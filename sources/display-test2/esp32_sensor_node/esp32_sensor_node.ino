/*
 * SafeNest ESP32 sensor node
 *
 * Sensors:
 *   - Seeed MR60BHA2 (UART): respiration and heart rate
 *   - Sensirion SCD40/SCD4x (I2C): CO2
 *   - PIR (digital input): motion
 *   - Waveshare Thermal Camera HAT / MI48xx (I2C control + SPI data): 80 x 62
 *
 * Transport:
 *   - Wi-Fi TCP to a Raspberry Pi
 *   - JSON packets for low-rate scalar telemetry
 *   - Big-endian uint16 binary packets for thermal frames
 *
 * The Arduino loop never calls delay(). Sensor scheduling uses millis().
 * TCP connection and potentially blocking writes run in a separate FreeRTOS task.
 * If the network is slow, the one-slot thermal queue keeps only the newest frame.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <Wire.h>
#include <SPI.h>
#include <SensirionI2cScd4x.h>
#include "Seeed_Arduino_mmWave.h"
#include "secrets.h"

// -----------------------------------------------------------------------------
// Device identity. Wi-Fi and Raspberry Pi settings live in ignored secrets.h.
// ----------------------------------------------------------------------------
constexpr char DEVICE_ID[] = "esp32-01";

// -----------------------------------------------------------------------------
// ESP32 Dev Module wiring (matches the existing standalone sensor tests).
// ----------------------------------------------------------------------------
constexpr int PIN_I2C_SDA = 21;
constexpr int PIN_I2C_SCL = 22;
constexpr int PIN_PIR = 13;
constexpr int PIN_MMWAVE_RX = 16;  // ESP32 RX <- MR60BHA2 TX
constexpr int PIN_MMWAVE_TX = 17;  // ESP32 TX -> MR60BHA2 RX

constexpr int PIN_THERMAL_SCLK = 18;
constexpr int PIN_THERMAL_MISO = 19;
constexpr int PIN_THERMAL_MOSI = 23;
constexpr int PIN_THERMAL_CS = 27;
constexpr int PIN_THERMAL_READY = 26;
constexpr int PIN_THERMAL_RESET = 25;

constexpr uint32_t USB_BAUD = 115200;
constexpr uint32_t MMWAVE_BAUD = 115200;
constexpr uint32_t THERMAL_SPI_HZ = 8000000;

// Runtime schedules. SCD4x periodic mode creates a new sample about every 5 s.
constexpr uint32_t PIR_PERIOD_MS = 20;
constexpr uint32_t CO2_POLL_PERIOD_MS = 250;
constexpr uint32_t TELEMETRY_PERIOD_MS = 1000;
constexpr uint32_t HEALTH_LOG_PERIOD_MS = 10000;
constexpr uint32_t MMWAVE_STALE_MS = 5000;
constexpr uint32_t CO2_STALE_MS = 15000;

// -----------------------------------------------------------------------------
// Thermal-camera constants (MI48xx + MI0801/MI0802, 80 x 62).
// ----------------------------------------------------------------------------
constexpr uint8_t THERMAL_ADDRESS_A = 0x40;
constexpr uint8_t THERMAL_ADDRESS_B = 0x41;
constexpr uint8_t SCD4X_ADDRESS = 0x62;

constexpr uint8_t REG_EVK_TEST = 0x00;
constexpr uint8_t REG_SENSOR_POWERUP = 0xB0;
constexpr uint8_t REG_FRAME_MODE = 0xB1;
constexpr uint8_t REG_FW_VERSION_1 = 0xB2;
constexpr uint8_t REG_FW_VERSION_2 = 0xB3;
constexpr uint8_t REG_FRAME_RATE = 0xB4;
constexpr uint8_t REG_STATUS = 0xB6;
constexpr uint8_t REG_SENSOR_TYPE = 0xBA;

constexpr uint8_t STATUS_DATA_READY = 0x10;
constexpr uint8_t STATUS_BOOTING = 0x20;
constexpr uint8_t MODE_CONTINUOUS = 0x02;

constexpr size_t THERMAL_WIDTH = 80;
constexpr size_t THERMAL_HEIGHT = 62;
constexpr size_t THERMAL_PIXEL_COUNT = THERMAL_WIDTH * THERMAL_HEIGHT;
constexpr size_t THERMAL_HEADER_WORDS = THERMAL_WIDTH;
constexpr size_t THERMAL_CAPTURE_WORDS =
    THERMAL_HEADER_WORDS + THERMAL_PIXEL_COUNT;

// Set divisor 4: for a 25 FPS sensor this requests about 6.25 FPS.
// Lowering this value raises bandwidth and ESP32 CPU/SPI load.
constexpr uint8_t THERMAL_FRAME_RATE_DIVIDER = 4;

// -----------------------------------------------------------------------------
// SafeNest TCP protocol v1.
// Outer header (16 bytes, network byte order):
//   magic[4]="SNST", version:u8, type:u8, flags:u16,
//   packet_sequence:u32, payload_length:u32
// Packet type 1 payload: UTF-8 JSON.
// Packet type 2 payload: 16-byte thermal metadata followed by 4960 uint16 words.
// -----------------------------------------------------------------------------
constexpr uint8_t PROTOCOL_VERSION = 1;
constexpr uint8_t PACKET_TELEMETRY_JSON = 1;
constexpr uint8_t PACKET_THERMAL_U16_BE = 2;
constexpr size_t PACKET_HEADER_SIZE = 16;
constexpr size_t THERMAL_META_SIZE = 16;

struct TelemetrySnapshot {
  uint32_t sequence;
  uint32_t uptimeMs;
  float respirationRate;
  float heartRate;
  uint16_t co2Ppm;
  bool pirMotion;
  bool respirationValid;
  bool heartValid;
  bool co2Valid;
};

struct ThermalTxFrame {
  uint32_t frameSequence;
  uint32_t uptimeMs;
  uint16_t minimumRaw;
  uint16_t maximumRaw;
  uint16_t pixels[THERMAL_PIXEL_COUNT];
};

SensirionI2cScd4x scd4x;
HardwareSerial mmWaveSerial(2);
SEEED_MR60BHA2 mmWave;
SPIClass thermalSpi(VSPI);

QueueHandle_t telemetryQueue = nullptr;
QueueHandle_t thermalQueue = nullptr;

uint16_t thermalCapture[THERMAL_CAPTURE_WORDS];
// These ~10 KiB objects are global on purpose. Keeping either on a task stack
// can overflow the default Arduino loop stack on a non-PSRAM ESP32.
ThermalTxFrame thermalProducerFrame;
ThermalTxFrame thermalNetworkFrame;
uint8_t thermalAddress = 0;
bool thermalStarted = false;
bool co2Started = false;

float respirationRate = NAN;
float heartRate = NAN;
uint16_t co2Ppm = 0;
bool pirMotion = false;

uint32_t lastRespirationMs = 0;
uint32_t lastHeartMs = 0;
uint32_t lastCo2Ms = 0;
uint32_t lastThermalMs = 0;
uint32_t lastThermalStatusPollMs = 0;
uint32_t lastPirPollMs = 0;
uint32_t lastCo2PollMs = 0;
uint32_t lastTelemetryMs = 0;
uint32_t lastHealthLogMs = 0;
uint32_t telemetrySequence = 0;
uint32_t thermalSequence = 0;

// Wrap-safe periodic scheduling helper. Updating by period, rather than assigning
// now, avoids gradual drift. A long overrun is collapsed to one execution.
bool scheduleDue(uint32_t now, uint32_t &lastRun, uint32_t period) {
  if (static_cast<uint32_t>(now - lastRun) < period) return false;
  lastRun += period;
  if (static_cast<uint32_t>(now - lastRun) >= period) lastRun = now;
  return true;
}

bool isFresh(uint32_t timestamp, uint32_t now, uint32_t timeout) {
  return timestamp != 0 && static_cast<uint32_t>(now - timestamp) < timeout;
}

// Blocking waits are used only during one-time hardware initialization. The
// runtime loop itself remains delay-free.
void setupWait(uint32_t milliseconds) {
  vTaskDelay(pdMS_TO_TICKS(milliseconds));
}

bool i2cPresent(uint8_t address) {
  Wire.beginTransmission(address);
  return Wire.endTransmission() == 0;
}

bool thermalWriteRegister(uint8_t reg, uint8_t value) {
  if (thermalAddress == 0) return false;
  Wire.beginTransmission(thermalAddress);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

bool thermalReadRegister(uint8_t reg, uint8_t &value) {
  if (thermalAddress == 0) return false;
  Wire.beginTransmission(thermalAddress);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(thermalAddress, static_cast<uint8_t>(1)) != 1) {
    return false;
  }
  value = Wire.read();
  return true;
}

void initializeThermalCamera() {
  if (i2cPresent(THERMAL_ADDRESS_A)) {
    thermalAddress = THERMAL_ADDRESS_A;
  } else if (i2cPresent(THERMAL_ADDRESS_B)) {
    thermalAddress = THERMAL_ADDRESS_B;
  } else {
    Serial.println("[thermal] ERROR: I2C address 0x40/0x41 not found");
    return;
  }

  // Active-low reset. The pulse and post-reset wait are required by hardware.
  digitalWrite(PIN_THERMAL_RESET, LOW);
  delayMicroseconds(100);
  digitalWrite(PIN_THERMAL_RESET, HIGH);
  setupWait(100);

  uint8_t evkTest = 0;
  if (!thermalReadRegister(REG_EVK_TEST, evkTest)) {
    Serial.println("[thermal] ERROR: register read failed");
    return;
  }

  // Non-bridge MI48 variants require the explicit sensor power-up command.
  if (evkTest != 0xFF) {
    if (!thermalWriteRegister(REG_SENSOR_POWERUP, 0x13)) {
      Serial.println("[thermal] ERROR: power-up command failed");
      return;
    }
    setupWait(100);
  }

  const uint32_t bootStarted = millis();
  uint8_t status = STATUS_BOOTING;
  while (static_cast<uint32_t>(millis() - bootStarted) < 3000) {
    if (!thermalReadRegister(REG_STATUS, status)) {
      Serial.println("[thermal] ERROR: status read failed");
      return;
    }
    if ((status & STATUS_BOOTING) == 0) break;
    setupWait(25);
  }
  if (status & STATUS_BOOTING) {
    Serial.println("[thermal] ERROR: boot timeout");
    return;
  }

  uint8_t fw1 = 0, fw2 = 0, sensorType = 0;
  thermalReadRegister(REG_FW_VERSION_1, fw1);
  thermalReadRegister(REG_FW_VERSION_2, fw2);
  thermalReadRegister(REG_SENSOR_TYPE, sensorType);

  thermalWriteRegister(REG_FRAME_MODE, 0x00);
  setupWait(30);
  if (!thermalWriteRegister(REG_FRAME_RATE, THERMAL_FRAME_RATE_DIVIDER) ||
      !thermalWriteRegister(REG_FRAME_MODE, MODE_CONTINUOUS)) {
    Serial.println("[thermal] ERROR: continuous stream start failed");
    return;
  }

  thermalStarted = true;
  Serial.printf("[thermal] ready: addr=0x%02X type=%u fw=%u.%u.%u\n",
                thermalAddress, sensorType, (fw1 >> 4) & 0x0F,
                fw1 & 0x0F, fw2);
}

bool thermalDataReady(uint32_t now) {
  if (digitalRead(PIN_THERMAL_READY) == HIGH) return true;

  // I2C polling is a fallback and also makes a disconnected READY wire visible.
  if (static_cast<uint32_t>(now - lastThermalStatusPollMs) < 25) return false;
  lastThermalStatusPollMs = now;
  uint8_t status = 0;
  return thermalReadRegister(REG_STATUS, status) &&
         (status & STATUS_DATA_READY) != 0;
}

void captureThermalIfReady(uint32_t now) {
  if (!thermalStarted || !thermalDataReady(now)) return;

  thermalSpi.beginTransaction(
      SPISettings(THERMAL_SPI_HZ, MSBFIRST, SPI_MODE0));
  digitalWrite(PIN_THERMAL_CS, LOW);
  delayMicroseconds(100);

  // MI48xx returns every word MSB first. Reading bytes explicitly avoids host
  // endianness ambiguity in SPI.transfer16().
  for (size_t i = 0; i < THERMAL_CAPTURE_WORDS; ++i) {
    const uint8_t highByte = thermalSpi.transfer(0x00);
    const uint8_t lowByte = thermalSpi.transfer(0x00);
    thermalCapture[i] = (static_cast<uint16_t>(highByte) << 8) | lowByte;
  }

  delayMicroseconds(100);
  digitalWrite(PIN_THERMAL_CS, HIGH);
  thermalSpi.endTransaction();

  ThermalTxFrame &frame = thermalProducerFrame;
  frame.frameSequence = ++thermalSequence;
  frame.uptimeMs = millis();
  frame.minimumRaw = UINT16_MAX;
  frame.maximumRaw = 0;

  for (size_t i = 0; i < THERMAL_PIXEL_COUNT; ++i) {
    const uint16_t raw = thermalCapture[THERMAL_HEADER_WORDS + i];
    frame.pixels[i] = raw;
    if (raw < frame.minimumRaw) frame.minimumRaw = raw;
    if (raw > frame.maximumRaw) frame.maximumRaw = raw;
  }

  lastThermalMs = frame.uptimeMs;
  // Queue length is one, so this replaces an unsent old frame under congestion.
  xQueueOverwrite(thermalQueue, &frame);
}

void initializeCo2() {
  if (!i2cPresent(SCD4X_ADDRESS)) {
    Serial.println("[co2] ERROR: I2C address 0x62 not found");
    return;
  }

  scd4x.begin(Wire, SCD41_I2C_ADDR_62);
  scd4x.stopPeriodicMeasurement();
  setupWait(500);
  const int16_t error = scd4x.startPeriodicMeasurement();
  if (error != 0) {
    Serial.printf("[co2] ERROR: startPeriodicMeasurement=%d\n", error);
    return;
  }
  co2Started = true;
  Serial.println("[co2] ready: first measurement takes about 5 seconds");
}

void pollCo2(uint32_t now) {
  if (!co2Started || !scheduleDue(now, lastCo2PollMs, CO2_POLL_PERIOD_MS)) {
    return;
  }

  bool ready = false;
  if (scd4x.getDataReadyStatus(ready) != 0 || !ready) return;

  uint16_t newCo2 = 0;
  float temperature = NAN;
  float humidity = NAN;
  if (scd4x.readMeasurement(newCo2, temperature, humidity) == 0 &&
      newCo2 != 0) {
    co2Ppm = newCo2;
    lastCo2Ms = millis();
  }
}

void pollMmWave(uint32_t now) {
  // timeout=0 makes the library consume currently buffered UART data without
  // waiting. Repeated loop calls drain all queued radar frames.
  if (!mmWave.update(0)) return;

  float value = 0.0f;
  if (mmWave.getBreathRate(value)) {
    respirationRate = value;
    lastRespirationMs = now;
  }
  if (mmWave.getHeartRate(value)) {
    heartRate = value;
    lastHeartMs = now;
  }
}

void publishTelemetrySnapshot(uint32_t now) {
  if (!scheduleDue(now, lastTelemetryMs, TELEMETRY_PERIOD_MS)) return;

  TelemetrySnapshot snapshot{};
  snapshot.sequence = ++telemetrySequence;
  snapshot.uptimeMs = now;
  snapshot.respirationRate = respirationRate;
  snapshot.heartRate = heartRate;
  snapshot.co2Ppm = co2Ppm;
  snapshot.pirMotion = pirMotion;
  snapshot.respirationValid = isFresh(lastRespirationMs, now, MMWAVE_STALE_MS);
  snapshot.heartValid = isFresh(lastHeartMs, now, MMWAVE_STALE_MS);
  snapshot.co2Valid = isFresh(lastCo2Ms, now, CO2_STALE_MS);
  xQueueOverwrite(telemetryQueue, &snapshot);
}

// Big-endian integer encoders keep the protocol independent of CPU endianness.
void putU16(uint8_t *destination, uint16_t value) {
  destination[0] = static_cast<uint8_t>(value >> 8);
  destination[1] = static_cast<uint8_t>(value);
}

void putU32(uint8_t *destination, uint32_t value) {
  destination[0] = static_cast<uint8_t>(value >> 24);
  destination[1] = static_cast<uint8_t>(value >> 16);
  destination[2] = static_cast<uint8_t>(value >> 8);
  destination[3] = static_cast<uint8_t>(value);
}

void makePacketHeader(uint8_t *header, uint8_t type, uint32_t sequence,
                      uint32_t payloadLength) {
  memcpy(header, "SNST", 4);
  header[4] = PROTOCOL_VERSION;
  header[5] = type;
  putU16(header + 6, 0);  // flags: reserved for protocol v1
  putU32(header + 8, sequence);
  putU32(header + 12, payloadLength);
}

// Runs only in the network task, so a slow peer can never stall sensor capture.
bool writeAll(WiFiClient &client, const uint8_t *data, size_t length) {
  size_t sent = 0;
  const uint32_t started = millis();
  while (sent < length && client.connected()) {
    const size_t written = client.write(data + sent, length - sent);
    if (written > 0) {
      sent += written;
    } else {
      vTaskDelay(pdMS_TO_TICKS(2));
    }
    if (static_cast<uint32_t>(millis() - started) > 3000) return false;
  }
  return sent == length;
}

void formatNullableFloat(char *output, size_t outputSize, bool valid,
                         float value) {
  if (valid && isfinite(value)) {
    snprintf(output, outputSize, "%.2f", value);
  } else {
    strlcpy(output, "null", outputSize);
  }
}

bool sendTelemetry(WiFiClient &client, const TelemetrySnapshot &snapshot) {
  char respiration[20], heart[20], co2[20];
  formatNullableFloat(respiration, sizeof(respiration),
                      snapshot.respirationValid, snapshot.respirationRate);
  formatNullableFloat(heart, sizeof(heart), snapshot.heartValid,
                      snapshot.heartRate);
  if (snapshot.co2Valid) {
    snprintf(co2, sizeof(co2), "%u", snapshot.co2Ppm);
  } else {
    strlcpy(co2, "null", sizeof(co2));
  }

  char json[512];
  const int length = snprintf(
      json, sizeof(json),
      "{\"schema\":\"safenest.telemetry.v1\",\"device_id\":\"%s\","
      "\"seq\":%lu,\"uptime_ms\":%lu,\"resp_rate_bpm\":%s,"
      "\"heart_rate_bpm\":%s,\"co2_ppm\":%s,\"pir_motion\":%s,"
      "\"valid\":{\"respiration\":%s,\"heart\":%s,\"co2\":%s}}",
      DEVICE_ID, static_cast<unsigned long>(snapshot.sequence),
      static_cast<unsigned long>(snapshot.uptimeMs), respiration, heart, co2,
      snapshot.pirMotion ? "true" : "false",
      snapshot.respirationValid ? "true" : "false",
      snapshot.heartValid ? "true" : "false",
      snapshot.co2Valid ? "true" : "false");
  if (length <= 0 || static_cast<size_t>(length) >= sizeof(json)) return false;

  uint8_t header[PACKET_HEADER_SIZE];
  makePacketHeader(header, PACKET_TELEMETRY_JSON, snapshot.sequence,
                   static_cast<uint32_t>(length));
  return writeAll(client, header, sizeof(header)) &&
         writeAll(client, reinterpret_cast<const uint8_t *>(json), length);
}

bool sendThermal(WiFiClient &client, const ThermalTxFrame &frame) {
  const uint32_t payloadLength =
      THERMAL_META_SIZE + THERMAL_PIXEL_COUNT * sizeof(uint16_t);
  uint8_t header[PACKET_HEADER_SIZE];
  makePacketHeader(header, PACKET_THERMAL_U16_BE, frame.frameSequence,
                   payloadLength);
  if (!writeAll(client, header, sizeof(header))) return false;

  // Thermal metadata: width, height, frame_seq, uptime_ms, min_raw, max_raw.
  uint8_t meta[THERMAL_META_SIZE];
  putU16(meta + 0, THERMAL_WIDTH);
  putU16(meta + 2, THERMAL_HEIGHT);
  putU32(meta + 4, frame.frameSequence);
  putU32(meta + 8, frame.uptimeMs);
  putU16(meta + 12, frame.minimumRaw);
  putU16(meta + 14, frame.maximumRaw);
  if (!writeAll(client, meta, sizeof(meta))) return false;

  // Convert in small chunks instead of allocating a second 9.7 KiB byte array.
  uint8_t chunk[512];
  size_t pixelIndex = 0;
  while (pixelIndex < THERMAL_PIXEL_COUNT) {
    const size_t pixelsThisChunk =
        min(sizeof(chunk) / 2, THERMAL_PIXEL_COUNT - pixelIndex);
    for (size_t i = 0; i < pixelsThisChunk; ++i) {
      putU16(chunk + i * 2, frame.pixels[pixelIndex + i]);
    }
    if (!writeAll(client, chunk, pixelsThisChunk * 2)) return false;
    pixelIndex += pixelsThisChunk;
  }
  return true;
}

void networkTask(void *parameter) {
  (void)parameter;
  WiFiClient client;
  client.setNoDelay(true);
  TelemetrySnapshot telemetry{};

  for (;;) {
    if (WiFi.status() != WL_CONNECTED) {
      if (client.connected()) client.stop();
      vTaskDelay(pdMS_TO_TICKS(250));
      continue;
    }

    if (!client.connected()) {
      Serial.printf("[network] connecting to %s:%u\n", RPI_HOST, RPI_PORT);
      if (!client.connect(RPI_HOST, RPI_PORT, 1500)) {
        vTaskDelay(pdMS_TO_TICKS(1000));
        continue;
      }
      client.setNoDelay(true);
      Serial.println("[network] Raspberry Pi connected");
    }

    // Scalar telemetry has priority and is small.
    if (xQueueReceive(telemetryQueue, &telemetry, 0) == pdTRUE) {
      if (!sendTelemetry(client, telemetry)) {
        client.stop();
        continue;
      }
    }

    if (xQueueReceive(thermalQueue, &thermalNetworkFrame, 0) == pdTRUE) {
      if (!sendThermal(client, thermalNetworkFrame)) {
        client.stop();
        continue;
      }
    }

    vTaskDelay(pdMS_TO_TICKS(2));
  }
}

void logHealth(uint32_t now) {
  if (!scheduleDue(now, lastHealthLogMs, HEALTH_LOG_PERIOD_MS)) return;
  Serial.printf(
      "[health] wifi=%s rpi=%s resp=%.1f heart=%.1f co2=%u pir=%d "
      "thermal_frames=%lu free_heap=%u\n",
      WiFi.status() == WL_CONNECTED ? "up" : "down", RPI_HOST,
      respirationRate, heartRate, co2Ppm, pirMotion,
      static_cast<unsigned long>(thermalSequence), ESP.getFreeHeap());
}

void setup() {
  Serial.begin(USB_BAUD);
  setupWait(500);
  Serial.println("\nSafeNest ESP32 sensor node starting");

  pinMode(PIN_PIR, INPUT);
  pinMode(PIN_THERMAL_CS, OUTPUT);
  digitalWrite(PIN_THERMAL_CS, HIGH);
  pinMode(PIN_THERMAL_READY, INPUT);
  pinMode(PIN_THERMAL_RESET, OUTPUT);
  digitalWrite(PIN_THERMAL_RESET, HIGH);

  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
  Wire.setClock(400000);
  thermalSpi.begin(PIN_THERMAL_SCLK, PIN_THERMAL_MISO,
                   PIN_THERMAL_MOSI, PIN_THERMAL_CS);

  // setPins() before the library's begin() makes its pin-less Serial.begin()
  // retain GPIO16/17 on both Arduino-ESP32 core 2.x and 3.x.
  mmWaveSerial.setPins(PIN_MMWAVE_RX, PIN_MMWAVE_TX);
  mmWave.begin(&mmWaveSerial, MMWAVE_BAUD, 0);

  telemetryQueue = xQueueCreate(1, sizeof(TelemetrySnapshot));
  thermalQueue = xQueueCreate(1, sizeof(ThermalTxFrame));
  if (telemetryQueue == nullptr || thermalQueue == nullptr) {
    Serial.println("[fatal] queue allocation failed");
    while (true) vTaskDelay(portMAX_DELAY);
  }

  initializeThermalCamera();
  initializeCo2();

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);  // lower latency for the continuous thermal stream
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.printf("[wifi] connecting to %s asynchronously\n", WIFI_SSID);

  // The networking task owns WiFiClient and all socket writes.
  xTaskCreatePinnedToCore(networkTask, "network", 16384, nullptr, 1, nullptr, 0);
}

void loop() {
  const uint32_t now = millis();

  pollMmWave(now);
  pollCo2(now);
  captureThermalIfReady(now);

  if (scheduleDue(now, lastPirPollMs, PIR_PERIOD_MS)) {
    pirMotion = digitalRead(PIN_PIR) == HIGH;
  }

  publishTelemetrySnapshot(now);
  logHealth(now);

  // Cooperative yield only; there is intentionally no delay() in runtime.
  taskYIELD();
}
