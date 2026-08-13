#include <Arduino.h>

#include <cmath>
#include <cstring>

#include "mmwave_config.h"

namespace {

using namespace safenest::mmwave_config;

constexpr size_t kHeaderSize = 8;
constexpr size_t kMaxDataSize = 512;
constexpr size_t kMaxFrameSize = kHeaderSize + kMaxDataSize + 1;
constexpr uint8_t kSof = 0x01;
constexpr uint16_t kTypePhases = 0x0A13;
constexpr uint16_t kTypeBreath = 0x0A14;
constexpr uint16_t kTypeHeart = 0x0A15;
constexpr uint16_t kTypeDistance = 0x0A16;
constexpr uint16_t kTypePresence = 0x0F09;
constexpr uint16_t kTypeFirmware = 0xFFFF;

HardwareSerial radarSerial(2);
uint8_t frameBuffer[kMaxFrameSize];
size_t frameLength = 0;
size_t expectedLength = 0;

uint32_t sequence = 0;
uint32_t validFrames = 0;
uint32_t checksumErrors = 0;
uint32_t parseErrors = 0;
uint32_t lastTelemetryMs = 0;
uint32_t lastValidFrameMs = 0;
uint8_t consecutiveUartErrors = 0;

bool presenceKnown = false;
bool presenceRaw = false;
bool presenceStableKnown = false;
bool presenceStable = false;
uint8_t presenceHistory = 0;
uint8_t presenceSamples = 0;
uint32_t presenceUpdatedMs = 0;
uint32_t stablePresenceStartedMs = 0;

float distanceRaw = NAN;
float breathRaw = NAN;
float heartRaw = NAN;
float totalPhase = NAN;
float breathPhase = NAN;
float heartPhase = NAN;
uint32_t distanceUpdatedMs = 0;
uint32_t breathUpdatedMs = 0;
uint32_t heartUpdatedMs = 0;
uint32_t phasesUpdatedMs = 0;
uint32_t firmwareRaw = 0;
bool firmwareKnown = false;

struct TimedSample {
  uint32_t timestampMs;
  float value;
};

TimedSample breathWindow[kBreathWindowCapacity];
TimedSample distanceWindow[kBreathWindowCapacity];
size_t breathWindowStart = 0;
size_t breathWindowCount = 0;
size_t distanceWindowStart = 0;
size_t distanceWindowCount = 0;
uint32_t lastBreathWindowSampleMs = 0;
uint32_t lastDistanceWindowSampleMs = 0;

struct WindowStats {
  bool ready;
  float stddev;
  float rate;
  size_t crossings;
};

enum class SensorState { kWarmup, kValid, kDegraded, kUnknown, kFault };

void appendSample(TimedSample* window, size_t* start, size_t* count,
                  uint32_t timestampMs, float value) {
  while (*count > 0 &&
         timestampMs - window[*start].timestampMs > kBreathWindowMs) {
    *start = (*start + 1) % kBreathWindowCapacity;
    --(*count);
  }
  if (*count == kBreathWindowCapacity) {
    *start = (*start + 1) % kBreathWindowCapacity;
    --(*count);
  }
  const size_t end = (*start + *count) % kBreathWindowCapacity;
  window[end] = {timestampMs, value};
  ++(*count);
}

bool windowReady(const TimedSample* window, size_t start, size_t count) {
  if (count < 2) return false;
  const size_t last = (start + count - 1) % kBreathWindowCapacity;
  return window[last].timestampMs - window[start].timestampMs >=
         kBreathWindowMs - kWindowReadyToleranceMs;
}

float windowStddev(const TimedSample* window, size_t start, size_t count) {
  if (count == 0) return NAN;
  double sum = 0.0;
  for (size_t i = 0; i < count; ++i) {
    sum += window[(start + i) % kBreathWindowCapacity].value;
  }
  const double mean = sum / count;
  double squared = 0.0;
  for (size_t i = 0; i < count; ++i) {
    const double delta =
        window[(start + i) % kBreathWindowCapacity].value - mean;
    squared += delta * delta;
  }
  return static_cast<float>(sqrt(squared / count));
}

WindowStats breathStats() {
  WindowStats result = {false, NAN, NAN, 0};
  result.ready = windowReady(breathWindow, breathWindowStart, breathWindowCount);
  if (!result.ready) return result;
  result.stddev =
      windowStddev(breathWindow, breathWindowStart, breathWindowCount);
  if (!std::isfinite(result.stddev) || result.stddev == 0.0F) return result;

  double sum = 0.0;
  for (size_t i = 0; i < breathWindowCount; ++i) {
    sum += breathWindow[(breathWindowStart + i) % kBreathWindowCapacity].value;
  }
  const float mean = static_cast<float>(sum / breathWindowCount);
  const float hysteresis = kBreathHysteresisFraction * result.stddev;
  int8_t state = 0;
  uint32_t firstCrossingMs = 0;
  uint32_t lastCrossingMs = 0;
  for (size_t i = 0; i < breathWindowCount; ++i) {
    const TimedSample& sample =
        breathWindow[(breathWindowStart + i) % kBreathWindowCapacity];
    const float centered = sample.value - mean;
    if (state <= 0 && centered > hysteresis) {
      state = 1;
      if (result.crossings == 0) firstCrossingMs = sample.timestampMs;
      lastCrossingMs = sample.timestampMs;
      ++result.crossings;
    } else if (state >= 0 && centered < -hysteresis) {
      state = -1;
    }
  }
  if (result.crossings >= kBreathMinCrossings &&
      lastCrossingMs > firstCrossingMs) {
    result.rate = 60000.0F * static_cast<float>(result.crossings - 1) /
                  static_cast<float>(lastCrossingMs - firstCrossingMs);
  }
  return result;
}

uint8_t checksum(const uint8_t* data, size_t length) {
  uint8_t value = 0;
  for (size_t i = 0; i < length; ++i) value ^= data[i];
  return static_cast<uint8_t>(~value);
}

float readFloat(const uint8_t* data) {
  float value;
  memcpy(&value, data, sizeof(value));
  return value;
}

uint32_t readU32(const uint8_t* data) {
  uint32_t value;
  memcpy(&value, data, sizeof(value));
  return value;
}

bool fresh(uint32_t updatedAt, uint32_t maxAge, uint32_t now) {
  return updatedAt != 0 && now - updatedAt <= maxAge;
}

void recordUartError() {
  if (consecutiveUartErrors < UINT8_MAX) ++consecutiveUartErrors;
}

void recordValidFrame(uint32_t now) {
  ++validFrames;
  consecutiveUartErrors = 0;
  lastValidFrameMs = now;
}

void updateStablePresence(bool raw, uint32_t now) {
  presenceHistory = static_cast<uint8_t>(
      ((presenceHistory << 1U) | (raw ? 1U : 0U)) &
      ((1U << kPresenceWindowSamples) - 1U));
  if (presenceSamples < kPresenceWindowSamples) ++presenceSamples;
  if (presenceSamples < kPresenceWindowSamples) {
    presenceStableKnown = false;
    return;
  }
  uint8_t trueCount = 0;
  for (uint8_t i = 0; i < kPresenceWindowSamples; ++i) {
    trueCount += (presenceHistory >> i) & 1U;
  }
  const bool previous = presenceStable;
  if (trueCount >= kPresenceRequiredTrue) {
    presenceStable = true;
    presenceStableKnown = true;
  } else if (kPresenceWindowSamples - trueCount >= kPresenceRequiredFalse) {
    presenceStable = false;
    presenceStableKnown = true;
  } else {
    presenceStableKnown = false;
  }
  if (presenceStableKnown && presenceStable && !previous) {
    stablePresenceStartedMs = now;
  } else if (presenceStableKnown && !presenceStable) {
    stablePresenceStartedMs = 0;
  }
}

void printNullableFloat(float value) {
  if (std::isfinite(value)) Serial.print(value, 2);
  else Serial.print("null");
}

void printNullableBool(bool known, bool value) {
  if (known) Serial.print(value ? "true" : "false");
  else Serial.print("null");
}

void printAge(bool known, uint32_t updatedAt, uint32_t now) {
  if (known && updatedAt != 0) Serial.print(now - updatedAt);
  else Serial.print("null");
}

SensorState sensorState(uint32_t now, bool heartRawValid,
                        const WindowStats& breath, float distanceStd,
                        bool distanceWindowReady, const char** errorCode) {
  const bool uartTimedOut =
      (lastValidFrameMs == 0 && now > kFrameTimeoutMs) ||
      (lastValidFrameMs != 0 && now - lastValidFrameMs > kFrameTimeoutMs);
  if (uartTimedOut) {
    *errorCode = "UART_FRAME_TIMEOUT";
    return SensorState::kFault;
  }
  if (consecutiveUartErrors >= kFaultConsecutiveUartErrors) {
    *errorCode = "UART_CONSECUTIVE_ERRORS";
    return SensorState::kFault;
  }
  if (!presenceStableKnown) {
    *errorCode = "PRESENCE_WINDOW_NOT_READY";
    return SensorState::kUnknown;
  }
  if (!fresh(presenceUpdatedMs, kFrameTimeoutMs, now)) {
    *errorCode = "PRESENCE_STALE";
    return SensorState::kUnknown;
  }
  if (!presenceStable) {
    *errorCode = "PRESENCE_NOT_DETECTED";
    return SensorState::kUnknown;
  }
  const bool distanceValid =
      std::isfinite(distanceRaw) && distanceRaw >= kDistanceMinCm &&
      distanceRaw <= kDistanceMaxCm &&
      fresh(distanceUpdatedMs, kDistanceMaxAgeMs, now);
  if (!distanceValid) {
    *errorCode = "DISTANCE_INVALID_OR_STALE";
    return SensorState::kUnknown;
  }
  const bool phaseValid =
      std::isfinite(totalPhase) && std::isfinite(breathPhase) &&
      std::isfinite(heartPhase) && fresh(phasesUpdatedMs, kPhaseMaxAgeMs, now);
  if (!phaseValid) {
    *errorCode = "PHASE_INVALID_OR_STALE";
    return SensorState::kUnknown;
  }
  if (stablePresenceStartedMs == 0 ||
      now - stablePresenceStartedMs < kWarmupMs) {
    *errorCode = "TARGET_WARMUP";
    return SensorState::kWarmup;
  }
  if (!breath.ready) {
    *errorCode = "BREATH_WINDOW_NOT_READY";
    return SensorState::kUnknown;
  }
  if (distanceWindowReady && distanceStd == 0.0F && !heartRawValid) {
    *errorCode = "LOCK_LOSS_FREEZE";
    return SensorState::kDegraded;
  }
  if (breath.stddev < kBreathMinPhaseStd) {
    *errorCode = "BREATH_PHASE_LOW_AMPLITUDE";
    return SensorState::kDegraded;
  }
  if (!std::isfinite(breath.rate)) {
    *errorCode = "BREATH_RATE_UNAVAILABLE";
    return SensorState::kDegraded;
  }
  *errorCode = nullptr;
  return SensorState::kValid;
}

const char* stateName(SensorState state) {
  switch (state) {
    case SensorState::kWarmup: return "WARMUP";
    case SensorState::kValid: return "VALID";
    case SensorState::kDegraded: return "DEGRADED";
    case SensorState::kFault: return "FAULT";
    default: return "UNKNOWN";
  }
}

void emitTelemetry(uint32_t now) {
  const bool breathRawValid =
      std::isfinite(breathRaw) && breathRaw > 0.0F &&
      fresh(breathUpdatedMs, kVitalMaxAgeMs, now);
  const bool heartRawValid =
      std::isfinite(heartRaw) && heartRaw > 0.0F &&
      fresh(heartUpdatedMs, kVitalMaxAgeMs, now);
  const WindowStats breath = breathStats();
  const bool distanceStatsReady = windowReady(
      distanceWindow, distanceWindowStart, distanceWindowCount);
  const float distanceStd = distanceStatsReady
                                ? windowStddev(distanceWindow,
                                               distanceWindowStart,
                                               distanceWindowCount)
                                : NAN;
  const bool freezeDetected = distanceStatsReady && distanceStd == 0.0F &&
                              !heartRawValid;
  const bool filteredBreathValid = breath.ready &&
      std::isfinite(breath.rate) && breath.stddev >= kBreathMinPhaseStd &&
      !freezeDetected;
  const char* errorCode = nullptr;
  const SensorState state = sensorState(now, heartRawValid, breath, distanceStd,
                                        distanceStatsReady, &errorCode);
  const bool communicationOk = state != SensorState::kFault;

  Serial.printf(
      "{\"schema_version\":\"%s\",\"device_id\":\"%s\","
      "\"seq\":%lu,\"ts_monotonic_ms\":%lu,"
      "\"uart_frame_ok\":%s,\"checksum_ok\":%s,"
      "\"uart_frames_total\":%lu,\"checksum_errors\":%lu,"
      "\"parse_errors\":%lu,\"consecutive_uart_errors\":%u,"
      "\"human_detected_raw\":",
      kSchemaVersion, kDeviceId, static_cast<unsigned long>(sequence++),
      static_cast<unsigned long>(now), communicationOk ? "true" : "false",
      communicationOk ? "true" : "false", static_cast<unsigned long>(validFrames),
      static_cast<unsigned long>(checksumErrors),
      static_cast<unsigned long>(parseErrors), consecutiveUartErrors);
  printNullableBool(presenceKnown, presenceRaw);
  Serial.print(",\"human_detected_stable\":");
  printNullableBool(presenceStableKnown, presenceStable);
  Serial.print(",\"presence_age_ms\":");
  printAge(presenceKnown, presenceUpdatedMs, now);
  Serial.print(",\"distance_cm_raw\":");
  printNullableFloat(distanceRaw);
  Serial.print(",\"distance_age_ms\":");
  printAge(std::isfinite(distanceRaw), distanceUpdatedMs, now);
  Serial.print(",\"breath_rate_raw\":");
  printNullableFloat(breathRaw);
  Serial.print(",\"breath_rate_filtered\":");
  printNullableFloat(filteredBreathValid ? breath.rate : NAN);
  Serial.print(",\"breath_filtered_valid\":");
  Serial.print(filteredBreathValid ? "true" : "false");
  Serial.print(",\"breath_phase_std\":");
  printNullableFloat(breath.stddev);
  Serial.print(",\"breath_window_ready\":");
  Serial.print(breath.ready ? "true" : "false");
  Serial.print(",\"breath_rate_raw_trusted\":false,\"breath_raw_valid\":");
  Serial.print(breathRawValid ? "true" : "false");
  Serial.print(",\"breath_age_ms\":");
  printAge(std::isfinite(breathRaw), breathUpdatedMs, now);
  Serial.print(",\"heart_rate_raw\":");
  printNullableFloat(heartRaw);
  Serial.print(",\"heart_raw_valid\":");
  Serial.print(heartRawValid ? "true" : "false");
  Serial.print(",\"vital_presence_detected\":");
  Serial.print(heartRawValid ? "true" : "false");
  Serial.print(",\"heart_verified\":false,\"heart_age_ms\":");
  printAge(std::isfinite(heartRaw), heartUpdatedMs, now);
  Serial.print(",\"total_phase\":");
  printNullableFloat(totalPhase);
  Serial.print(",\"breath_phase\":");
  printNullableFloat(breathPhase);
  Serial.print(",\"heart_phase\":");
  printNullableFloat(heartPhase);
  Serial.print(",\"phase_age_ms\":");
  printAge(std::isfinite(totalPhase), phasesUpdatedMs, now);
  Serial.print(",\"distance_std_cm\":");
  printNullableFloat(distanceStd);
  Serial.print(",\"freeze_detected\":");
  Serial.print(freezeDetected ? "true" : "false");
  Serial.printf(",\"firmware_version\":\"%s\",\"sensor_firmware_version\":",
                kEspFirmwareVersion);
  if (firmwareKnown) Serial.printf("\"0x%08lX\"", static_cast<unsigned long>(firmwareRaw));
  else Serial.print("null");
  Serial.printf(",\"config_hash\":\"%s\",\"sensor_state\":\"%s\",\"error_code\":",
                kConfigSha256, stateName(state));
  if (errorCode) Serial.printf("\"%s\"", errorCode);
  else Serial.print("null");
  Serial.println("}");
}

bool handleValidFrame(uint16_t type, const uint8_t* data, size_t dataLength,
                      uint32_t now) {
  switch (type) {
    case kTypePhases: {
      if (dataLength < 12) return false;
      const float total = readFloat(data);
      const float breath = readFloat(data + 4);
      const float heart = readFloat(data + 8);
      if (!std::isfinite(total) || !std::isfinite(breath) || !std::isfinite(heart)) return false;
      totalPhase = total;
      breathPhase = breath;
      heartPhase = heart;
      phasesUpdatedMs = now;
      if (breathWindowCount == 0 ||
          now - lastBreathWindowSampleMs >= kTelemetryIntervalMs) {
        appendSample(breathWindow, &breathWindowStart, &breathWindowCount, now,
                     breathPhase);
        lastBreathWindowSampleMs = now;
      }
      return true;
    }
    case kTypeBreath: {
      if (dataLength < 4) return false;
      breathRaw = readFloat(data);
      if (!std::isfinite(breathRaw)) breathRaw = NAN;
      breathUpdatedMs = now;
      return true;
    }
    case kTypeHeart: {
      if (dataLength < 4) return false;
      heartRaw = readFloat(data);
      if (!std::isfinite(heartRaw)) heartRaw = NAN;
      heartUpdatedMs = now;
      return true;
    }
    case kTypeDistance:
      if (dataLength < 8) return false;
      distanceRaw = readU32(data) ? readFloat(data + 4) : NAN;
      if (!std::isfinite(distanceRaw)) distanceRaw = NAN;
      distanceUpdatedMs = now;
      if (std::isfinite(distanceRaw) &&
          (distanceWindowCount == 0 ||
           now - lastDistanceWindowSampleMs >= kTelemetryIntervalMs)) {
        appendSample(distanceWindow, &distanceWindowStart, &distanceWindowCount,
                     now, distanceRaw);
        lastDistanceWindowSampleMs = now;
      }
      return true;
    case kTypePresence:
      if (dataLength < 1) return false;
      presenceKnown = true;
      presenceRaw = data[0] != 0;
      presenceUpdatedMs = now;
      updateStablePresence(presenceRaw, now);
      return true;
    case kTypeFirmware:
      if (dataLength < 4) return false;
      firmwareRaw = readU32(data);
      firmwareKnown = true;
      return true;
    default:
      return true;
  }
}

void processFrame(uint32_t now) {
  const size_t dataLength = (static_cast<size_t>(frameBuffer[3]) << 8) |
                            static_cast<size_t>(frameBuffer[4]);
  const bool headerOk = checksum(frameBuffer, 7) == frameBuffer[7];
  const bool dataOk = checksum(frameBuffer + kHeaderSize, dataLength) ==
                      frameBuffer[kHeaderSize + dataLength];
  if (!headerOk || !dataOk) {
    ++checksumErrors;
    recordUartError();
    return;
  }
  const uint16_t type = (static_cast<uint16_t>(frameBuffer[5]) << 8) |
                        frameBuffer[6];
  if (!handleValidFrame(type, frameBuffer + kHeaderSize, dataLength, now)) {
    ++parseErrors;
    recordUartError();
    return;
  }
  recordValidFrame(now);
}

void resetParserWithError() {
  ++parseErrors;
  recordUartError();
  frameLength = 0;
  expectedLength = 0;
}

void consumeByte(uint8_t value, uint32_t now) {
  if (frameLength == 0) {
    if (value == kSof) frameBuffer[frameLength++] = value;
    return;
  }
  if (frameLength >= kMaxFrameSize) {
    resetParserWithError();
    return;
  }
  frameBuffer[frameLength++] = value;
  if (frameLength == kHeaderSize) {
    const size_t dataLength = (static_cast<size_t>(frameBuffer[3]) << 8) |
                              frameBuffer[4];
    if (dataLength > kMaxDataSize) {
      resetParserWithError();
      return;
    }
    expectedLength = kHeaderSize + dataLength + 1;
  }
  if (expectedLength != 0 && frameLength == expectedLength) {
    processFrame(now);
    frameLength = 0;
    expectedLength = 0;
  }
}

}  // namespace

void setup() {
  Serial.begin(kUsbBaud);
  delay(1000);
  radarSerial.begin(kRadarBaud, SERIAL_8N1, kRadarRxPin, kRadarTxPin);
  Serial.printf(
      "{\"event\":\"boot\",\"board\":\"esp-wroom-32\","
      "\"firmware_version\":\"%s\",\"config_hash\":\"%s\","
      "\"radar_uart\":\"UART2\",\"rx_gpio\":%d,\"tx_gpio\":%d,"
      "\"baud\":%lu}\n",
      kEspFirmwareVersion, kConfigSha256, kRadarRxPin, kRadarTxPin,
      static_cast<unsigned long>(kRadarBaud));
}

void loop() {
  while (radarSerial.available() > 0) {
    consumeByte(static_cast<uint8_t>(radarSerial.read()), millis());
  }
  const uint32_t now = millis();
  if (now - lastTelemetryMs >= kTelemetryIntervalMs) {
    lastTelemetryMs = now;
    emitTelemetry(now);
  }
  delay(1);
}
