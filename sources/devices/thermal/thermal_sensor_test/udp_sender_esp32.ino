#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>
#include <WiFi.h>
#include <WiFiUdp.h>

// ==========================================
// 1. Wi-Fi 및 UDP 설정
// ==========================================
const char* ssid = "EELab04 2G";           // 공유기 SSID 입력
const char* password = "openlab206";   // 공유기 비밀번호 입력

const char* receiverIP = "192.168.1.44";      // 라즈베리파이의 IP 주소로 반드시 변경하세요
const int receiverPort = 5005;                 // 라즈베리파이 수신 포트

WiFiUDP udp;

// ==========================================
// 2. 하드웨어 핀 매핑 (XIAO-ESP32-C6 기준)
// ==========================================
#define PIN_SDA      D4   // I2C SDA
#define PIN_SCL      D5   // I2C SCL
#define PIN_MOSI     D10  // SPI MOSI
#define PIN_MISO     D9   // SPI MISO
#define PIN_CLK      D8   // SPI SCK (CLK)
#define PIN_CS       D3   // SPI Slave Select (CS)
#define PIN_D_READY  D1   // 데이터 준비 완료 알림 핀 (DATA_READY)
#define PIN_NRESET   D2   // 하드웨어 리셋 핀

// ==========================================
// 3. 센서 통신 파라미터 및 버퍼 (Thermal-90 Module, 80x62, 90° FOV)
// ==========================================
#define I2C_ADDR_SENSOR 0x40      // I2C 센서 기본 주소
#define SPI_SPEED       2000000   // SPI 통신 속도 (2MHz)
#define TOTAL_WORDS     5040      // 프레임 헤더(80) + 픽셀(4960)
#define TOTAL_BYTES     (TOTAL_WORDS * 2) // 10,080 Bytes

// 5,040 Words 수신용 프레임 버퍼 (10,080 Bytes 할당)
uint16_t frameBuffer[TOTAL_WORDS];

// D_READY 인터럽트 플래그
volatile bool isFrameReady = false;

void IRAM_ATTR onDataReady() {
  isFrameReady = true;
}

void setup() {
  Serial.begin(115200);
  while (!Serial);
  Serial.println("\n[ ESP32 Thermal-90 Camera UDP Sender Init ]");

  // ----------------------------------------
  // Step A: Wi-Fi 접속
  // ----------------------------------------
  Serial.print("Connecting to Wi-Fi: ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected successfully.");
  Serial.print("ESP32 IP address: ");
  Serial.println(WiFi.localIP());

  // ----------------------------------------
  // Step B: 하드웨어 핀 초기화
  // ----------------------------------------
  pinMode(PIN_CS, OUTPUT);
  pinMode(PIN_NRESET, OUTPUT);
  pinMode(PIN_D_READY, INPUT); 
  attachInterrupt(digitalPinToInterrupt(PIN_D_READY), onDataReady, RISING); 

  digitalWrite(PIN_CS, HIGH);  // SPI CS 기본 상태 HIGH (대기)
  
  // 센서 하드웨어 리셋 시퀀스
  digitalWrite(PIN_NRESET, LOW);
  delay(10);
  digitalWrite(PIN_NRESET, HIGH);
  delay(100);

  // ----------------------------------------
  // Step C: I2C 초기화 (설정용)
  // ----------------------------------------
  Wire.begin(PIN_SDA, PIN_SCL);
  Wire.setClock(400000); // 400kHz Fast-mode
  
  Wire.beginTransmission(I2C_ADDR_SENSOR);
  if (Wire.endTransmission() == 0) {
    Serial.println("> I2C: Thermal-90 Sensor successfully found at 0x40.");
    
    // 센서 초기화(Boot-up) 명령 시퀀스 전송
    Serial.println("  - Sending Power-Up Command (Reg 0xB0)...");
    Wire.beginTransmission(I2C_ADDR_SENSOR);
    Wire.write(0xB0); // SENXOR_POWERUP
    Wire.write(0x13);
    Wire.endTransmission();
    
    Serial.print("  - Waiting for Boot-up to complete...");
    bool booted = false;
    for (int i = 0; i < 50; i++) {
      Wire.beginTransmission(I2C_ADDR_SENSOR);
      Wire.write(0xB6); // STATUS register
      Wire.endTransmission(false);
      Wire.requestFrom((uint16_t)I2C_ADDR_SENSOR, (uint8_t)1);
      if (Wire.available()) {
        uint8_t status = Wire.read();
        if ((status & 0x20) == 0) { // BOOTING_UP (0x20) bit clear
          booted = true;
          break;
        }
      }
      delay(100);
      Serial.print(".");
    }
    Serial.println(booted ? " OK!" : " Timeout!");

    Serial.println("  - Setting Frame Rate to ~7 FPS (Reg 0xB4)...");
    Wire.beginTransmission(I2C_ADDR_SENSOR);
    Wire.write(0xB4); // FRAME_RATE
    Wire.write(0x04);
    Wire.endTransmission();

    Serial.println("  - Starting Continuous Stream Mode (Reg 0xB1)...");
    Wire.beginTransmission(I2C_ADDR_SENSOR);
    Wire.write(0xB1); // FRAME_MODE
    Wire.write(0x02);
    Wire.endTransmission();
    delay(100);

  } else {
    Serial.println("> I2C: Warning! Sensor NOT found at 0x40.");
  }

  // ----------------------------------------
  // Step D: SPI 초기화 (데이터 획득용)
  // ----------------------------------------
  SPI.begin(PIN_CLK, PIN_MISO, PIN_MOSI, PIN_CS);
  Serial.println("> SPI: VSPI Bus Initialized.");
  
  Serial.println("[ Initialization Complete. Waiting for frames... ]\n");
}

void loop() {
  // ----------------------------------------
  // 프레임 획득: 인터럽트 플래그 확인 (RISING EDGE 감지)
  // ----------------------------------------
  if (isFrameReady) {
    isFrameReady = false; // 플래그 초기화
    
    // SPI 통신 세팅 적용: Mode 0, MSB First
    SPI.beginTransaction(SPISettings(SPI_SPEED, MSBFIRST, SPI_MODE0));
    digitalWrite(PIN_CS, LOW); // 통신 시작
    delayMicroseconds(100); // CS 안정화 딜레이 (제조사 권장)

    // 0x0000 더미 전송으로 클럭 발생 및 5040 Words 수신
    for (int i = 0; i < TOTAL_WORDS; i++) {
      frameBuffer[i] = SPI.transfer16(0x0000);
    }
    
    delayMicroseconds(100); // 통신 종료 안정화 딜레이
    digitalWrite(PIN_CS, HIGH); // 통신 종료
    SPI.endTransaction();

    // ----------------------------------------
    // 데이터 추출 및 검증 (데이터시트 기준)
    // ----------------------------------------
    uint16_t frameCounter = frameBuffer[0];
    uint16_t maxPixelRaw  = frameBuffer[5]; // 새 데이터시트 반영 오프셋 5
    uint16_t minPixelRaw  = frameBuffer[6]; // 새 데이터시트 반영 오프셋 6
    
    // 중앙 픽셀 위치 산출 (해상도 80 x 62 기준)
    int centerX = 40;
    int centerY = 31;
    int pixelDataOffset = 80;
    int centerIdx = pixelDataOffset + (centerY * 80 + centerX);
    
    uint16_t centerPixelRaw = frameBuffer[centerIdx];

    // 읽어온 주요 데이터 시리얼 출력
    Serial.println("====== [ Frame Received ] ======");
    Serial.print("Frame Counter : "); Serial.println(frameCounter);
    Serial.print("Max Pixel RAW : "); Serial.println(maxPixelRaw);
    Serial.print("Min Pixel RAW : "); Serial.println(minPixelRaw);
    Serial.print("Center Pixel  : "); Serial.println(centerPixelRaw);
    Serial.println("================================\n");

    // ----------------------------------------
    // UDP 데이터 전송
    // ----------------------------------------
    // 효율적인 속도 확보를 위해 배열의 메모리 주소를 캐스팅하여 10,080 바이트를 통째로 전송
    udp.beginPacket(receiverIP, receiverPort);
    udp.write((uint8_t*)frameBuffer, TOTAL_BYTES);
    udp.endPacket();

    Serial.println("Frame sent via UDP (10,080 Bytes).");
  }
}
