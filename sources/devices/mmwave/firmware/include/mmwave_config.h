#pragma once

#include <Arduino.h>

namespace safenest::mmwave_config {

constexpr char kSchemaVersion[] = "1.2";
constexpr char kDeviceId[] = "safenest-node-01";
constexpr char kEspFirmwareVersion[] = "safenest-mr60-esp/1.2.0";
constexpr char kConfigSha256[] =
    "b817e8bfd5e52b18275626f7b6a9bd60098ea4b108428a5aaf63600dbc987834";

constexpr uint32_t kUsbBaud = 115200;
constexpr uint32_t kRadarBaud = 115200;
constexpr int kRadarRxPin = 16;
constexpr int kRadarTxPin = 17;
constexpr uint32_t kTelemetryIntervalMs = 100;
constexpr uint32_t kFrameTimeoutMs = 1000;
constexpr uint8_t kFaultConsecutiveUartErrors = 5;
constexpr uint8_t kPresenceWindowSamples = 3;
constexpr uint8_t kPresenceRequiredTrue = 2;
constexpr uint8_t kPresenceRequiredFalse = 2;
constexpr uint32_t kWarmupMs = 60000;
constexpr uint32_t kPhaseMaxAgeMs = 500;
constexpr uint32_t kDistanceMaxAgeMs = 1000;
constexpr uint32_t kVitalMaxAgeMs = 2000;
constexpr float kDistanceMinCm = 40.0F;
constexpr float kDistanceMaxCm = 150.0F;
constexpr uint32_t kBreathWindowMs = 30000;
constexpr size_t kBreathWindowCapacity = 640;
constexpr float kBreathHysteresisFraction = 0.15F;
constexpr float kBreathMinPhaseStd = 0.2F;
constexpr size_t kBreathMinCrossings = 2;
constexpr uint32_t kWindowReadyToleranceMs = 200;

static_assert(kPresenceWindowSamples <= 8, "presence history uses uint8_t");
static_assert(kPresenceRequiredTrue <= kPresenceWindowSamples,
              "presence true quorum exceeds window");
static_assert(kPresenceRequiredFalse <= kPresenceWindowSamples,
              "presence false quorum exceeds window");

}  // namespace safenest::mmwave_config
