# 센서 타이밍 및 rolling dataset 운영

## 실행 계약

- ESP32의 센서 수신과 TCP 전송은 기존 주기를 유지한다.
- Raspberry Pi는 유효 CO2 값을 60초마다 한 번 채택한다. `last_received_at`과 통신 상태는 매 telemetry 수신에 따라 별도로 갱신된다.
- Risk Engine은 시작 시 한 번 평가한 뒤 15초마다 최신 Sensor State로 평가한다.
- 다음 평가 전까지 `RuntimeStore`의 기존 publication을 그대로 유지하므로 FastAPI, WebSocket, Web Dashboard와 LCD compatibility view가 같은 risk 상태를 표시한다.
- mmWave, Thermal, PIR 수신 및 내부 Sensor State 갱신은 15초 risk scheduler와 독립적이다.

## 저장 형식

기본 위치는 기존 SQLite 위치와 같은 저장소 루트의 `data/` 아래다.

- `data/mmwave/YYYYMMDD_HH_mmwave.jsonl`: 수신 timestamp, device/sequence/uptime, 호흡수, 심박수, 각 validity. 현재 protocol에 raw respiration phase/window가 없으므로 존재하는 최소 가공 scalar만 저장한다.
- `data/co2/YYYYMMDD_HH_co2.jsonl`: 확장 sender에서는 `(device_id, boot_id, co2_measurement_event_id)`로 중복 제거한 실제 SCD4x measurement event, source measurement monotonic time, Pi receive time과 CO2 ppm을 저장한다. legacy sender는 event provenance가 없으므로 기존 60초 fallback을 유지한다. 온도·습도는 B-complete candidate 입력이 아니므로 전송·저장하지 않는다.
- `data/thermal/YYYYMMDD_HHMMSS_microseconds_first-last.npz`: 여러 raw uint16 80×62 frame, 수신 timestamp, frame sequence, source uptime, raw min/max, 해당 시점의 최신 Thermal AI 및 risk context JSON. `allow_pickle=False`로 재생할 수 있고 원시 pixel 수치를 손실 없이 보존한다.

수신 callback은 bounded memory queue에 `put_nowait()`만 수행한다. JSONL/NPZ write, compression 및 cleanup은 `safenest-sensor-data-writer` thread가 담당한다. 디스크가 느려 queue가 가득 차면 실시간 통신을 막지 않고 새 logging item을 drop하며 diagnostics의 `dropped`에 기록한다.

## 기본 저장 정책

`.env`에서 다음 값을 변경할 수 있다. GB는 10억 byte 기준이다.

```text
SAFENEST_SENSOR_DATA_MAX_GB=10
SAFENEST_MIN_FREE_DISK_GB=2
SAFENEST_MMWAVE_DATA_MAX_GB=1
SAFENEST_CO2_DATA_MAX_GB=0.25
SAFENEST_THERMAL_DATA_MAX_GB=8.5
```

먼저 센서별 quota를 넘긴 디렉터리에서 timestamp/mtime 기준 가장 오래된 완료 파일을 지운다. 다음으로 전체 sensor dataset limit를 적용하고, 마지막으로 filesystem 여유 공간이 2 GB 미만이면 전체 센서 중 가장 오래된 완료 파일부터 지운다. 현재 쓰는 JSONL segment와 임시 NPZ는 cleanup 대상에서 제외된다. SQLite DB는 sensor dataset 용량 계산과 삭제 대상에 포함하지 않는다.

## 저장량 추정

실제 firmware 설정을 사용한 압축 전 보수적 추정이다.

| 센서 | 확인된 주기/크기 | 시간당 | 일당 |
|---|---:|---:|---:|
| mmWave | telemetry 1 Hz, JSONL 약 180~250 B/record | 약 0.65~0.90 MB | 약 15.6~21.6 MB |
| CO2 | 60초당 1 record, JSONL 약 120~180 B | 약 7.2~10.8 KB | 약 0.17~0.26 MB |
| Thermal | raw 9,920 B/frame, 설정 약 6.25 FPS | 약 223.2 MB | 약 5.36 GB |

Thermal NPZ는 frame 내용에 따라 압축률이 달라 실제 사용량은 달라진다. metadata와 ZIP container overhead도 있으므로 운영 중 `data/thermal` 실측 증가량을 확인해야 한다. 기본 Thermal quota 8.5 GB는 압축이 전혀 없다는 보수적 기준으로 약 38시간 분량이다.

## 장애 의미

- `LIVE`, `STALE`, `DISCONNECTED`, `INVALID`, `NO_DATA`는 Sensor State의 통신/신선도 사실이다.
- `INPUT_UNAVAILABLE`/AI unavailable은 AI component 상태이며 가능한 기존 rule fallback을 방해하지 않는다.
- 최종 risk는 위 상태를 15초 평가 경계에서 반영한다. 센서 값이나 통신 상태를 정상값으로 위조하지 않는다.
