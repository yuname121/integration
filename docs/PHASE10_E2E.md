# PHASE 10 — 전체 End-to-End 테스트

## 검증 경로

`e2e/harness.py`는 실제 loopback TCP socket으로 SafeNest TCP v1 telemetry와 80×62 Thermal frame을 전송한다. 데이터는 운영 코드와 같은 경로를 지난다.

```text
TCP v1 receiver
→ strict parser / sequence tracker
→ SensorStateManager
→ OnDeviceAIPipeline
→ SafeNestRiskEngine
→ PersistentRuntimeStore(SQLite :memory:)
→ status API view
```

Thermal 모델 출력만 `ScriptedThermalModel`로 결정적으로 주입한다. 이는 파싱·전처리·장애 격리·후처리 계약을 검증하지만 실제 TFLite 정확도나 Pi LiteRT 호환성을 검증한 것으로 간주하지 않는다.

## 시나리오 결과

| Test | 자동 검증 결과 | 현재 판정 | 남은 검증 |
|---|---|---|---|
| 1. 사람 없음 | Thermal `NO_HUMAN`, fused presence false, risk NORMAL | PASS (software loopback) | 실제 빈 공간 HIL |
| 2. 사람 있음 + 정상 호흡 | Thermal presence, 15 rpm, motion, risk NORMAL | PASS (software loopback) | 실제 센서/TFLite HIL |
| 3. 사람 있음 + 즉시 무움직임 | PIR 0.5, 긴급/장기 무움직임 미승격 | PASS (software loopback) | 실제 15초 연속 HIL |
| 4. 호흡 이상 | 5 rpm → mmWave 0.75, 총 29.75, risk NORMAL | **INCONSISTENCY** | 요구값 WARNING/DANGER와 V4 잠금 계약 재결정 필요 |
| 5. CO₂ 상승 | 2,600 ppm, CO₂ component 1.0, HIGH_CO2_DANGER, risk WARNING | PASS (software loopback) | SCD40 가스/챔버 HIL |
| 6. mmWave false positive | downstream mismatch rule 확인 | **BLOCKED AT PROTOCOL** | TCP v1에 mmWave presence가 없음 |
| 7. 비생체 열원 | 주입된 `NO_HUMAN` 출력은 presence false 유지 | PASS (contract only) | 실제 모델 정확도 HIL 필수 |
| 8. TCP disconnect | disconnect 후 새 socket과 fragmented frame 수신, ONLINE 복귀 | PASS (Pi receiver loopback) | ESP32 자동 재연결 HIL |
| 9. ESP32 재부팅 | 새 연결에서 sequence 100→1 reset 허용, protocol error 0 | PASS (Pi receiver loopback) | 실제 전원 재부팅 HIL |
| 10. AI failure | Thermal failure 격리, mmWave/CO₂ rule fallback, SQLite/API 유지 | PASS (software loopback) | Pi에서 LiteRT 강제 실패 확인 |

자동 테스트 10개는 모두 현재 코드의 동작과 알려진 불일치를 회귀 계약으로 고정한다. 따라서 TEST 4와 TEST 6의 테스트가 통과한다는 것은 제품 acceptance가 충족됐다는 뜻이 아니라, 불일치가 조용히 사라지거나 추측값으로 대체되지 않는다는 뜻이다.

## PHASE 10에서 수정한 통합 누락

Risk Engine이 이미 계산하던 융합 presence가 결과에 노출되지 않던 문제를 수정했다.

- `presence_detected`: 현재 융합된 사람 존재 boolean
- `presence_source`: `MMWAVE`, `THERMAL`, `UNCONFIRMED`

대시보드는 실제 mmWave presence가 있으면 이를 우선하고, 현재 v1처럼 미제공이면 검증된 Thermal 융합 결과를 출처와 함께 표시한다. 센서값을 새로 추정하지 않는다.

## 자동 테스트 명령

[Windows 노트북에서 실행]

```powershell
cd C:\path\to\integration
python -m unittest tests.test_end_to_end -v
python -m unittest discover -s ./tests -p "test_*.py" -v
```

[Raspberry Pi에서 실행]

```bash
cd ~/integration
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r ./requirements-backend.txt
python -m unittest tests.test_end_to_end -v
python ./backend/run_backend.py
```

대시보드는 `http://<raspberry-pi-ip>:8000/dashboard`, 센서 TCP listener는 기본 `0.0.0.0:9000`이다.

## 실제 하드웨어 완료 조건

[Arduino IDE에서 실행]

1. `./sources/display-test2/esp32_sensor_node/esp32_sensor_node.ino`를 연다.
2. 같은 폴더의 `secrets.example.h`를 `secrets.h`로 복사하고 실제 Wi-Fi와 Pi 주소를 입력한다.
3. 현재 통합 firmware와 호환되는 ESP-WROOM-32/ESP32 Dev Module로 flash한 뒤 Serial Monitor에서 sensor poll과 TCP reconnect 로그를 보존한다. XIAO ESP32-C6에는 포팅 전 업로드하지 않는다.
4. Pi의 `/health`, `/api/status`, `/api/history`, `/api/events` 응답과 SQLite snapshot 증가를 함께 캡처한다.

HIL 완료 판정에는 빈 공간, 실제 사람, 정지 사람 15초 이상, 비생체 열원, CO₂ 상승, 케이블/전원 단절, ESP32 reboot, LiteRT 장애를 각각 독립 수행한 증거가 필요하다.

## 해결이 필요한 두 결정

1. mmWave presence를 생산하는 검증된 MR60 API가 정해지면 ESP32 sender와 Pi decoder를 같은 protocol revision으로 함께 확장해야 한다. v1 수신기가 모르는 값을 임의로 생성해서는 안 된다.
2. 비정상 호흡 단독으로 WARNING을 요구한다면 locked V4의 fallback score 또는 임계값을 공식적으로 변경해야 한다. 현재 0.75 × 35점에 정상 CO₂ 700 ppm 기여 3.5점을 더한 29.75점은 WARNING 30점보다 낮다.
