# SafeNest 실제 장비 HIL 실행 가이드

## 1. Raspberry Pi 준비 및 실행

[Raspberry Pi에서 실행]

```bash
cd ~/integration
bash ./deployment/run_pi.sh --install
```

이 명령은 `.venv` 생성, FastAPI/uvicorn/LiteRT 의존성 설치, Python·필수 파일·모델 SHA-256·8000/9000 포트 확인 후 통합 gateway/backend를 시작한다. 다음 실행부터는 설치 옵션을 제외한다.

```bash
cd ~/integration
bash ./deployment/run_pi.sh
```

다른 포트가 필요하면 backend 인자를 그대로 전달한다.

```bash
bash ./deployment/run_pi.sh --api-port 8080 --sensor-port 9100
```

정상 기동 후 별도 터미널에서 확인한다.

```bash
curl -fsS http://127.0.0.1:8000/health | python -m json.tool
curl -fsS http://127.0.0.1:8000/api/status | python -m json.tool
```

## 2. ESP32 Arduino 작업 복사본

동결 원본과 Wi-Fi 비밀정보를 섞지 않는다.

[Windows 노트북 PowerShell에서 실행]

```powershell
$repo = "C:\path\to\integration"
$source = Join-Path $repo "./sources\display-test2\esp32_sensor_node"
$sketch = Join-Path $HOME "Documents\Arduino\esp32_sensor_node"
Copy-Item -LiteralPath $source -Destination $sketch -Recurse
Copy-Item -LiteralPath (Join-Path $sketch "secrets.example.h") -Destination (Join-Path $sketch "secrets.h")
```

`secrets.h`에 실제 SSID, 비밀번호, Raspberry Pi IP를 입력한 뒤 Arduino IDE에서 `esp32_sensor_node.ino`를 **ESP-WROOM-32/ESP32 Dev Module**에 flash한다. Serial Monitor의 sensor freshness, TCP connect/disconnect, send failure 로그를 각 시나리오 증거와 함께 보존한다.

현재 통합 firmware의 GPIO/UART/SPI 설정은 XIAO ESP32-C6 핀맵과 호환되지 않는다. XIAO C6에는 별도 포팅과 compile/HIL 검증 전 업로드하지 않는다.

## 3. HIL evidence collector

collector는 `/api/status`와 `/health`를 지정 시간 동안 polling하고 원본 표본, 판정 check, 시작/종료 시각을 UTF-8 JSON으로 저장한다.

공통 형식:

```bash
source ~/integration/.venv/bin/activate
cd ~/integration
python -m hil.capture \
  --scenario <scenario-name> \
  --base-url http://127.0.0.1:8000 \
  --duration 20 \
  --interval 1
```

종료 코드는 `0=PASS`, `1=FAIL`, `2=INCONCLUSIVE`이다. 보고서는 기본적으로 `./hil/reports/`에 생성된다.

## 4. 물리 시나리오 실행

각 명령을 실행한 직후 설명된 물리 동작을 수행한다.

### TEST 1 — 사람 없음

감지 영역을 비우고 Thermal frame이 안정된 뒤 실행한다.

```bash
python -m hil.capture --scenario test01_no_person --duration 20
```

### TEST 2 — 사람 있음 + 정상 호흡

사람 한 명이 감지 영역에서 정상적으로 호흡하며 가볍게 움직인다.

```bash
python -m hil.capture --scenario test02_person_normal --duration 30
```

### TEST 3 — 정지 인체

사람이 정상 호흡을 유지하면서 움직이지 않는다. 즉시 DANGER로 오탐하지 않는지 확인한다.

```bash
python -m hil.capture --scenario test03_stationary_person --duration 10
```

15초 장기 무움직임 규칙은 별도로 25초 이상 수집해 event 전환을 확인한다.

### TEST 4 — 호흡 이상 입력

사람에게 위험한 호흡 행동을 요구하지 않는다. MR60 검증용 재생 데이터 또는 안전한 센서 simulator로 12 rpm 미만/20 rpm 초과 값을 주입한다.

```bash
python -m hil.capture --scenario test04_abnormal_breathing --duration 20
```

현재 locked V4는 5 rpm, CO₂ 700 ppm 조건에서 29.75점 NORMAL이므로 acceptance는 FAIL할 것으로 예상된다. 공식 threshold 결정 전에는 판정기를 완화하지 않는다.

### TEST 5 — CO₂ 상승

사람이 밀폐된 고농도 CO₂를 흡입하는 시험은 금지한다. 교정용 가스/안전 챔버 또는 SCD40 simulator를 사용한다.

```bash
python -m hil.capture --scenario test05_co2_rise --duration 90 --interval 2
```

### TEST 6 — mmWave false positive

mmWave에는 존재 신호, Thermal에는 `NOT_HUMAN`이 나오도록 안전한 표적을 배치한다.

```bash
python -m hil.capture --scenario test06_mmwave_false_positive --duration 30
```

현재 TCP v1은 mmWave presence를 보내지 않으므로 `INCONCLUSIVE`가 정상적인 차단 결과다.

### TEST 7 — 비생체 열원

사람 없이 안전한 온도의 온열팩 등 비생체 열원을 Thermal 시야에 둔다. 화염이나 과열 장치는 사용하지 않는다.

```bash
python -m hil.capture --scenario test07_thermal_nonhuman --duration 30
```

### TEST 8 — TCP 단절 및 복구

수집 중 ESP32의 Wi-Fi를 끊었다가 복구한다. `DISCONNECTED/STALE → LIVE`와 receiver connection 증가가 모두 필요하다.

```bash
python -m hil.capture --scenario test08_thermal_disconnect --duration 45
```

### TEST 9 — ESP32 재부팅

수집 중 ESP32 reset 버튼을 한 번 누른다. 새 연결, sequence 감소, 최종 ONLINE을 확인한다.

```bash
python -m hil.capture --scenario test09_esp32_reboot --duration 45
```

### TEST 10 — AI runtime 장애

운영 모델 파일을 삭제하거나 덮어쓰지 않는다. 별도 HIL 작업 복사본에서 잘못된 모델 경로를 지정하거나 LiteRT import를 차단한 프로세스로 실행한다.

```bash
python -m hil.capture --scenario test10_ai_failure --duration 20
```

Thermal AI unavailable 중에도 risk가 계산되고 `/health`와 SQLite가 살아 있어야 한다.

## 5. 완료 판정

HIL 완료에는 다음 자료가 모두 필요하다.

- 10개 collector JSON 보고서
- Raspberry Pi backend 로그
- ESP32 Serial Monitor 로그
- `/health`의 receiver/SQLite 진단
- TEST 8과 9의 연결 전환 시각
- TEST 4 정책 결정 기록
- TEST 6 protocol revision 또는 명시적인 acceptance 제외 결정

실제 하드웨어가 없는 개발 환경에서 생성한 loopback 결과는 물리 HIL 보고서를 대체하지 않는다.
