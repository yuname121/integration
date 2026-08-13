# SafeNest 전달 패키지 및 프로그램 실행·종료 가이드

## 1. 압축 폴더만 보내면 실행할 수 있는가?

이 저장소 루트에는 SafeNest 통합을 위해 작성·선정한 소스 코드, TFLite 모델, ESP32 firmware 원본, Raspberry Pi 실행 스크립트, 웹 대시보드, SQLite schema, 자동 테스트와 문서가 모두 포함되어 있다.

다만 ZIP만 풀면 의존성·Wi-Fi 비밀번호·하드웨어 설정까지 자동으로 생기는 것은 아니다. 받는 사람은 이 문서의 최초 설치를 수행해야 하며 Raspberry Pi가 인터넷에 연결되어 있어야 Python 의존성을 설치할 수 있다.

### 포함된 항목

- SafeNest TCP v1 Pi receiver와 strict parser
- Sensor State Manager와 freshness 처리
- 최신 `ondevice_ai/` 전체 snapshot, primary 모델 3개, mmWave candidate/experiment evidence와 inference adapter
- Risk Engine과 rule fallback
- FastAPI, WebSocket, SQLite repository
- 실시간 웹 대시보드
- loopback E2E 및 HIL 증거 수집 도구
- ESP-WROOM-32용 통합 ESP32 firmware와 `secrets.example.h`
- PHASE 1~10 및 HIL 상세 문서
- Pi 설치·점검·실행 스크립트

### 보안상 포함하지 않은 항목

- 실제 Wi-Fi 비밀번호가 든 `secrets.h`
- 개인 SSH key와 Raspberry Pi 계정 비밀번호
- 기존에 생성된 SQLite 운영 데이터
- `.venv`, Python package cache와 임시 파일

### 외부 설치가 필요한 항목

- Raspberry Pi OS 64-bit, Python 3.10 이상
- 인터넷을 통한 FastAPI, uvicorn, numpy, LiteRT 등의 Python package
- Arduino IDE와 ESP32 board core
- Arduino library `Sensirion I2C SCD4x`
- Arduino library `Seeed Arduino mmWave`
- 실제 센서, 배선, 전원, Raspberry Pi와 ESP32

## 2. 현재 지원 보드에 관한 중요 제한

현재 통합 firmware의 핀은 ESP-WROOM-32/Arduino `ESP32 Dev Module` 기준이다.

```text
I²C SDA/SCL: GPIO 21/22
PIR: GPIO 13
MR60 UART RX/TX: GPIO 16/17
Thermal SPI SCLK/MISO/MOSI/CS: GPIO 18/19/23/27
Thermal READY/RESET: GPIO 26/25
```

최종 목표 보드인 XIAO ESP32-C6에는 이 firmware를 그대로 업로드할 수 없다. XIAO C6 사용 시 핀, UART, SPI class와 FreeRTOS 동작을 포팅하고 compile/HIL 검증해야 한다. 이 제한 때문에 **XIAO C6 기준으로는 현재 ZIP만 받아 즉시 전체 시스템을 실행할 수 없다.**

## 3. 기존 실행 방식과 달라진 점

다음 기존 명령은 새 통합 패키지에 포함되지 않으며 실행하지 않는다.

```text
cd ~/raspberry_pi_lcd && bash start_lcd.sh
cd ~/SafeNest_Web && node server.js
http://RPI_IP:8080/thermal
http://RPI_IP:3000
```

새 통합 시스템은 Raspberry Pi에서 Python process 하나가 TCP 9000, HTTP 8000, SQLite와 대시보드를 함께 담당한다.

```text
ESP32 → TCP 9000 → SafeNest Runtime
                     ├─ FastAPI / WebSocket :8000
                     ├─ SQLite data/safenest.db
                     └─ Dashboard /dashboard
```

## 4. 받는 사람의 최초 설치 순서

### 4.1 압축 해제

전달받은 `integration_package.zip`을 Raspberry Pi home에 복사한다.

[Raspberry Pi에서 실행]

```bash
cd ~
unzip integration_package.zip
test -f ~/integration/deployment/run_pi.sh
```

압축 해제 결과가 `~/integration/`인지 확인한다.

Windows에서 먼저 확인할 경우 최신 mmWave 실험 파일 중 경로가 긴 항목이 있으므로, ZIP을 드라이브 루트에 가까운 짧은 경로(예: `C:\safenest`)에 7-Zip 또는 Python `python -m zipfile -e`로 푼다. Windows 기본 `Expand-Archive`는 긴 경로에서 중단될 수 있다. Raspberry Pi의 위 `unzip` 절차에는 해당 Windows 경로 제한이 없다.

### 4.2 OS 기본 package

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip python3-dev build-essential unzip
```

### 4.3 Python 의존성 설치와 최초 실행

```bash
cd ~
cd ~/integration
bash deployment/run_pi.sh --install
```

이 명령은 다음을 순서대로 수행한다.

1. `~/.venv` 생성
2. backend 및 Pi AI 의존성 설치
3. 필수 파일과 Python runtime 확인
4. manifest에 등록된 primary 3개와 보존된 후보 artifact의 SHA-256 확인

최신 mmWave BPF+Z-score 경로는 `runtime_manifest_path`를 명시적으로 전달하는 phase 전용 기능이다. 현재 SafeNest 통합은 운영 primary manifest를 사용하며, `deployment_ready=false` 오프라인 후보를 자동 선택하지 않는다. phase 후보를 별도로 실험할 때만 `scipy`와 해당 runtime manifest 계약을 추가 설치·검증한다.
5. TCP 9000과 HTTP 8000 포트 확인
6. FastAPI와 전체 sensor runtime 실행

설치 또는 LiteRT package가 Pi의 Python version에서 실패하면 그 오류를 해결하기 전에는 실행 완료로 간주하지 않는다.

## 5. ESP32 최초 준비

동결 원본에 비밀번호 파일을 추가하지 말고 Arduino sketch 작업 복사본을 만든다.

[Windows 노트북 PowerShell에서 실행]

```powershell
$repo = "압축을_푼_폴더의_상위_경로"
$source = Join-Path $repo "sources\display-test2\esp32_sensor_node"
$sketch = Join-Path $HOME "Documents\Arduino\esp32_sensor_node"
Copy-Item -LiteralPath $source -Destination $sketch -Recurse
Copy-Item -LiteralPath (Join-Path $sketch "secrets.example.h") -Destination (Join-Path $sketch "secrets.h")
```

`secrets.h`를 다음처럼 실제 환경에 맞춘다.

```cpp
constexpr char WIFI_SSID[] = "EELab04 2G";
constexpr char WIFI_PASSWORD[] = "실제_비밀번호";
constexpr char RPI_HOST[] = "Raspberry_Pi_IP";
constexpr uint16_t RPI_PORT = 9000;
```

Arduino IDE에서 다음을 확인한다.

1. Board: `ESP32 Dev Module`
2. 올바른 ESP-WROOM-32 USB port
3. `Sensirion I2C SCD4x`, `Seeed Arduino mmWave` library 설치
4. 센서 배선과 전압 확인
5. firmware upload
6. Serial Monitor 115200 baud에서 Wi-Fi와 TCP 연결 확인

## 6. 매번 프로그램 실행 순서

1. 노트북 Wi-Fi를 `EELab04 2G`로 연결한다.
2. Raspberry Pi 전원을 켠다.
3. Raspberry Pi가 부팅될 때까지 기다린다.
4. VS Code Remote SSH로 Raspberry Pi에 연결한다.
5. VS Code에서 터미널을 하나 연다.
6. Raspberry Pi 통합 프로그램을 실행한다.

```bash
cd ~/integration
bash deployment/run_pi.sh
```

7. 터미널에서 Uvicorn HTTP 8000과 sensor TCP 9000 listener가 시작됐는지 확인한다.
8. ESP-WROOM-32와 센서 전원을 켠다.
9. ESP32 Serial Monitor 또는 Pi 로그에서 TCP 연결과 telemetry 수신을 확인한다.
10. 노트북 브라우저에서 다음 주소를 연다.

```text
통합 웹 대시보드: http://RPI_IP:8000/dashboard
상태 API:         http://RPI_IP:8000/api/status
시스템 진단:      http://RPI_IP:8000/health
API 문서:         http://RPI_IP:8000/docs
```

11. 대시보드에서 `ONLINE`, 센서별 `LIVE`, 현재 값, risk와 최근 갱신 시각을 확인한다.

## 7. 프로그램 종료 순서

1. 브라우저 사용을 종료한다.
2. `run_pi.sh`가 실행 중인 Pi 터미널에서 `Ctrl+C`를 한 번 누른다.
3. Uvicorn shutdown과 backend 종료 메시지가 끝날 때까지 기다린다.
4. ESP32와 센서 전원을 분리한다.
5. Raspberry Pi에서 안전 종료를 실행한다.

```bash
sudo shutdown -h now
```

6. SSH 연결이 끊어지는 것을 확인한다.
7. Raspberry Pi activity LED가 멈춘 뒤 전원선을 분리한다.

별도의 `node server.js`, `start_lcd.sh`, `stop_lcd.sh` process는 새 통합 시스템에 없으므로 시작하거나 종료할 필요가 없다.

## 8. 정상 실행 확인

[Raspberry Pi의 별도 SSH 터미널에서 실행]

```bash
curl -fsS http://127.0.0.1:8000/health | python -m json.tool
curl -fsS http://127.0.0.1:8000/api/status | python -m json.tool
```

필수 확인 항목:

- `/health`의 `ok`가 `true`
- database `available`이 `true`
- receiver `connections`가 1 이상
- `/api/status`의 `ready`가 `true`
- 센서 연결 후 system `ONLINE`
- SQLite snapshot count가 계속 증가

## 9. 문제 발생 위치 구분

| 증상 | 먼저 확인할 계층 |
|---|---|
| ESP32가 Wi-Fi에 연결되지 않음 | `secrets.h`, 2.4 GHz SSID/비밀번호 |
| ESP32 TCP 연결 실패 | `RPI_HOST`, port 9000, Pi firewall, runtime 실행 여부 |
| 모든 센서 DISCONNECTED | ESP32 전원/TCP와 Pi receiver |
| 한 센서만 INVALID/STALE | 해당 센서 배선·driver·freshness |
| Thermal AI unavailable | LiteRT 설치, 모델 hash, frame 유효성 |
| API는 되지만 DB unavailable | DB 경로 권한과 SQLite 오류 |
| 브라우저 접속 실패 | port 8000, Pi IP, 같은 Wi-Fi 여부 |
| 대시보드 값이 갱신되지 않음 | `/api/status`, `/ws`, sensor timestamp |

## 10. 전달 전 체크리스트

- standalone 저장소 루트에 `README.md`, `ai/`, `backend/`, `gateway/`, `sources/`, `tests/`, `web/`가 있는가?
- `PACKAGE_AND_OPERATION_GUIDE.md`와 `INTEGRATION_PHASE_SUMMARY.md`가 있는가?
- `sources/ondevice_ai/models/`와 manifest 등록 primary/candidate artifact가 있는가?
- `requirements-backend.txt`와 `requirements-pi.txt`가 있는가?
- 실제 `secrets.h`, `.venv`, DB, SSH key가 포함되지 않았는가?
- 받는 사람이 WROOM-32와 XIAO C6 제한을 이해했는가?
- 실제 장비 HIL 미완료와 TEST 4/6 제약을 전달했는가?
