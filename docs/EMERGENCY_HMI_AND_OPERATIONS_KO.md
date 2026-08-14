# SafeNest 긴급 대응 HMI·운영 가이드

이 문서는 통합 패키지의 긴급 대응 기능을 Raspberry Pi와 터치 화면에서 시연하는 방법을 정리한다. 긴급 기능은 기존 TCP 수신기 → State Manager → AI/Risk Engine → FastAPI/WebSocket → SQLite 구조 위에 연결되어 있으며, 프론트엔드가 위험도를 다시 계산하지 않는다.

## 1. 안전 범위

- `DANGER` 전환이 처음 발생할 때만 서버가 알람 래치와 GPIO/mock buzzer를 활성화한다.
- 반복되는 `DANGER` WebSocket 패킷은 같은 `transition_id`를 유지하며 알람음·오버레이·이벤트를 중복 시작하지 않는다.
- `WARNING`은 긴급 오버레이를 열지 않는다. `DANGER`에서 `WARNING` 또는 `NORMAL`로 복귀한 실제 live publication만 알람을 해제한다.
- WebSocket이 끊겨 polling으로 전환되거나 위험도 계산값이 사라져도 기존 DANGER는 자동 해제하지 않는다.
- `경고 확인`은 buzzer만 끄고 Risk Engine의 위험 단계와 DANGER 래치를 바꾸지 않는다.
- `119 모의 신고`는 카운트다운과 이벤트 기록만 수행한다. 실제 119·소방·공공 긴급망과 통신하지 않는다.
- 담당자 SMS는 브라우저가 아니라 백엔드가 설정된 Naver Cloud SENS 자격증명으로만 요청한다. 자격증명이 없으면 성공으로 가장하지 않고 `SMS_NOT_CONFIGURED` 오류를 반환한다.

## 2. 최초 설정

패키지 루트에서 실행한다.

```bash
cd ~/integration
cp .env.example .env
${EDITOR:-vi} .env
```

`.env`에는 실제 비밀값을 넣을 수 있지만 Git에 추가하거나 ZIP에 포함하지 않는다. `run_pi.sh`는 이 파일이 있으면 실행 전에 환경변수로 읽는다.

### 환경변수

| 변수 | 용도 | 기본값/주의 |
|---|---|---|
| `SMS_ACCESS_KEY` | Naver Cloud API access key | 실제 SMS를 사용할 때만 입력 |
| `SMS_SECRET_KEY` | Naver Cloud API secret key | 로그·브라우저로 전달하지 않음 |
| `SMS_SERVICE_ID` | SENS 서비스 ID | 실제 서비스 ID |
| `SMS_FROM_NUMBER` | SENS 발신번호 | SENS에 등록된 번호 |
| `MANAGER_PHONE_NUMBER` | 담당자 수신번호 | 서버에만 보관, 화면에는 마스킹 표시 |
| `MANAGER_NAME` | 담당자 표시명 | `안전 담당자` |
| `SMS_API_BASE_URL` | SENS API base URL | `https://sens.apigw.ntruss.com` |
| `SMS_TIMEOUT_SECONDS` | SMS 요청 제한시간 | `8`초 |
| `SAFENEST_SMS_COOLDOWN_SECONDS` | SMS 성공 후 재전송 제한 | `60`초 |
| `SAFENEST_GPIO_MODE` | `auto`, `mock`, `off` | 개발 PC는 `mock` 권장 |
| `SAFENEST_BUZZER_GPIO_PIN` | BCM buzzer 핀 | `18` |
| `SAFENEST_BUZZER_FREQUENCY_HZ` | PWM 주파수 | `880` |

Raspberry Pi에서 실제 SMS를 보내려면 Naver Cloud SENS 발신번호와 서비스 ID를 먼저 검증하고, 소액 테스트 번호로 확인한다. 이 패키지는 테스트용 SMS provider를 기본으로 사용하지 않는다.

## 3. 설치·실행

Raspberry Pi에서 다음을 실행한다.

```bash
cd ~
cd ~/integration
bash deployment/run_pi.sh --install
bash deployment/run_pi.sh
```

통합 프로세스가 기본적으로 다음을 제공한다.

```text
ESP32 → TCP 9000 → Runtime / State / AI / Risk
                         ├─ FastAPI + WebSocket :8000
                         ├─ SQLite data/safenest.db
                         └─ Dashboard /dashboard
```

터치 HMI는 Chromium에서 같은 origin으로 연다.

```bash
chromium-browser --kiosk --app=http://127.0.0.1:8000/dashboard
```

일반 PC에서는 GPIO가 자동으로 mock fallback으로 바뀐다. 명시적으로 다음처럼 실행하면 buzzer 하드웨어를 사용하지 않는다.

```bash
SAFENEST_GPIO_MODE=mock bash deployment/run_pi.sh
```

## 4. API 계약

화면의 모든 변경 동작은 같은 origin의 백엔드 API로 전송한다.

| API | 동작 |
|---|---|
| `GET /api/emergency/state` | 알람 래치·확인·buzzer 현재 상태 |
| `POST /api/emergency/119/simulation/start` | DANGER에서만 모의 신고 시작 |
| `POST /api/emergency/119/simulation/complete` | 모의 신고 완료 기록 |
| `POST /api/emergency/contact` | 설정된 담당자에게 서버 측 SMS 요청 |
| `POST /api/emergency/acknowledge` | buzzer 음소거, DANGER 유지 |
| `POST /api/emergency/voice` | 로컬 음성 동작 이벤트 기록 |
| `POST /api/client-connection` | WebSocket/polling online·offline 기록 |

API 오류는 `ok: false`, `error_code`, `message` 구조로 반환한다. SMS 응답에는 수신번호 전체를 포함하지 않고 `010-****-1234` 형식의 마스킹 번호만 포함한다.

## 5. 터치 시연 순서

1. 대시보드에서 `ONLINE`, 센서 `LIVE`, 위험도 `NORMAL`을 확인한다.
2. 센서 또는 E2E 시나리오로 Risk Engine의 실제 `DANGER` 전환을 발생시킨다.
3. 대시보드가 전체 화면 긴급 오버레이, 위험 점수, 감지 원인, 호흡수·CO₂·무움직임 값을 표시하는지 확인한다.
4. `119 모의 신고`를 누르고 안내 문구와 3·2·1 카운트다운을 확인한다. 화면과 이벤트에는 실제 119 연결이 아니라는 고지가 남는다.
5. `담당자 연락`을 누른다.
   - SMS 환경변수가 설정된 장비: 백엔드 전송 성공, 요청 ID, 마스킹 번호, 60초 쿨다운 확인
   - 설정되지 않은 장비: 실패 상태와 `SMS_NOT_CONFIGURED` 확인
6. `경고 확인`을 누른다. buzzer는 꺼지지만 위험 단계는 `DANGER`로 남는지 확인한다.
7. 센서를 live 상태로 복구해 `WARNING` 또는 `NORMAL` publication을 만들고, 그때만 오버레이와 래치가 해제되는지 확인한다.
8. 브라우저에서 WebSocket을 끊거나 네트워크를 차단해 polling 전환, offline 배너, DANGER 유지, `SENSOR_OFFLINE`/연결 이벤트를 확인한다.

## 6. 오디오와 GPIO 확인

브라우저 음성 파일은 다음 경로에 팀이 녹음한 MP3를 배치한다.

```text
web/dashboard/audio/system_start.mp3
web/dashboard/audio/warning.mp3
web/dashboard/audio/danger.mp3
web/dashboard/audio/report_119.mp3
web/dashboard/audio/report_119_complete.mp3
web/dashboard/audio/sms_sent.mp3
web/dashboard/audio/sms_failed.mp3
web/dashboard/audio/sensor_offline.mp3
```

파일이 없거나 브라우저 autoplay가 차단되어도 화면·API 동작은 계속되어야 한다. Raspberry Pi가 아닌 환경에서 `auto` GPIO 초기화가 실패하면 상태에 `mock_fallback`이 표시된다. 실제 Pi 배선에서는 buzzer의 GND, 전원·저항·트랜지스터 구성과 BCM 핀 번호를 먼저 확인한다.

## 7. 종료·검증

```bash
curl -fsS http://127.0.0.1:8000/health | python -m json.tool
curl -fsS http://127.0.0.1:8000/api/emergency/state | python -m json.tool
```

정상 종료는 실행 터미널에서 `Ctrl+C` 한 번으로 수행한다. 개발 PC 또는 CI에서 긴급 기능만 검증하려면 저장소 루트에서 다음을 실행한다.

```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
node --check web/dashboard/app.js
```

실제 SMS 발송, 실제 GPIO 소리, Raspberry Pi 카메라·터치 입력은 장비와 외부 계정이 있어야 확인할 수 있으므로 표준 단위 테스트의 성공과 구분한다.
