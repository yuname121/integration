# PHASE 9 — 실시간 관제 대시보드

## 결과

`web/dashboard/`에 외부 CDN이나 빌드 도구가 필요 없는 반응형 대시보드를 추가했다. FastAPI가 `/dashboard`와 `/dashboard/`에서 HTML을, `/dashboard/assets/`에서 CSS와 JavaScript를 같은 origin으로 제공한다.

화면은 다음 통합 값을 그대로 표시한다.

- 전체 `NORMAL/WARNING/DANGER` 위험 단계, 점수, 판단 근거, system health
- mmWave 호흡수·제공 가능한 심박수·presence·AI·risk component
- Thermal AI 분류·human 확률·raw 범위·risk component·정규화 heatmap
- CO₂ ppm·경계 상태·risk component
- PIR 움직임·rule 상태·무움직임 시간·risk component
- SQLite 최근 위험도/CO₂ snapshot 추이와 전환 이벤트

## 실시간 연결

브라우저는 우선 `/ws`를 연결한다. 연결 실패 또는 종료 시 2초 간격 `/api/status` polling으로 자동 전환하고, 2.5초 간격으로 WebSocket 재연결을 시도한다. `/api/events`와 `/api/history`는 약 5초마다 갱신한다. DOM 문자열 삽입은 `textContent`를 사용하며 상태를 변경하는 POST 요청은 없다.

## Thermal 표시 계약

원본 80×62 U16 frame은 상태 JSON이나 SQLite에 저장하지 않는다. AI pipeline이 현재 frame을 메모리에서 20×16으로 축소하고 frame 내부 min-max로 0–1 정규화한 `heatmap_preview`만 publication metadata에 포함한다.

이 preview는 공간 패턴 확인용이며 온도가 아니다. 센서의 °C calibration 계약이 없으므로 화면에는 raw 최소/최대와 `온도 보정 미적용`을 명시하고 °C 값을 만들지 않는다. TFLite runtime 또는 모델 로드 실패 시에도 live frame이 유효하면 preview는 유지된다.

## 실행

저장소 루트에서 백엔드 의존성을 설치하고 실행한다.

```bash
python -m pip install -r ./requirements-backend.txt
python ./backend/run_backend.py
```

같은 네트워크의 브라우저에서 `http://<raspberry-pi-ip>:8000/dashboard`를 연다. 센서 노드는 기본 TCP 9000 포트로 연결한다.

## 검증 범위

자동 테스트는 필수 DOM id 중복, same-origin asset/API 경로, WebSocket/polling 경계, read-only 동작, Thermal 단위 표기, legacy demo 값 배제, 반응형/reduced-motion CSS, preview shape와 모델 장애 시 보존을 검사한다.

현재 개발 환경에는 FastAPI/uvicorn과 실제 브라우저가 설치되어 있지 않으므로 HTTP 렌더링 및 Raspberry Pi 화면 크기의 육안 검증은 수행하지 않았다. 이는 Pi 배포 후 smoke test 항목으로 남긴다.
