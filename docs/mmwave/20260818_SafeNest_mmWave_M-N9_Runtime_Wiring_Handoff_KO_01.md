# SafeNest mmWave M-N9 Runtime Wiring

## 변경 내용

활성 mmWave AI 경로의 과거 `respiration_phase_window[300]` 입력을 제거했다. TCP v1의 기존 필드는 유지하면서 선택적 MR60 `breath_phase`, `ts_monotonic_ms`, `phase_age_ms`, `human_detected_raw`, `session_id`를 정규화한다. 이 필드가 없는 기존 패킷은 계속 수신되지만 M-N9 입력으로 승격되지 않는다.

`MR60CanonicalWindowBuilder`는 동결된 `mmwave_m_n4_canonical.py`를 재사용한다. 즉 `ts_monotonic_ms - phase_age_ms`의 8 ms 초과 전진만 genuine event로 받아 R2를 만들고, boot/session 경계와 large gap을 끊는다. 최신 연속 30초를 8 Hz/240개로 linear resample한 뒤 MAD **divide-only** 정규화를 적용한다. 300개 텐서는 M-N9에 전달되지 않는다.

M-N9은 `MMWAVE_M_N9_FULL_INT8_V1` (`3b008af4be0facc4037c2afd3fe39292fb794208eb4370dbe6916b2d15aa38a4`)으로만 로드된다. 런타임은 interpreter metadata의 INT8 scale/zero point로 양자화·역양자화하고 0/NORMAL, 1/RAPID_OR_ABNORMAL, 2/APNEA-proxy를 반환한다.

## Presence 및 저장

기존 `human_detected_raw` boolean을 그대로 `presence`로 노출하며 숫자 임계값은 추가하지 않았다. false 또는 부재이면 inference 결과를 생리 상태로 노출하지 않고 각각 `NO_VALID_PERSON` 또는 `PRESENCE_STATE_UNAVAILABLE`으로 suppress한다. 특히 empty zero input의 APNEA-proxy는 backend/UI에 유효 호흡 결과로 남지 않는다.

mmWave JSONL은 phase, timestamp, age, presence, boot/session provenance를 매 수신 패킷별로 append한다. 최신 backend state와 별도로 재생 가능한 raw/near-raw evidence를 남긴다.

## Mac replay 결과와 제한

실제 MR60 snapshot은 workspace에서 찾지 못했다. 따라서 246개 8 Hz 합성 telemetry로 같은 pipeline을 재생했다. 30초 warm-up 뒤 canonical window를 만들고 실제 M-N9 INT8 interpreter를 두 번 호출했으며, no-person output suppression과 persistence를 확인했다. 이는 runtime mechanics 검증이며 생리적 모델 검증이 아니다.

- `MAC_REPLAY_RUNTIME_READY = YES`
- `PI_DEVICE_SMOKE = NOT_PERFORMED_ENVIRONMENT_UNAVAILABLE`
- `LIVE_MR60_STREAM = NOT_PERFORMED`
- `DEVICE_VALIDATED = NO`

다음 단계는 이 runtime 경계를 팀 저장소의 `RaspberryPi/Ondevice_AI`에 이식한 뒤 Pi 및 live MR60 smoke를 수행하는 것이다. M-N10 capture, firmware, 다른 sensor path는 변경하지 않았다.
