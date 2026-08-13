# PHASE 3 — ESP32 to Raspberry Pi Communication

## 확정된 wire contract

| 항목 | 값 |
|---|---|
| Transport | TCP |
| ESP32 역할 | client |
| Raspberry Pi 역할 | server |
| 기본 port | 9000 |
| Header | `!4sBBHII`, 16 bytes |
| Magic/version | `SNST` / `1` |
| Endian | big-endian |
| Type 1 | UTF-8 telemetry JSON, 최대 4,096 bytes |
| Type 2 | Thermal U16 BE, 정확히 9,936 bytes |
| CRC | 없음 |
| ESP32 connect timeout | 1.5초 |
| ESP32 write timeout | 3초 |
| Pi complete-field deadline | 기본 5초 |

## 구현 파일

- `gateway/protocol.py`: header, telemetry, thermal, sequence의 strict decoder
- `gateway/receiver.py`: 연결 종료·malformed packet 후에도 accept를 계속하는 TCP server
- `gateway/run_receiver.py`: Raspberry Pi 실행 entry point
- `tests/test_gateway_protocol.py`: fragmentation, timeout, malformed, sequence, reconnect 테스트

## 오류 처리 정책

1. header magic/version/flags/type/length가 잘못되면 해당 연결을 닫는다.
2. 부분 header/payload는 받은 바이트를 보존하며 계속 읽는다.
3. 전체 receive deadline이 끝나면 부분 packet을 재사용하지 않고 연결을 닫는다.
4. telemetry와 thermal sequence는 ESP32 firmware와 동일하게 별도로 추적한다.
5. duplicate/backward sequence는 연결 단위 protocol error다.
6. sequence gap은 수신 누락 통계로 기록하지만 뒤 packet은 처리한다.
7. ESP32 재부팅 후 새 TCP 연결에서는 sequence 0을 허용한다.
8. JSON의 NaN/Inf, value/valid 불일치, header/JSON sequence 불일치를 거부한다.
9. Thermal dimensions, exact payload length, header/meta sequence, pixel min/max를 검증한다.
10. consumer callback 실패는 framing을 훼손하지 않으며 receiver는 계속 작동한다.

CRC가 없으므로 min/max 검증만으로 모든 bit corruption을 찾을 수는 없다. Protocol v1 sender와의 호환성을
깨지 않기 위해 PHASE 3에서 CRC를 임의로 추가하지 않았다. CRC 도입은 protocol v2 변경으로 관리해야 한다.

## Raspberry Pi 실행

저장소를 Raspberry Pi에 복사한 뒤 저장소 루트에서 실행한다.

```bash
cd ~/integration
source .venv/bin/activate
python3 -m gateway.run_receiver \
  --host 0.0.0.0 \
  --port 9000 \
  --packet-deadline 5
```

포트 확인:

```bash
ss -ltn | grep ':9000 '
```

## Windows 정적/loopback 검증

```powershell
cd C:\path\to\integration
python -m unittest discover `
  -s ./tests `
  -p "test_*.py" -v
```

## Arduino IDE

1. `sources/display-test2/esp32_sensor_node/esp32_sensor_node.ino`를 연다.
2. `secrets.example.h`를 `secrets.h`로 복사한다.
3. `WIFI_SSID`, `WIFI_PASSWORD`, `RPI_HOST`, `RPI_PORT=9000`을 설정한다.
4. 현재 firmware는 ESP-WROOM-32 핀맵이다. XIAO ESP32-C6에 바로 업로드하지 않는다.
5. 업로드 후 Serial Monitor를 115200 baud로 연다.

정상 로그:

```text
[network] connecting to <RPI_IP>:9000
[network] Raspberry Pi connected
```

## PHASE 4로 전달하는 출력

`on_packet(packet, peer)` callback은 검증이 끝난 다음 객체만 전달한다.

- `TelemetryPayload`: device, uptime, 호흡, 심박, CO₂, PIR, valid flags
- `ThermalFrame`: 80×62 big-endian pixel bytes, uptime, raw min/max

PHASE 4의 Sensor State Manager는 이 객체를 받아 센서별 `last_update`, `connected`, `stale`을 관리한다.
