# SafeNest 로컬 음성 파일

Raspberry Pi Chromium kiosk에서 다음 파일을 이 디렉터리에 배치하면 대시보드가
브라우저의 로컬 오디오 재생으로 사용합니다.

- `system_start.mp3`
- `warning.mp3`
- `danger.mp3`
- `report_119.mp3`
- `report_119_complete.mp3`
- `sms_sent.mp3`
- `sms_failed.mp3`
- `sensor_offline.mp3`

파일이 없거나 브라우저 autoplay 정책으로 재생되지 않아도 화면·API·부저·로그는
계속 동작합니다. 실제 한국어 음성 파일은 팀이 직접 녹음한 안전한 자산을 사용하고,
비밀값이나 개인정보를 오디오 파일명/메타데이터에 넣지 않습니다.
