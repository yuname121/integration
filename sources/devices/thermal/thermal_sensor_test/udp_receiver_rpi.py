import socket
import numpy as np
import cv2
import os

# ==========================================
# 1. 보정 로직 (ThermalCalibrator)
# ==========================================
class ThermalCalibrator:
    def __init__(self):
        self.is_calibrated = False
        self.is_calibrating = False
        self.calibration_frames = 32
        self.current_frame_count = 0
        self.frame_accumulator = None
        self.offset_map = None
        self.offset_mean = 0.0
        self.die_temp_baseline = 0
        self.drift_coeff = -1.02  # 실측 선형 회귀 분석 결과 반영
        self.save_path = "thermal_calibration.npz"
        self.load()

    def start_calibration(self):
        self.is_calibrating = True
        self.current_frame_count = 0
        self.frame_accumulator = np.zeros((62, 80), dtype=np.float64)
        print("\n[Calibrator] Started collecting frames for FPN calibration...")

    def feed_calibration_frame(self, raw_matrix, die_temp):
        if self.is_calibrating:
            self.frame_accumulator += raw_matrix
            self.current_frame_count += 1
            if self.current_frame_count >= self.calibration_frames:
                self._complete_calibration(die_temp)

    def _complete_calibration(self, die_temp):
        self.offset_map = self.frame_accumulator / self.calibration_frames
        self.offset_mean = np.mean(self.offset_map)
        self.die_temp_baseline = die_temp
        self.is_calibrated = True
        self.is_calibrating = False
        print(f"[Calibrator] Calibration complete. Baseline Temp: {die_temp}")
        print("[Calibrator] Press 's' to save calibration data.")

    def correct(self, raw_matrix, die_temp):
        if not self.is_calibrated:
            return raw_matrix
        
        # Drift correction
        dt = float(die_temp) - float(self.die_temp_baseline)
        drift_correction = self.drift_coeff * dt
        
        # FPN correction + Drift correction
        corrected = raw_matrix - self.offset_map + self.offset_mean - drift_correction
        return corrected

    def save(self):
        if self.is_calibrated:
            np.savez(self.save_path, offset_map=self.offset_map, offset_mean=self.offset_mean, die_temp_baseline=self.die_temp_baseline)
            print(f"\n[Calibrator] Calibration data saved to {self.save_path}")
        else:
            print("\n[Calibrator] No calibration data to save.")

    def load(self):
        if os.path.exists(self.save_path):
            data = np.load(self.save_path)
            self.offset_map = data['offset_map']
            self.offset_mean = data['offset_mean']
            self.die_temp_baseline = data['die_temp_baseline']
            self.is_calibrated = True
            print(f"[Calibrator] Loaded calibration data from {self.save_path}")

    def reset(self):
        self.is_calibrated = False
        self.offset_map = None
        self.offset_mean = 0.0
        self.die_temp_baseline = 0
        if os.path.exists(self.save_path):
            os.remove(self.save_path)
        print("\n[Calibrator] Calibration reset.")

    def get_status_text(self, current_die_temp):
        if self.is_calibrating:
            return f"Calibrating... {self.current_frame_count}/{self.calibration_frames}"
        elif self.is_calibrated:
            dt = float(current_die_temp) - float(self.die_temp_baseline)
            return f"Cal: ON | dT: {dt} (a={self.drift_coeff})"
        else:
            return "Cal: OFF (Press 'c' to start)"

# ==========================================
# 2. UDP 네트워크 설정
# ==========================================
UDP_IP = "0.0.0.0"
UDP_PORT = 5005
BUFFER_SIZE = 65535

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(0.1)

print(f"[*] UDP Server listening on port {UDP_PORT}...")
print("[*] Keys: 'c' to calibrate, 's' to save, 'r' to reset, 'q' to quit\n")

packet_buffer = bytearray()
calibrator = ThermalCalibrator()

try:
    while True:
        try:
            data, addr = sock.recvfrom(BUFFER_SIZE)
            packet_buffer.extend(data)
            
            if len(packet_buffer) >= 10080:
                frame_data = packet_buffer[:10080]
                packet_buffer = packet_buffer[10080:]
                
                raw_array = np.frombuffer(frame_data, dtype=np.uint16)
                header = raw_array[:80]
                pixel_data = raw_array[80:]
                
                frame_counter = header[0]
                die_temp = header[2]
                max_pixel = header[5]
                min_pixel = header[6]
                
                # Float64로 캐스팅하여 보정 연산 정밀도 확보
                thermal_matrix = pixel_data.reshape((62, 80)).astype(np.float64)
                
                # 캘리브레이션 중이면 데이터 수집
                if calibrator.is_calibrating:
                    calibrator.feed_calibration_frame(thermal_matrix, die_temp)
                
                # 보정 적용 (Cal off 상태면 원본 그대로 반환)
                corrected_matrix = calibrator.correct(thermal_matrix, die_temp)
                
                # 정규화
                min_val = np.min(corrected_matrix)
                max_val = np.max(corrected_matrix)
                
                if max_val > min_val:
                    normalized = ((corrected_matrix - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
                else:
                    normalized = np.zeros((62, 80), dtype=np.uint8)
                    
                # 컬러맵
                color_img = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
                enlarged_img = cv2.resize(color_img, (80 * 8, 62 * 8), interpolation=cv2.INTER_CUBIC)
                
                # OSD 텍스트
                cv2.putText(enlarged_img, f"Frame: {frame_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(enlarged_img, f"Die Temp: {die_temp}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(enlarged_img, f"Min: {int(min_val)} / Max: {int(max_val)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # 보정 상태 텍스트 (노란색)
                cv2.putText(enlarged_img, calibrator.get_status_text(die_temp), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                cv2.imshow("Thermal-90 Camera with Calibration", enlarged_img)
                
                # 키보드 이벤트
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    calibrator.start_calibration()
                elif key == ord('s'):
                    calibrator.save()
                elif key == ord('r'):
                    calibrator.reset()
                    
        except socket.timeout:
            if len(packet_buffer) > 0:
                packet_buffer.clear()

except KeyboardInterrupt:
    pass
finally:
    sock.close()
    cv2.destroyAllWindows()
    print("[*] System safely shut down.")
