#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SafeNest 통합 센서 라즈베리파이 수신부 (TCP 기반)
=====================================================================
ESP32 센서 노드로부터 단일 TCP 연결(포트 9000)을 통해 데이터를 수신합니다.
열화상 센서(Type 2) 데이터를 파싱하여 화면에 표시합니다.

16-byte SNST Header Protocol:
  magic[4] = "SNST"
  version: u8
  type: u8 (1: JSON Telemetry, 2: Thermal U16 BE)
  flags: u16
  sequence: u32
  payload_length: u32
"""

import socket
import struct
import cv2
import numpy as np
import threading

PORT = 9000
FRAME_WIDTH = 80
FRAME_HEIGHT = 62
PIXEL_COUNT = FRAME_WIDTH * FRAME_HEIGHT
THERMAL_PAYLOAD_SIZE = 16 + (PIXEL_COUNT * 2) # 16 bytes metadata + 9920 bytes pixels

def receive_exact(sock, count):
    """지정된 바이트 수만큼 정확히 수신합니다."""
    buf = bytearray()
    while len(buf) < count:
        try:
            packet = sock.recv(count - len(buf))
            if not packet:
                return None
            buf.extend(packet)
        except (socket.timeout, BlockingIOError):
            continue
        except Exception as e:
            print(f"[Error] Socket receive error: {e}")
            return None
    return bytes(buf)

def handle_client(conn, addr):
    print(f"[TCP] Connected to {addr}")
    conn.settimeout(2.0)
    
    try:
        while True:
            # 1. 16바이트 SNST 헤더 수신
            header = receive_exact(conn, 16)
            if not header:
                print("[TCP] Connection closed by client (Header)")
                break
                
            # 헤더 파싱 (Big-Endian: >)
            magic = header[:4]
            if magic != b'SNST':
                print("[Warning] Invalid magic bytes, disconnecting.")
                break
                
            version, pkt_type, flags, seq, payload_len = struct.unpack('>BBHII', header[4:16])
            
            # 2. 페이로드 수신
            payload = receive_exact(conn, payload_len)
            if not payload:
                print("[TCP] Connection closed by client (Payload)")
                break
                
            # 3. 열화상 데이터 처리 (Type 2)
            if pkt_type == 2:
                if payload_len < THERMAL_PAYLOAD_SIZE:
                    print(f"[Warning] Invalid thermal payload size: {payload_len}")
                    continue
                    
                # 메타데이터 파싱
                # meta: width, height, frame_seq, uptime_ms, min_raw, max_raw
                width, height, frame_seq, uptime_ms, min_raw, max_raw = struct.unpack('>HHIIHH', payload[:16])
                
                # 픽셀 데이터 파싱 (Big-Endian uint16)
                pixel_bytes = payload[16:16 + (width * height * 2)]
                raw_pixels = np.frombuffer(pixel_bytes, dtype=np.dtype('>u2'))
                
                # 열화상 시각화 처리
                process_and_display(raw_pixels, width, height, frame_seq, min_raw, max_raw)
                
            elif pkt_type == 1:
                # 텔레메트리 데이터 (Type 1) - 열화상 전용 코드이므로 콘솔에만 간단히 표시
                # print(f"[Telemetry] Seq: {seq}")
                pass
                
            else:
                print(f"[Warning] Unknown packet type: {pkt_type}")
                
    except Exception as e:
        print(f"[TCP] Client handler exception: {e}")
    finally:
        conn.close()
        print(f"[TCP] Disconnected from {addr}")

def process_and_display(raw_pixels, width, height, frame_seq, min_raw, max_raw):
    """수신된 픽셀 데이터를 OpenCV를 이용해 정규화하고 표시합니다."""
    # 2D 배열로 변환
    try:
        thermal_matrix = raw_pixels.reshape((height, width))
    except ValueError:
        return
        
    # Min-Max 정규화 (0~255)
    _min = float(min_raw) if min_raw < max_raw else float(np.min(thermal_matrix))
    _max = float(max_raw) if max_raw > min_raw else float(np.max(thermal_matrix))
    
    if _max > _min:
        normalized = ((thermal_matrix - _min) / (_max - _min) * 255.0).astype(np.uint8)
    else:
        normalized = np.zeros((height, width), dtype=np.uint8)
        
    # 컬러맵 적용 (JET)
    color_img = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    # 이미지 8배 확대 (640x496)
    enlarged = cv2.resize(color_img, (width * 8, height * 8), interpolation=cv2.INTER_CUBIC)
    
    # 텍스트 오버레이
    cv2.putText(enlarged, f"Frame: {frame_seq}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(enlarged, f"Raw Min: {_min:.0f} Max: {_max:.0f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("SafeNest Thermal View (TCP)", enlarged)
    cv2.waitKey(1)

def main():
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_sock.bind(("0.0.0.0", PORT))
        server_sock.listen(5)
        print(f"[Main] TCP Server listening on port {PORT}...")
        print("[Main] Waiting for ESP32 sensor node connection...")
        
        while True:
            conn, addr = server_sock.accept()
            # 클라이언트 연결마다 새로운 스레드 생성 (멀티 클라이언트 지원)
            client_thread = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
            client_thread.start()
            
    except KeyboardInterrupt:
        print("\n[Main] Keyboard interrupt received.")
    finally:
        server_sock.close()
        cv2.destroyAllWindows()
        print("[Main] Server shut down.")

if __name__ == "__main__":
    main()
