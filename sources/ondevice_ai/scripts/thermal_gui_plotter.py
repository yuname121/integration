import os
import sys
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def run_thermal_gui_plotter(base_dir):
    print("🚀 [Step 3] Thermal-44 Camera 80x62 실시간 히트맵 GUI 시뮬레이터 구동 중...")
    
    # 1. TFLite 모델 로드 (tflite/ 폴더 또는 메인 디렉터리 탐색)
    tflite_candidates = [
        os.path.join(base_dir, "tflite", "thermal_fall_quant.tflite"),
        os.path.join(base_dir, "tflight", "thermal_fall_quant.tflite"),
        os.path.join(base_dir, "thermal_fall_quant.tflite")
    ]
    
    model_path = None
    for cand in tflite_candidates:
        if os.path.exists(cand):
            model_path = cand
            break
            
    if not model_path:
        print("⚠️ TFLite 모델을 찾지 못해 기본 경로 thermal_fall_quant.tflite 사용")
        model_path = os.path.join(base_dir, "thermal_fall_quant.tflite")
        
    print(f"📦 로드된 TFLite 모델 파일: {model_path}")
    
    # TFLite 인터프리터 초기화
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    scale, zero_point = input_details[0]['quantization']
    
    # 2. 테스트용 80x62 데이터셋 로드
    npz_path = os.path.join(base_dir, "thermal", "processed_thermal_80x62.npz")
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        X_test, y_test = data['X'], data['y']
    else:
        # 데이터가 없을 경우 시뮬레이션 가상 80x62 프레임 생성
        X_test = np.random.rand(100, 62, 80).astype(np.float32)
        y_test = np.random.choice([0, 1, 2], size=100)

    # 3. Matplotlib GUI 시각화 셋팅
    fig, ax = plt.subplots(figsize=(10, 7.5))
    fig.canvas.manager.set_window_title("SafeNest Edge AI - Thermal-44 80x62 Real-time Posture Monitor")
    
    initial_frame = X_test[0]
    im = ax.imshow(initial_frame, cmap='inferno', vmin=0.0, vmax=1.0, aspect='equal')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Temperature (0.0=Min, 1.0=Max)", fontsize=10)
    
    title_text = ax.set_title("Initializing Thermal-44 Camera Stream...", fontsize=12, fontweight='bold', pad=12)
    status_box = ax.text(0.5, -0.12, "", transform=ax.transAxes, ha='center', va='center',
                         fontsize=11, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))

    classes_str = ["0: Not Human / BG", "1: Human Normal Activity", "2: Human Fall / Fainted"]
    
    frame_counter = 0

    def update(frame_idx):
        nonlocal frame_counter
        idx = frame_counter % len(X_test)
        frame_counter += 1
        
        sample = X_test[idx] # Shape: (62, 80)
        im.set_array(sample)
        
        # TFLite 추론
        input_data = np.expand_dims(sample, axis=(0, -1)).astype(np.float32) # (1, 62, 80, 1)
        if scale > 0:
            input_int8 = np.int8(input_data / scale + zero_point)
        else:
            input_int8 = input_data.astype(np.int8)
            
        interpreter.set_tensor(input_details[0]['index'], input_int8)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Softmax 확률
        pred_class = np.argmax(output_data[0])
        prob = output_data[0][pred_class]
        
        # S4 융합 위험 점수 계산 (Fall 감지 시 S4 = 1.0)
        S4_score = 1.0 if pred_class == 2 else 0.0
        
        # UI 오버레이 업데이트
        if pred_class == 2: # FALL DETECTED
            ax.set_facecolor('#ffeeee')
            title_text.set_text(f"🚨 [FALL DETECTED] 바닥 쓰러짐/기절 감지! (Frame #{idx})")
            title_text.set_color('red')
            status_box.set_text(f"Status: FALL / FAINTED | S4 Score = {S4_score:.1f} | Conf: {prob*100:.1f}%")
            status_box.set_bbox(dict(boxstyle='round,pad=0.5', facecolor='#ffcccc', edgecolor='red', linewidth=2))
        elif pred_class == 1: # NORMAL
            ax.set_facecolor('white')
            title_text.set_text(f"🟢 [NORMAL] 인체 정상 활동 모니터링 중 (Frame #{idx})")
            title_text.set_color('#1e293b')
            status_box.set_text(f"Status: Normal Activity | S4 Score = {S4_score:.1f} | Conf: {prob*100:.1f}%")
            status_box.set_bbox(dict(boxstyle='round,pad=0.5', facecolor='#e0f2fe', edgecolor='#0284c7', linewidth=1))
        else: # NOT HUMAN
            ax.set_facecolor('white')
            title_text.set_text(f"⚪ [BG/NO HUMAN] 비생체 / 모니터링 대기 중 (Frame #{idx})")
            title_text.set_color('#64748b')
            status_box.set_text(f"Status: Not Human / BG | S4 Score = {S4_score:.1f} | Conf: {prob*100:.1f}%")
            status_box.set_bbox(dict(boxstyle='round,pad=0.5', facecolor='#f1f5f9', edgecolor='#94a3b8', linewidth=1))
            
        return im, title_text, status_box

    print("🖼️ Matplotlib GUI 실시간 시뮬레이션 창을 띄웁니다 (종료: 창 닫기)...")
    ani = animation.FuncAnimation(fig, update, interval=150, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_thermal_gui_plotter(base_dir)
