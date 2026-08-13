#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
thermal_prep.py
SafeNest Thermal-44 (80x62 IR Array) 열화상 데이터셋 물리/기하학 & Zenodo SDT 전수 전처리

1. thermal_new_dataset/ (Zenodo SDT 48,000 프레임: train 32k, val 8k, test 8k) stream 파싱
   - Class 3 (Background) -> 0: NOT_HUMAN
   - Class 1 & 2 (Standing/Sitting/Normal) -> 1: HUMAN_NORMAL
   - Class 0 (Lying/Fall) -> 2: HUMAN_FALL
2. thermal/ 폴더 내 기존 MLX90640 / TeraRanger 데이터셋 파싱 및 물리 기하학(Centroid Y & Aspect Ratio) 라벨링
3. 80x62 Resizing & Per-Frame Min-Max Normalization (0.0 ~ 1.0 float32)
4. thermal/processed_thermal_80x62.npz 저장
"""

import os
import glob
import io
import zipfile
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class MultiFileStream(io.BufferedIOBase):
    """14GB 분할 압축(train.zip.001~004)을 디스크 압축 해제 없이 메모리 스트림으로 직접 읽는 래퍼 클래스"""
    def __init__(self, paths):
        self.paths = [p for p in paths if os.path.exists(p)]
        self.sizes = [os.path.getsize(p) for p in self.paths]
        self.total_size = sum(self.sizes)
        self.pos = 0
        self.fps = [open(p, 'rb') for p in self.paths]

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET: self.pos = offset
        elif whence == io.SEEK_CUR: self.pos += offset
        elif whence == io.SEEK_END: self.pos = self.total_size + offset
        self.pos = max(0, min(self.pos, self.total_size))
        return self.pos

    def read(self, size=-1):
        if size == -1 or size is None: size = self.total_size - self.pos
        if size <= 0 or self.pos >= self.total_size: return b''
        res = bytearray()
        rem = size
        while rem > 0 and self.pos < self.total_size:
            cur = self.pos
            f_idx = 0
            for i, sz in enumerate(self.sizes):
                if cur < sz:
                    f_idx = i
                    break
                cur -= sz
            fp = self.fps[f_idx]
            fp.seek(cur)
            chunk = fp.read(min(rem, self.sizes[f_idx] - cur))
            if not chunk: break
            res.extend(chunk)
            self.pos += len(chunk)
            rem -= len(chunk)
        return bytes(res)

    def close(self):
        for fp in self.fps: fp.close()

def extract_posture_label(img_arr, default_label=1):
    """
    80x62 체온 이미지 배열(0.0 ~ 1.0)로부터 인체 열원의 공간 위치 및 형상 분석
    """
    mask = img_arr > 0.35
    pixel_count = np.sum(mask)
    
    if pixel_count < 20:
        return 0
        
    y_indices, x_indices = np.where(mask)
    y_center = np.mean(y_indices)
    
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    
    height = max(1, y_max - y_min + 1)
    width = max(1, x_max - x_min + 1)
    aspect_ratio = width / float(height)
    
    if aspect_ratio >= 1.20 or y_center >= 34.0:
        return 2  # Human Fall
    else:
        return 1  # Human Normal

def parse_sdt_zip(zf, split_name):
    """Zenodo SDT Zip 아카이브에서 thermal 이미지(image_t_x.png) 및 labels.txt 파싱"""
    label_filename = f"{split_name}/labels.txt"
    if label_filename not in zf.namelist():
        # fallback search
        found = [f for f in zf.namelist() if f.endswith('labels.txt')]
        if not found:
            print(f"⚠️ {split_name}에서 labels.txt를 찾지 못했습니다.")
            return [], []
        label_filename = found[0]

    lines = [l.strip() for l in zf.read(label_filename).decode('utf-8', errors='ignore').splitlines() if l.strip()]
    
    X_list = []
    y_list = []
    
    print(f"  - [{split_name}] SDT 레이블 레코드: {len(lines)}개 항목 파싱 진행 중...")
    for idx, line in enumerate(lines):
        try:
            parts = line.split(',')
            sdt_class = int(parts[0])
            
            # SDT Class 매핑:
            # 3: Background -> 0 (NOT_HUMAN)
            # 1, 2: Standing/Sitting -> 1 (HUMAN_NORMAL)
            # 0: Lying/Fallen -> 2 (HUMAN_FALL)
            if sdt_class == 3:
                label = 0
            elif sdt_class in (1, 2):
                label = 1
            elif sdt_class == 0:
                label = 2
            else:
                label = 1

            img_name = f"{split_name}/image_t_{idx}.png"
            if img_name not in zf.namelist():
                # Try fallback image name pattern
                img_name = f"image_t_{idx}.png"
                if img_name not in zf.namelist():
                    continue

            img_bytes = zf.read(img_name)
            img = Image.open(io.BytesIO(img_bytes))
            img_resized = img.resize((80, 62), Image.Resampling.BILINEAR)
            arr = np.array(img_resized, dtype=np.float32)
            
            # Per-frame Min-Max Normalization (0.0 ~ 1.0)
            arr_min, arr_max = arr.min(), arr.max()
            if arr_max > arr_min:
                arr_norm = (arr - arr_min) / (arr_max - arr_min)
            else:
                arr_norm = np.zeros_like(arr)
                
            X_list.append(arr_norm)
            y_list.append(label)
        except Exception:
            pass

    return X_list, y_list

def process_thermal_datasets(base_dir):
    print("🚀 [Step 1] Thermal-44 (80x62 IR Array) 데이터셋 전수 전처리 시작...")
    
    X_data = []
    y_data = []
    
    # --- 1. Zenodo SDT 데이터셋 파싱 (datasets/raw_archives/thermal_split_zips) ---
    new_ds_dir = os.path.join(base_dir, "..", "datasets", "raw_archives", "thermal_split_zips")
    if not os.path.exists(new_ds_dir):
        new_ds_dir = os.path.join(base_dir, "thermal_new_dataset")
    if os.path.exists(new_ds_dir):
        # A. train.zip.001 ~ 004
        train_parts = [os.path.join(new_ds_dir, f"train.zip.00{i}") for i in range(1, 5)]
        if all(os.path.exists(p) for p in train_parts):
            print("📦 [Zenodo SDT Train Set] 14GB 분할 압축 스트림 파싱 중...")
            stream = MultiFileStream(train_parts)
            with zipfile.ZipFile(stream) as zf:
                X_tr, y_tr = parse_sdt_zip(zf, "train")
                X_data.extend(X_tr)
                y_data.extend(y_tr)
                print(f"    └─ Train 수집 완료: {len(X_tr)} 개 프레임")
            stream.close()

        # B. validation.zip
        val_zip = os.path.join(new_ds_dir, "validation.zip")
        if os.path.exists(val_zip):
            print("📦 [Zenodo SDT Validation Set] 파싱 중...")
            with zipfile.ZipFile(val_zip) as zf:
                X_val, y_val = parse_sdt_zip(zf, "validation")
                X_data.extend(X_val)
                y_data.extend(y_val)
                print(f"    └─ Validation 수집 완료: {len(X_val)} 개 프레임")

        # C. test.zip
        test_zip = os.path.join(new_ds_dir, "test.zip")
        if os.path.exists(test_zip):
            print("📦 [Zenodo SDT Test Set] 파싱 중...")
            with zipfile.ZipFile(test_zip) as zf:
                X_ts, y_ts = parse_sdt_zip(zf, "test")
                X_data.extend(X_ts)
                y_data.extend(y_ts)
                print(f"    └─ Test 수집 완료: {len(X_ts)} 개 프레임")

    # --- 2. 기존 thermal/ 폴더 내 소형 데이터셋 병합 ---
    thermal_img_dir = os.path.join(base_dir, "thermal", "thermal image")
    archive_dir = os.path.join(base_dir, "thermal", "archive")
    
    not_human_dir = os.path.join(thermal_img_dir, "not human")
    human_dir = os.path.join(thermal_img_dir, "human")
    
    if os.path.exists(not_human_dir):
        files = glob.glob(os.path.join(not_human_dir, "*.[jJ][pP][gG]")) + glob.glob(os.path.join(not_human_dir, "*.[pP][nN][gG]"))
        for f in files:
            try:
                img = Image.open(f).convert('L')
                img_resized = img.resize((80, 62), Image.Resampling.BILINEAR)
                arr = np.array(img_resized, dtype=np.float32) / 255.0
                X_data.append(arr)
                y_data.append(0)
            except Exception:
                pass
                
    if os.path.exists(human_dir):
        files = glob.glob(os.path.join(human_dir, "*.[jJ][pP][gG]")) + glob.glob(os.path.join(human_dir, "*.[pP][nN][gG]"))
        for f in files:
            try:
                img = Image.open(f).convert('L')
                img_resized = img.resize((80, 62), Image.Resampling.BILINEAR)
                arr = np.array(img_resized, dtype=np.float32) / 255.0
                label = extract_posture_label(arr, default_label=1)
                X_data.append(arr)
                y_data.append(label)
            except Exception:
                pass

    fall_dataset_dir = os.path.join(archive_dir, "Thermal_Dataset_Fall_Non_Fall")
    if not os.path.exists(fall_dataset_dir):
        subdirs = [d for d in glob.glob(os.path.join(archive_dir, "**"), recursive=True) if os.path.isdir(d) and "Fall" in d]
        if subdirs:
            fall_dataset_dir = subdirs[0]
            
    if os.path.exists(fall_dataset_dir):
        files = glob.glob(os.path.join(fall_dataset_dir, "*.[pP][nN][gG]")) + glob.glob(os.path.join(fall_dataset_dir, "*.[jJ][pP][gG]"))
        for f in files:
            try:
                img = Image.open(f).convert('L')
                img_resized = img.resize((80, 62), Image.Resampling.BILINEAR)
                arr = np.array(img_resized, dtype=np.float32) / 255.0
                label = extract_posture_label(arr)
                X_data.append(arr)
                y_data.append(label)
            except Exception:
                pass

    X_arr = np.array(X_data, dtype=np.float32)  # (N, 62, 80)
    y_arr = np.array(y_data, dtype=np.int32)
    
    unique, counts = np.unique(y_arr, return_counts=True)
    class_dist = dict(zip(unique, counts))
    
    print(f"\n✅ [80x62 전수 전처리 완료 요약]")
    print(f"   - 총 수집 프레임: {len(X_arr)} 개")
    print(f"   - 이미지 배열 Shape: {X_arr.shape} (62 rows x 80 cols 그리드)")
    print(f"   - 클래스 분포: {class_dist}")
    print(f"     (0: NOT_HUMAN, 1: HUMAN_NORMAL, 2: HUMAN_FALL)")
    
    output_dir = os.path.join(base_dir, "thermal")
    os.makedirs(output_dir, exist_ok=True)
    output_npz = os.path.join(output_dir, "processed_thermal_80x62.npz")
    np.savez_compressed(output_npz, X=X_arr, y=y_arr)
    print(f"💾 80x62 파싱 데이터 압축 저장 완료: {output_npz} (용량: {os.path.getsize(output_npz)/1024/1024:.2f} MB)")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    process_thermal_datasets(base_dir)
