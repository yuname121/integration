#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
datasets/build_processed_npz.py
SafeNest deterministic synthetic smoke-fixture generator.

Usage:
  python3 datasets/build_processed_npz.py --dataset co2
  python3 datasets/build_processed_npz.py --dataset mmwave

This script does not parse UCI or Zenodo source files. Source paths are rejected
to prevent synthetic arrays from being mislabeled as real processed data.
"""

from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path
import numpy as np


def build_co2_npz(output_dir: Path, source_root: Path | None = None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "co2_occupancy_v1.npz"

    if source_root is not None:
        raise ValueError(
            "Real CO2 source conversion is not implemented in this smoke-fixture generator; "
            "refusing to synthesize data for a supplied source path."
        )

    print("📊 [CO2] Generating synthetic smoke NPZ fixture...")

    # Synthetic reproducible generation matching the legacy smoke split.
    np.random.seed(42)
    y_tr = np.random.choice([0, 1], size=8138, p=[0.75, 0.25]).astype(np.int64)
    X_tr = np.zeros((8138, 3), dtype=np.float32)
    X_tr[:, 0] = np.where(y_tr == 1, np.random.normal(25.0, 10.0, 8138), np.random.normal(1.5, 3.0, 8138))
    X_tr[:, 1] = np.where(y_tr == 1, np.random.normal(55.0, 8.0, 8138), np.random.normal(40.0, 5.0, 8138))
    X_tr[:, 2] = np.where(y_tr == 1, np.random.normal(1400.0, 300.0, 8138), np.random.normal(500.0, 80.0, 8138))

    y_va = np.random.choice([0, 1], size=2660, p=[0.75, 0.25]).astype(np.int64)
    X_va = np.zeros((2660, 3), dtype=np.float32)
    X_va[:, 0] = np.where(y_va == 1, np.random.normal(25.0, 10.0, 2660), np.random.normal(1.5, 3.0, 2660))
    X_va[:, 1] = np.where(y_va == 1, np.random.normal(55.0, 8.0, 2660), np.random.normal(40.0, 5.0, 2660))
    X_va[:, 2] = np.where(y_va == 1, np.random.normal(1400.0, 300.0, 2660), np.random.normal(500.0, 80.0, 2660))

    y_te = np.random.choice([0, 1], size=9747, p=[0.75, 0.25]).astype(np.int64)
    X_te = np.zeros((9747, 3), dtype=np.float32)
    X_te[:, 0] = np.where(y_te == 1, np.random.normal(25.0, 10.0, 9747), np.random.normal(1.5, 3.0, 9747))
    X_te[:, 1] = np.where(y_te == 1, np.random.normal(55.0, 8.0, 9747), np.random.normal(40.0, 5.0, 9747))
    X_te[:, 2] = np.where(y_te == 1, np.random.normal(1400.0, 300.0, 9747), np.random.normal(500.0, 80.0, 9747))

    # Calculate normalization stats using ONLY train split
    mean = np.mean(X_tr, axis=0)
    std = np.std(X_tr, axis=0)
    std = np.where(std == 0, 1.0, std)

    np.savez_compressed(
        npz_path,
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_va,
        y_val=y_va,
        X_test=X_te,
        y_test=y_te,
        mean=mean,
        std=std,
        feature_names=["CO2_slope", "Humidity", "CO2"]
    )
    print(f"  ✅ Saved CO2 NPZ: {npz_path} (Train: {len(X_tr)}, Val: {len(X_va)}, Test: {len(X_te)})")
    return npz_path


def build_mmwave_npz(output_dir: Path, source_root: Path | None = None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "mmwave_respiration_v1.npz"

    if source_root is not None:
        raise ValueError(
            "Real Zenodo rFFT conversion belongs to Phase A6; refusing to generate "
            "synthetic mmWave data for a supplied source path."
        )

    print("📊 [mmWave] Generating synthetic smoke NPZ fixture...")
    np.random.seed(42)

    # Total 3433 300x1 windows across 110 subjects (Train: 2491, Val: 474, Test: 468)
    # Labels: 0: NORMAL (1401), 1: RAPID_OR_ABNORMAL (1717), 2: APNEA (315)
    t = np.linspace(0, 30, 300, dtype=np.float32)

    def generate_windows(count: int):
        X = np.zeros((count, 300, 1), dtype=np.float32)
        # Class probabilities: NORMAL ~ 0.408, ABNORMAL ~ 0.500, APNEA ~ 0.092
        y = np.random.choice([0, 1, 2], size=count, p=[0.408, 0.500, 0.092]).astype(np.int64)
        for i in range(count):
            lbl = y[i]
            if lbl == 2:  # APNEA
                sig = np.full(300, 1.25, dtype=np.float32) + np.random.normal(0, 0.01, 300).astype(np.float32)
            elif lbl == 1:  # RAPID/ABNORMAL
                sig = 2.5 * np.sin(2 * np.pi * 0.75 * t).astype(np.float32) + np.random.normal(0, 0.05, 300).astype(np.float32)
            else:  # NORMAL
                sig = 2.5 * np.sin(2 * np.pi * 0.25 * t).astype(np.float32) + np.random.normal(0, 0.05, 300).astype(np.float32)
            X[i, :, 0] = sig
        return X, y

    X_tr, y_tr = generate_windows(2491)
    X_va, y_va = generate_windows(474)
    X_te, y_te = generate_windows(468)

    # Normalization mean and std calculated solely on train split
    mean = float(np.mean(X_tr))
    std = float(np.std(X_tr))
    if std <= 0:
        std = 1.0

    np.savez_compressed(
        npz_path,
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_va,
        y_val=y_va,
        X_test=X_te,
        y_test=y_te,
        mean=mean,
        std=std,
        class_map={0: "NORMAL", 1: "RAPID_OR_ABNORMAL", 2: "APNEA"},
        subject_split={"train_subjects": 80, "val_subjects": 15, "test_subjects": 15}
    )
    print(f"  ✅ Saved mmWave NPZ: {npz_path} (Train: {len(X_tr)}, Val: {len(X_va)}, Test: {len(X_te)})")
    return npz_path


def main():
    parser = argparse.ArgumentParser(description="SafeNest synthetic smoke NPZ fixture builder")
    parser.add_argument("--dataset", choices=["co2", "mmwave", "all"], default="all", help="Target dataset")
    parser.add_argument("--source-root", type=str, default=None, help="Rejected: real source conversion is not implemented here")
    parser.add_argument("--co2-root", type=str, default=None, help="Rejected: use a dedicated real-source converter")
    parser.add_argument("--mmwave-root", type=str, default=None, help="Rejected: Zenodo conversion is performed in Phase A6")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    co2_out_dir = project_root / "datasets/co2/processed"
    mmwave_out_dir = project_root / "datasets/mmwave/processed"

    if args.dataset in ["co2", "all"]:
        root = Path(args.co2_root or args.source_root) if (args.co2_root or args.source_root) else None
        build_co2_npz(co2_out_dir, source_root=root)

    if args.dataset in ["mmwave", "all"]:
        root = Path(args.mmwave_root or args.source_root) if (args.mmwave_root or args.source_root) else None
        build_mmwave_npz(mmwave_out_dir, source_root=root)


if __name__ == "__main__":
    main()
