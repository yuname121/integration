#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sensors/thermal44/frame_parser.py
Thermal-44 (80x62 IR Array) Frame Parsing & Normalization Utility
"""

from __future__ import annotations
import numpy as np


class ThermalFrameParser:
    @staticmethod
    def parse_raw_buffer(raw_buffer: bytes | np.ndarray) -> np.ndarray:
        if isinstance(raw_buffer, bytes):
            array = np.frombuffer(raw_buffer, dtype=np.float32)
        else:
            array = np.asarray(raw_buffer, dtype=np.float32)

        if array.size != 4960:
            raise ValueError(f"Thermal frame must contain exactly 4960 pixels (80x62), got {array.size}")

        grid = array.reshape((62, 80))
        if not np.all(np.isfinite(grid)):
            raise ValueError("Thermal frame contains NaN or infinite values")

        return grid

    @staticmethod
    def normalize_to_int8(grid_80x62: np.ndarray, scale: float = 0.003814697265625, zero_point: int = -128) -> np.ndarray:
        grid = np.clip(grid_80x62, 0.0, 100.0)  # Temperature range clipping
        quantized = np.rint(grid / scale + zero_point)
        quantized = np.clip(quantized, -128, 127)
        return quantized.astype(np.int8)
