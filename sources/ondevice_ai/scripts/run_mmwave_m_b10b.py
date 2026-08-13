#!/usr/bin/env python3
"""Command-line entry point for the SafeNest M-B10B final evaluation."""

from __future__ import annotations

import sys

from mmwave_m_b10b_final_eval import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
