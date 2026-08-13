#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scripts/audit_co2_slope_ablation.py — Phase C-B1 generation entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.co2.slope_ablation import run_slope_ablation


def main() -> int:
    result = run_slope_ablation()
    print(f"C-B1 artifacts: {result['artifact_dir']}")
    print(f"Winner: {result['winner']}")
    print(f"Incremental: {result['incremental_status']}")
    print(f"ENDPOINT_H150 parity: {result['parity']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
