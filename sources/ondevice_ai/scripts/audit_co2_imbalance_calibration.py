#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate Phase C-B2 controlled imbalance/calibration evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

repo_root_dir = Path(__file__).resolve().parent.parent
if str(repo_root_dir) not in sys.path:
    sys.path.insert(0, str(repo_root_dir))

from datasets.co2.imbalance_calibration import run_imbalance_calibration
from datasets.co2.raw_reader import get_repo_root


def main() -> int:
    result = run_imbalance_calibration(get_repo_root())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
