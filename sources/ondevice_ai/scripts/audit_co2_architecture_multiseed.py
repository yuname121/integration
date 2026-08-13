#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate SafeNest CO₂ Phase C-B3 architecture/multi-seed evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

repo_root_dir = Path(__file__).resolve().parent.parent
if str(repo_root_dir) not in sys.path:
    sys.path.insert(0, str(repo_root_dir))

from datasets.co2.architecture_multiseed import run_architecture_multiseed
from datasets.co2.raw_reader import get_repo_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate SafeNest CO2 Phase C-B3 evidence")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument(
        "--repeat-determinism",
        action="store_true",
        help="fit the fixed 20-run grid twice and compare probabilities",
    )
    args = parser.parse_args()
    result = run_architecture_multiseed(
        args.repo_root or get_repo_root(), verify_repeat=args.repeat_determinism
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
