#!/usr/bin/env python3
"""Generate SafeNest CO₂ C-B4 conversion/equivalence evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.co2.tflite_equivalence import run_c_b4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--no-repeat", action="store_true", help="Skip the second semantic conversion/inference run")
    args = parser.parse_args()
    result = run_c_b4(args.root, verify_repeat=not args.no_repeat)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
