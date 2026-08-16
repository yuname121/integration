"""CLI: python -m storage.capture_v1 validate PATH [PATH ...]"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .validator import validate_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate SafeNest Capture v1 documents")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="Validate session manifests or Capture events")
    validate.add_argument("paths", nargs="+", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = validate_paths(args.paths)
    if result.ok:
        print(f"ok: {len(args.paths)} document(s)")
        return 0
    print(result.format_errors(), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
