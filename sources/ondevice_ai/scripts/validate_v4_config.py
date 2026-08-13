#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Package-local entrypoint for SafeNest V5 model validation.

The historical filename is retained for automation compatibility. Validation is
always anchored to this script's package root, never to a sibling V4/archive.
"""

from __future__ import annotations
from pathlib import Path
import sys

package_root = Path(__file__).resolve().parent.parent
repo_root = package_root.parent

if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

from inference.validator import GroundTruthValidator


def main() -> int:
    validator = GroundTruthValidator(project_root=package_root)
    is_valid, _, errors = validator.validate_all(generate_inventory=True)
    if is_valid:
        print("✅ [V5] TFLite Model & Manifest Ground Truth Validation PASSED.")
        print(f"📄 Inventory report: {package_root.name}/docs/reports/model_inventory.json")
        return 0

    print("❌ [P0-2] TFLite Model & Manifest Ground Truth Validation FAILED:\n", file=sys.stderr)
    for error in errors:
        print(f"  {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
