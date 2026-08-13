#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Canonical-root entrypoint for SafeNest model validation.

Validation is anchored to the active repository root and never resolves a
versioned sibling or archived snapshot.
"""

from __future__ import annotations
from pathlib import Path
import sys

package_root = Path(__file__).resolve().parent.parent
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

from inference.validator import GroundTruthValidator


def main() -> int:
    validator = GroundTruthValidator(project_root=package_root)
    is_valid, _, errors = validator.validate_all(generate_inventory=True)
    if is_valid:
        print("✅ [ACTIVE] TFLite Model & Manifest Ground Truth Validation PASSED.")
        print("📄 Inventory report: docs/reports/model_inventory.json")
        return 0

    print("❌ [ACTIVE] TFLite Model & Manifest Ground Truth Validation FAILED:\n", file=sys.stderr)
    for error in errors:
        print(f"  {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
