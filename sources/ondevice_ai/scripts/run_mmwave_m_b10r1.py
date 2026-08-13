#!/usr/bin/env python3
"""CLI entry for M-B10R1 recovery harness.

Default / --pre-access / --write-prefreeze-evidence → never access recovery.
--execute-authorized-limited-reuse-recovery → requires token AND readiness;
M-B10R1-A MUST NOT run that flag.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.mmwave_m_b10r1_recovery_eval import (  # noqa: E402
    MB10R1EvalError,
    execute_authorized_recovery,
    readiness_summary,
    run_validation_smoke,
)
from scripts.mmwave_m_b10r1_recovery_access import RecoveryAccessError  # noqa: E402
from scripts.mmwave_m_b10r1a_prefreeze import generate_m_b10r1a_prefreeze  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M-B10R1 limited-reuse recovery harness (pre-access by default)",
    )
    parser.add_argument(
        "--pre-access",
        action="store_true",
        help="Print readiness summary only (default behavior).",
    )
    parser.add_argument(
        "--write-prefreeze-evidence",
        action="store_true",
        help="Generate M-B10R1-A pre-freeze manifests and report (no recovery access).",
    )
    parser.add_argument(
        "--validation-smoke",
        action="store_true",
        help="Optional VALIDATION-only smoke (never LOCKED_TEST / recovery).",
    )
    parser.add_argument(
        "--execute-authorized-limited-reuse-recovery",
        action="store_true",
        help="IRREVERSIBLE future recovery path. Requires --authorization-token. Do not use in M-B10R1-A.",
    )
    parser.add_argument(
        "--authorization-token",
        default=None,
        help="Recovery authorization token (required only for irreversible execute mode).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Typo/default must not release payload: execute only when explicit flag set.
    if args.execute_authorized_limited_reuse_recovery:
        if not args.authorization_token:
            print(
                json.dumps(
                    {
                        "status": "REFUSED",
                        "reason": "AUTHORIZATION_TOKEN_REQUIRED",
                        "recovery_accessor_invoked": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
        try:
            result = execute_authorized_recovery(ROOT_DIR, args.authorization_token)
        except (MB10R1EvalError, RecoveryAccessError) as exc:
            print(
                json.dumps(
                    {
                        "status": "REFUSED",
                        "reason": str(exc),
                        "recovery_payload_released": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
        # Never print full ledger by default in CLI success path summary.
        # Full 225-row ledger is already persisted under the B result directory.
        slim = {k: v for k, v in result.items() if k != "ledger"}
        print(json.dumps(slim, indent=2, sort_keys=True))
        return 0

    if args.write_prefreeze_evidence:
        result = generate_m_b10r1a_prefreeze(ROOT_DIR)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.validation_smoke:
        result = run_validation_smoke(ROOT_DIR, attempt_tflite=False)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    # Default and --pre-access: readiness only.
    summary = readiness_summary(ROOT_DIR)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
