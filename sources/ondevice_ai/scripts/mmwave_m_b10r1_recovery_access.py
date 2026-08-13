#!/usr/bin/env python3
"""M-B10R1 limited-reuse recovery access controller.

Separate from ``mmwave_phase_b_access.PhaseBAccessGuard`` final accessor.
Does NOT modify phase_b_access. On the authorized success path (M-B10R1-B only),
loads LOCKED_TEST via the private ``_get_split_dataset`` helper with
``include_ambiguous=False`` — never ``get_locked_test_final_evaluation_dataset``.

M-B10R1-A never supplies a valid authorization for real access.

Payload-release audit semantics
-------------------------------
Once ``_load_eligible_locked_test()`` successfully RETURNS a payload object:
  - Immediately record ``recovery_payload_release_events += 1``
  - ``historical_total = original_final + recovery_payload_release_events``
  - Persist BEFORE ``_verify_payload()``
If verification then fails: payload_consumed remains true, release remains 1,
historical total remains 2, no retry / no rollback.
If loading throws BEFORE returning a payload: do NOT increment
``recovery_payload_release_events``; accessor invocation + payload_consumed=true
are still recorded (conservative); historical total stays 1.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR_REL = Path("datasets/mmwave/manifests/M-B10R1A_recovery_prefreeze")
DEFAULT_AUDIT_STATE_REL = OUT_DIR_REL / "recovery_access_runtime_state.json"

RECOVERY_AUTHORIZATION_TOKEN = "M_B10R1_LIMITED_REUSE_RECOVERY_AUTHORIZATION_V1"
# Governance control string, not a secret. M-B10R1-A never supplies it for real access.
ORIGINAL_FINAL_TOKEN = "AUTHORIZED_FINAL_LOCKED_TEST_EVALUATION_TOKEN_V1"  # must be REJECTED for recovery
RESULT_LIMITATION = "REUSED_LOCKED_TEST_AFTER_PREINFERENCE_STRUCTURAL_ABORT"
EXPECTED_ELIGIBLE = 75
EXPECTED_SUBJECTS = 16
EXPECTED_STRUCTURAL = 88
EXPECTED_AMBIGUOUS = 13
EXPECTED_INFERENCES = 225


class RecoveryAccessError(Exception):
    """Fail-closed recovery access refusal."""


@dataclass(frozen=True)
class RecoveryReadiness:
    """Authorization / readiness flags for recovery payload release."""

    recovery_execution_authorized: bool = False
    recovery_payload_release_authorized: bool = False
    independent_review_required: bool = True
    mechanism_implemented: bool = False
    runner_implemented: bool = False
    pre_access_validator_pass: bool = False
    M_B10R1B_started: bool = False


def _default_audit_state() -> dict[str, Any]:
    return {
        "schema_version": "M-B10R1_RECOVERY_ACCESS_RUNTIME_STATE_V1",
        # Historical facts — NEVER reset.
        "original_final_accessor_invocations": 1,
        "original_locked_test_consumed": True,
        "historical_total_payload_release_events": 1,
        "original_final_payload_release_events": 1,
        # Separate recovery counters.
        "recovery_accessor_invocations": 0,
        "recovery_payload_release_events": 0,
        "payload_consumed": False,
        "rerun_performed": False,
        "automatic_retry": False,
    }


class LimitedReuseRecoveryAccessController:
    """Auditable limited-reuse recovery accessor (at most one payload release)."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        audit_state_path: Path | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else ROOT_DIR
        self.audit_state_path = (
            Path(audit_state_path)
            if audit_state_path is not None
            else self.root / DEFAULT_AUDIT_STATE_REL
        )
        self._state = self._load_or_create_state()

    def _load_or_create_state(self) -> dict[str, Any]:
        if self.audit_state_path.is_file():
            state = json.loads(self.audit_state_path.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                raise RecoveryAccessError("RECOVERY_AUDIT_STATE_INVALID")
            # Preserve historical facts if present; refuse zeroed originals.
            if int(state.get("original_final_accessor_invocations", 0)) < 1:
                raise RecoveryAccessError("ORIGINAL_FINAL_ACCESSOR_HISTORY_RESET_FORBIDDEN")
            if state.get("original_locked_test_consumed") is not True:
                raise RecoveryAccessError("ORIGINAL_LOCKED_TEST_CONSUMED_MUST_REMAIN_TRUE")
            if int(state.get("historical_total_payload_release_events", 0)) < 1:
                raise RecoveryAccessError("HISTORICAL_PAYLOAD_RELEASE_HISTORY_RESET_FORBIDDEN")
            return state
        state = _default_audit_state()
        self.audit_state_path.parent.mkdir(parents=True, exist_ok=True)
        self._persist(state)
        return state

    def _persist(self, state: dict[str, Any] | None = None) -> None:
        payload = state if state is not None else self._state
        path = self.audit_state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self._state)

    def assert_preaccess_only(self) -> None:
        """Helper for validators: recovery must remain unused during M-B10R1-A."""
        if int(self._state.get("recovery_accessor_invocations", 0)) != 0:
            raise RecoveryAccessError("PREACCESS_RECOVERY_ACCESSOR_NOT_ZERO")
        if int(self._state.get("recovery_payload_release_events", 0)) != 0:
            raise RecoveryAccessError("PREACCESS_RECOVERY_PAYLOAD_NOT_ZERO")
        if self._state.get("payload_consumed") is True:
            raise RecoveryAccessError("PREACCESS_PAYLOAD_ALREADY_CONSUMED")
        if int(self._state.get("original_final_accessor_invocations", 0)) != 1:
            raise RecoveryAccessError("PREACCESS_ORIGINAL_ACCESSOR_HISTORY_CORRUPT")
        if self._state.get("original_locked_test_consumed") is not True:
            raise RecoveryAccessError("PREACCESS_ORIGINAL_CONSUMED_CORRUPT")

    def get_locked_test_recovery_evaluation_dataset(
        self,
        authorization_token: str | None,
        bound_contract_identity: dict[str, Any],
        readiness: RecoveryReadiness | dict[str, Any],
        *,
        phase_b_guard: Any | None = None,
    ) -> dict[str, Any]:
        """One-shot recovery payload release. Forbidden during M-B10R1-A."""
        if authorization_token == ORIGINAL_FINAL_TOKEN:
            raise RecoveryAccessError(
                "ORIGINAL_FINAL_TOKEN_REJECTED_FOR_RECOVERY:"
                "AUTHORIZED_FINAL_LOCKED_TEST_EVALUATION_TOKEN_V1_NOT_VALID_FOR_RECOVERY"
            )
        if not authorization_token or authorization_token != RECOVERY_AUTHORIZATION_TOKEN:
            raise RecoveryAccessError("RECOVERY_AUTHORIZATION_REFUSED")

        flags = readiness if isinstance(readiness, RecoveryReadiness) else RecoveryReadiness(
            recovery_execution_authorized=bool(readiness.get("recovery_execution_authorized")),
            recovery_payload_release_authorized=bool(readiness.get("recovery_payload_release_authorized")),
            independent_review_required=bool(readiness.get("independent_review_required", True)),
            mechanism_implemented=bool(readiness.get("mechanism_implemented")),
            runner_implemented=bool(readiness.get("runner_implemented")),
            pre_access_validator_pass=bool(readiness.get("pre_access_validator_pass")),
            M_B10R1B_started=bool(readiness.get("M-B10R1B_started", readiness.get("M_B10R1B_started", False))),
        )
        if flags.recovery_execution_authorized is not True:
            raise RecoveryAccessError("RECOVERY_EXECUTION_NOT_AUTHORIZED")
        if flags.recovery_payload_release_authorized is not True:
            raise RecoveryAccessError("RECOVERY_PAYLOAD_RELEASE_NOT_AUTHORIZED")

        if int(self._state.get("recovery_payload_release_events", 0)) >= 1:
            raise RecoveryAccessError("SECOND_RECOVERY_PAYLOAD_RELEASE_REFUSED")
        if self._state.get("payload_consumed") is True:
            raise RecoveryAccessError("RECOVERY_PAYLOAD_ALREADY_CONSUMED_NO_RETRY")

        # Hard-forbid include_ambiguous; eligibility is A6 assignment_status != AMBIGUOUS.
        include_ambiguous = False
        if bound_contract_identity.get("include_ambiguous") is True:
            raise RecoveryAccessError("INCLUDE_AMBIGUOUS_FORBIDDEN")
        if bound_contract_identity.get("expected_eligible_windows") not in (None, EXPECTED_ELIGIBLE):
            if int(bound_contract_identity.get("expected_eligible_windows", EXPECTED_ELIGIBLE)) != EXPECTED_ELIGIBLE:
                raise RecoveryAccessError("BOUND_ELIGIBLE_COUNT_MISMATCH")

        self._verify_bound_contract(bound_contract_identity)

        # Mark consumed before load so post-release failures still refuse retry.
        self._state["recovery_accessor_invocations"] = int(self._state.get("recovery_accessor_invocations", 0)) + 1
        self._state["payload_consumed"] = True
        self._state["rerun_performed"] = False
        self._state["automatic_retry"] = False
        self._persist()

        try:
            payload = self._load_eligible_locked_test(
                phase_b_guard=phase_b_guard, include_ambiguous=include_ambiguous
            )
        except Exception:
            # Load threw BEFORE returning a payload object:
            # do NOT increment recovery_payload_release_events; historical total stays 1.
            # accessor_invocations already incremented; payload_consumed remains true.
            self._persist()
            raise

        # Payload object successfully returned — record release BEFORE verify.
        # Order is intentional: verification failure must not roll back the release.
        self._state["recovery_payload_release_events"] = (
            int(self._state.get("recovery_payload_release_events", 0)) + 1
        )
        self._state["historical_total_payload_release_events"] = (
            int(self._state.get("original_final_payload_release_events", 1))
            + int(self._state["recovery_payload_release_events"])
        )
        self._persist()

        try:
            self._verify_payload(payload)
        except Exception:
            # Keep release=1, historical_total=2, consumed=true; no retry / no rollback.
            self._persist()
            raise

        return payload

    def _verify_bound_contract(self, bound: dict[str, Any]) -> None:
        required_keys = (
            "policy_decision_sha256",
            "reuse_exception_gate_results_sha256",
            "proposed_recovery_evaluation_contract_sha256",
            "future_recovery_access_requirements_sha256",
            "m_b10r0_summary_sha256",
            "m_b10a_metric_contract_sha256",
            "selected_model_sha256",
            "baseline_v01_sha256",
            "baseline_v02_sha256",
            "executor_sha256",
            "metadata_v01_sha256",
            "metadata_v02_sha256",
            "result_limitation",
            "expected_eligible_windows",
            "expected_subjects",
        )
        for key in required_keys:
            if key not in bound:
                raise RecoveryAccessError(f"BOUND_CONTRACT_MISSING:{key}")
        if bound.get("result_limitation") != RESULT_LIMITATION:
            raise RecoveryAccessError("BOUND_RESULT_LIMITATION_MISMATCH")
        if int(bound["expected_eligible_windows"]) != EXPECTED_ELIGIBLE:
            raise RecoveryAccessError("BOUND_ELIGIBLE_MISMATCH")
        if int(bound["expected_subjects"]) != EXPECTED_SUBJECTS:
            raise RecoveryAccessError("BOUND_SUBJECTS_MISMATCH")

        # Live recompute against frozen files under root.
        from hashlib import sha256

        def _sha(rel: str) -> str:
            path = self.root / rel
            if not path.is_file():
                raise RecoveryAccessError(f"BOUND_FILE_MISSING:{rel}")
            return sha256(path.read_bytes()).hexdigest()

        checks = {
            "policy_decision_sha256": "datasets/mmwave/manifests/M-B10R0_holdout_policy_review/policy_decision.json",
            "reuse_exception_gate_results_sha256": "datasets/mmwave/manifests/M-B10R0_holdout_policy_review/reuse_exception_gate_results.json",
            "proposed_recovery_evaluation_contract_sha256": "datasets/mmwave/manifests/M-B10R0_holdout_policy_review/proposed_recovery_evaluation_contract.json",
            "future_recovery_access_requirements_sha256": "datasets/mmwave/manifests/M-B10R0_holdout_policy_review/future_recovery_access_requirements.json",
            "m_b10r0_summary_sha256": "datasets/mmwave/manifests/M-B10R0_holdout_policy_review/m_b10r0_summary.json",
            "m_b10a_metric_contract_sha256": "datasets/mmwave/manifests/M-B10A_candidate_selection_setup/locked_test_evaluation_contract.json",
            "selected_model_sha256": "models/mmwave/experiments/M-B6_stage_equivalence/M-B3_CONV1D_GAP_BASELINE_seed42_M-B5_CAL_CLASS_BALANCED_120_int8.tflite",
            "baseline_v01_sha256": "models/mmwave/mmwave_resp_int8_v0.1.0.tflite",
            "baseline_v02_sha256": "models/mmwave/mmwave_resp_int8_v0.2.0_candidate.tflite",
            "executor_sha256": "scripts/mmwave_m_b10b_baseline_preprocessing.py",
            "metadata_v01_sha256": "models/mmwave/sensor_stats_metadata_v0.1.0.json",
            "metadata_v02_sha256": "models/mmwave/mmwave_resp_int8_v0.2.0_candidate_metadata.json",
        }
        for key, rel in checks.items():
            live = _sha(rel)
            if live != bound[key]:
                raise RecoveryAccessError(f"BOUND_CONTRACT_SHA_MISMATCH:{key}")

    def _load_eligible_locked_test(self, *, phase_b_guard: Any | None, include_ambiguous: bool) -> dict[str, Any]:
        if include_ambiguous:
            raise RecoveryAccessError("INCLUDE_AMBIGUOUS_HARDCODED_FALSE_ONLY")
        # Import locally to keep module import side-effect free of guard construction.
        from scripts.mmwave_phase_b_access import PhaseBAccessGuard

        guard = phase_b_guard if phase_b_guard is not None else PhaseBAccessGuard(root_dir=self.root)
        # ONLY private split loader — NEVER get_locked_test_final_evaluation_dataset.
        return guard._get_split_dataset("LOCKED_TEST", include_ambiguous=False)  # noqa: SLF001

    def _verify_payload(self, payload: dict[str, Any]) -> None:
        windows = payload.get("windows") or []
        if int(payload.get("total_count", len(windows))) != EXPECTED_ELIGIBLE:
            raise RecoveryAccessError(f"RECOVERY_PAYLOAD_COUNT_MISMATCH:{payload.get('total_count')}")
        if len(windows) != EXPECTED_ELIGIBLE:
            raise RecoveryAccessError(f"RECOVERY_PAYLOAD_WINDOW_LEN_MISMATCH:{len(windows)}")
        subjects = {str(w.get("subject_id")) for w in windows}
        if len(subjects) != EXPECTED_SUBJECTS:
            raise RecoveryAccessError(f"RECOVERY_PAYLOAD_SUBJECT_MISMATCH:{len(subjects)}")
        for window in windows:
            if window.get("assignment_status") == "AMBIGUOUS":
                raise RecoveryAccessError("RECOVERY_PAYLOAD_CONTAINS_AMBIGUOUS")
            if window.get("split") != "LOCKED_TEST":
                raise RecoveryAccessError("RECOVERY_PAYLOAD_SPLIT_MISMATCH")
