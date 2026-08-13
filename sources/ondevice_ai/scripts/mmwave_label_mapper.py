#!/usr/bin/env python3
"""SafeNest Phase A4 — Annotation Alignment & Deterministic Label Mapper.

Phase A4 semantically connects original dataset test conditions, voluntary
non-breathing annotations, and Movesense chest accelerometer reference respiration rates
to SafeNest target classes (NORMAL, RAPID_OR_ABNORMAL, APNEA).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
import math
import re
from typing import Any, Sequence

import numpy as np


PROFILE_ID = "MMWAVE_LABEL_MAPPING_PROFILE_001"


class LabelMappingError(ValueError):
    """Raised when annotation parsing or label mapping fails."""


@dataclass(frozen=True)
class LabelMappingProfile:
    """Deterministic configuration for label mapping and policy evaluation."""

    profile_id: str = PROFILE_ID
    target_classes: dict[str, int] = field(
        default_factory=lambda: {"NORMAL": 0, "RAPID_OR_ABNORMAL": 1, "APNEA": 2}
    )
    apnea_min_overlap_seconds: float = 6.0
    apnea_min_event_duration_seconds: float = 8.0
    rapid_min_rr_bpm: float = 25.0
    normal_min_rr_bpm: float = 10.0
    movesense_rr_search_band_hz: tuple[float, float] = (0.1, 0.7)
    post_exercise_auto_rapid: bool = False
    clinical_apnea_claimed: bool = False
    a3_window_contract_modified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "target_classes": self.target_classes,
            "apnea_policy": {
                "min_overlap_seconds": self.apnea_min_overlap_seconds,
                "min_event_duration_seconds": self.apnea_min_event_duration_seconds,
                "voluntary_breath_hold_as_apnea_proxy": True,
                "clinical_apnea_claimed": self.clinical_apnea_claimed,
            },
            "normal_policy": {
                "rest_condition_as_normal_proxy": True,
                "requires_zero_non_breathing_overlap": True,
                "movesense_acc_normal_rr_range_bpm": [self.normal_min_rr_bpm, self.rapid_min_rr_bpm],
            },
            "rapid_or_abnormal_policy": {
                "rapid_min_rr_bpm": self.rapid_min_rr_bpm,
                "bradypnea_max_rr_bpm": self.normal_min_rr_bpm,
                "post_exercise_auto_rapid": self.post_exercise_auto_rapid,
                "requires_independent_respiration_rate_reference": True,
                "reference_sensor": "MOVESENSE_CHEST_ACC",
                "movesense_rr_search_band_hz": list(self.movesense_rr_search_band_hz),
            },
            "a3_window_contract_modified": self.a3_window_contract_modified,
        }


def _parse_iso_string(s: str) -> dt.datetime:
    """Parse ISO-8601 string into a datetime object with microsecond precision."""
    cleaned = s.strip().rstrip("Z").replace(" ", "T")
    match = re.fullmatch(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?", cleaned)
    if not match:
        raise LabelMappingError(f"Invalid ISO timestamp string: {s!r}")
    base, frac = match.groups()
    frac = (frac or "").ljust(6, "0")[:6]
    return dt.datetime.fromisoformat(f"{base}.{frac}")


def parse_annotation_file(
    raw: bytes | str, radar_start_iso: str
) -> list[dict[str, Any]]:
    """Parse non_breathing_ts.csv lines (begin, end) relative to radar_start_iso."""
    text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    radar_t0 = _parse_iso_string(radar_start_iso)
    events: list[dict[str, Any]] = []

    begin_dt: dt.datetime | None = None
    end_dt: dt.datetime | None = None
    begin_raw: str = ""
    end_raw: str = ""

    for line in lines:
        parts = line.split(",", 1)
        if len(parts) != 2:
            continue
        key, val = parts[0].strip().lower(), parts[1].strip()
        if key == "begin":
            begin_raw = val
            begin_dt = _parse_iso_string(val)
        elif key == "end":
            end_raw = val
            end_dt = _parse_iso_string(val)

    if begin_dt is not None and end_dt is not None:
        t_start_rel = (begin_dt - radar_t0).total_seconds()
        t_end_rel = (end_dt - radar_t0).total_seconds()
        duration = (end_dt - begin_dt).total_seconds()

        events.append(
            {
                "event_id": "EVT_0001",
                "annotation_type_original": "VOLUNTARY_NON_BREATHING",
                "start_timestamp_iso": begin_raw,
                "end_timestamp_iso": end_raw,
                "start_seconds_relative": float(t_start_rel),
                "end_seconds_relative": float(t_end_rel),
                "duration_seconds": float(duration),
            }
        )

    return events


def extract_movesense_respiration_rate(
    acc_raw: bytes | str,
    radar_t0_iso: str,
    window_start_sec: float,
    window_end_sec: float,
    search_band_hz: tuple[float, float] = (0.1, 0.7),
) -> dict[str, Any] | None:
    """Extract respiration rate (in bpm and Hz) from Movesense chest ACC data for a 30s window."""
    text = acc_raw.decode("utf-8") if isinstance(acc_raw, bytes) else str(acc_raw)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 10:
        return None

    radar_t0 = _parse_iso_string(radar_t0_iso)
    timestamps: list[dt.datetime] = []
    mags: list[float] = []

    for l in lines[1:]:
        parts = l.split(",")
        if len(parts) < 4:
            continue
        try:
            t = _parse_iso_string(parts[0])
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            mag = math.sqrt(x * x + y * y + z * z)
            timestamps.append(t)
            mags.append(mag)
        except Exception:
            continue

    if not timestamps:
        return None

    t_sec = np.array([(t - radar_t0).total_seconds() for t in timestamps], dtype=np.float64)
    mags_arr = np.array(mags, dtype=np.float64)

    mask = (t_sec >= window_start_sec) & (t_sec < window_end_sec)
    if np.sum(mask) < 50:
        return None

    sub_t = t_sec[mask]
    sub_m = mags_arr[mask]

    dt_grid = 0.04  # 25 Hz
    grid_t = np.arange(sub_t[0], sub_t[-1], dt_grid)
    if len(grid_t) < 50:
        return None

    grid_m = np.interp(grid_t, sub_t, sub_m)

    # Detrend
    p = np.polyfit(grid_t - grid_t[0], grid_m, 1)
    grid_m_detrend = grid_m - np.polyval(p, grid_t - grid_t[0])

    # Hanning window
    h_win = np.hanning(len(grid_m_detrend))
    grid_m_win = grid_m_detrend * h_win

    # Spectral analysis over search band (f_low to f_high)
    f_low, f_high = search_band_hz
    freqs = np.fft.rfftfreq(len(grid_m_win), d=dt_grid)
    fft_mag = np.abs(np.fft.rfft(grid_m_win))

    valid_f_mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(valid_f_mask):
        return None

    peak_idx = np.argmax(fft_mag[valid_f_mask])
    peak_freq_hz = float(freqs[valid_f_mask][peak_idx])
    rr_bpm = peak_freq_hz * 60.0

    return {
        "peak_freq_hz": round(peak_freq_hz, 4),
        "rr_bpm": round(rr_bpm, 2),
        "sample_count": int(np.sum(mask)),
        "reference_sensor": "MOVESENSE_CHEST_ACC",
        "search_band_hz": list(search_band_hz),
    }


def compute_window_annotation_overlap(
    window_start_sec: float,
    window_end_exclusive_sec: float,
    events: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Compute exact 1D interval intersection between a window and annotation events."""
    total_overlap_sec = 0.0
    overlapping_events: list[dict[str, Any]] = []
    win_len_sec = window_end_exclusive_sec - window_start_sec

    for ev in events:
        e_start = ev["start_seconds_relative"]
        e_end = ev["end_seconds_relative"]

        o_start = max(window_start_sec, e_start)
        o_end = min(window_end_exclusive_sec, e_end)

        overlap_sec = max(0.0, o_end - o_start)
        if overlap_sec > 0:
            total_overlap_sec += overlap_sec
            overlapping_events.append(
                {
                    "event_id": ev["event_id"],
                    "event_start_seconds": e_start,
                    "event_end_seconds": e_end,
                    "overlap_start_seconds": o_start,
                    "overlap_end_seconds": o_end,
                    "overlap_seconds": float(overlap_sec),
                }
            )

    overlap_fraction = total_overlap_sec / win_len_sec if win_len_sec > 0 else 0.0

    return {
        "annotation_overlap_seconds": round(total_overlap_sec, 6),
        "annotation_overlap_fraction": round(overlap_fraction, 6),
        "overlapping_event_count": len(overlapping_events),
        "overlapping_events": overlapping_events,
    }


def map_window_label(
    window_record: dict[str, Any],
    events: Sequence[dict[str, Any]],
    source_condition: str,
    posture: str,
    profile: LabelMappingProfile = LabelMappingProfile(),
    movesense_rr_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deterministically assign SafeNest target label and mapping provenance for a window."""
    win_start_sec = float(window_record.get("window_index", 0) * 30.0)
    win_end_sec = win_start_sec + profile.target_classes.get("window_duration_seconds", 30.0)

    overlap_info = compute_window_annotation_overlap(win_start_sec, win_end_sec, events)
    overlap_sec = overlap_info["annotation_overlap_seconds"]

    safenest_label: str | None = None
    safenest_label_id: int | None = None
    mapping_type: str = "AMBIGUOUS"
    mapping_rule_id: str = "A4_RULE_UNMAPPED"
    assignment_status: str = "UNMAPPED"
    ambiguity_reasons: list[str] = []
    mapping_evidence: list[str] = []

    # Precedence Rule Evaluation:
    # 1. APNEA Proxy: Non-breathing overlap >= apnea_min_overlap_seconds (6.0s)
    if overlap_sec >= profile.apnea_min_overlap_seconds:
        safenest_label = "APNEA"
        safenest_label_id = profile.target_classes["APNEA"]
        mapping_type = "DERIVED"
        mapping_rule_id = "A4_RULE_APNEA_VOLUNTARY_PROXY"
        assignment_status = "ASSIGNED"
        mapping_evidence.append(
            f"Voluntary non-breathing annotation overlap {overlap_sec:.3f}s >= threshold {profile.apnea_min_overlap_seconds}s"
        )
    # 2. Transition Window: Non-zero overlap but < 6.0s
    elif overlap_sec > 0.0:
        safenest_label = None
        safenest_label_id = None
        mapping_type = "AMBIGUOUS"
        mapping_rule_id = "A4_RULE_TRANSITION_WINDOW"
        assignment_status = "AMBIGUOUS"
        ambiguity_reasons.append(
            f"Non-breathing overlap {overlap_sec:.3f}s is non-zero but below APNEA threshold {profile.apnea_min_overlap_seconds}s (transition state)"
        )
    # 3. Movesense ACC Respiration Rate Reference (for non-apnea windows)
    elif movesense_rr_info is not None:
        rr_bpm = movesense_rr_info["rr_bpm"]
        if rr_bpm >= profile.rapid_min_rr_bpm:
            safenest_label = "RAPID_OR_ABNORMAL"
            safenest_label_id = profile.target_classes["RAPID_OR_ABNORMAL"]
            mapping_type = "DERIVED"
            mapping_rule_id = "A4_RULE_RAPID_MOVESENSE_ACC_REF"
            assignment_status = "ASSIGNED"
            mapping_evidence.append(
                f"Movesense chest ACC reference respiration rate {rr_bpm:.1f} bpm >= threshold {profile.rapid_min_rr_bpm} bpm"
            )
        elif rr_bpm >= profile.normal_min_rr_bpm:
            safenest_label = "NORMAL"
            safenest_label_id = profile.target_classes["NORMAL"]
            mapping_type = "DERIVED"
            mapping_rule_id = "A4_RULE_NORMAL_MOVESENSE_ACC_REF"
            assignment_status = "ASSIGNED"
            mapping_evidence.append(
                f"Movesense chest ACC reference respiration rate {rr_bpm:.1f} bpm is in normal range ({profile.normal_min_rr_bpm} <= RR < {profile.rapid_min_rr_bpm} bpm)"
            )
        else:
            # Abnormally slow respiration rate (< 10.0 bpm bradypnea)
            safenest_label = "RAPID_OR_ABNORMAL"
            safenest_label_id = profile.target_classes["RAPID_OR_ABNORMAL"]
            mapping_type = "DERIVED"
            mapping_rule_id = "A4_RULE_ABNORMAL_BRADYPNEA_MOVESENSE_ACC_REF"
            assignment_status = "ASSIGNED"
            mapping_evidence.append(
                f"Movesense chest ACC reference respiration rate {rr_bpm:.1f} bpm is abnormally slow bradypnea (< {profile.normal_min_rr_bpm} bpm)"
            )
    # 4. Rest Condition Proxy Fallback (when Movesense ACC unavailable)
    elif source_condition == "Rest" and overlap_sec == 0.0:
        safenest_label = "NORMAL"
        safenest_label_id = profile.target_classes["NORMAL"]
        mapping_type = "DERIVED"
        mapping_rule_id = "A4_RULE_NORMAL_REST_PROXY"
        assignment_status = "ASSIGNED"
        mapping_evidence.append("Controlled Rest condition with 0.0s non-breathing annotation overlap (ACC reference fallback)")
    # 5. Post-exercise Unverified (when Movesense ACC unavailable)
    elif source_condition == "Post-exercise":
        safenest_label = None
        safenest_label_id = None
        mapping_type = "AMBIGUOUS"
        mapping_rule_id = "A4_RULE_POST_EXERCISE_UNVERIFIED"
        assignment_status = "AMBIGUOUS"
        ambiguity_reasons.append(
            "Post-exercise condition lacks independent validated respiration rate reference ground truth"
        )

    out = dict(window_record)
    out.update(
        {
            "posture": posture,
            "source_test_condition": source_condition,
            "original_annotation_type": "VOLUNTARY_NON_BREATHING" if events else "NONE",
            "annotation_events_overlapping": overlap_info["overlapping_events"],
            "annotation_overlap_seconds": overlap_info["annotation_overlap_seconds"],
            "annotation_overlap_fraction": overlap_info["annotation_overlap_fraction"],
            "movesense_reference_rr": movesense_rr_info,
            "safenest_label": safenest_label,
            "safenest_label_id": safenest_label_id,
            "mapping_type": mapping_type,
            "mapping_rule_id": mapping_rule_id,
            "assignment_status": assignment_status,
            "mapping_evidence": mapping_evidence,
            "ambiguity_reasons": ambiguity_reasons,
        }
    )
    return out


__all__ = [
    "PROFILE_ID",
    "LabelMappingError",
    "LabelMappingProfile",
    "compute_window_annotation_overlap",
    "extract_movesense_respiration_rate",
    "map_window_label",
    "parse_annotation_file",
]
