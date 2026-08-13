#!/usr/bin/env python3
"""M-B10R1 metric engine (copied from M-B10B semantics; unit-testable).

Identical support-zero and subject aggregation semantics to
``mmwave_m_b10b_final_eval.metric_bundle`` / ``subject_metrics``.
Independent module to avoid circular imports and accidental final-access paths.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

CLASS_MAP = {"0": "NORMAL", "1": "RAPID_OR_ABNORMAL", "2": "APNEA"}
LABELS = ("NORMAL", "RAPID_OR_ABNORMAL", "APNEA")


class MB10R1MetricsError(Exception):
    """Raised when metric inputs violate the frozen contract."""


def metric_bundle(
    labels: Iterable[int],
    predictions: Iterable[int],
    *,
    evaluated_sample_count: int | None = None,
) -> dict[str, Any]:
    """Frozen M-B0/M-B10A metrics with support-zero values defined as 0.0.

    Guard: never claim a positive evaluated_sample_count from empty labels
    (e.g. ``metric_bundle([], [], evaluated_sample_count=75)`` is refused).
    When ``evaluated_sample_count`` is provided it must equal ``len(labels)``.
    """
    truth_list = list(labels)
    pred_list = list(predictions)
    if evaluated_sample_count is not None:
        if int(evaluated_sample_count) > 0 and len(truth_list) == 0:
            raise MB10R1MetricsError("METRIC_EMPTY_LABELS_WITH_POSITIVE_EVALUATED_COUNT")
        if int(evaluated_sample_count) != len(truth_list):
            raise MB10R1MetricsError("METRIC_EVALUATED_SAMPLE_COUNT_MISMATCH")
    truth = np.asarray(truth_list, dtype=np.int64)
    pred = np.asarray(pred_list, dtype=np.int64)
    if truth.shape != pred.shape:
        raise MB10R1MetricsError("METRIC_LABEL_PREDICTION_SHAPE_MISMATCH")
    if truth.size and (np.any((truth < 0) | (truth > 2)) or np.any((pred < 0) | (pred > 2))):
        raise MB10R1MetricsError("METRIC_CLASS_INDEX_OUT_OF_RANGE")
    confusion = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(truth.tolist(), pred.tolist()):
        confusion[t, p] += 1
    per_class: dict[str, dict[str, Any]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for index, label in enumerate(LABELS):
        tp = int(confusion[index, index])
        support = int(confusion[index, :].sum())
        fp = int(confusion[:, index].sum() - tp)
        fn = int(support - tp)
        tn = int(confusion.sum() - tp - fp - fn)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        row = {
            "support": support,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "f1_score": round(float(f1), 6),
            "fpr": round(float(fpr), 6),
        }
        per_class[label] = row
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    distribution = {label: int(np.sum(pred == index)) for index, label in enumerate(LABELS)}
    zero_prediction = [label for label in LABELS if distribution[label] == 0]
    zero_recall = [label for label in LABELS if per_class[label]["recall"] == 0.0]
    return {
        "evaluated_sample_count": int(truth.size if evaluated_sample_count is None else evaluated_sample_count),
        "accuracy": round(float(np.mean(truth == pred)) if truth.size else 0.0, 6),
        "macro_f1": round(float(np.mean(f1s)) if f1s else 0.0, 6),
        "macro_precision": round(float(np.mean(precisions)) if precisions else 0.0, 6),
        "macro_recall": round(float(np.mean(recalls)) if recalls else 0.0, 6),
        "per_class": per_class,
        "apnea_proxy": {
            "precision": per_class["APNEA"]["precision"],
            "recall": per_class["APNEA"]["recall"],
            "misses": per_class["APNEA"]["fn"],
            "fpr": per_class["APNEA"]["fpr"],
        },
        "rapid_or_abnormal_recall": per_class["RAPID_OR_ABNORMAL"]["recall"],
        "confusion_matrix": confusion.tolist(),
        "prediction_distribution": distribution,
        "class_collapse": {
            "collapsed": bool(zero_prediction or zero_recall),
            "zero_prediction_classes": zero_prediction,
            "zero_recall_classes": zero_recall,
        },
    }


def subject_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Subject-level aggregation matching M-B10B semantics."""
    by_subject: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record.get("invalid"):
            continue
        by_subject.setdefault(str(record["subject_id"]), []).append(record)
    per_subject: dict[str, Any] = {}
    for subject_id in sorted(by_subject):
        rows = by_subject[subject_id]
        labels = [int(row["true_class_index"]) for row in rows]
        predictions = [int(row["predicted_class_index"]) for row in rows]
        metrics = metric_bundle(labels, predictions)
        per_subject[subject_id] = {
            "window_count": len(rows),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "per_class": metrics["per_class"],
            "prediction_distribution": metrics["prediction_distribution"],
        }
    values = [float(per_subject[item]["macro_f1"]) for item in sorted(per_subject)]
    if values:
        worst_id = min(sorted(per_subject), key=lambda subject: (per_subject[subject]["macro_f1"], subject))
        median = float(np.median(np.asarray(values, dtype=np.float64)))
        worst = float(per_subject[worst_id]["macro_f1"])
    else:
        worst_id, median, worst = None, 0.0, 0.0
    return {
        "subject_count": len(per_subject),
        "per_subject": per_subject,
        "median_subject_macro_f1": round(median, 6),
        "worst_subject_macro_f1": round(worst, 6),
        "worst_subject_id": worst_id,
    }


def saturation_audit_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate seed42 pre-clamp saturation diagnostics from ledger-like rows."""
    if not rows:
        return {
            "total_quantized_elements": 0,
            "pre_clamp_out_of_range_count": 0,
            "input_saturation_ratio": 0.0,
            "samples_with_any_saturation": 0,
            "worst_sample_saturation_ratio": 0.0,
            "worst_sample_window_id": None,
            "saturation_source": "pre-clamp quantized values before int8 clipping",
        }
    total_elements = 0
    saturated = 0
    affected = 0
    worst_ratio = -1.0
    worst_id: str | None = None
    for row in rows:
        count = int(row.get("input_saturation_count", 0))
        ratio = float(row.get("input_saturation_ratio", 0.0))
        # Each model-ready window is 300 int8 elements under frozen contract.
        total_elements += 300
        saturated += count
        if count > 0:
            affected += 1
        if ratio > worst_ratio:
            worst_ratio = ratio
            worst_id = str(row.get("window_id"))
    return {
        "total_quantized_elements": total_elements,
        "pre_clamp_out_of_range_count": saturated,
        "input_saturation_ratio": round(float(saturated / total_elements) if total_elements else 0.0, 6),
        "samples_with_any_saturation": affected,
        "worst_sample_saturation_ratio": round(float(max(worst_ratio, 0.0)), 6),
        "worst_sample_window_id": worst_id,
        "saturation_source": "pre-clamp quantized values before int8 clipping",
    }


def quantize_with_saturation(
    model_ready: np.ndarray,
    scale: float,
    zero_point: int,
    *,
    contract_id: str,
) -> dict[str, Any]:
    """Quantize float32 model-ready input and record pre-clamp saturation."""
    model_ready = np.asarray(model_ready, dtype=np.float32).reshape(1, 300, 1)
    if not np.all(np.isfinite(model_ready)) or not np.isfinite(scale) or scale <= 0:
        raise MB10R1MetricsError("INVALID_MODEL_READY_OR_QUANTIZATION")
    raw = np.rint(model_ready / np.float32(scale) + np.int32(zero_point))
    limits = np.iinfo(np.int8)
    saturation = (raw < limits.min) | (raw > limits.max)
    return {
        "contract_id": contract_id,
        "model_ready": model_ready,
        "input_int8": np.clip(raw, limits.min, limits.max).astype(np.int8),
        "input_saturation_count": int(np.sum(saturation)),
        "input_saturation_ratio": float(np.mean(saturation)),
    }
