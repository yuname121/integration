#!/usr/bin/env python3
"""SafeNest Phase M-B2 — Real-Data Class-Imbalance Strategy Module.

Defines the 4 pre-registered class-imbalance strategies (Standard Unweighted CE,
Real-TRAIN Class Weighting, TRAIN-Only Random Oversampling, and Multiclass Focal Loss),
and helper utilities for dataset calculation, focal loss, and evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parents[1]

LABEL_NAMES = ["NORMAL", "RAPID_OR_ABNORMAL", "APNEA"]

STRATEGIES = [
    {
        "strategy_id": "M-B2_CE_UNWEIGHTED",
        "name": "CE_UNWEIGHTED",
        "type": "standard_ce",
        "description": "Standard unweighted sparse categorical cross-entropy baseline",
    },
    {
        "strategy_id": "M-B2_CE_CLASS_WEIGHT",
        "name": "CE_CLASS_WEIGHT",
        "type": "class_weighting",
        "description": "Balanced inverse-frequency class-weighted cross-entropy (TRAIN-only)",
    },
    {
        "strategy_id": "M-B2_CE_RANDOM_OVERSAMPLE",
        "name": "CE_RANDOM_OVERSAMPLE",
        "type": "random_oversample",
        "description": "TRAIN-only random oversampling of minority classes to match largest class",
    },
    {
        "strategy_id": "M-B2_FOCAL_CLASS_ALPHA",
        "name": "FOCAL_CLASS_ALPHA",
        "type": "focal_loss",
        "description": "Multiclass focal loss (gamma=2.0) with TRAIN-derived class alpha weights",
    },
]


def compute_train_class_weights(train_labels: list[int]) -> dict[int, float]:
    """Compute balanced inverse-frequency class weights w_c = N_train / (K * n_c) from TRAIN labels."""
    n_train = len(train_labels)
    k_classes = 3
    counts = {c: train_labels.count(c) for c in range(k_classes)}
    weights = {}
    for c in range(k_classes):
        if counts[c] == 0:
            raise ValueError(f"Class {c} has 0 samples in TRAIN split!")
        weights[c] = float(n_train / (k_classes * counts[c]))
    return weights


def build_oversampling_plan(
    train_windows: list[dict[str, Any]], seed: int = 42
) -> tuple[list[int], list[dict[str, Any]]]:
    """Build deterministic minority-only oversampling plan preserving all original TRAIN windows and duplicating only minority classes."""
    labels = [w["safenest_label_id"] for w in train_windows]
    counts = {c: labels.count(c) for c in range(3)}
    max_count = max(counts.values())

    rng = np.random.RandomState(seed)

    # All 327 original TRAIN indices are preserved
    all_original_indices = list(range(len(train_windows)))
    duplicate_counter = {idx: 0 for idx in all_original_indices}

    oversampled_indices = list(all_original_indices)

    # Draw duplicates ONLY for minority classes
    for c in range(3):
        c_indices = [idx for idx, l in enumerate(labels) if l == c]
        if len(c_indices) == 0:
            raise ValueError(f"Class {c} has 0 samples in TRAIN split!")
        extra_needed = max_count - len(c_indices)
        if extra_needed > 0:
            sampled_c = rng.choice(c_indices, size=extra_needed, replace=True).tolist()
            oversampled_indices.extend(sampled_c)
            for idx in sampled_c:
                duplicate_counter[idx] += 1

    plan_records = []
    for idx, w in enumerate(train_windows):
        plan_records.append(
            {
                "train_index": idx,
                "canonical_sample_index": int(w["canonical_sample_index"]),
                "window_id": w["window_id"],
                "subject_id": w["subject_id"],
                "recording_id": w["recording_id"],
                "class_id": int(w["safenest_label_id"]),
                "class_name": LABEL_NAMES[w["safenest_label_id"]],
                "original_occurrence": 1,
                "additional_duplicate_count": duplicate_counter[idx],
                "effective_multiplicity": 1 + duplicate_counter[idx],
            }
        )

    return oversampled_indices, plan_records


def build_multiclass_focal_loss(
    alpha_weights: dict[int, float], gamma: float = 2.0, epsilon: float = 1e-7
) -> tf.keras.losses.Loss:
    """Construct custom multiclass Focal Loss: FL(y, p) = -sum_c alpha_c * I(y=c) * (1-p_c)^gamma * log(max(p_c, eps))."""
    alpha_tensor = tf.constant([alpha_weights[0], alpha_weights[1], alpha_weights[2]], dtype=tf.float32)

    class MulticlassFocalLoss(tf.keras.losses.Loss):
        def __init__(self, name="multiclass_focal_loss", **kwargs):
            super().__init__(name=name, **kwargs)

        def call(self, y_true, y_pred):
            # y_true is sparse integer (batch,), y_pred is softmax probabilities (batch, 3)
            y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=3)
            y_pred_clipped = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

            # Focal factor: (1 - p_t)^gamma
            p_t = tf.reduce_sum(y_true_one_hot * y_pred_clipped, axis=-1)
            focal_factor = tf.pow(1.0 - p_t, gamma)

            # Alpha factor
            alpha_factor = tf.reduce_sum(y_true_one_hot * alpha_tensor, axis=-1)

            # Cross entropy
            ce = -tf.math.log(p_t)

            loss = alpha_factor * focal_factor * ce
            return tf.reduce_mean(loss)

    return MulticlassFocalLoss()


def compute_one_vs_rest_false_positives(
    val_true: np.ndarray, val_preds: np.ndarray
) -> dict[str, dict[str, Any]]:
    """Compute per-class confusion matrix, precision, recall, F1, and one-vs-rest FPR."""
    metrics = {}
    for cid, cname in enumerate(LABEL_NAMES):
        tp = int(np.sum((val_preds == cid) & (val_true == cid)))
        fp = int(np.sum((val_preds == cid) & (val_true != cid)))
        tn = int(np.sum((val_preds != cid) & (val_true != cid)))
        fn = int(np.sum((val_preds != cid) & (val_true == cid)))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        metrics[cname] = {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "f1_score": round(f1, 6),
            "fpr": round(fpr, 6),
            "support": int(np.sum(val_true == cid)),
        }
    return metrics


def rank_imbalance_strategies(
    results: list[dict[str, Any]], eps: float = 1e-5
) -> list[dict[str, Any]]:
    """Rank imbalance strategies under pre-registered 7-step rule with numerical tie tolerance eps."""
    candidates = [r for r in results if not r.get("is_class_collapsed", False)]
    if not candidates:
        raise ValueError("ALL 4 CLASS-IMBALANCE STRATEGIES COLLAPSED! No valid candidate winner.")

    simplicity_order = {
        "M-B2_CE_UNWEIGHTED": 0,
        "M-B2_CE_CLASS_WEIGHT": 1,
        "M-B2_CE_RANDOM_OVERSAMPLE": 2,
        "M-B2_FOCAL_CLASS_ALPHA": 3,
    }

    def compare_pair(a: dict[str, Any], b: dict[str, Any]) -> int:
        if a["strategy_id"] == b["strategy_id"]:
            return 0

        # Step 2: Macro F1
        f1_diff = a["macro_f1"] - b["macro_f1"]
        if abs(f1_diff) > eps:
            return 1 if f1_diff > 0 else -1

        # Step 3: Min per-class recall
        rec_diff = a["min_per_class_recall"] - b["min_per_class_recall"]
        if abs(rec_diff) > eps:
            return 1 if rec_diff > 0 else -1

        # Step 4: Macro precision
        prec_diff = a["macro_precision"] - b["macro_precision"]
        if abs(prec_diff) > eps:
            return 1 if prec_diff > 0 else -1

        # Step 5: Macro FPR (lower is better)
        fpr_diff = b["macro_fpr"] - a["macro_fpr"]
        if abs(fpr_diff) > eps:
            return 1 if fpr_diff > 0 else -1

        # Step 6: Simpler intervention
        simp_diff = simplicity_order.get(b["strategy_id"], 99) - simplicity_order.get(a["strategy_id"], 99)
        if simp_diff != 0:
            return 1 if simp_diff > 0 else -1

        # Step 7: Lexicographic strategy ID
        return 1 if a["strategy_id"] < b["strategy_id"] else -1

    import functools

    candidates.sort(key=functools.cmp_to_key(compare_pair), reverse=True)
    return candidates


    simplicity_order = {
        "M-B2_CE_UNWEIGHTED": 0,
        "M-B2_CE_CLASS_WEIGHT": 1,
        "M-B2_CE_RANDOM_OVERSAMPLE": 2,
        "M-B2_FOCAL_CLASS_ALPHA": 3,
    }

    def compare_pair(a: dict[str, Any], b: dict[str, Any]) -> int:
        # Step 2: Macro F1
        f1_diff = a["macro_f1"] - b["macro_f1"]
        if abs(f1_diff) > eps:
            return 1 if f1_diff > 0 else -1

        # Step 3: Min recall
        rec_diff = a["min_per_class_recall"] - b["min_per_class_recall"]
        if abs(rec_diff) > eps:
            return 1 if rec_diff > 0 else -1

        # Step 4: Macro precision
        prec_diff = a["macro_precision"] - b["macro_precision"]
        if abs(prec_diff) > eps:
            return 1 if prec_diff > 0 else -1

        # Step 5: Macro FPR (lower is better)
        fpr_diff = b["macro_fpr"] - a["macro_fpr"]
        if abs(fpr_diff) > eps:
            return 1 if fpr_diff > 0 else -1

        # Step 6: Simpler intervention
        simp_diff = simplicity_order.get(b["strategy_id"], 99) - simplicity_order.get(a["strategy_id"], 99)
        if simp_diff != 0:
            return 1 if simp_diff > 0 else -1

        # Step 7: Lexicographic strategy ID
        return 1 if a["strategy_id"] < b["strategy_id"] else -1

    import functools

    candidates.sort(key=functools.cmp_to_key(compare_pair), reverse=True)
    return candidates


def compute_subject_level_diagnostics(
    val_windows: list[dict[str, Any]], val_preds: np.ndarray
) -> dict[str, Any]:
    """Compute per-subject validation accuracy, Macro F1, and class support breakdown."""
    subject_map = {}
    for idx, w in enumerate(val_windows):
        sid = w["subject_id"]
        if sid not in subject_map:
            subject_map[sid] = {"true": [], "pred": []}
        subject_map[sid]["true"].append(w["safenest_label_id"])
        subject_map[sid]["pred"].append(val_preds[idx])

    subject_results = {}
    macro_f1s = []
    accuracies = []

    for sid, data in sorted(subject_map.items()):
        y_true = np.array(data["true"])
        y_pred = np.array(data["pred"])
        n_sub = len(y_true)

        acc = float(np.mean(y_true == y_pred))
        accuracies.append(acc)

        sub_f1s = []
        c_metrics = {}
        for cid, cname in enumerate(LABEL_NAMES):
            sub_c_true = np.sum(y_true == cid)
            tp = int(np.sum((y_pred == cid) & (y_true == cid)))
            fp = int(np.sum((y_pred == cid) & (y_true != cid)))
            tn = int(np.sum((y_pred != cid) & (y_true != cid)))
            fn = int(np.sum((y_pred != cid) & (y_true == cid)))

            if sub_c_true == 0:
                c_metrics[cname] = {
                    "support": 0,
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                    "recall": "NOT_DEFINED_NO_SUPPORT",
                    "precision": "NOT_DEFINED_NO_SUPPORT" if (tp + fp) == 0 else round(float(tp / (tp + fp)), 6),
                    "f1": "NOT_DEFINED_NO_SUPPORT",
                }
            else:
                prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
                sub_f1s.append(f1)
                c_metrics[cname] = {
                    "support": int(sub_c_true),
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                    "recall": round(rec, 6),
                    "precision": round(prec, 6),
                    "f1": round(f1, 6),
                }

        sub_macro_f1 = float(np.mean(sub_f1s)) if sub_f1s else 0.0
        macro_f1s.append(sub_macro_f1)

        apnea_tp = int(np.sum((y_pred == 2) & (y_true == 2)))
        apnea_fp = int(np.sum((y_pred == 2) & (y_true != 2)))
        apnea_fn = int(np.sum((y_pred != 2) & (y_true == 2)))

        rapid_tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        rapid_fp = int(np.sum((y_pred == 1) & (y_true != 1)))
        rapid_fn = int(np.sum((y_pred != 1) & (y_true == 1)))

        subject_results[sid] = {
            "window_count": n_sub,
            "accuracy": round(acc, 6),
            "subject_macro_f1": round(sub_macro_f1, 6),
            "apnea_fp": apnea_fp,
            "apnea_fn": apnea_fn,
            "rapid_fp": rapid_fp,
            "rapid_fn": rapid_fn,
            "class_metrics": c_metrics,
            "prediction_distribution": {
                "NORMAL": int(np.sum(y_pred == 0)),
                "RAPID_OR_ABNORMAL": int(np.sum(y_pred == 1)),
                "APNEA": int(np.sum(y_pred == 2)),
            },
        }

    return {
        "per_subject": subject_results,
        "summary_across_subjects": {
            "subject_count": len(subject_results),
            "mean_accuracy": round(float(np.mean(accuracies)), 6),
            "median_accuracy": round(float(np.median(accuracies)), 6),
            "std_accuracy": round(float(np.std(accuracies)), 6),
            "min_accuracy": round(float(np.min(accuracies)), 6),
            "max_accuracy": round(float(np.max(accuracies)), 6),
            "mean_macro_f1": round(float(np.mean(macro_f1s)), 6),
            "median_macro_f1": round(float(np.median(macro_f1s)), 6),
            "std_macro_f1": round(float(np.std(macro_f1s)), 6),
            "min_macro_f1": round(float(np.min(macro_f1s)), 6),
            "max_macro_f1": round(float(np.max(macro_f1s)), 6),
        },
    }
