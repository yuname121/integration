# SafeNest mmWave Track — Phase M-B4 Multi-Seed Reproducibility & Stability Module

import gc
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from mmwave_m_b2_imbalance import LABEL_NAMES, compute_one_vs_rest_false_positives, compute_subject_level_diagnostics
from mmwave_m_b3_architecture import (
    build_architecture_a,
    build_architecture_b,
    build_model_by_id,
    compute_numerical_weights_sha256,
    reset_seeds,
)

SEEDS = [42, 43, 44]


def train_architecture_seed(
    arch_id: str,
    seed: int,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    batch_size: int = 32,
    epochs: int = 25,
    learning_rate: float = 0.001,
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """Train a shortlisted model architecture under a specific training-initialization seed."""
    reset_seeds(seed)

    model = build_model_by_id(arch_id)
    initial_weights_sha = compute_numerical_weights_sha256(model)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=0,
        )
    ]

    history = model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    final_weights_sha = compute_numerical_weights_sha256(model)
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    epochs_run = len(history.history["loss"])

    info = {
        "architecture_id": arch_id,
        "seed": seed,
        "initial_weights_sha256": initial_weights_sha,
        "final_weights_sha256": final_weights_sha,
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "param_counts": {
            "total_params": int(model.count_params()),
            "trainable_params": int(sum([tf.keras.backend.count_params(p) for p in model.trainable_weights])),
        },
        "history": {
            "loss": [float(v) for v in history.history["loss"]],
            "val_loss": [float(v) for v in history.history["val_loss"]],
            "accuracy": [float(v) for v in history.history.get("accuracy", [])],
            "val_accuracy": [float(v) for v in history.history.get("val_accuracy", [])],
        },
    }

    return model, info


def rank_multiseed_architectures(
    multiseed_results: List[Dict[str, Any]],
    eps: float = 1e-5,
) -> List[Dict[str, Any]]:
    """Rank shortlisted architectures across multi-seed runs according to preregistered M-B4 rules.
    
    Ranking Rules:
    1. Step 1: Exclude collapsed architectures (collapsed_seed_count > 0).
    2. Step 2: Higher worst_seed_macro_f1 wins.
    3. Step 3: Higher mean_macro_f1 (within eps=1e-5).
    4. Step 4: Higher worst_seed_min_per_class_recall (within eps=1e-5).
    5. Step 5: Lower std_macro_f1 (within eps=1e-5).
    6. Step 6: Lower total_params.
    7. Step 7: Smaller M-B3 strict INT8 size.
    8. Step 8: Lexicographic architecture_id.
    """
    eligible = [a for a in multiseed_results if a.get("collapsed_seed_count", 0) == 0]
    ineligible = [a for a in multiseed_results if a.get("collapsed_seed_count", 0) > 0]

    def compare_key(a: Dict[str, Any]) -> Tuple:
        w_f1 = a["macro_f1"]["worst_seed_val"]
        m_f1 = a["macro_f1"]["mean"]
        w_min_rec = a["min_per_class_recall"]["worst_seed_val"]
        std_f1 = a["macro_f1"]["std"]
        params = a["total_params"]
        int8_sz = a.get("strict_int8_bytes", 999999999) or 999999999
        aid = a["architecture_id"]

        # Discretize continuous float metrics by eps=1e-5 for stable tie handling
        return (
            round(w_f1 / eps),
            round(m_f1 / eps),
            round(w_min_rec / eps),
            -round(std_f1 / eps),
            -params,
            -int8_sz,
            aid,
        )

    ranked_eligible = sorted(eligible, key=compare_key, reverse=True)
    ranked_ineligible = sorted(ineligible, key=compare_key, reverse=True)

    return ranked_eligible + ranked_ineligible
