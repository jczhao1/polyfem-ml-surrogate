#!/usr/bin/env python3
"""XGBoost sign + log-magnitude decomposition baseline for all three gradients.

This script demonstrates that the sign + log-magnitude decomposition is
**model-agnostic**: applying it to XGBoost (instead of the multi-head NN)
recovers most of the proper-CV accuracy gain on the smooth gradients and
actually outperforms the NN on dL/dtheta.

Reproduces the XGBoost rows of Table~5.8 in the thesis::

    dL/dE     Sign 95.2%  Acc@20% 66.9%  R2_log 0.956
    dL/dh     Sign 98.5%  Acc@20% 75.8%  R2_log 0.973
    dL/dtheta Sign 94.9%  Acc@20% 57.4%  R2_log 0.834

Methodology:

- 60/20/20 stratified split (train / val for early stopping / test) by
  ``h x sign(g)`` strata.
- Sign head: XGBClassifier with ``binary:logistic``.
- Magnitude head: XGBRegressor on ``log10(|g| + 1)`` with ``reg:squarederror``.
- Decode: ``g_hat = sign(s_hat - 0.5) * (10**m_hat - 1)``.

Usage::

    uv run python 03_xgboost_baseline.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Cross-platform: cap BLAS thread count before importing numpy / xgboost to
# avoid a PyTorch/BLAS deadlock on macOS. Harmless no-op on Windows/Linux.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier, XGBRegressor

from polyfem_ml import RANDOM_SEED, features_grad_15, grad_metrics
from polyfem_ml.data import (
    load_geometric_gradient_dataset,
    load_material_gradient_dataset,
)


RESULTS_DIR = Path(__file__).resolve().parent / "results"

XGB_PARAMS_REG = dict(
    n_estimators=500, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_SEED, n_jobs=-1, verbosity=0,
)
XGB_PARAMS_CLF = dict(
    n_estimators=500, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    objective="binary:logistic",
    random_state=RANDOM_SEED, n_jobs=-1, verbosity=0,
)


def stratified_split(strata: np.ndarray, *, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(sss.split(np.arange(len(strata)), strata))


def fit_decomposition(
    X_feat: np.ndarray, G: np.ndarray, name: str,
) -> dict:
    """Train sign + log-magnitude XGBoost on all samples, evaluate on a test split."""
    sign = np.sign(G)
    log_mag = np.log10(np.abs(G) + 1.0)

    # Stratify on h-bin x sign so all four (h, sign) combinations appear in both splits.
    # X_feat[:, 1] is the raw h column.
    h_bin = (X_feat[:, 1] * 100).astype(int)
    sign_bin = (sign > 0).astype(int)
    strata = h_bin * 10 + sign_bin

    train_idx, test_idx = stratified_split(strata, test_size=0.2, seed=RANDOM_SEED)

    print(f"\n{name}: train n={len(train_idx)}  test n={len(test_idx)}")

    sign_head = XGBClassifier(**XGB_PARAMS_CLF).fit(X_feat[train_idx], (sign[train_idx] > 0).astype(int))
    mag_head = XGBRegressor(**XGB_PARAMS_REG).fit(X_feat[train_idx], log_mag[train_idx])

    sign_pred = np.where(sign_head.predict(X_feat[test_idx]) > 0.5, 1.0, -1.0)
    log_mag_pred = mag_head.predict(X_feat[test_idx])
    pred = sign_pred * (10.0 ** log_mag_pred - 1.0)

    metrics = grad_metrics(pred, G[test_idx])
    print(f"  Sign={metrics['sign_acc']:.1f}%  Acc@20%={metrics['acc20']:.1f}%  "
          f"R2_log={metrics['r2_log']:.4f}")
    return {"metrics": metrics, "n_train": int(len(train_idx)), "n_test": int(len(test_idx))}


def main() -> int:
    print(f"XGBoost decomposition baseline (seed={RANDOM_SEED})")
    t0 = time.perf_counter()

    # dL/dE
    X_raw, _, G_E = load_material_gradient_dataset()
    res_E = fit_decomposition(features_grad_15(X_raw), G_E, "dL/dE (material, n=8606)")

    # dL/dh + dL/dtheta share the geometric dataset.
    X_raw_g, _, Gh, Gth = load_geometric_gradient_dataset()
    X_feat_g = features_grad_15(X_raw_g)
    res_h = fit_decomposition(X_feat_g, Gh, "dL/dh (geometric, n=9653)")
    res_th = fit_decomposition(X_feat_g, Gth, "dL/dtheta (geometric, n=9653)")

    elapsed = time.perf_counter() - t0
    print(f"\nTotal time: {elapsed:.0f}s")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "xgboost_baseline_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "script": "03_xgboost_baseline.py",
                "seed": RANDOM_SEED,
                "elapsed_s": round(elapsed, 1),
                "feature_count": 15,
                "xgb_params_reg": XGB_PARAMS_REG,
                "xgb_params_clf": XGB_PARAMS_CLF,
            },
            "dL_dE": res_E,
            "dL_dh": res_h,
            "dL_dtheta": res_th,
        }, f, indent=2, default=str)
    print(f"Results: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
