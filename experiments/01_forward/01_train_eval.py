#!/usr/bin/env python3
"""Train and evaluate the forward surrogate (XGBoost + 22 features + log1p).

Reproduces the headline result on the held-out test set (n = 1,889):

    XGBoost (optimized): R^2 = 0.9684, MAPE = 10.2%, Acc@10% = 74.1%

The script trains three baseline configurations (raw 3 features) and three
optimized configurations (22 features + log1p), then evaluates the
optimized models on the held-out test set. The optimized XGBoost row is the
headline number cited in the thesis.

Pre-requisite: run ``00_split_data.py`` first to produce
``results/split_indices.npz``.
"""

from __future__ import annotations

import os

# Cross-platform: cap BLAS thread count before importing numpy / xgboost /
# sklearn so the same command works on macOS (where uncapped threads can
# trigger a PyTorch/BLAS deadlock) and on Windows/Linux (harmless no-op).
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import time
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from polyfem_ml import RANDOM_SEED, compute_metrics, features_forward_22
from polyfem_ml.data import load_forward_dataset


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def train_eval(
    name: str, model, X_tr: np.ndarray, Y_tr: np.ndarray, X_va: np.ndarray, Y_va: np.ndarray,
    *, use_log1p: bool,
) -> tuple[dict, object, StandardScaler, np.ndarray]:
    """Standard-scale, fit, predict (with log1p inverse if needed), score."""
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_va_s = scaler.transform(X_tr), scaler.transform(X_va)
    target = np.log1p(Y_tr) if use_log1p else Y_tr

    t0 = time.perf_counter()
    model.fit(X_tr_s, target)
    train_time = time.perf_counter() - t0

    pred_va = model.predict(X_va_s)
    if use_log1p:
        pred_va = np.expm1(pred_va)
    pred_va = np.maximum(pred_va, 0)

    metrics = compute_metrics(Y_va, pred_va)
    metrics["train_time_s"] = round(train_time, 2)
    return metrics, model, scaler, pred_va


def regional_breakdown(theta: np.ndarray, Y_true: np.ndarray, Y_pred: np.ndarray) -> list[dict]:
    """Per-theta-region Acc@10/20% on whatever array is passed (val or test)."""
    bins = [
        ("<80",       0,    80),
        ("[80,87)",   80,   87),
        ("[87,89)",   87,   89),
        ("[89,93)",   89,   93),
        ("[93,100)",  93,  100),
        (">=100",    100,  200),
    ]
    out = []
    for label, lo, hi in bins:
        mask = (theta >= lo) & (theta < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        m = compute_metrics(Y_true[mask], Y_pred[mask])
        m["region"] = label
        m["n"] = n
        out.append(m)
    return out


def main() -> None:
    # ---------- Load data and splits ----------
    X_raw, Y = load_forward_dataset()
    splits_path = RESULTS_DIR / "split_indices.npz"
    if not splits_path.exists():
        raise FileNotFoundError(f"Run 00_split_data.py first to produce {splits_path}")
    splits = np.load(splits_path)
    train_idx, val_idx, test_idx = splits["train_idx"], splits["val_idx"], splits["test_idx"]

    X_feat = features_forward_22(X_raw)
    Y_tr, Y_va, Y_te = Y[train_idx], Y[val_idx], Y[test_idx]
    theta_te = X_raw[test_idx, 0]

    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")
    print(f"Features: {X_feat.shape[1]} (raw 3 + 12 engineered + 7 h-onehot)\n")

    # ---------- PART A: Baselines (raw 3 features, no log1p) ----------
    print("=" * 70 + "\nPART A: Baselines (3 raw features)\n" + "=" * 70)
    X3_tr, X3_va = X_raw[train_idx], X_raw[val_idx]
    baselines = [
        ("RF (baseline)", RandomForestRegressor(
            n_estimators=500, min_samples_leaf=3, random_state=RANDOM_SEED, n_jobs=-1)),
        ("GBR (baseline)", GradientBoostingRegressor(
            n_estimators=500, max_depth=5, learning_rate=0.05, random_state=RANDOM_SEED)),
        ("XGBoost (baseline)", xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            random_state=RANDOM_SEED, n_jobs=-1, verbosity=0)),
        ("SVR (baseline)", SVR(kernel="rbf", C=1e6, gamma="scale")),
        ("MLP (baseline)", MLPRegressor(
            hidden_layer_sizes=(128, 64), max_iter=1000,
            random_state=RANDOM_SEED, early_stopping=True)),
    ]
    baseline_results: dict[str, dict] = {}
    for name, model in baselines:
        m, *_ = train_eval(name, model, X3_tr, Y_tr, X3_va, Y_va, use_log1p=False)
        baseline_results[name] = m
        print(f"  {name:>22}: R2={m['r2']:.4f}  MAPE={m['mape']:.1f}%  "
              f"Acc@10%={m['acc10']:.1f}%  Acc@20%={m['acc20']:.1f}%  ({m['train_time_s']:.1f}s)")

    # ---------- PART B: Optimized (22 features + log1p) ----------
    print("\n" + "=" * 70 + "\nPART B: Optimized (22 features + log1p target)\n" + "=" * 70)
    Xf_tr, Xf_va, Xf_te = X_feat[train_idx], X_feat[val_idx], X_feat[test_idx]
    optimized = [
        ("RF (optimized)", RandomForestRegressor(
            n_estimators=3000, min_samples_leaf=1, max_features=0.5,
            random_state=RANDOM_SEED, n_jobs=-1)),
        ("XGBoost (optimized)", xgb.XGBRegressor(
            n_estimators=5000, max_depth=8, learning_rate=0.01,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=1,
            random_state=RANDOM_SEED, n_jobs=-1, verbosity=0)),
        ("GBR (optimized)", GradientBoostingRegressor(
            n_estimators=2000, max_depth=8, learning_rate=0.03,
            subsample=0.8, min_samples_leaf=3, random_state=RANDOM_SEED)),
    ]
    opt_results: dict[str, dict] = {}
    opt_models: dict[str, object] = {}
    opt_scalers: dict[str, StandardScaler] = {}
    for name, model in optimized:
        m, trained, scaler, _ = train_eval(name, model, Xf_tr, Y_tr, Xf_va, Y_va, use_log1p=True)
        opt_results[name] = m
        opt_models[name] = trained
        opt_scalers[name] = scaler
        print(f"  {name:>22}: R2={m['r2']:.4f}  MAPE={m['mape']:.1f}%  "
              f"Acc@10%={m['acc10']:.1f}%  Acc@20%={m['acc20']:.1f}%  ({m['train_time_s']:.1f}s)")

    # ---------- PART C: Held-out test evaluation ----------
    print("\n" + "=" * 70 + "\nPART C: Held-Out Test Set (n = {})\n".format(len(test_idx))
          + "=" * 70)
    test_results: dict[str, dict] = {}
    test_preds: dict[str, np.ndarray] = {}
    for name in opt_models:
        X_te_s = opt_scalers[name].transform(Xf_te)
        pred = np.maximum(np.expm1(opt_models[name].predict(X_te_s)), 0)
        test_preds[name] = pred
        m = compute_metrics(Y_te, pred)
        test_results[name] = m
        print(f"  {name:>22}: R2={m['r2']:.4f}  MAPE={m['mape']:.1f}%  "
              f"Acc@10%={m['acc10']:.1f}%  Acc@20%={m['acc20']:.1f}%")

    # Regional breakdown for the headline (XGBoost optimized) test predictions.
    headline = "XGBoost (optimized)"
    regions = regional_breakdown(theta_te, Y_te, test_preds[headline])
    print(f"\n  Per-region (test, {headline}):")
    for r in regions:
        print(f"    {r['region']:>9}: n={r['n']:>4}  Acc@10%={r['acc10']:.1f}%  "
              f"Acc@20%={r['acc20']:.1f}%  R2={r['r2']:.4f}")

    # ---------- Save artifacts ----------
    RESULTS_DIR.mkdir(exist_ok=True)
    all_results = {
        "split": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "baseline_val": baseline_results,
        "optimized_val": opt_results,
        "test_results": test_results,
        "test_regional": regions,
        "metadata": {
            "seed": RANDOM_SEED,
            "feature_count": int(X_feat.shape[1]),
            "headline_test": test_results[headline],
        },
    }
    out_json = RESULTS_DIR / "all_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    np.save(RESULTS_DIR / "test_preds_xgboost.npy", test_preds[headline])
    np.save(RESULTS_DIR / "test_Y_true.npy", Y_te)

    # Save trained optimized models (XGBoost ~62 MB, RF ~4 GB, GBR ~35 MB).
    for name, model in opt_models.items():
        short = name.split(" ")[0].lower()
        joblib.dump(
            {"model": model, "scaler": opt_scalers[name]},
            RESULTS_DIR / f"model_{short}.joblib",
        )
    print(f"\nResults: {out_json}")

    # Also publish the headline (XGBoost) bundle to the package checkpoints
    # directory so `StressSurrogate.load_default()` works out of the box.
    package_ckpt = Path(__file__).resolve().parent.parent.parent / "polyfem_ml" / "checkpoints"
    package_ckpt.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": opt_models[headline], "scaler": opt_scalers[headline]},
        package_ckpt / "stress_xgboost.joblib",
    )
    print(f"Published headline bundle -> {package_ckpt / 'stress_xgboost.joblib'}")


if __name__ == "__main__":
    main()
