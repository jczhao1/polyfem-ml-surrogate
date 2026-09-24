#!/usr/bin/env python3
"""Proper 5-fold CV for the dL/dh + dL/dtheta (geometric) gradient surrogate.

Reproduces the headline results
    dL/dh:     Acc@20% = 74.5 +/- 1.0%, Sign = 98.6 +/- 0.3%
    dL/dtheta: Acc@20% = 46.7 +/- 1.7%, Sign = 94.8 +/- 0.3%
plus a per-theta-region breakdown of dL/dtheta from out-of-fold predictions.

Methodology and hyperparameters match :file:`01_proper_cv_dLdE.py`; the
architecture is :class:`MultiHead5` (one extra mag/sign head pair for the
second gradient component).

Usage::

    uv run python 02_proper_cv_htheta.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Cross-platform: cap BLAS thread count before importing numpy/torch to avoid
# a PyTorch/BLAS deadlock on macOS. Harmless no-op on Windows/Linux.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

from polyfem_ml import (
    MultiHead5,
    RANDOM_SEED,
    N_FOLDS,
    TrainConfig,
    decode_gradient,
    grad_metrics,
    inner_split,
    loss_ramp,
)
from polyfem_ml.data import load_geometric_gradient_dataset


RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Theta region boundaries used to break down dL/dtheta performance by physics regime.
THETA_REGIONS = [
    (60, 75, "[60,75)", "Smooth"),
    (75, 85, "[75,85)", "Smooth"),
    (85, 91, "[85,91)", "Transition"),
    (91, 100, "[91,100)", "Hardest"),
    (100, 111, "[100,111)", "Hard"),
]


def train_one_fold(
    X: np.ndarray, Y_log: np.ndarray,
    Gh: np.ndarray, Gtheta: np.ndarray,
    Gh_log_abs: np.ndarray, Gtheta_log_abs: np.ndarray,
    Gh_sign: np.ndarray, Gtheta_sign: np.ndarray,
    outer_train_idx: np.ndarray, outer_test_idx: np.ndarray,
    *, seed: int, cfg: TrainConfig, device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Train one fold; return ``(pred_h, pred_theta, artifacts)``.

    ``artifacts`` carries the best model's ``state_dict`` and the per-fold
    normalisation statistics, so the fold can later be republished as a
    checkpoint consumed by :class:`polyfem_ml.GradientSurrogate`.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    inner_train_idx, inner_val_idx = inner_split(outer_train_idx, val_frac=0.1, rng=rng)

    X_mean = X[inner_train_idx].mean(axis=0)
    X_std = X[inner_train_idx].std(axis=0) + 1e-8
    Y_mean = Y_log[inner_train_idx].mean()
    Y_std = Y_log[inner_train_idx].std() + 1e-8
    Gh_mean = Gh_log_abs[inner_train_idx].mean()
    Gh_std = Gh_log_abs[inner_train_idx].std() + 1e-8
    Gth_mean = Gtheta_log_abs[inner_train_idx].mean()
    Gth_std = Gtheta_log_abs[inner_train_idx].std() + 1e-8

    def _norm_x(idx: np.ndarray) -> np.ndarray:
        return (X[idx] - X_mean) / X_std

    Xtr = torch.tensor(_norm_x(inner_train_idx), dtype=torch.float32, device=device)
    Xva = torch.tensor(_norm_x(inner_val_idx), dtype=torch.float32, device=device)
    Xte = torch.tensor(_norm_x(outer_test_idx), dtype=torch.float32, device=device)
    Ytr = torch.tensor((Y_log[inner_train_idx] - Y_mean) / Y_std, dtype=torch.float32, device=device)
    Ghtr = torch.tensor((Gh_log_abs[inner_train_idx] - Gh_mean) / Gh_std, dtype=torch.float32, device=device)
    Gthtr = torch.tensor((Gtheta_log_abs[inner_train_idx] - Gth_mean) / Gth_std, dtype=torch.float32, device=device)
    Sh_tr = torch.tensor((Gh_sign[inner_train_idx] > 0).astype(np.float32), device=device)
    Sth_tr = torch.tensor((Gtheta_sign[inner_train_idx] > 0).astype(np.float32), device=device)

    model = MultiHead5(in_dim=3, hidden=cfg.hidden, n_layers=cfg.n_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.n_epochs)

    n_train = len(inner_train_idx)
    n_batches = (n_train + cfg.batch_size - 1) // cfg.batch_size
    best_metric = -float("inf")
    wait = 0
    best_state: dict | None = None

    for epoch in range(cfg.n_epochs):
        model.train()
        ramp = loss_ramp(epoch, warmup=cfg.warmup_epochs, window=cfg.ramp_window)
        perm_t = torch.randperm(n_train, device=device)

        for i in range(n_batches):
            idx = perm_t[i * cfg.batch_size : (i + 1) * cfg.batch_size]
            pv, pmh, psh, pmth, psth = model(Xtr[idx])
            loss_val = nn.functional.mse_loss(pv, Ytr[idx])
            loss_mh = nn.functional.mse_loss(pmh, Ghtr[idx])
            loss_sh = nn.functional.binary_cross_entropy_with_logits(psh, Sh_tr[idx])
            loss_mth = nn.functional.mse_loss(pmth, Gthtr[idx])
            loss_sth = nn.functional.binary_cross_entropy_with_logits(psth, Sth_tr[idx])
            # Loss weights match the thesis: more on dL/dtheta (3.0/1.5) than dL/dh (2.0/1.0)
            # to encourage the harder gradient.
            loss = loss_val + ramp * (2.0 * loss_mh + 1.0 * loss_sh
                                      + 3.0 * loss_mth + 1.5 * loss_sth)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        sch.step()

        if (epoch + 1) % cfg.check_every == 0:
            model.eval()
            with torch.no_grad():
                _, pmh_va, psh_va, pmth_va, psth_va = model(Xva)
            pred_h_va = decode_gradient(pmh_va.cpu().numpy(), psh_va.cpu().numpy(),
                                        log_mag_mean=Gh_mean, log_mag_std=Gh_std)
            pred_th_va = decode_gradient(pmth_va.cpu().numpy(), psth_va.cpu().numpy(),
                                         log_mag_mean=Gth_mean, log_mag_std=Gth_std)
            gt_h = Gh[inner_val_idx]
            gt_th = Gtheta[inner_val_idx]
            mask_h = np.abs(gt_h) > 1e-6
            mask_th = np.abs(gt_th) > 1e-6
            acc20_h = ((np.abs(pred_h_va[mask_h] - gt_h[mask_h]) / np.abs(gt_h[mask_h])) <= 0.20).mean() * 100 if mask_h.any() else 0.0
            acc20_th = ((np.abs(pred_th_va[mask_th] - gt_th[mask_th]) / np.abs(gt_th[mask_th])) <= 0.20).mean() * 100 if mask_th.any() else 0.0
            # Weighted toward dL/dtheta (the harder one) for early-stopping.
            metric = 0.4 * acc20_h + 0.6 * acc20_th
            if metric > best_metric:
                best_metric = metric
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
            if wait >= cfg.patience_checks:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _, pmh_te, psh_te, pmth_te, psth_te = model(Xte)
    pred_h = decode_gradient(pmh_te.cpu().numpy(), psh_te.cpu().numpy(),
                             log_mag_mean=Gh_mean, log_mag_std=Gh_std)
    pred_th = decode_gradient(pmth_te.cpu().numpy(), psth_te.cpu().numpy(),
                              log_mag_mean=Gth_mean, log_mag_std=Gth_std)
    artifacts = {
        "state_dict": best_state,
        "best_inner_val_metric": best_metric,
        "normalisation": {
            "X_mean":  X_mean.tolist(),
            "X_std":   X_std.tolist(),
            "Y_mean":  float(Y_mean),
            "Y_std":   float(Y_std),
            # The h_theta branch decodes two gradients, so we store both
            # (Gh_mean, Gh_std) and (Gth_mean, Gth_std). The inference API
            # uses ``G_mean`` / ``G_std`` as the *primary* (h) statistics
            # and reads the theta-specific stats from the nested keys.
            "G_mean":     float(Gh_mean),
            "G_std":      float(Gh_std),
            "Gtheta_mean": float(Gth_mean),
            "Gtheta_std":  float(Gth_std),
        },
    }
    return pred_h, pred_th, artifacts


def per_region_breakdown(theta: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> list[dict]:
    """Compute Acc@10/20% and sign agreement for each theta region from OOF predictions."""
    out = []
    for lo, hi, label, regime in THETA_REGIONS:
        mask = (theta >= lo) & (theta < hi)
        n_r = int(mask.sum())
        if n_r == 0:
            continue
        gt_r = gt[mask]
        pred_r = pred[mask]
        sign_ag = ((pred_r > 0) == (gt_r > 0)).mean() * 100
        valid = np.abs(gt_r) > 1e-6
        rel = np.abs(pred_r[valid] - gt_r[valid]) / np.abs(gt_r[valid])
        out.append({
            "region": label, "n": n_r, "regime": regime,
            "sign_acc": round(float(sign_ag), 1),
            "acc20": round(float((rel <= 0.20).mean() * 100), 1),
            "acc10": round(float((rel <= 0.10).mean() * 100), 1),
        })
    return out


def main() -> int:
    print(f"5-fold CV for dL/dh + dL/dtheta (seed={RANDOM_SEED}, folds={N_FOLDS})")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X_raw, Y, Gh, Gth = load_geometric_gradient_dataset()
    n = len(Gh)
    print(f"Loaded {n} samples")

    Y_log = np.log1p(Y)
    Gh_log_abs = np.log10(np.abs(Gh) + 1.0)
    Gth_log_abs = np.log10(np.abs(Gth) + 1.0)
    Gh_sign = np.sign(Gh)
    Gth_sign = np.sign(Gth)

    cfg = TrainConfig()
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fm_h, fm_th, oof_h, oof_th = [], [], np.zeros(n), np.zeros(n)
    fold_artifacts: list[dict] = []

    t0 = time.perf_counter()
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(np.arange(n))):
        ft0 = time.perf_counter()
        pred_h, pred_th, artifacts = train_one_fold(
            X_raw, Y_log, Gh, Gth, Gh_log_abs, Gth_log_abs, Gh_sign, Gth_sign,
            train_idx, test_idx, seed=RANDOM_SEED + fold_i, cfg=cfg, device=device,
        )
        oof_h[test_idx] = pred_h
        oof_th[test_idx] = pred_th
        m_h = grad_metrics(pred_h, Gh[test_idx])
        m_th = grad_metrics(pred_th, Gth[test_idx])
        fm_h.append(m_h)
        fm_th.append(m_th)
        fold_artifacts.append(artifacts)
        print(f"  Fold {fold_i + 1} ({time.perf_counter() - ft0:.0f}s): "
              f"dL/dh Acc@20%={m_h['acc20']:.1f}%  |  dL/dtheta Acc@20%={m_th['acc20']:.1f}%")

    elapsed = time.perf_counter() - t0

    def agg(folds: list[dict]) -> tuple[dict, dict]:
        m = {k: round(float(np.mean([f[k] for f in folds])), 4 if "r2" in k else 1) for k in folds[0]}
        s = {k: round(float(np.std([f[k] for f in folds])), 4 if "r2" in k else 1) for k in folds[0]}
        return m, s

    mean_h, std_h = agg(fm_h)
    mean_th, std_th = agg(fm_th)

    print(f"\nTotal: {elapsed:.0f}s ({elapsed / 60:.1f}min)\n")
    print(f"  dL/dh   :  Sign {mean_h['sign_acc']:.1f}+/-{std_h['sign_acc']:.1f}%  "
          f"Acc@20% {mean_h['acc20']:.1f}+/-{std_h['acc20']:.1f}%  "
          f"R2_log {mean_h['r2_log']:.4f}")
    print(f"  dL/dth  :  Sign {mean_th['sign_acc']:.1f}+/-{std_th['sign_acc']:.1f}%  "
          f"Acc@20% {mean_th['acc20']:.1f}+/-{std_th['acc20']:.1f}%  "
          f"R2_log {mean_th['r2_log']:.4f}")

    region_breakdown = per_region_breakdown(X_raw[:, 2], Gth, oof_th)
    print("\n  Per-theta region (dL/dtheta, OOF):")
    for r in region_breakdown:
        print(f"    {r['region']:12s} n={r['n']:4d}  Sign={r['sign_acc']:.1f}%  "
              f"Acc@20%={r['acc20']:.1f}%  Acc@10%={r['acc10']:.1f}%  ({r['regime']})")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "proper_cv_htheta_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "script": "02_proper_cv_htheta.py",
                "seed": RANDOM_SEED,
                "n_folds": N_FOLDS,
                "elapsed_s": round(elapsed, 1),
                "config": cfg.__dict__,
            },
            "n_samples": n,
            "dL_dh": {"fold_metrics": fm_h, "mean": mean_h, "std": std_h},
            "dL_dtheta": {
                "fold_metrics": fm_th, "mean": mean_th, "std": std_th,
                "per_region_oof": region_breakdown,
            },
        }, f, indent=2)
    print(f"\nResults: {out_path}")

    # ------------------------------------------------------------------
    # Publish the best fold's model + normalisation to the package
    # checkpoints directory so `GradientSurrogate.load_default()` works.
    # We score each fold by the average of its two Acc@20% values, since
    # the network solves both heads jointly.
    # ------------------------------------------------------------------
    avg_acc20 = [0.5 * (fm_h[i]["acc20"] + fm_th[i]["acc20"]) for i in range(N_FOLDS)]
    best_fold = max(range(N_FOLDS), key=lambda i: avg_acc20[i])
    print(f"\nBest fold for h_theta checkpoint: fold {best_fold + 1} "
          f"(dL/dh Acc@20% = {fm_h[best_fold]['acc20']:.1f}%, "
          f"dL/dtheta Acc@20% = {fm_th[best_fold]['acc20']:.1f}%)")
    package_ckpt = Path(__file__).resolve().parent.parent.parent / "polyfem_ml" / "checkpoints"
    package_ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(fold_artifacts[best_fold]["state_dict"], package_ckpt / "model_h_theta.pt")

    meta_path = package_ckpt / "gradient_surrogate.json"
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
    meta["hidden"] = cfg.hidden
    meta["n_layers"] = cfg.n_layers
    meta.setdefault("normalisation", {})
    meta["normalisation"]["h_theta"] = fold_artifacts[best_fold]["normalisation"]
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Published checkpoint -> {package_ckpt / 'model_h_theta.pt'}")
    print(f"Updated metadata     -> {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
