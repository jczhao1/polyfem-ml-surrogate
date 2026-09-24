#!/usr/bin/env python3
"""Proper 5-fold CV for the dL/dE (material) gradient surrogate.

Reproduces the headline result Acc@20% = 68.7 ± 1.3% (Sign 97.2 ± 0.3%).

Methodology (see Section 4.4.2 of the thesis):

- Outer 5-fold KFold over n = 8,606 samples, seed 42.
- Within each outer training fold, an inner 90/10 random split provides the
  early-stopping validation set; the held-out outer test fold is never used
  during model selection.
- Architecture: ``MultiHead3`` with hidden=512, n_layers=6.
- Loss = MSE(value) + ramp * (3.0 * MSE(log_mag) + 1.5 * BCE(sign)).
  Ramp = clip((epoch - 30) / 40, 0, 1).
- AdamW lr=5e-4 wd=1e-5, cosine annealing T_max=400, batch 512, grad clip 1.
- Early stop on inner-val Acc@20%, patience 40 checks (every 20 epochs).

Usage::

    uv run python 01_proper_cv_dLdE.py
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
    MultiHead3,
    RANDOM_SEED,
    N_FOLDS,
    TrainConfig,
    decode_gradient,
    grad_metrics,
    inner_split,
    loss_ramp,
)
from polyfem_ml.data import load_material_gradient_dataset


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def train_one_fold(
    X: np.ndarray, Y_log: np.ndarray, G: np.ndarray, G_log_abs: np.ndarray, G_sign: np.ndarray,
    outer_train_idx: np.ndarray, outer_test_idx: np.ndarray,
    *, seed: int, cfg: TrainConfig, device: torch.device,
) -> tuple[np.ndarray, dict]:
    """Train one fold; return (predictions on outer test, fold artifacts).

    ``fold artifacts`` is a dict containing the best model's ``state_dict``
    and the input/output normalisation statistics needed to re-use the model
    via :class:`polyfem_ml.GradientSurrogate`.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # Inner 90/10 split (never the outer test fold).
    inner_train_idx, inner_val_idx = inner_split(outer_train_idx, val_frac=0.1, rng=rng)

    # Per-fold normalization (computed on inner train only).
    X_mean = X[inner_train_idx].mean(axis=0)
    X_std = X[inner_train_idx].std(axis=0) + 1e-8
    Y_mean = Y_log[inner_train_idx].mean()
    Y_std = Y_log[inner_train_idx].std() + 1e-8
    G_mean = G_log_abs[inner_train_idx].mean()
    G_std = G_log_abs[inner_train_idx].std() + 1e-8

    def _norm_x(idx: np.ndarray) -> np.ndarray:
        return (X[idx] - X_mean) / X_std

    Xtr = torch.tensor(_norm_x(inner_train_idx), dtype=torch.float32, device=device)
    Xva = torch.tensor(_norm_x(inner_val_idx), dtype=torch.float32, device=device)
    Xte = torch.tensor(_norm_x(outer_test_idx), dtype=torch.float32, device=device)
    Ytr = torch.tensor((Y_log[inner_train_idx] - Y_mean) / Y_std, dtype=torch.float32, device=device)
    Gtr = torch.tensor((G_log_abs[inner_train_idx] - G_mean) / G_std, dtype=torch.float32, device=device)
    Str = torch.tensor((G_sign[inner_train_idx] > 0).astype(np.float32), device=device)

    model = MultiHead3(in_dim=3, hidden=cfg.hidden, n_layers=cfg.n_layers).to(device)
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
            pv, pmag, psign = model(Xtr[idx])
            loss_val = nn.functional.mse_loss(pv, Ytr[idx])
            loss_mag = nn.functional.mse_loss(pmag, Gtr[idx])
            loss_sign = nn.functional.binary_cross_entropy_with_logits(psign, Str[idx])
            loss = loss_val + ramp * (3.0 * loss_mag + 1.5 * loss_sign)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        sch.step()

        # Early stopping on inner val Acc@20% (never the test fold).
        if (epoch + 1) % cfg.check_every == 0:
            model.eval()
            with torch.no_grad():
                _, pmag_va, psign_va = model(Xva)
            pred_va = decode_gradient(
                pmag_va.cpu().numpy(), psign_va.cpu().numpy(),
                log_mag_mean=G_mean, log_mag_std=G_std,
            )
            gt_va = G[inner_val_idx]
            mask = np.abs(gt_va) > 1e-6
            if mask.any():
                rel = np.abs(pred_va[mask] - gt_va[mask]) / np.abs(gt_va[mask])
                acc20 = (rel <= 0.20).mean() * 100
            else:
                acc20 = 0.0
            if acc20 > best_metric:
                best_metric = acc20
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
            if wait >= cfg.patience_checks:
                break

    assert best_state is not None, "model never validated; increase n_epochs or check_every"
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _, pmag_te, psign_te = model(Xte)
    preds = decode_gradient(
        pmag_te.cpu().numpy(), psign_te.cpu().numpy(),
        log_mag_mean=G_mean, log_mag_std=G_std,
    )
    artifacts = {
        "state_dict": best_state,
        "best_inner_val_acc20": best_metric,
        "normalisation": {
            "X_mean": X_mean.tolist(),
            "X_std":  X_std.tolist(),
            "Y_mean": float(Y_mean),
            "Y_std":  float(Y_std),
            "G_mean": float(G_mean),
            "G_std":  float(G_std),
        },
    }
    return preds, artifacts


def main() -> int:
    print(f"5-fold CV for dL/dE (seed={RANDOM_SEED}, folds={N_FOLDS})")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X_raw, Y, G = load_material_gradient_dataset()
    n = len(G)
    print(f"Loaded {n} samples")

    Y_log = np.log1p(Y)
    G_log_abs = np.log10(np.abs(G) + 1.0)
    G_sign = np.sign(G)

    cfg = TrainConfig()  # defaults match the thesis
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_metrics, all_preds = [], np.zeros(n)
    fold_artifacts: list[dict] = []

    t0 = time.perf_counter()
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(np.arange(n))):
        ft0 = time.perf_counter()
        pred, artifacts = train_one_fold(
            X_raw, Y_log, G, G_log_abs, G_sign, train_idx, test_idx,
            seed=RANDOM_SEED + fold_i, cfg=cfg, device=device,
        )
        all_preds[test_idx] = pred
        fm = grad_metrics(pred, G[test_idx])
        fold_metrics.append(fm)
        fold_artifacts.append(artifacts)
        print(f"  Fold {fold_i + 1} ({time.perf_counter() - ft0:.0f}s): "
              f"Sign={fm['sign_acc']:.1f}%  Acc@20%={fm['acc20']:.1f}%  R2_log={fm['r2_log']:.4f}")

    elapsed = time.perf_counter() - t0
    mean = {k: round(float(np.mean([f[k] for f in fold_metrics])), 4 if "r2" in k else 1)
            for k in fold_metrics[0]}
    std = {k: round(float(np.std([f[k] for f in fold_metrics])), 4 if "r2" in k else 1)
           for k in fold_metrics[0]}

    print(f"\nTotal: {elapsed:.0f}s ({elapsed / 60:.1f}min)")
    print(f"  Sign:    {mean['sign_acc']:.1f} +/- {std['sign_acc']:.1f}%")
    print(f"  Acc@20%: {mean['acc20']:.1f} +/- {std['acc20']:.1f}%")
    print(f"  R2_log:  {mean['r2_log']:.4f} +/- {std['r2_log']:.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "proper_cv_dLdE_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "script": "01_proper_cv_dLdE.py",
                "seed": RANDOM_SEED,
                "n_folds": N_FOLDS,
                "elapsed_s": round(elapsed, 1),
                "config": cfg.__dict__,
            },
            "fold_metrics": fold_metrics,
            "mean": mean,
            "std": std,
            "n_samples": n,
        }, f, indent=2)
    print(f"\nResults: {out_path}")

    # ------------------------------------------------------------------
    # Publish the best fold's model + normalisation to the package
    # checkpoints directory so `GradientSurrogate.load_default()` works.
    # ------------------------------------------------------------------
    best_fold = max(range(N_FOLDS), key=lambda i: fold_metrics[i]["acc20"])
    print(f"\nBest fold for dL/dE checkpoint: fold {best_fold + 1} "
          f"(Acc@20% = {fold_metrics[best_fold]['acc20']:.1f}%)")
    package_ckpt = Path(__file__).resolve().parent.parent.parent / "polyfem_ml" / "checkpoints"
    package_ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(fold_artifacts[best_fold]["state_dict"], package_ckpt / "model_dLdE.pt")

    # The two NN models (dL/dE + dL/dh,dL/dtheta) share one JSON metadata file.
    # If the partner script (02_proper_cv_htheta.py) has already populated its
    # "h_theta" entry, preserve it; otherwise just write our part.
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
    meta["normalisation"]["dLdE"] = fold_artifacts[best_fold]["normalisation"]
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Published checkpoint -> {package_ckpt / 'model_dLdE.pt'}")
    print(f"Updated metadata     -> {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
