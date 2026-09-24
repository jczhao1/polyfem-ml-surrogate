# Experiment 2: Gradient Surrogate

Predicts the three adjoint-gradient components ``dL/dE``, ``dL/dh``, ``dL/dtheta``
of a smooth-max Von Mises objective on the lattice (body 1).

## Headline results (proper 5-fold CV, mean ± std)

| Component | Method                | Sign            | Acc@20%          | Acc@10%          |
|-----------|-----------------------|-----------------|-------------------|------------------|
| dL/dE     | Multi-head NN         | 97.2 ± 0.3 %    | **68.7 ± 1.3 %** | 51.5 ± 1.9 %     |
| dL/dE     | XGBoost decomposition | 95.2 %          | 66.9 %            | 51.0 %           |
| dL/dh     | Multi-head NN         | 98.6 ± 0.3 %    | **74.5 ± 1.0 %** | 60.5 ± 2.6 %     |
| dL/dh     | XGBoost decomposition | 98.5 %          | 75.8 %            | 65.4 %           |
| dL/dtheta | Multi-head NN         | 94.8 ± 0.3 %    | 46.7 ± 1.7 %      | 30.7 ± 1.7 %     |
| dL/dtheta | XGBoost decomposition | 94.9 %          | **57.4 %**        | 37.2 %           |

## Reproducing

```bash
uv run python 00_prepare_data.py        # Builds X.npy / Y.npy / G.npy from raw
uv run python 01_proper_cv_dLdE.py      # 5-fold CV for dL/dE
uv run python 02_proper_cv_htheta.py    # 5-fold CV for dL/dh + dL/dtheta
uv run python 03_xgboost_baseline.py    # XGBoost decomposition baseline (all 3)
```

Each script sets ``OMP_NUM_THREADS=1`` internally (before importing numpy /
torch) to avoid a PyTorch / BLAS threading deadlock that can hit macOS. The
setting is harmless on Linux and Windows.

Each script writes a JSON to ``results/`` with per-fold metrics and means.

## Methodology

- **Sign + log-magnitude decomposition.** Each gradient ``g`` is split into
  ``sign(g) ∈ {-1, +1}`` and a stabilized log-magnitude ``log10(|g| + 1)``.
  Inference: ``g_hat = sign_hat · (10^logmag_hat - 1)``.
- **Architecture.** Shared backbone of 6 ``Linear+SiLU`` stages with hidden
  width 512; one Linear-SiLU-Linear head per output (3 for dL/dE, 5 for
  dL/dh + dL/dtheta).
- **Training.** AdamW lr=5e-4, weight decay 1e-5, batch 512, gradient
  clipping ‖·‖₂ ≤ 1, cosine annealing T_max = 400 epochs.
- **Evaluation protocol.** Proper 5-fold CV: each outer fold's test partition
  is never seen during training or model selection. Within the training
  partition, an inner 90/10 random split provides the early-stopping
  validation set.

## Files

- ``00_prepare_data.py`` — Build X.npy / Y.npy / G.npy from raw PolyFEM JSON.
- ``01_proper_cv_dLdE.py`` — Headline 5-fold CV run for dL/dE.
- ``02_proper_cv_htheta.py`` — Headline 5-fold CV run for dL/dh and dL/dtheta.
- ``03_xgboost_baseline.py`` — XGBoost sign+log-magnitude baseline (single split).
