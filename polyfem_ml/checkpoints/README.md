# Trained checkpoints

This directory holds the trained surrogates consumed by
`StressSurrogate.load_default()` and `GradientSurrogate.load_default()`.

The checkpoints are tracked via **Git LFS** (~76 MB total), so a normal
`git clone` pulls them automatically (after `git lfs install`).

## Layout

```
polyfem_ml/checkpoints/
├── stress_xgboost.joblib       # forward surrogate: dict with "model" + "scaler"
├── model_dLdE.pt               # gradient surrogate (3-head NN), state_dict
├── model_h_theta.pt            # gradient surrogate (5-head NN), state_dict
└── gradient_surrogate.json     # arch hyperparams + per-branch normalisation stats
```

`gradient_surrogate.json` has the shape:

```json
{
  "hidden": 512,
  "n_layers": 6,
  "normalisation": {
    "dLdE":   {"X_mean": [...], "X_std": [...], "Y_mean": ..., "Y_std": ..., "G_mean": ..., "G_std": ...},
    "h_theta":{"X_mean": [...], "X_std": [...], "Y_mean": ..., "Y_std": ..., "G_mean": ..., "G_std": ...}
  }
}
```

The two gradient training scripts emit the per-branch normalisation stats
alongside their model weights and merge them into this file.

## Repopulating after a fresh retrain

```bash
# Forward
uv run python experiments/01_forward/00_split_data.py
uv run python experiments/01_forward/01_train_eval.py
# (writes stress_xgboost.joblib directly into polyfem_ml/checkpoints/)

# Gradient
uv run python experiments/02_gradient/01_proper_cv_dLdE.py
uv run python experiments/02_gradient/02_proper_cv_htheta.py
# (each writes its model_*.pt and merges its stats into gradient_surrogate.json)
```
