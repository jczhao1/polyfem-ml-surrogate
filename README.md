# polyfem_ml

Machine learning surrogates for triangular mesh impact simulation in PolyFEM. Each FEM call takes minutes on CPU; each surrogate call takes sub-millisecond inference.

Two surrogates: a **forward** one (predicts the block peak Von Mises stress) and a **gradient** one (predicts the three adjoint sensitivities `dL/dE`, `dL/dh`, `dL/dtheta`).

## Structure

```
polyfem_ml/                importable package
├── inference.py           public API: StressSurrogate / GradientSurrogate
├── features.py            22 physics features, log(1+y) transform
├── models.py              XGBoost + multihead nets (sign + log-magnitude decomposition)
├── training.py            train config, inner split, gradient decode
├── metrics.py             Acc@K%, Sign%, R²
├── data.py                load forward / material / geometric npy bundles
└── checkpoints/           trained models (reproduce the thesis numbers)

experiments/
├── 01_forward/            00_split_data.py, 01_train_eval.py
└── 02_gradient/           00_prepare_data.py + 3 CV / baseline scripts

data/                      cached .npy bundles (symlinks into ../archived/polyfem_ML_dataset/)
tools/check_inference.py   smoke test
pyproject.toml + uv.lock   pinned deps (managed by uv)
```

## Install

Requirements: Python ≥ 3.12, [`uv`](https://github.com/astral-sh/uv), [`git-lfs`](https://git-lfs.com).

```bash
git lfs install                    # one-time, system-wide
uv sync                            # creates .venv, pulls LFS blobs (npy bundles + checkpoints)
uv run python tools/check_inference.py   # verify the install
```

## Run — inference

```python
from polyfem_ml import StressSurrogate, GradientSurrogate

sigma = StressSurrogate.load_default().predict(theta=87.3, h=0.05, E=18.0)
dE, dh, dtheta = GradientSurrogate.load_default().predict(theta=87.3, h=0.05, E=18.0)
```

`predict()` takes scalars or 1-D NumPy arrays, and returns a scalar for scalar input, otherwise an array.

## Run — retraining

The checkpoints already reproduce the thesis numbers. To retrain from the cached npy bundles:

```bash
uv run python experiments/01_forward/00_split_data.py         # ~seconds
uv run python experiments/01_forward/01_train_eval.py         # ~2 min
uv run python experiments/02_gradient/01_proper_cv_dLdE.py    # ~15 min
uv run python experiments/02_gradient/02_proper_cv_htheta.py  # ~15 min
```

Each script writes into `polyfem_ml/checkpoints/`, so later `.load_default()` calls pick up the retrained weights. Run the two gradient scripts **sequentially** (both write `gradient_surrogate.json`).

## Raw data

The raw PolyFEM outputs live in a companion dataset (a local copy is in `../archived/polyfem_ML_dataset/`). Only needed to re-run preprocessing from scratch:

```bash
uv run python experiments/02_gradient/00_prepare_data.py
```
