# Experiment 1: Forward Surrogate

Predicts the peak Von Mises stress on the impacting block from
``(theta, h, E)`` lattice design parameters.

## Headline result

XGBoost with 22 physics-informed features and ``log(1+y)`` target transform:

- **R² = 0.9684**
- **Acc@10% = 74.1%**
- **Acc@20% = 86.9%**

on the test set (n = 1,889, never seen during training) under an 80/10/10
stratified split. Cross-validated 5-fold mean is 73.4% ± 0.5%.

## Reproducing

```bash
uv run python 00_split_data.py    # Creates 80/10/10 stratified split
uv run python 01_train_eval.py    # Trains all baselines and the optimized XGBoost
```

Outputs:
- ``results/all_results.json`` — all model performance numbers.
- ``polyfem_ml/checkpoints/stress_xgboost.joblib`` — final XGBoost model
  (consumed by ``StressSurrogate.load_default()``).

## Files

- ``00_split_data.py`` — 80/10/10 stratified split by 5 θ-regions, seed 42.
- ``01_train_eval.py`` — Trains RF, XGBoost, GBR, MLP, SVR with raw and 22-feat configs.
