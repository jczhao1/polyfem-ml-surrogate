# Data

The training data bundles used by the surrogates live in this directory.
They are stored via **Git LFS**, so a normal `git clone` pulls them
automatically (after `git lfs install`).

## Layout

```
data/
├── forward/                    n = 18,887 forward simulations
│   ├── X.npy                   (n, 3) float64 — inputs (theta_deg, h, E_MPa)
│   └── Y.npy                   (n,)   float64 — peak Von Mises stress on body 2 (Pa)
├── gradient_material/          n = 8,606 dL/dE samples
│   ├── X.npy                   (n, 3)
│   ├── Y.npy                   (n,)   — smooth-max Von Mises objective on body 1
│   └── G.npy                   (n,)   — adjoint gradient dL/dE
└── gradient_geometric/         n = 9,653 dL/dh and dL/dtheta samples
    ├── X.npy
    ├── Y.npy
    ├── Gh.npy                  (n,)   — adjoint gradient dL/dh
    └── Gtheta.npy              (n,)   — adjoint gradient dL/dtheta
```

Total size: ~1.4 MB.

## Where the data comes from

These npy bundles are the output of preprocessing applied to the raw PolyFEM
simulation logs. The raw logs (per-simulation JSON / npy files, ~50 GB total
across the three experiments) are **not** redistributed in this repo. They
live in a separate dataset repository maintained by the dataset authors:

  https://github.com/peteryu0131/polyfem_ML_dataset

For inference and for retraining from the bundles above, the raw logs are
not needed.

## Regenerating from raw simulation (optional)

If you want to re-run preprocessing from the raw PolyFEM output:

```bash
git clone https://github.com/peteryu0131/polyfem_ML_dataset.git raw_data
uv run python experiments/02_gradient/00_prepare_data.py
```

The preprocessing script raises a clear error if `raw_data/` is missing
rather than silently substituting synthetic data.
