"""Dataset loading.

The thesis numbers depend on three datasets that are too large for git
(roughly 1.5 GB total). They live under ``data/`` as cached NumPy arrays
prepared once from the raw PolyFEM JSON output.

The expected layout is::

    data/
    ├── forward/                n = 18,887
    │   ├── X.npy              (n, 3) inputs (theta, h, E)
    │   └── Y.npy              (n,)   peak Von Mises stress on body 2
    ├── gradient_material/      n = 8,606  (dL/dE)
    │   ├── X.npy
    │   ├── Y.npy
    │   └── G.npy              dL/dE
    └── gradient_geometric/     n = 9,653  (dL/dh, dL/dtheta)
        ├── X.npy
        ├── Y.npy
        ├── Gh.npy             dL/dh
        └── Gtheta.npy         dL/dtheta

To produce these from the raw PolyFEM training bundles, run::

    cd experiments/02_gradient
    uv run python 00_prepare_data.py

Unlike the original development scripts, this loader **raises** when data is
missing instead of silently substituting synthetic data. Set the
``POLYFEM_ML_DATA`` environment variable to override the default location.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Project root: <project>/surrogate/data.py -> <project>/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Resolve the data directory in this priority order:
#   1. POLYFEM_ML_DATA env var (explicit override)
#   2. <repo>/data/                          (canonical location)
DATA_DIR = Path(os.environ.get("POLYFEM_ML_DATA", _PROJECT_ROOT / "data"))


def _load_or_raise(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Required data file not found: {path}\n"
            f"Run experiments/02_gradient/00_prepare_data.py to build the cache, "
            f"or see data/README.md for download instructions, "
            f"or set POLYFEM_ML_DATA to point at an existing data directory."
        )
    return np.load(path)


def load_forward_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X, Y)`` for the forward surrogate.

    Shape: ``X (18887, 3)`` (theta, h, E); ``Y (18887,)`` peak Von Mises
    stress on body 2.
    """
    base = DATA_DIR / "forward"
    return _load_or_raise(base / "X.npy"), _load_or_raise(base / "Y.npy")


def load_material_gradient_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(X, Y, G)`` for the dL/dE gradient surrogate.

    Shape: ``X (8606, 3)``, ``Y (8606,)`` smooth-max objective, ``G (8606,)``
    adjoint gradient ``dL/dE``.
    """
    base = DATA_DIR / "gradient_material"
    return (
        _load_or_raise(base / "X.npy"),
        _load_or_raise(base / "Y.npy"),
        _load_or_raise(base / "G.npy"),
    )


def load_geometric_gradient_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(X, Y, Gh, Gtheta)`` for the dL/dh + dL/dtheta surrogate.

    Shape: ``X (9653, 3)``, ``Y (9653,)``, ``Gh (9653,)``, ``Gtheta (9653,)``.
    """
    base = DATA_DIR / "gradient_geometric"
    return (
        _load_or_raise(base / "X.npy"),
        _load_or_raise(base / "Y.npy"),
        _load_or_raise(base / "Gh.npy"),
        _load_or_raise(base / "Gtheta.npy"),
    )


__all__ = [
    "DATA_DIR",
    "load_forward_dataset",
    "load_material_gradient_dataset",
    "load_geometric_gradient_dataset",
]
