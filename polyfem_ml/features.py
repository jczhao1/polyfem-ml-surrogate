"""Physics-informed feature engineering.

Two feature sets are used in the thesis:

- :func:`features_forward_22` — 22-dim, used by the forward surrogate. Includes
  a 7-level one-hot over the discrete ``h`` grid.
- :func:`features_grad_15` — 15-dim, used by the gradient XGBoost baselines.
  Same physics-informed transforms, **without** the ``h`` one-hot, since the
  gradient datasets sample ``h`` on a different (5-level) grid.

All transforms are derived from dimensional analysis of cellular-solid
mechanics (Gibson & Ashby, 1997) plus explicit transition-distance features
around the buckling bifurcation at ``theta = 88 deg``.
"""

from __future__ import annotations

import numpy as np

# Discrete h levels in the forward dataset. Treated as categorical via one-hot.
H_LEVELS_FORWARD = (0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.07)


def _common_engineered(theta: np.ndarray, h: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Return the 12 engineered features common to both feature sets."""
    return np.column_stack([
        np.log(h),                        # log h
        np.log(E),                        # log E
        np.sin(theta * np.pi / 180.0),    # trigonometric encoding
        np.abs(theta - 88.0),             # transition distance (1st order)
        (theta - 88.0) ** 2,              # transition distance (2nd order)
        (theta - 88.0) ** 3,              # transition distance (3rd order, signed)
        h ** 2 * E,                       # bending stiffness proxy
        h ** 3 * E,                       # higher-order bending
        h / E,                            # compliance
        theta * h,                        # interaction
        theta * E,                        # interaction
        h * E,                            # interaction
    ])


def features_forward_22(X: np.ndarray) -> np.ndarray:
    """Return the 22-dim forward feature matrix for inputs ``(theta, h, E)``.

    Columns: 3 raw + 12 engineered + 7 one-hot over ``h``.
    """
    theta, h, E = X[:, 0], X[:, 1], X[:, 2]
    onehot = np.column_stack([np.isclose(h, lv).astype(np.float32) for lv in H_LEVELS_FORWARD])
    return np.column_stack([X, _common_engineered(theta, h, E), onehot])


def features_grad_15(X: np.ndarray) -> np.ndarray:
    """Return the 15-dim gradient feature matrix for inputs ``(theta, h, E)`` --- no h one-hot.

    The gradient datasets use a 5-level ``h`` grid that does not match the
    forward 7-level grid, so a forward-style one-hot would create cross-dataset
    leakage. Columns: 3 raw + 12 engineered.
    """
    theta, h, E = X[:, 0], X[:, 1], X[:, 2]
    return np.column_stack([X, _common_engineered(theta, h, E)])
