"""Machine-learning surrogates for PolyFEM triangular-mesh impact simulation.

Public API (stable interface for ``polyfem-python`` and downstream users)::

    from polyfem_ml import StressSurrogate, GradientSurrogate

    stress = StressSurrogate.load_default()
    sigma_peak = stress.predict(theta=87.3, h=0.05, E=18.0)

    grad = GradientSurrogate.load_default()
    dE, dh, dtheta = grad.predict(theta=87.3, h=0.05, E=18.0)

The training scripts under ``experiments/`` reproduce the headline numbers
from a clean checkout; you don't need them at inference time.

Submodules
----------
- :mod:`polyfem_ml.inference` — public ``StressSurrogate`` / ``GradientSurrogate`` classes.
- :mod:`polyfem_ml.data` — dataset loading (forward / material / geometric npy bundles).
- :mod:`polyfem_ml.features` — physics-informed feature construction (22-dim forward, 15-dim gradient).
- :mod:`polyfem_ml.metrics` — Acc@K%, Sign agreement, R^2 in log space.
- :mod:`polyfem_ml.models` — ``MultiHead3`` and ``MultiHead5`` torch modules.
- :mod:`polyfem_ml.training` — shared training utilities used by the experiment scripts.

Constants
---------
``RANDOM_SEED``
    The fixed seed used throughout the thesis (42).
``N_FOLDS``
    Default number of outer cross-validation folds (5).
"""

from polyfem_ml.metrics import compute_metrics, grad_metrics
from polyfem_ml.features import features_forward_22, features_grad_15
from polyfem_ml.models import MultiHead3, MultiHead5
from polyfem_ml.training import TrainConfig, inner_split, decode_gradient, loss_ramp
from polyfem_ml.inference import StressSurrogate, GradientSurrogate

RANDOM_SEED = 42
N_FOLDS = 5

__all__ = [
    # Public inference API (preferred entry point)
    "StressSurrogate",
    "GradientSurrogate",
    # Lower-level building blocks (useful for retraining / extension)
    "compute_metrics",
    "grad_metrics",
    "features_forward_22",
    "features_grad_15",
    "MultiHead3",
    "MultiHead5",
    "TrainConfig",
    "inner_split",
    "decode_gradient",
    "loss_ramp",
    "RANDOM_SEED",
    "N_FOLDS",
]
