"""Public inference API for the trained surrogates.

This module is the **stable interface** that downstream consumers (notably
``polyfem-python``) should depend on. The training scripts under
``experiments/`` re-train the models, but you don't need any of that to
run inference — just::

    from polyfem_ml import StressSurrogate, GradientSurrogate

    stress = StressSurrogate.load_default()
    sigma_peak = stress.predict(theta=87.3, h=0.05, E=18.0)

    grad = GradientSurrogate.load_default()
    dE, dh, dtheta = grad.predict(theta=87.3, h=0.05, E=18.0)

Both classes accept either scalar inputs (returning a scalar / 3-tuple) or
1-D numpy arrays of shape ``(n,)`` (returning an array / 3-array).

The pickled artifacts these classes load live in
``polyfem_ml/checkpoints/`` — see ``polyfem_ml/checkpoints/README.md`` for
provenance and retraining instructions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

import joblib
import numpy as np

from polyfem_ml.features import features_forward_22

ArrayLike = Union[float, Sequence[float], np.ndarray]

CHECKPOINTS_DIR = Path(__file__).resolve().parent / "checkpoints"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _broadcast_inputs(theta: ArrayLike, h: ArrayLike, E: ArrayLike) -> np.ndarray:
    """Stack ``(theta, h, E)`` into an ``(n, 3)`` float array.

    Each argument may be a Python scalar, a list/tuple, or a 1-D numpy
    array; they are broadcast to a common length so that single-point
    calls and batch calls work the same way.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=np.float64))
    h     = np.atleast_1d(np.asarray(h, dtype=np.float64))
    E     = np.atleast_1d(np.asarray(E, dtype=np.float64))
    theta, h, E = np.broadcast_arrays(theta, h, E)
    return np.column_stack([theta, h, E])


# ---------------------------------------------------------------------------
# Forward surrogate
# ---------------------------------------------------------------------------

class StressSurrogate:
    """Predict the peak Von Mises stress on the impacting block.

    The trained pipeline is:

    1. compute 22 physics-informed features from ``(theta, h, E)``;
    2. apply the training-time ``StandardScaler``;
    3. predict ``z = log(1 + sigma_peak)`` with XGBoost;
    4. invert with ``sigma_peak = max(e^z - 1, 0)``.

    Inference is ``< 1 ms`` per sample on CPU.
    """

    def __init__(self, *, model, scaler):
        self.model = model
        self.scaler = scaler

    # ------------------------------------------------------------------
    @classmethod
    def load_default(cls, checkpoint_dir: Path | None = None) -> "StressSurrogate":
        """Load the model + scaler shipped with the package."""
        d = Path(checkpoint_dir) if checkpoint_dir else CHECKPOINTS_DIR
        path = d / "stress_xgboost.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"Trained stress surrogate not found at {path}. "
                f"Run ``experiments/01_forward/01_train_eval.py`` first, "
                f"then copy ``results/model_xgboost.joblib`` here."
            )
        bundle = joblib.load(path)
        return cls(model=bundle["model"], scaler=bundle["scaler"])

    # ------------------------------------------------------------------
    def predict(self, theta: ArrayLike, h: ArrayLike, E: ArrayLike) -> np.ndarray:
        """Predict peak Von Mises stress (Pa) for one or more designs."""
        X_raw = _broadcast_inputs(theta, h, E)
        X_feat = features_forward_22(X_raw)
        X_scaled = self.scaler.transform(X_feat)
        z = self.model.predict(X_scaled)
        sigma = np.maximum(np.expm1(z), 0.0)
        return float(sigma.item()) if sigma.size == 1 else sigma


# ---------------------------------------------------------------------------
# Gradient surrogate
# ---------------------------------------------------------------------------

@dataclass
class _Normalizer:
    """Per-fold input/output statistics computed at training time."""
    X_mean: np.ndarray
    X_std: np.ndarray
    Y_mean: float
    Y_std: float
    G_mean: float
    G_std: float
    # Only used by the 5-head h_theta branch. Stays at NaN for the dLdE branch.
    Gtheta_mean: float = float("nan")
    Gtheta_std: float = float("nan")


class GradientSurrogate:
    """Predict ``(dL/dE, dL/dh, dL/dtheta)`` of the smooth-max Von Mises objective.

    This wraps the multi-head NN trained under proper 5-fold CV (see
    ``experiments/02_gradient/01_proper_cv_dLdE.py`` and
    ``experiments/02_gradient/02_proper_cv_htheta.py``).

    The decomposition decode is::

        sign_hat = sigmoid(sign_logit) > 0.5
        m_hat    = magnitude_logit * G_std + G_mean
        g_hat    = (+1 if sign_hat else -1) * (10**m_hat - 1)

    Inference is ``< 0.1 ms`` per sample on CPU; outputs all three gradient
    components in one call.
    """

    def __init__(
        self,
        *,
        model_dLdE,
        model_h_theta,
        norm_dLdE: _Normalizer,
        norm_h_theta: _Normalizer,
    ):
        self.model_dLdE = model_dLdE
        self.model_h_theta = model_h_theta
        self.norm_dLdE = norm_dLdE
        self.norm_h_theta = norm_h_theta

    # ------------------------------------------------------------------
    @classmethod
    def load_default(cls, checkpoint_dir: Path | None = None) -> "GradientSurrogate":
        """Load both trained sub-models and their normalisation statistics."""
        import torch  # local: keep torch import out of module import path

        d = Path(checkpoint_dir) if checkpoint_dir else CHECKPOINTS_DIR
        meta_path = d / "gradient_surrogate.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Trained gradient surrogate metadata not found at {meta_path}. "
                f"See polyfem_ml/checkpoints/README.md for how to populate this."
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        from polyfem_ml.models import MultiHead3, MultiHead5

        m_dLdE = MultiHead3(in_dim=3, hidden=meta["hidden"], n_layers=meta["n_layers"])
        m_dLdE.load_state_dict(torch.load(d / "model_dLdE.pt", map_location="cpu"))
        m_dLdE.eval()

        m_ht = MultiHead5(in_dim=3, hidden=meta["hidden"], n_layers=meta["n_layers"])
        m_ht.load_state_dict(torch.load(d / "model_h_theta.pt", map_location="cpu"))
        m_ht.eval()

        def _norm(key):
            n = meta["normalisation"][key]
            return _Normalizer(
                X_mean=np.array(n["X_mean"]),
                X_std=np.array(n["X_std"]),
                Y_mean=n["Y_mean"], Y_std=n["Y_std"],
                G_mean=n["G_mean"], G_std=n["G_std"],
                Gtheta_mean=n.get("Gtheta_mean", float("nan")),
                Gtheta_std=n.get("Gtheta_std", float("nan")),
            )

        return cls(
            model_dLdE=m_dLdE,
            model_h_theta=m_ht,
            norm_dLdE=_norm("dLdE"),
            norm_h_theta=_norm("h_theta"),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _decode(mag_logit: np.ndarray, sign_logit: np.ndarray,
                log_mean: float, log_std: float) -> np.ndarray:
        """Recombine sign-classifier + log-magnitude regressor into a gradient."""
        m = mag_logit * log_std + log_mean
        # Threshold the sign-head logit directly to avoid sigmoid overflow on
        # very confident predictions; the cut-point ``logit > 0`` is the same
        # as ``sigmoid(logit) > 0.5``.
        sign = np.where(sign_logit > 0.0, 1.0, -1.0)
        return sign * (10.0**m - 1.0)

    # ------------------------------------------------------------------
    def predict(self, theta: ArrayLike, h: ArrayLike, E: ArrayLike):
        """Return ``(dL/dE, dL/dh, dL/dtheta)`` for one or more designs."""
        import torch

        X_raw = _broadcast_inputs(theta, h, E)

        # --- dL/dE branch ---------------------------------------------------
        n = self.norm_dLdE
        X_E = (X_raw - n.X_mean) / n.X_std
        with torch.no_grad():
            _, mag_E, sign_E = self.model_dLdE(torch.tensor(X_E, dtype=torch.float32))
        dE = self._decode(mag_E.numpy(), sign_E.numpy(), n.G_mean, n.G_std)

        # --- dL/dh + dL/dtheta branch ---------------------------------------
        n = self.norm_h_theta
        X_ht = (X_raw - n.X_mean) / n.X_std
        with torch.no_grad():
            _, mag_h, sign_h, mag_th, sign_th = self.model_h_theta(
                torch.tensor(X_ht, dtype=torch.float32)
            )
        # h uses the primary G_mean/G_std; theta uses its own pair.
        dh = self._decode(mag_h.numpy(), sign_h.numpy(), n.G_mean, n.G_std)
        dtheta = self._decode(mag_th.numpy(), sign_th.numpy(),
                              n.Gtheta_mean, n.Gtheta_std)

        if dE.size == 1:
            return float(dE.item()), float(dh.item()), float(dtheta.item())
        return dE, dh, dtheta


__all__ = ["StressSurrogate", "GradientSurrogate"]
