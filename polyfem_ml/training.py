"""Shared training utilities for the gradient surrogates.

Encapsulates the training-loop pattern used by both :file:`01_proper_cv_dLdE`
and :file:`02_proper_cv_htheta`:

- inner 90/10 train/val split for early stopping (never the test fold);
- per-fold normalization (X, Y_log, gradient log-magnitudes);
- BCE-with-logits sign + MSE log-magnitude losses with linear ramp warmup;
- AdamW + cosine annealing + gradient clipping;
- early stop checked every ``check_every`` epochs on inner-val Acc@20%.

The decode rule for predictions is the same in both:
    g_hat = sign(s_hat - 0.5) * (10 ** m_hat - 1)

with ``m_hat`` un-normalized by the training-fold log-magnitude statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class TrainConfig:
    """Hyperparameters for the proper-CV training loop. Defaults match the thesis."""
    hidden: int = 512
    n_layers: int = 6
    n_epochs: int = 400
    lr: float = 5e-4
    weight_decay: float = 1e-5
    batch_size: int = 512
    grad_clip: float = 1.0
    warmup_epochs: int = 30
    ramp_window: int = 40
    patience_checks: int = 40   # patience in units of "checks", not epochs
    check_every: int = 20       # validate every N epochs


def inner_split(outer_train_idx: np.ndarray, *, val_frac: float = 0.1, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Random 90/10 split of an outer training fold into inner train / val.

    The outer test fold is *never* passed in here — it is held back at the
    caller for unbiased evaluation.
    """
    n = len(outer_train_idx)
    if rng is None:
        rng = np.random.default_rng()
    perm = rng.permutation(n)
    n_val = max(1, int(val_frac * n))
    return outer_train_idx[perm[n_val:]], outer_train_idx[perm[:n_val]]


def decode_gradient(
    log_mag_pred: np.ndarray,
    sign_logit: np.ndarray,
    *,
    log_mag_mean: float,
    log_mag_std: float,
) -> np.ndarray:
    """Recombine sign + log-magnitude head outputs into a gradient prediction.

    Parameters
    ----------
    log_mag_pred
        Magnitude head output, *normalized* (the head sees standardized
        log-magnitudes during training).
    sign_logit
        Sign head pre-sigmoid logits.
    log_mag_mean, log_mag_std
        Training-fold mean / std of ``log10(|g| + 1)`` used to normalize the
        magnitude target.

    Returns
    -------
    Gradient prediction in the original linear scale.
    """
    mag = log_mag_pred * log_mag_std + log_mag_mean
    sign = np.where(_sigmoid(sign_logit) > 0.5, 1.0, -1.0)
    return sign * (10.0 ** mag - 1.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid for arbitrary input.
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def loss_ramp(epoch: int, *, warmup: int, window: int) -> float:
    """Compute the gradient-loss weight ramp for the given epoch.

    Returns 0 for epochs <= warmup, linearly increases to 1 over the next
    ``window`` epochs. Matches the thesis configuration:
        ramp(epoch) = clip((epoch - 30) / 40, 0, 1)
    """
    return min(1.0, max(0.0, (epoch - warmup) / window))


__all__ = [
    "TrainConfig",
    "inner_split",
    "decode_gradient",
    "loss_ramp",
]
