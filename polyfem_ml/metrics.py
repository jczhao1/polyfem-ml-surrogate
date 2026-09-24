"""Evaluation metrics: Acc@K%, Sign agreement, R^2 in log space.

The Acc@K% metric is the primary deployment metric used throughout the thesis.
It directly answers the question "is this prediction close enough to replace
FEM?" rather than measuring average distance.
"""

from __future__ import annotations

import numpy as np


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Forward (scalar) regression metrics.

    Returns a dict with: ``r2``, ``mape``, ``acc10``, ``acc20``.
    """
    mask = np.abs(y_true) > 1e-6
    rel_err = np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    return {
        "r2": round(float(r2), 4),
        "mape": round(float(rel_err.mean() * 100), 1),
        "acc10": round(float((rel_err <= 0.10).mean() * 100), 1),
        "acc20": round(float((rel_err <= 0.20).mean() * 100), 1),
    }


def grad_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Gradient prediction metrics: sign agreement, log-space R^2, Acc@K%.

    Returns a dict with: ``sign_acc``, ``r2_log``, ``acc10``, ``acc20``,
    ``acc50``, ``mape``. Sign convention: positive sign of the *predicted*
    value vs. positive sign of the ground-truth value, with zero counted as
    negative on both sides for consistency.
    """
    sign_ag = ((pred > 0) == (gt > 0)).mean() * 100
    mask = np.abs(gt) > 1e-6
    rel_err = np.abs(pred[mask] - gt[mask]) / np.abs(gt[mask])

    mask_p = (pred != 0) & (gt != 0)
    if mask_p.sum() > 0:
        log_p = np.log10(np.abs(pred[mask_p]))
        log_g = np.log10(np.abs(gt[mask_p]))
        ss_res = ((log_p - log_g) ** 2).sum()
        ss_tot = ((log_g - log_g.mean()) ** 2).sum()
        r2_log = 1 - ss_res / (ss_tot + 1e-10)
    else:
        r2_log = 0.0

    return {
        "sign_acc": round(float(sign_ag), 1),
        "r2_log": round(float(r2_log), 4),
        "acc10": round(float((rel_err <= 0.10).mean() * 100), 1),
        "acc20": round(float((rel_err <= 0.20).mean() * 100), 1),
        "acc50": round(float((rel_err <= 0.50).mean() * 100), 1),
        "mape": round(float(rel_err.mean() * 100), 1),
    }
