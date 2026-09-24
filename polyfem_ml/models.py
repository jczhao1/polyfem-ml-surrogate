"""Multi-head neural network architectures for the gradient surrogate.

Two variants:

- :class:`MultiHead3` — three heads (value, magnitude, sign), used for the
  single-component dL/dE task.
- :class:`MultiHead5` — five heads (value, magnitude_h, sign_h,
  magnitude_theta, sign_theta), used for the joint dL/dh + dL/dtheta task.

Both share the same backbone shape (n_layers Linear+SiLU stages of width
`hidden`). Each head is a Linear-SiLU-Linear projection of width `hidden`
to a single scalar.

Default config matches the headline thesis numbers:
    hidden = 512, n_layers = 6.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _make_backbone(in_dim: int, hidden: int, n_layers: int) -> nn.Sequential:
    """Stack of Linear+SiLU layers; returns a feature extractor of width ``hidden``.

    Note: ``n_layers`` counts the *Linear* modules, including the input
    projection. So ``n_layers=6`` produces 6 Linear layers in total.
    """
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.SiLU()]
    for _ in range(n_layers - 2):
        layers += [nn.Linear(hidden, hidden), nn.SiLU()]
    return nn.Sequential(*layers)


def _head(hidden: int) -> nn.Sequential:
    """Linear-SiLU-Linear head projecting backbone features to a scalar."""
    return nn.Sequential(
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, 1),
    )


class MultiHead3(nn.Module):
    """Three-head model for the dL/dE gradient surrogate.

    Heads: value (log1p(L)), magnitude (log10(|g| + 1)), sign logit (P(g > 0)).
    """

    def __init__(self, in_dim: int = 3, hidden: int = 512, n_layers: int = 6):
        super().__init__()
        self.backbone = _make_backbone(in_dim, hidden, n_layers)
        self.value_head = _head(hidden)
        self.grad_mag_head = _head(hidden)
        self.grad_sign_head = _head(hidden)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return (
            self.value_head(h).squeeze(-1),
            self.grad_mag_head(h).squeeze(-1),
            self.grad_sign_head(h).squeeze(-1),
        )


class MultiHead5(nn.Module):
    """Five-head model for the joint dL/dh + dL/dtheta surrogate.

    Heads: value, magnitude_h, sign_h logit, magnitude_theta, sign_theta logit.
    """

    def __init__(self, in_dim: int = 3, hidden: int = 512, n_layers: int = 6):
        super().__init__()
        self.backbone = _make_backbone(in_dim, hidden, n_layers)
        self.value_head = _head(hidden)
        self.mag_h_head = _head(hidden)
        self.sign_h_head = _head(hidden)
        self.mag_theta_head = _head(hidden)
        self.sign_theta_head = _head(hidden)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return (
            self.value_head(h).squeeze(-1),
            self.mag_h_head(h).squeeze(-1),
            self.sign_h_head(h).squeeze(-1),
            self.mag_theta_head(h).squeeze(-1),
            self.sign_theta_head(h).squeeze(-1),
        )


__all__ = ["MultiHead3", "MultiHead5"]
