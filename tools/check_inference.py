#!/usr/bin/env python3
"""End-to-end smoke test: load the published checkpoints and predict on a few
representative designs, then assert the outputs are finite and in a sane range.

Run from the repo root::

    uv run python tools/check_inference.py
"""
from __future__ import annotations

import os

# Cross-platform: cap BLAS thread count before importing numpy/torch to avoid
# a PyTorch/BLAS deadlock on macOS. Harmless no-op on Windows/Linux.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from polyfem_ml import StressSurrogate, GradientSurrogate


def main() -> None:
    print("Loading checkpoints…")
    stress = StressSurrogate.load_default()
    grad = GradientSurrogate.load_default()
    print("  OK\n")

    # Three representative design points:
    # 1. Smooth pre-bifurcation (forward Acc@10% ≈ 99% region)
    # 2. Buckling transition (forward Acc@10% ≈ 60% region)
    # 3. Post-bifurcation (forward Acc@10% ≈ 85% region)
    designs = [
        ("smooth (θ=70°)",      dict(theta=70.0,  h=0.05, E=20.0)),
        ("bifurcation (θ=88°)", dict(theta=88.0,  h=0.05, E=20.0)),
        ("post-buckling (θ=100°)", dict(theta=100.0, h=0.05, E=20.0)),
    ]

    print(f"{'Design':<28} {'sigma (Pa)':>14} {'dL/dE':>14} {'dL/dh':>14} {'dL/dθ':>14}")
    print("-" * 90)
    for label, inputs in designs:
        sigma = stress.predict(**inputs)
        dE, dh, dtheta = grad.predict(**inputs)
        print(f"{label:<28} {sigma:>14.3e} {dE:>14.3e} {dh:>14.3e} {dtheta:>14.3e}")

        # Sanity checks: stress must be a finite positive Pa value in a
        # plausible range; gradients must be finite. If a checkpoint were
        # corrupted or a normalisation stat were missing, one of these fails.
        assert np.isfinite(sigma) and sigma > 0.0, f"bad sigma at {label}: {sigma}"
        assert 1e3 < sigma < 1e9, f"sigma out of range at {label}: {sigma}"
        assert all(np.isfinite(v) for v in (dE, dh, dtheta)), (
            f"non-finite gradient at {label}: {(dE, dh, dtheta)}"
        )

    # Vectorised call (batched)
    print("\nBatched call (5 samples at once):")
    thetas = np.array([65.0, 80.0, 88.0, 95.0, 105.0])
    sigmas = stress.predict(theta=thetas, h=0.04, E=15.0)
    print("  thetas =", thetas)
    print("  sigmas =", np.round(sigmas, 2))
    assert sigmas.shape == thetas.shape, f"batched shape mismatch: {sigmas.shape}"
    assert np.all(np.isfinite(sigmas)), "batched call produced non-finite values"

    print("\nInference smoke test passed.")


if __name__ == "__main__":
    main()
