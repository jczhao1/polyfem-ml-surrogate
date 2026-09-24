#!/usr/bin/env python3
"""Stratified 80/10/10 split of the forward dataset by theta region.

Stratification uses 5 theta bins so the bifurcation zone is proportionally
represented in train, validation, and test. Test set is held out for the
final evaluation in 01_train_eval.py and is never seen during model
selection.

Output: ``results/split_indices.npz`` with arrays
``train_idx``, ``val_idx``, ``test_idx``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from polyfem_ml import RANDOM_SEED
from polyfem_ml.data import load_forward_dataset


RESULTS_DIR = Path(__file__).resolve().parent / "results"

REGION_BOUNDS = (80.0, 87.0, 93.0, 100.0)
REGION_NAMES = ("<80", "[80,87)", "[87,93)", "[93,100)", ">=100")


def theta_region(theta: np.ndarray) -> np.ndarray:
    """Return integer region labels in [0, 4] for each theta value."""
    return np.searchsorted(REGION_BOUNDS, theta, side="right")


def main() -> None:
    X, Y = load_forward_dataset()
    n = len(Y)
    print(f"Dataset: {n} samples")

    regions = theta_region(X[:, 0])
    print("\nRegion distribution:")
    for r, name in enumerate(REGION_NAMES):
        print(f"  {name:>10}: {(regions == r).sum()}")

    idx = np.arange(n)
    train_idx, temp_idx = train_test_split(
        idx, test_size=0.20, random_state=RANDOM_SEED, stratify=regions,
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=RANDOM_SEED, stratify=regions[temp_idx],
    )

    print(
        f"\nSplit sizes:"
        f"\n  Train: {len(train_idx)} ({len(train_idx) / n * 100:.1f}%)"
        f"\n  Val:   {len(val_idx)} ({len(val_idx) / n * 100:.1f}%)"
        f"\n  Test:  {len(test_idx)} ({len(test_idx) / n * 100:.1f}%)"
    )
    print("\nPer-region:")
    for r, name in enumerate(REGION_NAMES):
        ntr = (regions[train_idx] == r).sum()
        nva = (regions[val_idx] == r).sum()
        nte = (regions[test_idx] == r).sum()
        print(f"  {name:>10}: train={ntr:>5}  val={nva:>4}  test={nte:>4}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "split_indices.npz"
    np.savez(out_path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
