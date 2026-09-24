#!/usr/bin/env python3
"""Build the cached NumPy bundles used by the gradient surrogate experiments.

This script reads the raw PolyFEM simulation outputs and produces the compact
``X.npy`` / ``Y.npy`` / ``G*.npy`` bundles consumed by the training scripts.

The raw simulation outputs are NOT redistributed in this repo. They live in a
separate dataset repository:

    https://github.com/peteryu0131/polyfem_ML_dataset

To regenerate the npy bundles from raw simulation data:

    git clone https://github.com/peteryu0131/polyfem_ML_dataset.git raw_data
    uv run python experiments/02_gradient/00_prepare_data.py

For most uses (inference, retraining from the published bundles in ``data/``),
this script is not needed.

If raw data is missing, this script raises with a clear pointer rather than
silently substituting synthetic data.
"""

from __future__ import annotations

import io
import json
import os
import sys
import zipfile
from pathlib import Path

# Cross-platform: cap BLAS thread count on macOS to avoid a PyTorch/BLAS
# deadlock. Harmless no-op on Windows/Linux.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

# Repository root: experiments/02_gradient/00_prepare_data.py -> three up.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_ROOT = REPO_ROOT / "raw_data"
DATA_ROOT = REPO_ROOT / "data"

# Support two layouts for raw_data/:
#   (a) raw_data/Experiment02_* (legacy, when subdirs were placed directly)
#   (b) raw_data/polyfem_ML_dataset/Experiment02_* (when the dataset repo
#       was cloned as a whole into raw_data/)
def _resolve_raw(subdir: str) -> Path:
    direct = RAW_ROOT / subdir
    if direct.exists():
        return direct
    nested = RAW_ROOT / "polyfem_ML_dataset" / subdir
    if nested.exists():
        return nested
    return direct  # return the canonical path so the error message points at it


MATERIAL_RAW = _resolve_raw("Experiment02_material_training_data") / "training_data_E_diff_success_minimal"
H_THETA_RAW = _resolve_raw("Experiment02_h_theta_training_data")


def _raise_missing_raw_data() -> None:
    raise FileNotFoundError(
        "Raw PolyFEM simulation outputs were not found under "
        f"{RAW_ROOT}/. The raw data lives in a separate repository:\n\n"
        "    https://github.com/peteryu0131/polyfem_ML_dataset\n\n"
        "To regenerate the npy bundles from raw data, clone that repo into "
        f"{RAW_ROOT}/ and re-run this script:\n\n"
        f"    git clone https://github.com/peteryu0131/polyfem_ML_dataset.git {RAW_ROOT}\n"
        "    uv run python experiments/02_gradient/00_prepare_data.py\n\n"
        "For inference and for retraining from the bundles already in data/, "
        "this step is not required."
    )


def find_material_case_dirs() -> list[Path]:
    if not MATERIAL_RAW.exists():
        return []
    out = []
    for case_dir in sorted(MATERIAL_RAW.iterdir()):
        if not case_dir.is_dir():
            continue
        shard_dirs = [d for d in case_dir.iterdir() if d.is_dir()]
        if shard_dirs and (shard_dirs[0] / "metadata.json").exists():
            out.append(shard_dirs[0])
    return out


def load_material() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load (X, Y, G) for dL/dE from raw bundles.

    Raises ``FileNotFoundError`` if no cases are found — never falls back to
    synthetic data.
    """
    case_dirs = find_material_case_dirs()
    if not case_dirs:
        _raise_missing_raw_data()

    X_list, Y_list, G_list = [], [], []
    skipped = 0
    for cd in case_dirs:
        try:
            meta = json.loads((cd / "metadata.json").read_text(encoding="utf-8"))
            eg = np.load(cd / "E_gradient.npy")
        except (FileNotFoundError, json.JSONDecodeError):
            skipped += 1
            continue
        case = meta["case"]
        X_list.append([case["E"], case["h"], case["theta_deg"]])
        obj = meta.get("objective", {})
        loss = obj.get("value", obj.get("loss", meta.get("loss", 0.0)))
        Y_list.append(float(loss))
        G_list.append(float(eg[0]))
    if skipped:
        print(f"  (material: skipped {skipped} incomplete cases)")
    return (
        np.asarray(X_list, dtype=np.float64),
        np.asarray(Y_list, dtype=np.float64),
        np.asarray(G_list, dtype=np.float64),
    )


def load_geometric() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load (X, Y, Gh, Gtheta) for the joint dL/dh + dL/dtheta task."""
    zips = sorted(H_THETA_RAW.glob("part_*.zip"))
    if not zips:
        _raise_missing_raw_data()

    X_list, Y_list, Gh_list, Gtheta_list = [], [], [], []
    skipped = 0
    for zp in zips:
        with zipfile.ZipFile(zp) as zf:
            grad_files = [n for n in zf.namelist() if n.endswith("h_theta_gradient.npy")]
            for gf in grad_files:
                try:
                    meta = json.loads(zf.read(gf.replace("h_theta_gradient.npy", "metadata.json")))
                    grad = np.load(io.BytesIO(zf.read(gf)))
                except (KeyError, json.JSONDecodeError):
                    skipped += 1
                    continue
                case = meta["case"]
                X_list.append([case["E"], case["h"], case["theta_deg"]])
                obj = meta.get("objective", {})
                loss = obj.get("value", obj.get("loss", meta.get("loss", 0.0)))
                Y_list.append(float(loss))
                Gh_list.append(float(grad[0]))
                Gtheta_list.append(float(grad[1]))
    if skipped:
        print(f"  (geometric: skipped {skipped} incomplete cases)")
    return (
        np.asarray(X_list, dtype=np.float64),
        np.asarray(Y_list, dtype=np.float64),
        np.asarray(Gh_list, dtype=np.float64),
        np.asarray(Gtheta_list, dtype=np.float64),
    )


def main() -> int:
    print("Building gradient datasets from raw PolyFEM bundles…")
    print(f"  raw_data: {RAW_ROOT}")
    print(f"  data:     {DATA_ROOT}")

    if not RAW_ROOT.exists():
        try:
            _raise_missing_raw_data()
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2

    # --- Material (dL/dE) ---
    print("\n[1/2] dL/dE (material gradient)")
    X_m, Y_m, G_m = load_material()
    out_m = DATA_ROOT / "gradient_material"
    out_m.mkdir(parents=True, exist_ok=True)
    np.save(out_m / "X.npy", X_m)
    np.save(out_m / "Y.npy", Y_m)
    np.save(out_m / "G.npy", G_m)
    print(f"  -> wrote {out_m} (n={len(X_m)})")

    # --- Geometric (dL/dh, dL/dtheta) ---
    print("\n[2/2] dL/dh + dL/dtheta (geometric gradient)")
    X_g, Y_g, Gh, Gth = load_geometric()
    out_g = DATA_ROOT / "gradient_geometric"
    out_g.mkdir(parents=True, exist_ok=True)
    np.save(out_g / "X.npy", X_g)
    np.save(out_g / "Y.npy", Y_g)
    np.save(out_g / "Gh.npy", Gh)
    np.save(out_g / "Gtheta.npy", Gth)
    print(f"  -> wrote {out_g} (n={len(X_g)})")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
