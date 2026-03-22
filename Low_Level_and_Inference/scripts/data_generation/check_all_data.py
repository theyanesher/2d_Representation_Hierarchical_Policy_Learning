"""
Check all HDF5 files in a directory for expected keys and shapes.

Usage:
    python scripts/data_generation/check_all_data.py data/rgb/41510/
    python scripts/data_generation/check_all_data.py data/rgb/41510/ --verbose
    python scripts/data_generation/check_all_data.py data/rgb/41510/ --strict
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


CAM_IDS = [0, 1, 2]

# Expected trailing shapes (after the T dimension)
EXPECTED_SHAPES = {
    # actions
    "action/hybrid":          None,       # checked against action/delta
    "action/delta":           None,       # checked against action/hybrid

    # rgb images: (T, 256, 256, 3)
    **{f"obs/cam{c}_image":    (256, 256, 3) for c in CAM_IDS},

    # pointmaps: (T, 3, 256, 256)
    **{f"obs/cam{c}_pointmap": (3, 256, 256) for c in CAM_IDS},

    # plucker: (T, 6, 256, 256)
    **{f"obs/cam{c}_plucker":  (6, 256, 256) for c in CAM_IDS},

    # depth: (T, 256, 256)
    **{f"obs/cam{c}_depth":    (256, 256)    for c in CAM_IDS},

    # intrinsics: (T, 3, 3)
    **{f"obs/cam{c}_intrinsic": (3, 3)       for c in CAM_IDS},

    # extrinsics: (T, 4, 4)
    **{f"obs/cam{c}_extrinsic": (4, 4)       for c in CAM_IDS},

    # state: (T, 10)
    "obs/state":              (10,),

    # gripper to world: (T, 4, 4)
    "obs/gripper_to_world":   (4, 4),
}


def check_file(h5_path: Path, verbose: bool = False) -> list[str]:
    """
    Check a single HDF5 file. Returns a list of error strings (empty = OK).
    """
    errors = []

    try:
        f = h5py.File(h5_path, "r")
    except Exception as e:
        return [f"Cannot open file: {e}"]

    with f:
        # -----------------------------------------------------------------------
        # 1. action/hybrid and action/delta must both exist and match shape
        # -----------------------------------------------------------------------
        has_hybrid = "action/hybrid" in f
        has_delta  = "action/delta"  in f

        if not has_hybrid:
            errors.append("Missing key: action/hybrid")
        if not has_delta:
            errors.append("Missing key: action/delta")

        if has_hybrid and has_delta:
            sh = f["action/hybrid"].shape
            sd = f["action/delta"].shape
            if sh != sd:
                errors.append(
                    f"action/hybrid shape {sh} != action/delta shape {sd}"
                )
            T = sh[0]
        elif has_hybrid:
            T = f["action/hybrid"].shape[0]
        elif has_delta:
            T = f["action/delta"].shape[0]
        else:
            # Can't do further T-consistency checks
            T = None

        # -----------------------------------------------------------------------
        # 2. Check all other keys
        # -----------------------------------------------------------------------
        for key, expected_suffix in EXPECTED_SHAPES.items():
            if key in ("action/hybrid", "action/delta"):
                continue  # already handled above

            if key not in f:
                errors.append(f"Missing key: {key}")
                continue

            shape = f[key].shape
            if len(shape) == 0:
                errors.append(f"{key}: scalar, expected at least 1-D")
                continue

            t_dim = shape[0]

            # T-consistency check
            if T is not None and t_dim != T:
                errors.append(
                    f"{key}: T={t_dim} but action T={T}"
                )

            # Trailing-shape check
            if expected_suffix is not None:
                actual_suffix = shape[1:]
                if actual_suffix != expected_suffix:
                    errors.append(
                        f"{key}: expected shape (T, {', '.join(str(s) for s in expected_suffix)}) "
                        f"but got {shape}"
                    )

        if verbose and not errors:
            print(f"  [OK] T={T}  {h5_path.name}")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate HDF5 dataset files.")
    parser.add_argument("data_dir", type=str, help="Directory containing .h5 files.")
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print OK lines in addition to errors."
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit with code 1 if any file has errors."
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"ERROR: {data_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    h5_paths = sorted(data_dir.glob("*.h5"))
    if not h5_paths:
        print(f"No .h5 files found in {data_dir}.", file=sys.stderr)
        sys.exit(1)

    print(f"Checking {len(h5_paths)} files in {data_dir} ...\n")

    n_ok = 0
    n_bad = 0

    for path in h5_paths:
        errors = check_file(path, verbose=args.verbose)
        if errors:
            n_bad += 1
            print(f"[FAIL] {path.name}")
            for e in errors:
                print(f"       - {e}")
        else:
            n_ok += 1
            if args.verbose:
                pass  # already printed inside check_file

    print(f"\nSummary: {n_ok} OK, {n_bad} FAILED out of {len(h5_paths)} files.")

    if args.strict and n_bad > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
