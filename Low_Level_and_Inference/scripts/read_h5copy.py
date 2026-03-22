"""
Script to read and inspect .h5 files in a given directory.
Usage:
    python scripts/read_h5.py <directory>
    python scripts/read_h5copy.py data/rgb_mino_data/41510
"""

import sys
import h5py
import numpy as np
from pathlib import Path


def print_tree(f: h5py.File, verbose: bool = True):
    """Recursively print all keys, shapes, dtypes, and scalar values."""
    def visitor(name, obj):
        indent = "  " * name.count("/")
        if isinstance(obj, h5py.Dataset):
            ds = obj
            shape_str = f"shape={ds.shape}, dtype={ds.dtype}"
            # Print scalar or small 1-D values inline
            if ds.shape == ():
                val = ds[()]
                if isinstance(val, bytes):
                    val = val.decode("utf-8", errors="replace")
                # print(f"  {indent}{name}: {shape_str}  -> {val}")
            elif ds.ndim == 1 and ds.shape[0] <= 12 and verbose:
                val = ds[()]
                if ds.dtype == object:
                    val = [v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v for v in val]
                # print(f"  {indent}{name}: {shape_str}  -> {val}")
            else:
                print(f"  {indent}{name}: {shape_str}")
        else:
            print(f"  {indent}{name}/")

    f.visititems(visitor)


def summarize_file(path: Path, verbose: bool = True):
    # print("=" * 70)
    # print(f"FILE: {path.name}")
    # print("=" * 70)

    with h5py.File(path, "r") as f:
        # Top-level keys
        print(f"\nTop-level keys: {list(f.keys())}\n")

        print("--- Full dataset tree ---")
        print_tree(f, verbose=verbose)

        # ---- Quick summary of key data ----
        print("\n--- Key values summary ---")

        # Task info
        if "sim_states/task_config" in f:
            tc = f["sim_states/task_config"]
            for field in ["name", "lang", "type"]:
                # import pdb; pdb.set_trace()
                if field in tc:
                    val = tc[field][()]
                    if isinstance(val, bytes):
                        val = val.decode()
                    # print(f"  task_config/{field}: {val}")

        # Trajectory label
        if "sim_states/label" in f:
            lbl = f["sim_states/label"]
            if "good_traj" in lbl:
                print(f"  good_traj: {lbl['good_traj'][()]}")
            if "failure reason" in lbl:
                reason = lbl["failure reason"][()]
                if isinstance(reason, bytes):
                    reason = reason.decode()
                # print(f"  failure reason: {reason!r}")

        # Observation shapes
        if "obs" in f:
            # print(f"\n  Observations ({len(f['obs'].keys())} keys):")
            for k in sorted(f["obs"].keys()):
                import pdb; pdb.set_trace()
                ds = f[f"obs/{k}"]
                # print(f"    obs/{k}: shape={ds.shape}, dtype={ds.dtype}")

        # Action shapes
        if "action" in f:
            # print(f"\n  Actions ({len(f['action'].keys())} keys):")
            for k in f["action"].keys():
                ds = f[f"action/{k}"]
                # print(f"    action/{k}: shape={ds.shape}, dtype={ds.dtype}")
                # Print min/max range
                arr = ds[()]
                # print(f"      min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")

        # NaN check across all numeric datasets
        print("\n--- NaN check ---")
        nan_found = False
        def check_nan(name, obj):
            nonlocal nan_found
            if isinstance(obj, h5py.Dataset) and np.issubdtype(obj.dtype, np.floating):
                arr = obj[()]
                n = np.isnan(arr).sum()
                if n > 0:
                    print(f"  [NaN] {name}: {n} NaN values out of {arr.size}")
                    nan_found = True
        f.visititems(check_nan)
        if not nan_found:
            print("  No NaN values found.")

        # Camera intrinsics/extrinsics
        if "_physical" in f:
            # print(f"\n  Physical camera params:")
            for k in f["_physical"].keys():
                ds = f[f"_physical/{k}"]
                # print(f"    {k}:\n{np.array2string(ds[()], precision=4, suppress_small=True)}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/read_h5.py <directory_or_file>")
        sys.exit(1)

    target = Path(sys.argv[1])

    # Determine files to process
    if target.is_dir():
        files = sorted(target.glob("*.h5"))
        if not files:
            print(f"No .h5 files found in {target}")
            sys.exit(1)
        print(f"Found {len(files)} .h5 files in {target}\n")
        # By default summarize all; pass --all to print tree for each
        verbose = "--verbose" in sys.argv
        for i, fpath in enumerate(files):
            # import pdb; pdb.set_trace()
            summarize_file(fpath, verbose=verbose)
            if i == 0 and not verbose:
                print("\n(Pass --verbose to show full tree for every file. Showing summary for remaining files.)\n")
    elif target.is_file():
        summarize_file(target, verbose=True)
    else:
        print(f"Path not found: {target}")
        sys.exit(1)


if __name__ == "__main__":
    main()
