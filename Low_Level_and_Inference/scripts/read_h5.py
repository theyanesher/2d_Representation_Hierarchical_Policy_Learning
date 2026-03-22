"""
Script to read and inspect .h5 files in a given directory.
Usage:
    python scripts/read_h5.py <directory>
    python scripts/read_h5.py data/rgb_mino_data/41510
"""

import sys
import h5py
import numpy as np
from pathlib import Path


def summarize_file(path: Path):
    with h5py.File(path, "r") as f:
        print(f"\nFILE: {path.name}")

        # NaN check across all floating-point datasets
        nan_found = False
        def check_nan(name, obj):
            nonlocal nan_found
            if isinstance(obj, h5py.Dataset) and np.issubdtype(obj.dtype, np.floating):
                n = np.isnan(obj[()]).sum()
                if n > 0:
                    print(f"  [NaN] {name}: {n} NaN values out of {obj.size}")
                    nan_found = True
        f.visititems(check_nan)
        if not nan_found:
            print("  NaN check: clean")
        return nan_found

        # Heatmap min/max
        for key in sorted(f["obs"].keys()):
            if "heatmap" in key:
                arr = f[f"obs/{key}"][()].astype(np.float32)
                print(f"  obs/{key}: min={arr.min():.6f}, max={arr.max():.6f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/read_h5.py <directory_or_file>")
        sys.exit(1)

    target = Path(sys.argv[1])

    if target.is_dir():
        files = sorted(target.glob("*.h5"))
        if not files:
            print(f"No .h5 files found in {target}")
            sys.exit(1)
        nan_files = []
        for fpath in files:
            had_nan = summarize_file(fpath)
            if had_nan:
                nan_files.append(fpath.name)
        print("\n=== Summary ===")
        if nan_files:
            print(f"Files with NaN ({len(nan_files)}/{len(files)}):")
            for name in nan_files:
                print(f"  {name}")
        else:
            print(f"All {len(files)} files are NaN-free.")
    elif target.is_file():
        summarize_file(target)
    else:
        print(f"Path not found: {target}")
        sys.exit(1)


if __name__ == "__main__":
    main()
