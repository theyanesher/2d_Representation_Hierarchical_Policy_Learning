#!/usr/bin/env python3
import h5py
import tqdm

def copy_top_level(src, dst):
    """
    Copy all keys from src to dst, except the 'data' group.
    """
    for key in src:
        if key != 'data':
            src.copy(key, dst, name=key)


def copy_demos(src_data, dst_data, demo_keys):
    """
    Copy specified demos from src_data to dst_data, preserving dataset contents.
    """
    # Copy attributes of the data group
    for a_key, a_val in src_data.attrs.items():
        dst_data.attrs[a_key] = a_val

    for demo_key in tqdm.tqdm(demo_keys, desc=f"Copying {len(demo_keys)} demos"):
        src_data.copy(demo_key, dst_data, name=demo_key)


def split_file(src_path: str, dst1_path: str, dst2_path: str,
               first_n: int = 1000, next_n: int = 150):
    """
    Split the HDF5 file at src_path into two files:
      - dst1_path will contain the first `first_n` demos
      - dst2_path will contain the next `next_n` demos
    Assumes demos are named with prefix 'demo' and sorted lexicographically.
    """
    with h5py.File(src_path, 'r') as src:
        src_data = src.get('data')
        if src_data is None:
            raise RuntimeError("No 'data' group found in source file")

        # Collect and sort demo keys
        demo_keys = [f'demo_{i}' for i in range(len(src_data))]

        part1 = demo_keys[:first_n]
        part2 = demo_keys[first_n:first_n + next_n]

        # Write first part
        with h5py.File(dst1_path, 'w') as dst1:
            copy_top_level(src, dst1)
            dst1_data = dst1.create_group('data')
            copy_demos(src_data, dst1_data, part1)

        # Write second part
        with h5py.File(dst2_path, 'w') as dst2:
            copy_top_level(src, dst2)
            dst2_data = dst2.create_group('data')
            copy_demos(src_data, dst2_data, part2)

    print(f"Wrote {len(part1)} demos to {dst1_path}")
    print(f"Wrote {len(part2)} demos to {dst2_path}")


if __name__ == "__main__":
    # Adjust these paths as needed
    task = 'nut_assembly_d0'
    base = f"data/robomimic/datasets/{task}/for_oracle"
    # src_hdf5 = f"{base}/{task}_1150_abs.hdf5"
    src_hdf5 = f"{base}/demo.hdf5"
    dst1_hdf5 = f"{base}/{task}_1000_abs.hdf5"
    dst2_hdf5 = f"{base}/{task}_150_abs.hdf5"

    split_file(src_hdf5, dst1_hdf5, dst2_hdf5, first_n=1000, next_n=150)
