import h5py
import numpy as np
import tqdm

def copy_dataset(src_dataset, dst_group, key, data=None):
    """
    Copy a single dataset (with compression/chunking) into dst_group[key].
    If data is provided, use that instead of src_dataset[()].
    """
    if data is None:
        data = src_dataset[()]

    creation_args = {
        'dtype':          src_dataset.dtype,
        'compression':    src_dataset.compression,
        'compression_opts': src_dataset.compression_opts,
        'shuffle':        src_dataset.shuffle,
        'fletcher32':     src_dataset.fletcher32,
        'chunks':         src_dataset.chunks,
        'maxshape':       src_dataset.maxshape,
    }
    filtered = {k: v for k, v in creation_args.items() if v is not None}
    dst_group.create_dataset(key, data=data, **filtered)

def process_file(src_path: str, dst_path: str):
    with h5py.File(src_path, 'r') as src, h5py.File(dst_path, 'w') as dst:
        # 1) copy all top‐level keys except 'data'
        for key in src:
            if key != 'data':
                src.copy(key, dst, name=key)

        # 2) recreate & copy 'data' group demo-by-demo
        src_data = src.get('data')
        if src_data is None:
            raise RuntimeError("No 'data' group found in source file")
        dst_data = dst.create_group('data')
        # copy any attrs on the data group
        for a_key, a_val in src_data.attrs.items():
            dst_data.attrs[a_key] = a_val

        # 3) iterate demos inside data
        for demo_key in tqdm.tqdm(src_data, desc="demos"):
            # copy the demo group
            src_data.copy(demo_key, dst_data, name=demo_key)

            # only process keys starting with 'demo'
            if not demo_key.startswith('demo'):
                continue

            demo_dst = dst_data[demo_key]
            obs_dst  = demo_dst.get('obs')
            if obs_dst is None:
                print(f"[SKIP] no obs in data/{demo_key}")
                continue

            # drop color if present
            if 'point_cloud_color' in obs_dst:
                del obs_dst['point_cloud_color']

# def process_file(src_path, dst_path):
#     with h5py.File(src_path, 'r') as src_file:
#         with h5py.File(dst_path, 'w') as dst_file:
#             copy_and_modify(src_file, dst_file)

if __name__ == "__main__":
    task = 'square_d2'
    src_hdf5 = f"data/robomimic/datasets/{task}/{task}_pcd_abs_better.hdf5"
    dst_hdf5 = f"data/robomimic/datasets/{task}/{task}_pcd_abs_better_no_color.hdf5"
    process_file(src_hdf5, dst_hdf5)
