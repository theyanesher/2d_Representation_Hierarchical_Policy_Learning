#!/usr/bin/env python3
import h5py
import numpy as np
import tqdm
import third_party.robogen.robogen_utils as ru
import torch
from matplotlib import pyplot as plt

def copy_dataset(src_dataset, dst_group, key, data=None):
    """
    Copy a single dataset (with compression/chunking) into dst_group[key].
    If data is provided, use that instead of src_dataset[()].
    """
    if data is None:
        data = src_dataset[()]

    creation_args = {
        'dtype':           src_dataset.dtype,
        'compression':     src_dataset.compression,
        'compression_opts':src_dataset.compression_opts,
        'shuffle':         src_dataset.shuffle,
        'fletcher32':      src_dataset.fletcher32,
        'chunks':          src_dataset.chunks,
        'maxshape':        src_dataset.maxshape,
    }
    filtered = {k: v for k, v in creation_args.items() if v is not None}
    dst_group.create_dataset(key, data=data, **filtered)

def process_file(src_path: str, dst_path: str):
    with h5py.File(src_path, 'r') as src, h5py.File(dst_path, 'w') as dst:
        # 1) copy all top‐level keys except 'data'
        for key in src:
            if key != 'data':
                src.copy(key, dst, name=key)

        # 2) recreate & copy 'data' group demo‐by‐demo
        src_data = src.get('data')
        if src_data is None:
            raise RuntimeError("No 'data' group found in source file")
        dst_data = dst.create_group('data')
        # copy any attrs on the data group
        for a_key, a_val in src_data.attrs.items():
            dst_data.attrs[a_key] = a_val

        # 3) iterate demos inside data
        for demo_key in tqdm.tqdm(src_data, desc="demos"):
            # copy the demo group wholesale
            src_data.copy(demo_key, dst_data, name=demo_key)

            # only process keys starting with 'demo'
            if not demo_key.startswith('demo'):
                continue

            obs_dst = dst_data[demo_key].get('obs')
            actions = dst_data[demo_key].get('actions')
            if obs_dst is None:
                print(f"[SKIP] no obs in data/{demo_key}")
                continue
            if actions is None:
                print(f"[SKIP] no actions in data/{demo_key}")
                continue
            # check required datasets
            required = ('gripper_pcd', 'robot0_gripper_qpos', 'goal_gripper_pcd')
            if all(k in obs_dst for k in required):
                grip_pcd  = obs_dst['gripper_pcd'][()]
                grip_qpos = obs_dst['robot0_gripper_qpos'][()]
                # compute your replacement
                new_pcd = ru.compute_new_goal_gripper_pcd(grip_pcd, grip_qpos, actions)

                # replace the old goal_gripper_pcd in place
                old_ds = obs_dst['goal_gripper_pcd']
                del obs_dst['goal_gripper_pcd']
                copy_dataset(old_ds, obs_dst, 'goal_gripper_pcd', data=new_pcd)

                print(f"[OK] replaced goal_gripper_pcd in data/{demo_key}")
            else:
                missing = [k for k in required if k not in obs_dst]
                print(f"[SKIP] missing {missing} in data/{demo_key}/obs")

if __name__ == "__main__":
    # adjust these paths as needed
    task     = 'square_d2'
    src_hdf5 = f"data/robomimic/datasets/{task}/{task}_pcd_abs_512.hdf5"
    dst_hdf5 = f"data/robomimic/datasets/{task}/{task}_pcd_abs_better.hdf5"

    process_file(src_hdf5, dst_hdf5)
    print(f"Done ➜ wrote modified file to {dst_hdf5}")
