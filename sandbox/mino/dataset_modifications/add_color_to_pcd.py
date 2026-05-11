#!/usr/bin/env python3
import h5py
import numpy as np
import tqdm
import third_party.robogen.robogen_utils as ru
import torch
import matplotlib.pyplot as plt

def add_dataset(src_ds, dst_group, new_key, data):
    """
    Create dst_group[new_key] with `data`, preserving src_ds creation args and attrs.
    """
    creation_args = {
        'dtype':            src_ds.dtype,
        'compression':      src_ds.compression,
        'compression_opts': src_ds.compression_opts,
        'shuffle':          src_ds.shuffle,
        'fletcher32':       src_ds.fletcher32,
    }
    filtered = {k: v for k, v in creation_args.items() if v is not None}
    new_ds = dst_group.create_dataset(new_key, data=data, **filtered)
    new_ds.attrs.update(src_ds.attrs)
    return new_ds

def process_file(src_path: str, dst_path: str):
    with h5py.File(src_path, 'r') as src, h5py.File(dst_path, 'w') as dst:
        # copy top‐level except 'data'
        for key in src:
            if key != 'data':
                src.copy(key, dst, name=key)

        src_data = src.get('data')
        if src_data is None:
            raise RuntimeError("No 'data' group found in source file")
        dst_data = dst.create_group('data')
        dst_data.attrs.update(src_data.attrs)
        demo_keys = [f'demo_{i}' for i in range(len(src_data))]

        for demo_key in tqdm.tqdm(demo_keys):
            if demo_key not in dst_data:
                src_data.copy(demo_key, dst_data, name=demo_key)
                print(f'adding {demo_key}')
            else:
                print(f'wtf is {demo_key}')
            if not demo_key.startswith('demo'):
                continue
            obs_dst = dst_data[demo_key]['obs']

            # compute goal channels and merge
            required = ('goal_gripper_pcd', 'gripper_pcd', 'point_cloud', 'point_cloud_color')
            if all(k in obs_dst for k in required):

                color_pcd = np.concatenate([
                    obs_dst['point_cloud'],
                    obs_dst['point_cloud_color']
                ], axis=-1)
                src_ds = obs_dst['point_cloud']
                obs_dst.pop('point_cloud', None)
                add_dataset(src_ds, obs_dst, 'point_cloud', color_pcd)
                
                color_gripper_pcd = np.concatenate([
                    obs_dst['gripper_pcd'],
                    np.ones_like(obs_dst['gripper_pcd'])
                ], axis=-1)
                src_ds = obs_dst['gripper_pcd']
                obs_dst.pop('gripper_pcd', None)
                add_dataset(src_ds, obs_dst, 'gripper_pcd', color_gripper_pcd)

                color_goal_gripper_pcd = np.concatenate([
                    obs_dst['goal_gripper_pcd'],
                    np.ones_like(obs_dst['goal_gripper_pcd'])
                ], axis=-1)
                src_ds = obs_dst['goal_gripper_pcd']
                obs_dst.pop('goal_gripper_pcd', None)
                add_dataset(src_ds, obs_dst, 'goal_gripper_pcd', color_goal_gripper_pcd)

                print('adding color / encodings to pcds')
            else:
                missing = [k for k in required if k not in obs_dst]
                print(f"[SKIP] missing {missing} in data/{demo_key}/obs")

if __name__ == "__main__":
    task = 'stack_three_d1'
    src_hdf5 = f"data/robomimic/datasets/{task}/{task}_pcd_abs.hdf5"
    dst_hdf5 = f"data/robomimic/datasets/{task}/{task}_pcd_abs_color.hdf5"
    process_file(src_hdf5, dst_hdf5)
    print(f"Done ➜ wrote modified file to {dst_hdf5}")
