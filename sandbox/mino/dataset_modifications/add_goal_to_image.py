#!/usr/bin/env python3
import argparse
import h5py
import numpy as np
import tqdm
import third_party.robogen.robogen_utils as ru
import third_party.robogen.goal_cond_diffpo_utils as gdu
import torch
import matplotlib.pyplot as plt
from equi_diffpo.common.goal_to_image_converter import Goal2ImageConverterNumpy

def copy_dataset(src_dataset, dst_group, key, data=None):
    """
    Copy a single dataset (with compression/chunking) into dst_group[key].
    If data is provided, use that instead of src_dataset[()].
    """
    if data is None:
        data = src_dataset[()]

    creation_args = {
        'dtype':           data.dtype,
        'compression':     src_dataset.compression,
        'compression_opts':src_dataset.compression_opts,
        'shuffle':         src_dataset.shuffle,
        'fletcher32':      src_dataset.fletcher32,
        # 'chunks':          src_dataset.chunks,
        # 'maxshape':        src_dataset.maxshape,
    }
    filtered = {k: v for k, v in creation_args.items() if v is not None}
    dst_group.create_dataset(key, data=data, **filtered)

def process_file(src_path: str, dst_path: str, conditioning_type: str):
    goal_to_image_converter = Goal2ImageConverterNumpy(
        img_size=(84,84),
        conditioning_type=conditioning_type
    )

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

        for demo_key in tqdm.tqdm(src_data, desc="demos"):
            src_data.copy(demo_key, dst_data, name=demo_key)
            if not demo_key.startswith('demo'):
                continue
            obs_dst = dst_data[demo_key]['obs']

            # compute goal channels and merge
            required = ('goal_gripper_pcd', 'gripper_pcd', 'point_cloud')
            if all(k in obs_dst for k in required):
                agent_ch, eye_ch = goal_to_image_converter.generate_image_conditioning(obs_dst)

                src_ds = obs_dst['agentview_image_84']
                copy_dataset(src_ds, obs_dst, 'agentview_cond_84', agent_ch)

                src_eye_ds = obs_dst['robot0_eye_in_hand_image_84']
                copy_dataset(src_eye_ds, obs_dst, 'robot0_eye_in_hand_cond_84', eye_ch)

                print(f"[OK] merged goal channels into images for data/{demo_key}")
            else:
                missing = [k for k in required if k not in obs_dst]
                print(f"[SKIP] missing {missing} in data/{demo_key}/obs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process HDF5 dataset with goal conditioning")
    parser.add_argument("--task", default="square_d2", help="Name of MimicGen task")
    parser.add_argument("--src_h5_suffix", default="pcd_abs_images.hdf5", help="Suffix for source hdf5 file")
    parser.add_argument("--dst_h5_suffix", default="pcd_abs_images_dddxyz2.hdf5", help="Suffix for destination suffix hdf5 file")
    parser.add_argument("--conditioning-type", default="3d_flow_world_frame", 
                       help="Conditioning type for Goal2ImageConverter")
    
    args = parser.parse_args()

    task = args.task
    src_hdf5 = f"data/robomimic/datasets/{task}/{task}_{args.src_h5_suffix}"
    dst_hdf5 = f"data/robomimic/datasets/{task}/{task}_{args.dst_h5_suffix}"
    
    process_file(src_hdf5, dst_hdf5, args.conditioning_type)
    print(f"Done ➜ wrote modified file to {dst_hdf5}")
