#!/usr/bin/env python3
import h5py
import numpy as np
import tqdm
import third_party.robogen.robogen_utils as ru
import torch

batch_size = 40
high_level_policy = ru.load_high_level_weighted_displacement_policy('third_party/robogen/test_PointNet2/exps/pointnet2_super_model_invariant_2025-05-11_use_all_data_square_D2_abs-obj_square_D2_abs/model_60.pth')

def compute_new_goal_gripper_pcd(
        old_goal_pcd: np.ndarray,
        gripper_pcd:    np.ndarray,
        point_cloud:    np.ndarray
    ) -> np.ndarray:
    """
    Replace this stub with your actual logic.
    Takes the original goal_gripper_pcd, the current gripper_pcd,
    and the point_cloud, and returns a new prediction array.
    """
    N, _, _ = old_goal_pcd.shape
    pred_goals = []
    for i in range(0, N, batch_size):
        start = i
        end   = min(N, i + batch_size)
        batch = {
            'point_cloud': torch.from_numpy(point_cloud[start:end]).to('cuda').unsqueeze(1).float(),
            'gripper_pcd': torch.from_numpy(gripper_pcd[start:end]).to('cuda').unsqueeze(1).float()
        }
        with torch.no_grad():
            pred = ru.run_high_level_policy_inference(high_level_policy, batch).squeeze(1)
            pred_goals.append(pred.cpu().numpy())
            del pred
    pred_goals = np.concatenate(pred_goals, axis=0)
    assert pred_goals.shape == old_goal_pcd.shape
    return pred_goals

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

            # if all required datasets exist, compute & write pred
            required = ('goal_gripper_pcd', 'gripper_pcd', 'point_cloud')
            if all(k in obs_dst for k in required):
                old_goal    = obs_dst['goal_gripper_pcd'][()]
                grip_pcd    = obs_dst['gripper_pcd'][()]
                point_cloud = obs_dst['point_cloud'][()]

                pred = compute_new_goal_gripper_pcd(old_goal, grip_pcd, point_cloud)

                # write out under a new name, preserving layout & attrs
                old_ds = obs_dst['goal_gripper_pcd']
                copy_dataset(old_ds, obs_dst, 'pred_goal_gripper_pcd', data=pred)
                obs_dst['pred_goal_gripper_pcd'].attrs.update(old_ds.attrs)

                print(f"[OK] added pred_goal_gripper_pcd in data/{demo_key}")
            else:
                missing = [k for k in required if k not in obs_dst]
                print(f"[SKIP] missing {missing} in data/{demo_key}/obs")

if __name__ == "__main__":
    # adjust these paths as needed
    task     = 'square_d2'
    src_hdf5 = f"data/robomimic/datasets/{task}/{task}_pcd_abs_512.hdf5"
    dst_hdf5 = f"data/robomimic/datasets/{task}/{task}_pcd_abs_512_mixed_goal.hdf5"

    process_file(src_hdf5, dst_hdf5)
    print(f"Done ➜ wrote modified file to {dst_hdf5}")
