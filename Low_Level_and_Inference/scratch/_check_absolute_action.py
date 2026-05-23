"""Standalone sanity check for action_mode='absolute' on Coffee_D2."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, REPO)                              # for `manipulation`, `common`
sys.path.insert(0, os.path.join(REPO, "diffusion_policy"))  # for diffusion_policy.*

import numpy as np
from diffusion_policy.dataset.lazy_articubot_dataset import LazyArticuBotDataset
from manipulation.utils import rotation_transfer_6D_to_matrix


def main():
    shape_meta = {
        "obs": {
            "cam0_image": {"shape": [3, 256, 256], "type": "rgb"},
            "cam1_image": {"shape": [3, 256, 256], "type": "rgb"},
            "state": {"shape": [10], "type": "low_dim"},
            "gmm_all_goals": {"shape": [4500, 4, 3], "type": "gmm_goals"},
            "gmm_all_weights": {"shape": [4500], "type": "gmm_weights"},
        },
        "action": {"shape": [10]},
    }

    ds = LazyArticuBotDataset(
        shape_meta=shape_meta,
        data_dir=("/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/"
                  "Articubot_Data_For_RVT/SMITH_High_Level_FineTune/"
                  "LOW_LEVEL_WITH_GMM_DATASET_GROOT_STYLE_DATASET/D2/Coffee_D2/"),
        horizon=16, pad_before=1, pad_after=7, n_obs_steps=2,
        val_ratio=0.02, max_train_episodes=4, action_mode="absolute",
    )
    print(f"len(ds)={len(ds)}, episode_ends={ds.replay_buffer.episode_ends}")

    # Pick a non-boundary index for clean cross-check
    test_idx = 50
    sample = ds[test_idx]
    b0, b1, s0, s1 = ds.sampler.indices[test_idx]
    print(f"idx={test_idx}: buf=[{b0}:{b1}] sample=[{s0}:{s1}]")
    print(f"sample['action'].shape = {sample['action'].shape}")
    print(f"sample['obs']['state'].shape = {sample['obs']['state'].shape}")

    # Cross-check vs manual composition from the same replay buffer.
    state_arr = ds.replay_buffer["state"]
    action_arr = ds.replay_buffer["action"]
    vs = state_arr[b0:b1].astype(np.float32)
    va = action_arr[b0:b1].astype(np.float32)
    abs_p = vs[:, :3] + va[:, :3]
    abs_g = vs[:, 9:10] + va[:, 9:10]
    sR = np.stack([rotation_transfer_6D_to_matrix(s[3:9]) for s in vs])
    dR = np.stack([rotation_transfer_6D_to_matrix(a[3:9]) for a in va])
    abs_R = sR @ dR

    ds_act = sample["action"].numpy()
    n = vs.shape[0]
    xerr = np.linalg.norm(ds_act[s0:s0 + n, :3] - abs_p, axis=1).max()
    gerr = np.abs(ds_act[s0:s0 + n, 9:10] - abs_g).max()
    dsR = np.stack([rotation_transfer_6D_to_matrix(ds_act[s0 + i, 3:9]) for i in range(n)])
    rerr = np.linalg.norm(dsR - abs_R, axis=(1, 2)).max()

    print("\nCross-check (dataset vs manual state⊕action composition):")
    print(f"  xyz max err : {xerr:.3e}")
    print(f"  grip max err: {gerr:.3e}")
    print(f"  rot max err : {rerr:.3e}")

    print("\nFirst 3 dataset absolute xyz:")
    print(ds_act[s0:s0 + 3, :3])
    print("First 3 manual abs_p:")
    print(abs_p[:3])

    ok = xerr < 1e-5 and gerr < 1e-5 and rerr < 1e-5
    print("\n=>", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
