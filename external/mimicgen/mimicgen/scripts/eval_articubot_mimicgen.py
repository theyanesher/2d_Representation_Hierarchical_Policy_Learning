#!/usr/bin/env python3
"""
Evaluate trained ArticuBot high-level and low-level policies in the MimicGen squared2 env.

- Loads high-level and low-level policies (see mimicgen/utils/control_robot.py).
- Builds a MimicGen robomimic env from a dataset (see mimicgen/utils/train_articubot_workspace.py
  and mimicgen/scripts/articubot_pcd_env_runner.py).
- Runs the loop: obs -> high-level subgoal -> low-level action -> convert action -> env.step.

Important conversions (done in mimicgen/utils/articubot_eval_actions.py):
1. **Action normalization**: Robomimic expects actions in [-1, 1]. The policy outputs unnormalized
   deltas; we normalize using the controller's max_dpos and max_drot (and clip gripper to [-1, 1]).
2. **Delta rotation frame**: The low-level policy predicts delta rotation in the **gripper frame**
   (next_rotation = cur_rotation @ delta_rotation). The MimicGen env expects **world frame**
   (next_rotation = delta_rotation @ cur_rotation). We convert before normalizing.

Usage:
  python -m mimicgen.scripts.eval_articubot_mimicgen \\
    --dataset_path /path/to/square_d2.hdf5 \\
    --high_level_path /path/to/high_level/model.pth \\
    --high_level_type weighted_displacement \\
    --low_level_exp_dir /path/to/low_level_exp \\
    --low_level_checkpoint epoch=100.ckpt \\
    --n_episodes 10 --max_steps 400 --seed 42
"""

import argparse
import collections
import os
import sys

import numpy as np
import torch

# Ensure mimicgen envs are registered
try:
    import mimicgen  # noqa: F401
except ImportError:
    pass

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

from mimicgen.utils.articubot_eval_actions import policy_action_batch_to_env_action


def _get_shape_meta():
    """Shape meta for articubot PCD obs (matches runner / dataset)."""
    return {
        "obs": {
            "point_cloud": {"shape": [4500, 3], "type": "point_cloud"},
            "gripper_pcd": {"shape": [4, 3], "type": "low_dim"},
            "state": {"shape": [10], "type": "low_dim"},
            "robot0_eef_pos": {"shape": [3], "type": "low_dim"},
            "robot0_eef_quat": {"shape": [4], "type": "low_dim"},
            "robot0_eef_rot6d": {"shape": [6], "type": "low_dim"},
            "robot0_gripper_qpos": {"shape": [1], "type": "low_dim"},
        },
        "action": {"shape": [7]},
    }


def create_mimicgen_env(dataset_path, shape_meta, disable_object_obs=True):
    """Build a single MimicGen robosuite env from dataset metadata."""
    dataset_path = os.path.expanduser(dataset_path)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    if disable_object_obs:
        env_meta["env_kwargs"]["use_object_obs"] = False

    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta["obs"].items():
        modality_mapping[attr.get("type", "low_dim")].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=True,  # needed for point cloud from cameras
    )
    return env, env_meta


def build_obs_dict_for_policy(obs, n_obs_steps=2, device="cuda"):
    """
    Turn raw env obs (single step) into batched obs dict for policy.
    Repeats the same obs n_obs_steps times to match training.
    """
    point_cloud = obs["point_cloud"]  # (N, 3) or (N, 6) if color
    gripper_pcd = obs["gripper_pcd"]  # (4, 3)
    state = obs["state"]  # (10,) = eef_pos(3) + eef_rot6d(6) + gripper_q(1)

    # Add batch and time steps: (1, n_obs_steps, ...)
    pc = np.tile(point_cloud[np.newaxis, np.newaxis, ...], (1, n_obs_steps, 1, 1))
    gp = np.tile(gripper_pcd[np.newaxis, np.newaxis, ...], (1, n_obs_steps, 1, 1))

    obs_dict = {
        "point_cloud": torch.from_numpy(pc).float().to(device),
        "gripper_pcd": torch.from_numpy(gp).float().to(device),
    }
    if "goal_gripper_pcd" in obs:
        goal = obs["goal_gripper_pcd"]
        goal_batch = np.tile(goal[np.newaxis, np.newaxis, ...], (1, n_obs_steps, 1, 1))
        obs_dict["goal_gripper_pcd"] = torch.from_numpy(goal_batch).float().to(device)
    return obs_dict, state, obs.get("robot0_eef_quat")


def run_high_level_inference(obs_dict, high_level_policy, high_level_type, device="cuda", high_level_args=None):
    """Predict goal gripper PCD (4 points) from current obs."""
    if high_level_type == "weighted_displacement":
        try:
            import third_party.robogen.robogen_utils as ru
        except ImportError:
            from mimicgen.utils import articubot_util as ru
        with torch.no_grad():
            subgoal_pred, _ = ru.run_high_level_policy_inference(
                high_level_policy, obs_dict, return_weights=True
            )
        return subgoal_pred
    elif high_level_type == "multitask":
        from mimicgen.utils.control_robot import infer_multitask_high_level_model
        pc = obs_dict["point_cloud"]
        gp = obs_dict["gripper_pcd"]
        pc_last = pc[:, -1, :, :]
        gp_last = gp[:, -1, :, :]
        inputs = torch.cat([pc_last, gp_last], dim=1)
        with torch.no_grad():
            prediction = infer_multitask_high_level_model(
                inputs, high_level_policy,
                cat_embedding=None,
                high_level_args=high_level_args or {},
                extra=None,
            )
        return prediction.unsqueeze(1)
    else:
        raise ValueError("high_level_type must be 'weighted_displacement' or 'multitask'")


def load_policies(args):
    """Load high-level and low-level policies."""
    from mimicgen.utils import control_robot

    low_level_policy = control_robot.load_low_level_policy(
        args.low_level_exp_dir,
        args.low_level_checkpoint,
    )
    low_level_policy.eval()

    high_level_args = None
    high_level_policy, high_level_args = control_robot.load_multitask_high_level_model(args.high_level_path)

    high_level_policy.eval()
    high_level_policy = high_level_policy.to("cuda")
    low_level_policy = low_level_policy.to("cuda")

    return high_level_policy, low_level_policy, high_level_args


def run_episode(
    env,
    high_level_policy,
    low_level_policy,
    high_level_type,
    controller,
    n_obs_steps,
    max_steps,
    device,
    cat_idx=13,
    high_level_args=None,
):
    """Run one episode; return total reward and success."""
    obs = env.reset()
    if isinstance(obs, dict) and "obs" in obs:
        obs = obs["obs"]
    total_reward = 0.0
    max_dpos = float(controller.output_max[0])
    max_drot = float(controller.output_max[3])

    for step in range(max_steps):
        obs_dict, state, eef_quat = build_obs_dict_for_policy(obs, n_obs_steps, device)
        eef_quat = np.array(eef_quat, dtype=np.float64)

        goal_pcd = run_high_level_inference(
            obs_dict, high_level_policy, high_level_type, device,
            high_level_args=high_level_args,
        )
        obs_dict["goal_gripper_pcd"] = goal_pcd.repeat(1, n_obs_steps, 1, 1)

        with torch.no_grad():
            if hasattr(low_level_policy, "predict_action"):
                cat_t = torch.tensor([cat_idx], device=device)
                action_dict = low_level_policy.predict_action(obs_dict, cat_t)
            else:
                action_dict = low_level_policy.predict_action(obs_dict)
        action_raw = action_dict.get("action_pred", action_dict["action"]).detach().cpu().numpy()

        eef_quats_batch = np.tile(eef_quat[np.newaxis, :], (1, 1))
        env_action_arm = policy_action_batch_to_env_action(
            action_raw, eef_quats_batch, max_dpos, max_drot
        )
        env_action = env_action_arm[0]

        obs, reward, done, info = env.step(env_action)
        if isinstance(obs, dict) and "obs" in obs:
            obs = obs["obs"]
        total_reward += reward
        if done:
            break

    success = bool(env.is_success().get("task", False))
    return total_reward, success


def main():
    parser = argparse.ArgumentParser(description="Eval ArticuBot policies in MimicGen squared2 env")
    parser.add_argument("--dataset_path", type=str, required=True, help="HDF5 dataset (e.g. square_d2.hdf5)")
    parser.add_argument("--high_level_path", type=str, required=True, help="High-level model path")
    parser.add_argument(
        "--high_level_type",
        type=str,
        choices=("weighted_displacement", "multitask"),
        default="weighted_displacement",
        help="High-level model type",
    )
    parser.add_argument("--low_level_exp_dir", type=str, required=True, help="Low-level experiment dir (with .hydra)")
    parser.add_argument("--low_level_checkpoint", type=str, default="latest.ckpt", help="Low-level checkpoint name")
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_obs_steps", type=int, default=2)
    parser.add_argument("--cat_idx", type=int, default=13, help="Category index for low-level policy")
    parser.add_argument("--use_pc_color", action="store_true", help="Use point cloud color for high-level")
    args = parser.parse_args()

    shape_meta = _get_shape_meta()
    env, env_meta = create_mimicgen_env(args.dataset_path, shape_meta)
    inner = env.env if hasattr(env, "env") else env
    controller = inner.robots[0].controller

    high_level_policy, low_level_policy, high_level_args = load_policies(args)
    device = next(low_level_policy.parameters()).device

    rewards = []
    successes = []
    for ep in range(args.n_episodes):
        env.seed(args.seed + ep)
        r, succ = run_episode(
            env,
            high_level_policy,
            low_level_policy,
            args.high_level_type,
            controller,
            args.n_obs_steps,
            args.max_steps,
            device,
            cat_idx=args.cat_idx,
            high_level_args=high_level_args,
        )
        rewards.append(r)
        successes.append(succ)
        print(f"Episode {ep + 1}/{args.n_episodes}  reward={r:.2f}  success={succ}")

    env.close()
    print("\n--- Summary ---")
    print(f"Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Success rate: {np.mean(successes):.2%} ({sum(successes)}/{args.n_episodes})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
