#!/usr/bin/env python3
"""
Hierarchical eval for MimicGen using:
  - HL: SMITH multitask weighted-displacement PointNet2 (cat_idx-conditioned)
  - LL: 2D DiT image policy from `2D_Hierarchical_Policy_Learning_Github`
        (FlowMatchingDiTImagePolicy with frozen DINOv2)

Per step: env obs -> HL(point_cloud + gripper_pcd) -> 4-pt subgoal
        -> LL(cam0_image, cam1_image, state, goal_gripper_pts) -> hybrid_delta action
        -> SMITH's policy_action_batch_to_env_action -> env.step

Two foreign code roots must be on sys.path BEFORE the LL checkpoint unpickles:
  1. SMITH_HL_REPO  (RoboGen-sim2real) -> test_PointNet2.model_invariant.PointNet2_super_multitask
  2. DIT_2D_REPO    (Low_Level_and_Inference/diffusion_policy) ->
                    diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace.TrainDiffusionUnetHybridWorkspace

Pass them via --smith_hl_repo / --dit_2d_repo (or the wrapper shell script).
"""

import argparse
import collections
import copy
import os
import sys
from pathlib import Path

import numpy as np

# MUST run before any direct or transitive `import mujoco` / robosuite import.
# Populates `OpenGL.EGL.EGLDeviceEXT` at top level so mujoco's egl_ext.py:34 finds it.
# Also requires PYOPENGL_PLATFORM=egl in the env (set by the shell wrapper).
import OpenGL.EGL.EXT.device_base  # noqa: F401  (side-effect import only)


# -----------------------------------------------------------------------------
# sys.path setup MUST happen before any import that would unpickle a foreign
# dill payload. We prepend so foreign packages shadow any same-named internal
# mirror (SMITH ships an older test_PointNet2 in third_party/robogen).
# -----------------------------------------------------------------------------
def _prepend_sys_path(*roots):
    for root in roots:
        root = os.path.abspath(os.path.expanduser(root))
        if not os.path.isdir(root):
            raise FileNotFoundError(f"sys.path prepend target does not exist: {root}")
        if root not in sys.path:
            sys.path.insert(0, root)


def _bootstrap_paths(args):
    # SMITH HL repo bundles its own dp3 at <repo>/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/.
    # test_PointNet2.model_invariant imports `diffusion_policy_3d.model.vision.layers`
    # which only resolves with that nested dir on sys.path.
    smith_hl_dp3 = Path(args.smith_hl_repo) / "3d_diffusion_policy" / "3D-Diffusion-Policy" / "3D-Diffusion-Policy"
    # The 2D LL workspace transitively imports siblings of diffusion_policy/ (e.g. vggt/).
    # So we add Low_Level_and_Inference/ (parent of diffusion_policy/) too.
    dit_2d_parent = Path(args.dit_2d_repo).parent
    _prepend_sys_path(
        args.smith_hl_repo,
        str(smith_hl_dp3),
        args.dit_2d_repo,
        str(dit_2d_parent),
    )
    # Also ensure the SMITH_MimicGen root is importable so eval_smith_utils etc. resolve.
    # This script lives at:
    #   SMITH_on_mimicgen/external/mimicgen/mimicgen/scripts/eval_2d_dit_mimicgen.py
    # parents: [0]=scripts [1]=mimicgen(inner) [2]=mimicgen(outer) [3]=external [4]=SMITH_on_mimicgen
    smith_root = Path(__file__).resolve().parents[4]
    # SMITH vendors robomimic/mimicgen under external/ but does NOT add them to
    # its pixi activation.env PYTHONPATH. Prepend them from Python so they're
    # available regardless of how pixi mangles the shell PYTHONPATH.
    # Note: SMITH's external/mimicgen/offcial_robosuite/ is INCOMPLETE
    # (no robosuite.models subpackage). Use the full copy under MimicGen/.
    _prepend_sys_path(
        str(smith_root),
        str(smith_root / "external" / "robomimic"),
        str(smith_root / "external" / "mimicgen"),
    )
    if args.robosuite_root:
        _prepend_sys_path(args.robosuite_root)


def _stub_unused_smith_imports():
    """
    eval_smith_utils.py:4 has a stale `from train_ddp import TrainDP3Workspace`
    that is already broken in SMITH_MimicGen (no such file defines that class).
    We never invoke its only consumer (load_low_level_policy), so we stub the
    module to make the top-level import succeed without modifying SMITH code.
    """
    import types
    if "train_ddp" not in sys.modules:
        stub = types.ModuleType("train_ddp")
        stub.TrainDP3Workspace = type("TrainDP3Workspace", (), {})
        sys.modules["train_ddp"] = stub


# -----------------------------------------------------------------------------
# Shape meta describing the obs dict the LL model expects.
# Mirrors `MimicGen_Tasks/coffee_goal_gripper.yaml` from the 2D codebase.
# Extra keys (point_cloud, gripper_pcd, robot0_eef_quat) are needed by the HL
# inference path and the action-conversion step but never reach the LL model.
# -----------------------------------------------------------------------------
def get_shape_meta(camera_h: int, camera_w: int):
    return {
        "obs": {
            # LL inputs (2D DiT)
            "cam0_image":       {"shape": [3, camera_h, camera_w], "type": "rgb"},
            "cam1_image":       {"shape": [3, camera_h, camera_w], "type": "rgb"},
            "state":            {"shape": [10],                    "type": "low_dim"},
            "goal_gripper_pts": {"shape": [4, 3],                  "type": "low_dim"},
            # HL inputs / action conversion
            "point_cloud":      {"shape": [4500, 3],               "type": "point_cloud"},
            "gripper_pcd":      {"shape": [4, 3],                  "type": "low_dim"},
            "robot0_eef_quat":  {"shape": [4],                     "type": "low_dim"},
        },
        "action": {"shape": [10]},
    }


# -----------------------------------------------------------------------------
# Build a MimicGen robosuite env, forcing camera resolution to 256x256.
# This is the only known training/eval distribution gap fix: the LL model was
# trained on 256x256 RGBs (--camera_height 256 in convert_dataset.py), while
# env_meta from the dataset usually defaults to ~84x84.
# -----------------------------------------------------------------------------
def create_mimicgen_env(dataset_path: str, shape_meta: dict, camera_h: int, camera_w: int):
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils

    env_meta = FileUtils.get_env_metadata_from_dataset(os.path.expanduser(dataset_path))
    env_meta["env_kwargs"]["use_object_obs"] = False
    env_meta["env_kwargs"]["camera_heights"] = camera_h
    env_meta["env_kwargs"]["camera_widths"]  = camera_w

    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta["obs"].items():
        modality_mapping[attr.get("type", "low_dim")].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=True,
    )
    return env, env_meta


# -----------------------------------------------------------------------------
# Load the 2D DiT low-level policy. Mirrors lines 633-643 of
# `eval_hierarchical_diffpo_single_object.py` in the 2D codebase.
# -----------------------------------------------------------------------------
def load_low_level_2d_dit(exp_dir: str, ckpt_name: str, device: str = "cuda"):
    import hydra
    from omegaconf import OmegaConf

    cfg_path = Path(exp_dir) / ".hydra" / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"LL hydra config missing: {cfg_path}")
    cfg = OmegaConf.load(str(cfg_path))

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    ckpt_path = Path(exp_dir) / "checkpoints" / ckpt_name
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"LL checkpoint missing: {ckpt_path}")
    workspace.load_checkpoint(path=str(ckpt_path))

    policy = copy.deepcopy(workspace.model)
    if OmegaConf.select(workspace.cfg, "training.use_ema", default=False):
        policy = copy.deepcopy(workspace.ema_model)
    policy.eval()
    policy.reset()
    return policy.to(device), cfg


# -----------------------------------------------------------------------------
# Obs-dict builders.
# -----------------------------------------------------------------------------
def _maybe_unwrap(obs):
    """Some robomimic configs return {"obs": ...} after env.reset/step."""
    if isinstance(obs, dict) and "obs" in obs and not {"agentview_image", "state"} & obs.keys():
        return obs["obs"]
    return obs


def build_hl_inputs(obs, device):
    """HL takes [point_cloud (N,3) ; gripper_pcd (4,3)] concatenated -> (1, N+4, 3)."""
    import torch
    pc = torch.from_numpy(obs["point_cloud"]).float().to(device)[None]   # (1, N, 3)
    gp = torch.from_numpy(obs["gripper_pcd"]).float().to(device)[None]   # (1, 4, 3)
    return torch.cat([pc, gp], dim=1)


def build_ll_obs_dict(obs, n_obs_steps: int, device):
    """
    LL takes (B=1, T=n_obs_steps, ...) tensors.

    The 2D dataset class converts h5 (T, H, W, 3) uint8 -> (T, 3, H, W) float32 / 255.
    SMITH's robomimic env emits images already as (3, H, W) float32 in [0, 1] (after
    ObsUtils initialization), so we just stack across T and add a batch dim.
    """
    import torch

    cam0 = obs["agentview_image"]              # (3, H, W) float32 [0,1]
    cam1 = obs["robot0_eye_in_hand_image"]     # (3, H, W) float32 [0,1]
    state = obs["state"]                       # (10,) float32

    cam0_b  = np.tile(cam0[None, None],  (1, n_obs_steps, 1, 1, 1))
    cam1_b  = np.tile(cam1[None, None],  (1, n_obs_steps, 1, 1, 1))
    state_b = np.tile(state[None, None], (1, n_obs_steps, 1))

    return {
        "cam0_image": torch.from_numpy(cam0_b).float().to(device),
        "cam1_image": torch.from_numpy(cam1_b).float().to(device),
        "state":      torch.from_numpy(state_b).float().to(device),
    }


# -----------------------------------------------------------------------------
# Single episode rollout.
# -----------------------------------------------------------------------------
def run_episode(
    env,
    hl_model,
    ll_model,
    cat_embedding,
    controller,
    n_obs_steps: int,
    n_action_steps: int,
    max_steps: int,
    device: str,
):
    import torch
    from eval_smith_utils import (
        infer_multitask_high_level_model,
        policy_action_batch_to_env_action,
    )

    obs = _maybe_unwrap(env.reset())

    max_dpos = float(controller.output_max[0])
    max_drot = float(controller.output_max[3])

    total_reward = 0.0
    success = False
    step = 0

    while step < max_steps:
        # ------------------- HL: PCD -> 4-pt subgoal -------------------------
        with torch.no_grad():
            hl_inputs = build_hl_inputs(obs, device)            # (1, N+4, 3)
            subgoal = infer_multitask_high_level_model(
                hl_inputs, hl_model,
                cat_embedding=cat_embedding,                    # (768,) on cuda
                high_level_args=None, extra=None,
            )                                                   # (1, 4, 3)
            subgoal_t = subgoal.unsqueeze(1).repeat(1, n_obs_steps, 1, 1)  # (1, T, 4, 3)

        # ------------------- LL: image obs + subgoal -> action ---------------
        ll_obs = build_ll_obs_dict(obs, n_obs_steps, device)
        ll_obs["goal_gripper_pts"] = subgoal_t

        with torch.no_grad():
            action_dict = ll_model.predict_action(ll_obs)
        # 2D codebase uses "action" key; some variants also expose "action_pred".
        action_raw = action_dict.get("action_pred", action_dict["action"]).detach().cpu().numpy()
        # action_raw: (1, n_action_steps, 10) hybrid_delta

        # ------------------- hybrid_delta -> env action ----------------------
        eef_quat = np.array(obs["robot0_eef_quat"], dtype=np.float64)
        eef_quats_b = np.tile(eef_quat[np.newaxis, :], (1, 1))             # (1, 4)
        env_action_arm = policy_action_batch_to_env_action(
            action_raw, eef_quats_b, max_dpos, max_drot,
        )                                                                  # (1, T, 7)
        env_action_seq = env_action_arm[0, :n_action_steps]                # (T, 7)

        # ------------------- step env n_action_steps times -------------------
        for t_idx in range(env_action_seq.shape[0]):
            obs, reward, done, _info = env.step(env_action_seq[t_idx])
            obs = _maybe_unwrap(obs)
            total_reward += float(reward)
            step += 1
            if env.is_success().get("task", False):
                success = True
                done = True
            if done or step >= max_steps:
                break

        if success or step >= max_steps:
            break

    return total_reward, success


# -----------------------------------------------------------------------------
# Entrypoint.
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical eval: SMITH HL + 2D DiT LL on a MimicGen task")
    parser.add_argument("--dataset_path", type=str, required=True,
        help="HDF5 dataset whose env_meta defines the MimicGen env (e.g. coffee_d0.hdf5)")
    parser.add_argument("--high_level_path", type=str, required=True,
        help="Path to a SMITH multitask HL .pth (config.json must sit next to it)")
    parser.add_argument("--low_level_exp_dir", type=str, required=True,
        help="Hydra output dir of the 2D DiT training run (.hydra/config.yaml inside)")
    parser.add_argument("--low_level_checkpoint", type=str, default="epoch_60.ckpt")
    parser.add_argument("--siglip_features_path", type=str, required=True,
        help=".pt file containing {'values': [...]} indexed by cat_idx")
    parser.add_argument("--cat_idx", type=int, default=0,
        help="Index into siglip['values']. Coffee_Task fine-tune used 0 — see "
             "RoboGen-sim2real/test_PointNet2/dataset_from_disk.py:135 for the "
             "string-matching logic that produced this.")
    parser.add_argument("--smith_hl_repo", type=str, required=True,
        help="Path to SMITH_High_Level_FineTune/RoboGen-sim2real (provides "
             "test_PointNet2.model_invariant.PointNet2_super_multitask)")
    parser.add_argument("--dit_2d_repo", type=str, required=True,
        help="Path to .../Low_Level_and_Inference/diffusion_policy (so that "
             "the LL ckpt's _target_ resolves on unpickle)")
    parser.add_argument("--robosuite_root", type=str, default="",
        help="Optional path to a complete robosuite checkout (parent dir of "
             "the robosuite/ package). SMITH's vendored offcial_robosuite/ is "
             "incomplete; use MimicGen/robosuite/ instead.")
    parser.add_argument("--n_episodes",     type=int, default=10)
    parser.add_argument("--max_steps",      type=int, default=400)
    parser.add_argument("--seed",           type=int, default=100000)
    parser.add_argument("--n_obs_steps",    type=int, default=2)
    parser.add_argument("--n_action_steps", type=int, default=8)
    parser.add_argument("--camera_h",       type=int, default=256)
    parser.add_argument("--camera_w",       type=int, default=256)
    parser.add_argument("--output_dir",     type=str, default=None,
        help="Where to save args.json / results.jsonl / summary.json. "
             "Default: outputs_eval/<HL_dir>__<LL_dir>_<ckpt>/<timestamp>/.")
    args = parser.parse_args()

    # Resolve output_dir before any heavy work — fail fast on mkdir issues.
    import json
    from datetime import datetime
    if args.output_dir is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        hl_tag = Path(args.high_level_path).parent.name
        ll_tag = Path(args.low_level_exp_dir).name
        ckpt_tag = Path(args.low_level_checkpoint).stem
        args.output_dir = f"outputs_eval/{hl_tag}__{ll_tag}_{ckpt_tag}/{ts}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] saving results to {output_dir.resolve()}")
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Bootstrap sys.path BEFORE importing torch / loading any foreign ckpt.
    _bootstrap_paths(args)
    _stub_unused_smith_imports()

    import torch
    from eval_smith_utils import load_multitask_high_level_model

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build env and grab the controller (for max_dpos / max_drot during action conversion).
    shape_meta = get_shape_meta(args.camera_h, args.camera_w)
    env, env_meta = create_mimicgen_env(
        args.dataset_path, shape_meta, args.camera_h, args.camera_w)
    inner = env.env if hasattr(env, "env") else env
    controller = inner.robots[0].controller

    # Load HL.
    print(f"[HL] loading multitask weighted-displacement model from {args.high_level_path}")
    hl_model, _hl_args = load_multitask_high_level_model(args.high_level_path)
    hl_model.eval().to(device)

    # Load LL (2D DiT).
    print(f"[LL] loading 2D DiT from {args.low_level_exp_dir}/{args.low_level_checkpoint}")
    ll_model, _ll_cfg = load_low_level_2d_dit(
        args.low_level_exp_dir, args.low_level_checkpoint, device=device)

    # SigLIP cat embedding for the HL.
    siglip = torch.load(args.siglip_features_path, map_location="cpu")
    siglip_values = siglip["values"] if isinstance(siglip, dict) else siglip
    cat_embedding = siglip_values[args.cat_idx].float().to(device)
    print(f"[HL] cat_idx={args.cat_idx}  cat_embedding shape={tuple(cat_embedding.shape)}")

    # Roll out. Stream per-episode results to results.jsonl so partial output
    # survives Ctrl+C / OOM.
    rewards, successes = [], []
    results_path = output_dir / "results.jsonl"
    with open(results_path, "w") as results_f:
        for ep in range(args.n_episodes):
            seed = args.seed + ep
            env.seed(seed)
            r, succ = run_episode(
                env, hl_model, ll_model, cat_embedding,
                controller,
                args.n_obs_steps, args.n_action_steps, args.max_steps,
                device=device,
            )
            rewards.append(r)
            successes.append(succ)
            print(f"Episode {ep + 1}/{args.n_episodes}  seed={seed}  reward={r:.2f}  success={succ}")
            results_f.write(json.dumps({
                "episode": ep + 1, "seed": seed,
                "reward": float(r), "success": bool(succ),
            }) + "\n")
            results_f.flush()

    env.close()
    summary = {
        "n_episodes":   args.n_episodes,
        "mean_reward":  float(np.mean(rewards)),
        "std_reward":   float(np.std(rewards)),
        "success_rate": float(np.mean(successes)),
        "successes":    int(sum(successes)),
        "args":         vars(args),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n--- Summary ---")
    print(f"Mean reward:  {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
    print(f"Success rate: {summary['success_rate']:.2%} ({summary['successes']}/{args.n_episodes})")
    print(f"\nSaved: {output_dir.resolve()}")
    print(f"  - args.json")
    print(f"  - results.jsonl")
    print(f"  - summary.json")


if __name__ == "__main__":
    sys.exit(main())
