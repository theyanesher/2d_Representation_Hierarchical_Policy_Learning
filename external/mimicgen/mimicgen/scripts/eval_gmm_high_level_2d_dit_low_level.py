#!/usr/bin/env python3
"""
Hierarchical eval for MimicGen using:
  - HL: lfd3d ArticubotNetwork (PointNet++ with optional FiLM text conditioning,
        GMM head -> per-anchor 4x3 displacement + 1 weight logit)
  - LL: 2D DiT image policy from `2D_Hierarchical_Policy_Learning_Github`
        trained with task=coffee_gmm_goal (consumes the FULL GMM distribution
        via WeightedCrossAttention with policy.gmm_top_k=1024 applied internally)

Per step: env obs -> HL(scene_pcd + gripper_pcd, gripper-first concat with mask)
        -> ALL N=4500 anchor (4-pt goal, softmax-prob) candidates in world frame
        -> LL(cam0_image, cam1_image, state, gmm_all_goals, gmm_all_weights)
        -> hybrid_delta -> policy_action_batch_to_env_action -> env.step

Sister script of eval_ghost_high_level_2d_dit_low_level.py. The only differences
are isolated to:
  - infer_articubot_gmm (returns the full distribution instead of an argmax sample)
  - build_ll_obs_dict (injects gmm_all_goals + gmm_all_weights instead of goal_gripper_pts)
  - run_episode wiring
Everything else (sys.path bootstrap, lfd3d stubs, env build, LL load, action
conversion, video recorder, summary stats) is intentionally identical to the
ghost script.

Weights format on the wire — verified to match the GMM training data produced by
lfd3d/scripts/run_gmm_on_dataset_batch_optimized.py:
  - obs/gmm_all_weights: softmax probabilities (sum-to-1 per timestep) AFTER stripping
                          the K=4 gripper anchors.
  - obs/gmm_all_goals  : (anchor_xyz + displacement) AFTER stripping the K=4 gripper
                          anchors.
Top-K (1024 by default) is applied INSIDE the LL policy from the training cfg —
we pass all 4500 candidates here.
"""

import argparse
import collections
import copy
import os
import sys
import types
from pathlib import Path

import numpy as np

# MUST run before any direct or transitive `import mujoco` / robosuite import.
# Populates `OpenGL.EGL.EGLDeviceEXT` at top level so mujoco's egl_ext.py:34 finds it.
# Also requires PYOPENGL_PLATFORM=egl in the env (set by the shell wrapper).
import OpenGL.EGL.EXT.device_base  # noqa: F401  (side-effect import only)


# -----------------------------------------------------------------------------
# sys.path setup MUST happen before any import that touches lfd3d / 2D DiT.
# -----------------------------------------------------------------------------
def _prepend_sys_path(*roots):
    for root in roots:
        root = os.path.abspath(os.path.expanduser(root))
        if not os.path.isdir(root):
            raise FileNotFoundError(f"sys.path prepend target does not exist: {root}")
        if root not in sys.path:
            sys.path.insert(0, root)


def _bootstrap_paths(args):
    # The 2D LL workspace transitively imports siblings of diffusion_policy/ (e.g. vggt/).
    # So we add Low_Level_and_Inference/ (parent of diffusion_policy/) too.
    dit_2d_parent = Path(args.dit_2d_repo).parent
    # lfd3d packages live under <lfd3d_repo>/src/.
    lfd3d_src = Path(args.lfd3d_repo) / "src"
    _prepend_sys_path(
        args.dit_2d_repo,
        str(dit_2d_parent),
        str(lfd3d_src),
    )
    # SMITH_on_mimicgen root + its vendored external/ deps.
    # This script lives at:
    #   SMITH_on_mimicgen/external/mimicgen/mimicgen/scripts/eval_gmm_high_level_2d_dit_low_level.py
    # parents: [0]=scripts [1]=mimicgen(inner) [2]=mimicgen(outer) [3]=external [4]=SMITH_on_mimicgen
    smith_root = Path(__file__).resolve().parents[4]
    _prepend_sys_path(
        str(smith_root),
        str(smith_root / "external" / "robomimic"),
        str(smith_root / "external" / "mimicgen"),
    )
    if args.robosuite_root:
        _prepend_sys_path(args.robosuite_root)


# -----------------------------------------------------------------------------
# Stub the train_ddp module so eval_smith_utils' top-level import succeeds.
# Same trick the SMITH eval uses — eval_smith_utils never invokes the path that
# actually needs TrainDP3Workspace (we only import policy_action_batch_to_env_action).
# -----------------------------------------------------------------------------
def _stub_unused_smith_imports():
    if "train_ddp" not in sys.modules:
        stub = types.ModuleType("train_ddp")
        stub.TrainDP3Workspace = type("TrainDP3Workspace", (), {})
        sys.modules["train_ddp"] = stub


# -----------------------------------------------------------------------------
# Stub training-time deps that lfd3d.models.articubot and its transitive
# lfd3d.* imports load at module level. ArticubotNetwork's forward never calls
# any of these — they're only needed by GoalRegressionModule (training/viz).
# Try-import first; only stub if genuinely missing.
# -----------------------------------------------------------------------------
def _ensure_stub(name, attrs=None):
    """Try to import; if missing, install a minimal stub. Never overrides a real module."""
    try:
        __import__(name)
        return
    except ImportError:
        pass
    parts = name.split(".")
    for i in range(len(parts)):
        sub = ".".join(parts[: i + 1])
        if sub not in sys.modules:
            sys.modules[sub] = types.ModuleType(sub)
        if i > 0:
            setattr(sys.modules[".".join(parts[:i])], parts[i], sys.modules[sub])
    if attrs:
        for k, v in attrs.items():
            setattr(sys.modules[name], k, v)


def _force_stub_module(name, attrs=None):
    """Force a stub into sys.modules even if a real module exists. Use only for
    sub-modules whose attributes we know are never reached on the inference path."""
    m = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(m, k, v)
    sys.modules[name] = m
    parent_name, _, leaf = name.rpartition(".")
    if parent_name and parent_name in sys.modules:
        setattr(sys.modules[parent_name], leaf, m)


def _stub_unused_lfd3d_deps():
    """
    1) Stub PyPI training-time deps lfd3d may import at module load.
    2) Force-stub three lfd3d sub-modules (dino_heatmap, tax3d, viz_utils)
       whose symbols articubot.py imports at module-load but uses only in
       GoalRegressionModule methods we never call. This short-circuits the
       deep import chain (tax3d -> lfd3d.models.dit.models -> timm.layers etc.).
    """
    _ensure_stub(
        "pytorch_lightning",
        {"LightningModule": type("LightningModule", (), {})},
    )
    _ensure_stub(
        "diffusers",
        {"get_cosine_schedule_with_warmup": lambda *a, **k: None},
    )
    _ensure_stub("pytorch3d")
    _ensure_stub(
        "pytorch3d.structures",
        {"Pointclouds": type("Pointclouds", (), {})},
    )
    _ensure_stub(
        "pytorch3d.ops",
        {"sample_farthest_points": lambda *a, **k: None},
    )
    _ensure_stub("wandb")
    _ensure_stub("trimesh")
    _ensure_stub("imageio")
    _ensure_stub("imageio.v3")
    _ensure_stub(
        "transformers",
        {
            "AutoImageProcessor": type("AutoImageProcessor", (), {}),
            "AutoModel": type("AutoModel", (), {}),
        },
    )

    # Pre-load the REAL lfd3d parent packages (their __init__.py files are
    # essentially empty), so subsequent sub-module stubs attach correctly.
    import importlib
    importlib.import_module("lfd3d")
    importlib.import_module("lfd3d.models")

    _noop = lambda *a, **k: None
    _force_stub_module("lfd3d.models.dino_heatmap", {"calc_pix_metrics": _noop})
    _force_stub_module("lfd3d.models.tax3d", {"calc_pcd_metrics": _noop})
    # lfd3d/utils has no __init__.py (namespace package). Stub the package
    # entry too so attribute lookups on it don't fall back to disk.
    _force_stub_module("lfd3d.utils")
    _force_stub_module(
        "lfd3d.utils.viz_utils",
        {
            "get_action_anchor_pcd": _noop,
            "get_img_and_track_pcd": _noop,
            "invert_augmentation_and_normalization": _noop,
            "project_pcd_on_image": _noop,
        },
    )


# -----------------------------------------------------------------------------
# Shape meta describing the obs dict the env populates and the LL/HL paths consume.
# Identical to the ghost eval (the env emits the same keys regardless of which
# LL we plug in). gmm_all_goals / gmm_all_weights are NOT registered here — same
# pattern goal_gripper_pts uses in the ghost script: those keys are not produced
# by the env, they are injected by us in build_ll_obs_dict.
# -----------------------------------------------------------------------------
def get_shape_meta(camera_h: int, camera_w: int):
    return {
        "obs": {
            "agentview_image":          {"shape": [3, camera_h, camera_w], "type": "rgb"},
            "robot0_eye_in_hand_image": {"shape": [3, camera_h, camera_w], "type": "rgb"},
            "state":                    {"shape": [10],                    "type": "low_dim"},
            "point_cloud":              {"shape": [4500, 3],               "type": "point_cloud"},
            "gripper_pcd":              {"shape": [4, 3],                  "type": "low_dim"},
            "robot0_eef_quat":          {"shape": [4],                     "type": "low_dim"},
        },
        "action": {"shape": [10]},
    }


# -----------------------------------------------------------------------------
# Build a MimicGen robosuite env at the requested camera resolution.
# Mirrors the ghost eval (camera set must match training distribution).
# -----------------------------------------------------------------------------
def create_mimicgen_env(dataset_path: str, shape_meta: dict, camera_h: int, camera_w: int):
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils

    env_meta = FileUtils.get_env_metadata_from_dataset(os.path.expanduser(dataset_path))

    env_name = env_meta["env_name"]
    if env_name.startswith("PickPlace"):
        camera_names = ["birdview", "agentview", "robot0_eye_in_hand"]
    else:
        camera_names = ["birdview", "agentview", "sideview", "robot0_eye_in_hand"]

    env_kwargs = dict(env_meta["env_kwargs"])
    env_kwargs.pop("env_name", None)
    env_kwargs["camera_names"]   = camera_names
    env_kwargs["camera_heights"] = camera_h
    env_kwargs["camera_widths"]  = camera_w

    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta["obs"].items():
        modality_mapping[attr.get("type", "low_dim")].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env_type = EnvUtils.get_env_type(env_meta=env_meta)
    env = EnvUtils.create_env(
        env_type=env_type,
        env_name=env_name,
        render=False,
        render_offscreen=True,
        use_image_obs=True,
        **env_kwargs,
    )
    return env, env_meta


# -----------------------------------------------------------------------------
# Load the 2D DiT low-level policy. Identical to the ghost eval path —
# the policy class (FlowMatchingDiTImagePolicy) automatically picks up the
# GMM mode from the saved Hydra config (has_gmm=True via gmm_goals/gmm_weights
# keys in shape_meta), use_goal_cross_attention=True, use_weighted_cross_attention=True,
# gmm_top_k=1024.
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
# lfd3d ArticubotNetwork loading + inference.
# -----------------------------------------------------------------------------
def build_articubot_model_cfg(in_channels: int = 4, use_rgb: bool = False):
    """Mirror scripts/run_gmm_on_dataset.py:build_model_cfg for the coffee_task ckpt.
    Identical to the ghost eval — same model architecture, only the LL changes."""
    from omegaconf import OmegaConf
    return OmegaConf.create(
        {
            "name": "articubot",
            "type": "cross_displacement",
            "in_channels": in_channels,
            "num_classes": 13,
            "keep_gripper_in_fps": False,
            "add_action_pcd_masked": True,
            "use_text_embedding": True,
            "use_rgb": use_rgb,
            "use_dual_head": False,
            "is_gmm": True,
            "fixed_variance": [0.01, 0.05, 0.1, 0.25, 0.5],
            "uniform_weights_coeff": 0.1,
        }
    )


def load_articubot_high_level(ckpt_path: str, model_cfg, device: str = "cuda"):
    """
    Load lfd3d ArticubotNetwork from a Lightning .ckpt.
    The Lightning module wraps the network as `self.network`, so saved keys are
    prefixed `network.` — strip that to load directly into ArticubotNetwork.
    """
    import torch
    from lfd3d.models.articubot import ArticubotNetwork

    network = ArticubotNetwork(model_cfg=model_cfg).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    network_state = {
        k[len("network."):]: v for k, v in state.items() if k.startswith("network.")
    }
    missing, unexpected = network.load_state_dict(network_state, strict=False)
    if missing or unexpected:
        print(f"[HL] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    print(f"[HL] loaded ArticubotNetwork from {ckpt_path}")
    network.eval()
    return network


# -----------------------------------------------------------------------------
# CHANGED vs ghost eval: return the FULL GMM distribution instead of one sample.
# Mirrors lfd3d/scripts/run_gmm_on_dataset_batch_optimized.py:infer_gmm — identical
# math (gripper-first concat with mask, strip K=4 gripper anchors, softmax over
# scene anchors only). The training data on disk under `obs/gmm_all_weights` is
# exactly these post-softmax probabilities, and `obs/gmm_all_goals` is exactly
# `gaussian_means[:, K:]`. This function returns both, ready to be tiled across
# n_obs_steps and fed into the LL.
# -----------------------------------------------------------------------------
def infer_articubot_gmm(
    network,
    scene_pcd_t,    # (1, N, 3)
    gripper_pcd_t,  # (1, K=4, 3)
    text_embed_t,   # (1, 1152) — zeros for the coffee ckpt
):
    """
    Returns:
        gaussian_means : (1, N, 4, 3) world-frame 4-keypoint goal candidates
                         (gripper anchors stripped — N=4500 for the coffee task)
        probabilities  : (1, N)       softmax probabilities over those N anchors
                         (sum-to-1 per batch)
    """
    import torch
    import torch.nn.functional as F

    device = scene_pcd_t.device
    B, K, _ = gripper_pcd_t.shape
    N = scene_pcd_t.shape[1]

    # gripper-FIRST concat with a 1/0 mask channel — matches
    # GoalRegressionModule.prepare_scene_pcd (add_action_pcd_masked=True, use_rgb=False).
    gripper_w_mask = torch.cat([gripper_pcd_t, torch.ones(B, K, 1, device=device)], dim=2)
    scene_w_mask   = torch.cat([scene_pcd_t,   torch.zeros(B, N, 1, device=device)], dim=2)
    full = torch.cat([gripper_w_mask, scene_w_mask], dim=1)        # (B, K+N, 4)
    anchor_xyz_full = full[:, :, :3].clone()
    net_in = full.permute(0, 2, 1).contiguous()                    # (B, 4, K+N)

    with torch.no_grad():
        # data_source string is unused when use_dual_head=False (the coffee ckpt).
        outputs = network(
            net_in, text_embedding=text_embed_t, data_source=["libero_franka"],
        )                                                          # (B, K+N, 13)

    B, KN, _ = outputs.shape
    weights = outputs[:, :, -1]                                    # (B, K+N) raw logits
    displacements = outputs[:, :, :-1].reshape(B, KN, 4, 3)        # (B, K+N, 4, 3)
    gaussian_means = anchor_xyz_full[:, :, None, :] + displacements  # (B, K+N, 4, 3)

    # Strip the K=4 gripper anchors — matches the training-data generator.
    weights = weights[:, K:]                                       # (B, N) raw logits
    gaussian_means = gaussian_means[:, K:, :, :]                   # (B, N, 4, 3)

    probabilities = F.softmax(weights, dim=1)                      # (B, N)
    return gaussian_means, probabilities


# -----------------------------------------------------------------------------
# Obs-dict builders.
# -----------------------------------------------------------------------------
def _maybe_unwrap(obs):
    if isinstance(obs, dict) and "obs" in obs and not {"agentview_image", "state"} & obs.keys():
        return obs["obs"]
    return obs


def build_hl_inputs_lfd3d(obs, device):
    """
    Pull scene_pcd + gripper_pcd from env obs, defensively slice to xyz only
    (env may emit point_cloud as (N, 6) = xyz+rgb), and add batch dim.
    Returns:
        scene_pcd_t   : (1, N, 3) on device
        gripper_pcd_t : (1, 4, 3) on device
    """
    import torch
    pc = np.asarray(obs["point_cloud"])
    if pc.ndim == 2 and pc.shape[1] >= 3:
        pc = pc[:, :3]
    gp = np.asarray(obs["gripper_pcd"])
    if gp.ndim == 2 and gp.shape[1] >= 3:
        gp = gp[:, :3]
    scene_pcd_t   = torch.from_numpy(pc).float().to(device).unsqueeze(0)   # (1, N, 3)
    gripper_pcd_t = torch.from_numpy(gp).float().to(device).unsqueeze(0)   # (1, 4, 3)
    return scene_pcd_t, gripper_pcd_t


# CHANGED vs ghost eval: this builds only the always-present LL keys (cams + state).
# The GMM keys are tiled and injected by run_episode after the HL forward.
def build_ll_obs_dict(obs, n_obs_steps: int, device):
    """LL takes (B=1, T=n_obs_steps, ...) tensors. Same image+state path as the ghost eval."""
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


def _agentview_to_uint8_rgb(obs):
    img = np.asarray(obs["agentview_image"])
    frame = np.transpose(img, (1, 2, 0))
    frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


# -----------------------------------------------------------------------------
# Single episode rollout.
# CHANGED vs ghost eval:
#   - HL returns (gaussian_means, probabilities) instead of a single (1,4,3) subgoal.
#   - LL obs gets `gmm_all_goals` and `gmm_all_weights` (tiled over n_obs_steps)
#     instead of `goal_gripper_pts`.
# Everything else is identical.
# -----------------------------------------------------------------------------
def run_episode(
    env,
    hl_network,
    ll_model,
    text_embed_t,
    controller,
    n_obs_steps: int,
    n_action_steps: int,
    max_steps: int,
    device: str,
):
    import torch
    from eval_smith_utils import policy_action_batch_to_env_action

    obs = _maybe_unwrap(env.reset())

    max_dpos = float(controller.output_max[0])
    max_drot = float(controller.output_max[3])

    total_reward = 0.0
    success = False
    step = 0
    frames = [_agentview_to_uint8_rgb(obs)]

    while step < max_steps:
        # ------------------- HL: PCD -> full GMM distribution ----------------
        scene_pcd_t, gripper_pcd_t = build_hl_inputs_lfd3d(obs, device)
        gmm_means, gmm_probs = infer_articubot_gmm(
            hl_network, scene_pcd_t, gripper_pcd_t, text_embed_t,
        )                                                            # (1, N, 4, 3), (1, N)

        # Tile over n_obs_steps so the LL sees one entry per obs step.
        # No top-K truncation here — the LL policy has gmm_top_k=1024 baked in
        # and applies it internally inside _encode_obs. We pass all candidates.
        gmm_means_t = gmm_means.unsqueeze(1).repeat(1, n_obs_steps, 1, 1, 1)   # (1, T, N, 4, 3)
        gmm_probs_t = gmm_probs.unsqueeze(1).repeat(1, n_obs_steps, 1)         # (1, T, N)

        # ------------------- LL: image obs + GMM dist -> action --------------
        ll_obs = build_ll_obs_dict(obs, n_obs_steps, device)
        ll_obs["gmm_all_goals"]   = gmm_means_t
        ll_obs["gmm_all_weights"] = gmm_probs_t

        with torch.no_grad():
            action_dict = ll_model.predict_action(ll_obs)
        action_raw = action_dict.get("action_pred", action_dict["action"]).detach().cpu().numpy()

        # ------------------- hybrid_delta -> env action ----------------------
        eef_quat = np.array(obs["robot0_eef_quat"], dtype=np.float64)
        eef_quats_b = np.tile(eef_quat[np.newaxis, :], (1, 1))
        env_action_arm = policy_action_batch_to_env_action(
            action_raw, eef_quats_b, max_dpos, max_drot,
        )
        env_action_seq = env_action_arm[0, :n_action_steps]          # (T, 7)

        # ------------------- step env n_action_steps times -------------------
        for t_idx in range(env_action_seq.shape[0]):
            obs, reward, done, _info = env.step(env_action_seq[t_idx])
            obs = _maybe_unwrap(obs)
            frames.append(_agentview_to_uint8_rgb(obs))
            total_reward += float(reward)
            step += 1
            if env.is_success().get("task", False):
                success = True
                done = True
            if done or step >= max_steps:
                break

        if success or step >= max_steps:
            break

    return total_reward, success, frames


# -----------------------------------------------------------------------------
# Entrypoint.
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical eval: lfd3d ArticubotNetwork HL + 2D DiT GMM-LL on a MimicGen task")
    parser.add_argument("--dataset_path", type=str, required=True,
        help="HDF5 dataset whose env_meta defines the MimicGen env (e.g. coffee_d0.hdf5)")
    parser.add_argument("--high_level_ckpt", type=str, required=True,
        help="Path to a lfd3d ArticubotNetwork Lightning .ckpt")
    parser.add_argument("--low_level_exp_dir", type=str, required=True,
        help="Hydra output dir of the GMM-LL training run "
             "(.hydra/config.yaml inside; expected to have policy.gmm_top_k>0 and "
             "use_goal_cross_attention=True, use_weighted_cross_attention=True)")
    parser.add_argument("--low_level_checkpoint", type=str, default="epoch_30.ckpt")
    parser.add_argument("--lfd3d_repo", type=str, required=True,
        help="Path to lfd3d/lfd3d/ (provides lfd3d.models.articubot.ArticubotNetwork via src/)")
    parser.add_argument("--dit_2d_repo", type=str, required=True,
        help="Path to .../Low_Level_and_Inference/diffusion_policy")
    parser.add_argument("--robosuite_root", type=str, default="",
        help="Optional path to a complete robosuite checkout (parent dir of the robosuite/ package)")
    parser.add_argument("--text_embed_cache", type=str, default="",
        help="Optional .npy with a (1152,) text embedding for the HL FiLM block. "
             "Default: zeros — matches the coffee ckpt's training (use_text_embed=False). "
             "Feeding a real embedding here is off-distribution for that ckpt.")
    parser.add_argument("--hl_in_channels", type=int, default=4,
        help="ArticubotNetwork in_channels. 4 = xyz + 1 mask (use_rgb=False).")
    parser.add_argument("--n_episodes",     type=int, default=10)
    parser.add_argument("--max_steps",      type=int, default=400)
    parser.add_argument("--seed",           type=int, default=100000)
    parser.add_argument("--n_obs_steps",    type=int, default=2)
    parser.add_argument("--n_action_steps", type=int, default=8)
    parser.add_argument("--camera_h",       type=int, default=256)
    parser.add_argument("--camera_w",       type=int, default=256)
    parser.add_argument("--output_dir",     type=str, default=None,
        help="Where to save args.json / results.jsonl / summary.json. "
             "Default: outputs_eval_gmm/<HL_dir>__<LL_dir>_<ckpt>/<timestamp>/.")
    parser.add_argument("--save_videos", action=argparse.BooleanOptionalAction, default=True,
        help="Write an mp4 of every episode to <output_dir>/media/. "
             "Filenames carry the outcome (..._success.mp4 / ..._failure.mp4). Pass --no-save-videos to disable.")
    parser.add_argument("--video_fps", type=int, default=10,
        help="Frame rate for saved mp4s.")
    args = parser.parse_args()

    import json
    from datetime import datetime
    if args.output_dir is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        hl_tag = Path(args.high_level_ckpt).stem
        ll_tag = Path(args.low_level_exp_dir).name
        ckpt_tag = Path(args.low_level_checkpoint).stem
        args.output_dir = f"outputs_eval_gmm/{hl_tag}__{ll_tag}_{ckpt_tag}/{ts}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] saving results to {output_dir.resolve()}")
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Bootstrap sys.path + stub training-time deps BEFORE importing torch /
    # loading any foreign ckpt.
    _bootstrap_paths(args)
    _stub_unused_smith_imports()
    _stub_unused_lfd3d_deps()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build env and grab the controller (for max_dpos / max_drot during action conversion).
    shape_meta = get_shape_meta(args.camera_h, args.camera_w)
    env, env_meta = create_mimicgen_env(
        args.dataset_path, shape_meta, args.camera_h, args.camera_w)
    inner = env.env if hasattr(env, "env") else env
    controller = inner.robots[0].controller

    # Load HL (lfd3d ArticubotNetwork).
    print(f"[HL] loading lfd3d ArticubotNetwork from {args.high_level_ckpt}")
    hl_cfg = build_articubot_model_cfg(in_channels=args.hl_in_channels, use_rgb=False)
    hl_network = load_articubot_high_level(args.high_level_ckpt, hl_cfg, device=device)

    # Text embedding for the HL FiLM block.
    if args.text_embed_cache and os.path.exists(args.text_embed_cache):
        text_embed_np = np.load(args.text_embed_cache).astype(np.float32)
        print(f"[HL] loaded text embedding from {args.text_embed_cache} (shape={text_embed_np.shape})")
    else:
        text_embed_np = np.zeros(1152, dtype=np.float32)
        print("[HL] using zero (1152,) text embedding (matches coffee ckpt's use_text_embed=False)")
    text_embed_t = torch.from_numpy(text_embed_np).float().unsqueeze(0).to(device)  # (1, 1152)

    # Load LL (2D DiT GMM-conditioned).
    print(f"[LL] loading 2D DiT GMM-LL from {args.low_level_exp_dir}/{args.low_level_checkpoint}")
    ll_model, ll_cfg = load_low_level_2d_dit(
        args.low_level_exp_dir, args.low_level_checkpoint, device=device)

    # Sanity-check the LL config: this script only makes sense when the LL was
    # actually trained on the GMM task — print a clear warning otherwise.
    from omegaconf import OmegaConf
    ll_use_gca   = OmegaConf.select(ll_cfg, "policy.use_goal_cross_attention",   default=False)
    ll_use_wca   = OmegaConf.select(ll_cfg, "policy.use_weighted_cross_attention", default=False)
    ll_top_k     = OmegaConf.select(ll_cfg, "policy.gmm_top_k",                  default=None)
    print(f"[LL] use_goal_cross_attention={ll_use_gca}, "
          f"use_weighted_cross_attention={ll_use_wca}, gmm_top_k={ll_top_k}")
    if not (ll_use_gca and ll_use_wca):
        print("[LL][WARN] this LL run was NOT trained with weighted GMM cross-attention. "
              "The GMM tensors will be ignored unless the LL's shape_meta has gmm_goals/gmm_weights typed obs.")

    # Video setup.
    video_recorder_cls = None
    videos_dir = None
    if args.save_videos:
        from equi_diffpo.gym_util.video_recording_wrapper import VideoRecorder as video_recorder_cls
        videos_dir = output_dir / "media"
        videos_dir.mkdir(parents=True, exist_ok=True)
        print(f"[video] all rollouts -> {videos_dir.resolve()} (fps={args.video_fps}, h264 crf=22)")

    # Roll out — stream per-episode results to results.jsonl.
    rewards, successes = [], []
    results_path = output_dir / "results.jsonl"
    with open(results_path, "w") as results_f:
        for ep in range(args.n_episodes):
            seed = args.seed + ep
            np.random.seed(seed)
            torch.manual_seed(seed)
            r, succ, frames = run_episode(
                env, hl_network, ll_model, text_embed_t,
                controller,
                args.n_obs_steps, args.n_action_steps, args.max_steps,
                device=device,
            )

            video_path = None
            if args.save_videos:
                outcome_tag = "success" if succ else "failure"
                video_path = videos_dir / f"episode_{ep + 1:03d}_seed_{seed}_{outcome_tag}.mp4"
                recorder = video_recorder_cls.create_h264(fps=args.video_fps, crf=22)
                recorder.start(str(video_path))
                for frame in frames:
                    recorder.write_frame(frame)
                recorder.stop()

            rewards.append(r)
            successes.append(succ)
            video_tag = f"  video={video_path.name}" if video_path is not None else ""
            print(f"Episode {ep + 1}/{args.n_episodes}  seed={seed}  reward={r:.2f}  success={succ}{video_tag}")
            results_f.write(json.dumps({
                "episode": ep + 1, "seed": seed,
                "reward": float(r), "success": bool(succ),
                "video": str(video_path) if video_path is not None else None,
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
