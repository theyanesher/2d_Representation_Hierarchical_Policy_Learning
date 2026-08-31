#!/usr/bin/env python3
"""
Hierarchical eval, PARALLEL variant of the STANDARD (non-RoPE) low-level.

DERIVED BY COPY from eval_gmm_high_level_rope_2d_dit_low_level.py, keeping that
script's parallel rollout scheduler verbatim and reverting its four RoPE-specific
pieces back to the plain eval_gmm_high_level_2d_dit_low_level.py behaviour.

Use this for a visual_encoder_type=dinov2 low-level (e.g. the Kitchen_D1 beta /
WCA ablation sweep) when you want the multi-env speedup. For a
dinov2_rope4d_grounded[_goals] low-level use the RoPE script instead -- this one
does not emit depth/intrinsics/extrinsics, so that trunk would have nothing to
unproject.

The low-level receives exactly the 5 keys the standard eval supplies:
    cam0_image cam1_image state          (from the env)
    gmm_all_goals gmm_all_weights        (injected from the HL forward)

Differences from the RoPE parent, and ONLY these:
  1. get_shape_meta drops the two *_depth keys and the four intrinsics /
     extrinsics keys; back to the standard six-key obs set.
  2. create_mimicgen_env no longer sets camera_depths=True.
  3. create_mimicgen_env no longer overrides DepthModality's processor. That
     override only existed to keep metric depth unclipped for the RoPE trunk;
     with no depth requested it would be a no-op with global side effects.
  4. build_ll_obs_dict / build_batched_ll_obs_dict supply only cam0_image,
     cam1_image and state.

Everything else is untouched and therefore shared with the RoPE script:
  * Spawned MuJoCo workers parallelize rendering / point-cloud construction;
    their active observations share batched Articubot and DiT forwards.
    Independent per-episode noise generators keep stochastic policy noise
    invariant to worker batch size and sibling completion order.
  * The full four-camera point cloud is constructed only at policy boundaries,
    not for the seven intermediate observations discarded within an 8-action
    chunk. Articubot therefore receives the same observation type as before.
  * Only the first --num_video_episodes episodes are recorded.

The HL math, GMM injection, WCA, action conversion, termination rules, seeds and
obs TILING convention all match the sequential standard eval, so numbers from
this script are directly comparable to runs made with it.

--use_gmm_modes is NOT supported here (same restriction as the RoPE parent); use
the sequential eval_gmm_high_level_2d_dit_low_level.py for mode-based runs.
"""

import argparse
import collections
import copy
import functools
import os
import sys
import time
import types
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np

# MUST run before any direct or transitive `import mujoco` / robosuite import.
# Kept lazy so spawned workers create their own EGL state and CPU-only tests can
# import the scheduling helpers without an installed EGL stack.
def _prepare_egl_import():
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
    # No camera_depths / DepthModality override here: the standard dinov2 encoder
    # never reads depth. See the RoPE sibling script if you need those.

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
            net_in,
            text_embedding=text_embed_t,
            data_source=["libero_franka"] * B,
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
# Halo-collapse the per-anchor GMM into a few discrete modes.
#
# LOGIC-IDENTICAL copy of reduce_gmm_to_modes in the HL dataset generator
# (lfd3d/scripts/run_gmm_on_dataset_batch_optimized.py). Keep the two IN SYNC so
# the modes the LL sees at inference match exactly what it was trained on
# (same radius / max_modes / weighting). Inlined (not imported) because the
# inference env stubs out the HL repo's heavy training deps.
#
#   weight-ordered greedy NMS in goal-centroid space:
#     mode weight = SUM of member anchor weights (total probability mass)
#     mode goal   = weight-weighted average of member goals (4, 3)
# -----------------------------------------------------------------------------
def reduce_gmm_to_modes(all_goals, all_weights, radius=0.03, max_modes=3, w_thresh=1e-4):
    """(T,N,4,3)+(T,N) -> (T,max_modes,4,3)+(T,max_modes), zero-padded/sorted by weight."""
    T, N = all_weights.shape
    modes = np.zeros((T, max_modes, 4, 3), dtype=np.float32)
    mode_weights = np.zeros((T, max_modes), dtype=np.float32)
    for t in range(T):
        w = all_weights[t]
        keep = np.where(w > w_thresh)[0]
        if keep.size == 0:
            continue
        g = all_goals[t, keep]                 # (n, 4, 3)
        wk = w[keep].astype(np.float64)        # (n,)
        cent = g.mean(axis=1)                  # (n, 3) goal centroids
        used = np.zeros(keep.size, dtype=bool)
        collected = []                         # list of (mode_weight, mode_goal(4,3))
        for idx in np.argsort(wk)[::-1]:       # seeds in descending weight
            if used[idx]:
                continue
            d = np.linalg.norm(cent - cent[idx], axis=1)
            members = np.where((d <= radius) & (~used))[0]
            used[members] = True
            mw = wk[members].sum()
            mg = (g[members] * wk[members][:, None, None]).sum(axis=0) / mw
            collected.append((mw, mg.astype(np.float32)))
        collected.sort(key=lambda x: x[0], reverse=True)
        for k, (mw, mg) in enumerate(collected[:max_modes]):
            modes[t, k] = mg
            mode_weights[t, k] = mw
    return modes, mode_weights


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


# Same TILING convention as the sequential standard eval (one env observation
# repeated across n_obs_steps), and the same key set.
# The GMM keys are still tiled and injected by run_episode after the HL forward.
def build_ll_obs_dict(obs, n_obs_steps: int, device):
    """LL takes (B=1, T=n_obs_steps, ...) tensors.

    Only the always-present standard keys (cams + state). The GMM keys are tiled
    and injected after the HL forward, as in the sequential standard eval.
    """
    import torch

    cam0 = np.asarray(obs["agentview_image"], dtype=np.float32)               # (3,H,W) [0,1]
    cam1 = np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.float32)
    state = np.asarray(obs["state"], dtype=np.float32)                        # (10,)

    def tile(x):
        return np.tile(x[None, None], (1, n_obs_steps) + (1,) * x.ndim)

    out = {
        "cam0_image": tile(cam0),
        "cam1_image": tile(cam1),
        "state":      tile(state),
    }
    return {k: torch.from_numpy(v).float().to(device) for k, v in out.items()}


def build_batched_hl_inputs_lfd3d(obs, active_indices, device):
    """Transfer active vector-env point clouds as one contiguous GPU batch."""
    import torch

    active_indices = np.asarray(active_indices, dtype=np.int64)
    pc = np.asarray(obs["point_cloud"])[active_indices, ..., :3]
    gp = np.asarray(obs["gripper_pcd"])[active_indices, ..., :3]
    scene = torch.from_numpy(np.ascontiguousarray(pc)).to(device=device, dtype=torch.float32)
    gripper = torch.from_numpy(np.ascontiguousarray(gp)).to(device=device, dtype=torch.float32)
    return scene, gripper


def build_batched_ll_obs_dict(obs, active_indices, n_obs_steps: int, device):
    """Build active LL inputs and repeat the current observation with ``expand``."""
    import torch

    key_map = {
        "cam0_image": "agentview_image",
        "cam1_image": "robot0_eye_in_hand_image",
        "state": "state",
    }
    active_indices = np.asarray(active_indices, dtype=np.int64)
    result = {}
    for policy_key, env_key in key_map.items():
        value = np.asarray(obs[env_key])[active_indices]
        tensor = torch.from_numpy(np.ascontiguousarray(value)).to(
            device=device, dtype=torch.float32
        )
        result[policy_key] = tensor.unsqueeze(1).expand(
            (-1, n_obs_steps) + tuple(tensor.shape[1:])
        )
    return result


def policy_action_batch_to_env_action_vectorized(
    action_batch: np.ndarray,
    cur_eef_quats: np.ndarray,
    max_dpos: float,
    max_drot: float,
) -> np.ndarray:
    """Vectorized equivalent of eval_smith_utils' hybrid-delta conversion."""
    from scipy.spatial.transform import Rotation

    action_batch = np.asarray(action_batch, dtype=np.float64)
    cur_eef_quats = np.asarray(cur_eef_quats, dtype=np.float64)
    if action_batch.ndim != 3 or action_batch.shape[-1] != 10:
        raise ValueError(f"expected action_batch (B,T,10), got {action_batch.shape}")
    if cur_eef_quats.shape != (action_batch.shape[0], 4):
        raise ValueError(
            f"expected cur_eef_quats ({action_batch.shape[0]},4), got {cur_eef_quats.shape}"
        )

    rot6d = action_batch[..., 3:9].reshape(*action_batch.shape[:2], 2, 3)
    a1, a2 = rot6d[..., 0, :], rot6d[..., 1, :]
    eps = np.finfo(np.float64).eps
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), eps)
    b2 = a2 - np.sum(a2 * b1, axis=-1, keepdims=True) * b1
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), eps)
    b3 = np.cross(b1, b2)
    delta_rot_gripper = np.stack((b1, b2, b3), axis=-1)

    cur_rot = Rotation.from_quat(cur_eef_quats).as_matrix()
    delta_rot_world = (
        cur_rot[:, None] @ delta_rot_gripper @ np.swapaxes(cur_rot[:, None], -1, -2)
    )
    delta_axisangle_world = Rotation.from_matrix(
        delta_rot_world.reshape(-1, 3, 3)
    ).as_rotvec().reshape(*action_batch.shape[:2], 3)

    result = np.empty(action_batch.shape[:2] + (7,), dtype=np.float32)
    result[..., :3] = np.clip(action_batch[..., :3] / max_dpos, -1.0, 1.0)
    result[..., 3:6] = np.clip(delta_axisangle_world / max_drot, -1.0, 1.0)
    result[..., 6] = np.clip(action_batch[..., 9] / -0.01, -1.0, 1.0)
    return result


def _make_observation_space(shape_meta):
    from gym import spaces

    result = spaces.Dict()
    for key, value in shape_meta["obs"].items():
        low, high = -np.inf, np.inf
        if key.endswith("image"):
            low, high = 0.0, 1.0
        result[key] = spaces.Box(
            low=low, high=high, shape=tuple(value["shape"]), dtype=np.float32
        )
    return result


class _SpaceOnlyEnv:
    """OpenGL-free env used only for AsyncVectorEnv space discovery."""

    metadata = {}

    def __init__(self, shape_meta, n_action_steps):
        from gym import spaces

        self.observation_space = _make_observation_space(shape_meta)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_action_steps, 7), dtype=np.float32
        )

    def close(self):
        return None


class _ParallelMimicGenEpisodeEnv:
    """Worker-owned Approach 1 env that consumes one action chunk per step."""

    metadata = {}

    def __init__(
        self,
        dataset_path,
        shape_meta,
        camera_h,
        camera_w,
        n_action_steps,
        max_steps,
        video_fps,
    ):
        from gym import spaces

        _prepare_egl_import()
        self.observation_space = _make_observation_space(shape_meta)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_action_steps, 7), dtype=np.float32
        )
        self.env, _ = create_mimicgen_env(dataset_path, shape_meta, camera_h, camera_w)
        inner = self.env.env if hasattr(self.env, "env") else self.env
        controller = inner.robots[0].controller
        self.controller_limits = (
            float(controller.output_max[0]),
            float(controller.output_max[3]),
        )
        self.max_steps = int(max_steps)
        self.video_fps = int(video_fps)
        self.pending_seed = None
        self.pending_video_path = None
        self.active = True
        self.done = False
        self.success = False
        self.steps = 0
        self.total_reward = 0.0
        self.video_seconds = 0.0
        self.last_obs = None
        self.video_recorder = None
        self.point_cloud_builds = 0
        self.point_cloud_skips = 0

    def seed(self, seed=None):
        self.pending_seed = seed

    def configure_episode(self, seed, video_path=None, active=True):
        self._stop_video()
        self.pending_seed = int(seed)
        self.pending_video_path = video_path
        self.active = bool(active)

    def reset(self):
        if self.pending_seed is not None:
            np.random.seed(self.pending_seed)
        self.env.set_point_cloud_enabled(True)
        self.last_obs = _maybe_unwrap(self.env.reset())
        self.done = not self.active
        self.success = False
        self.steps = 0
        self.total_reward = 0.0
        self.video_seconds = 0.0
        self.point_cloud_builds = 1
        self.point_cloud_skips = 0
        if self.active and self.pending_video_path:
            from equi_diffpo.gym_util.video_recording_wrapper import VideoRecorder

            t0 = time.perf_counter()
            self.video_recorder = VideoRecorder.create_h264(
                fps=self.video_fps, crf=22, thread_type="FRAME", thread_count=1
            )
            self.video_recorder.start(str(self.pending_video_path))
            self.video_recorder.write_frame(_agentview_to_uint8_rgb(self.last_obs))
            self.video_seconds += time.perf_counter() - t0
        return self._filtered_obs(self.last_obs)

    def step(self, action_sequence):
        if self.done:
            return self._filtered_obs(self.last_obs), 0.0, True, self._info()

        chunk_reward = 0.0
        actions = np.asarray(action_sequence, dtype=np.float32)
        try:
            for action_index, action in enumerate(actions):
                # The next HL/LL forward only consumes the observation returned
                # after this whole chunk. Avoid four segmentation renders,
                # Open3D fusion and FPS for every discarded intermediate obs.
                policy_boundary = (
                    action_index == len(actions) - 1
                    or self.steps + 1 >= self.max_steps
                )
                self.env.set_point_cloud_enabled(policy_boundary)
                if policy_boundary:
                    self.point_cloud_builds += 1
                else:
                    self.point_cloud_skips += 1

                obs, reward, _done, _info = self.env.step(action)
                self.last_obs = _maybe_unwrap(obs)
                reward = float(reward)
                chunk_reward += reward
                self.total_reward += reward
                self.steps += 1
                if self.video_recorder is not None:
                    t0 = time.perf_counter()
                    self.video_recorder.write_frame(_agentview_to_uint8_rgb(self.last_obs))
                    self.video_seconds += time.perf_counter() - t0
                if self.env.is_success().get("task", False):
                    self.success = True
                    self.done = True
                if self.steps >= self.max_steps:
                    self.done = True
                if self.done:
                    # A success may terminate before the planned chunk boundary.
                    # Rebuild once so AsyncVectorEnv still receives its fixed
                    # observation schema. No further policy inference is run.
                    if not policy_boundary:
                        self.env.set_point_cloud_enabled(True)
                        self.last_obs = _maybe_unwrap(self.env.get_observation())
                        self.point_cloud_builds += 1
                    break
        finally:
            # Resets and any out-of-band observation request retain legacy
            # behavior even if stepping raises.
            self.env.set_point_cloud_enabled(True)
        return self._filtered_obs(self.last_obs), chunk_reward, self.done, self._info()

    def get_controller_limits(self):
        return self.controller_limits

    def get_episode_result(self):
        self._stop_video()
        return {
            "reward": self.total_reward,
            "success": self.success,
            "steps": self.steps,
            "video": self.pending_video_path,
            "video_seconds": self.video_seconds,
            "point_cloud_builds": self.point_cloud_builds,
            "point_cloud_skips": self.point_cloud_skips,
        }

    def close(self):
        self._stop_video()
        close = getattr(self.env, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    def _filtered_obs(self, obs):
        return {
            key: np.asarray(obs[key], dtype=np.float32)
            for key in self.observation_space.keys()
        }

    def _info(self):
        return {"success": self.success, "steps": self.steps}

    def _stop_video(self):
        if self.video_recorder is not None:
            t0 = time.perf_counter()
            self.video_recorder.stop()
            self.video_seconds += time.perf_counter() - t0
            self.video_recorder = None


def _make_parallel_env(**kwargs):
    return _ParallelMimicGenEpisodeEnv(**kwargs)


def _make_space_only_env(shape_meta, n_action_steps):
    return _SpaceOnlyEnv(shape_meta, n_action_steps)


def _agentview_to_uint8_rgb(obs):
    img = np.asarray(obs["agentview_image"])
    frame = np.transpose(img, (1, 2, 0))
    frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


# -----------------------------------------------------------------------------
# Goal-overlay helpers (NEW). Purely additive — used only when
# --save_goal_overlay_videos is on. Never touched by the existing inference path.
# Duplicated from the ghost eval script (intentional, to keep these two scripts
# fully independent — refactor into a shared module later if desired).
# -----------------------------------------------------------------------------
def _get_agentview_world_to_pixel(env, camera_h: int, camera_w: int):
    """4x4 world->pixel transform for the 'agentview' camera. None on any failure."""
    try:
        from robosuite.utils.camera_utils import get_camera_transform_matrix
        inner = env.env if hasattr(env, "env") else env
        if not hasattr(inner, "sim"):
            return None
        return get_camera_transform_matrix(inner.sim, "agentview", camera_h, camera_w)
    except Exception as e:
        print(f"[overlay] disabled — camera transform unavailable: {e}")
        return None


def _get_robot_base_world_pos(env, body_name_substring: str = "robot0_base"):
    """
    Return the world-frame xyz of the mujoco body whose name contains
    `body_name_substring`. The HL/LL obs frames are robot-base relative, so this
    offset must be added to each keypoint before projecting through the camera matrix.
    """
    try:
        inner = env.env if hasattr(env, "env") else env
        if not hasattr(inner, "sim"):
            return None
        sim = inner.sim
        for i in range(sim.model.nbody):
            name = sim.model.body_id2name(i)
            if name is not None and body_name_substring in name:
                return np.array(sim.data.body_xpos[sim.model.body_name2id(name)], dtype=np.float64)
        return None
    except Exception as e:
        print(f"[overlay] robot-base lookup failed: {e}")
        return None


def _project_world_points_to_uv(points_world: np.ndarray, w2p: np.ndarray, H: int, W: int):
    """
    points_world : (N, 3) world-frame xyz
    w2p          : (4, 4) world->pixel transform
    Returns:
        uv         : (N, 2) int  — [u_col, v_row] in standard CV convention
        visibility : (N,) bool   — True iff z_cam>0 AND inside the image
    """
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    n = pts.shape[0]
    homo = np.concatenate([pts, np.ones((n, 1))], axis=1)
    proj = (w2p @ homo.T).T
    z = proj[:, 2]
    safe_z = np.where(np.abs(z) > 1e-6, z, 1e-6)
    u = np.round(proj[:, 0] / safe_z).astype(int)
    v = np.round(proj[:, 1] / safe_z).astype(int)
    visible = (z > 0.0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return np.stack([u, v], axis=1), visible


def _draw_gmm_goal_overlay(frame: np.ndarray, uv: np.ndarray, visible: np.ndarray,
                            weights: np.ndarray, radius: int = 2) -> None:
    """
    GMM overlay: ONE filled circle per candidate (not 4), colored blue with
    brightness proportional to weight / max(weight) over the kept candidates.

    - Skips candidates where visible=False (off-screen / behind camera).
    - Sorts ascending by weight so high-weight candidates are drawn last (on top).
    - Caller is expected to have already filtered out below-threshold candidates.

    `frame` is modified IN-PLACE (RGB uint8).
    """
    try:
        import cv2
    except Exception as e:
        print(f"[overlay] disabled mid-episode — cv2 unavailable: {e}")
        return
    if uv.size == 0:
        return
    visible = np.asarray(visible, dtype=bool)
    if not visible.any():
        return
    uvs = uv[visible]
    ws = np.asarray(weights, dtype=np.float64)[visible]
    max_w = float(ws.max()) + 1e-12
    # High-weight on top
    order = np.argsort(ws)
    for idx in order:
        u, v = int(uvs[idx, 0]), int(uvs[idx, 1])
        w_norm = float(ws[idx]) / max_w        # in (0, 1]
        b = int(100 + 155 * w_norm)            # 100..255
        # RGB tuple — frame is RGB. Use a deep blue with brightness scaled.
        cv2.circle(frame, (u, v), radius, (40, 40, b), thickness=-1)


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
    use_gmm_modes: bool = False,
    mode_radius: float = 0.03,
    max_modes: int = 3,
    save_goal_overlay: bool = False,
    gmm_overlay_weight_threshold: float = 0.001,
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

    # ---- Goal-overlay setup (NEW, additive) ----
    frames_overlay = None
    w2p = None
    base_offset_world = None
    # Latest GMM cached for drawing: centroids (N, 3) base-frame, weights (N,) probs.
    current_gmm_centroids_base = None
    current_gmm_weights = None
    if save_goal_overlay:
        H, W = frames[0].shape[:2]
        w2p = _get_agentview_world_to_pixel(env, H, W)
        base_offset_world = _get_robot_base_world_pos(env, "robot0_base")
        if w2p is not None:
            frames_overlay = [frames[0].copy()]
            try:
                gp = np.asarray(obs.get("gripper_pcd", []), dtype=np.float64)
                if gp.ndim == 2 and gp.shape[1] >= 3:
                    gp = gp[:, :3]
                if base_offset_world is not None and gp.size > 0:
                    gp_world = gp + base_offset_world
                    _uv_g, vis_g = _project_world_points_to_uv(gp_world, w2p, H, W)
                    print(f"[overlay] base_offset_world={tuple(base_offset_world.tolist())}  "
                          f"sanity-check gripper_pcd: {int(vis_g.sum())}/{gp.shape[0]} keypoints visible  "
                          f"(weight_threshold={gmm_overlay_weight_threshold})")
                elif base_offset_world is None:
                    print("[overlay] WARNING: no robot0_base body found — projecting as if obs were "
                          "world-frame, which may be wrong.")
            except Exception as _e:
                print(f"[overlay] sanity-check failed (non-fatal): {_e}")

    while step < max_steps:
        # ------------------- HL: PCD -> full GMM distribution ----------------
        scene_pcd_t, gripper_pcd_t = build_hl_inputs_lfd3d(obs, device)
        gmm_means, gmm_probs = infer_articubot_gmm(
            hl_network, scene_pcd_t, gripper_pcd_t, text_embed_t,
        )                                                            # (1, N, 4, 3), (1, N)

        # (GMM goals/weights are reduced-or-not and tiled below, right before
        # injection into the LL obs — see the use_gmm_modes branch. The overlay
        # below still uses the full untiled GMM gmm_means/gmm_probs.)

        # Cache GMM centroids+weights for overlay drawing during the next env-step batch.
        if frames_overlay is not None:
            # centroid = mean over the 4 keypoints  → (N, 3)
            means_np = gmm_means[0].detach().cpu().numpy()   # (N, 4, 3) base frame
            probs_np = gmm_probs[0].detach().cpu().numpy()   # (N,) probabilities
            current_gmm_centroids_base = means_np.mean(axis=1)  # (N, 3)
            current_gmm_weights = probs_np

        # ------------------- LL: image obs + GMM dist -> action --------------
        ll_obs = build_ll_obs_dict(obs, n_obs_steps, device)
        if use_gmm_modes:
            # Halo-collapse the full GMM into discrete modes with the SAME params
            # the dataset generator used, then feed under the keys a modes-trained
            # LL expects. Zero-weight padded modes are auto-masked by the LL's
            # WeightedCrossAttention log(w) bias, so 1/2/3 modes all just work.
            modes_np, mode_w_np = reduce_gmm_to_modes(
                gmm_means.detach().cpu().numpy(),   # (1, N, 4, 3)
                gmm_probs.detach().cpu().numpy(),   # (1, N)
                radius=mode_radius, max_modes=max_modes,
            )                                                        # (1, K, 4, 3), (1, K)
            goals   = torch.from_numpy(modes_np).float().to(device)
            weights = torch.from_numpy(mode_w_np).float().to(device)
            goals_key, weights_key = "gmm_modes", "gmm_mode_weights"
        else:
            goals, weights = gmm_means, gmm_probs                   # (1, N, 4, 3), (1, N)
            goals_key, weights_key = "gmm_all_goals", "gmm_all_weights"

        # Tile over n_obs_steps so the LL sees one entry per obs step.
        ll_obs[goals_key]   = goals.unsqueeze(1).repeat(1, n_obs_steps, 1, 1, 1)
        ll_obs[weights_key] = weights.unsqueeze(1).repeat(1, n_obs_steps, 1)

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
            frame = _agentview_to_uint8_rgb(obs)
            frames.append(frame)

            # Build the overlay frame.
            if frames_overlay is not None:
                overlay = frame.copy()
                if current_gmm_centroids_base is not None and current_gmm_weights is not None:
                    H, W = overlay.shape[:2]
                    # 1. Filter by absolute weight threshold (probabilities, sum to 1).
                    keep_mask = current_gmm_weights >= gmm_overlay_weight_threshold
                    if keep_mask.any():
                        kept_centroids_base = current_gmm_centroids_base[keep_mask]  # (M, 3)
                        kept_weights        = current_gmm_weights[keep_mask]         # (M,)
                        # 2. Base-frame → world-frame.
                        if base_offset_world is not None:
                            kept_centroids_world = kept_centroids_base + base_offset_world
                        else:
                            kept_centroids_world = kept_centroids_base
                        # 3. Project.
                        uv, vis = _project_world_points_to_uv(kept_centroids_world, w2p, H, W)
                        # 4. Draw.
                        _draw_gmm_goal_overlay(overlay, uv, vis, kept_weights)
                frames_overlay.append(overlay)

            total_reward += float(reward)
            step += 1
            if env.is_success().get("task", False):
                success = True
                done = True
            if done or step >= max_steps:
                break

        if success or step >= max_steps:
            break

    return total_reward, success, frames, frames_overlay


def _inference_autocast_context(device, inference_dtype):
    import torch

    if inference_dtype == "fp32":
        return nullcontext()
    if not str(device).startswith("cuda"):
        raise RuntimeError(
            f"--inference_dtype={inference_dtype} requires CUDA; current device is {device}"
        )
    if inference_dtype == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bfloat16 inference is not supported by this CUDA device")
        dtype = torch.bfloat16
    elif inference_dtype == "float16":
        dtype = torch.float16
    else:
        raise ValueError(f"unknown inference dtype: {inference_dtype}")
    return torch.autocast(device_type="cuda", dtype=dtype)


def _make_episode_generators(episode_seeds, device):
    import torch

    generators = []
    for episode_seed in episode_seeds:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(episode_seed))
        generators.append(generator)
    return generators


def _sample_episode_initial_noise(ll_model, generators, active_indices, device):
    import torch

    samples = [
        torch.randn(
            (1, ll_model.action_horizon, ll_model.action_dim),
            dtype=ll_model.dtype,
            device=device,
            generator=generators[int(slot)],
        )[0]
        for slot in active_indices
    ]
    return torch.stack(samples, dim=0)


@contextmanager
def _supply_initial_noise_to_legacy_policy(initial_noise):
    """Supply deterministic batched noise to older FlowMatchingDiT policies.

    The Approach 1 policy predates the explicit ``initial_noise`` argument used
    by the optimized Approach 2 policy. Its predict path makes exactly one
    matching ``torch.randn(B, H, D)`` call. Intercepting that call preserves an
    independent RNG stream per episode without modifying the external LL repo.
    """
    import torch

    original_randn = torch.randn
    expected_shape = tuple(initial_noise.shape)
    consumed = False

    def supplied_randn(*size, **kwargs):
        nonlocal consumed
        if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)):
            requested_shape = tuple(size[0])
        else:
            requested_shape = tuple(size)
        if not consumed and requested_shape == expected_shape:
            consumed = True
            dtype = kwargs.get("dtype", initial_noise.dtype)
            device = kwargs.get("device", initial_noise.device)
            return initial_noise.to(device=device, dtype=dtype)
        return original_randn(*size, **kwargs)

    torch.randn = supplied_randn
    try:
        yield
    finally:
        torch.randn = original_randn
    if not consumed:
        raise RuntimeError(
            "legacy LL policy did not request the expected initial-noise tensor; "
            "its predict_action implementation may have changed"
        )


def _predict_action_with_initial_noise(ll_model, ll_obs, initial_noise):
    """Use the modern explicit API when present, otherwise the legacy shim."""
    import inspect

    parameters = inspect.signature(ll_model.predict_action).parameters
    if "initial_noise" in parameters:
        return ll_model.predict_action(ll_obs, initial_noise=initial_noise)
    with _supply_initial_noise_to_legacy_policy(initial_noise):
        return ll_model.predict_action(ll_obs)


def run_parallel_episodes(
    vector_env,
    hl_network,
    ll_model,
    text_embed_t,
    *,
    n_episodes,
    seed,
    n_obs_steps,
    n_action_steps,
    device,
    inference_dtype,
    save_videos,
    num_video_episodes,
    videos_dir,
    results_f,
):
    """Run parallel simulators and batch both Approach 1 policy forwards."""
    import json
    import torch

    n_envs = vector_env.num_envs
    controller_limits = vector_env.call("get_controller_limits")
    max_dpos, max_drot = controller_limits[0]
    if not all(np.allclose(x, controller_limits[0]) for x in controller_limits[1:]):
        raise RuntimeError(f"worker controller limits differ: {controller_limits}")

    rewards, successes = [], []
    timing = collections.defaultdict(float)
    eval_start = time.perf_counter()

    for chunk_start in range(0, n_episodes, n_envs):
        chunk_stop = min(chunk_start + n_envs, n_episodes)
        n_active = chunk_stop - chunk_start
        episode_indices = list(range(chunk_start, chunk_stop))

        configure_args = []
        for slot in range(n_envs):
            active_slot = slot < n_active
            episode_index = episode_indices[slot] if active_slot else chunk_start
            episode_seed = seed + episode_index
            video_path = None
            if active_slot and save_videos and episode_index < num_video_episodes:
                video_path = str(
                    videos_dir
                    / f"episode_{episode_index + 1:03d}_seed_{episode_seed}_pending.mp4"
                )
            configure_args.append((episode_seed, video_path, active_slot))

        t0 = time.perf_counter()
        vector_env.call_each("configure_episode", args_list=configure_args)
        obs = vector_env.reset()
        timing["environment_seconds"] += time.perf_counter() - t0

        generators = _make_episode_generators(
            (seed + episode_index for episode_index in episode_indices), device
        )
        active = np.zeros(n_envs, dtype=bool)
        active[:n_active] = True

        while np.any(active):
            active_indices = np.flatnonzero(active)
            timing["policy_calls"] += 1
            timing["policy_batch_samples"] += len(active_indices)

            t0 = time.perf_counter()
            scene_pcd_t, gripper_pcd_t = build_batched_hl_inputs_lfd3d(
                obs, active_indices, device
            )
            ll_obs = build_batched_ll_obs_dict(
                obs, active_indices, n_obs_steps, device
            )
            timing["preprocess_transfer_seconds"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            with torch.inference_mode(), _inference_autocast_context(
                device, inference_dtype
            ):
                batch_text_embed = text_embed_t.expand(scene_pcd_t.shape[0], -1)
                gmm_means, gmm_probs = infer_articubot_gmm(
                    hl_network, scene_pcd_t, gripper_pcd_t, batch_text_embed
                )
                ll_obs["gmm_all_goals"] = gmm_means.unsqueeze(1).expand(
                    -1, n_obs_steps, -1, -1, -1
                )
                ll_obs["gmm_all_weights"] = gmm_probs.unsqueeze(1).expand(
                    -1, n_obs_steps, -1
                )
                initial_noise = _sample_episode_initial_noise(
                    ll_model, generators, active_indices, device
                )
                action_dict = _predict_action_with_initial_noise(
                    ll_model, ll_obs, initial_noise
                )
            action_tensor = (
                action_dict["action_pred"]
                if "action_pred" in action_dict
                else action_dict["action"]
            )
            action_raw = action_tensor.detach().to("cpu", dtype=torch.float32).numpy()
            timing["policy_seconds"] += time.perf_counter() - t0

            if action_raw.shape[1] < n_action_steps:
                raise RuntimeError(
                    f"policy returned {action_raw.shape[1]} action steps, "
                    f"but --n_action_steps={n_action_steps}"
                )

            t0 = time.perf_counter()
            eef_quats = np.asarray(obs["robot0_eef_quat"])[active_indices]
            converted = policy_action_batch_to_env_action_vectorized(
                action_raw[:, :n_action_steps], eef_quats, max_dpos, max_drot
            )
            env_actions = np.zeros((n_envs, n_action_steps, 7), dtype=np.float32)
            env_actions[active_indices] = converted
            timing["action_conversion_seconds"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            obs, _reward, done, _info = vector_env.step(env_actions)
            timing["environment_seconds"] += time.perf_counter() - t0
            active &= ~done

        t0 = time.perf_counter()
        worker_results = vector_env.call("get_episode_result")
        timing["video_finalize_seconds"] += time.perf_counter() - t0

        for slot, episode_index in enumerate(episode_indices):
            episode_seed = seed + episode_index
            result = worker_results[slot]
            timing["video_worker_seconds"] += float(result.get("video_seconds", 0.0))
            timing["point_cloud_builds"] += int(result.get("point_cloud_builds", 0))
            timing["point_cloud_skips"] += int(result.get("point_cloud_skips", 0))
            reward = float(result["reward"])
            success = bool(result["success"])
            video_path = result["video"]
            if video_path is not None:
                pending_path = Path(video_path)
                outcome = "success" if success else "failure"
                final_path = pending_path.with_name(
                    pending_path.name.replace("_pending.mp4", f"_{outcome}.mp4")
                )
                pending_path.replace(final_path)
                video_path = str(final_path)

            rewards.append(reward)
            successes.append(success)
            video_tag = f"  video={Path(video_path).name}" if video_path else ""
            print(
                f"Episode {episode_index + 1}/{n_episodes}  seed={episode_seed}  "
                f"reward={reward:.2f}  success={success}{video_tag}"
            )
            results_f.write(
                json.dumps(
                    {
                        "episode": episode_index + 1,
                        "seed": episode_seed,
                        "reward": reward,
                        "success": success,
                        "video": video_path,
                        "video_with_goal_overlay": None,
                    }
                )
                + "\n"
            )
            results_f.flush()

    timing["total_seconds"] = time.perf_counter() - eval_start
    timing["episodes_per_hour"] = (
        3600.0 * n_episodes / max(timing["total_seconds"], 1e-9)
    )
    timing["mean_policy_batch_size"] = (
        timing["policy_batch_samples"] / max(timing["policy_calls"], 1)
    )
    timing["point_cloud_skip_fraction"] = (
        timing["point_cloud_skips"]
        / max(timing["point_cloud_builds"] + timing["point_cloud_skips"], 1)
    )
    return rewards, successes, dict(timing)


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
    parser.add_argument("--num_envs",       type=int, default=4,
        help="Number of spawned MuJoCo environments. Approach 1 point-cloud "
             "construction is CPU/RAM heavy, so 4 is the conservative default.")
    parser.add_argument("--camera_h",       type=int, default=256)
    parser.add_argument("--camera_w",       type=int, default=256)
    parser.add_argument("--output_dir",     type=str, default=None,
        help="Where to save args.json / results.jsonl / summary.json. "
             "Default: outputs_eval_gmm/<HL_dir>__<LL_dir>_<ckpt>/<timestamp>/.")
    parser.add_argument("--save_videos", action=argparse.BooleanOptionalAction, default=True,
        help="Write selected episode mp4s to <output_dir>/media/. Pass --no-save_videos to disable.")
    parser.add_argument("--num_video_episodes", type=int, default=4,
        help="Record only the first N episodes (default: 4) to limit encoding cost.")
    parser.add_argument("--save_goal_overlay_videos", action=argparse.BooleanOptionalAction, default=True,
        help="In addition to the regular video, write a sister mp4 to "
             "<output_dir>/media_with_goal_overlay/ with the GMM goal centroids "
             "projected onto each frame (one point per kept candidate). "
             "Requires --save_videos; pass --no-save_goal_overlay_videos to disable.")
    parser.add_argument("--gmm_overlay_weight_threshold", type=float, default=0.001,
        help="Absolute probability threshold for plotting a GMM candidate in the overlay. "
             "Keeps candidates with weight >= threshold (weights are softmax probs over N anchors). "
             "Lower => more points shown.")
    parser.add_argument("--video_fps", type=int, default=10,
        help="Frame rate for saved mp4s.")
    parser.add_argument(
        "--inference_dtype",
        choices=("fp32", "bfloat16", "float16"),
        default="fp32",
        help="HL/LL autocast precision. FP32 is the benchmark-compatible default.",
    )
    parser.add_argument("--use_gmm_modes", action="store_true", default=False,
        help="Halo-collapse the HL's full per-anchor GMM to discrete modes "
             "(reduce_gmm_to_modes) and feed the LL gmm_modes/gmm_mode_weights "
             "instead of gmm_all_goals/gmm_all_weights. Use when the LL was "
             "trained on the mode-based task (mugcleanup_D1_modes_goal).")
    parser.add_argument("--mode_radius", type=float, default=0.03,
        help="Merge radius (m) for GMM mode reduction. MUST match the value the "
             "LL's training dataset was generated with (default 0.03). "
             "Only used with --use_gmm_modes.")
    parser.add_argument("--max_modes", type=int, default=3,
        help="Max modes kept per step (must match the LL's gmm_modes shape). "
             "Only used with --use_gmm_modes.")
    args = parser.parse_args()

    if args.n_episodes <= 0:
        parser.error("--n_episodes must be positive")
    if args.num_envs <= 0:
        parser.error("--num_envs must be positive")
    if args.num_video_episodes < 0:
        parser.error("--num_video_episodes cannot be negative")
    if args.save_goal_overlay_videos:
        print("[overlay] disabled in parallel mode; regular videos are streamed by workers")

    _prepare_egl_import()

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
    if args.inference_dtype != "fp32" and device == "cpu":
        raise RuntimeError(
            f"--inference_dtype={args.inference_dtype} requires CUDA, but CUDA is unavailable"
        )

    # Spawn CPU-heavy MuJoCo / segmentation / Open3D workers before loading
    # either CUDA model. A dummy env provides spaces without creating another
    # OpenGL context in the parent.
    shape_meta = get_shape_meta(args.camera_h, args.camera_w)
    effective_num_envs = min(args.num_envs, args.n_episodes)
    from equi_diffpo.gym_util.async_vector_env import AsyncVectorEnv

    env_kwargs = {
        "dataset_path": args.dataset_path,
        "shape_meta": shape_meta,
        "camera_h": args.camera_h,
        "camera_w": args.camera_w,
        "n_action_steps": args.n_action_steps,
        "max_steps": args.max_steps,
        "video_fps": args.video_fps,
    }
    env_fn = functools.partial(_make_parallel_env, **env_kwargs)
    dummy_env_fn = functools.partial(
        _make_space_only_env, shape_meta, args.n_action_steps
    )
    print(
        f"[env] starting {effective_num_envs} parallel Approach 1 worker(s); "
        f"policy device={device}, dtype={args.inference_dtype}"
    )
    env = AsyncVectorEnv(
        [env_fn] * effective_num_envs,
        dummy_env_fn=dummy_env_fn,
        shared_memory=True,
        copy=False,
        context="spawn",
    )

    # Load HL (lfd3d ArticubotNetwork).
    print(f"[HL] loading lfd3d ArticubotNetwork from {args.high_level_ckpt}")
    hl_cfg = build_articubot_model_cfg(in_channels=args.hl_in_channels, use_rgb=False)
    try:
        hl_network = load_articubot_high_level(args.high_level_ckpt, hl_cfg, device=device)

    # Text embedding for the HL FiLM block.
        if args.text_embed_cache and os.path.exists(args.text_embed_cache):
            text_embed_np = np.load(args.text_embed_cache).astype(np.float32)
            print(f"[HL] loaded text embedding from {args.text_embed_cache} (shape={text_embed_np.shape})")
        else:
            text_embed_np = np.zeros(1152, dtype=np.float32)
            print("[HL] using zero (1152,) text embedding (matches coffee ckpt's use_text_embed=False)")
        text_embed_t = torch.from_numpy(text_embed_np).float().unsqueeze(0).to(device)

    # Load LL (2D DiT GMM-conditioned).
        print(f"[LL] loading 2D DiT GMM-LL from {args.low_level_exp_dir}/{args.low_level_checkpoint}")
        ll_model, ll_cfg = load_low_level_2d_dit(
            args.low_level_exp_dir, args.low_level_checkpoint, device=device)

    # Sanity-check the LL config: this script only makes sense when the LL was
    # actually trained on the GMM task — print a clear warning otherwise.
        from omegaconf import OmegaConf
        ll_use_gca = OmegaConf.select(ll_cfg, "policy.use_goal_cross_attention", default=False)
        ll_use_wca = OmegaConf.select(ll_cfg, "policy.use_weighted_cross_attention", default=False)
        ll_top_k = OmegaConf.select(ll_cfg, "policy.gmm_top_k", default=None)
        print(f"[LL] use_goal_cross_attention={ll_use_gca}, "
              f"use_weighted_cross_attention={ll_use_wca}, gmm_top_k={ll_top_k}")
        if not (ll_use_gca and ll_use_wca):
            print("[LL][WARN] this LL run was NOT trained with weighted GMM cross-attention. "
                  "The GMM tensors may be ignored.")

    # GMM-modes cross-check: when --use_gmm_modes, the LL must read gmm_modes.
        if args.use_gmm_modes:
            raise NotImplementedError(
                "parallel evaluation currently supports full-GMM Approach 1 runs only; "
                "use the sequential eval_gmm_high_level_2d_dit_low_level.py for --use_gmm_modes"
            )

    # Video setup.
        videos_dir = output_dir / "media"
        if args.save_videos and args.num_video_episodes > 0:
            videos_dir.mkdir(parents=True, exist_ok=True)
            n_videos = min(args.num_video_episodes, args.n_episodes)
            print(
                f"[video] recording first {n_videos} episode(s) -> "
                f"{videos_dir.resolve()} (fps={args.video_fps}, h264 crf=22)"
            )

    # Roll out — stream per-episode results to results.jsonl.
        results_path = output_dir / "results.jsonl"
        with open(results_path, "w") as results_f:
            rewards, successes, timing = run_parallel_episodes(
                env,
                hl_network,
                ll_model,
                text_embed_t,
                n_episodes=args.n_episodes,
                seed=args.seed,
                n_obs_steps=args.n_obs_steps,
                n_action_steps=args.n_action_steps,
                device=device,
                inference_dtype=args.inference_dtype,
                save_videos=args.save_videos,
                num_video_episodes=args.num_video_episodes,
                videos_dir=videos_dir,
                results_f=results_f,
            )
    finally:
        try:
            env.close()
        except Exception as e:
            print(f"[warn] parallel env close failed, ignoring: {e}")
    summary = {
        "n_episodes":   args.n_episodes,
        "num_envs":     effective_num_envs,
        "mean_reward":  float(np.mean(rewards)),
        "std_reward":   float(np.std(rewards)),
        "success_rate": float(np.mean(successes)),
        "successes":    int(sum(successes)),
        "timing":       timing,
        "args":         vars(args),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n--- Summary ---")
    print(f"Mean reward:  {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
    print(f"Success rate: {summary['success_rate']:.2%} ({summary['successes']}/{args.n_episodes})")
    print(f"Throughput:   {timing['episodes_per_hour']:.2f} episodes/hour")
    print(f"PCD skipped:  {timing['point_cloud_skip_fraction']:.1%} of observations")
    print(f"\nSaved: {output_dir.resolve()}")
    print(f"  - args.json")
    print(f"  - results.jsonl")
    print(f"  - summary.json")


if __name__ == "__main__":
    sys.exit(main())
