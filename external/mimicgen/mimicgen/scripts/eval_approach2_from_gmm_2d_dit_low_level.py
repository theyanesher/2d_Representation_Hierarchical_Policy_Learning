#!/usr/bin/env python3
"""
APPROACH 2 eval for MimicGen: the 2D DiT low-level policy ALONE.

DERIVED BY COPY from eval_gmm_high_level_2d_dit_low_level.py, which reproduces
the reference numbers. Everything not listed below is byte-identical to that
script on purpose -- a from-scratch reimplementation (eval_approach2_2d_dit_low_level.py)
scored ~16 points lower and the cause was never fully isolated, so this variant
inherits the proven behaviour instead of re-deriving it.

Differences from the parent script, and ONLY these:
  1. No high-level policy. The Approach 2 LL is never given a goal --
     goal_gripper_pts only supervised its visual representation during training
     via the auxiliary GMM head, so at rollout the policy is self-contained.
     All HL loading, the per-step HL forward, and the gmm_all_goals /
     gmm_all_weights injection are removed.
  2. Depth is requested (camera_depths=True) and registered in shape_meta,
     because Approach 2's RoPE4D trunk unprojects depth to ground its visual
     tokens. The parent script never reads depth.
  3. robomimic's DepthModality processor is overridden so metric depth is not
     clipped to [0, 1] -- see the comment in create_mimicgen_env.
  4. build_ll_obs_dict additionally supplies cam*_depth, cam*_intrinsic,
     cam*_extrinsic and present_gripper_pts. It KEEPS the parent's tiling of a
     single observation across n_obs_steps.
  5. The goal overlay is dropped (there is no goal to draw).

Per step: env obs -> LL(cam0/cam1 image+depth+intrinsic+extrinsic, state,
          present_gripper_pts) -> hybrid_delta
          -> policy_action_batch_to_env_action -> env.step
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
    # No lfd3d here: Approach 2 has no high-level policy, so <lfd3d_repo>/src is
    # not on the path at all (the parent script needed it for ArticubotNetwork).
    _prepend_sys_path(
        args.dit_2d_repo,
        str(dit_2d_parent),
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
            # Approach 2 only: the env populates *_depth solely for keys registered
            # as the "depth" modality here. The parent script omits these.
            "agentview_depth":          {"shape": [1, camera_h, camera_w], "type": "depth"},
            "robot0_eye_in_hand_depth": {"shape": [1, camera_h, camera_w], "type": "depth"},
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
    env_kwargs["camera_depths"]  = True     # Approach 2 unprojects depth

    # ---- metric depth must NOT be clipped to [0, 1] ----------------------
    # robomimic's DepthModality processor is process_frame(channel_dim=1, scale=1.),
    # and process_frame ends with `frame.clip(0.0, 1.0)`. Harmless for RGB
    # (scale=255, so uint8/255 is already <= 1) but the env hands depth to it in
    # METRES, so everything past 1 m is truncated to exactly 1.0. Measured on
    # Coffee_Preperation_D1: the training h5 spans 0.743-3.064 m, the unpatched
    # env returns 0.7435-1.0000. The training loader does
    # process_depth(compress=False) = raw/1000.0 with no clip, so unclipped
    # metres is what the policy was trained on.
    def _depth_processor_keep_metres(obs):
        import robomimic.utils.tensor_utils as TU
        return ObsUtils.batch_image_hwc_to_chw(TU.to_float(obs))

    ObsUtils.DepthModality.set_obs_processor(_depth_processor_keep_metres)

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


# Approach 2: same TILING convention as the parent script (one env observation
# repeated across n_obs_steps), plus the extra keys this policy consumes.
# The tiling is deliberate -- see the module docstring.
def build_ll_obs_dict(obs, n_obs_steps: int, device):
    """LL takes (B=1, T=n_obs_steps, ...) tensors."""
    import torch

    cam0  = np.asarray(obs["agentview_image"], dtype=np.float32)              # (3,H,W) [0,1]
    cam1  = np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.float32)
    d0    = np.asarray(obs["agentview_depth"], dtype=np.float32)              # (1,H,W) metres
    d1    = np.asarray(obs["robot0_eye_in_hand_depth"], dtype=np.float32)
    k0    = np.asarray(obs["agentview_intrinsics"], dtype=np.float32)         # (3,3)
    k1    = np.asarray(obs["robot0_eye_in_hand_intrinsics"], dtype=np.float32)
    e0    = np.asarray(obs["agentview_extrinsics"], dtype=np.float32)         # (4,4) cam->base
    e1    = np.asarray(obs["robot0_eye_in_hand_extrinsics"], dtype=np.float32)
    state = np.asarray(obs["state"], dtype=np.float32)                        # (10,)
    gp    = np.asarray(obs["gripper_pcd"], dtype=np.float32)[:, :3]           # (4,3)

    def tile(x):
        return np.tile(x[None, None], (1, n_obs_steps) + (1,) * x.ndim)

    out = {
        "cam0_image":          tile(cam0),
        "cam1_image":          tile(cam1),
        "cam0_depth":          tile(d0),
        "cam1_depth":          tile(d1),
        "cam0_intrinsic":      tile(k0),
        "cam1_intrinsic":      tile(k1),
        "cam0_extrinsic":      tile(e0),
        "cam1_extrinsic":      tile(e1),
        "state":               tile(state),
        "present_gripper_pts": tile(gp),
    }
    return {k: torch.from_numpy(v).float().to(device) for k, v in out.items()}


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
    ll_model,
    controller,
    n_obs_steps: int,
    n_action_steps: int,
    max_steps: int,
    device: str,
):
    """Approach 2 rollout: no HL, no goal. Stepping logic copied verbatim from
    the parent GMM eval -- same action extraction, same n_action_steps slice,
    same success/termination handling."""
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
        # ------------------- LL: image+depth obs -> action -------------------
        ll_obs = build_ll_obs_dict(obs, n_obs_steps, device)

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
        description="Approach 2 eval: 2D DiT low-level policy alone (no high-level, no goal)")
    parser.add_argument("--dataset_path", type=str, required=True,
        help="HDF5 dataset whose env_meta defines the MimicGen env (e.g. coffee_d0.hdf5)")
    parser.add_argument("--low_level_exp_dir", type=str, required=True,
        help="Hydra output dir of the Approach 2 LL training run "
             "(.hydra/config.yaml inside; expected policy.aux_gmm_loss_weight>0 "
             "and use_goal_cross_attention=False)")
    parser.add_argument("--low_level_checkpoint", type=str, default="epoch_99.ckpt")
    parser.add_argument("--dit_2d_repo", type=str, required=True,
        help="Path to .../Low_Level_and_Inference/diffusion_policy")
    parser.add_argument("--robosuite_root", type=str, default="",
        help="Optional path to a complete robosuite checkout (parent dir of the robosuite/ package)")
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
        ll_tag = Path(args.low_level_exp_dir).name
        ckpt_tag = Path(args.low_level_checkpoint).stem
        args.output_dir = f"outputs_eval_approach2/{ll_tag}_{ckpt_tag}/{ts}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] saving results to {output_dir.resolve()}")
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Bootstrap sys.path + stub training-time deps BEFORE importing torch /
    # loading any foreign ckpt.
    _bootstrap_paths(args)
    _stub_unused_smith_imports()
    # NOT _stub_unused_lfd3d_deps(): that helper imports lfd3d to patch its
    # transitive deps, and Approach 2 has no high-level policy so lfd3d is not
    # on sys.path at all. Calling it here raises ModuleNotFoundError.)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build env and grab the controller (for max_dpos / max_drot during action conversion).
    shape_meta = get_shape_meta(args.camera_h, args.camera_w)
    env, env_meta = create_mimicgen_env(
        args.dataset_path, shape_meta, args.camera_h, args.camera_w)
    inner = env.env if hasattr(env, "env") else env
    controller = inner.robots[0].controller

    # Load LL (2D DiT GMM-conditioned).
    print(f"[LL] loading 2D DiT GMM-LL from {args.low_level_exp_dir}/{args.low_level_checkpoint}")
    ll_model, ll_cfg = load_low_level_2d_dit(
        args.low_level_exp_dir, args.low_level_checkpoint, device=device)

    # Sanity-check the LL config. Approach 2 expects the OPPOSITE of the parent
    # script: the goal must NOT be an input, and the auxiliary GMM head must have
    # been trained (aux_gmm_loss_weight > 0). Warn loudly rather than fail, since
    # a c1=0 ablation is a legitimate thing to evaluate.
    from omegaconf import OmegaConf
    ll_use_gca = OmegaConf.select(ll_cfg, "policy.use_goal_cross_attention", default=False)
    ll_c1      = OmegaConf.select(ll_cfg, "policy.aux_gmm_loss_weight",      default=None)
    ll_vis     = OmegaConf.select(ll_cfg, "policy.visual_encoder_type",      default=None)
    print(f"[LL] visual_encoder_type={ll_vis}, aux_gmm_loss_weight={ll_c1}, "
          f"use_goal_cross_attention={ll_use_gca}")
    if ll_use_gca:
        print("[LL][WARN] this LL was trained WITH goal cross-attention, i.e. it expects a "
              "goal at its input. That is Approach 1, not Approach 2 — use "
              "eval_gmm_high_level_2d_dit_low_level.py instead.")
    if ll_c1 in (None, 0):
        print(f"[LL][WARN] aux_gmm_loss_weight={ll_c1}: this checkpoint had no auxiliary GMM "
              "loss, so nothing shaped its visual representation toward the goal.")

    # Video setup.
    video_recorder_cls = None
    videos_dir = None
    if args.save_videos:
        from equi_diffpo.gym_util.video_recording_wrapper import VideoRecorder as video_recorder_cls
        videos_dir = output_dir / "media"
        videos_dir.mkdir(parents=True, exist_ok=True)
        print(f"[video] all rollouts -> {videos_dir.resolve()} (fps={args.video_fps}, h264 crf=22)")
        # No media_with_goal_overlay/: Approach 2 has no goal to project. The
        # shell wrapper's do_merge creates that directory itself and tolerates it
        # being empty, so the on-disk layout stays compatible with the A1 evals.

    # Roll out — stream per-episode results to results.jsonl.
    rewards, successes = [], []
    results_path = output_dir / "results.jsonl"
    with open(results_path, "w") as results_f:
        for ep in range(args.n_episodes):
            seed = args.seed + ep
            np.random.seed(seed)
            torch.manual_seed(seed)
            r, succ, frames = run_episode(
                env, ll_model, controller,
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
                # Kept (always null) so results.jsonl stays schema-compatible with
                # the A1 evals and the shell do_merge helper.
                "video_with_goal_overlay": None,
            }) + "\n")
            results_f.flush()

    # EnvRobosuite doesn't implement close(); guard so a cleanup error never
    # aborts the run. Without this the AttributeError propagates out of main(),
    # skips summary.json below, and (under `set -e`) kills a multi-seed shell
    # loop before later seeds run.
    _close = getattr(env, "close", None)
    if callable(_close):
        try:
            _close()
        except Exception as e:
            print(f"[warn] env.close() failed, ignoring: {e}")
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
