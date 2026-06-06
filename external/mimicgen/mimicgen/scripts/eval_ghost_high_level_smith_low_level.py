#!/usr/bin/env python3
"""
Hierarchical eval for MimicGen using:
  - HL: lfd3d ArticubotNetwork (PointNet++ with optional FiLM text conditioning,
        GMM head -> per-anchor 4x3 displacement + 1 weight logit). Same as
        eval_ghost_high_level_2d_dit_low_level.py.
  - LL: diffusion_policy_3d.policy.dp3.DP3 (Act3D encoder, UNet noise model)
        trained by RoboGen-sim2real/3d_diffusion_policy/.../train_ddp.py. Same
        as eval_smith.py.

Per step: env obs -> HL(scene_pcd + gripper_pcd, gripper-first concat with mask)
        -> argmax-anchor 4-pt subgoal in world frame
        -> LL(point_cloud, agent_pos, gripper_pcd, goal_gripper_pcd, cat_idx)
        -> hybrid_delta
        -> SMITH's policy_action_batch_to_env_action -> env.step

This file is the GHOST-HL + SMITH-LL hybrid of:
  - eval_ghost_high_level_2d_dit_low_level.py (GHOST HL + 2D DiT LL)
  - eval_smith.py                              (SMITH HL + SMITH LL)

Stub policy:
  - lfd3d training-time deps (pytorch_lightning, pytorch3d, diffusers, ...)
    are stubbed if missing — they're never reached on the HL inference path.
  - train_ddp is NOT stubbed: we use the real TrainDP3Workspace to load LL.
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
# sys.path setup MUST happen before any import that touches lfd3d / SMITH.
# -----------------------------------------------------------------------------
def _prepend_sys_path(*roots):
    """Remove-and-reinsert, so the LAST listed root ends up first on sys.path.
    Needed because both SMITH_on_mimicgen/third_party/robogen and
    RoboGen-sim2real ship `test_PointNet2/`; we must force RoboGen's to win.
    """
    for root in roots:
        root = os.path.abspath(os.path.expanduser(root))
        if not os.path.isdir(root):
            raise FileNotFoundError(f"sys.path prepend target does not exist: {root}")
        if root in sys.path:
            sys.path.remove(root)
        sys.path.insert(0, root)


def _bootstrap_paths(args):
    # lfd3d packages live under <lfd3d_repo>/src/.
    lfd3d_src = Path(args.lfd3d_repo) / "src"

    # SMITH_on_mimicgen root + its vendored external/ deps.
    # This script lives at:
    #   SMITH_on_mimicgen/external/mimicgen/mimicgen/scripts/eval_ghost_high_level_smith_low_level.py
    # parents: [0]=scripts [1]=mimicgen(inner) [2]=mimicgen(outer) [3]=external [4]=SMITH_on_mimicgen
    smith_root = Path(__file__).resolve().parents[4]

    # SMITH training side. RoboGen-sim2real must end up BEFORE third_party/robogen
    # (which is reachable from smith_root) so that `from test_PointNet2.* import ...`
    # resolves to the multitask version. train_ddp.py lives inside the dp3 dir.
    dp3_dir = os.path.join(
        args.smith_robogen_repo,
        "3d_diffusion_policy", "3D-Diffusion-Policy", "3D-Diffusion-Policy",
    )

    # Insertion order: later wins (insert(0, ...) prepends each time).
    _prepend_sys_path(
        str(lfd3d_src),
        str(smith_root),
        str(smith_root / "external" / "robomimic"),
        str(smith_root / "external" / "mimicgen"),
    )
    if args.robosuite_root:
        _prepend_sys_path(args.robosuite_root)
    _prepend_sys_path(
        dp3_dir,
        args.smith_robogen_repo,
    )


# -----------------------------------------------------------------------------
# Stub training-time deps that lfd3d.models.articubot and its transitive
# lfd3d.* imports load at module level. ArticubotNetwork's forward never calls
# any of these — they're only needed by GoalRegressionModule (training/viz).
# Try-import first; only stub if genuinely missing.
#
# NOTE: We do NOT stub `train_ddp` here (cf. eval_ghost_high_level_2d_dit_low_level.py
# which does). The SMITH LL load path uses the real TrainDP3Workspace.
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
       GoalRegressionModule methods we never call.
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

    import importlib
    importlib.import_module("lfd3d")
    importlib.import_module("lfd3d.models")

    _noop = lambda *a, **k: None
    _force_stub_module("lfd3d.models.dino_heatmap", {"calc_pix_metrics": _noop})
    _force_stub_module("lfd3d.models.tax3d", {"calc_pcd_metrics": _noop})
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
# Shape meta. SMITH LL consumes point_cloud / agent_pos / gripper_pcd /
# goal_gripper_pcd; we also keep agentview/eye_in_hand images for video +
# goal-overlay projection.
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
# Identical to the other two eval scripts.
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
# lfd3d ArticubotNetwork loading + inference. Verbatim from
# eval_ghost_high_level_2d_dit_low_level.py.
# -----------------------------------------------------------------------------
def build_articubot_model_cfg(in_channels: int = 4, use_rgb: bool = False):
    """Mirror scripts/run_gmm_on_dataset.py:build_model_cfg for the coffee_task ckpt."""
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
    """Load lfd3d ArticubotNetwork from a Lightning .ckpt. Strips the `network.` prefix."""
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


def infer_articubot_subgoal(
    network,
    scene_pcd_t,    # (1, N, 3)
    gripper_pcd_t,  # (1, K=4, 3)
    text_embed_t,   # (1, 1152) — zeros for the NO_MULTIMODAL coffee ckpt
    argmax_weight: bool = True,
):
    """
    Mirror scripts/run_gmm_on_dataset.py:infer_gmm.
    Returns the sampled subgoal as (1, 4, 3) world-frame xyz.
    """
    import torch

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
        outputs = network(
            net_in, text_embedding=text_embed_t, data_source=["libero_franka"],
        )                                                          # (B, K+N, 13)

    B, KN, _ = outputs.shape
    weights = outputs[:, :, -1]                                    # (B, K+N)
    displacements = outputs[:, :, :-1].reshape(B, KN, 4, 3)        # (B, K+N, 4, 3)
    gaussian_means = anchor_xyz_full[:, :, None, :] + displacements  # (B, K+N, 4, 3)

    # Strip the K=4 gripper anchors — sample only over scene anchors.
    weights = weights[:, K:]
    gaussian_means = gaussian_means[:, K:, :, :]

    probabilities = torch.nn.functional.softmax(weights, dim=1)
    if argmax_weight:
        sampled_idx = torch.argmax(probabilities, dim=1)
    else:
        sampled_idx = torch.multinomial(probabilities, num_samples=1).squeeze(1)
    batch_idx = torch.arange(B, device=outputs.device)
    return gaussian_means[batch_idx, sampled_idx]                  # (B, 4, 3)


# -----------------------------------------------------------------------------
# SMITH LL loading. Verbatim from eval_smith.py:load_smith_low_level — bypasses
# eval_smith_utils.load_low_level_policy whose hydra.initialize relative path
# doesn't resolve under SMITH_on_mimicgen/.
# -----------------------------------------------------------------------------
def load_smith_low_level(exp_dir: str, ckpt_name: str, device: str = "cuda"):
    """Load diffusion_policy_3d.policy.dp3.DP3 from a SMITH training run dir.
    Returns the EMA model if use_ema, else the raw model. Always .eval()."""
    from omegaconf import OmegaConf
    from train_ddp import TrainDP3Workspace

    cfg_path = Path(exp_dir) / ".hydra" / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"LL hydra config missing: {cfg_path}")
    cfg = OmegaConf.load(str(cfg_path))

    # The training run's saved config preserves `load_policy_path`. We're about
    # to overwrite weights via load_checkpoint, so the pretrained init is moot.
    cfg.load_policy_path = None

    workspace = TrainDP3Workspace(cfg)
    ckpt_path = Path(exp_dir) / "checkpoints" / ckpt_name
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"LL checkpoint missing: {ckpt_path}")
    workspace.load_checkpoint(path=str(ckpt_path))

    if OmegaConf.select(workspace.cfg, "training.use_ema", default=False):
        policy = copy.deepcopy(workspace.ema_model)
    else:
        policy = copy.deepcopy(workspace.model)
    policy.eval()
    if hasattr(policy, "reset"):
        policy.reset()
    return policy.to(device), cfg


# -----------------------------------------------------------------------------
# Canonical Franka-gripper 4-keypoint template (lifted from articubot_util.py).
# The env's runtime gripper_pcd is built from this template, and SMITH LL was
# trained expecting index 3 to be the canonical grasp_center. But the lfd3d
# trainer (lfd3d/models/articubot.py:extract_gt_4_points) replaces the GT
# 4th keypoint with `midpoint(index 0, index 1)` before computing the loss
# (because NpyDataset's GRIPPER_IDX = [0, 1, 2]). So GHOST's output index 3
# is midpoint(top, right), not grasp_center. The two helpers below reconstruct
# the canonical grasp_center from indices [top, right, left] via Kabsch
# alignment, so the resulting 4-tuple is what SMITH LL expects.
# -----------------------------------------------------------------------------
_TEMPLATE_GRIPPER_PCD = np.array([
    [ 0.5648266,   0.05482348,  0.34434554],   # 0 = top
    [ 0.5642125,   0.02702148,  0.2877661 ],   # 1 = right finger (open template)
    [ 0.53906703,  0.01263776,  0.38347825],   # 2 = left finger  (open template)
    [ 0.54250515, -0.00441092,  0.32957944],   # 3 = grasp center
], dtype=np.float64)
_TEMPLATE_RIGHT_CLOSED = np.array([0.55415434, 0.02126799, 0.32605097], dtype=np.float64)
_TEMPLATE_LEFT_CLOSED  = np.array([0.54912525, 0.01839125, 0.34519340], dtype=np.float64)
_TEMPLATE_CLOSED_ANGLE = 2.6652539383870777e-05
_TEMPLATE_OPEN_ANGLE   = 0.04


def _canonical_template_at_finger_angle(joint_angle: float) -> np.ndarray:
    """Return (4, 3) canonical Franka template with fingers interpolated to joint_angle."""
    tpl = _TEMPLATE_GRIPPER_PCD.copy()
    span = _TEMPLATE_OPEN_ANGLE - _TEMPLATE_CLOSED_ANGLE
    frac = (joint_angle - _TEMPLATE_CLOSED_ANGLE) / span
    tpl[1] = _TEMPLATE_RIGHT_CLOSED + (_TEMPLATE_GRIPPER_PCD[1] - _TEMPLATE_RIGHT_CLOSED) * frac
    tpl[2] = _TEMPLATE_LEFT_CLOSED  + (_TEMPLATE_GRIPPER_PCD[2] - _TEMPLATE_LEFT_CLOSED)  * frac
    return tpl


def _estimate_finger_angle_from_right_left_dist(right_to_left_dist: float) -> float:
    """Recover gripper joint angle from the right-to-left finger distance."""
    open_tpl = _canonical_template_at_finger_angle(_TEMPLATE_OPEN_ANGLE)
    closed_tpl = _canonical_template_at_finger_angle(_TEMPLATE_CLOSED_ANGLE)
    open_d = float(np.linalg.norm(open_tpl[1] - open_tpl[2]))
    closed_d = float(np.linalg.norm(closed_tpl[1] - closed_tpl[2]))
    frac = float(np.clip((right_to_left_dist - closed_d) / (open_d - closed_d), 0.0, 1.0))
    return _TEMPLATE_CLOSED_ANGLE + frac * (_TEMPLATE_OPEN_ANGLE - _TEMPLATE_CLOSED_ANGLE)


def _kabsch_rigid_transform(src: np.ndarray, dst: np.ndarray):
    """Kabsch: find R (3,3) and t (3,) minimizing ||R @ src + t - dst||² rowwise."""
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    H = src_c.T @ dst_c
    U, _, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = dst_mean - R @ src_mean
    return R, t


def fix_ghost_subgoal_for_smith_ll(ghost_subgoal_np: np.ndarray) -> np.ndarray:
    """
    Replace GHOST's index-3 keypoint (which lfd3d trains as midpoint(0, 1))
    with the canonical grasp_center implied by indices [top, right, left].

    Args:
        ghost_subgoal_np: (4, 3) GHOST HL output, in robot-base frame.
    Returns:
        (4, 3) float32 with indices 0, 1, 2 unchanged and index 3 replaced by
        the canonical grasp_center from Kabsch alignment to the Franka template.
    """
    pts = np.asarray(ghost_subgoal_np, dtype=np.float64).reshape(4, 3)
    finger_dist = float(np.linalg.norm(pts[1] - pts[2]))
    joint_angle = _estimate_finger_angle_from_right_left_dist(finger_dist)
    template = _canonical_template_at_finger_angle(joint_angle)
    R, t = _kabsch_rigid_transform(template[:3], pts[:3])
    grasp_center_pred = R @ template[3] + t
    fixed = pts.copy()
    fixed[3] = grasp_center_pred
    return fixed.astype(np.float32)


# -----------------------------------------------------------------------------
# Obs-dict builders.
# -----------------------------------------------------------------------------
def _maybe_unwrap(obs):
    if isinstance(obs, dict) and "obs" in obs and not {"agentview_image", "state"} & obs.keys():
        return obs["obs"]
    return obs


def build_hl_inputs_lfd3d(obs, device):
    """Pull scene_pcd + gripper_pcd from env obs, slice to xyz only, add batch dim."""
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


def build_ll_obs_dict_smith(obs, subgoal_4x3, n_obs_steps: int, device):
    """
    SMITH LL takes (B=1, T=n_obs_steps, ...) tensors:
      - point_cloud      : (1, T, N, 3)
      - agent_pos        : (1, T, 10)
      - gripper_pcd      : (1, T, 4, 3)
      - goal_gripper_pcd : (1, T, 4, 3)
    Tile single-step env obs across T.
    """
    import torch
    pc = np.asarray(obs["point_cloud"])
    if pc.ndim == 2 and pc.shape[1] >= 3:
        pc = pc[:, :3]
    gp = np.asarray(obs["gripper_pcd"])
    if gp.ndim == 2 and gp.shape[1] >= 3:
        gp = gp[:, :3]
    state = np.asarray(obs["state"], dtype=np.float32)

    pc_b    = np.tile(pc[None, None],    (1, n_obs_steps, 1, 1))   # (1, T, N, 3)
    gp_b    = np.tile(gp[None, None],    (1, n_obs_steps, 1, 1))   # (1, T, 4, 3)
    state_b = np.tile(state[None, None], (1, n_obs_steps, 1))      # (1, T, 10)
    subgoal_b = subgoal_4x3.unsqueeze(1).repeat(1, n_obs_steps, 1, 1)

    return {
        "point_cloud":      torch.from_numpy(pc_b).float().to(device),
        "agent_pos":        torch.from_numpy(state_b).float().to(device),
        "gripper_pcd":      torch.from_numpy(gp_b).float().to(device),
        "goal_gripper_pcd": subgoal_b.to(device),
    }


def _agentview_to_uint8_rgb(obs):
    img = np.asarray(obs["agentview_image"])
    frame = np.transpose(img, (1, 2, 0))
    frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


# -----------------------------------------------------------------------------
# Goal-overlay helpers. Lifted verbatim from the other two eval scripts.
# -----------------------------------------------------------------------------
def _get_agentview_world_to_pixel(env, camera_h: int, camera_w: int):
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


_GOAL_KP_COLORS_RGB = (
    (255,  50,  50),
    ( 50, 255,  50),
    ( 50,  50, 255),
    (255, 220,  20),
)
_GOAL_LINK_COLOR_RGB = (255, 255, 255)


def _draw_goal_overlay(frame, uv, visible, radius=4, draw_links=True):
    try:
        import cv2
    except Exception as e:
        print(f"[overlay] disabled mid-episode — cv2 unavailable: {e}")
        return
    pts_visible = []
    for i, ((u, v), vis) in enumerate(zip(uv, visible)):
        if not vis:
            continue
        color = _GOAL_KP_COLORS_RGB[i % len(_GOAL_KP_COLORS_RGB)]
        cv2.circle(frame, (int(u), int(v)), radius, color, thickness=-1)
        cv2.circle(frame, (int(u), int(v)), radius + 1, (0, 0, 0), thickness=1)
        pts_visible.append((int(u), int(v)))
    if draw_links and len(pts_visible) >= 2:
        for i in range(len(pts_visible)):
            cv2.line(frame, pts_visible[i], pts_visible[(i + 1) % len(pts_visible)],
                     _GOAL_LINK_COLOR_RGB, thickness=1)


# -----------------------------------------------------------------------------
# Single episode rollout.
# -----------------------------------------------------------------------------
def run_episode(
    env,
    hl_network,
    ll_model,
    text_embed_t,
    cat_idx: int,
    controller,
    n_obs_steps: int,
    n_action_steps: int,
    max_steps: int,
    device: str,
    argmax_weight: bool = True,
    save_goal_overlay: bool = False,
):
    import torch
    from eval_smith_utils import policy_action_batch_to_env_action, low_level_policy_infer

    obs = _maybe_unwrap(env.reset())

    max_dpos = float(controller.output_max[0])
    max_drot = float(controller.output_max[3])

    total_reward = 0.0
    success = False
    step = 0
    frames = [_agentview_to_uint8_rgb(obs)]

    frames_overlay = None
    w2p = None
    base_offset_world = None
    current_subgoal_np = None
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
                    uv_g, vis_g = _project_world_points_to_uv(gp_world, w2p, H, W)
                    n_vis = int(vis_g.sum())
                    print(f"[overlay] base_offset_world={tuple(base_offset_world.tolist())}  "
                          f"sanity-check gripper_pcd: {n_vis}/{gp.shape[0]} keypoints visible "
                          f"(first uv={tuple(uv_g[0].tolist()) if uv_g.shape[0] else None})")
                elif base_offset_world is None:
                    print("[overlay] WARNING: no robot0_base body found — projecting as if obs were "
                          "world-frame, which may be wrong.")
            except Exception as _e:
                print(f"[overlay] sanity-check failed (non-fatal): {_e}")

    while step < max_steps:
        # ------------------- HL (GHOST/lfd3d): PCD -> 4-pt subgoal -----------
        scene_pcd_t, gripper_pcd_t = build_hl_inputs_lfd3d(obs, device)
        subgoal = infer_articubot_subgoal(
            hl_network, scene_pcd_t, gripper_pcd_t, text_embed_t,
            argmax_weight=argmax_weight,
        )                                                            # (1, 4, 3)

        # GHOST/lfd3d trains its 4th keypoint as midpoint(top, right_finger),
        # not the canonical grasp_center. SMITH LL was trained expecting the
        # grasp_center at index 3 (and uses all 4 keypoints to recover the
        # gripper's pos+orient). Reconstruct the canonical 4-tuple from the
        # well-supervised [top, right, left] subset before handing off.
        subgoal_raw_np = subgoal[0].detach().cpu().numpy()        # (4, 3)
        subgoal_fixed_np = fix_ghost_subgoal_for_smith_ll(subgoal_raw_np)
        subgoal = torch.from_numpy(subgoal_fixed_np).float().to(device).unsqueeze(0)

        # --- DIAG: print pre-fix subgoal, post-fix subgoal, and obs ---
        # Pre-fix should show the lfd3d midpoint(0,1) bug (03≈13≈half of 01);
        # post-fix should match the obs template within finger-opening tolerance.
        try:
            sg = subgoal_raw_np
            sf = subgoal_fixed_np
            gp = np.asarray(obs["gripper_pcd"], dtype=np.float64)
            if gp.ndim == 2 and gp.shape[1] >= 3:
                gp = gp[:, :3]
            def _pd(p):
                return (
                    float(np.linalg.norm(p[0] - p[1])),
                    float(np.linalg.norm(p[0] - p[2])),
                    float(np.linalg.norm(p[1] - p[2])),
                    float(np.linalg.norm(p[0] - p[3])),
                    float(np.linalg.norm(p[1] - p[3])),
                    float(np.linalg.norm(p[2] - p[3])),
                )
            d01,d02,d12,d03,d13,d23 = _pd(sg)
            f01,f02,f12,f03,f13,f23 = _pd(sf)
            e01,e02,e12,e03,e13,e23 = _pd(gp)
            print(f"[diag step={step:03d}] raw subgoal: "
                  f"01={d01:.4f} 02={d02:.4f} 12={d12:.4f} "
                  f"03={d03:.4f} 13={d13:.4f} 23={d23:.4f}")
            print(f"[diag step={step:03d}] fix subgoal: "
                  f"01={f01:.4f} 02={f02:.4f} 12={f12:.4f} "
                  f"03={f03:.4f} 13={f13:.4f} 23={f23:.4f}")
            print(f"[diag step={step:03d}] obs        : "
                  f"01={e01:.4f} 02={e02:.4f} 12={e12:.4f} "
                  f"03={e03:.4f} 13={e13:.4f} 23={e23:.4f}")
        except Exception as _e:
            print(f"[diag] pair-dist check failed (non-fatal): {_e}")

        if frames_overlay is not None:
            current_subgoal_np = subgoal[0].detach().cpu().numpy()    # (4, 3)

        # ------------------- LL (SMITH/DP3): PCD obs + subgoal -> action -----
        ll_obs = build_ll_obs_dict_smith(obs, subgoal, n_obs_steps, device)
        with torch.no_grad():
            action_raw_t = low_level_policy_infer(
                ll_obs["point_cloud"],
                ll_obs["agent_pos"],
                ll_obs["goal_gripper_pcd"],
                ll_obs["gripper_pcd"],
                ll_model,
                cat_idx=cat_idx,
            )                                                       # (B, horizon, 10)
        action_raw = action_raw_t.detach().cpu().numpy()

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

            if frames_overlay is not None:
                overlay = frame.copy()
                if current_subgoal_np is not None:
                    H, W = overlay.shape[:2]
                    if base_offset_world is not None:
                        pts_world = current_subgoal_np + base_offset_world
                    else:
                        pts_world = current_subgoal_np
                    uv, vis = _project_world_points_to_uv(pts_world, w2p, H, W)
                    _draw_goal_overlay(overlay, uv, vis)
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


# -----------------------------------------------------------------------------
# Entrypoint.
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical eval: lfd3d ArticubotNetwork HL + SMITH DP3 LL on a MimicGen task")
    parser.add_argument("--dataset_path", type=str, required=True,
        help="HDF5 dataset whose env_meta defines the MimicGen env (e.g. coffee_d2.hdf5)")
    parser.add_argument("--high_level_ckpt", type=str, required=True,
        help="Path to a lfd3d ArticubotNetwork Lightning .ckpt")
    parser.add_argument("--low_level_exp_dir", type=str, required=True,
        help="SMITH LL training run dir (must contain .hydra/config.yaml and checkpoints/)")
    parser.add_argument("--low_level_checkpoint", type=str, default="latest.ckpt")
    parser.add_argument("--lfd3d_repo", type=str, required=True,
        help="Path to lfd3d/lfd3d/ (provides lfd3d.models.articubot.ArticubotNetwork via src/)")
    parser.add_argument("--smith_robogen_repo", type=str, required=True,
        help="Path to RoboGen-sim2real (provides test_PointNet2.* and the dp3 train_ddp)")
    parser.add_argument("--robosuite_root", type=str, default="",
        help="Optional path to a complete robosuite checkout (parent dir of the robosuite/ package)")
    parser.add_argument("--text_embed_cache", type=str, default="",
        help="Optional .npy with a (1152,) text embedding for the HL FiLM block. "
             "Default: zeros — matches the NO_MULTIMODAL coffee ckpt's training.")
    parser.add_argument("--hl_in_channels", type=int, default=4,
        help="ArticubotNetwork in_channels. 4 = xyz + 1 mask (use_rgb=False).")
    parser.add_argument("--hl_argmax_weight", type=int, default=1,
        help="1 = argmax over softmaxed scene anchor weights; 0 = multinomial sample.")
    parser.add_argument("--cat_idx", type=int, default=0,
        help="SigLIP category index for SMITH LL conditioning (0 = 'open the storage furniture')")
    parser.add_argument("--n_episodes",     type=int, default=10)
    parser.add_argument("--max_steps",      type=int, default=400)
    parser.add_argument("--seed",           type=int, default=100000)
    parser.add_argument("--n_obs_steps",    type=int, default=2)
    parser.add_argument("--n_action_steps", type=int, default=8)
    parser.add_argument("--camera_h",       type=int, default=256)
    parser.add_argument("--camera_w",       type=int, default=256)
    parser.add_argument("--output_dir",     type=str, default=None,
        help="Where to save args.json / results.jsonl / summary.json. "
             "Default: outputs_eval_ghost_smith/<HL>__<LL>_<ckpt>/<timestamp>/.")
    parser.add_argument("--save_videos", action=argparse.BooleanOptionalAction, default=True,
        help="Write an mp4 of every episode. Pass --no-save-videos to disable.")
    parser.add_argument("--save_goal_overlay_videos", action=argparse.BooleanOptionalAction, default=True,
        help="Also write a sister mp4 with the HL 4-keypoint subgoal projected onto each frame.")
    parser.add_argument("--video_fps", type=int, default=10)
    args = parser.parse_args()

    import json
    from datetime import datetime
    if args.output_dir is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        hl_tag = Path(args.high_level_ckpt).stem
        ll_tag = Path(args.low_level_exp_dir).name
        ckpt_tag = Path(args.low_level_checkpoint).stem
        args.output_dir = f"outputs_eval_ghost_smith/{hl_tag}__{ll_tag}_{ckpt_tag}/{ts}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] saving results to {output_dir.resolve()}")
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Bootstrap sys.path + stub training-time deps BEFORE importing torch /
    # loading any foreign ckpt.
    _bootstrap_paths(args)
    _stub_unused_lfd3d_deps()

    # PROJECT_DIR is required by diffusion_policy_3d.policy.dp3.DP3.__init__ —
    # it loads SigLIP features from ${PROJECT_DIR}/siglip_text_features_w_pick_and_place_w_grasp_and_lift.pt.
    if "PROJECT_DIR" not in os.environ:
        raise RuntimeError(
            "PROJECT_DIR env var is required (SMITH DP3 loads SigLIP from it). "
            "Set it to the directory containing siglip_text_features_w_pick_and_place_w_grasp_and_lift.pt."
        )

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
        print("[HL] using zero (1152,) text embedding (matches NO_MULTIMODAL coffee ckpt's training)")
    text_embed_t = torch.from_numpy(text_embed_np).float().unsqueeze(0).to(device)  # (1, 1152)

    # Load LL (SMITH DP3).
    print(f"[LL] loading SMITH DP3 from {args.low_level_exp_dir}/{args.low_level_checkpoint}")
    ll_model, _ll_cfg = load_smith_low_level(
        args.low_level_exp_dir, args.low_level_checkpoint, device=device)

    # Video setup.
    video_recorder_cls = None
    videos_dir = None
    videos_dir_overlay = None
    save_overlay = bool(args.save_videos and args.save_goal_overlay_videos)
    if args.save_videos:
        from equi_diffpo.gym_util.video_recording_wrapper import VideoRecorder as video_recorder_cls
        videos_dir = output_dir / "media"
        videos_dir.mkdir(parents=True, exist_ok=True)
        print(f"[video] all rollouts -> {videos_dir.resolve()} (fps={args.video_fps}, h264 crf=22)")
        if save_overlay:
            videos_dir_overlay = output_dir / "media_with_goal_overlay"
            videos_dir_overlay.mkdir(parents=True, exist_ok=True)
            print(f"[overlay] goal-overlay rollouts -> {videos_dir_overlay.resolve()}")

    # Roll out — stream per-episode results to results.jsonl.
    rewards, successes = [], []
    results_path = output_dir / "results.jsonl"
    with open(results_path, "w") as results_f:
        for ep in range(args.n_episodes):
            seed = args.seed + ep
            np.random.seed(seed)
            torch.manual_seed(seed)
            r, succ, frames, frames_overlay = run_episode(
                env, hl_network, ll_model, text_embed_t, args.cat_idx,
                controller,
                args.n_obs_steps, args.n_action_steps, args.max_steps,
                device=device,
                argmax_weight=bool(args.hl_argmax_weight),
                save_goal_overlay=save_overlay,
            )

            video_path = None
            overlay_video_path = None
            if args.save_videos:
                outcome_tag = "success" if succ else "failure"
                video_path = videos_dir / f"episode_{ep + 1:03d}_seed_{seed}_{outcome_tag}.mp4"
                recorder = video_recorder_cls.create_h264(fps=args.video_fps, crf=22)
                recorder.start(str(video_path))
                for frame in frames:
                    recorder.write_frame(frame)
                recorder.stop()

                if save_overlay and videos_dir_overlay is not None and frames_overlay is not None:
                    overlay_video_path = videos_dir_overlay / f"episode_{ep + 1:03d}_seed_{seed}_{outcome_tag}.mp4"
                    rec_ov = video_recorder_cls.create_h264(fps=args.video_fps, crf=22)
                    rec_ov.start(str(overlay_video_path))
                    for frame in frames_overlay:
                        rec_ov.write_frame(frame)
                    rec_ov.stop()

            rewards.append(r)
            successes.append(succ)
            video_tag = f"  video={video_path.name}" if video_path is not None else ""
            overlay_tag = f"  overlay={overlay_video_path.name}" if overlay_video_path is not None else ""
            print(f"Episode {ep + 1}/{args.n_episodes}  seed={seed}  reward={r:.2f}  success={succ}{video_tag}{overlay_tag}")
            results_f.write(json.dumps({
                "episode": ep + 1, "seed": seed,
                "reward": float(r), "success": bool(succ),
                "video": str(video_path) if video_path is not None else None,
                "video_with_goal_overlay": str(overlay_video_path) if overlay_video_path is not None else None,
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


if __name__ == "__main__":
    sys.exit(main())
