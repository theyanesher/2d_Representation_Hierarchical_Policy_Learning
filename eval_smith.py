#!/usr/bin/env python3
"""
Hierarchical eval for MimicGen using SMITH HL + SMITH LL.

  - HL: PointNet2_super_multitask from RoboGen-sim2real (SigLIP-conditioned,
        LayerNorm, two heads: per-anchor 4x3 displacement + 1 weight logit).
        Loaded via eval_smith_utils.load_multitask_high_level_model (reads
        config.json next to the .pth).
  - LL: diffusion_policy_3d.policy.dp3.DP3 (Act3D encoder, UNet noise model)
        trained by RoboGen-sim2real/3d_diffusion_policy/.../train_ddp.py.
        Loaded directly from <exp_dir>/.hydra/config.yaml + checkpoints/<ckpt>.

Per step:
  env_obs -> HL(scene_pcd + gripper_pcd, cat_embedding=SigLIP[cat_idx])
          -> argmax-anchor 4-pt subgoal in world frame
          -> LL(point_cloud, agent_pos, gripper_pcd, goal_gripper_pcd, cat_idx)
          -> hybrid_delta action
          -> policy_action_batch_to_env_action -> env.step

Structure mirrors eval_ghost_high_level_2d_dit_low_level.py. The HL/LL load
and infer functions are swapped to SMITH equivalents; the rest (env build,
EGL bootstrap, action conversion, video saving, results.jsonl) is unchanged.
"""

import argparse
import collections
import copy
import os
import sys
from pathlib import Path

import numpy as np

# MUST run before any direct or transitive `import mujoco` / robosuite import.
import OpenGL.EGL.EXT.device_base  # noqa: F401  (side-effect import only)


# -----------------------------------------------------------------------------
# sys.path setup MUST happen before any import that touches the SMITH training
# packages (test_PointNet2, diffusion_policy_3d, train_ddp).
# -----------------------------------------------------------------------------
def _prepend_sys_path(*roots):
    for root in roots:
        root = os.path.abspath(os.path.expanduser(root))
        if not os.path.isdir(root):
            raise FileNotFoundError(f"sys.path prepend target does not exist: {root}")
        if root in sys.path:
            sys.path.remove(root)
        sys.path.insert(0, root)


def _bootstrap_paths(args):
    # SMITH_on_mimicgen root + its vendored external/ deps (robomimic, mimicgen).
    smith_root = Path(__file__).resolve().parent
    _prepend_sys_path(
        str(smith_root),
        str(smith_root / "external" / "robomimic"),
        str(smith_root / "external" / "mimicgen"),
    )
    if args.robosuite_root:
        _prepend_sys_path(args.robosuite_root)

    # SMITH training side. Order matters: prepend RoboGen-sim2real BEFORE
    # third_party/robogen (which is also on sys.path via smith_root) so that
    # `from test_PointNet2.model_invariant import PointNet2_super_multitask`
    # resolves to the real multitask class, not the cut-down version in
    # third_party/robogen/test_PointNet2/.
    dp3_dir = os.path.join(
        args.smith_robogen_repo,
        "3d_diffusion_policy", "3D-Diffusion-Policy", "3D-Diffusion-Policy",
    )
    _prepend_sys_path(
        dp3_dir,                # for `from train_ddp import TrainDP3Workspace`
        args.smith_robogen_repo,  # for `test_PointNet2.*`, `scripts.datasets.*`
    )


# -----------------------------------------------------------------------------
# SMITH HL loading + inference.
# Wraps eval_smith_utils.load_multitask_high_level_model and
# eval_smith_utils.infer_multitask_high_level_model.
# -----------------------------------------------------------------------------
def load_smith_high_level(ckpt_path: str, device: str = "cuda"):
    """Load PointNet2_super_multitask. Returns (model, training_args)."""
    from eval_smith_utils import load_multitask_high_level_model
    model, train_args = load_multitask_high_level_model(ckpt_path)
    model.eval()
    return model.to(device), train_args


def load_siglip_cat_embedding(project_dir: str, cat_idx: int, device: str = "cuda"):
    """
    Return the (768,) SigLIP text embedding for `cat_idx` from
    <project_dir>/siglip_text_features_w_pick_and_place_w_grasp_and_lift.pt.

    The file is a dict with keys ['keys', 'values']; values is (16, 768) float32.
    cat_idx=0 corresponds to 'open the storage furniture' — the default for
    paths that don't match any category substring in dataset_from_disk.py
    (e.g. Coffee_D2).
    """
    import torch
    pt_path = Path(project_dir) / "siglip_text_features_w_pick_and_place_w_grasp_and_lift.pt"
    if not pt_path.is_file():
        raise FileNotFoundError(f"SigLIP features missing: {pt_path}")
    data = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    values = data["values"] if isinstance(data, dict) else data
    return values[cat_idx].float().to(device)  # (768,)


def infer_smith_subgoal(
    hl_model,
    scene_pcd_t,    # (1, N, 3)
    gripper_pcd_t,  # (1, 4, 3)
    cat_embedding,  # (768,) on device
):
    """
    Mirror articubot_pcd_runner.py:307-316. Concat scene + gripper along the
    point axis (scene first), call infer_multitask_high_level_model which
    permutes to (B, 3, N+4), runs the network, samples by argmax over scene
    anchor weights, returns the (B, 4, 3) world-frame subgoal.
    """
    import torch
    from eval_smith_utils import infer_multitask_high_level_model

    inputs = torch.cat([scene_pcd_t, gripper_pcd_t], dim=1)  # (B, N+4, 3)
    subgoal = infer_multitask_high_level_model(
        inputs, hl_model,
        cat_embedding=cat_embedding,
        high_level_args=None,
        extra=None,
    )
    return subgoal  # (B, 4, 3)


# -----------------------------------------------------------------------------
# SMITH LL loading.
# We do NOT use eval_smith_utils.load_low_level_policy — its hydra.initialize
# relative path resolves to a non-existent dir in SMITH_on_mimicgen/. Instead
# load OmegaConf config directly and instantiate the workspace ourselves.
# -----------------------------------------------------------------------------
def load_smith_low_level(exp_dir: str, ckpt_name: str, device: str = "cuda"):
    """
    Load diffusion_policy_3d.policy.dp3.DP3 from a SMITH training run dir.
    Returns the EMA model if use_ema, else the raw model. Always .eval().
    """
    import copy
    from omegaconf import OmegaConf
    # train_ddp lives at <smith_robogen>/3d_diffusion_policy/.../train_ddp.py,
    # already on sys.path via _bootstrap_paths.
    from train_ddp import TrainDP3Workspace

    cfg_path = Path(exp_dir) / ".hydra" / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"LL hydra config missing: {cfg_path}")
    cfg = OmegaConf.load(str(cfg_path))

    # The training run's saved config preserves `load_policy_path` (the
    # pretrained ckpt used as init). On a fresh machine that path likely
    # doesn't exist — and we're about to overwrite weights via load_checkpoint
    # anyway, so the pretrained init is moot. Null it out to skip the
    # FileNotFoundError in TrainDP3Workspace.__init__ -> load_policy.
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
# Shape meta. The SMITH LL only consumes point_cloud / agent_pos /
# gripper_pcd / goal_gripper_pcd — but we keep agentview_image in env obs
# for video saving + goal-overlay projection.
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
# Env construction. Identical to ghost script.
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
# Obs builders.
# -----------------------------------------------------------------------------
def _maybe_unwrap(obs):
    if isinstance(obs, dict) and "obs" in obs and not {"agentview_image", "state"} & obs.keys():
        return obs["obs"]
    return obs


def build_hl_inputs(obs, device):
    """
    Pull scene_pcd + gripper_pcd from env obs, slice to xyz only, add batch dim.
    Returns scene_pcd_t (1, N, 3) and gripper_pcd_t (1, 4, 3).
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


def build_ll_obs_dict(obs, subgoal_4x3, n_obs_steps: int, device):
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

    # subgoal_4x3 is (1, 4, 3); tile to (1, T, 4, 3).
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
# Goal-overlay helpers (additive — used only when --save_goal_overlay_videos).
# Lifted verbatim from eval_ghost_high_level_2d_dit_low_level.py.
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
    hl_model,
    ll_model,
    cat_embedding,
    cat_idx,
    controller,
    n_obs_steps: int,
    n_action_steps: int,
    max_steps: int,
    device: str,
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
        # ------------------- HL: PCD -> 4-pt subgoal -------------------------
        scene_pcd_t, gripper_pcd_t = build_hl_inputs(obs, device)
        subgoal = infer_smith_subgoal(hl_model, scene_pcd_t, gripper_pcd_t, cat_embedding)  # (1, 4, 3)

        if frames_overlay is not None:
            current_subgoal_np = subgoal[0].detach().cpu().numpy()  # (4, 3)

        # ------------------- LL: PCD obs + subgoal -> action -----------------
        ll_obs = build_ll_obs_dict(obs, subgoal, n_obs_steps, device)
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
        description="Hierarchical eval: SMITH HL + SMITH LL on a MimicGen task")
    parser.add_argument("--dataset_path", type=str, required=True,
        help="HDF5 dataset whose env_meta defines the MimicGen env (e.g. coffee_d2.hdf5)")
    parser.add_argument("--high_level_ckpt", type=str, required=True,
        help="Path to SMITH HL .pth (config.json must sit next to it)")
    parser.add_argument("--low_level_exp_dir", type=str, required=True,
        help="SMITH LL training run dir (must contain .hydra/config.yaml and checkpoints/)")
    parser.add_argument("--low_level_checkpoint", type=str, default="latest.ckpt",
        help="Filename inside <low_level_exp_dir>/checkpoints/")
    parser.add_argument("--smith_robogen_repo", type=str, required=True,
        help="Path to RoboGen-sim2real (provides test_PointNet2.* and the dp3 train_ddp)")
    parser.add_argument("--robosuite_root", type=str, default="",
        help="Optional path to a complete robosuite checkout (parent dir of the robosuite/ package)")
    parser.add_argument("--cat_idx", type=int, default=0,
        help="SigLIP category index for HL conditioning (0 = 'open the storage furniture', "
             "the default for Coffee_D2 since its path matches no category substring)")
    parser.add_argument("--n_episodes",     type=int, default=10)
    parser.add_argument("--max_steps",      type=int, default=400)
    parser.add_argument("--seed",           type=int, default=100000)
    parser.add_argument("--n_obs_steps",    type=int, default=2)
    parser.add_argument("--n_action_steps", type=int, default=8)
    parser.add_argument("--camera_h",       type=int, default=256)
    parser.add_argument("--camera_w",       type=int, default=256)
    parser.add_argument("--output_dir",     type=str, default=None,
        help="Where to save args.json / results.jsonl / summary.json. "
             "Default: outputs_eval_smith/<HL_dir>__<LL_dir>_<ckpt>/<timestamp>/.")
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
        args.output_dir = f"outputs_eval_smith/{hl_tag}__{ll_tag}_{ckpt_tag}/{ts}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] saving results to {output_dir.resolve()}")
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Bootstrap sys.path BEFORE importing torch / SMITH model classes.
    _bootstrap_paths(args)

    # PROJECT_DIR is required by diffusion_policy_3d.policy.dp3.DP3.__init__ —
    # it loads SigLIP features from ${PROJECT_DIR}/siglip_text_features_w_pick_and_place_w_grasp_and_lift.pt.
    # The shell wrapper sets it; bail loudly if missing rather than at model __init__.
    if "PROJECT_DIR" not in os.environ:
        raise RuntimeError(
            "PROJECT_DIR env var is required (SMITH DP3 loads SigLIP from it). "
            "Set it to the directory containing siglip_text_features_w_pick_and_place_w_grasp_and_lift.pt."
        )

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    shape_meta = get_shape_meta(args.camera_h, args.camera_w)
    env, env_meta = create_mimicgen_env(
        args.dataset_path, shape_meta, args.camera_h, args.camera_w)
    inner = env.env if hasattr(env, "env") else env
    controller = inner.robots[0].controller

    print(f"[HL] loading SMITH PointNet2_super_multitask from {args.high_level_ckpt}")
    hl_model, _hl_args = load_smith_high_level(args.high_level_ckpt, device=device)

    cat_embedding = load_siglip_cat_embedding(
        os.environ["PROJECT_DIR"], cat_idx=args.cat_idx, device=device)
    print(f"[HL] using cat_idx={args.cat_idx} SigLIP embedding (shape={tuple(cat_embedding.shape)})")

    print(f"[LL] loading SMITH DP3 from {args.low_level_exp_dir}/{args.low_level_checkpoint}")
    ll_model, _ll_cfg = load_smith_low_level(
        args.low_level_exp_dir, args.low_level_checkpoint, device=device)

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

    rewards, successes = [], []
    results_path = output_dir / "results.jsonl"
    with open(results_path, "w") as results_f:
        for ep in range(args.n_episodes):
            seed = args.seed + ep
            np.random.seed(seed)
            torch.manual_seed(seed)
            r, succ, frames, frames_overlay = run_episode(
                env, hl_model, ll_model, cat_embedding, args.cat_idx,
                controller,
                args.n_obs_steps, args.n_action_steps, args.max_steps,
                device=device,
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
    main()
