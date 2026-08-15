"""
Approach 2 eval: 2D DiT low-level policy ALONE (no high-level policy).

The Approach 2 LL is not given a goal. The goal only ever supervised its visual
representation through the auxiliary GMM head during training, so at rollout the
policy is fully self-contained:

    obs (RGB + depth + camera matrices + state + gripper keypoints)
        -> DINOv2 -> RoPE4D grounded trunk -> DiT -> hybrid_delta actions

Consequently there is no lfd3d / ArticubotNetwork / subgoal inference here, and
no --high_level_ckpt. Output layout (args.json / results.jsonl / summary.json /
media/) is byte-compatible with eval_gmm_high_level_2d_dit_low_level.py so the
shell-side resume+merge bookkeeping and any downstream analysis work unchanged.

Four things this script must get right, all of which fail SILENTLY otherwise:

  1. Full-res *_image / *_depth keys must survive get_observation(). The env
     gates that deletion on data_gen (default False), so nothing needs passing.
     NOTE: an earlier revision of this script passed is_eval=True here. That
     kwarg belongs to the repo-root env_robosuite.py, NOT to the one actually
     used (external/robomimic/robomimic/envs/env_robosuite.py), which would
     forward it to robosuite.make() and fail.
  2. camera_depths=True, so depth is rendered at all.
  3. Depth must reach the policy in METRES. env_robosuite already applies
     get_real_depth_map() + the [::-1] vertical flip to depth and RGB alike, so
     the keys are consumed as-is; do not re-flip or re-scale here.
  4. Everything is in ROBOT-BASE frame, not world. The env applies
     base_world_T_base_robot to point_cloud, gripper_pcd and robot0_eef_pos,
     and pre-multiplies the camera extrinsics by it, so unprojected depth lands
     in base frame consistently with state / gripper_pcd / the training h5.
     Those extrinsics are camera-to-base; do not add a robot-base offset.

Observation history: training fed two CONSECUTIVE frames (obs step 0 = frame t,
obs step 1 = frame t+1), so this script keeps a real 2-deep environment-obs
buffer rather than tiling one frame. At the first step of an episode there is no
history and the frame is repeated -- which is exactly what training did, since
the dataset's pad_before=1 makes each episode's first sample a repeated frame.
"""

import argparse
import collections
import copy
import json
import os
import sys
import types
from datetime import datetime
from pathlib import Path

import numpy as np

# EGL device extensions must be imported before mujoco touches EGL.
try:
    import OpenGL.EGL.EXT.device_base  # noqa: F401
except Exception:
    pass


# -----------------------------------------------------------------------------
# sys.path bootstrap
# -----------------------------------------------------------------------------
def _prepend_sys_path(*roots):
    for r in roots:
        if r and os.path.isdir(r) and r not in sys.path:
            sys.path.insert(0, r)


def _bootstrap_paths(args):
    """Put the low-level training repo ahead of everything else.

    NOTE: <LL>/diffusion_policy/diffusion_policy/ has no __init__.py, i.e. it is
    a NAMESPACE package. Python prefers a regular package of the same name found
    anywhere on sys.path regardless of order, so if some environment also ships a
    `diffusion_policy` with an __init__.py it will win and the Approach 2 policy
    class will appear to be missing. Keep the env clean of that.
    """
    dit_2d_parent = Path(args.dit_2d_repo).parent      # .../Low_Level_and_Inference
    _prepend_sys_path(args.dit_2d_repo, str(dit_2d_parent))

    # This file: <INFERENCE_ROOT>/external/mimicgen/mimicgen/scripts/<this>.py
    inference_root = Path(__file__).resolve().parents[4]
    _prepend_sys_path(
        str(inference_root),
        str(inference_root / "external" / "mimicgen"),
        # REQUIRED: robomimic must resolve to this source tree, not to the copy
        # in site-packages. Only this one defines the EnvRobosuite that emits
        # base-frame `state` and base-frame camera extrinsics; the site-packages
        # one (symlinked to the repo-root env_robosuite.py) emits world-frame
        # `agent_pos` and strips full-res images, which fails silently as
        # KeyError: 'agentview_image' several frames later.
        str(inference_root / "external" / "robomimic"),
    )
    if args.robosuite_root:
        _prepend_sys_path(args.robosuite_root)


def _stub_unused_training_imports():
    """The LL workspace transitively imports a couple of training-only modules
    that are irrelevant at eval time and absent here."""
    if "train_ddp" not in sys.modules:
        stub = types.ModuleType("train_ddp")
        # eval_smith_utils does `from train_ddp import TrainDP3Workspace` at
        # import time but we only need policy_action_batch_to_env_action from it.
        stub.TrainDP3Workspace = object
        sys.modules["train_ddp"] = stub


# -----------------------------------------------------------------------------
# shape_meta / env
# -----------------------------------------------------------------------------
def get_shape_meta(camera_h: int, camera_w: int):
    """Only used to drive robomimic's obs-modality registration."""
    return {
        "obs": {
            "agentview_image":          {"shape": [3, camera_h, camera_w], "type": "rgb"},
            "robot0_eye_in_hand_image": {"shape": [3, camera_h, camera_w], "type": "rgb"},
            "agentview_depth":          {"shape": [1, camera_h, camera_w], "type": "depth"},
            "robot0_eye_in_hand_depth": {"shape": [1, camera_h, camera_w], "type": "depth"},
            "state":                    {"shape": [10],                    "type": "low_dim"},
            "point_cloud":              {"shape": [4500, 3],               "type": "point_cloud"},
            "gripper_pcd":              {"shape": [4, 3],                  "type": "low_dim"},
            "robot0_eef_quat":          {"shape": [4],                     "type": "low_dim"},
        },
        "action": {"shape": [10]},
    }


def create_mimicgen_env(dataset_path: str, shape_meta: dict, camera_h: int, camera_w: int):
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils
    # Importing mimicgen is what REGISTERS its robosuite tasks (HammerCleanup_D1
    # etc.). Without it robosuite.make() only knows the stock envs.
    import mimicgen  # noqa: F401

    env_meta = FileUtils.get_env_metadata_from_dataset(os.path.expanduser(dataset_path))
    env_name = env_meta["env_name"]
    camera_names = (["birdview", "agentview", "robot0_eye_in_hand"]
                    if env_name.startswith("PickPlace")
                    else ["birdview", "agentview", "sideview", "robot0_eye_in_hand"])

    env_kwargs = dict(env_meta["env_kwargs"])
    env_kwargs.pop("env_name", None)
    env_kwargs["camera_names"]   = camera_names
    env_kwargs["camera_heights"] = camera_h
    env_kwargs["camera_widths"]  = camera_w
    env_kwargs["camera_depths"]  = True          # (2) Approach 2 unprojects depth

    # ---- metric depth must NOT be clipped to [0, 1] -----------------------
    # robomimic's DepthModality processor is process_frame(channel_dim=1, scale=1.),
    # and process_frame ends with `frame.clip(0.0, 1.0)`. That is harmless for RGB
    # (scale=255, so uint8/255 is already <= 1) but the env hands depth to it in
    # METRES, so every pixel past 1 m gets truncated to exactly 1.0. Measured on
    # Coffee_Preperation_D1: the training h5 has depth spanning 0.743-3.064 m,
    # while the unpatched env returns min 0.7435 / max 1.0000. Approach 2's
    # RoPE4D trunk unprojects this depth to place its visual tokens, so the whole
    # scene beyond 1 m collapses onto a shell and the 3D grounding is destroyed.
    # (Approach 1 never reads depth, which is why only this eval regressed.)
    #
    # set_obs_processor is robomimic's sanctioned override hook. Same as the
    # default minus the /scale and the clip: to float, then HWC -> CHW.
    def _depth_processor_keep_metres(obs):
        import robomimic.utils.tensor_utils as TU
        return ObsUtils.batch_image_hwc_to_chw(TU.to_float(obs))

    ObsUtils.DepthModality.set_obs_processor(_depth_processor_keep_metres)

    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta["obs"].items():
        modality_mapping[attr.get("type", "low_dim")].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = EnvUtils.create_env(
        env_type=EnvUtils.get_env_type(env_meta=env_meta),
        env_name=env_name,
        render=False,
        render_offscreen=True,
        use_image_obs=True,
        # (1) full-res image/depth keys are kept by default: the env only strips
        # them when data_gen=True, which we leave at its default.
        **env_kwargs,
    )
    return env, env_meta


def load_low_level_2d_dit(exp_dir: str, ckpt_name: str, device: str = "cuda"):
    import hydra
    from omegaconf import OmegaConf

    cfg_path = Path(exp_dir) / ".hydra" / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"LL hydra config missing: {cfg_path}")
    cfg = OmegaConf.load(str(cfg_path))

    workspace = hydra.utils.get_class(cfg._target_)(cfg)
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
# obs plumbing
# -----------------------------------------------------------------------------
def _maybe_unwrap(obs):
    if isinstance(obs, dict) and "obs" in obs and not {"agentview_image", "state"} & obs.keys():
        return obs["obs"]
    return obs


# env obs key -> policy obs key. cam0 = agentview, cam1 = wrist, matching training.
_CAM_MAP = {
    "agentview_image":               "cam0_image",
    "robot0_eye_in_hand_image":      "cam1_image",
    "agentview_depth":               "cam0_depth",
    "robot0_eye_in_hand_depth":      "cam1_depth",
    "agentview_intrinsics":          "cam0_intrinsic",
    "robot0_eye_in_hand_intrinsics": "cam1_intrinsic",
    "agentview_extrinsics":          "cam0_extrinsic",
    "robot0_eye_in_hand_extrinsics": "cam1_extrinsic",
}


def _state_from_obs(obs):
    """10-D [pos(3), rot6d(6), gripper(1)]; env_robosuite exposes it as agent_pos."""
    if "state" in obs:
        return np.asarray(obs["state"], dtype=np.float32)
    return np.asarray(obs["agent_pos"], dtype=np.float32)


def build_ll_obs_dict(obs_hist, n_obs_steps: int, device):
    """Stack a REAL 2-deep observation history into (B=1, T=n_obs_steps, ...).

    obs_hist is a deque of raw env obs, oldest first. When it is shorter than
    n_obs_steps (only at the first step of an episode) the oldest entry is
    repeated -- matching the dataset's pad_before=1 behaviour, which makes each
    episode's first training sample a repeated frame.
    """
    import torch

    hist = list(obs_hist)
    while len(hist) < n_obs_steps:
        hist.insert(0, hist[0])
    hist = hist[-n_obs_steps:]

    out = {}
    for env_key, pol_key in _CAM_MAP.items():
        stacked = np.stack([np.asarray(o[env_key], dtype=np.float32) for o in hist], axis=0)
        out[pol_key] = torch.from_numpy(stacked[None]).float().to(device)

    out["state"] = torch.from_numpy(
        np.stack([_state_from_obs(o) for o in hist], 0)[None]).float().to(device)
    out["present_gripper_pts"] = torch.from_numpy(
        np.stack([np.asarray(o["gripper_pcd"], dtype=np.float32)[:, :3] for o in hist], 0)[None]
    ).float().to(device)
    return out


def _agentview_to_uint8_rgb(obs):
    img = np.asarray(obs["agentview_image"])
    frame = np.transpose(img, (1, 2, 0))
    return np.ascontiguousarray((frame * 255.0).clip(0, 255).astype(np.uint8))


def _write_mp4(path, frames, fps):
    """imageio + imageio-ffmpeg (equi_diffpo's recorder is not installed here)."""
    import imageio
    with imageio.get_writer(str(path), fps=fps, codec="libx264",
                            quality=None, ffmpeg_params=["-crf", "22", "-pix_fmt", "yuv420p"],
                            macro_block_size=1) as w:
        for f in frames:
            w.append_data(f)


# -----------------------------------------------------------------------------
# rollout
# -----------------------------------------------------------------------------
def run_episode(env, ll_model, controller, n_obs_steps, n_action_steps, max_steps, device):
    import torch
    from eval_smith_utils import policy_action_batch_to_env_action

    obs = _maybe_unwrap(env.reset())
    max_dpos = float(controller.output_max[0])
    max_drot = float(controller.output_max[3])

    obs_hist = collections.deque([obs], maxlen=n_obs_steps)
    total_reward, success, step = 0.0, False, 0
    frames = [_agentview_to_uint8_rgb(obs)]

    while step < max_steps:
        ll_obs = build_ll_obs_dict(obs_hist, n_obs_steps, device)
        with torch.no_grad():
            action_dict = ll_model.predict_action(ll_obs)
        action_raw = action_dict.get("action_pred", action_dict["action"]).detach().cpu().numpy()

        eef_quat = np.array(obs["robot0_eef_quat"], dtype=np.float64)
        env_action_arm = policy_action_batch_to_env_action(
            action_raw, np.tile(eef_quat[None, :], (1, 1)), max_dpos, max_drot,
        )
        env_action_seq = env_action_arm[0, :n_action_steps]              # (T, 7)

        for t_idx in range(env_action_seq.shape[0]):
            obs, reward, done, _info = env.step(env_action_seq[t_idx])
            obs = _maybe_unwrap(obs)
            obs_hist.append(obs)                     # real t-1 / t history
            frames.append(_agentview_to_uint8_rgb(obs))
            total_reward += float(reward)
            step += 1
            if env.is_success()["task"]:
                success = True
                break
            if done or step >= max_steps:
                break
        if success or step >= max_steps:
            break

    return total_reward, success, frames


# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Approach 2 eval: 2D DiT LL alone (no high-level policy).")
    parser.add_argument("--dataset_path", type=str, required=True,
        help="MimicGen hdf5 supplying env metadata / init states.")
    parser.add_argument("--low_level_exp_dir", type=str, required=True,
        help="LL run dir containing .hydra/config.yaml and checkpoints/.")
    parser.add_argument("--low_level_checkpoint", type=str, default="epoch_99.ckpt")
    parser.add_argument("--dit_2d_repo", type=str, required=True,
        help="<Low_Level_and_Inference>/diffusion_policy")
    parser.add_argument("--robosuite_root", type=str, default="")
    parser.add_argument("--n_episodes",     type=int, default=50)
    parser.add_argument("--max_steps",      type=int, default=800)
    parser.add_argument("--seed",           type=int, default=100000)
    parser.add_argument("--n_obs_steps",    type=int, default=2)
    parser.add_argument("--n_action_steps", type=int, default=8)
    parser.add_argument("--camera_h",       type=int, default=256)
    parser.add_argument("--camera_w",       type=int, default=256)
    parser.add_argument("--output_dir",     type=str, default=None)
    parser.add_argument("--save_videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video_fps",      type=int, default=10)
    args = parser.parse_args()

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

    _bootstrap_paths(args)
    _stub_unused_training_imports()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    shape_meta = get_shape_meta(args.camera_h, args.camera_w)
    env, _env_meta = create_mimicgen_env(
        args.dataset_path, shape_meta, args.camera_h, args.camera_w)
    inner = env.env if hasattr(env, "env") else env
    controller = inner.robots[0].controller

    print(f"[LL] loading Approach 2 policy from {args.low_level_exp_dir}/{args.low_level_checkpoint}")
    ll_model, ll_cfg = load_low_level_2d_dit(
        args.low_level_exp_dir, args.low_level_checkpoint, device=device)

    # Sanity-check that this really is an Approach 2 run.
    from omegaconf import OmegaConf
    tgt = str(OmegaConf.select(ll_cfg, "policy._target_", default="")).split(".")[-1]
    aux_w = OmegaConf.select(ll_cfg, "policy.aux_gmm_loss_weight", default=None)
    print(f"[LL] policy={tgt}  aux_gmm_loss_weight={aux_w}  "
          f"encoder={type(getattr(ll_model, 'visual_encoder', None)).__name__}")
    if tgt != "FlowMatchingDiTGoalGMMPolicy":
        print(f"[LL][WARN] expected FlowMatchingDiTGoalGMMPolicy, got {tgt!r}. "
              "This script feeds no goal, so a goal-conditioned LL would run blind.")

    videos_dir = None
    if args.save_videos:
        videos_dir = output_dir / "media"
        videos_dir.mkdir(parents=True, exist_ok=True)
        # Created for parity with the hierarchical evals' merge bookkeeping;
        # Approach 2 has no goal, so there is nothing to overlay.
        (output_dir / "media_with_goal_overlay").mkdir(parents=True, exist_ok=True)
        print(f"[video] rollouts -> {videos_dir.resolve()} (fps={args.video_fps}, h264 crf=22)")

    rewards, successes = [], []
    with open(output_dir / "results.jsonl", "w") as results_f:
        for ep in range(args.n_episodes):
            seed = args.seed + ep
            np.random.seed(seed)
            torch.manual_seed(seed)
            r, succ, frames = run_episode(
                env, ll_model, controller,
                args.n_obs_steps, args.n_action_steps, args.max_steps, device=device,
            )

            video_path = None
            if args.save_videos:
                outcome_tag = "success" if succ else "failure"
                video_path = videos_dir / f"episode_{ep + 1:03d}_seed_{seed}_{outcome_tag}.mp4"
                _write_mp4(video_path, frames, args.video_fps)

            rewards.append(r)
            successes.append(succ)
            vtag = f"  video={video_path.name}" if video_path is not None else ""
            print(f"Episode {ep + 1}/{args.n_episodes}  seed={seed}  "
                  f"reward={r:.2f}  success={succ}{vtag}")
            results_f.write(json.dumps({
                "episode": ep + 1, "seed": seed,
                "reward": float(r), "success": bool(succ),
                "video": str(video_path) if video_path is not None else None,
                "video_with_goal_overlay": None,
            }) + "\n")
            results_f.flush()

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
    print("  - args.json\n  - results.jsonl\n  - summary.json")


if __name__ == "__main__":
    sys.exit(main())
