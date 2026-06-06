"""
Convert RLBench smooth-dataset episodes into per-step .npz files (D2-style layout).

Per frame t (leading axis = 1, single frame, NO history), 4 cameras
[front, left_shoulder, right_shoulder, wrist] at 128x128:

  state              (1,8)   [x,y,z,qx,qy,qz,qw, gripper_open]  at t   (qw>=0)
  eef_pos            (1,3)   gripper_pose[:3] at t
  eef_quat           (1,4)   gripper_pose[3:7] at t (qw>=0)
  gripper_qpos       (1,2)   gripper_joint_positions at t, clipped [0,0.04]
  action             (1,8)   RAW gripper_pose+open at t+1 (qw>=0)
  point_cloud        (1,4500,3)        fused 4 cams + SCENE_BOUNDS crop + FPS
  {cam}_point_cloud  (1,3,128,128)     per-cam dense world-frame pcd
  {cam}_rgb          (1,3,128,128)     channels-first float
  {cam}_depth        (1,1,128,128)     meters
  {cam}_camera_extrinsics (1,4,4) / _intrinsics (1,3,3)
  goal_gripper_pcd   (1,4,3)  next-keypoint 4-pt pcd from keyframe_gripper_points.npz (rdp)
  lang_goal          (1,) object
  episode_idx, reward, terminal, timeout

Depth/point-cloud math replicate peract_colab.rlbench.utils.get_stored_demo exactly.
"""

import os
import sys
import types
import pickle
import importlib.util
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

# ----------------------------------------------------------------------------
RLBENCH_PKG = ("/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/"
               "RVT2_GMM_Codebase/Bimanual_Manipulation/rvt/libs/RLBench/rlbench")
EPISODES_DIR = ("/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/"
                "Articubot_Data_For_RVT/SMITH_High_Level_FineTune/RL_BENCH_SMOOTH_DATASET/"
                "sweep_to_dustpan_of_size/sweep_to_dustpan_of_size/all_variations/episodes")
OUT_ROOT = ("/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/Articubot_Data_For_RVT/SMITH_High_Level_FineTune/LOW_LEVEL_NO_GMM_DATASET_GROOT_STYLE_DATASET/D2/RL_BENCH_DATASET/sweep_to_dustpan_of_size")

CAMERAS = ["front", "left_shoulder", "right_shoulder", "wrist"]
IMAGE_SIZE = 128
DEPTH_SCALE = 2 ** 24 - 1
N_POINTS = 4500
SCENE_BOUNDS = [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]   # [x0,y0,z0,x1,y1,z1] world frame


# ----------------------------------------------------------------------------
def install_rlbench_classes():
    for name in ["rlbench", "rlbench.backend"]:
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = []
            sys.modules[name] = m

    def _load(modname, path):
        spec = importlib.util.spec_from_file_location(modname, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[modname] = mod
        spec.loader.exec_module(mod)

    _load("rlbench.backend.observation", f"{RLBENCH_PKG}/backend/observation.py")
    _load("rlbench.demo", f"{RLBENCH_PKG}/demo.py")


def load_demo(episode_idx):
    with open(os.path.join(EPISODES_DIR, f"episode{episode_idx}", "low_dim_obs.pkl"), "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------------------------
# depth / point cloud (same logic as peract_colab)
# ----------------------------------------------------------------------------
def image_to_float_array(png_path, scale_factor):
    arr = np.array(Image.open(png_path)).astype(np.float32)
    packed = np.sum(arr * [65536.0, 256.0, 1.0], axis=2)
    return packed / scale_factor


def _uniform_pixel_coords(h, w):
    px = np.reshape(np.tile(np.arange(w), [h]), (h, w, 1)).astype(np.float32)
    py = np.reshape(np.tile(np.arange(h), [w]), (w, h, 1)).astype(np.float32)
    py = np.transpose(py, (1, 0, 2))
    return np.concatenate([px, py, np.ones_like(px)], -1)


def pointcloud_from_depth(depth, extrinsics, intrinsics):
    h, w = depth.shape
    upc = _uniform_pixel_coords(h, w)
    pc = upc * depth[..., None]
    C = extrinsics[:3, 3:4]
    R = extrinsics[:3, :3]
    R_inv = R.T
    R_inv_C = R_inv @ C
    ext = np.concatenate([R_inv, -R_inv_C], axis=-1)
    cam_proj = intrinsics @ ext
    cam_proj_homo = np.concatenate([cam_proj, [[0, 0, 0, 1.0]]], axis=0)
    cam_proj_inv = np.linalg.inv(cam_proj_homo)[:3]
    pc_homo = np.concatenate([pc, np.ones((h, w, 1))], -1)
    flat = pc_homo.reshape(h * w, 4).T
    world = (cam_proj_inv @ flat).T.reshape(h, w, 3)
    return world.astype(np.float32)


def farthest_point_sampling(xyz, n_samples):
    N = xyz.shape[0]
    centroids = np.zeros((n_samples,), dtype=np.int64)
    distances = np.full((N,), 1e10, dtype=np.float32)
    farthest = 0
    for i in range(n_samples):
        centroids[i] = farthest
        dist = np.sum((xyz - xyz[farthest:farthest + 1]) ** 2, axis=-1)
        distances = np.minimum(distances, dist)
        farthest = int(np.argmax(distances))
    return centroids


def crop_scene_bounds(pc):
    x0, y0, z0, x1, y1, z1 = SCENE_BOUNDS
    m = ((pc[:, 0] >= x0) & (pc[:, 0] <= x1) &
         (pc[:, 1] >= y0) & (pc[:, 1] <= y1) &
         (pc[:, 2] >= z0) & (pc[:, 2] <= z1))
    return pc[m]


def canon_quat(pose7):
    p = np.asarray(pose7, dtype=np.float32).copy()
    if p[6] < 0:
        p[3:7] = -p[3:7]
    return p


# ----------------------------------------------------------------------------
def build_camera_arrays(obs, episode_idx, t):
    out = {}
    fused = []
    ep_dir = os.path.join(EPISODES_DIR, f"episode{episode_idx}")
    for cam in CAMERAS:
        rgb = np.array(Image.open(os.path.join(ep_dir, f"{cam}_rgb", f"{t}.png")))
        rgb = np.transpose(rgb, (2, 0, 1)).astype(np.float32)

        d_norm = image_to_float_array(os.path.join(ep_dir, f"{cam}_depth", f"{t}.png"), DEPTH_SCALE)
        near = obs.misc[f"{cam}_camera_near"]
        far = obs.misc[f"{cam}_camera_far"]
        depth = (near + d_norm * (far - near)).astype(np.float32)

        extr = np.asarray(obs.misc[f"{cam}_camera_extrinsics"], dtype=np.float32)
        intr = np.asarray(obs.misc[f"{cam}_camera_intrinsics"], dtype=np.float32)
        pcd = pointcloud_from_depth(depth, extr, intr)               # (H,W,3)

        out[f"{cam}_rgb"] = rgb[None]
        out[f"{cam}_depth"] = depth[None, None]
        out[f"{cam}_point_cloud"] = np.transpose(pcd, (2, 0, 1))[None]
        out[f"{cam}_camera_extrinsics"] = extr[None]
        out[f"{cam}_camera_intrinsics"] = intr[None]
        fused.append(pcd.reshape(-1, 3))

    fused = np.concatenate(fused, axis=0)
    cropped = crop_scene_bounds(fused)
    if cropped.shape[0] == 0:
        cropped = fused
    if cropped.shape[0] >= N_POINTS:
        sampled = cropped[farthest_point_sampling(cropped, N_POINTS)]
    else:
        pad = np.random.choice(cropped.shape[0], N_POINTS - cropped.shape[0], replace=True)
        sampled = np.concatenate([cropped, cropped[pad]], axis=0)
    out["point_cloud"] = sampled.astype(np.float32)[None]
    return out


def make_goal_lookup(episode_idx):
    """Returns f(t) -> (4,3) goal pcd = next keypoint strictly after t (never kp 0)."""
    kf = np.load(os.path.join(EPISODES_DIR, f"episode{episode_idx}", "keyframe_gripper_points.npz"),
                 allow_pickle=True)
    kps = kf["keypoints"].astype(int)            # e.g. [0,56,71,95,107,133]
    gpts = kf["gripper_points"].astype(np.float32)  # (K,4,3)

    def goal(t):
        j = 0
        while j < len(kps) and t >= kps[j]:
            j += 1
        if j >= len(kps):
            j = len(kps) - 1
        return gpts[j]
    return goal


def build_step(demo, episode_idx, t, T, desc, goal_fn):
    obs = demo[t]
    obs_next = demo[min(t + 1, T - 1)]

    pose_t = canon_quat(obs.gripper_pose)
    open_t = np.float32(obs.gripper_open)
    qpos_t = np.clip(np.asarray(obs.gripper_joint_positions, np.float32), 0.0, 0.04)

    pose_tp1 = canon_quat(obs_next.gripper_pose)
    open_tp1 = np.float32(obs_next.gripper_open)

    rec = {}
    rec["state"] = np.concatenate([pose_t, [open_t]]).astype(np.float32)[None]
    rec["eef_pos"] = pose_t[:3].astype(np.float32)[None]
    rec["eef_quat"] = pose_t[3:7].astype(np.float32)[None]
    rec["gripper_qpos"] = qpos_t[None]
    rec["action"] = np.concatenate([pose_tp1, [open_tp1]]).astype(np.float32)[None]
    rec["goal_gripper_pcd"] = goal_fn(t).astype(np.float32)[None]      # (1,4,3)
    rec.update(build_camera_arrays(obs, episode_idx, t))
    rec["lang_goal"] = np.array([desc], dtype=object)
    rec["episode_idx"] = np.int64(episode_idx)
    rec["reward"] = np.float32(t == T - 2)
    rec["terminal"] = bool(t == T - 2)
    rec["timeout"] = False
    return rec


def convert_episode(episode_idx, demo_idx=None, max_steps=None, out_root=OUT_ROOT):
    if demo_idx is None:
        demo_idx = episode_idx
    demo = load_demo(episode_idx)
    T = len(demo)
    with open(os.path.join(EPISODES_DIR, f"episode{episode_idx}", "variation_descriptions.pkl"), "rb") as f:
        descs = pickle.load(f)
    desc = descs[0] if isinstance(descs, (list, tuple)) else str(descs)
    goal_fn = make_goal_lookup(episode_idx)

    out_dir = os.path.join(out_root, f"demo_{demo_idx}")
    os.makedirs(out_dir, exist_ok=True)

    n_steps = T - 1
    if max_steps is not None:
        n_steps = min(n_steps, max_steps)
    for t in range(n_steps):
        rec = build_step(demo, episode_idx, t, T, desc, goal_fn)
        np.savez(os.path.join(out_dir, f"{t}.npz"), **rec)
    tqdm.write(f"episode{episode_idx} -> demo_{demo_idx}: wrote {n_steps}/{T-1} steps to {out_dir}")
    return out_dir, T


def list_available_episodes():
    """Sorted episode indices present in EPISODES_DIR (episode0, episode1, ...)."""
    eps = []
    for name in os.listdir(EPISODES_DIR):
        if name.startswith("episode") and os.path.isdir(os.path.join(EPISODES_DIR, name)):
            try:
                eps.append(int(name[len("episode"):]))
            except ValueError:
                pass
    return sorted(eps)


def _episode_num_frames(ep):
    """T = number of frames in the episode (count of front_rgb PNGs)."""
    return len([x for x in os.listdir(os.path.join(EPISODES_DIR, f"episode{ep}", "front_rgb"))
                if x.endswith(".png")])


def is_done(ep, out_root):
    """A demo counts as already generated if demo_<ep> holds all T-1 step npz files."""
    out_dir = os.path.join(out_root, f"demo_{ep}")
    if not os.path.isdir(out_dir):
        return False
    n_npz = len([x for x in os.listdir(out_dir) if x.endswith(".npz")])
    return n_npz >= _episode_num_frames(ep) - 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_demos", type=int, default=None,
                    help="Generate at most this many demos that are NOT already in out_root "
                         "(the next N missing, in ascending episode order). "
                         "Default: generate ALL missing demos.")
    ap.add_argument("--episodes", type=int, nargs="+", default=None,
                    help="Explicit episode indices to (re)generate, overriding missing-demo discovery.")
    ap.add_argument("--max_steps", type=int, default=None, help="Limit steps per demo (debug).")
    ap.add_argument("--out_root", type=str, default=OUT_ROOT)
    args = ap.parse_args()

    install_rlbench_classes()

    if args.episodes is not None:
        todo = args.episodes
        print(f"[plan] explicit episodes: {todo}")
    else:
        available = list_available_episodes()
        missing = [ep for ep in available if not is_done(ep, args.out_root)]
        n_done = len(available) - len(missing)
        if args.num_demos is not None:
            missing = missing[:args.num_demos]
        todo = missing
        print(f"[plan] {len(available)} episodes available, {n_done} already done; "
              f"generating {len(todo)}: {todo}")

    for ep in tqdm(todo, desc="converting demos", unit="demo"):
        convert_episode(ep, max_steps=args.max_steps, out_root=args.out_root)
    print(f"[done] generated {len(todo)} demos into {args.out_root}")
