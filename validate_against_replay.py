"""Validate converted .npz steps against the RVT/.replay reference (Dir A)."""
import os
import sys
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rlbench_to_d2_converter as C

REPLAY_DIR = ("/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/"
              "Articubot_Data_For_RVT/SMITH_High_Level_FineTune/RL_BENCH_SMOOTH_DATASET/"
              "sweep_to_dustpan_of_size/replay_buffer/sweep_to_dustpan_of_size")

CAMERAS = C.CAMERAS


def quat_to_R(q):
    q = q / (np.linalg.norm(q) + 1e-8)
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def pose_to_4pt(pose7, open_):
    lf = np.array([0, -0.0405, 0.08]); rf = np.array([0, 0.0405, 0.08]); w = np.array([0, 0, 0.])
    tmpl = np.stack([lf, rf, w, 0.5 * (lf + rf)], 0)
    t = pose7[:3]; R = quat_to_R(pose7[3:7]); p = tmpl.copy()
    d = 0.021 * (np.clip(open_, 0, 1) - 0.5); p[0, 1] -= d; p[1, 1] += d; p[3] = 0.5 * (p[0] + p[1])
    return (R @ p.T).T + t


def cmp(name, a, b, atol=1e-4):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    if a.shape != b.shape:
        print(f"   [SHAPE ] {name:34s} npz{a.shape} vs replay{b.shape}")
        return
    md = float(np.abs(a - b).max())
    tag = "OK    " if md <= atol else "DIFF  "
    print(f"   [{tag}] {name:34s} max|Δ|={md:.3e}")


def validate(npz_dir, t):
    npz = np.load(os.path.join(npz_dir, f"{t}.npz"), allow_pickle=True)
    rp = pickle.load(open(os.path.join(REPLAY_DIR, f"{t}.replay"), "rb"))   # demo0: npz t -> t.replay
    print(f"\n--- t={t}  (episode_idx npz={int(npz['episode_idx'])}, replay={int(rp['episode_idx'])}) ---")

    cmp("state[:7] vs gripper_pose[1]", npz["state"][0, :7], rp["gripper_pose"][1])
    cmp("state[7] vs gripper_open[1]", npz["state"][0, 7], rp["gripper_open"][1, 0])
    cmp("eef_pos vs gripper_pose[1,:3]", npz["eef_pos"][0], rp["gripper_pose"][1, :3])
    cmp("eef_quat vs gripper_pose[1,3:7]", npz["eef_quat"][0], rp["gripper_pose"][1, 3:7])
    cmp("gripper_qpos vs low_dim_state[1,1:3]", npz["gripper_qpos"][0], rp["low_dim_state"][1, 1:3])
    cmp("action vs replay action[0]", npz["action"][0], rp["action"][0])

    for cam in CAMERAS:
        cmp(f"{cam}_rgb", npz[f"{cam}_rgb"][0], rp[f"{cam}_rgb"][1])
        cmp(f"{cam}_depth", npz[f"{cam}_depth"][0], rp[f"{cam}_depth"][1])
        cmp(f"{cam}_point_cloud", npz[f"{cam}_point_cloud"][0], rp[f"{cam}_point_cloud"][1])
        cmp(f"{cam}_camera_extrinsics", npz[f"{cam}_camera_extrinsics"][0], rp[f"{cam}_camera_extrinsics"][1])
        cmp(f"{cam}_camera_intrinsics", npz[f"{cam}_camera_intrinsics"][0], rp[f"{cam}_camera_intrinsics"][1])

    # goal: replay stores pose (rdp_gripper); convert to 4pt and compare to our pcd (rdp)
    goal_replay_pcd = pose_to_4pt(rp["goal_gripper_pose"], float(rp["goal_gripper_open"][0]))
    cmp("goal_gripper_pcd (rdp vs rdp_gripper)", npz["goal_gripper_pcd"][0], goal_replay_pcd)

    # fused point cloud: no replay counterpart -> sanity only
    pc = npz["point_cloud"][0]
    x0, y0, z0, x1, y1, z1 = C.SCENE_BOUNDS
    inb = np.mean((pc[:, 0] >= x0) & (pc[:, 0] <= x1) & (pc[:, 1] >= y0) &
                  (pc[:, 1] <= y1) & (pc[:, 2] >= z0) & (pc[:, 2] <= z1))
    print(f"   [INFO  ] point_cloud shape={pc.shape}  frac in SCENE_BOUNDS={inb:.3f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--steps", type=int, nargs="+", default=[0, 1, 2, 3])
    args = ap.parse_args()
    for t in args.steps:
        validate(args.npz_dir, t)
