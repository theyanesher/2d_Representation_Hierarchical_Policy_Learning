"""
Anchor-set diagnostic for the Approach-2 goal-GMM auxiliary head.

Answers the design fork: should the auxiliary GMM be anchored on DINOv2 patch
centres (unprojected from depth) or on obs/point_cloud (the same 4500 points the
high-level policy uses)?

Measures, per frame:
  * d_min   : distance from the GT goal centroid to the NEAREST anchor. The GMM's
              per-anchor displacement is Delta_n = goal - anchor_n, so d_min is
              the shortest displacement the head ever has to regress.
  * d_med   : median distance over all anchors. nll_loss also runs a
              uniform-weights term, so EVERY anchor must predict a good
              displacement — this is the typical regression magnitude.
  * in_bbox : fraction of patch anchors inside the point_cloud bounding box,
              i.e. on task-relevant geometry rather than far wall/floor.

Also validates the depth->world unprojection against obs/point_cloud, which
resolves the extrinsic convention and would catch a silently-wrong unprojection.

Result on COFFEE_PREPERATION_D1 (600 frames / 40 demos), which is why the
implementation anchors on patch centres and keeps the variance ladder unchanged:

    extrinsics are CAMERA-TO-WORLD  (5.4 mm reconstruction error vs 545 mm as w2c)

    goal centroid -> nearest anchor       median     p90     p99
      patch centres (512)                  35.3    96.1   163.2   mm
      obs/point_cloud (4500)               41.8   108.6   171.1   mm

    sigma^2=0.01 |exponent| at median d_min = 0.25  -> tightest rung still live
    patch anchors inside the object bbox: 56%  (~288 of 512)

Usage:
    pixi run python anchor_diagnostic.py --data_dir <NO_GMM h5 dir> \
        --n_demos 40 --n_frames 15
"""

import argparse
import os

import h5py
import numpy as np

PATCH = 14
CROP = 224
IMG = 256


def unproject(depth_mm, K, E, c2w):
    """(H,W) uint16 mm + 3x3 intrinsic + 4x4 extrinsic -> (H,W,3) world XYZ."""
    z = depth_mm.astype(np.float32) / 1000.0
    H, W = z.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    cam = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], axis=-1)
    R, t = E[:3, :3], E[:3, 3]
    if c2w:
        return cam @ R.T + t
    # extrinsic is world->camera: world = R^T (cam - t)
    return (cam - t) @ R


def patch_centers(world_xyz):
    """(256,256,3) -> (16*16,3): centre of each 14x14 patch of the 224 centre crop."""
    off = (IMG - CROP) // 2
    idx = off + PATCH // 2 + PATCH * np.arange(CROP // PATCH)
    return world_xyz[np.ix_(idx, idx)].reshape(-1, 3)


def nn_dist(query, anchors):
    """min L2 distance from a single (3,) point to (N,3) anchors."""
    return float(np.sqrt(((anchors - query) ** 2).sum(-1)).min())


def check_convention(f, t=0):
    """Return (is_camera_to_world, error) by reconstructing obs/point_cloud."""
    pcd = f["obs/point_cloud"][t]
    best, best_err = None, None
    for c2w in (False, True):
        pts = np.concatenate([
            unproject(f[f"obs/{cam}_depth"][t], f[f"obs/{cam}_intrinsic"][t],
                      f[f"obs/{cam}_extrinsic"][t], c2w).reshape(-1, 3)
            for cam in ("cam0", "cam1")
        ], 0)
        sub = pcd[np.random.default_rng(0).choice(len(pcd), 200, replace=False)]
        err = np.mean([nn_dist(p, pts) for p in sub])
        print(f"  c2w={c2w!s:5s} -> mean NN dist from point_cloud to unprojected: "
              f"{err*1000:7.2f} mm")
        if best_err is None or err < best_err:
            best, best_err = c2w, err
    return best, best_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--n_demos", type=int, default=40)
    ap.add_argument("--n_frames", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    files = sorted(
        [x for x in os.listdir(args.data_dir) if x.endswith(".h5")],
        key=lambda s: int(s.split("_")[1].split(".")[0]),
    )[: args.n_demos]

    print(f"[convention check] {files[0]}")
    with h5py.File(os.path.join(args.data_dir, files[0]), "r") as f:
        c2w, err = check_convention(f)
    print(f"  -> using c2w={c2w} (error {err*1000:.2f} mm)\n")
    if err > 0.02:
        print("  WARNING: neither convention reproduces point_cloud within 20mm.\n")

    d_min_patch, d_min_pcd, d_med_patch, in_bbox = [], [], [], []

    for name in files:
        with h5py.File(os.path.join(args.data_dir, name), "r") as f:
            T = f["obs/state"].shape[0]
            for t in sorted(rng.choice(T, min(args.n_frames, T), replace=False)):
                t = int(t)
                anchors = np.concatenate([
                    patch_centers(unproject(f[f"obs/{c}_depth"][t],
                                            f[f"obs/{c}_intrinsic"][t],
                                            f[f"obs/{c}_extrinsic"][t], c2w))
                    for c in ("cam0", "cam1")
                ], 0)                                   # (512, 3)
                pcd = f["obs/point_cloud"][t]           # (4500, 3)
                goal_c = f["obs/goal_gripper_pts"][t].mean(0)

                d_min_patch.append(nn_dist(goal_c, anchors))
                d_min_pcd.append(nn_dist(goal_c, pcd))
                d_med_patch.append(float(np.median(
                    np.sqrt(((anchors - goal_c) ** 2).sum(-1)))))

                lo, hi = pcd.min(0), pcd.max(0)
                in_bbox.append(np.all((anchors >= lo) & (anchors <= hi), axis=1).mean())

    def stats(name, v):
        q = np.percentile(np.array(v) * 1000.0, [50, 75, 90, 95, 99])
        print(f"  {name:34s} median {q[0]:7.1f}  p75 {q[1]:7.1f}  "
              f"p90 {q[2]:7.1f}  p95 {q[3]:7.1f}  p99 {q[4]:7.1f}  [mm]")

    print(f"[results] {len(d_min_patch)} frames from {len(files)} demos\n")
    print("Distance from GT goal centroid to nearest anchor:")
    stats("patch anchors (512)", d_min_patch)
    stats("point_cloud anchors (4500)", d_min_pcd)
    print("\nTypical displacement the head must regress (median over all anchors):")
    stats("patch anchors (512)", d_med_patch)
    print(f"\nPatch anchors inside the point_cloud bbox: "
          f"mean {np.mean(in_bbox)*100:.1f}%  "
          f"(~{np.mean(in_bbox)*512:.0f} of 512 on task-relevant geometry)")

    dm = np.array(d_min_patch)
    print("\nFraction of frames with nearest patch anchor beyond a threshold:")
    for thr in (0.02, 0.05, 0.08, 0.10, 0.15, 0.20):
        print(f"  > {thr*1000:5.0f} mm : {(dm > thr).mean()*100:5.1f}%")

    print("\nsigma^2 ladder reachability (exponent ~ -2*d^2/sigma^2 at 4 keypoints):")
    for var in (0.01, 0.05, 0.1, 0.25, 0.5):
        print(f"  sigma^2={var:5.2f}  |exponent| at median d_min = "
              f"{2*np.median(dm)**2/var:8.4f}")


if __name__ == "__main__":
    main()
