"""
Convert the real-world LeRobot Franka dataset (franka_push_block: 2x Azure
Kinect + 1x uncalibrated wrist ZED) into the same per-timestep .npz layout
that `external/mimicgen/mimicgen/scripts/convert_dataset.py` produces from
simulated MimicGen hdf5s. This is the format `NpyDataset`
(src/lfd3d/datasets/npy/npy_dataset.py) reads for high-level (HL) policy
training:

    <output_dir>/<demo_i>/<t>.npz
        point_cloud            (1, N, 3)   float32  robot-base frame, fused front+left
        gripper_pcd            (1, 4, 3)   float32  robot-base frame
        goal_gripper_pcd        (1, 4, 3)   float32  robot-base frame
        rgb_agentview           (1, H, W, 3) uint8
        depth_agentview         (1, H, W, 1) float32  metres
        agentview_intrinsics     (1, 3, 3)   float32  matches rgb/depth_agentview resolution
        agentview_extrinsics     (1, 4, 4)   float32  camera-to-base (world_from_cam)
        rgb_wrist / depth_wrist / wrist_intrinsics / wrist_extrinsics  (uncalibrated -> zeros/eye)
        eef_pos, eef_quat, gripper_qpos, state, action, lang_goal

ASSUMPTIONS THAT ARE NOT VERIFIED BY DATA IN THIS REPO (flagged inline too):
  1. gripper_pcd geometry: reuses MimicGen's `get_4_points_from_gripper_pos_orient`
     (mimicgen/utils/articubot_util.py), which encodes a specific Franka Hand
     keypoint layout + reference orientation captured in SIM. It assumes the
     real `observation.right_eef_pose` rotation is expressed in a frame whose
     axis convention matches the sim robot-base convention (both should be
     the standard Franka DH base frame; the eef translation/rotation convention
     itself IS cross-checked now -- panda_fk_joint_positions's joint7-frame
     origin, offset ~0.21m along its local z (0.107m flange + Franka Hand TCP),
     lines up with this same field's translation to ~1-2cm across sampled
     frames -- but the exact axis convention assumed by get_4_points_from_
     gripper_pos_orient's SIM-captured reference orientation is still unverified).
  2. `observation.right_eef_pose[..., 9]` ("gripper_articulation") is treated
     as a [0=closed, 1=open] normalized value and scaled by 0.04 m (the real
     Franka Hand's per-finger travel) to get `cur_joint_angle` / `eef_qpos`.
  3. `state` is written as zeros — NpyDataset.__getitem__ never reads "state"
     for HL training (only the LL eval path does, from a different key set),
     so this is inert for the HL conversion this script targets.
  4. goal_gripper_pcd subgoal decomposition reuses
     third_party/robogen/subgoal_decomp.py::compute_subgoal_gripper_pcd
     unmodified; its gripper-action sign convention (+1 close / -1 open) is
     approximated from the real `action` gripper channel via a 0.5 threshold
     since the real data's gripper action is a normalized target, not a
     signed delta.

Camera calibration source: <lerobot_dir>/camera_extrinsics.json (intrinsics +
extrinsics for cam_azure_kinect_front/left, written by
lerobot/scripts/verify_camera_calibration.py). cam_wrist has no calibration
and is intentionally stored as RGB-only with zero/identity intrinsics/extrinsics.

Robot-arm exclusion from `point_cloud` (--mask_robot_arm): MimicGen's sim
`point_cloud` observable (external/robomimic/robomimic/envs/env_robosuite.py,
`scene_pcd_ids`) explicitly drops every geom whose body name contains
"robot"/"gripper" before backprojection -- only scene/object points are kept,
the gripper is represented separately via the 4-point `gripper_pcd`. Real
depth has no per-pixel semantic labels to replicate that segmentation, so
--mask_robot_arm prompts SAM (facebookresearch/segment-anything) per frame,
per static camera with points sampled along the known 3D line from the robot
base origin (0,0,0 in the calibration file's base frame) to the current
`eef_pos`, projected into that camera's undistorted pixel space. The
highest-scoring returned mask that contains the projected EEF pixel is taken
as the arm mask, dilated a few pixels, and excluded from that camera's
backprojected points before fusion. This is NOT the same guarantee as sim's
exact geom-id segmentation -- it depends on SAM's mask quality and can
occasionally miss/over-include pixels (e.g. under motion blur or when the
arm silhouette merges with a similarly-colored object). `depth_agentview`
(the raw single-camera depth map) is left untouched, matching sim, where only
the fused `point_cloud` masks the robot out.

Example:
    python scripts/convert_lerobot_franka_to_mimicgen.py \
        --lerobot_dir /data/theya/data/uncertainity_subgoal/franka_push_block \
        --output_dir /data/theya/data/uncertainity_subgoal/franka_push_block_mimicgen_npz \
        --camera_h 256 --camera_w 256 --num_scene_points 4500 \
        --workspace_bounds 0.0 0.78 -0.40 0.30 -0.03 0.32 \
        --mask_robot_arm --mask_table --sam_checkpoint checkpoints/sam_vit_b_01ec64.pth \
        --sam_model_type vit_b --sam_device cuda:1
"""
import argparse
import json
import os
import sys
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "external" / "mimicgen"))
sys.path.insert(0, str(_REPO_ROOT / "third_party" / "robogen"))

from mimicgen.utils.articubot_util import (  # noqa: E402
    get_4_points_from_gripper_pos_orient,
    rotation_transfer_6D_to_matrix,
)
from subgoal_decomp import compute_subgoal_gripper_pcd  # noqa: E402

FINGER_TRAVEL_M = 0.04  # real Franka Hand per-finger travel, 0 (closed) .. 0.04 (open)
STATIC_CAMS = ["cam_azure_kinect_front", "cam_azure_kinect_left"]

# Modified (Craig) DH parameters for the Franka Panda arm: (alpha_{i-1}, a_{i-1}, d_i)
# per joint 1..7. Standard published values (frankaemika.github.io). Verified against
# this dataset: FK's joint7-frame origin + ~0.21m along its local z lines up with
# `observation.right_eef_pose`'s translation (0.107m flange + ~0.1034m Franka Hand TCP
# offset -- matches to within ~1cm across sampled frames), confirming both the DH table
# and the joint-angle units/order (observation.state[:7], radians) are correct.
_PANDA_DH = [
    (0.0,        0.0,     0.333),
    (-np.pi / 2, 0.0,     0.0),
    (np.pi / 2,  0.0,     0.316),
    (np.pi / 2,  0.0825,  0.0),
    (-np.pi / 2, -0.0825, 0.384),
    (np.pi / 2,  0.0,     0.0),
    (np.pi / 2,  0.088,   0.0),
]


def _dh_transform(alpha, a, d, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st, 0, a],
        [st * ca, ct * ca, -sa, -sa * d],
        [st * sa, ct * sa, ca, ca * d],
        [0, 0, 0, 1],
    ])


def panda_fk_joint_positions(q):
    """q: (7,) joint angles (rad) -> (8, 3) array of [base_origin, joint1..joint7
    frame origins] in the robot-base frame. These lie ON the physical links (not a
    straight-line approximation), which is what makes them usable as SAM prompts
    for a bent arm configuration -- a straight base->eef line prompt (the previous
    approach) can land in free space whenever the elbow is bent, which is most of
    the time for this task."""
    Tf = np.eye(4)
    pts = [Tf[:3, 3].copy()]
    for i in range(7):
        alpha, a, d = _PANDA_DH[i]
        Tf = Tf @ _dh_transform(alpha, a, d, q[i])
        pts.append(Tf[:3, 3].copy())
    return np.stack(pts, axis=0)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def load_calibration(lerobot_dir: Path):
    calib_path = lerobot_dir / "camera_extrinsics.json"
    if not calib_path.exists():
        raise FileNotFoundError(
            f"Camera calibration not found at {calib_path}. This script requires "
            "camera_extrinsics.json (intrinsics + extrinsics) to build point clouds."
        )
    with open(calib_path) as fh:
        calib = json.load(fh)

    cams = {}
    for name in STATIC_CAMS:
        c = calib["cameras"][name]
        K = np.array(c["color_intrinsics"]["K"], dtype=np.float64)
        dist = np.array(c["color_intrinsics"]["distortion"], dtype=np.float64)
        res = c["color_intrinsics"]["resolution"]  # [W, H]
        T_color_to_base = np.array(c["T_color_to_base"], dtype=np.float64)
        cams[name] = dict(K=K, dist=dist, resolution=res, T_color_to_base=T_color_to_base)
    return cams


def build_undistort_maps(K, dist, resolution):
    """Undistortion map + the new (still pinhole) intrinsics valid after remap."""
    w, h = resolution
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, new_K, (w, h), cv2.CV_32FC1)
    return map1, map2, new_K


def scale_intrinsics(K, src_wh, dst_wh):
    sx = dst_wh[0] / src_wh[0]
    sy = dst_wh[1] / src_wh[1]
    K = K.copy()
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy
    return K


# ---------------------------------------------------------------------------
# Video decoding
# ---------------------------------------------------------------------------
def decode_video_frames(path: Path, n_expected: int, gray16=False):
    container = av.open(str(path))
    frames = []
    for frame in container.decode(video=0):
        arr = frame.to_ndarray() if gray16 else frame.to_ndarray(format="rgb24")
        frames.append(arr)
    container.close()
    if len(frames) != n_expected:
        raise ValueError(
            f"{path}: decoded {len(frames)} frames, expected {n_expected} "
            "(episode length from meta/episodes.jsonl)."
        )
    return frames


# ---------------------------------------------------------------------------
# Point cloud backprojection
# ---------------------------------------------------------------------------
def depth_to_base_pointcloud(
    depth_m, K, T_color_to_base, stride=4, arm_mask=None,
    table_mask=None, table_height=None, table_tol=0.06,
):
    """depth_m: (H, W) float32 metres, already in the COLOUR grid/undistorted.
    arm_mask: (H, W) bool, True where the pixel belongs to the robot arm and
    must be EXCLUDED (mirrors sim's scene_pcd_ids body-name filter).
    table_mask: (H, W) bool, True where the pixel is bare table (already has
    the white target patch carved out, see get_table_exclusion_mask). Only
    excluded when its BACKPROJECTED base-frame z also lands within
    table_tol of table_height -- this height-gate is what stops a static,
    once-per-episode table mask from swallowing the pushed block once it
    slides into a pixel region that used to show bare table."""
    h, w = depth_m.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    z = depth_m[ys, xs]
    valid = z > 0
    if arm_mask is not None:
        valid = valid & ~arm_mask[ys, xs]
    table_flag = table_mask[ys, xs] if table_mask is not None else None
    xs, ys, z = xs[valid], ys[valid], z[valid]
    if table_flag is not None:
        table_flag = table_flag[valid]
    if len(z) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x_cam = (xs - cx) * z / fx
    y_cam = (ys - cy) * z / fy
    pts_cam = np.stack([x_cam, y_cam, z], axis=-1)  # (M, 3)

    pts_base = pts_cam @ T_color_to_base[:3, :3].T + T_color_to_base[:3, 3]

    if table_flag is not None and table_height is not None:
        near_table_height = np.abs(pts_base[:, 2] - table_height) <= table_tol
        keep = ~(table_flag & near_table_height)
        pts_base = pts_base[keep]

    return pts_base.astype(np.float32)


def crop_workspace(pts, bounds):
    """bounds: (xmin,xmax,ymin,ymax,zmin,zmax) in robot-base frame, or None."""
    if bounds is None or len(pts) == 0:
        return pts
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    keep = (
        (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax)
        & (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax)
        & (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax)
    )
    return pts[keep]


def fuse_and_subsample(point_clouds, n_points, rng, bounds=None):
    pts = np.concatenate(point_clouds, axis=0) if point_clouds else np.zeros((0, 3), np.float32)
    pts = crop_workspace(pts, bounds)
    if len(pts) == 0:
        return np.zeros((n_points, 3), dtype=np.float32)
    if len(pts) >= n_points:
        idx = rng.choice(len(pts), size=n_points, replace=False)
    else:
        idx = rng.choice(len(pts), size=n_points, replace=True)
    return pts[idx].astype(np.float32)


# ---------------------------------------------------------------------------
# SAM-based robot-arm masking (--mask_robot_arm)
# ---------------------------------------------------------------------------
def build_sam_predictor(checkpoint_path, model_type, device):
    from segment_anything import SamPredictor, sam_model_registry

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)
    return SamPredictor(sam)


def project_base_point_to_pixel(p_base, K, T_color_to_base, image_wh):
    """p_base: (3,) point in robot-base frame -> (u, v) pixel in this camera's
    undistorted image, or None if behind the camera / outside the frame."""
    Rmat = T_color_to_base[:3, :3]
    t = T_color_to_base[:3, 3]
    p_cam = Rmat.T @ (p_base - t)
    if p_cam[2] <= 1e-6:  # behind the camera
        return None
    px = K @ p_cam
    u, v = px[0] / px[2], px[1] / px[2]
    w, h = image_wh
    if not (0 <= u < w and 0 <= v < h):
        return None
    return np.array([u, v], dtype=np.float32)


def arm_prompt_points(joint_positions_base, eef_pos_base, K, T_color_to_base, image_wh):
    """Project the FK joint-chain points (base_origin..joint7, ON the physical
    links) plus the dataset's exact eef_pos (the real gripper/TCP location) into
    this camera, keeping only points landing inside the frame. These become
    SAM's foreground point prompts for segmenting the arm+gripper silhouette --
    grounded in the actual (possibly bent) arm pose, not a straight-line guess."""
    pts_3d = np.concatenate([joint_positions_base, eef_pos_base[None, :]], axis=0)
    pts = []
    for p3d in pts_3d:
        px = project_base_point_to_pixel(p3d, K, T_color_to_base, image_wh)
        if px is not None:
            pts.append(px)
    return np.stack(pts, axis=0) if pts else None


def get_arm_mask(predictor, rgb, prompt_pts, dilate_px=8, max_area_frac=0.15):
    """Returns (H, W) bool mask, True = robot-arm pixel to exclude. All-False if
    no valid prompt points landed in frame.

    Queries each FK joint point as its OWN single-point SAM prompt and unions
    the results (same reasoning as get_table_exclusion_mask): a single call with
    every point as simultaneous prompts forces SAM to find one mask containing
    all of them, which is fragile if any single point's local click is
    ambiguous -- per-point queries degrade gracefully instead.

    A joint can be occluded (behind another link, or off the visible arm from
    this viewpoint) even though its FK position is geometrically correct --
    the pixel it projects to then shows whatever background/table IS visible
    there, and a single-point click on a large flat surface returns a huge
    ambiguous SAM mask. The real arm+gripper is never anywhere near
    max_area_frac of the frame, so any single point's best mask exceeding it
    is treated as a bad click and dropped from the union rather than flooding
    the whole mask."""
    h, w = rgb.shape[:2]
    if prompt_pts is None or len(prompt_pts) == 0:
        return np.zeros((h, w), dtype=bool)

    predictor.set_image(rgb)
    mask = np.zeros((h, w), dtype=bool)
    max_area = max_area_frac * h * w
    for pt in prompt_pts:
        masks, scores, _ = predictor.predict(
            point_coords=pt[None, :], point_labels=np.array([1]), multimask_output=True
        )
        best = masks[int(np.argmax(scores))]
        if best.sum() <= max_area:
            mask |= best

    if dilate_px > 0:
        kernel = np.ones((dilate_px, dilate_px), dtype=np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return mask


# ---------------------------------------------------------------------------
# SAM-based table masking (--mask_table), with the white target patch carved
# back out. Sim's point_cloud never includes the table plane at all (only
# objects on top of it -- 'table' is explicitly in robot_body_names alongside
# 'robot'/'gripper' in env_robosuite.py's scene_pcd_ids filter), so this
# mirrors that: exclude bare table, keep the white patch (it's the task's
# goal marker, not background) and keep any object sitting on the table
# (protected by the table_height gate in depth_to_base_pointcloud, since an
# object's backprojected z won't be within table_tol of the table plane).
# ---------------------------------------------------------------------------
def detect_white_patch_mask(rgb, min_area=800, max_area=40000, dilate_px=6):
    """Colour-threshold the white target patch (high value, low saturation)
    and keep only the largest plausibly-patch-sized connected component.
    Returns (H, W) bool, True = white patch pixel (to PROTECT from the table
    exclusion), or an all-False mask if nothing plausible was found.

    The Franka Hand's white plastic housing also passes a bare
    value/saturation threshold and can be larger than the patch, so
    candidates are additionally required to (a) not touch the image's top
    row -- the arm/gripper always enters frame from the top edge in both
    static cameras here, the patch never does -- and (b) look roughly
    square and reasonably filled-in (gripper silhouettes are elongated /
    irregular, the taped-down patch is a compact square)."""
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat, val = hsv[..., 1], hsv[..., 2]
    bright = (val > 190) & (sat < 60)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(bright.astype(np.uint8), connectivity=8)
    best_label, best_area = None, 0
    for lbl in range(1, n):  # 0 is background
        x0, y0, bw, bh, area = stats[lbl]
        if not (min_area <= area <= max_area):
            continue
        if y0 <= 0:  # touches the top edge -> arm/gripper, not the patch
            continue
        aspect = bw / max(bh, 1)
        solidity = area / max(bw * bh, 1)
        if not (0.5 <= aspect <= 2.0 and solidity >= 0.5):
            continue
        if area > best_area:
            best_label, best_area = lbl, area

    mask = np.zeros((h, w), dtype=bool)
    if best_label is None:
        return mask
    mask = labels == best_label
    if dilate_px > 0:
        kernel = np.ones((dilate_px, dilate_px), dtype=np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return mask


def table_prompt_points(K, T_color_to_base, image_wh, workspace_bounds, table_height,
                         margin=0.05, grid_n=7):
    """Project a grid of known-flat 3D points spanning the workspace (at table
    height) into this camera. Physically-grounded, like arm_prompt_points --
    no manual per-camera pixel picking needed. A large flat surface can have
    shadows/scratches that split it into several SAM segments, so this
    deliberately samples a grid (not just 4 corners) -- get_table_exclusion_mask
    queries each point separately and unions the results, so coverage scales
    with how many of these land on visually-distinct patches of table."""
    if workspace_bounds is None:
        return None
    xmin, xmax, ymin, ymax, _, _ = workspace_bounds
    xs = np.linspace(xmin + margin, xmax - margin, grid_n)
    ys = np.linspace(ymin + margin, ymax - margin, grid_n)
    pts = []
    for x in xs:
        for y in ys:
            px = project_base_point_to_pixel(
                np.array([x, y, table_height]), K, T_color_to_base, image_wh
            )
            if px is not None:
                pts.append(px)
    return np.stack(pts, axis=0) if pts else None


def get_table_exclusion_mask(predictor, rgb, prompt_pts, white_patch_mask, dilate_px=4):
    """Returns (H, W) bool, True = bare-table pixel to exclude (white patch
    already carved out). All-False if no prompt points landed in frame.

    Queries each grid point as its OWN single-point SAM prompt and unions the
    resulting masks, rather than one predict() call with all points at once --
    a single merged prompt can under-segment a large flat surface when shadows
    or surface marks make SAM split it into several plausible regions."""
    h, w = rgb.shape[:2]
    if prompt_pts is None or len(prompt_pts) == 0:
        return np.zeros((h, w), dtype=bool)

    predictor.set_image(rgb)
    mask = np.zeros((h, w), dtype=bool)
    for pt in prompt_pts:
        u, v = int(round(pt[0])), int(round(pt[1]))
        if white_patch_mask[min(v, h - 1), min(u, w - 1)]:
            continue  # this grid point landed on the white patch -- skip it
        masks, scores, _ = predictor.predict(
            point_coords=pt[None, :], point_labels=np.array([1]), multimask_output=True
        )
        mask |= masks[int(np.argmax(scores))]

    if dilate_px > 0:
        kernel = np.ones((dilate_px, dilate_px), dtype=np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    return mask & ~white_patch_mask


# ---------------------------------------------------------------------------
# Mask logging (--debug_mask_dir)
# ---------------------------------------------------------------------------
def save_mask_overlay(out_path: Path, rgb, mask, color=(255, 0, 0), alpha=0.6):
    """Save rgb with `mask` alpha-blended in `color` (BGR-order file, RGB-order
    input) to out_path, creating parent dirs as needed."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = rgb.copy()
    color_arr = np.array(color, dtype=np.float32)
    overlay_f = overlay.astype(np.float32)
    overlay_f[mask] = (1 - alpha) * overlay_f[mask] + alpha * color_arr
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay_f.astype(np.uint8), cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# Output frame conversion (--output_frame camera)
# ---------------------------------------------------------------------------
def invert_rigid(T_a_to_b):
    """4x4 rigid transform a->b -> its inverse, b->a."""
    R_ab, t_ab = T_a_to_b[:3, :3], T_a_to_b[:3, 3]
    T_b_to_a = np.eye(4)
    T_b_to_a[:3, :3] = R_ab.T
    T_b_to_a[:3, 3] = -R_ab.T @ t_ab
    return T_b_to_a


def transform_points(pts, T):
    """pts: (..., 3) -> apply 4x4 rigid transform T."""
    return pts @ T[:3, :3].T + T[:3, 3]


def transform_quat(quat_xyzw, T):
    """Rotate a quaternion by the rotation part of a 4x4 rigid transform."""
    R_new = T[:3, :3] @ R.from_quat(quat_xyzw).as_matrix()
    return R.from_matrix(R_new).as_quat()


# ---------------------------------------------------------------------------
# Gripper keypoints
# ---------------------------------------------------------------------------
def eef_pose_to_gripper_pcd(pose_row):
    """pose_row: (10,) [rot6d(6), trans(3), gripper_articulation(1)] -> (gripper_pcd(4,3), quat_xyzw, joint_angle, eef_pos)"""
    rot6d = pose_row[:6]
    eef_pos = pose_row[6:9].astype(np.float64)
    gripper_norm = float(pose_row[9])  # 0=closed .. 1=open (assumption #2, see module docstring)

    Rmat = rotation_transfer_6D_to_matrix(rot6d)
    quat_xyzw = R.from_matrix(Rmat).as_quat()
    cur_joint_angle = gripper_norm * FINGER_TRAVEL_M

    gripper_pcd = get_4_points_from_gripper_pos_orient(eef_pos, quat_xyzw, cur_joint_angle)
    return gripper_pcd.astype(np.float32), quat_xyzw.astype(np.float32), cur_joint_angle, eef_pos.astype(np.float32)


# ---------------------------------------------------------------------------
# Per-episode conversion
# ---------------------------------------------------------------------------
def convert_episode(  # noqa: PLR0914
    ep_idx: int,
    lerobot_dir: Path,
    output_dir: Path,
    cams,
    camera_h: int,
    camera_w: int,
    num_scene_points: int,
    fps: float,
    lang_goal: str,
    rng: np.random.Generator,
    curvature_threshold: float,
    min_segment_len: int,
    warmup_steps: int,
    pcd_stride: int,
    workspace_bounds,
    sam_predictor=None,
    sam_dilate_px: int = 8,
    mask_table: bool = False,
    table_height: float = 0.0,
    table_tol: float = 0.06,
    output_frame: str = "base",
    debug_mask_dir: Path = None,
):
    parquet_path = lerobot_dir / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
    df = pd.read_parquet(parquet_path)
    T_full = len(df)  # original length, incl. the bad frame 0 -- needed for video decode validation

    eef_pose_full = np.stack(df["observation.right_eef_pose"].to_numpy())  # (T_full, 10)
    action_pose_full = np.stack(df["action.right_eef_pose"].to_numpy())  # (T_full, 10)
    action_joint_full = np.stack(df["action"].to_numpy())  # (T_full, 8)
    joint_angles_full = np.stack(df["observation.state"].to_numpy())[:, :7].astype(np.float64)  # (T_full, 7) rad

    # Drop frame 0: a recording artifact captures the PREVIOUS episode's last
    # frame there instead of this episode's actual first frame. Everything
    # below (including subgoal decomposition's warmup_steps) operates on the
    # trimmed T = T_full - 1 frames only, so the bogus frame never contaminates
    # goal_gripper_pcd or gets written to demo_i/0.npz.
    T = T_full - 1
    eef_pose = eef_pose_full[1:]
    action_pose = action_pose_full[1:]
    action_joint = action_joint_full[1:]
    joint_angles = joint_angles_full[1:]

    # FK joint-chain points (T, 8, 3), used to ground SAM's arm prompts on the
    # real (possibly bent) arm geometry instead of a straight base->eef line.
    fk_joint_positions = np.stack([panda_fk_joint_positions(joint_angles[t]) for t in range(T)])

    # --- Gripper keypoints for every frame ---
    gripper_pcd = np.zeros((T, 4, 3), dtype=np.float32)
    eef_quat = np.zeros((T, 4), dtype=np.float32)
    eef_qpos = np.zeros((T, 2), dtype=np.float32)
    eef_pos = np.zeros((T, 3), dtype=np.float32)
    for t in range(T):
        gp, quat, joint_angle, pos = eef_pose_to_gripper_pcd(eef_pose[t])
        gripper_pcd[t] = gp
        eef_quat[t] = quat
        eef_qpos[t] = joint_angle  # duplicate scalar -> both fingers (assumption #2)
        eef_pos[t] = pos

    # --- goal_gripper_pcd via the same subgoal decomposition convert_dataset.py uses ---
    dt = 1.0 / fps
    eef_vel_lin = np.gradient(eef_pos, dt, axis=0).astype(np.float32)
    # actions[:, -1] sign convention: mimicgen uses +1=close, -1=open. The real
    # action gripper channel is a normalized open-target in [0, 1]; approximate
    # the sign via a midpoint threshold (assumption #4).
    gripper_action_sign = np.where(action_pose[:, 9] > 0.5, -1.0, 1.0).astype(np.float32)
    pseudo_actions = np.concatenate(
        [action_joint[:, :-1], gripper_action_sign[:, None]], axis=1
    )
    goal_gripper_pcd, _switch_idxs = compute_subgoal_gripper_pcd(
        gripper_pcd=gripper_pcd,
        eef_qpos=eef_qpos,
        actions=pseudo_actions,
        eef_vel_lin=eef_vel_lin,
        curvature_threshold=curvature_threshold,
        min_segment_len=min_segment_len,
        warmup_steps=warmup_steps,
        return_switch_idxs=True,
    )

    # --- Decode + undistort camera streams ---
    static_frames = {}
    static_new_K = {}
    for cam in STATIC_CAMS:
        color_dir = lerobot_dir / "videos" / "chunk-000" / f"observation.images.{cam}.color"
        depth_dir = lerobot_dir / "videos" / "chunk-000" / f"observation.images.{cam}.transformed_depth"
        # Decode against T_full (the video file's real length), then drop the
        # same bogus frame 0 the parquet arrays already dropped above.
        color_frames = decode_video_frames(color_dir / f"episode_{ep_idx:06d}.mp4", T_full, gray16=False)[1:]
        depth_frames = decode_video_frames(depth_dir / f"episode_{ep_idx:06d}.mkv", T_full, gray16=True)[1:]

        K, dist, res = cams[cam]["K"], cams[cam]["dist"], cams[cam]["resolution"]
        map1, map2, new_K = build_undistort_maps(K, dist, res)
        static_new_K[cam] = new_K

        und_color = [cv2.remap(f, map1, map2, cv2.INTER_LINEAR) for f in color_frames]
        und_depth_mm = [
            cv2.remap(f, map1, map2, cv2.INTER_NEAREST) for f in depth_frames
        ]
        static_frames[cam] = dict(color=und_color, depth_mm=und_depth_mm)

    wrist_dir = lerobot_dir / "videos" / "chunk-000" / "observation.images.cam_wrist"
    wrist_frames = decode_video_frames(wrist_dir / f"episode_{ep_idx:06d}.mp4", T_full, gray16=False)[1:]

    # --- Table exclusion mask (--mask_table): computed ONCE per camera per
    # episode from a reference frame -- camera + table are static within an
    # episode, so this is a fixed 2D region. depth_to_base_pointcloud's
    # table_height gate is what protects the pushed block once it slides
    # into a pixel region this static mask marks as "table".
    static_table_mask = {}
    if mask_table and sam_predictor is not None:
        ref_t = 0
        for cam in STATIC_CAMS:
            rgb_ref = static_frames[cam]["color"][ref_t]
            image_wh = (rgb_ref.shape[1], rgb_ref.shape[0])
            white_patch_mask = detect_white_patch_mask(rgb_ref)
            t_prompts = table_prompt_points(
                static_new_K[cam], cams[cam]["T_color_to_base"], image_wh,
                workspace_bounds, table_height,
            )
            static_table_mask[cam] = get_table_exclusion_mask(
                sam_predictor, rgb_ref, t_prompts, white_patch_mask
            )
            if debug_mask_dir is not None:
                save_mask_overlay(
                    debug_mask_dir / f"demo_{ep_idx}" / f"table_{cam}.png",
                    rgb_ref, static_table_mask[cam], color=(255, 0, 0),
                )
                save_mask_overlay(
                    debug_mask_dir / f"demo_{ep_idx}" / f"white_patch_{cam}.png",
                    rgb_ref, white_patch_mask, color=(0, 255, 0),
                )

    ep_out_dir = output_dir / f"demo_{ep_idx}"
    ep_out_dir.mkdir(parents=True, exist_ok=True)

    # --output_frame camera: everything below is computed in robot-base frame
    # (workspace_bounds, SAM prompts, table_height gating all key off the real
    # calibration's base-frame convention) and only transformed into the
    # reference camera's (front, i.e. rgb_agentview's) own frame right before
    # writing -- T_base_to_ref is that camera's inverse extrinsic.
    T_base_to_ref = invert_rigid(cams[STATIC_CAMS[0]]["T_color_to_base"])

    for t in range(T):
        pcs = []
        for cam in STATIC_CAMS:
            depth_m = static_frames[cam]["depth_mm"][t].astype(np.float32) / 1000.0
            T_color_to_base = cams[cam]["T_color_to_base"]

            arm_mask = None
            if sam_predictor is not None:
                rgb_cam = static_frames[cam]["color"][t]
                image_wh = (rgb_cam.shape[1], rgb_cam.shape[0])
                prompt_pts = arm_prompt_points(
                    fk_joint_positions[t], eef_pos[t].astype(np.float64),
                    static_new_K[cam], T_color_to_base, image_wh
                )
                arm_mask = get_arm_mask(
                    sam_predictor, rgb_cam, prompt_pts, dilate_px=sam_dilate_px
                )
                if debug_mask_dir is not None:
                    save_mask_overlay(
                        debug_mask_dir / f"demo_{ep_idx}" / f"arm_{cam}_{t}.png",
                        rgb_cam, arm_mask, color=(255, 0, 0),
                    )

            pcs.append(
                depth_to_base_pointcloud(
                    depth_m, static_new_K[cam], T_color_to_base, stride=pcd_stride,
                    arm_mask=arm_mask,
                    table_mask=static_table_mask.get(cam),
                    table_height=table_height if mask_table else None,
                    table_tol=table_tol,
                )
            )
        point_cloud = fuse_and_subsample(pcs, num_scene_points, rng, bounds=workspace_bounds)

        # agentview = front camera, resized to (camera_h, camera_w)
        front = STATIC_CAMS[0]
        rgb_full = static_frames[front]["color"][t]
        depth_full_m = static_frames[front]["depth_mm"][t].astype(np.float32) / 1000.0
        native_wh = (rgb_full.shape[1], rgb_full.shape[0])
        rgb_resized = cv2.resize(rgb_full, (camera_w, camera_h), interpolation=cv2.INTER_AREA)
        depth_resized = cv2.resize(
            depth_full_m, (camera_w, camera_h), interpolation=cv2.INTER_NEAREST
        )
        agentview_K = scale_intrinsics(static_new_K[front], native_wh, (camera_w, camera_h))
        agentview_extrinsics = cams[front]["T_color_to_base"]

        wrist_resized = cv2.resize(
            wrist_frames[t], (camera_w, camera_h), interpolation=cv2.INTER_AREA
        )

        out_point_cloud = point_cloud
        out_gripper_pcd = gripper_pcd[t]
        out_goal_gripper_pcd = goal_gripper_pcd[t]
        out_eef_pos = eef_pos[t]
        out_eef_quat = eef_quat[t]
        out_agentview_extrinsics = agentview_extrinsics
        if output_frame == "camera":
            out_point_cloud = transform_points(point_cloud, T_base_to_ref)
            out_gripper_pcd = transform_points(gripper_pcd[t], T_base_to_ref)
            out_goal_gripper_pcd = transform_points(goal_gripper_pcd[t], T_base_to_ref)
            out_eef_pos = transform_points(eef_pos[t], T_base_to_ref)
            out_eef_quat = transform_quat(eef_quat[t], T_base_to_ref)
            # points are now already in the front camera's own frame, so the
            # transform FROM that camera TO the point frame is the identity.
            out_agentview_extrinsics = np.eye(4, dtype=np.float32)

        np.savez_compressed(
            ep_out_dir / f"{t}.npz",
            point_cloud=out_point_cloud.astype(np.float32)[None, :],
            gripper_pcd=out_gripper_pcd.astype(np.float32)[None, :],
            goal_gripper_pcd=out_goal_gripper_pcd.astype(np.float32)[None, :],
            rgb_agentview=rgb_resized[None, :].astype(np.uint8),
            depth_agentview=depth_resized[None, :, :, None].astype(np.float32),
            agentview_intrinsics=agentview_K[None, :].astype(np.float32),
            agentview_extrinsics=out_agentview_extrinsics.astype(np.float32)[None, :],
            # cam_wrist has no extrinsic calibration (verified absent in
            # camera_extrinsics.json) -> RGB only, per project decision.
            rgb_wrist=wrist_resized[None, :].astype(np.uint8),
            depth_wrist=np.zeros((1, camera_h, camera_w, 1), dtype=np.float32),
            wrist_intrinsics=np.zeros((1, 3, 3), dtype=np.float32),
            wrist_extrinsics=np.eye(4, dtype=np.float32)[None, :],
            eef_pos=out_eef_pos.astype(np.float32)[None, :],
            eef_quat=out_eef_quat.astype(np.float32)[None, :],
            gripper_qpos=eef_qpos[t][None, :],
            action=action_pose[t][None, :],
            # NpyDataset.__getitem__ never reads "state" for HL training
            # (see assumption #3) -- kept only for schema parity.
            state=np.zeros((1, 10), dtype=np.float32),
            lang_goal=np.array([lang_goal], dtype=object),
        )

    return T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lerobot_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=None,
                         help="subset of episode indices to convert (default: all)")
    parser.add_argument("--camera_h", type=int, default=256)
    parser.add_argument("--camera_w", type=int, default=256)
    parser.add_argument("--num_scene_points", type=int, default=4500)
    parser.add_argument("--pcd_stride", type=int, default=4,
                         help="pixel stride when backprojecting depth (before subsampling)")
    parser.add_argument("--workspace_bounds", type=float, nargs=6, default=None,
                         metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                         help="robot-base-frame crop applied to the fused point_cloud "
                              "before subsampling (drops floor/background). No default -- "
                              "measure your table extents and pass explicitly, e.g. "
                              "0.2 0.9 -0.5 0.5 -0.05 0.5")
    parser.add_argument("--curvature_threshold", type=float, default=0.5)
    parser.add_argument("--min_segment_len", type=int, default=10)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mask_robot_arm", action="store_true",
                         help="use SAM to exclude the robot arm from the fused "
                              "point_cloud, mirroring sim's scene_pcd_ids body filter "
                              "(see module docstring). Off by default.")
    parser.add_argument("--sam_checkpoint", type=str,
                         default="checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--sam_model_type", type=str, default="vit_b",
                         choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--sam_device", type=str, default="cuda:1")
    parser.add_argument("--sam_dilate_px", type=int, default=8,
                         help="dilation applied to the SAM arm mask before exclusion, "
                              "to cover anti-aliased/motion-blurred edges")
    parser.add_argument("--mask_table", action="store_true",
                         help="use SAM to also exclude the bare table surface from the "
                              "fused point_cloud (sim never includes the table plane "
                              "either -- see module docstring), while explicitly "
                              "preserving the white target patch and any object sitting "
                              "on the table (height-gated, see --table_height/--table_tol). "
                              "Requires --mask_robot_arm (reuses the same SAM predictor).")
    parser.add_argument("--table_height", type=float, default=0.0,
                         help="table plane's z in the robot-base frame (metres), used both "
                              "to pick the SAM table-prompt points and to height-gate the "
                              "table exclusion so a moving object is never swept up in it")
    parser.add_argument("--table_tol", type=float, default=0.06,
                         help="+/- z band (metres) around --table_height treated as bare "
                              "table for the height gate. Default 0.06: measured backprojection "
                              "error grows toward the table's edges (observed up to ~4.5cm here, "
                              "beyond the camera_extrinsics.json accuracy block's stated typical "
                              "~2-3cm) -- a tighter value leaves real table points unexcluded "
                              "near those edges.")
    parser.add_argument("--robot_base_frame", action="store_true",
                         help="save point_cloud/gripper_pcd/goal_gripper_pcd/eef_pos/eef_quat "
                              "in robot-base frame instead of the default (front/agentview "
                              "camera's own frame). Everything upstream (workspace crop, SAM "
                              "prompts, table height gate) always runs in robot-base frame "
                              "internally regardless of this flag -- it only affects the final "
                              "saved arrays. In camera mode (default), agentview_extrinsics is "
                              "identity since the points are already in that camera's frame.")
    parser.add_argument("--debug_mask_dir", type=str, default=None,
                         help="if set, save SAM mask overlays (red=excluded, green=protected "
                              "white patch) as PNGs under <dir>/demo_i/: arm_<cam>_<t>.png per "
                              "frame/camera (only when --mask_robot_arm), table_<cam>.png and "
                              "white_patch_<cam>.png once per episode/camera (only when "
                              "--mask_table). No files written otherwise.")
    args = parser.parse_args()
    args.output_frame = "base" if args.robot_base_frame else "camera"

    if args.mask_table and not args.mask_robot_arm:
        parser.error("--mask_table requires --mask_robot_arm (shares the SAM predictor)")

    lerobot_dir = Path(args.lerobot_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(lerobot_dir / "meta" / "info.json") as fh:
        info = json.load(fh)
    fps = float(info["fps"])

    with open(lerobot_dir / "meta" / "tasks.jsonl") as fh:
        task_line = json.loads(fh.readline())
    lang_goal = task_line["task"]

    episodes_meta = []
    with open(lerobot_dir / "meta" / "episodes.jsonl") as fh:
        for line in fh:
            episodes_meta.append(json.loads(line))
    n_total = len(episodes_meta)

    episode_ids = args.episodes if args.episodes is not None else list(range(n_total))

    cams = load_calibration(lerobot_dir)
    rng = np.random.default_rng(args.seed)

    sam_predictor = None
    if args.mask_robot_arm:
        print(f"[sam] loading {args.sam_model_type} from {args.sam_checkpoint} "
              f"on {args.sam_device} ...")
        sam_predictor = build_sam_predictor(
            args.sam_checkpoint, args.sam_model_type, args.sam_device
        )

    for ep_idx in episode_ids:
        n_steps = convert_episode(
            ep_idx=ep_idx,
            lerobot_dir=lerobot_dir,
            output_dir=output_dir,
            cams=cams,
            camera_h=args.camera_h,
            camera_w=args.camera_w,
            num_scene_points=args.num_scene_points,
            fps=fps,
            lang_goal=lang_goal,
            rng=rng,
            curvature_threshold=args.curvature_threshold,
            min_segment_len=args.min_segment_len,
            warmup_steps=args.warmup_steps,
            pcd_stride=args.pcd_stride,
            workspace_bounds=args.workspace_bounds,
            sam_predictor=sam_predictor,
            sam_dilate_px=args.sam_dilate_px,
            mask_table=args.mask_table,
            table_height=args.table_height,
            table_tol=args.table_tol,
            output_frame=args.output_frame,
            debug_mask_dir=Path(args.debug_mask_dir) if args.debug_mask_dir else None,
        )
        print(f"[done] demo_{ep_idx}: {n_steps} steps")

    print(f"Wrote {len(episode_ids)} episodes to {output_dir}")


if __name__ == "__main__":
    main()
