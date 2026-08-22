#!/usr/bin/env python
"""
Convert a LeRobot Franka dataset (2x calibrated Azure Kinect + 1x wrist ZED) into
per-timestep .npz files for the low-level policy.

WHAT THIS WRITES (per timestep, demo_<i>/<t>.npz)
------------------------------------------------
    rgb_agentview           (1, H, W, 3) uint8
    depth_agentview         (1, H, W, 1) float32  METRES
    agentview_intrinsics    (1, 3, 3)    float32  for (H, W), post-undistortion
    agentview_extrinsics    (1, 4, 4)    float32  CAMERA-TO-BASE (c2w)
    rgb_wrist / depth_wrist / wrist_intrinsics / wrist_extrinsics   -- "cam1",
                            same layout; which physical camera fills these is
                            chosen by --cam1 (see below)
    gripper_pcd             (1, 4, 3)    float32  4 gripper keypoints, base frame
    goal_gripper_pcd        (1, 4, 3)    float32  subgoal keypoints, base frame
    eef_pos / eef_quat / gripper_qpos / action / state / lang_goal

STATE / ACTION CONVENTION
-------------------------
    state   (1, 10)  [xyz(3), rot6d(6), gripper(1)]  absolute measured pose,
                     gripper = normalized width, 0 closed .. 1 open
    action  (1, 10)  [dxyz(3), rot6d(6), gripper(1)]  HYBRID DELTA obeying the
                     trainer's rule  state[t] (+) action[t] = state[t+1]:
                     world-frame dxyz, BODY-frame dR (R_{t+1} = R_t @ dR),
                     gripper = binary open/close scaled to +-0.01
                     (+0.01 open, -0.01 close, matching D1).

Neither field can be copied out of the parquet as recorded. The recording orders
its 10-D pose [rot6d, xyz, gripper] and encodes the rotation with pytorch3d's
row convention, and `action.right_eef_pose` is not a command at all -- its pose
channels are a duplicate of the measured eef. See the convention block and
panda_fk() below for the full story and the fix. --action_source selects where
the delta's target comes from.

    point_cloud             (1, N, 3)    float32  fused scene cloud, base frame

POINT CLOUD: KEPT. SAM MASKING: REMOVED
---------------------------------------
`point_cloud` still fuses both calibrated Kinects, crops to --workspace_bounds
and subsamples to --num_scene_points, exactly as before. What is gone is the SAM
2 segmentation that used to carve the robot arm and the table plane out of it
first, along with the scene-prompt picker and the GPU requirement.

The consequence is real and worth stating: the cloud now CONTAINS the robot arm
and the table surface. Sim's scene_pcd_ids filter drops both, so a real cloud
from this script is no longer distribution-matched to a sim one. --workspace_bounds
is now the ONLY thing bounding it, which makes those numbers matter more than
they used to -- measure your table and pass them explicitly.

Approach 2 is unaffected either way: it never reads obs/point_cloud (absent from
every *_goal_gmm_aux shape_meta, and LazyArticuBotDataset loads only declared
keys). The cloud is here for the GMM high-level path and anything else that wants
it. Note `gripper_pcd` / `goal_gripper_pcd` are NOT sensor clouds -- they are 4
keypoints derived analytically from the EEF pose.

EXTRINSIC CONVENTION -- do not "fix" this
-----------------------------------------
Both *_extrinsics arrays are CAMERA-TO-BASE (c2w): p_base = T[:3,:3] @ p_cam +
T[:3,3], matching camera_extrinsics.json's stated convention. That is what
Approach 2 wants: grounded_encoder.py passes E straight into
unproject_depth_to_world(depth, K, extrinsic_c2w) with NO inversion.

Be aware that model/vision/rope_3d.py documents the opposite ("dataset stores
w2c") and inverts internally -- that is a DIFFERENT encoder used by a different
policy. Writing w2c here would silently misplace every patch token for
Approach 2.

CAM1 SELECTION (--cam1)
-----------------------
  kinect_left  (default)  The left Azure Kinect. Fully calibrated in
                          camera_extrinsics.json and recorded with
                          transformed_depth, so this works on data you already
                          have. Static camera -> one extrinsic for all frames.

  wrist                   The wrist ZED. Requires BOTH:
                            * a depth stream in the dataset -- only present if it
                              was recorded with ZedCameraConfig(use_depth=True);
                              older datasets are colour-only and this script will
                              say so rather than emit zeros, and
                            * --wrist_calibration JSON carrying the ZED's
                              intrinsics AND T_eef_cam, the constant camera->EEF
                              mount transform from an EYE-IN-HAND calibration.
                          The wrist camera MOVES, so its extrinsic is rebuilt
                          every frame as T_base_eef(t) @ T_eef_cam.

  none                    Zero-fill cam1. Only for single-camera runs where the
                          task shape_meta declares no cam1_* keys.

Camera calibration source: <lerobot_dir>/camera_extrinsics.json (intrinsics +
extrinsics for cam_azure_kinect_front/left, written by
lerobot/scripts/verify_camera_calibration.py).

Example:
    python scripts/convert_lerobot_franka_to_mimicgen.py \
        --lerobot_dir /data/theya/uncertain_subgoal_data/franka_push_block \
        --output_dir  /data/theya/uncertain_subgoal_data/franka_push_block_npz \
        --cam1 kinect_left --camera_h 256 --camera_w 256
"""
import argparse
import json
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



def _load_articubot_util():
    """Load articubot_util directly from its file, bypassing mimicgen/__init__.

    We only want two pure-maths helpers out of it, but importing it as
    `mimicgen.utils.articubot_util` executes mimicgen/__init__.py first, which
    pulls in robosuite and mujoco. That is noisy at best (the "robosuite
    WARNING" banner on every run) and fatal at worst -- in the low_level pixi
    env mujoco.egl and PyOpenGL disagree and the import dies with
    `EGL has no attribute EGLDeviceEXT`. articubot_util itself needs only
    numpy/scipy/torch, so loading the file directly sidesteps all of it.
    """
    import importlib.util
    path = _REPO_ROOT / "external" / "mimicgen" / "mimicgen" / "utils" / "articubot_util.py"
    spec = importlib.util.spec_from_file_location("_articubot_util", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_au = _load_articubot_util()
get_4_points_from_gripper_pos_orient = _au.get_4_points_from_gripper_pos_orient

from subgoal_decomp import compute_subgoal_gripper_pcd  # noqa: E402


def rotation_6d_to_matrix_recorded(rot6d):
    """Decode observation/action.right_eef_pose's 6D rotation as it was actually
    ENCODED at record time.

    control_utils.py's add_eef_pose() encodes the real robot's rotation with
    pytorch3d.transforms.matrix_to_rotation_6d, which packs the first two ROWS
    of R into the 6D vector and does NOT transpose the Gram-Schmidt result.
    mimicgen's articubot_util.rotation_transfer_6D_to_matrix -- used
    everywhere else in the wider articubot/mimicgen/robogen ecosystem this
    repo builds on -- decodes the OTHER convention (first two COLUMNS, i.e.
    Zhou et al.'s original layout), which is a `.T` away from pytorch3d's.

    That mismatch is silent (both produce a valid rotation matrix) and huge:
    round-tripping through record-encode -> mimicgen-decode gives a median
    ~89 deg, max ~180 deg error on random rotations. It has no effect on
    STATIC-camera geometry (T_color_to_base never touches eef_pose) but
    corrupts everything derived from observation.right_eef_pose's rotation
    here: eef_quat, gripper_pcd, goal_gripper_pcd, and -- critically -- the
    wrist camera's per-frame extrinsic T_base_eef(t) @ T_eef_cam, which is
    exactly why the wrist cloud was misaligned while the two Kinects agreed
    with each other fine.

    This is mimicgen's own Gram-Schmidt with the final transpose removed --
    verified to agree with pytorch3d.transforms.rotation_6d_to_matrix to
    <0.02 deg over 3000 random rotations -- so no new dependency is needed
    just to undo one `.T`.

    Do NOT swap articubot_util.rotation_transfer_6D_to_matrix itself: it is
    used by many other, unrelated consumers (robogen, diffusion_policy,
    mino_utils, ...) that expect its native column convention, presumably
    because they're self-consistent with a simulation encoder that already
    matches it. The bug is specific to this real-robot recording path.
    """
    d6 = np.asarray(rot6d, dtype=np.float64).reshape(2, 3)
    a1, a2 = d6[0], d6[1]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=0)


# ---------------------------------------------------------------------------
# 10-D state/action convention  (see low_level .../common/action_util.py:74-104
# and .../dataset/lazy_articubot_dataset.py:498-535)
# ---------------------------------------------------------------------------
# The trainer's contract is  state[t] (+) action[t] = target[t], with
#     ch 0:3  world-frame  dxyz      xyz_target = xyz_t + dxyz
#     ch 3:9  BODY-frame   dR (6D)   R_target   = R_t @ dR
#     ch 9    gripper                additive
# and BOTH state and action order the channels [xyz(3), rot6d(6), gripper(1)].
#
# The 6D rotation encoding used by the trainer is the FIRST TWO COLUMNS of R:
# manipulation/utils.py:1055 rotation_transfer_matrix_to_6D_batch() just takes
# the leading 6 numbers of the flattened input, and every call site feeds it
# R.transpose(0, 2, 1) -- so what lands in the file is [R[:,0], R[:,1]].
# rotation_transfer_6D_to_matrix_batch_mino() inverts exactly that.
#
# This is NOT the convention the raw recording uses: add_eef_pose() encodes with
# pytorch3d (first two ROWS) -- see rotation_6d_to_matrix_recorded() above. Every
# rotation read out of the parquet must therefore be decoded row-wise and
# re-encoded column-wise before it is written to the npz.

def rot6d_from_matrix_mimicgen(Rmat):
    """R (3,3) -> 6D in the trainer's column convention: [R[:,0], R[:,1]]."""
    Rmat = np.asarray(Rmat, dtype=np.float64).reshape(3, 3)
    return np.concatenate([Rmat[:, 0], Rmat[:, 1]])


# --- Franka Panda forward kinematics -----------------------------------------
# Only used by --action_source fk_commanded.
#
# deoxys itself never needs FK: it reads the measured O_T_EE straight out of
# libfranka's robot state (franka_interface.py:543 last_eef_rot_and_pos). The
# COMMANDED pose was equally available live -- libfranka publishes q_d and
# O_T_EE_d, and deoxys wraps the former as last_q_d -- but add_eef_pose() read
# neither, so the parquet holds only the measured pose in both fields (see the
# state/action block above). Offline, with just the 8-D GELLO joint command to
# work from, FK is the only way back to a commanded pose.
#
# lerobot/.../robot_devices/robots/franka_kinematics.py already implements it
# ("deoxys ships inverse kinematics (mujoco-backed) but no forward kinematics"),
# and reports matching this rig's own O_T_EE to 0.00 mm / 0.000 deg -- so load
# that rather than keeping a second copy of the DH table in sync. It is pure
# numpy, so path-loading it costs nothing even though lerobot is a sibling repo.
_LEROBOT_FK = (_REPO_ROOT.parent / "lerobot" / "lerobot" / "common"
               / "robot_devices" / "robots" / "franka_kinematics.py")


def _load_franka_kinematics():
    import importlib.util
    if not _LEROBOT_FK.exists():
        raise FileNotFoundError(
            f"--action_source fk_commanded needs {_LEROBOT_FK}, which is not "
            f"there. Pass --action_source achieved, or point this at the "
            f"lerobot checkout."
        )
    spec = importlib.util.spec_from_file_location("_franka_kinematics", _LEROBOT_FK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def panda_fk(joints7):
    """(7,) joint angles -> (4,4) base->TCP transform, i.e. deoxys' O_T_EE."""
    global _fk_mod
    try:
        _fk_mod
    except NameError:
        _fk_mod = _load_franka_kinematics()
    return _fk_mod.fk_tcp(np.asarray(joints7, dtype=np.float64))


def _fk_selfcheck(obs_joint, eef_pose, ep_idx):
    """FK(observation joints) must reproduce observation.right_eef_pose.

    This is the only check that the DH chain, the flange offset and the row-wise
    6D decode all agree. It is a sanity gate, not a tight tolerance: the joint
    stream and last_eef_rot_and_pos are two separate hardware reads a few ms
    apart, so a few mm of disagreement during fast motion is expected and does
    not affect the delta (which is FK-to-FK and cancels any constant bias).
    """
    n = len(obs_joint)
    idx = np.arange(0, n, max(1, n // 64))
    pos_err, rot_err = [], []
    for t in idx:
        T = panda_fk(obs_joint[t, :7])
        pos_err.append(np.linalg.norm(T[:3, 3] - eef_pose[t, 6:9]))
        dR = T[:3, :3].T @ rotation_6d_to_matrix_recorded(eef_pose[t, :6])
        rot_err.append(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
    pos_med, rot_med = float(np.median(pos_err)), float(np.median(rot_err))
    if pos_med > 0.02 or rot_med > 2.0:
        print(f"  [WARN] ep{ep_idx}: FK disagrees with recorded eef pose "
              f"(median {pos_med * 1000:.1f} mm / {rot_med:.2f} deg). The commanded "
              f"action delta is probably wrong -- check the DH chain or pass "
              f"--action_source achieved.")
    return pos_med, rot_med


def build_state_action(eef_pose, obs_joint, action_joint, action_source, ep_idx):
    """(T,10) recorded eef pose + (T,8) joint streams -> (state, action), both
    (T,10) in the trainer's [xyz, rot6d, gripper] layout.

    state  : the measured absolute pose.
    action : a hybrid delta obeying state[t] (+) action[t] = target[t].

    action_source:
      'achieved' (default) -- target is the measured pose at t+1, so
          state[t] (+) action[t] == state[t+1] holds exactly and the action is
          simply what the arm did. Nothing outside this file has to be trusted:
          no DH chain, no assumption that GELLO joints map 1:1 to the follower,
          no reliance on a field the recorder got wrong. The one cost is scale
          -- these deltas are ~5x smaller than D1's, which store a controller
          command the arm only partly tracks (see below).
      'fk_commanded' -- target is FK(commanded joints), i.e. where the
          teleoperator asked the arm to go, which is what D1 actually holds.
          Measured on this dataset the arm achieves 0.18-0.22x the commanded
          translation and 0.11-0.12x the commanded rotation per step, matching
          D1's 0.16-0.22x / 0.07-0.16x tracking ratios. Use this if you are
          mixing real and D1 data under one normalizer.
    """
    T = len(eef_pose)
    R_meas = np.stack([rotation_6d_to_matrix_recorded(eef_pose[t, :6]) for t in range(T)])
    xyz_meas = eef_pose[:, 6:9].astype(np.float64)

    state = np.zeros((T, 10), dtype=np.float32)
    state[:, :3] = xyz_meas
    for t in range(T):
        state[t, 3:9] = rot6d_from_matrix_mimicgen(R_meas[t])
    # Channel 9 stays the measured normalized width (0=closed .. 1=open), which
    # is what obs/state means everywhere else in the pipeline.
    state[:, 9] = eef_pose[:, 9]

    if action_source == "fk_commanded":
        _fk_selfcheck(obs_joint, eef_pose, ep_idx)
        # FK both streams and difference them. Doing it FK-to-FK rather than
        # against the recorded pose makes the delta invariant to any constant
        # error in the DH chain or the flange offset, and sidesteps the few-ms
        # skew between the joint and eef reads.
        T_obs = np.stack([panda_fk(obs_joint[t, :7]) for t in range(T)])
        T_cmd = np.stack([panda_fk(action_joint[t, :7]) for t in range(T)])
        dxyz = T_cmd[:, :3, 3] - T_obs[:, :3, 3]
        dR = np.einsum("tij,tjk->tik", T_obs[:, :3, :3].transpose(0, 2, 1), T_cmd[:, :3, :3])
    elif action_source == "achieved":
        dxyz = np.zeros((T, 3))
        dxyz[:-1] = xyz_meas[1:] - xyz_meas[:-1]
        dR = np.tile(np.eye(3), (T, 1, 1))
        dR[:-1] = np.einsum("tij,tjk->tik", R_meas[:-1].transpose(0, 2, 1), R_meas[1:])
    else:
        raise ValueError(f"unknown action_source: {action_source}")

    action = np.zeros((T, 10), dtype=np.float32)
    action[:, :3] = dxyz
    for t in range(T):
        action[t, 3:9] = rot6d_from_matrix_mimicgen(dR[t])
    # Gripper: D1 carries a binary open/close command scaled to +-0.01, not a
    # width delta -- mimicgen/scripts/convert_dataset.py:187 does
    # `gripper *= -0.01` on the robosuite action, where MimicGen uses +1=close
    # and -1=open. So +0.01 = OPEN, -0.01 = CLOSE. The real command channel is a
    # normalized open-target that only ever takes {0, 1} (verified: cmd=1 is
    # followed by widening, cmd=0 by narrowing), so 1 -> +0.01, 0 -> -0.01.
    action[:, 9] = 0.01 * (2.0 * _grip_cmd(action_joint) - 1.0)
    return state, action


def _grip_cmd(action_joint):
    """Binary open(1)/close(0) command from the 8-D GELLO action's last channel."""
    return (action_joint[:, -1] > 0.5).astype(np.float64)


FINGER_TRAVEL_M = 0.04  # real Franka Hand per-finger travel, 0 (closed) .. 0.04 (open)
STATIC_CAMS = ["cam_azure_kinect_front", "cam_azure_kinect_left"]
AGENTVIEW_CAM = "cam_azure_kinect_front"


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def load_calibration(lerobot_dir: Path, calib_path: Path = None):
    """Kinect intrinsics + extrinsics.

    Defaults to <lerobot_dir>/camera_extrinsics.json, but newer recordings do not
    always carry a copy, so --calibration can point at the snapshot directly.
    """
    calib_path = Path(calib_path) if calib_path else lerobot_dir / "camera_extrinsics.json"
    if not calib_path.exists():
        raise FileNotFoundError(
            f"Camera calibration not found at {calib_path}. Pass --calibration "
            "pointing at the camera_extrinsics.json for the rig this was recorded on."
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


def load_wrist_calibration(path: Path):
    """Intrinsics + T_eef_cam for the wrist ZED.

    Expected JSON (a superset of what lerobot/scripts/dump_zed_intrinsics.py
    writes -- that script emits the intrinsics half; T_eef_cam has to come from
    an eye-in-hand calibration and be merged in):

        {"cameras": {"cam_wrist": {
            "color_intrinsics": {"K": [[..]], "distortion": [..], "resolution": [W, H]},
            "T_eef_cam": [[..4x4..]]
        }}}

    T_eef_cam maps a point in the CAMERA frame to the EEF frame, so the
    per-frame camera-to-base transform is T_base_eef(t) @ T_eef_cam.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"--cam1 wrist needs a wrist calibration, not found at {path}. "
            "Generate the intrinsics half with lerobot/scripts/dump_zed_intrinsics.py "
            "and add T_eef_cam from an eye-in-hand calibration."
        )
    with open(path) as fh:
        calib = json.load(fh)

    try:
        c = calib["cameras"]["cam_wrist"]
    except KeyError as exc:
        raise KeyError(f"{path}: expected cameras.cam_wrist") from exc

    if "T_eef_cam" not in c:
        raise KeyError(
            f"{path}: cameras.cam_wrist has no 'T_eef_cam'. Intrinsics alone are "
            "not enough -- the wrist camera moves with the gripper, so without the "
            "camera->EEF mount transform its patch tokens cannot be placed in the "
            "base frame. This comes from an EYE-IN-HAND calibration; note that "
            "estimated_tag_to_hand.npz is eye-to-hand and does NOT contain it."
        )

    K = np.array(c["color_intrinsics"]["K"], dtype=np.float64)
    dist = np.array(c["color_intrinsics"].get("distortion", [0.0] * 8), dtype=np.float64)
    res = c["color_intrinsics"]["resolution"]
    T_eef_cam = np.array(c["T_eef_cam"], dtype=np.float64)
    if T_eef_cam.shape != (4, 4):
        raise ValueError(f"{path}: T_eef_cam must be 4x4, got {T_eef_cam.shape}")

    if not np.any(K):
        raise ValueError(f"{path}: wrist K is all-zero -- unprojection would divide by zero.")

    return dict(K=K, dist=dist, resolution=res, T_eef_cam=T_eef_cam)


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


def find_wrist_depth_dir(lerobot_dir: Path):
    """Locate the wrist depth stream, or return None if it was never recorded.

    ZedCameraConfig names it `<cam>.depth`, unlike the Kinects'
    `<cam>.transformed_depth`; older datasets recorded the wrist colour-only
    (use_depth=False) and have neither.
    """
    videos = lerobot_dir / "videos" / "chunk-000"
    for name in ("observation.images.cam_wrist.depth",
                 "observation.images.cam_wrist.transformed_depth"):
        if (videos / name).is_dir():
            return videos / name
    return None


def find_wrist_color_dir(lerobot_dir: Path):
    """Wrist colour, which franka_2cam records under the FLAT key for ZEDs."""
    videos = lerobot_dir / "videos" / "chunk-000"
    for name in ("observation.images.cam_wrist",
                 "observation.images.cam_wrist.color"):
        if (videos / name).is_dir():
            return videos / name
    raise FileNotFoundError(f"No wrist colour stream under {videos}")


# ---------------------------------------------------------------------------
# Point cloud backprojection
# ---------------------------------------------------------------------------
def depth_to_base_pointcloud(depth_m, K, T_color_to_base, stride=4):
    """depth_m: (H, W) float32 metres, already undistorted onto the colour grid.

    Unmasked: every valid depth pixel is kept. The SAM arm/table exclusion that
    used to run here is gone, so the fused cloud INCLUDES the robot arm and the
    table plane -- unlike sim's scene_pcd_ids filter, which drops both. Crop with
    --workspace_bounds to bound it.
    """
    h, w = depth_m.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    z = depth_m[ys, xs]
    valid = z > 0
    xs, ys, z = xs[valid], ys[valid], z[valid]
    if len(z) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pts_cam = np.stack([(xs - cx) * z / fx, (ys - cy) * z / fy, z], axis=-1)
    pts_base = pts_cam @ T_color_to_base[:3, :3].T + T_color_to_base[:3, 3]
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
    idx = rng.choice(len(pts), size=n_points, replace=len(pts) < n_points)
    return pts[idx].astype(np.float32)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def eef_pose_to_gripper_pcd(pose_row):
    """pose_row: (10,) [rot6d(6), trans(3), gripper_articulation(1)]
    -> (gripper_pcd(4,3), quat_xyzw, joint_angle, eef_pos)"""
    rot6d = pose_row[:6]
    eef_pos = pose_row[6:9].astype(np.float64)
    gripper_norm = float(pose_row[9])  # 0=closed .. 1=open

    Rmat = rotation_6d_to_matrix_recorded(rot6d)
    quat_xyzw = R.from_matrix(Rmat).as_quat()
    cur_joint_angle = gripper_norm * FINGER_TRAVEL_M

    gripper_pcd = get_4_points_from_gripper_pos_orient(eef_pos, quat_xyzw, cur_joint_angle)
    return gripper_pcd.astype(np.float32), quat_xyzw.astype(np.float32), cur_joint_angle, eef_pos.astype(np.float32)


def T_base_eef_from(quat_xyzw, pos):
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T


def prepare_camera(cam_calib, camera_w, camera_h, crop_lr=0):
    """Undistort maps + the intrinsics valid for the final (camera_w, camera_h).

    crop_lr trims that many columns off BOTH the left and right edge of the
    undistorted full-res image before the resize. That is a translation of the
    image origin, so the principal point moves with it (cx -= crop_lr) and the
    subsequent scaling runs from the CROPPED width, not the native one --
    scale_intrinsics only scales, it knows nothing about a crop offset.
    """
    map1, map2, new_K = build_undistort_maps(
        cam_calib["K"], cam_calib["dist"], cam_calib["resolution"]
    )
    native_w, native_h = tuple(cam_calib["resolution"])
    if crop_lr:
        if 2 * crop_lr >= native_w:
            raise ValueError(
                f"--crop_lr {crop_lr} removes {2 * crop_lr}px from a {native_w}px-wide "
                "image, leaving nothing."
            )
        new_K = new_K.copy()
        new_K[0, 2] -= crop_lr
    K_out = scale_intrinsics(new_K, (native_w - 2 * crop_lr, native_h), (camera_w, camera_h))
    return map1, map2, K_out


def crop_lr_pair(rgb, depth_m, crop_lr):
    """Symmetric width crop of an undistorted full-res (rgb, depth) pair.

    Applied to the camera IMAGES only. The fused `point_cloud` is backprojected
    from the uncropped native-resolution depth with its own full-res intrinsics,
    so it keeps the full horizontal field of view regardless of this setting.
    """
    if not crop_lr:
        return rgb, depth_m
    return rgb[:, crop_lr:-crop_lr], depth_m[:, crop_lr:-crop_lr]


def resize_pair(rgb, depth_m, camera_w, camera_h):
    rgb_r = cv2.resize(rgb, (camera_w, camera_h), interpolation=cv2.INTER_AREA)
    # NEAREST for depth: averaging across a depth discontinuity invents surfaces
    # that sit between the foreground and the background.
    depth_r = cv2.resize(depth_m, (camera_w, camera_h), interpolation=cv2.INTER_NEAREST)
    return rgb_r, depth_r


# ---------------------------------------------------------------------------
# Episode conversion
# ---------------------------------------------------------------------------
def convert_episode(
    lerobot_dir: Path,
    ep_idx: int,
    out_dir: Path,
    cams,
    fps: float,
    lang_goal: str,
    camera_h: int,
    camera_w: int,
    crop_lr: int,
    cam1_mode: str,
    wrist_calib,
    num_scene_points: int,
    pcd_stride: int,
    workspace_bounds,
    rng,
    curvature_threshold: float,
    min_segment_len: int,
    warmup_steps: int,
    action_source: str = "achieved",
):
    parquet_path = lerobot_dir / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
    df = pd.read_parquet(parquet_path)
    T = len(df)

    eef_pose = np.stack(df["observation.right_eef_pose"].to_numpy())      # (T, 10)
    action_pose = np.stack(df["action.right_eef_pose"].to_numpy())        # (T, 10)
    action_joint = np.stack(df["action"].to_numpy())                      # (T, 8)
    obs_joint = np.stack(df["observation.state"].to_numpy())              # (T, 8)

    # 10-D state/action in the trainer's convention. NOTE: `action_pose` is NOT
    # used for the action -- its pose channels are a copy of the measured eef
    # (see panda_fk's comment block); it survives only to drive the subgoal
    # decomposition's gripper sign below.
    state10, action10 = build_state_action(
        eef_pose, obs_joint, action_joint, action_source, ep_idx
    )

    # --- Gripper keypoints for every frame ---
    gripper_pcd = np.zeros((T, 4, 3), dtype=np.float32)
    eef_quat = np.zeros((T, 4), dtype=np.float32)
    eef_qpos = np.zeros((T, 2), dtype=np.float32)
    eef_pos = np.zeros((T, 3), dtype=np.float32)
    for t in range(T):
        gp, quat, joint_angle, pos = eef_pose_to_gripper_pcd(eef_pose[t])
        gripper_pcd[t] = gp
        eef_quat[t] = quat
        eef_qpos[t] = joint_angle  # duplicate scalar -> both fingers
        eef_pos[t] = pos

    # --- goal_gripper_pcd via the shared subgoal decomposition ---
    dt = 1.0 / fps
    eef_vel_lin = np.gradient(eef_pos, dt, axis=0).astype(np.float32)
    # actions[:, -1] sign convention: mimicgen uses +1=close, -1=open. The real
    # action gripper channel is a normalized open-target in [0, 1]; approximate
    # the sign via a midpoint threshold.
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

    videos = lerobot_dir / "videos" / "chunk-000"

    # --- agentview (front Kinect) ---
    front = AGENTVIEW_CAM
    map1_a, map2_a, agentview_K = prepare_camera(cams[front], camera_w, camera_h, crop_lr)
    color_a = decode_video_frames(
        videos / f"observation.images.{front}.color" / f"episode_{ep_idx:06d}.mp4", T)
    depth_a = decode_video_frames(
        videos / f"observation.images.{front}.transformed_depth" / f"episode_{ep_idx:06d}.mkv",
        T, gray16=True)
    agentview_extrinsics = cams[front]["T_color_to_base"]

    # --- both Kinects, undistorted, for the fused scene cloud. This is
    # independent of --cam1: the cloud always fuses the two calibrated static
    # cameras, whichever one also fills the cam1 image slot.
    static = {}
    for cam in STATIC_CAMS:
        if cam == front:
            m1, m2, col, dep = map1_a, map2_a, color_a, depth_a
        else:
            m1, m2, _K = prepare_camera(cams[cam], camera_w, camera_h)
            col = decode_video_frames(
                videos / f"observation.images.{cam}.color" / f"episode_{ep_idx:06d}.mp4", T)
            dep = decode_video_frames(
                videos / f"observation.images.{cam}.transformed_depth" / f"episode_{ep_idx:06d}.mkv",
                T, gray16=True)
        # The full-resolution K after undistortion -- the cloud is built at
        # native resolution, NOT the resized (camera_w, camera_h) grid.
        _, _, newK = build_undistort_maps(cams[cam]["K"], cams[cam]["dist"], cams[cam]["resolution"])
        static[cam] = dict(map1=m1, map2=m2, color=col, depth=dep, K=newK,
                           T=cams[cam]["T_color_to_base"])

    # --- cam1 ---
    color_1 = depth_1 = None
    map1_1 = map2_1 = None
    cam1_K = np.zeros((3, 3), dtype=np.float64)
    cam1_static_extrinsics = np.eye(4)
    cam1_T_eef_cam = None

    if cam1_mode == "kinect_left":
        left = "cam_azure_kinect_left"
        map1_1, map2_1, cam1_K = prepare_camera(cams[left], camera_w, camera_h, crop_lr)
        color_1 = decode_video_frames(
            videos / f"observation.images.{left}.color" / f"episode_{ep_idx:06d}.mp4", T)
        depth_1 = decode_video_frames(
            videos / f"observation.images.{left}.transformed_depth" / f"episode_{ep_idx:06d}.mkv",
            T, gray16=True)
        cam1_static_extrinsics = cams[left]["T_color_to_base"]

    elif cam1_mode == "wrist":
        depth_dir = find_wrist_depth_dir(lerobot_dir)
        if depth_dir is None:
            raise FileNotFoundError(
                f"--cam1 wrist but episode {ep_idx} has no wrist depth stream under "
                f"{videos}. That dataset was recorded colour-only "
                "(ZedCameraConfig use_depth=False). Re-record with use_depth=True, "
                "or use --cam1 kinect_left."
            )
        map1_1, map2_1, cam1_K = prepare_camera(wrist_calib, camera_w, camera_h, crop_lr)
        color_1 = decode_video_frames(
            find_wrist_color_dir(lerobot_dir) / f"episode_{ep_idx:06d}.mp4", T)
        depth_1 = decode_video_frames(depth_dir / f"episode_{ep_idx:06d}.mkv", T, gray16=True)
        cam1_T_eef_cam = wrist_calib["T_eef_cam"]

    ep_out_dir = out_dir / f"demo_{ep_idx}"
    ep_out_dir.mkdir(parents=True, exist_ok=True)

    for t in range(T):
        pcs = []
        for cam, st in static.items():
            d_full = cv2.remap(st["depth"][t], st["map1"], st["map2"],
                               cv2.INTER_NEAREST).astype(np.float32) / 1000.0
            pcs.append(depth_to_base_pointcloud(d_full, st["K"], st["T"], stride=pcd_stride))
        point_cloud = fuse_and_subsample(pcs, num_scene_points, rng, bounds=workspace_bounds)

        rgb_a = cv2.remap(color_a[t], map1_a, map2_a, cv2.INTER_LINEAR)
        dep_a = cv2.remap(depth_a[t], map1_a, map2_a, cv2.INTER_NEAREST).astype(np.float32) / 1000.0
        rgb_a, dep_a = crop_lr_pair(rgb_a, dep_a, crop_lr)
        rgb_a, dep_a = resize_pair(rgb_a, dep_a, camera_w, camera_h)

        if cam1_mode == "none":
            rgb_1 = np.zeros((camera_h, camera_w, 3), dtype=np.uint8)
            dep_1 = np.zeros((camera_h, camera_w), dtype=np.float32)
            cam1_extrinsics = np.eye(4)
        else:
            rgb_1 = cv2.remap(color_1[t], map1_1, map2_1, cv2.INTER_LINEAR)
            dep_1 = cv2.remap(depth_1[t], map1_1, map2_1, cv2.INTER_NEAREST).astype(np.float32) / 1000.0
            rgb_1, dep_1 = crop_lr_pair(rgb_1, dep_1, crop_lr)
            rgb_1, dep_1 = resize_pair(rgb_1, dep_1, camera_w, camera_h)
            if cam1_T_eef_cam is None:
                cam1_extrinsics = cam1_static_extrinsics
            else:
                # The wrist camera rides the gripper, so its camera-to-base
                # transform is rebuilt every frame from that frame's EEF pose.
                cam1_extrinsics = T_base_eef_from(eef_quat[t], eef_pos[t]) @ cam1_T_eef_cam

        np.savez_compressed(
            ep_out_dir / f"{t}.npz",
            point_cloud=point_cloud.astype(np.float32)[None, :],
            gripper_pcd=gripper_pcd[t].astype(np.float32)[None, :],
            goal_gripper_pcd=goal_gripper_pcd[t].astype(np.float32)[None, :],
            rgb_agentview=rgb_a[None, :].astype(np.uint8),
            depth_agentview=dep_a[None, :, :, None].astype(np.float32),
            agentview_intrinsics=agentview_K[None, :].astype(np.float32),
            agentview_extrinsics=agentview_extrinsics.astype(np.float32)[None, :],
            rgb_wrist=rgb_1[None, :].astype(np.uint8),
            depth_wrist=dep_1[None, :, :, None].astype(np.float32),
            wrist_intrinsics=cam1_K[None, :].astype(np.float32),
            wrist_extrinsics=cam1_extrinsics.astype(np.float32)[None, :],
            eef_pos=eef_pos[t].astype(np.float32)[None, :],
            eef_quat=eef_quat[t].astype(np.float32)[None, :],
            gripper_qpos=eef_qpos[t][None, :],
            # Hybrid delta: [world dxyz(3), body dR as 6D(6), gripper(1)].
            # generate_non_gmm_goals_for_low_level.py:432 copies this straight
            # into both action/delta and action/hybrid.
            action=action10[t][None, :],
            # The real 10-D EEF pose, [xyz(3), rot6d(6), gripper(1)]. Approach 2
            # declares state as a live obs (shape [10], type low_dim) and
            # LazyArticuBotDataset REQUIRES it when action_mode='absolute', so a
            # zero placeholder would train the policy against a constant.
            state=state10[t][None, :],
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
    parser.add_argument("--crop_lr", type=int, default=0,
                        help="columns trimmed from BOTH the left and right edge of each "
                             "undistorted full-res camera image before it is resized to "
                             "(camera_w, camera_h). The principal point follows the crop. "
                             "Use it to cut the 16:9 source down toward the square output "
                             "aspect: 1280x720 -> --crop_lr 280 is exactly 1:1, less than "
                             "that leaves a residual horizontal squash. Affects the camera "
                             "IMAGES only -- point_cloud is backprojected from the "
                             "uncropped native-resolution depth either way.")
    parser.add_argument("--cam1", choices=["kinect_left", "wrist", "none"],
                        default="kinect_left",
                        help="which physical camera fills the rgb_wrist/depth_wrist/"
                             "wrist_* arrays. kinect_left works on existing data; "
                             "wrist needs a depth-enabled recording plus "
                             "--wrist_calibration (see module docstring).")
    parser.add_argument("--calibration", type=str, default=None,
                        help="camera_extrinsics.json for the Kinects. Defaults to "
                             "<lerobot_dir>/camera_extrinsics.json.")
    parser.add_argument("--wrist_calibration", type=str, default="wrist_calibration.json",
                        help="JSON with the wrist ZED's intrinsics and T_eef_cam. "
                             "Only read when --cam1 wrist.")
    parser.add_argument("--num_scene_points", type=int, default=4500,
                        help="points in the fused scene cloud after subsampling")
    parser.add_argument("--pcd_stride", type=int, default=4,
                        help="pixel stride when backprojecting depth (before subsampling)")
    parser.add_argument("--workspace_bounds", type=float, nargs=6,
                        default=[0.2, 0.78, -0.40, 0.40, -0.03, 0.2],
                        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                        help="robot-base-frame crop applied to the fused cloud before "
                             "subsampling. Matters MORE now that SAM masking is gone: it "
                             "is the only thing bounding the cloud, which otherwise "
                             "includes the arm, the table plane and the background.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--curvature_threshold", type=float, default=0.5)
    parser.add_argument("--min_segment_len", type=int, default=10)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--action_source", choices=["achieved", "fk_commanded"],
                        default="achieved",
                        help="what the hybrid delta targets. 'achieved' (default) uses "
                             "the measured pose at t+1, so state[t] + action[t] == "
                             "state[t+1] exactly. 'fk_commanded' instead reconstructs the "
                             "teleoperator's commanded pose via forward kinematics on the "
                             "8-D GELLO action joints, matching what D1 stores (a "
                             "controller command the arm only partly tracks); its deltas "
                             "are ~5x larger. Use it if you train real and D1 data under "
                             "one normalizer.")
    args = parser.parse_args()

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

    cams = load_calibration(lerobot_dir, args.calibration)
    rng = np.random.default_rng(args.seed)

    wrist_calib = None
    if args.cam1 == "wrist":
        wrist_calib = load_wrist_calibration(Path(args.wrist_calibration))
        print(f"[cam1] wrist ZED, per-frame extrinsics from T_base_eef(t) @ T_eef_cam")
    elif args.cam1 == "kinect_left":
        print("[cam1] cam_azure_kinect_left (static extrinsic)")
    else:
        print("[cam1] disabled -- cam1 arrays are zero-filled")

    print(f"[convert] {len(episode_ids)} episodes -> {output_dir}")
    for i, ep_idx in enumerate(episode_ids):
        T = convert_episode(
            lerobot_dir=lerobot_dir,
            ep_idx=ep_idx,
            out_dir=output_dir,
            cams=cams,
            fps=fps,
            lang_goal=lang_goal,
            camera_h=args.camera_h,
            camera_w=args.camera_w,
            crop_lr=args.crop_lr,
            cam1_mode=args.cam1,
            wrist_calib=wrist_calib,
            num_scene_points=args.num_scene_points,
            pcd_stride=args.pcd_stride,
            workspace_bounds=args.workspace_bounds,
            rng=rng,
            curvature_threshold=args.curvature_threshold,
            min_segment_len=args.min_segment_len,
            warmup_steps=args.warmup_steps,
            action_source=args.action_source,
        )
        print(f"[convert] ({i + 1}/{len(episode_ids)}) demo_{ep_idx}: {T} frames")

    print(f"[convert] done -> {output_dir}")


if __name__ == "__main__":
    main()
