#!/usr/bin/env bash
# Convert the `pushblock` LeRobot dataset -> per-timestep MimicGen .npz.
#
# Point clouds: KEPT (fused front+left Kinect, base frame, cropped to
# WORKSPACE_BOUNDS, subsampled to NUM_SCENE_POINTS).
# SAM 2 masking: NOT RUN -- convert_lerobot_franka_to_mimicgen.py has no SAM
# path at all, so no GPU / checkpoint / scene-prompt picking is involved. The
# consequence is that the cloud contains the robot arm and the table plane;
# WORKSPACE_BOUNDS is the only thing bounding it.
#
# WORKSPACE_BOUNDS below are the same numbers used for the earlier
# lerobot/data/pushblock_test_npz run (they are also the script's own
# defaults); they are passed explicitly so a later default change cannot
# silently move them.
#
# Usage:  bash scripts/convert_pushblock.sh [extra args passed through]
#         OUTPUT_DIR=/somewhere/else bash scripts/convert_pushblock.sh
#         bash scripts/convert_pushblock.sh --episodes 0        # single demo
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

LEROBOT_DIR="${LEROBOT_DIR:-/home/madhavan/uncertain_subgoal/lerobot/data/pushblock}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/madhavan/uncertain_subgoal/lerobot/data/pushblock_npz}"

# Kinect intrinsics + T_color_to_base. The pushblock recording carries no local
# copy, so point at the rig snapshot it was recorded on (2026-08-21). This is
# the calibration the earlier pushblock_test_npz run used -- confirmed by its
# agentview_extrinsics matching cam_azure_kinect_front in that file exactly.
CALIBRATION="${CALIBRATION:-/home/madhavan/uncertain_subgoal/calibration_2026-08-21/camera_extrinsics.json}"

# cam1 = wrist ZED (per-frame extrinsic T_base_eef(t) @ T_eef_cam).
# pushblock records observation.images.cam_wrist.depth, so this works.
# Set CAM1=kinect_left to fill cam1 from the static left Kinect instead.
CAM1="${CAM1:-wrist}"
WRIST_CALIBRATION="${WRIST_CALIBRATION:-${REPO_ROOT}/wrist_calibration.json}"

# 448x252 is exactly 16:9, matching the 1280x720 source, so the emitted
# intrinsics come out isotropic (fx/fy = 1.000 instead of the 0.563 a square
# 256x256 target produces). Both dims are divisible by 14, so a DINOv2 patch-14
# backbone tiles the frame as 32x18 patches and a 224x224 crop as 16x16.
CAMERA_H="${CAMERA_H:-252}"
CAMERA_W="${CAMERA_W:-448}"

# Symmetric width crop applied before the resize (principal point follows it).
# Left at 0: the 16:9 output above already fixes the aspect, so nothing needs
# to be thrown away. Only useful when forcing a square output -- see --crop_lr
# in the converter. Images only; point_cloud ignores it either way.
CROP_LR="${CROP_LR:-0}"
NUM_SCENE_POINTS="${NUM_SCENE_POINTS:-4500}"
PCD_STRIDE="${PCD_STRIDE:-4}"

# xmin xmax ymin ymax zmin zmax, robot-base frame, metres -- bounds from earlier.
WORKSPACE_BOUNDS="${WORKSPACE_BOUNDS:-0.2 0.78 -0.40 0.40 -0.03 0.2}"

python "${REPO_ROOT}/scripts/convert_lerobot_franka_to_mimicgen.py" \
    --lerobot_dir "${LEROBOT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --calibration "${CALIBRATION}" \
    --cam1 "${CAM1}" \
    --wrist_calibration "${WRIST_CALIBRATION}" \
    --camera_h "${CAMERA_H}" \
    --camera_w "${CAMERA_W}" \
    --crop_lr "${CROP_LR}" \
    --num_scene_points "${NUM_SCENE_POINTS}" \
    --pcd_stride "${PCD_STRIDE}" \
    --workspace_bounds ${WORKSPACE_BOUNDS} \
    "$@"
