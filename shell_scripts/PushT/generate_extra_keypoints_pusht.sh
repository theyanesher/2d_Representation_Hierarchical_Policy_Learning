#!/usr/bin/env bash
# Non-VLM keypoints / subgoals for PushT, mirroring the method set used for D1's
#   EXTRA_KEYPOINTS_bspline_bspline_greville_awe_gripper_heuristic_orientation_heuristic_uvd_mix_bspline_bspline_greville_mix_gripper_heuristic_orientation_heuristic
# tree, minus the methods PushT's embodiment cannot support:
#
#   bspline                       -- every interior knot of the sparsest spline fit
#   bspline_greville              -- only the high-influence control points
#   awe                           -- Automatic Waypoint Extraction
#   uvd                           -- Universal Visual Decomposer (rgb_agentview)
#   mix_bspline_bspline_greville  -- the two spline methods mingled (bspline wins ties)
#
# DROPPED vs. the D1 tree, and why -- PushT is a PLANAR PUSHER, not a gripper arm:
#   gripper_heuristic                       -- needs gripper open/close transitions.
#       PushT's .npz has no `gripper_qpos` key at all and its `action` is a bare
#       (x, y) target with no gripper channel; there is no signal to detect.
#       --gripperless rejects this method outright rather than emit empty results.
#   orientation_heuristic                   -- needs EEF orientation drift. PushT's
#       `eef_quat` is the constant identity [0, 0, 0, 1] for every frame of every
#       demo, so the geodesic angle is exactly 0 everywhere and this method can
#       only ever return zero boundaries.
#   mix_gripper_heuristic_orientation_heuristic -- a mix of the two above, so it
#       degenerates along with them.
#
# --gripperless also matters for AWE specifically: AWE reads `actions[:, -1]` as
# the gripper open/close command and forces a waypoint wherever it changes.
# Without the flag, PushT's y-target would be fed in as that "gripper command"
# and flip nearly every frame, forcing a waypoint at almost every timestep.
# The flag synthesizes a constant-zero gripper channel, making that term inert.
#
# Output (original dataset stays READ-ONLY, new keys land in a mirror tree):
#   ${DATA_ROOT}/EXTRA_KEYPOINTS_bspline_bspline_greville_awe_uvd_mix_bspline_bspline_greville/${TASK}/demo_*/<t>.npz
# holding goal_gripper_pcd_bspline, _bspline_greville, _awe, _uvd, and
# _mix_bspline_bspline_greville -- each (1, 4, 3) float32, drop-in
# interchangeable with the original goal_gripper_pcd.
#
# Resumable: already-complete demos are skipped, so re-running after an
# interruption picks up where it left off. Pass --force to recompute.
#
# Run from the repo root. UVD shells out to the `uvd` pixi environment
# (Python 3.9) once per demo, so it needs pixi.toml's [feature.uvd.*] env built.
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/theya/data/uncertainity_subgoal/PushT}"
TASK="${TASK:-PushT_npz}"
EPISODES="${EPISODES:-}"          # e.g. EPISODES=5 to inspect a few first
NUM_WORKERS="${NUM_WORKERS:-4}"   # parallelism is ACROSS demos only; each worker
                                  # running uvd holds its own frozen encoder on
                                  # the GPU, so raise this only if VRAM allows

# PushT's world units are the same order of magnitude as MimicGen's metres
# (EEF spans roughly 0.18..0.73 on both axes, z pinned to 0), so the D1
# thresholds carry over directly. Retune here if the keypoint counts come out
# too dense or too sparse for the planar task.
MAX_ERROR="${MAX_ERROR:-0.08}"            # bspline reconstruction budget
AWE_SOLVER="${AWE_SOLVER:-greedy}"        # greedy (fast, near-optimal); dp is the
                                          # optimal O(T^3) alternative, affordable at
                                          # PushT's T ~= 100-200 if you want it
AWE_ERR_THRESHOLD="${AWE_ERR_THRESHOLD:-0.1}"
MIX_WINDOW="${MIX_WINDOW:-5}"

EXTRA_ARGS=()
if [[ -n "${EPISODES}" ]]; then
  EXTRA_ARGS+=(--episodes "${EPISODES}")
fi
# UVD locates its Python 3.9 env from the pixi.toml next to the generator
# script, which is right when running from a normal checkout. Point this at the
# main checkout's pixi.toml when running from a git worktree or a second clone,
# where no .pixi environments have been built.
UVD_PIXI_MANIFEST="${UVD_PIXI_MANIFEST:-}"
if [[ -n "${UVD_PIXI_MANIFEST}" ]]; then
  EXTRA_ARGS+=(--uvd_pixi_manifest "${UVD_PIXI_MANIFEST}")
fi

python external/mimicgen/mimicgen/scripts/generate_extra_keypoints.py \
  --data_root "${DATA_ROOT}" \
  --task "${TASK}" \
  --gripperless \
  --methods bspline bspline_greville awe uvd \
  --mix_methods bspline bspline_greville \
  --mix_window "${MIX_WINDOW}" \
  --max_error "${MAX_ERROR}" \
  --awe_solver "${AWE_SOLVER}" \
  --awe_err_threshold "${AWE_ERR_THRESHOLD}" \
  --uvd_camera agentview \
  --num_workers "${NUM_WORKERS}" \
  --dump_indices \
  "${EXTRA_ARGS[@]}"
