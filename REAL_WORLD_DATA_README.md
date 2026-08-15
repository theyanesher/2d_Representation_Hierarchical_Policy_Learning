# Real-world (LeRobot Franka) → MimicGen HL training pipeline

Status/log of converting the real-world `franka_push_block` LeRobot recording into
the per-frame `.npz` format `NpyDataset` (`src/lfd3d/datasets/npy/npy_dataset.py`)
reads for high-level (HL) policy training — same target format MimicGen sim data
is converted into by `external/mimicgen/mimicgen/scripts/convert_dataset.py`.

Run all commands from the repository root:

```bash
cd /home/theyanesh/2d_Representation_Hierarchical_Policy_Learning
```

## Data

```text
Source (raw LeRobot dataset): /data/theya/data/uncertainity_subgoal/franka_push_block
  - 50 episodes, 15 fps, task: "push the block to the white spot"
  - Cameras: cam_azure_kinect_front (color+depth), cam_azure_kinect_left (color+depth),
    cam_wrist (ZED, color only, UNCALIBRATED)
  - observation.right_eef_pose / action.right_eef_pose: rot6d(6) + trans(3) + gripper_articulation(1)
  - KNOWN BUG: frame 0 of every episode's parquet/video actually captures the PREVIOUS
    episode's last frame, not this episode's real first frame. The converter drops it
    unconditionally (every demo loses exactly 1 frame, e.g. 399 -> 398) before any
    downstream processing, so it never contaminates goal_gripper_pcd's subgoal
    decomposition or gets written to demo_i/0.npz.

Calibration: <source>/camera_extrinsics.json
  - Written by lerobot/scripts/verify_camera_calibration.py
  - Per camera (front/left only — wrist has none): color_intrinsics, depth_intrinsics,
    T_color_to_base (4x4, camera->robot-base), T_depth_to_base
  - Frame convention: robot base frame (deoxys O_T_* frame), metres

Converted output: /data/theya/data/uncertainity_subgoal/franka_push_block_mimicgen_npz
  - <output>/demo_i/t.npz — see convert script docstring for the full key list
```

## 1. Convert LeRobot → per-frame npz

Script: `scripts/convert_lerobot_franka_to_mimicgen.py` (full docstring has the
exact npz schema + every unverified assumption baked into the conversion).

```bash
.pixi/envs/eval/bin/python scripts/convert_lerobot_franka_to_mimicgen.py \
  --lerobot_dir /data/theya/data/uncertainity_subgoal/franka_push_block \
  --output_dir /data/theya/data/uncertainity_subgoal/franka_push_block_mimicgen_npz \
  --camera_h 256 --camera_w 256 --num_scene_points 4500 \
  --workspace_bounds 0.0 0.78 -0.40 0.30 -0.03 0.32 \
  --mask_robot_arm --mask_table --sam_checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --sam_model_type vit_b --sam_device cuda:1
```

Drop `--episodes` to convert all 50 demos; pass e.g. `--episodes 0 1 2` to spot-check
a subset first. `--workspace_bounds` (robot-base frame, metres: `XMIN XMAX YMIN YMAX
ZMIN ZMAX`) crops the fused point cloud to the table/workspace — verify it visually
(§3) before committing to a full run, since there's no default. The command above
saves in the default front-camera frame; add `--robot_base_frame` for robot-base
frame instead (see the "Output frame" note below).

**Known open items / assumptions (see script docstring for full detail):**
1. `gripper_pcd` reuses MimicGen's sim-specific 4-keypoint gripper geometry
   (`mimicgen/utils/articubot_util.py::get_4_points_from_gripper_pos_orient`) —
   assumes the real EEF rotation frame convention matches sim's. The eef
   translation/rotation field itself is now cross-checked (panda_fk_joint_positions's
   joint7 origin + Panda TCP offset lines up with `observation.right_eef_pose` to
   ~1-2cm, see §2), but `get_4_points_from_gripper_pos_orient`'s SIM-captured
   reference orientation constant is still unverified against the real convention.
2. `gripper_articulation` (0=closed..1=open) is scaled by 0.04 m (real Franka Hand
   per-finger travel) to get `cur_joint_angle` — a reasonable guess, not a documented spec.
3. `state` is written as zeros (harmless: `NpyDataset.__getitem__` never reads it for HL).
4. `goal_gripper_pcd` reuses the sim subgoal heuristic
   (`third_party/robogen/subgoal_decomp.py::compute_subgoal_gripper_pcd`) with an
   approximated gripper-action sign convention.
5. `cam_wrist` has no calibration → stored RGB-only, zero/identity intrinsics/extrinsics.

**Output frame (default: front/agentview camera frame; `--robot_base_frame` to switch
to robot-base):** `workspace_bounds`, SAM prompts, and the table-height gate always
run internally in robot-base frame (that's the calibration's native frame, and where
physical measurements like table extent are intuitive) regardless of this flag. By
default, the *saved* `point_cloud`/`gripper_pcd`/`goal_gripper_pcd`/`eef_pos`/
`eef_quat` are then rigid-transformed into the front camera's own frame right before
writing each npz; `--robot_base_frame` skips that final transform and saves
robot-base-frame values instead. `gripper_pcd[3]` equals `eef_pos` exactly in either
mode (that invariant is frame-independent), and `agentview_extrinsics` is identity in
the default camera mode since the points are already expressed in that camera's
frame. Note on why `base` is even an option: checked `external/robomimic/robomimic/
envs/env_robosuite.py` directly — sim's `point_cloud` is computed from raw
camera-to-world extrinsics (`pose = ext_mat`, line ~368) and is **not** premultiplied
by `self.base_world_T_base_robot`, while `gripper_pcd`/`eef_pos`/`agentview_extrinsics`
*are* (lines ~480-509) — i.e. sim's `point_cloud` is technically in MuJoCo world
frame, not robot-base frame, and only matches `gripper_pcd`'s frame if the robot's
root body sits at (or very near) the world origin for these particular MimicGen
environments. This real-data pipeline doesn't rely on that coincidence in
`--robot_base_frame` mode (everything explicitly uses the measured `T_color_to_base`).

## 2. Robot-arm + table exclusion from `point_cloud` (SAM)

Sim's `point_cloud` observable excludes the robot/gripper AND the table plane via
exact geom-id segmentation (`external/robomimic/robomimic/envs/env_robosuite.py::
scene_pcd_ids`, drops any body named `robot*`/`gripper*`, plus `table`/`world`/mount
bodies explicitly). Real depth has no such per-pixel labels, so this is approximated
with SAM (Meta's Segment Anything):

- `--mask_robot_arm`: per frame/camera, prompts SAM with points on the ACTUAL arm
  geometry — forward kinematics (`panda_fk_joint_positions`, standard Franka Panda
  modified-DH table) over `observation.state[:7]` (joint angles, radians) gives each
  joint frame's 3D origin, plus the dataset's exact `eef_pos` for the gripper tip.
  Verified against the data: FK's joint7-frame origin, offset ~0.21m along its local
  z (0.107m flange + ~0.1034m Franka Hand TCP, matching published Panda specs),
  lines up with `observation.right_eef_pose`'s translation to within ~1-2cm across
  sampled frames — confirms both the DH table and the joint-angle convention are
  right. Each joint point is queried as its own single-point SAM prompt and unioned
  (a first version used a straight 3D line from the robot base to eef_pos instead —
  broke badly whenever the elbow was visibly bent, which is most of the time for
  this task: interior "points" on that line land in free space, not on the arm
  surface, so SAM either grossly under-segments or over-merges into the
  background/table depending on where the bad click lands). Any single joint's best
  mask exceeding `max_area_frac` (15% of frame, hardcoded) is dropped from the union
  instead of included — an occluded joint's projected pixel can show background/table
  instead of arm, and a single click on a large flat surface returns a huge ambiguous
  SAM mask; the real arm is never that big, so oversized single-point masks are
  always spurious.
- `--mask_table` (requires `--mask_robot_arm`, shares the predictor): computed ONCE
  per camera per episode (table + camera are static within an episode) from a
  reference frame. Prompts SAM with a 7x7 grid of known-flat 3D points spanning
  `--workspace_bounds` at `--table_height` (each queried as its own single-point
  prompt and unioned — a single merged multi-point prompt under-segments a large
  flat surface when shadows/scratches split it into several SAM regions). The white
  target patch is auto-detected by colour (bright, low-saturation, square-ish,
  doesn't touch the image's top edge — that last check is what stops the detector
  from grabbing the Franka Hand's white plastic housing instead) and carved out of
  the table mask, so it's preserved. A pushed object sliding into a pixel region the
  static mask calls "table" is protected by `--table_tol`: the table mask only
  actually excludes a pixel if its *backprojected* z also lands within
  `--table_height +/- --table_tol` — an object standing above the table plane fails
  that check regardless of its 2D mask membership.

`depth_agentview` (raw per-camera depth) is left untouched by both flags, matching
sim, where only the fused `point_cloud` gets these exclusions.

Setup (one-time):
```bash
.pixi/envs/eval/bin/python -m pip install segment-anything
mkdir -p checkpoints
curl -sSL -o checkpoints/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```
Not yet pinned in `pixi.toml`/`pixi.lock` — currently a plain `pip install` into the
`eval` env. Add properly if this becomes a permanent dependency.

Measured cost: ~77 ms/frame warm on GPU (ViT-B) for the per-frame arm mask. Full
dataset (50 episodes x 399 frames x 2 cams ≈ 40k SAM calls) ≈ 50-60 min on one GPU.
The table mask adds ~49 SAM calls x 2 cams **once per episode** (reference frame
only) — a few extra seconds/episode, negligible against the arm-mask cost.

**`--table_tol` needed tuning empirically.** `camera_extrinsics.json`'s own
`accuracy` block states ~2-3cm typical backprojection error, but debugging left-over
unmasked table near the workspace edges showed real error up to ~4.5cm there (error
grows away from the calibration board's center). Default is `0.06` (6cm) to cover
that with margin — a tighter value leaves slivers of real table unexcluded near
`--workspace_bounds` edges. Verified on episode 0 (`--mask_robot_arm --mask_table`):
diffuse table bleed and a corner/edge gap present at the original `0.015`/4x4-grid
settings are gone; the only remaining unmasked-by-either-mask cluster is the
blue tape border around the white patch (expected/harmless — same height as table
but neither "bare table" colour nor "white patch" colour, so it's simply not
claimed by either mask) and a handful of near-black points at the table's far back
edge (curtain/mount-rail shadow, low point count).

Caveat: this is SAM's best guess per frame/reference-frame, not sim's exact geom
segmentation — can occasionally miss/over-include pixels (motion blur, silhouettes
merging with similarly-colored surfaces). Spot-check with the viser viewer (§3), or
a quick top-down matplotlib scatter of `point_cloud` colored by z, before trusting a
full run.

**`--debug_mask_dir <dir>`**: save the actual SAM masks as overlay PNGs (red =
excluded, green = protected white patch) instead of/alongside inspecting the final
point cloud — useful for telling apart "SAM segmented wrong" from "the mask was fine
but something downstream (crop, subsample, height gate) dropped the points anyway".
Writes, per episode, into `<dir>/demo_i/`:
- `arm_<cam>_<t>.png` — one per frame per camera, only with `--mask_robot_arm`
- `table_<cam>.png` / `white_patch_<cam>.png` — once per episode per camera (the
  reference-frame masks table exclusion is actually computed from), only with
  `--mask_table`

No files are written if the flag is omitted. Costs disk + a bit of I/O time (a full
398-frame episode with both flags on writes exactly `398*2 (arm) + 2 (table) + 2
(white patch) = 800` PNGs) — fine for spot-checking a couple of episodes, probably
skip it for the full 50-episode production run.

## 3. Visual sanity check (viser)

Uses the same viewer that reads AWE/RDP/etc. subgoal npz trees — needs the
`default` pixi env (has `viser`; `eval` does not).

```bash
.pixi/envs/default/bin/python scripts/visualize_npz_demo_viser.py \
  --demo_root /data/theya/data/uncertainity_subgoal/franka_push_block_mimicgen_npz \
  --goal_root /data/theya/data/uncertainity_subgoal/franka_push_block_mimicgen_npz \
  --scene --port 8080
```

Open `http://localhost:8080` (forward the port first if connecting remotely, e.g.
`ssh -L 8080:localhost:8080 <host>`). Enable the "scene" checkbox, scrub the
timestep slider, switch the demo dropdown. Recommended order: (1) verify
`--workspace_bounds` on a no-SAM conversion of a few episodes first — the arm/gripper
will still be visible in the scene cloud, that's expected — then (2) re-check with
`--mask_robot_arm` on to confirm the arm is actually excluded. Kill the server with
`pkill -f visualize_npz_demo_viser.py` when done (it's a long-running foreground
process, not meant to be left running unattended).

## 4. Extra keypoints / subgoal methods (e.g. UVD)

```bash
.pixi/envs/eval/bin/python external/mimicgen/mimicgen/scripts/generate_extra_keypoints.py \
  --data_root /data/theya/data/uncertainity_subgoal \
  --task franka_push_block_mimicgen_npz \
  --methods uvd \
  --uvd_camera agentview \
  --uvd_preprocessor vip
```

`--uvd_camera agentview` reads the `rgb_agentview` key our converter writes — matches
directly, no extra plumbing needed. UVD itself runs in the separate `uvd` pixi env
(Python 3.9 stack); the generator shells out to it automatically (`--uvd_pixi_env
uvd`, default), so launch the outer command from `eval` as shown. `vip`/`r3m`/`liv`/
`vc1` preprocessors need a manual install first, see `external/UVD/README.md`.
Output: `/data/theya/data/uncertainity_subgoal/EXTRA_KEYPOINTS_uvd/franka_push_block_mimicgen_npz/`
(`goal_gripper_pcd_uvd` key, original npz untouched). See `GENERATE_KEYPOINTS.md`
for the full flag reference (other methods, `--dump_indices`, `--mix_methods`, etc).

## 5. HL training

Not yet run / command not finalized for the real-world data — pick up from
`shell_scripts/train_all_high_level.sh` and the existing sim HL runs under
`/data/theya/logs/uncertain_subgoals/train_HAMMER_CLEANUP_D1_*` as reference once
ready.
