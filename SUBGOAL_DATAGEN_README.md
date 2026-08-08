# Subgoal Generation & Visualization (`MimicGen_DataGen_and_Infer`)

This branch adds tooling to generate proprioceptive, velocity-free **subgoal**
(`goal_gripper_pcd`) keypoint labels for the high-level policy, using several
interchangeable methods, plus tools to inspect them visually before training.

All methods are offline and read-only against the original dataset: they
never modify the source `.npz` files, and every method writes into its own
output tree using the same per-frame convention, so any method's output is a
drop-in replacement for any other's.

## Data layout convention

Original per-frame dataset (produced upstream by `convert_dataset.py` /
MimicGen / RL Bench conversion):

```
<DATA_ROOT>/<TASK>/demo_N/t.npz
    point_cloud       (1, P, 3)   scene point cloud
    gripper_pcd       (1, 4, 3)   gripper keypoints at this frame
    eef_pos           (1, 3)
    eef_quat          (1, 4)      (x, y, z, w)
    gripper_qpos      (1, 2)      finger joint positions
    action            (1, D)      last dim = gripper open/close command
    rgb_agentview / depth_agentview / agentview_intrinsics / agentview_extrinsics
    (or front_rgb / front_depth / front_camera_intrinsics / front_camera_extrinsics for RL Bench)
```

Every subgoal generator writes a **mirror tree** with the same `demo_N/t.npz`
structure, holding only the new `goal_gripper_pcd_<method>` key(s) `(1, 4, 3)`:

```
<DATA_ROOT>/EXTRA_KEYPOINTS/<TASK>/demo_N/t.npz   # RDP-family (see below)
<AWE_OUTPUT_DIR>/demo_N/t.npz                      # AWE (own output dir)
```

`NpyDataset` (`src/lfd3d/datasets/npy/npy_dataset.py`) reads whichever tree
you point it at via `dataset.goal_source` / `dataset.extra_goals_dir` —
see [Training on generated subgoals](#training-on-generated-subgoals).

## Methods

| Method | Library (pure function) | CLI |
|---|---|---|
| `rdp`, `rdp_gripper`, `random`, `fixed_interval` | `third_party/robogen/rdp_subgoal_decomp.py` | `external/mimicgen/mimicgen/scripts/generate_extra_keypoints.py` |
| curvature/jerk | `third_party/robogen/subgoal_decomp.py` | (run directly, see its `__main__`) |
| BOCPD | `third_party/robogen/bayesian_subgoal_decomp.py` | — |
| `awe` (Automatic Waypoint Extraction) | `third_party/robogen/awe_subgoal_decomp.py` | `scripts/generate_awe_subgoals.py` |

Each library module exposes one pure function
(`compute_<method>_subgoal_gripper_pcd(...)`) that takes per-demo arrays and
returns `(expanded_goal_gripper_pcd, switch_indices)` — no file I/O, no CLI.
The CLI scripts are thin wrappers: they walk `demo_*/` directories, call the
library function per demo, and write the mirror tree. This split means the
core algorithm can be unit-tested or called from a notebook without touching
disk.

### RDP family

```bash
pixi run python external/mimicgen/mimicgen/scripts/generate_extra_keypoints.py \
    --data_root /path/to/GROOT_STYLE_DATASET/D2 \
    --task COFFEE_PREPERATION_D1 \
    --methods rdp rdp_gripper \
    --episodes 5   # inspect a few first; drop for the full run
```

Writes to `<data_root>/EXTRA_KEYPOINTS/<task>/demo_N/t.npz`. Incremental and
resumable — rerunning skips demos that already look complete (`--force` to
recompute anyway).

### AWE (Automatic Waypoint Extraction)

```bash
pixi run python scripts/generate_awe_subgoals.py \
    --dataset_dir /path/to/GROOT_STYLE_DATASET/D2/COFFEE_PREPERATION_D1 \
    --output_dir  /path/to/GROOT_STYLE_DATASET/D2/COFFEE_PREPERATION_D1_AWE \
    --err_threshold 0.03 --method greedy --num_workers 8
```

AWE picks the sparsest set of frames ("waypoints") such that linearly
interpolating the end-effector pose between consecutive waypoints
reconstructs the full trajectory within `--err_threshold`. `--method greedy`
(default) is fast and near-optimal; `--method dp` is optimal but `O(T^3)` —
only use it on short demos (roughly ≤ a few hundred frames). Each demo is
processed independently, so `--num_workers` scales close to linearly; a
watcher thread prints periodic per-demo progress, safe for both a live
terminal and a SLURM log file.

**Environment note:** the `waypoint_extraction` PyPI package imports
`robosuite` purely for a math helper, but importing `robosuite` at all
eagerly pulls in its full mujoco/EGL rendering stack. `awe_subgoal_decomp.py`
works around this with a stub-module import so AWE never needs a working
EGL/OSMesa OpenGL backend (see the docstring at the top of that file for
details). `--pos_only` currently hits a known bug inside the
`waypoint_extraction` library itself (documented in the same file) — avoid it
for now.

## Visualization

Two standalone viewers, both driven by the same `--demo_root` (original
per-frame data) / `--goal_root` (a mirror tree, RDP's `EXTRA_KEYPOINTS/...` or
AWE's own `--output_dir` — either works, they use the identical
`goal_gripper_pcd_<type>` key convention) pair, and auto-detect every
`goal_gripper_pcd*` key present so multiple methods can be compared at once
by pointing `--goal_root` at a directory that has more than one:

**Static matplotlib plots** (`scripts/visualize_npz_demo_matplotlib.py`) —
one PNG per demo per keypoint type, plus a combined overlay:

```bash
pixi run python scripts/visualize_npz_demo_matplotlib.py \
    --demo_root /path/to/D2/COFFEE_PREPERATION_D1 \
    --goal_root /path/to/D2/EXTRA_KEYPOINTS/COFFEE_PREPERATION_D1 \
    --out_dir   /path/to/D2/KEYPOINT_PLOTS/COFFEE_PREPERATION_D1 --scene
```

**Interactive viser viewer** (`scripts/visualize_npz_demo_viser.py`) — live,
scrubbable 3D scene in the browser; demo dropdown, timestep slider, and a
scene-point-cloud toggle (re-colorized from the current frame's camera, not
frozen at frame 0):

```bash
pixi run python scripts/visualize_npz_demo_viser.py \
    --demo_root /path/to/D2/COFFEE_PREPERATION_D1 \
    --goal_root /path/to/D2/COFFEE_PREPERATION_D1_AWE --scene
```

Verified compatible end-to-end with both `generate_extra_keypoints.py`
(RDP-family) and `generate_awe_subgoals.py` (AWE) output on a synthetic demo:
both loaders correctly auto-detect the respective `goal_gripper_pcd_rdp*` /
`goal_gripper_pcd_awe` keys, and the viser server starts and serves the scene
without modification.

## Training on generated subgoals

Point a dataset config at the mirror tree you generated:

```yaml
dataset:
  goal_source: awe          # or: rdp, rdp_gripper, random, fixed_interval
  extra_goals_dir: /path/to/D2/COFFEE_PREPERATION_D1_AWE
```

`NpyDataset._load_goal` reads `goal_gripper_pcd_<goal_source>` from
`<extra_goals_dir>/demo_N/t.npz` per frame, in place of the default inline
`goal_gripper_pcd`. See `VALID_GOAL_SOURCES` in
`src/lfd3d/datasets/npy/npy_dataset.py`.

## Environment setup (`pixi.toml`)

This branch's `pixi.lock` needed a few fixes to resolve/install cleanly —
noted here since they're easy to accidentally revert:

- `huggingface_hub` pinned `>=0.34.0,<1.0` — an unconstrained conda-forge pull
  to `1.27.0` was conflicting with `transformers==4.56.0`'s `<1.0` requirement.
- `cmake` added — needed to build `egl-probe`'s C extension during `pixi install`.
- `scipy` pinned `>=1.10,<1.11` — `scipy==1.15.2` is ABI-incompatible with the
  hard-pinned `numpy==1.23.*`, crashing on `import scipy.special`.
- `waypoint-extraction`, `mujoco-py`, `cython`, `robosuite` added — needed by
  the AWE pipeline (see the EGL workaround note above for why `robosuite`
  doesn't need a real rendering backend despite being installed).

If you hit `pixi lock` / `pixi install` failures on this branch, check
whether one of these pins has drifted before troubleshooting further.
