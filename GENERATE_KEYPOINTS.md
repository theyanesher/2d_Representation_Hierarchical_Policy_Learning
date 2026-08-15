# Master guide: generating extra keypoints and VLM subgoals

This is the authoritative command and flag reference for:

```text
external/mimicgen/mimicgen/scripts/generate_extra_keypoints.py
```

The generator reads the existing per-frame NPZ demonstrations without modifying
them. It writes method-specific goal point clouds into a separate mirror tree so
they can be consumed like the original `goal_gripper_pcd` key.

The examples below use:

```text
DATA_ROOT=/data/theya/data/uncertainity_subgoal/D1
TASK=COFFEE_PREPERATION_D1
```

Run all commands from the repository root:

```bash
cd /home/theyanesh/2d_Representation_Hierarchical_Policy_Learning
```

## Quick start

Install or update the repository environment:

```bash
pixi install
```

Generate one method for one demo and write human-readable boundary indices:

```bash
pixi run python \
  external/mimicgen/mimicgen/scripts/generate_extra_keypoints.py \
  --data_root /data/theya/data/uncertainity_subgoal/D1 \
  --task COFFEE_PREPERATION_D1 \
  --methods rdp \
  --episodes 1 \
  --dump_indices
```

Inspect the result, then remove `--episodes 1` to process the complete task:

```bash
pixi run python \
  external/mimicgen/mimicgen/scripts/generate_extra_keypoints.py \
  --data_root /data/theya/data/uncertainity_subgoal/D1 \
  --task COFFEE_PREPERATION_D1 \
  --methods rdp \
  --dump_indices
```

Print the parser's current built-in help:

```bash
pixi run python \
  external/mimicgen/mimicgen/scripts/generate_extra_keypoints.py --help
```

## Input and output layout

The source tree must contain one directory per demonstration and one NPZ per
original trajectory frame:

```text
<DATA_ROOT>/<TASK>/
  demo_0/
    0.npz
    1.npz
    ...
  demo_1/
    0.npz
    ...
```

The numeric filename is the original trajectory index. Every method loads these
base keys:

```text
eef_pos
eef_quat
gripper_qpos
action
gripper_pcd
```

`vlm` additionally loads `rgb_<--vlm_camera>`, and `uvd` loads
`rgb_<--uvd_camera>`. The Coffee dataset provides `rgb_agentview` and
`rgb_wrist`.

Output is written under a method-set-specific mirror directory:

```text
<DATA_ROOT>/EXTRA_KEYPOINTS_<methods>/<TASK>/<demo>/<frame>.npz
```

For example:

```text
/data/theya/data/uncertainity_subgoal/D1/
  EXTRA_KEYPOINTS_vlm/
    COFFEE_PREPERATION_D1/
      demo_0/
        0.npz
        ...
        704.npz
        _keypoints.json
```

Each output frame contains one drop-in key per selected method:

```text
goal_gripper_pcd_<method>    shape=(1, 4, 3), dtype=float32
```

At each trajectory step, this array holds the gripper point cloud at the next
selected subgoal. All schedules include `T - 1` as their terminal target. With
`--dump_indices`, each demo also gets:

```json
{
  "T": 705,
  "keypoints": {
    "vlm": [105, 200, 420, 515, 704]
  }
}
```

The source dataset is never modified. Existing keys in a mirror NPZ are
preserved when another method is added to that same mirror tree.

## Supported methods

| Method | What selects a keypoint | Main flags |
|---|---|---|
| `rdp` | Ramer-Douglas-Peucker corners in end-effector position | `--epsilon` |
| `rdp_gripper` | RDP corners snapped to nearby gripper transitions | `--epsilon`, `--snap_window` |
| `random` | Random trajectory indices | `--n_random`, `--seed` |
| `fixed_interval` | Even cadence derived from the episode unless overridden | `--interval` |
| `bspline` | Interior knots of the sparsest acceptable B-spline | `--max_error`, `--degree` |
| `bspline_greville` | High-influence B-spline control points mapped through Greville abscissae | `--max_error`, `--degree`, `--influence_threshold` |
| `awe` | Automatic Waypoint Extraction reconstruction error | `--awe_err_threshold`, `--awe_solver` |
| `vlm` | Meaningful visual manipulation/world-state transitions | all `--vlm_*` flags |
| `gripper_heuristic` | Every gripper open/close transition | no method-specific flag |
| `fixed_interval_const` | Constant cadence shared across all demos | `--const_interval` |
| `orientation_heuristic` | Accumulated orientation drift since the previous subgoal | `--orientation_threshold` |
| `uvd` | Universal Visual Decomposer visual representation changes | all `--uvd_*` flags |

Generate every supported base method with:

```bash
pixi run python \
  external/mimicgen/mimicgen/scripts/generate_extra_keypoints.py \
  --data_root /data/theya/data/uncertainity_subgoal/D1 \
  --task COFFEE_PREPERATION_D1 \
  --methods all \
  --episodes 1 \
  --dump_indices
```

`all` includes VLM and UVD, so their separate runtimes must also be ready.

## Common method recipes

RDP and gripper-aware RDP:

```bash
pixi run python \
  external/mimicgen/mimicgen/scripts/generate_extra_keypoints.py \
  --task COFFEE_PREPERATION_D1 \
  --methods rdp rdp_gripper \
  --epsilon 0.02 \
  --snap_window 5 \
  --dump_indices
```

B-spline and AWE:

```bash
pixi run python \
  external/mimicgen/mimicgen/scripts/generate_extra_keypoints.py \
  --task COFFEE_PREPERATION_D1 \
  --methods bspline bspline_greville awe \
  --max_error 0.08 \
  --degree 3 \
  --influence_threshold 0.08 \
  --awe_err_threshold 0.2 \
  --awe_solver dp \
  --dump_indices
```

Local heuristics:

```bash
pixi run python \
  external/mimicgen/mimicgen/scripts/generate_extra_keypoints.py \
  --task COFFEE_PREPERATION_D1 \
  --methods gripper_heuristic fixed_interval_const orientation_heuristic \
  --const_interval 50 \
  --orientation_threshold 0.5235987756 \
  --dump_indices
```

Random and interval baselines:

```bash
pixi run python \
  external/mimicgen/mimicgen/scripts/generate_extra_keypoints.py \
  --task COFFEE_PREPERATION_D1 \
  --methods random fixed_interval \
  --n_random 20 \
  --seed 0 \
  --interval 35 \
  --dump_indices
```

## Local Qwen VLM setup

The local default is the multimodal `Qwen/Qwen3.6-35B-A3B-FP8` checkpoint,
served as `qwen3.6-local` through a local OpenAI-compatible vLLM endpoint. It
uses both RTX 4090 GPUs and does not require an API key.

The one-time setup downloads approximately 37.5 GB to
`/data/theya/models/Qwen3.6-35B-A3B-FP8` and installs an isolated runtime under
`/data/theya/qwen_local`:

```bash
scripts/setup_qwen_local.sh
```

Start the server in its own terminal:

```bash
scripts/serve_qwen_local.sh
```

The server stays in the foreground. Stop it with Ctrl-C. Check readiness from
another terminal:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

Optional server environment overrides:

```bash
QWEN_RUNTIME_DIR=/data/theya/qwen_local \
QWEN_MODEL_DIR=/data/theya/models/Qwen3.6-35B-A3B-FP8 \
QWEN_SERVED_MODEL=qwen3.6-local \
QWEN_HOST=127.0.0.1 \
QWEN_PORT=8001 \
QWEN_MAX_MODEL_LEN=16384 \
QWEN_GPU_MEMORY_UTILIZATION=0.86 \
CUDA_VISIBLE_DEVICES=0,1 \
scripts/serve_qwen_local.sh
```

The setup script accepts `QWEN_RUNTIME_DIR`, `QWEN_MODEL_DIR`, and
`QWEN_MODEL_REPO` as environment overrides. If `QWEN_SERVED_MODEL` changes,
pass the same name through `--vlm_model`.

If the port changes, point the generator at it:

```bash
--vlm_qwen_base_url http://127.0.0.1:8001/v1
```

### Inspect standalone VLM boundaries first

Sparse-only inspection performs one VLM request and returns only sampled
original indices:

```bash
pixi run python examples/detect_subtask_boundaries.py \
  /data/theya/data/uncertainity_subgoal/D1/COFFEE_PREPERATION_D1/demo_0 \
  --rgb-key rgb_agentview \
  --provider qwen \
  --instruction "prepare coffee" \
  --sample-every-n-frames 15 \
  --sheet-overlap-frames 2 \
  --stop-after-sparse-annotation
```

Dense refinement is enabled by default. Omit the sparse-stop flag:

```bash
pixi run python examples/detect_subtask_boundaries.py \
  /data/theya/data/uncertainity_subgoal/D1/COFFEE_PREPERATION_D1/demo_0 \
  --rgb-key rgb_agentview \
  --provider qwen \
  --instruction "prepare coffee" \
  --sample-every-n-frames 15 \
  --refinement-radius 15 \
  --refinement-stride 1
```

The standalone detector returns transition-only indices. It does not append the
terminal target; the keypoint generator appends `T - 1` when constructing the
goal schedule.

### Generate VLM keypoints

Recommended one-demo dense run:

```bash
pixi run python \
  external/mimicgen/mimicgen/scripts/generate_extra_keypoints.py \
  --data_root /data/theya/data/uncertainity_subgoal/D1 \
  --task COFFEE_PREPERATION_D1 \
  --methods vlm \
  --episodes 1 \
  --dump_indices \
  --vlm_provider qwen \
  --vlm_camera agentview \
  --vlm_instruction "prepare coffee" \
  --vlm_sample_every_n_frames 15 \
  --vlm_refinement_radius 15 \
  --vlm_refinement_stride 1 \
  --vlm_sheet_overlap_frames 2
```

Full task after inspecting `demo_0`:

```bash
pixi run python \
  external/mimicgen/mimicgen/scripts/generate_extra_keypoints.py \
  --data_root /data/theya/data/uncertainity_subgoal/D1 \
  --task COFFEE_PREPERATION_D1 \
  --methods vlm \
  --dump_indices \
  --vlm_provider qwen \
  --vlm_camera agentview \
  --vlm_instruction "prepare coffee"
```

Sparse-only generation, which skips all local refinement calls:

```bash
pixi run python \
  external/mimicgen/mimicgen/scripts/generate_extra_keypoints.py \
  --task COFFEE_PREPERATION_D1 \
  --methods vlm \
  --dump_indices \
  --vlm_stop_after_sparse_annotation
```

`--no_vlm_refine` has the same practical effect on the returned boundaries,
but `--vlm_stop_after_sparse_annotation` states the intended stopping behavior
explicitly.

### VLM contact sheets and indexing

With the defaults:

- frames are sampled every 15 original trajectory indices;
- `T - 1` is always appended if the stride does not land on it;
- every tile is 224 pixels wide and preserves its input aspect ratio;
- at most 20 frames are placed in five columns;
- adjacent sheets repeat two sampled frames;
- the final sheet contains only the rows required by its frames;
- every tile visibly says `FRAME <original index>`;
- Qwen must return only indices shown on the sheets;
- each coarse boundary is independently refined in a dense local window.

For a 705-frame episode the sampled indices end in `690, 704`. The sheets are:

```text
sheet 1: 0 ... 270, 285
sheet 2: 270, 285, 300 ... 540, 555
sheet 3: 540, 555, 570 ... 690, 704
```

For square Coffee frames, each tile is 224x224, a full sheet is 1120x896, and
the 12-tile final sheet is 1120x672.

### VLM logs

Every coarse detector call creates a unique run directory. Generator logs are
nested by demo:

```text
logs/subtask_boundaries/<demo>/run_<UTC-time>_<pid>_<id>/
  prompt.txt
  input.json
  contact_sheet_001.jpg
  contact_sheet_002.jpg
  ...
  output.json
```

`input.json` records the sampled original indices, overlap, and per-sheet
indices. `output.json` contains the parsed coarse structured result. A failed
request leaves the inputs in place and adds `error.txt`.

## Hosted VLM providers

Local Qwen is recommended, but the generator also supports hosted providers.

QwenCloud:

```bash
export DASHSCOPE_API_KEY="..."

pixi run python \
  external/mimicgen/mimicgen/scripts/generate_extra_keypoints.py \
  --task COFFEE_PREPERATION_D1 \
  --methods vlm \
  --episodes 1 \
  --vlm_provider qwen_cloud \
  --vlm_model qwen3.6-flash \
  --dump_indices
```

`QWEN_API_KEY` is accepted as an alias for `DASHSCOPE_API_KEY`.

Gemini:

```bash
export GEMINI_API_KEY="..."

pixi run python \
  external/mimicgen/mimicgen/scripts/generate_extra_keypoints.py \
  --task COFFEE_PREPERATION_D1 \
  --methods vlm \
  --episodes 1 \
  --vlm_provider gemini \
  --vlm_model gemini-3.5-flash \
  --dump_indices
```

OpenAI:

```bash
export OPENAI_API_KEY="..."

pixi run python \
  external/mimicgen/mimicgen/scripts/generate_extra_keypoints.py \
  --task COFFEE_PREPERATION_D1 \
  --methods vlm \
  --episodes 1 \
  --vlm_provider openai \
  --vlm_model gpt-5.4 \
  --dump_indices
```

## Mixing methods

`--mix_methods` produces another key containing the union of two or more
methods. Listed order is priority order. Every boundary from the first method
is kept; a lower-priority boundary is discarded when it is within
`--mix_window` frames of a boundary already retained.

Every method in a mix group must also appear in `--methods`:

```bash
pixi run python \
  external/mimicgen/mimicgen/scripts/generate_extra_keypoints.py \
  --task COFFEE_PREPERATION_D1 \
  --methods gripper_heuristic orientation_heuristic \
  --mix_methods gripper_heuristic orientation_heuristic \
  --mix_window 5 \
  --dump_indices
```

This adds:

```text
goal_gripper_pcd_mix_gripper_heuristic_orientation_heuristic
```

Repeat `--mix_methods` for independent groups:

```bash
pixi run python \
  external/mimicgen/mimicgen/scripts/generate_extra_keypoints.py \
  --task COFFEE_PREPERATION_D1 \
  --methods gripper_heuristic orientation_heuristic rdp awe \
  --mix_methods gripper_heuristic orientation_heuristic \
  --mix_methods rdp awe \
  --mix_window 5 \
  --dump_indices
```

## UVD

UVD runs in the isolated `uvd` pixi environment declared by this repository.
The generator starts one UVD subprocess per demo because UVD's dependency stack
is incompatible with the main environment.

```bash
pixi install -e uvd

pixi run python \
  external/mimicgen/mimicgen/scripts/generate_extra_keypoints.py \
  --task COFFEE_PREPERATION_D1 \
  --methods uvd \
  --episodes 1 \
  --uvd_camera agentview \
  --uvd_preprocessor dinov2 \
  --uvd_device cuda:0 \
  --dump_indices
```

Available preprocessors are `vip`, `r3m`, `liv`, `clip`, `vc1`, `dinov2`, and
`resnet`. Some require manual installation; consult `external/UVD/README.md`.

## Resume, recompute, and parallelism

The generator resumes by default. A demo is skipped only when:

1. every output frame from `0` through `T - 1` exists;
2. the last output NPZ contains every requested method key; and
3. when `--dump_indices` is enabled, `_keypoints.json` exists with matching `T`.

Recompute selected demos even when complete:

```bash
pixi run python \
  external/mimicgen/mimicgen/scripts/generate_extra_keypoints.py \
  --task COFFEE_PREPERATION_D1 \
  --methods vlm \
  --episodes 1 \
  --force \
  --dump_indices
```

Parallelize across demos:

```bash
pixi run python \
  external/mimicgen/mimicgen/scripts/generate_extra_keypoints.py \
  --task COFFEE_PREPERATION_D1 \
  --methods rdp \
  --num_workers 8 \
  --dump_indices
```

`--num_workers` does not parallelize one demo. Keep it at `1` for local VLM or
GPU-bound UVD unless the serving/runtime capacity has been intentionally tuned
for concurrent requests.

## Complete generator flag reference

### Dataset and execution

| Flag | Default | Meaning |
|---|---:|---|
| `--data_root PATH` | `/data/theya/data/uncertainity_subgoal/D1` | Root containing task folders. |
| `--task NAME` | required | Source task directory name. |
| `--methods METHOD [METHOD ...]` | `rdp` | One or more supported methods, or exactly `all`. |
| `--episodes N`, `-n N` | all | Process only the first `N` demos. |
| `--dump_indices` | off | Write `_keypoints.json` inside each output demo. |
| `--force` | off | Disable resume skipping and recompute selected demos. |
| `--num_workers N` | `1` | Number of demos processed concurrently. |

### RDP, random, and interval methods

| Flag | Default | Meaning |
|---|---:|---|
| `--epsilon FLOAT` | `0.02` | RDP position tolerance in metres. |
| `--snap_window N` | `5` | Maximum frame distance for `rdp_gripper` snapping. |
| `--n_random N` | `20` | Number of random keypoints requested. |
| `--seed N` | `0` | Random method seed. |
| `--interval N` | `T // 20` | Explicit `fixed_interval` cadence. |
| `--const_interval N` | `50` | Cross-demo cadence for `fixed_interval_const`. |

### B-spline and AWE

| Flag | Default | Meaning |
|---|---:|---|
| `--max_error FLOAT` | `0.08` | B-spline maximum Chebyshev EEF reconstruction error in metres. |
| `--degree N` | `3` | B-spline degree. |
| `--influence_threshold FLOAT` | `--max_error` | Minimum control-polygon deviation used by `bspline_greville`. |
| `--awe_err_threshold FLOAT` | `0.2` | AWE reconstruction error threshold combining position and rotation. |
| `--awe_solver {greedy,dp}` | `dp` | Greedy is faster; DP is optimal but approximately `O(T^3)`. |

### Local heuristic and mixing flags

| Flag | Default | Meaning |
|---|---:|---|
| `--orientation_threshold FLOAT` | `pi / 6` (`0.5236`) | Geodesic orientation drift since the previous subgoal. |
| `--mix_methods METHOD [METHOD ...]` | none | Create one priority-ordered mixed method; repeat for multiple groups. |
| `--mix_window N` | `5` | Drop lower-priority predictions within `N` frames of a retained prediction. |

### VLM flags

| Flag | Default | Meaning |
|---|---:|---|
| `--vlm_provider {qwen,qwen_cloud,openai,gemini}` | `qwen` | Boundary VLM backend; `qwen` is local. |
| `--vlm_model MODEL` | provider default | `qwen3.6-local`, `qwen3.6-flash`, `gpt-5.4`, or `gemini-3.5-flash` according to provider. |
| `--vlm_qwen_base_url URL` | `http://127.0.0.1:8000/v1` | Local Qwen OpenAI-compatible endpoint. |
| `--vlm_camera NAME` | `agentview` | Loads `rgb_<NAME>`. |
| `--vlm_instruction TEXT` | none | Optional episode task context; output remains indices only. |
| `--vlm_sample_every_n_frames N` | `15` | Coarse sampling stride in original frame indices. |
| `--vlm_refine` | on | Enable per-boundary dense local refinement. |
| `--no_vlm_refine` | off | Disable dense refinement. |
| `--vlm_stop_after_sparse_annotation`, `--vlm-stop-after-sparse-annotation` | off | Explicitly stop after the coarse request. |
| `--vlm_refinement_radius N` | `15` | Frames on each side of a coarse boundary. |
| `--vlm_refinement_stride N` | `1` | Original-frame stride within refinement windows. |
| `--vlm_min_boundary_distance_frames N` | `0` | Merge closer predictions by keeping the earlier one; zero disables merging. |
| `--vlm_frame_width N` | `224` | Contact-sheet tile width; aspect ratio is preserved. |
| `--vlm_frames_per_sheet N` | `20` | Maximum number of tiles per sheet. |
| `--vlm_columns N` | `5` | Contact-sheet columns. |
| `--vlm_sheet_overlap_frames N` | `2` | Sampled frames repeated between adjacent sheets; must be below frames-per-sheet. |
| `--vlm_logs_dir PATH` | `logs/subtask_boundaries` | Root for prompts, sheets, sampled indices, outputs, and errors. |

### UVD flags

| Flag | Default | Meaning |
|---|---:|---|
| `--uvd_camera NAME` | `agentview` | Loads `rgb_<NAME>`. |
| `--uvd_preprocessor {vip,r3m,liv,clip,vc1,dinov2,resnet}` | repository UVD default | Frozen visual encoder. |
| `--uvd_device DEVICE` | auto | Examples: `cuda`, `cuda:0`, or `cpu`. |
| `--uvd_pixi_env NAME` | `uvd` | Pixi environment used by the UVD subprocess. |
| `--uvd_pixi_manifest PATH` | repository `pixi.toml` | Manifest declaring the UVD environment. |

## Standalone detector flags

The standalone VLM inspection command accepts:

| Flag | Default | Meaning |
|---|---:|---|
| positional `input` | required | Episode-level NPZ or directory of numeric per-step NPZ files. |
| `--rgb-key KEY` | `rgb` | Episode array key or per-step RGB key. |
| `--provider {qwen,qwen_cloud,openai,gemini}` | `qwen` | VLM backend. |
| `--model MODEL` | provider default | Provider model/served-model name. |
| `--qwen-base-url URL` | `http://127.0.0.1:8000/v1` | Local endpoint. |
| `--sample-every-n-frames N` | `15` | Coarse stride. |
| `--refinement-radius N` | `15` | Dense window radius. |
| `--refinement-stride N` | `1` | Dense window stride. |
| `--no-refine` | off | Disable refinement. |
| `--stop-after-sparse-annotation` | off | Explicitly stop after coarse prediction. |
| `--min-boundary-distance-frames N` | `0` | Conservative close-boundary merge distance. |
| `--sheet-overlap-frames N` | `2` | Sampled frames repeated across sheet seams. |
| `--instruction TEXT` | none | Optional overall task context. |
| `--logs-dir PATH` | `logs/subtask_boundaries` | Sparse input/output log root. |

## Validation

Run the complete offline suite; it does not need a model server or API key:

```bash
pixi run python -m unittest discover -s tests -v
```

Check shell launchers and Python syntax:

```bash
bash -n scripts/setup_qwen_local.sh scripts/serve_qwen_local.sh

pixi run python -m py_compile \
  subtask_boundaries/contact_sheet.py \
  subtask_boundaries/detector.py \
  examples/detect_subtask_boundaries.py \
  external/mimicgen/mimicgen/scripts/generate_extra_keypoints.py
```

## Troubleshooting

**The generator says `[skip] demo_0: already complete`.** Use `--force` if
parameters changed and the existing output must be recomputed. Resume checks
output completeness, not whether the current numerical parameters differ from
the earlier invocation.

**Local Qwen connection is refused.** Start `scripts/serve_qwen_local.sh`, wait
until it prints that the HTTP server started, then check `/health` and
`/v1/models`.

**Local Qwen takes a while to start.** The 35B-A3B FP8 weights are loaded across
both GPUs and Triton kernels are warmed. Subsequent requests reuse the running
server.

**A transition is absent despite a valid third contact sheet.** Inspect the
logged JPEGs. The current generator sends one selected camera. If the
manipulation is outside that camera, smaller sampling stride and sheet overlap
cannot restore missing visual evidence; select a more informative camera with
`--vlm_camera`.

**VLM boundaries are too coarse.** Keep refinement enabled, reduce
`--vlm_sample_every_n_frames`, increase `--vlm_refinement_radius`, or keep
`--vlm_refinement_stride 1`.

**Too many nearly identical VLM boundaries.** Set a small conservative
`--vlm_min_boundary_distance_frames`, such as `3` or `5`. The earlier boundary
is retained.

**AWE DP is slow.** Use `--awe_solver greedy`, process fewer demos with
`--episodes`, or use multiple workers across demos. `--num_workers` cannot split
one DP search.

**UVD import/dependency failure.** Ensure `pixi install -e uvd` succeeds, then
check `--uvd_pixi_manifest`, `--uvd_pixi_env`, and the manual encoder
requirements in `external/UVD/README.md`.
