# RGB temporal subtask boundaries

The consolidated setup, commands, generator integration, and complete flag
reference now live in the repository's
[master keypoint-generation guide](../GENERATE_KEYPOINTS.md).

This package adapts Refiner's sparse, visibly indexed contact-sheet method to
numpy RGB trajectories. It does not decode video or depend on LeRobot.

By default each contact-sheet tile is 224 pixels wide and its height is scaled
to preserve the source aspect ratio. Thus the Coffee data's 256x256 frames
become 224x224 tiles; a full 5-by-4 sheet is 1120x896. The `FRAME <index>` badge
is drawn directly over the top-left of each tile. The terminal frame `T - 1` is
always sampled, adjacent sheets overlap by two sampled frames, and a partial
final sheet is cropped to only the rows containing frames.

```text
NPZ -> RGB frames -> sampled original indices -> FRAME <index> contact sheets
    -> structured coarse VLM prediction -> optional dense local refinement
    -> sorted unique original transition indices
```

Direct use:

```python
import numpy as np

from subtask_boundaries import detect_subtask_boundaries

frames = np.load("episode.npz")["rgb"]
points = detect_subtask_boundaries(
    frames,
    sample_every_n_frames=15,
    refine=True,
    refinement_radius=15,
    refinement_stride=1,
    instruction="put all objects into the drawer",
)
print(points)  # transition-only list[int], e.g. [48, 117, 204]
```

The default backend is local Qwen. It needs no API key and talks to the vLLM
server at `http://127.0.0.1:8000/v1`. The selected checkpoint is the official
multimodal `Qwen/Qwen3.6-35B-A3B-FP8`, served to the detector as
`qwen3.6-local`.

One-time setup (the checkpoint is about 37.5 GB):

```bash
scripts/setup_qwen_local.sh
```

Start the server in a dedicated terminal before detection:

```bash
scripts/serve_qwen_local.sh
```

This machine-specific launcher tensor-parallelizes across GPUs 0 and 1 and
caps context at 16K to leave inference headroom on two 24 GB RTX 4090s. Override
the endpoint with `QWEN_LOCAL_BASE_URL` or `qwen_base_url=` if the server runs
elsewhere.

Gemini remains available using `GEMINI_API_KEY` and defaults to
`gemini-3.5-flash`:

```python
points = detect_subtask_boundaries(frames, provider="gemini")
```

To stop after sparse annotation and avoid the per-boundary dense VLM calls:

```python
points = detect_subtask_boundaries(
    frames,
    provider="qwen",
    stop_after_sparse_annotation=True,
)
```

These returned indices are restricted to the sampled frames, such as
`[45, 120, 210]`. The equivalent command-line flags are
`--stop-after-sparse-annotation` for the standalone example and
`--vlm_stop_after_sparse_annotation` for `generate_extra_keypoints.py`.

Every high-level detector call logs its initial sparse VLM input and output
under a unique directory in `logs/subtask_boundaries`:

```text
logs/subtask_boundaries/run_<UTC time>_<pid>_<id>/
  prompt.txt
  input.json
  contact_sheet_001.jpg
  contact_sheet_002.jpg
  output.json
```

`input.json` records all sampled original frame indices and the indices shown
on each sheet. `output.json` is the exact parsed `BoundaryPrediction` returned
by the VLM. If the request fails, the inputs remain and `error.txt` records the
exception. Set `logs_dir=None` in the Python API to explicitly disable logging.
The extra-keypoint generator nests these run directories below each `demo_*`.

Pass a custom object implementing `BoundaryVLM.predict(...)` to `vlm=` to use
another provider or an offline test double. The old hosted QwenCloud path is
still available explicitly as `provider="qwen_cloud"`; it uses
`DASHSCOPE_API_KEY` and defaults to `qwen3.6-flash`:

```bash
export DASHSCOPE_API_KEY="your-key"
```

```python
points = detect_subtask_boundaries(frames, provider="qwen_cloud")
```

`QWEN_API_KEY` is accepted as a convenience alias. Qwen runs in non-thinking
mode because its visual structured-output interface requires that mode. Sparse
inputs and outputs use the same automatic logging as OpenAI and Gemini.

Run local Qwen3.6 on the Coffee demo and stop after sparse annotation:

```bash
pixi run python \
  examples/detect_subtask_boundaries.py \
  /data/theya/data/uncertainity_subgoal/D1/COFFEE_PREPERATION_D1/demo_0 \
  --rgb-key rgb_agentview \
  --provider qwen \
  --stop-after-sparse-annotation
```

The extra-keypoint generator defaults to the same local provider.

`min_boundary_distance_frames=0` disables distance merging. If set above zero,
sorted predictions closer than that distance are treated as duplicates and the
earlier index is retained, consistent with a boundary meaning the first frame of
the new state. Range sanitation clamps to `[0, T - 1]`, then sorts and removes
exact duplicates.

Provider outputs are also checked against what was visibly shown: invalid coarse
indices are ignored, and an invalid refinement index is snapped to the nearest
visible local index (ties choose the earlier frame).

The repository's per-step keypoint generator also exposes `--methods vlm`. It
stores `goal_gripper_pcd_vlm` in the same mirror format as AWE/B-spline. That
goal schedule includes `T - 1` as a terminal target, but the public detector API
returns only the VLM's task-state transitions.

Run the included example on the provided per-step Coffee demo:

```bash
pixi run python examples/detect_subtask_boundaries.py \
  /data/theya/data/uncertainity_subgoal/D1/COFFEE_PREPERATION_D1/demo_0 \
  --rgb-key rgb_agentview \
  --provider qwen \
  --instruction "prepare coffee"
```

Generate drop-in high-level-policy goals for one demo first:

```bash
pixi run python \
  external/mimicgen/mimicgen/scripts/generate_extra_keypoints.py \
  --data_root /data/theya/data/uncertainity_subgoal/D1 \
  --task COFFEE_PREPERATION_D1 \
  --methods vlm --episodes 1 --dump_indices \
  --vlm_provider qwen \
  --vlm_instruction "prepare coffee"
```

This writes under
`/data/theya/data/uncertainity_subgoal/D1/EXTRA_KEYPOINTS_vlm/COFFEE_PREPERATION_D1`.
After inspection, omit `--episodes 1` for the complete task.

Run the offline tests (no API key or VLM request):

```bash
pixi run python -m unittest discover -s tests -v
```
