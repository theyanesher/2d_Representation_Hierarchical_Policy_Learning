"""
Offline keypoint / subgoal generator — velocity-free, no simulator.

Reads the already-rendered per-timestep .npz dataset produced by
convert_dataset.py and writes ADDITIONAL goal_gripper_pcd variants computed by
the RDP-family methods (rdp, rdp_gripper, random, fixed_interval), the
B-spline methods (bspline: every interior knot; bspline_greville: only the
high-influence control points, via their Greville abscissae), AWE / Automatic
Waypoint Extraction (awe), VLM temporal boundaries (vlm), UVD / Universal
Visual Decomposer (uvd), and three local heuristic methods (gripper_heuristic: subgoal at
every gripper open/close transition, no EEF-path geometry at all;
fixed_interval_const: subgoal every `--const_interval` steps, fixed at 50 by
default and independent of `--interval`/T; orientation_heuristic: subgoal
whenever the gripper's orientation drifts more than `--orientation_threshold`
radians, combined across all 3 axes, from the orientation at the last
subgoal). The original dataset is treated as READ-ONLY; new keys live in a
mirror tree, one per requested method-set so different --methods runs never
collide or overwrite each other:

    <DATA_ROOT>/<TASK>/<demo>/<t>.npz                                (original, untouched)
    <DATA_ROOT>/EXTRA_KEYPOINTS_<method1>_<method2>/<TASK>/<demo>/<t>.npz   (new keys, this script)

'awe' appears in that directory name as awe-<solver>-th<err_threshold> (e.g.
EXTRA_KEYPOINTS_awe-greedy-th0.3), since --awe_solver/--awe_err_threshold change
its keypoints without changing its key name; every other method is named plainly.

Each new .npz holds one key per method, saved IDENTICALLY to the original
`goal_gripper_pcd` ((1, 4, 3) float32) so they are drop-in interchangeable:

    goal_gripper_pcd_rdp, goal_gripper_pcd_rdp_gripper,
    goal_gripper_pcd_random, goal_gripper_pcd_fixed_interval,
    goal_gripper_pcd_bspline, goal_gripper_pcd_awe, goal_gripper_pcd_vlm,
    goal_gripper_pcd_uvd,
    goal_gripper_pcd_gripper_heuristic, goal_gripper_pcd_fixed_interval_const,
    goal_gripper_pcd_orientation_heuristic

--mix_methods mingles 2+ of the methods above into one extra
goal_gripper_pcd_mix_<method1>_<method2>_... key: every boundary from the
highest-priority (first-listed) method is kept, and a lower-priority
method's boundary is dropped instead of added whenever it falls within
--mix_window frames of a boundary already kept from a higher-priority method
(default: 5 frames, shared by every mix group in the run). Every method
named in a --mix_methods group MUST also be named in --methods (this is
enforced, not auto-added). Repeat the flag for more than one independent
mix, e.g. --methods gripper_heuristic orientation_heuristic rdp awe
--mix_methods gripper_heuristic orientation_heuristic --mix_methods rdp awe
produces gripper_heuristic, orientation_heuristic, rdp, and awe normally,
plus goal_gripper_pcd_mix_gripper_heuristic_orientation_heuristic and
goal_gripper_pcd_mix_rdp_awe.

Incremental & safe: adding a method later loads the existing mirror .npz, keeps
all prior keys verbatim, adds the new ones, and rewrites the whole file. Nothing
irreplaceable lives in the mirror tree, so a mistake costs only a recompute.

Example:
    python generate_extra_keypoints.py \
        --data_root /scratch/.../GROOT_STYLE_DATASET/D2 \
        --task COFFEE_PREPERATION_D1 \
        --methods gripper_heuristic orientation_heuristic \
        --mix_methods gripper_heuristic orientation_heuristic --mix_window 5 \
        --episodes 5            # inspect a few first; drop for the full run
"""
import os
import sys
import glob
import json
import argparse
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.spatial.transform import Rotation

# third_party/robogen on the path (rdp_subgoal_decomp + subgoal_decomp +
# bspline_subgoal_decomp + awe_subgoal_decomp + uvd_subgoal_decomp live there)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "third_party", "robogen"))
from rdp_subgoal_decomp import (
    compute_rdp_subgoal_gripper_pcd, VALID_METHODS as RDP_METHODS,
    _finalize_indices, _expand_to_goal,
)
from bspline_subgoal_decomp import compute_bspline_subgoal_gripper_pcd, VALID_METHODS as BSPLINE_METHODS
from subgoal_decomp import gripper_switch_indices
# uvd_subgoal_decomp itself only imports numpy/argparse at module scope (its
# torch/uvd imports are deferred into compute_uvd_subgoal_gripper_pcd/_main),
# so importing it here -- just to get VALID_METHODS/DEFAULT_PREPROCESSOR -- is
# safe even though this script runs in the default Python 3.10 environment,
# not the `uvd` Python 3.9 one those deferred imports require. See
# _compute_uvd_via_subprocess below for how the actual computation crosses
# that interpreter boundary.
from uvd_subgoal_decomp import VALID_METHODS as UVD_METHODS, DEFAULT_PREPROCESSOR as UVD_DEFAULT_PREPROCESSOR
_UVD_SCRIPT = os.path.join(_ROOT, "third_party", "robogen", "uvd_subgoal_decomp.py")
# awe_subgoal_decomp is imported LAZILY (see _get_awe_fn below), not here: it
# eagerly pulls in robosuite -> numba-jitted transform utils on import, which
# sets the CPU's flush-denormals-to-zero flag process-wide -- that silently
# perturbs scipy.interpolate.splprep's convergence in bspline_subgoal_decomp's
# _fit_sparsest_bspline, making bspline's knot count depend on whether AWE was
# ever imported in the same process. An unconditional top-level import would
# contaminate every non-awe run; importing only when 'awe' is requested keeps
# rdp/bspline-only runs byte-for-byte reproducible.
_awe_fn = None


def _get_awe_fn():
    global _awe_fn
    if _awe_fn is None:
        from awe_subgoal_decomp import compute_awe_subgoal_gripper_pcd
        _awe_fn = compute_awe_subgoal_gripper_pcd
    return _awe_fn

EXTRA_DIRNAME_PREFIX = "EXTRA_KEYPOINTS"
# awe_subgoal_decomp's own VALID_METHODS=("greedy","dp") names its internal
# solver, not an output key -- "awe" is the single goal_gripper_pcd_awe key
# this script produces; --awe_solver below picks which of greedy/dp computes it.
AWE_METHODS = ("awe",)
# VLM temporal segmentation uses raw RGB frames and original trajectory
# indices. Its OpenAI adapter is imported only when this method runs.
VLM_METHODS = ("vlm",)
# Local heuristic methods (implemented in this file, not third_party/robogen):
#   gripper_heuristic   -- subgoal at every gripper open/close transition,
#                           with NO EEF-path geometry at all (unlike
#                           rdp_gripper, which snaps RDP corners onto these
#                           transitions instead of using them directly).
#   fixed_interval_const -- subgoal every --const_interval steps (default 50),
#                           always -- unlike 'fixed_interval' this ignores T
#                           and gives every demo/task the same cadence.
#   orientation_heuristic -- subgoal whenever the gripper's orientation has
#                           rotated more than --orientation_threshold radians
#                           (default pi/6), combined across all 3 axes as a
#                           single geodesic angle, away from the orientation
#                           at the last subgoal boundary; the base
#                           orientation starts out as the very first frame's
#                           orientation.
HEURISTIC_METHODS = ("gripper_heuristic", "fixed_interval_const", "orientation_heuristic")
VALID_METHODS = (
    RDP_METHODS + BSPLINE_METHODS + AWE_METHODS + VLM_METHODS
    + HEURISTIC_METHODS + UVD_METHODS
)
# Methods that read a real gripper open/close signal (gripper_qpos and/or the
# action's gripper channel) and are therefore meaningless -- not merely
# degenerate -- on a gripperless embodiment. --gripperless rejects these
# instead of running them against synthesized zeros, which would silently
# yield "0 keypoints everywhere" results that look like a tuning problem.
GRIPPER_DEPENDENT_METHODS = ("rdp_gripper", "gripper_heuristic")
METHOD_KEY = {m: "goal_gripper_pcd_{}".format(m) for m in VALID_METHODS}


def _mix_name(group):
    """Method/key name for one --mix_methods group, e.g.
    ["gripper_heuristic", "orientation_heuristic"] -> "mix_gripper_heuristic_
    orientation_heuristic" -- so the key itself says which methods (and in
    which priority order) it mingled, instead of a single generic 'mixed'
    name that would collide across different --mix_methods groups."""
    return "mix_" + "_".join(group)


def _gripper_heuristic_keypoints(gripper_qpos, actions):
    """Pure gripper open/close heuristic: a subgoal boundary at every
    open<->close transition. Reuses gripper_switch_indices() (the same
    source BOCPD and rdp_gripper use) so it stays apples-to-apples with the
    other methods' gripper-transition handling."""
    actions_raw = np.asarray(actions, dtype=np.float64).copy()
    actions_raw[:, -1] *= -1.0   # undo convert_dataset.py's gripper sign-flip/scale
    return gripper_switch_indices(gripper_qpos, actions_raw)


def _fixed_interval_const_keypoints(T, interval):
    """A subgoal boundary every `interval` timesteps, unconditionally."""
    return np.arange(0, T, interval, dtype=int)


def _exact_count_keypoints(T, n_keypoints):
    """Exactly `n_keypoints` evenly spaced boundaries, ending at T-1.

    fixed_interval's default (interval = T//K) does NOT give K subgoals: the
    integer floor leaves a remainder that _finalize_indices turns into extra
    boundaries, and the overshoot grows with K and varies demo to demo
    (K=60 on hammer_cleanup yields 64-73). That makes 'number of keypoints'
    unusable as a swept x-axis, so parameterise by the count directly and let
    the spacing fall out of it."""
    return np.unique(np.linspace(0, T - 1, int(n_keypoints) + 1).astype(int))


def _quat_angle(q1, q2):
    """Combined absolute rotation (radians) taking q1 -> q2, as the geodesic
    angle of the relative rotation R2 @ R1^-1 -- i.e. the single angle of the
    axis-angle representation, which folds the x/y/z contributions into one
    magnitude (not a per-axis check).
    """
    r1 = Rotation.from_quat(q1)
    r2 = Rotation.from_quat(q2)
    rel = r2 * r1.inv()
    return np.linalg.norm(rel.as_rotvec())


def _orientation_heuristic_keypoints(eef_quat, threshold):
    """A subgoal boundary wherever the gripper orientation's combined
    rotation (all 3 axes together, as one geodesic angle) exceeds `threshold`
    radians away from the orientation at the last subgoal (initially the
    very first frame's orientation). Each time a boundary is placed, the base
    orientation resets to that frame, so drift is measured relative to the
    most recent subgoal rather than accumulating from t=0."""
    eef_quat = np.asarray(eef_quat, dtype=np.float64)
    T = eef_quat.shape[0]
    base = eef_quat[0]
    idxs = []
    for t in range(1, T):
        if _quat_angle(base, eef_quat[t]) > threshold:
            idxs.append(t)
            base = eef_quat[t]
    return np.asarray(idxs, dtype=int)


def _merge_switch_idxs(idx_lists, window):
    """Mingle several methods' switch-index lists into one, in priority
    order: idx_lists[0] is highest priority. Every one of its indices is
    kept; for each subsequent (lower-priority) list, an index is kept only
    if it is NOT within `window` frames of an index already kept, so a
    near-duplicate boundary from a lower-priority method collapses onto the
    higher-priority one instead of adding a redundant extra subgoal.

    The window check only ever applies ACROSS lists, never within one list:
    the first (highest-priority) list is taken in full, unfiltered, before
    any window check runs -- checking a list's own entries against `kept`
    while `kept` is still being populated FROM that same list would silently
    drop that list's own close-together boundaries too, breaking "every one
    of idx_lists[0]'s indices is kept" whenever it has any two boundaries
    within `window` frames of each other (common in practice: e.g. bspline's
    knots are frequently only a few frames apart)."""
    kept = []
    for list_idx, idxs in enumerate(idx_lists):
        sorted_idxs = sorted(int(x) for x in idxs)
        if list_idx == 0:
            kept.extend(sorted_idxs)
            continue
        for idx in sorted_idxs:
            if all(abs(idx - k) > window for k in kept):
                kept.append(idx)
    return np.asarray(sorted(kept), dtype=int)


def _awe_token(opts):
    """awe-<solver>-th<err_threshold>, e.g. awe-greedy-th0.3.

    awe is the one method whose output depends on flags outside --methods:
    --awe_solver and --awe_err_threshold change the keypoints but not the
    key name (always goal_gripper_pcd_awe). Without those flags in the
    directory name, a second run with different settings would resolve to
    the same mirror tree, be judged 'already complete' by the resume check,
    and silently either skip or (with --force) overwrite the earlier
    settings' results. Every other method is fully described by its name."""
    tok = "awe-{}-th{:g}".format(opts.awe_solver, opts.awe_err_threshold)
    # --awe_use_gripper changes the keypoints the same way solver/threshold do,
    # so it belongs in the tree name too. Absent from the name for the default
    # (off) case, so existing geometric-only trees keep resolving as before.
    if getattr(opts, "awe_use_gripper", False):
        tok += "-grip"
    return tok


def _extra_dirname(methods, opts):
    """EXTRA_KEYPOINTS_<method1>_<method2>_... -- keeps different --methods
    runs (e.g. rdp-only vs. bspline-only vs. all) in separate mirror trees so
    they never collide or partially overwrite each other. 'awe' expands to
    _awe_token(opts) so its solver/threshold variants separate too, and
    --dirname_suffix separates runs that differ in something the name cannot
    otherwise express (a reworked vlm prompt, a different sampling stride)."""
    toks = [_awe_token(opts) if m == "awe" else m for m in methods]
    name = "_".join([EXTRA_DIRNAME_PREFIX] + toks)
    suffix = getattr(opts, "dirname_suffix", "") or ""
    return "{}_{}".format(name, suffix.strip("_")) if suffix.strip("_") else name


def _sorted_step_files(demo_dir):
    files = glob.glob(os.path.join(demo_dir, "*.npz"))
    return sorted(files, key=lambda p: int(os.path.basename(p)[:-4]))


def _load_demo_arrays(step_files, gripperless=False):
    """Stack the per-step keys this generator needs into (T, ...) arrays.

    `gripperless` is for datasets whose end-effector has no fingers at all --
    PushT, a planar pusher, is the case this was added for: its .npz carries
    no `gripper_qpos` key, and its `action` is a bare (x, y) target with no
    gripper channel appended. Both are synthesized here as constant zeros so
    every downstream consumer keeps its usual array shapes -- crucially AWE,
    which reads `actions[:, -1]` as the gripper open/close command and would
    otherwise treat PushT's y-target as a gripper signal flipping almost
    every frame, forcing a waypoint at nearly every timestep. Constant zeros
    make that term inert instead. Methods that genuinely *read* a gripper
    signal are rejected up front in main() rather than quietly handed these
    zeros (see GRIPPER_DEPENDENT_METHODS)."""
    eef_pos, eef_quat, gripper_qpos, action, gripper_pcd = [], [], [], [], []
    for f in step_files:
        d = np.load(f)
        eef_pos.append(d["eef_pos"][0])
        eef_quat.append(d["eef_quat"][0])
        gripper_qpos.append(
            np.zeros(1, dtype=np.float64) if gripperless else d["gripper_qpos"][0])
        action.append(
            np.concatenate([np.asarray(d["action"][0], dtype=np.float64), [0.0]])
            if gripperless else d["action"][0])
        gripper_pcd.append(d["gripper_pcd"][0])
    return (np.asarray(eef_pos), np.asarray(eef_quat), np.asarray(gripper_qpos),
            np.asarray(action), np.asarray(gripper_pcd, dtype=np.float32))


def _load_rgb_frames(step_files, camera):
    """Stack one demo's rgb_<camera> frames into a (T, H, W, 3) uint8 array.
    Only called when an RGB method (vlm or uvd) is requested, so runs using
    only proprioceptive methods do not decompress every image."""
    key = "rgb_{}".format(camera)
    frames = [np.load(f)[key][0] for f in step_files]
    return np.asarray(frames, dtype=np.uint8)


def _compute_vlm_boundaries(rgb_frames, opts, logs_dir):
    """Return transition-only original indices from coarse-to-fine VLM analysis."""
    from subtask_boundaries import detect_subtask_boundaries

    return detect_subtask_boundaries(
        rgb_frames,
        provider=opts.vlm_provider,
        model=opts.vlm_model,
        qwen_base_url=opts.vlm_qwen_base_url,
        sample_every_n_frames=opts.vlm_sample_every_n_frames,
        refine=opts.vlm_refine,
        stop_after_sparse_annotation=opts.vlm_stop_after_sparse_annotation,
        refinement_radius=opts.vlm_refinement_radius,
        refinement_stride=opts.vlm_refinement_stride,
        min_boundary_distance_frames=opts.vlm_min_boundary_distance_frames,
        frame_width=opts.vlm_frame_width,
        frames_per_sheet=opts.vlm_frames_per_sheet,
        columns=opts.vlm_columns,
        sheet_overlap_frames=opts.vlm_sheet_overlap_frames,
        instruction=opts.vlm_instruction,
        logs_dir=logs_dir,
    )


def _compute_uvd_via_subprocess(gripper_pcd, rgb_frames, opts):
    """UVD (external/UVD) needs its own Python 3.9 stack (torch==2.0.1,
    gym==0.21.0, dm_control==1.0.11, ...), isolated in this repo's `uvd`
    pixi environment -- entirely separate from the Python 3.10 interpreter
    this script runs under (see pixi.toml [feature.uvd.*]). So unlike every
    other method here, UVD can't be called in-process: this shells out to
    uvd_subgoal_decomp.py's CLI via `pixi run -e uvd`, passing the RGB
    frames + gripper keypoints through a temp .npz and reading the result
    back the same way. That's one subprocess (and one frozen-encoder reload)
    per demo -- see uvd_subgoal_decomp.py's module docstring for the
    tradeoff and how to fix it if throughput ever matters."""
    with tempfile.TemporaryDirectory(prefix="uvd_subgoal_") as tmpdir:
        in_path = os.path.join(tmpdir, "in.npz")
        out_path = os.path.join(tmpdir, "out.npz")
        np.savez(in_path, rgb=rgb_frames, gripper_pcd=gripper_pcd)

        cmd = [
            "pixi", "run", "--manifest-path", opts.uvd_pixi_manifest,
            "-e", opts.uvd_pixi_env, "python", _UVD_SCRIPT,
            "--input", in_path, "--output", out_path,
            "--preprocessor_name", opts.uvd_preprocessor,
        ]
        if opts.uvd_device:
            cmd += ["--device", opts.uvd_device]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "uvd_subgoal_decomp.py failed (exit {}):\n{}".format(
                    result.returncode, result.stderr[-4000:]))

        with np.load(out_path) as d:
            goal = d["goal_gripper_pcd"]
            switch_idxs = d["switch_idxs"]
    return goal, switch_idxs


def _write_mirror(out_demo_dir, method_goals, n_steps):
    """Write per-step mirror .npz, preserving any keys already there.

    method_goals: {key_name -> (T, 4, 3) float32}
    """
    os.makedirs(out_demo_dir, exist_ok=True)
    for t in range(n_steps):
        out_path = os.path.join(out_demo_dir, "{}.npz".format(t))
        payload = {}
        if os.path.exists(out_path):                      # preserve-keys rewrite
            try:
                with np.load(out_path) as existing:
                    payload = {k: existing[k] for k in existing.files}
            except Exception:                             # truncated by a mid-write kill
                payload = {}
        for key, goal in method_goals.items():
            payload[key] = goal[t][None, :].astype(np.float32)   # (1, 4, 3)
        np.savez_compressed(out_path, **payload)


def _demo_is_complete(demo_dir, out_demo_dir, methods, dump_indices):
    """True iff out_demo_dir already holds a finished result for `methods`.

    Completion signals, cheapest-first:
      - every mirror step file 0..T-1 exists,
      - the last mirror file carries all requested method keys (so a run that
        only had *some* methods before is treated as incomplete and re-run),
      - if dump_indices, _keypoints.json exists with matching T (written LAST,
        so its presence proves the demo finished, not just wrote some steps).
    A demo killed mid-write fails one of these and is regenerated.
    """
    step_files = _sorted_step_files(demo_dir)
    T = len(step_files)
    if T == 0:
        return True                                       # nothing to generate
    for t in range(T):
        if not os.path.exists(os.path.join(out_demo_dir, "{}.npz".format(t))):
            return False
    last = os.path.join(out_demo_dir, "{}.npz".format(T - 1))
    try:
        with np.load(last) as d:
            have = set(d.files)
    except Exception:
        return False
    if not all(METHOD_KEY[m] in have for m in methods):
        return False
    if dump_indices:
        kp = os.path.join(out_demo_dir, "_keypoints.json")
        if not os.path.exists(kp):
            return False
        try:
            with open(kp) as fh:
                if json.load(fh).get("T") != T:
                    return False
        except Exception:
            return False
    return True


def process_demo(demo_dir, out_demo_dir, methods, opts, dump_indices=False):
    step_files = _sorted_step_files(demo_dir)
    T = len(step_files)
    if T == 0:
        return 0, {}
    eef_pos, eef_quat, gripper_qpos, action, gripper_pcd = _load_demo_arrays(
        step_files, gripperless=getattr(opts, "gripperless", False))
    rgb_by_camera = {}
    if any(m in VLM_METHODS for m in methods):
        rgb_by_camera[opts.vlm_camera] = _load_rgb_frames(step_files, opts.vlm_camera)
    if any(m in UVD_METHODS for m in methods) and opts.uvd_camera not in rgb_by_camera:
        rgb_by_camera[opts.uvd_camera] = _load_rgb_frames(step_files, opts.uvd_camera)

    mix_groups = opts.mix_groups or []
    mix_names = {_mix_name(g) for g in mix_groups}

    method_goals = {}
    idx_record = {}
    for m in methods:
        if m in mix_names:
            continue   # needs its sub-methods' switch_idxs; handled below, once this loop is done
        if m in VLM_METHODS:
            boundaries = _compute_vlm_boundaries(
                rgb_by_camera[opts.vlm_camera],
                opts,
                os.path.join(opts.vlm_logs_dir, os.path.basename(demo_dir)),
            )
            # The public detector returns only true transitions. This generator's
            # goal schedule additionally needs T-1 as its terminal target, matching
            # AWE/B-spline and every existing goal_gripper_pcd method.
            switch_idxs = _finalize_indices(np.asarray(boundaries, dtype=int), T)
            goal = _expand_to_goal(gripper_pcd, switch_idxs).astype(np.float32)
        elif m in UVD_METHODS:
            goal, switch_idxs = _compute_uvd_via_subprocess(
                gripper_pcd, rgb_by_camera[opts.uvd_camera], opts)
        elif m in BSPLINE_METHODS:
            goal, switch_idxs = compute_bspline_subgoal_gripper_pcd(
                gripper_pcd=gripper_pcd,
                eef_pos=eef_pos,
                method=m,
                max_error=opts.max_error,
                degree=opts.degree,
                influence_threshold=opts.influence_threshold,
                return_switch_idxs=True,
            )
        elif m in HEURISTIC_METHODS:
            if m == "gripper_heuristic":
                idxs = _gripper_heuristic_keypoints(gripper_qpos, action)
            elif m == "orientation_heuristic":
                idxs = _orientation_heuristic_keypoints(eef_quat, opts.orientation_threshold)
            else:  # fixed_interval_const
                idxs = _fixed_interval_const_keypoints(T, opts.const_interval)
            switch_idxs = _finalize_indices(idxs, T)
            goal = _expand_to_goal(gripper_pcd, switch_idxs).astype(np.float32)
        elif m in AWE_METHODS:
            goal, switch_idxs = _get_awe_fn()(
                gripper_pcd=gripper_pcd,
                eef_pos=eef_pos,
                eef_quat=eef_quat,
                actions=action,
                err_threshold=opts.awe_err_threshold,
                method=opts.awe_solver,
                pos_only=False,
                use_gripper_seeding=opts.awe_use_gripper,
                return_switch_idxs=True,
            )
        elif m == "fixed_interval" and getattr(opts, "n_keypoints", None):
            # Count-parameterised fixed interval (see _exact_count_keypoints).
            switch_idxs = _finalize_indices(
                _exact_count_keypoints(T, opts.n_keypoints), T)
            goal = _expand_to_goal(gripper_pcd, switch_idxs).astype(np.float32)
        else:
            goal, switch_idxs = compute_rdp_subgoal_gripper_pcd(
                gripper_pcd=gripper_pcd,
                eef_pos=eef_pos,
                method=m,
                eef_qpos=gripper_qpos,
                actions=action,
                epsilon=opts.epsilon,
                interval=opts.interval,
                n_random=opts.n_random,
                seed=opts.seed,
                snap_window=opts.snap_window,
                return_switch_idxs=True,
            )
        method_goals[METHOD_KEY[m]] = goal
        idx_record[m] = [int(x) for x in switch_idxs]

    for group in mix_groups:
        name = _mix_name(group)
        sub_idx_lists = [idx_record[m] for m in group]
        merged = _merge_switch_idxs(sub_idx_lists, opts.mix_window)
        merged = _finalize_indices(merged, T)
        method_goals[METHOD_KEY[name]] = _expand_to_goal(gripper_pcd, merged).astype(np.float32)
        idx_record[name] = [int(x) for x in merged]

    _write_mirror(out_demo_dir, method_goals, T)

    if dump_indices:
        with open(os.path.join(out_demo_dir, "_keypoints.json"), "w") as fh:
            json.dump({"T": T, "keypoints": idx_record}, fh, indent=2)

    return T, idx_record


def _process_demo_worker(demo, demo_dir, out_demo_dir, methods, opts, dump_indices):
    """Top-level (picklable) wrapper so process_demo can run inside a
    ProcessPoolExecutor worker -- one worker per demo. Parallelism is only
    ACROSS demos: each demo's waypoint search (dp in particular, O(T^3) and
    strictly sequential over the trajectory) cannot itself be split across
    workers, so --num_workers only pays off with multiple demos queued."""
    T, idx_record = process_demo(demo_dir, out_demo_dir, methods, opts, dump_indices=dump_indices)
    return demo, T, idx_record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/data/theya/data/uncertainity_subgoal/D1",
                    help="dataset root that holds <TASK>/ folders (default: "
                         "/data/theya/data/uncertainity_subgoal/D1)")
    ap.add_argument("--task", required=True,
                    help="task folder name, e.g. COFFEE_PREPERATION_D1")
    ap.add_argument("--gripperless", action="store_true",
                    help="dataset has no gripper: its .npz carries no gripper_qpos "
                         "and its action has no gripper channel (e.g. PushT, a planar "
                         "pusher). Synthesizes both as constant zeros so AWE's "
                         "actions[:, -1] gripper term stays inert, and rejects the "
                         "methods needing a real gripper signal ({}).".format(
                             ", ".join(GRIPPER_DEPENDENT_METHODS)))
    ap.add_argument("--methods", nargs="+", default=["rdp"],
                    help="any of {} or 'all'".format(list(VALID_METHODS)))
    ap.add_argument("--episodes", "-n", type=int, default=None,
                    help="process only the first N demos (for inspection)")
    ap.add_argument("--epsilon", type=float, default=0.02, help="RDP tolerance (metres)")
    ap.add_argument("--interval", type=int, default=None,
                    help="fixed_interval step (default: T//20)")
    ap.add_argument("--n_keypoints", type=int, default=None,
                    help="fixed_interval: place EXACTLY this many evenly spaced "
                         "subgoals per demo (overrides --interval). Use this when "
                         "sweeping the keypoint count as an experimental variable -- "
                         "--interval's T//K floor overshoots the requested count by a "
                         "demo-dependent amount that grows with K.")
    ap.add_argument("--out_root", default=None,
                    help="root to write the EXTRA_KEYPOINTS_* mirror tree under "
                         "(default: --data_root). Point this elsewhere to keep a "
                         "sweep out of the main dataset tree.")
    ap.add_argument("--out_suffix", default="",
                    help="appended to the EXTRA_KEYPOINTS_<methods> directory name, so "
                         "runs of the SAME method with different settings (e.g. one per "
                         "--n_keypoints value) land in separate trees instead of "
                         "overwriting each other.")
    ap.add_argument("--const_interval", type=int, default=50,
                    help="fixed_interval_const: timesteps between subgoals, "
                         "constant across demos/tasks (default: 50)")
    ap.add_argument("--n_random", type=int, default=20, help="keypoints for 'random'")
    ap.add_argument("--seed", type=int, default=0, help="seed for 'random'")
    ap.add_argument("--snap_window", type=int, default=5,
                    help="frames within which rdp_gripper snaps to a gripper transition")
    ap.add_argument("--max_error", type=float, default=0.08,
                    help="bspline: max-abs (Chebyshev) EEF reconstruction error budget, metres")
    ap.add_argument("--degree", type=int, default=3,
                    help="bspline: spline degree (3 = cubic, matches bspline-policy's default)")
    ap.add_argument("--influence_threshold", type=float, default=None,
                    help="bspline_greville: min control-polygon deviation (metres) for a "
                         "control point to become a subgoal boundary; default: --max_error")
    ap.add_argument("--awe_err_threshold", type=float, default=0.2, # 0.2 default
                    help="awe: max reconstruction error (position in metres, "
                         "+ rotation in radians) before AWE adds another waypoint")
    ap.add_argument("--awe_use_gripper", action="store_true",
                    help="awe: let upstream AWE force a waypoint at every gripper "
                         "open/close transition (actions[:, -1] changing), on top of "
                         "its geometric waypoints. OFF by default -- see "
                         "awe_subgoal_decomp._select_waypoints. Both solvers seed "
                         "this way, so the flag applies to greedy and dp alike. "
                         "Trees built with it get a '-grip' suffix in the AWE token.")
    ap.add_argument("--awe_solver", choices=["greedy", "dp"], default="dp",
                    help="awe: greedy (fast, near-optimal) or dp (optimal, "
                         "O(T^3) -- short demos only, roughly <= a few hundred frames)")
    ap.add_argument("--vlm_provider",
                    choices=["qwen", "qwen_cloud", "openai", "gemini"],
                    default="qwen",
                    help="vlm: API provider (default: local qwen)")
    ap.add_argument("--vlm_model", default=None,
                    help="vlm: model ID (default: qwen3.6-local for local Qwen, "
                         "qwen3.6-flash for QwenCloud, gpt-5.4 for OpenAI, "
                         "gemini-3.5-flash for Gemini)")
    ap.add_argument("--vlm_qwen_base_url", default=None,
                    help="vlm: local Qwen endpoint "
                         "(default: http://127.0.0.1:8000/v1)")
    ap.add_argument("--vlm_camera", default="agentview",
                    help="vlm: which rgb_<camera> key to inspect (default: agentview)")
    ap.add_argument("--vlm_sample_every_n_frames", type=int, default=15,
                    help="vlm: coarse contact-sheet stride in original frames (default: 15)")
    ap.add_argument("--vlm_refine", dest="vlm_refine", action="store_true",
                    help="vlm: densely refine each coarse transition (default)")
    ap.add_argument("--no_vlm_refine", dest="vlm_refine", action="store_false",
                    help="vlm: keep sparse coarse transition indices")
    ap.set_defaults(vlm_refine=True)
    ap.add_argument("--vlm_stop_after_sparse_annotation",
                    "--vlm-stop-after-sparse-annotation",
                    dest="vlm_stop_after_sparse_annotation", action="store_true",
                    help="vlm: stop after the sparse coarse annotation pass and "
                         "skip all dense boundary-refinement requests")
    ap.add_argument("--vlm_refinement_radius", type=int, default=15,
                    help="vlm: frames on each side of a coarse boundary (default: 15)")
    ap.add_argument("--vlm_refinement_stride", type=int, default=1,
                    help="vlm: original-frame stride inside refinement windows (default: 1)")
    ap.add_argument("--vlm_min_boundary_distance_frames", type=int, default=0,
                    help="vlm: merge refined boundaries closer than this many frames, "
                         "keeping the earlier one; 0 disables merging (default: 0)")
    ap.add_argument("--vlm_frame_width", type=int, default=224,
                    help="vlm: resized contact-sheet tile width (default: 224)")
    ap.add_argument("--vlm_frames_per_sheet", type=int, default=20,
                    help="vlm: maximum contact-sheet tiles (default: 20)")
    ap.add_argument("--vlm_columns", type=int, default=5,
                    help="vlm: contact-sheet columns (default: 5)")
    ap.add_argument("--vlm_sheet_overlap_frames", type=int, default=2,
                    help="vlm: sampled frames repeated across consecutive contact "
                         "sheets (default: 2)")
    ap.add_argument("--vlm_instruction", default=None,
                    help="vlm: optional episode task instruction used only as context")
    ap.add_argument("--vlm_logs_dir", default=os.path.join("logs", "subtask_boundaries"),
                    help="vlm: root directory for per-demo sparse input/output logs "
                         "(default: logs/subtask_boundaries)")
    ap.add_argument("--orientation_threshold", type=float, default=np.pi / 4,
                    help="orientation_heuristic: radians of gripper-orientation drift, "
                         "combined across all 3 axes as one geodesic angle, from the "
                         "last subgoal before a new one is placed (default: pi/6)")
    ap.add_argument("--mix_methods", nargs="+", action="append", default=None,
                    help="mingle 2+ methods' subgoal boundaries into one extra "
                         "'mix_<method1>_<method2>_...' method/key (e.g. --mix_methods "
                         "gripper_heuristic orientation_heuristic produces "
                         "goal_gripper_pcd_mix_gripper_heuristic_orientation_heuristic). "
                         "Listed order sets priority -- earlier methods' boundaries win "
                         "over nearby later ones, see --mix_window. Every method listed "
                         "here must ALSO be listed in --methods (not auto-added). Repeat "
                         "the flag for multiple independent mixes, e.g. --mix_methods "
                         "gripper_heuristic orientation_heuristic --mix_methods rdp awe.")
    ap.add_argument("--mix_window", type=int, default=5,
                    help="mingling (--mix_methods): a lower-priority method's boundary "
                         "within this many frames of an already-kept higher-priority "
                         "boundary is dropped instead of kept as a separate subgoal -- "
                         "applies to every --mix_methods group (default: 5)")
    ap.add_argument("--uvd_camera", default="agentview",
                    help="uvd: which rgb_<camera> key to decompose (default: agentview)")
    ap.add_argument("--uvd_preprocessor", default=UVD_DEFAULT_PREPROCESSOR,
                    choices=["vip", "r3m", "liv", "clip", "vc1", "dinov2", "resnet"],
                    help="uvd: frozen visual encoder (default: {}, installed at external/vip; "
                         "r3m/liv/vc1 still need a manual install first, see "
                         "external/UVD/README.md)".format(UVD_DEFAULT_PREPROCESSOR))
    ap.add_argument("--uvd_device", default=None,
                    help="uvd: e.g. cuda, cuda:0, cpu (default: auto-detect inside the "
                         "uvd subprocess)")
    ap.add_argument("--uvd_pixi_env", default="uvd",
                    help="uvd: pixi environment name UVD's Python 3.9 stack lives in "
                         "(default: uvd, see pixi.toml [feature.uvd.*])")
    ap.add_argument("--uvd_pixi_manifest", default=os.path.join(_ROOT, "pixi.toml"),
                    help="uvd: path to the pixi.toml declaring --uvd_pixi_env")
    ap.add_argument("--dump_indices", action="store_true",
                    help="also write _keypoints.json per demo (for viser inspection)")
    ap.add_argument("--dirname_suffix", default="",
                    help="append this tag to the EXTRA_KEYPOINTS_<methods> directory "
                         "name, so runs differing in something the name cannot express "
                         "(vlm prompt revision, sampling stride) land in separate trees "
                         "instead of resuming or overwriting each other")
    ap.add_argument("--force", action="store_true",
                    help="recompute every demo even if it looks complete (disable resume)")
    ap.add_argument("--num_workers", type=int, default=1,
                    help="parallelize ACROSS demos (one process per demo in flight). "
                         "Each demo's own waypoint search -- dp especially, O(T^3) and "
                         "inherently sequential -- cannot be split further, so this only "
                         "helps when multiple demos are queued (default: 1, sequential)")
    args = ap.parse_args()

    methods = list(VALID_METHODS) if args.methods == ["all"] else args.methods
    bad = [m for m in methods if m not in VALID_METHODS]
    if bad:
        raise SystemExit("Unknown method(s) {}. Valid: {}".format(bad, list(VALID_METHODS)))
    if args.gripperless and args.awe_use_gripper:
        raise SystemExit(
            "--awe_use_gripper needs a gripper open/close signal, which a "
            "--gripperless dataset does not have. Drop one of the two flags.")
    if args.gripperless:
        needs_gripper = [m for m in methods if m in GRIPPER_DEPENDENT_METHODS]
        if needs_gripper:
            raise SystemExit(
                "--gripperless cannot run {}: they read a real gripper open/close "
                "signal, which this dataset does not have. Drop them from --methods "
                "(and from any --mix_methods group).".format(needs_gripper))
    # canonical order so e.g. --methods bspline rdp and --methods rdp bspline
    # land in the same mirror tree instead of silently forking into two.
    methods = sorted(methods, key=list(VALID_METHODS).index)

    args.mix_groups = args.mix_methods   # list of groups (each a priority-ordered list), or None
    if args.mix_groups is not None:
        for group in args.mix_groups:
            if len(group) < 2:
                raise SystemExit("--mix_methods needs at least 2 methods to mingle, got {}".format(group))
            bad_mix = [m for m in group if m not in VALID_METHODS]
            if bad_mix:
                raise SystemExit("Unknown --mix_methods method(s) {}. Valid: {}".format(
                    bad_mix, list(VALID_METHODS)))
            # sub-methods must run this pass so process_demo has their switch_idxs
            # to mingle -- require them in --methods rather than silently adding.
            missing_mix = [m for m in group if m not in methods]
            if missing_mix:
                raise SystemExit(
                    "--mix_methods {} also need to be in --methods (missing: {}). "
                    "Add them explicitly, e.g. --methods {} ...".format(
                        group, missing_mix, " ".join(group)))
            name = _mix_name(group)
            METHOD_KEY[name] = "goal_gripper_pcd_{}".format(name)
            if name not in methods:
                methods.append(name)
            print("[extra-keypoints] mixing {} -> '{}' (priority order left-to-right, "
                  "window={} frames)".format(group, name, args.mix_window))

    task_dir = os.path.join(args.data_root, args.task)
    out_task_dir = os.path.join(
        args.out_root or args.data_root,
        _extra_dirname(methods, args) + args.out_suffix,
        args.task,
    )
    if not os.path.isdir(task_dir):
        raise SystemExit("Task dir not found: {}".format(task_dir))

    demos = sorted(
        [d for d in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, d))],
        key=lambda s: int(s.split("_")[-1]) if s.split("_")[-1].isdigit() else 1 << 30,
    )
    if args.episodes is not None:
        demos = demos[:args.episodes]

    print("[extra-keypoints] methods={} | {} demos | src={}".format(methods, len(demos), task_dir))
    print("[extra-keypoints] writing -> {}".format(out_task_dir))

    counts = {m: [] for m in methods}
    n_skipped = 0
    pending = []
    for demo in demos:
        demo_dir = os.path.join(task_dir, demo)
        out_demo_dir = os.path.join(out_task_dir, demo)
        if not args.force and _demo_is_complete(demo_dir, out_demo_dir, methods, args.dump_indices):
            n_skipped += 1
            print("[skip] {}: already complete".format(demo))
            continue
        pending.append((demo, demo_dir, out_demo_dir))

    n_done = 0
    if args.num_workers <= 1:
        for demo, demo_dir, out_demo_dir in pending:
            _, T, idx_record = _process_demo_worker(
                demo, demo_dir, out_demo_dir, methods, args, args.dump_indices)
            n_done += 1
            for m in methods:
                counts[m].append(len(idx_record.get(m, [])))
            print("[done] {} ({}/{}): T={} | {}".format(
                demo, n_done, len(pending), T,
                " ".join("{}={}kp".format(m, len(idx_record.get(m, []))) for m in methods)))
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futures = {
                ex.submit(_process_demo_worker, demo, demo_dir, out_demo_dir, methods, args,
                          args.dump_indices): demo
                for demo, demo_dir, out_demo_dir in pending
            }
            for fut in as_completed(futures):
                demo, T, idx_record = fut.result()
                n_done += 1
                for m in methods:
                    counts[m].append(len(idx_record.get(m, [])))
                print("[done] {} ({}/{}): T={} | {}".format(
                    demo, n_done, len(pending), T,
                    " ".join("{}={}kp".format(m, len(idx_record.get(m, []))) for m in methods)))

    if n_skipped:
        print("[resume] skipped {}/{} already-complete demos (use --force to recompute)".format(
            n_skipped, len(demos)))

    print("\n[summary] keypoints per demo:")
    for m in methods:
        c = np.array(counts[m]) if counts[m] else np.array([0])
        print("  {:16s} mean={:.1f}  min={}  max={}".format(m, c.mean(), int(c.min()), int(c.max())))


if __name__ == "__main__":
    main()
