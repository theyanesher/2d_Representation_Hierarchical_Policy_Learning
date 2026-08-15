"""
UVD (Universal Visual Decomposer, external/UVD) subgoal decomposition for
trajectory keypoint extraction (https://github.com/zcczhang/UVD).

Companion to subgoal_decomp.py (curvature/jerk), bayesian_subgoal_decomp.py
(BOCPD), rdp_subgoal_decomp.py (RDP-family), bspline_subgoal_decomp.py, and
awe_subgoal_decomp.py. Same public contract: compute_uvd_subgoal_gripper_pcd
returns an expanded goal_gripper_pcd of shape (T, 4, 3) plus the switch-index
array, so the output is a drop-in interchangeable replacement for the
BOCPD/RDP/B-spline/AWE `goal_gripper_pcd`.

Unlike every other method here (proprioceptive-only: EEF position, gripper
state, actions -- no pixels), UVD decomposes purely from RGB video: it
embeds each frame with a frozen visual encoder (VIP/R3M/LIV/CLIP/VC1/DINOv2/
ResNet) and finds "milestones" as local extrema of the embedding-distance
curve computed backward from the final frame (see
external/UVD/uvd/decomp/decomp.py:embedding_decomp). No sim, no
end-effector geometry.

CROSS-INTERPRETER SPLIT -- read this before changing the CLI contract below:
external/UVD requires Python 3.9 with old, tightly-pinned deps (torch==2.0.1,
gym==0.21.0, dm_control==1.0.11, ...), isolated in this repo's own `uvd` pixi
environment (see pixi.toml [feature.uvd.*] / `pixi run -e uvd`) -- entirely
separate from the Python 3.10 default environment
generate_extra_keypoints.py runs in. That means this module CANNOT be
imported directly from generate_extra_keypoints.py the way
rdp/bspline/awe_subgoal_decomp.py are (`import uvd` would fail outright, or
worse, partially succeed against the wrong torch/numpy ABI). Instead this
file doubles as a standalone CLI: generate_extra_keypoints.py shells out to
it via `pixi run -e uvd python uvd_subgoal_decomp.py --input ... --output
...`, one subprocess per demo (see _compute_uvd_via_subprocess in
generate_extra_keypoints.py). That subprocess cost (reloading the frozen
encoder each demo) is the trade-off for correctness across the interpreter
boundary; if throughput ever matters, the fix is a persistent worker process
reading a queue of demos instead of one-shot invocations, not a change to
the public contract here.

Everything below (compute_uvd_subgoal_gripper_pcd et al.) only ever runs
inside the `uvd` environment -- never assume default-env deps (this repo's
own numpy/torch pins) are present when editing this file.
"""

import argparse

import numpy as np


VALID_METHODS = ("uvd",)
# UVD's own README notes vip/r3m/liv/vc1 each need a separate manual install
# (clone + `pip install -e .` + a one-off checkpoint download) that this
# repo's `pixi run -e uvd install-uvd` task deliberately does NOT do (see
# pixi.toml [feature.uvd.tasks]) -- only external/UVD/requirements.txt
# itself is installed. "resnet" (torchvision, ImageNet-pretrained) and
# "dinov2" (torch.hub) work with nothing beyond that, so "resnet" is the
# default here; pass --preprocessor_name explicitly for the others once
# you've installed them per external/UVD/README.md.
DEFAULT_PREPROCESSOR = "resnet"


def _finalize_indices(idxs: np.ndarray, T: int) -> np.ndarray:
    """Drop index 0 and any out-of-range, dedup, sort, force a boundary at
    T-1 -- identical convention to rdp_subgoal_decomp.py's _finalize_indices,
    duplicated here (rather than imported) so this file has zero dependency
    on the default-env-only modules; see the cross-interpreter note above."""
    idxs = np.asarray(idxs, dtype=int)
    idxs = idxs[(idxs > 0) & (idxs < T)]
    idxs = np.unique(idxs)
    if len(idxs) == 0 or idxs[-1] != T - 1:
        idxs = np.append(idxs, T - 1)
    return idxs


def _expand_to_goal(gripper_pcd: np.ndarray, switch_indices: np.ndarray) -> np.ndarray:
    """Build (T, 4, 3) goal_gripper_pcd: at each step the gripper PCD at the
    next subgoal boundary. Duplicated from rdp_subgoal_decomp.py -- see
    _finalize_indices above for why."""
    T = gripper_pcd.shape[0]
    switch_indices = np.asarray(switch_indices, dtype=int)

    repeat_count = np.insert(np.diff(switch_indices), 0, switch_indices[0])
    repeat_count[-1] += 1

    goal = gripper_pcd[switch_indices]
    expanded = np.repeat(goal, repeat_count, axis=0)
    assert expanded.shape == gripper_pcd.shape, \
        f"Shape mismatch: {expanded.shape} vs {gripper_pcd.shape}"
    return expanded


def compute_uvd_subgoal_gripper_pcd(
    gripper_pcd: np.ndarray,  # (T, 4, 3)
    rgb_frames: np.ndarray,  # (T, H, W, 3) uint8
    preprocessor_name: str = DEFAULT_PREPROCESSOR,
    device: str = None,
    return_switch_idxs: bool = False,
):
    """
    Compute a goal_gripper_pcd via UVD (Universal Visual Decomposer).

    Returns:
        expanded_goal_gripper_pcd: (T, 4, 3) float32
        switch_indices (optional): (K,) int array (boundaries, ending at T-1)
    """
    import torch
    import uvd

    T = gripper_pcd.shape[0]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    milestone_indices = uvd.get_uvd_subgoals(
        rgb_frames,
        preprocessor_name=preprocessor_name,
        device=device,
        return_indices=True,
    )

    switch_indices = _finalize_indices(np.asarray(milestone_indices), T)
    expanded = _expand_to_goal(gripper_pcd, switch_indices).astype(np.float32)

    if return_switch_idxs:
        return expanded, switch_indices
    return expanded


def _main():
    ap = argparse.ArgumentParser(
        description="UVD subgoal decomposition (runs inside the `uvd` pixi "
                    "environment -- see the module docstring)."
    )
    ap.add_argument("--input", required=True,
                    help="npz with 'rgb' (T,H,W,3) uint8 and 'gripper_pcd' (T,4,3) float32")
    ap.add_argument("--output", required=True,
                    help="npz to write with 'goal_gripper_pcd' (T,4,3) float32 and "
                         "'switch_idxs' (K,) int64")
    ap.add_argument("--preprocessor_name", default=DEFAULT_PREPROCESSOR,
                    choices=["vip", "r3m", "liv", "clip", "vc1", "dinov2", "resnet"])
    ap.add_argument("--device", default=None, help="e.g. cuda, cuda:0, cpu (default: auto)")
    args = ap.parse_args()

    data = np.load(args.input)
    goal, switch_idxs = compute_uvd_subgoal_gripper_pcd(
        gripper_pcd=data["gripper_pcd"],
        rgb_frames=data["rgb"],
        preprocessor_name=args.preprocessor_name,
        device=args.device,
        return_switch_idxs=True,
    )
    np.savez(args.output, goal_gripper_pcd=goal, switch_idxs=switch_idxs.astype(np.int64))


if __name__ == "__main__":
    _main()
