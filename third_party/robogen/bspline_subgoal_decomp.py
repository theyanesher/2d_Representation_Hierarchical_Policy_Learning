"""
B-spline subgoal decomposition for trajectory keypoint extraction, ported
from the knot-fitting logic in the sibling `bspline-policy` repo
(bspline_policy/bspline_policy/common/bspline_action.py's
ScipyBSplineCompression / extract_unique_knots) -- same algorithm, not an
approximation of it (this module previously used a scipy.interpolate.splprep
substitute while this repo's SciPy was pinned to 1.10.1; SciPy is now
>=1.15, installed via [pypi-dependencies] since the conda-forge >=1.15 build
is broken here, so the real generate_knots-based fit is used directly).

Companion to subgoal_decomp.py (curvature/jerk), bayesian_subgoal_decomp.py
(BOCPD), rdp_subgoal_decomp.py (RDP-family), and awe_subgoal_decomp.py (AWE).
Same public contract: compute_bspline_subgoal_gripper_pcd returns an expanded
goal_gripper_pcd of shape (T, 4, 3) plus the switch-index array, so the output
is a drop-in interchangeable replacement for the other methods' goal_gripper_pcd.

Method: fit a 3-D B-spline to the EEF position path (eef_pos) via
scipy.interpolate.generate_knots + make_lsq_spline -- the exact FITPACK
knot-refinement loop ScipyBSplineCompression.compress() uses -- find the
SPARSEST knot vector along that refinement path whose reconstruction stays
within `max_error` (Chebyshev / max-abs-error), and treat each interior knot
(via bspline-policy's own extract_unique_knots) as a subgoal boundary --
exactly the same "proprioceptive-only, purely geometric" flavor as RDP, just
with the spline's own adaptive knot placement standing in for RDP's
perpendicular-distance mask. Each frame's subgoal is the gripper keypoints at
the next knot it hasn't passed yet (inclusive) -- goals repeat until the demo
passes each knot, the same convention every sibling module uses for
goal_gripper_pcd.
"""

import numpy as np
from scipy.interpolate import generate_knots, make_lsq_spline

VALID_METHODS = ("bspline",)


def _fit_sparsest_bspline(pos: np.ndarray, u: np.ndarray, degree: int, max_error: float, s: float = 1e-12):
    """Port of bspline-policy's ScipyBSplineCompression.compress(): walk
    generate_knots's refinement sequence (for smoothing factor `s`) and stop
    at the first (sparsest) knot vector whose max-abs reconstruction error
    over all dimensions drops below max_error."""
    last_knots, last_err = None, None
    for knots in generate_knots(u, pos, k=degree, s=s):
        spl = make_lsq_spline(u, pos, knots, k=degree)
        err = float(np.max(np.abs(spl(u) - pos)))
        last_knots, last_err = knots, err
        if err < max_error:
            return knots
    # Search cap reached without hitting the budget -- return the finest
    # (highest knot count) fit found, matching compress()'s own fallback.
    return last_knots


def _bspline_knot_indices(t_full: np.ndarray, degree: int) -> np.ndarray:
    """bspline-policy's extract_unique_knots: strip `degree` (not degree+1)
    repeated boundary knots per side. This leaves one boundary knot at each
    end (0 and T-1) in the result, but _finalize_indices below drops index 0
    and force-appends T-1 regardless, so the extra boundary copies are
    inert -- kept identical to the source repo rather than the mathematically
    "purer" degree+1 strip for exact fidelity."""
    interior = t_full[degree:-degree]
    return np.round(interior).astype(int)


def _finalize_indices(idxs: np.ndarray, num_frames: int) -> np.ndarray:
    """Drop index 0 and any out-of-range, dedup, sort, force a boundary at T-1.
    Same convention as rdp_subgoal_decomp.py's _finalize_indices."""
    idxs = np.asarray(idxs, dtype=int)
    idxs = idxs[(idxs > 0) & (idxs < num_frames)]
    idxs = np.unique(idxs)
    if len(idxs) == 0 or idxs[-1] != num_frames - 1:
        idxs = np.append(idxs, num_frames - 1)
    return idxs


def _expand_to_goal(gripper_pcd: np.ndarray, switch_indices: np.ndarray) -> np.ndarray:
    """Build (T, 4, 3) goal_gripper_pcd: at each step the gripper PCD at the
    next subgoal boundary. Same convention as rdp_subgoal_decomp.py /
    awe_subgoal_decomp.py's expansion."""
    num_frames = gripper_pcd.shape[0]
    switch_indices = np.asarray(switch_indices, dtype=int)

    repeat_count = np.insert(np.diff(switch_indices), 0, switch_indices[0])
    repeat_count[-1] += 1

    goal = gripper_pcd[switch_indices]
    expanded = np.repeat(goal, repeat_count, axis=0)
    assert expanded.shape == gripper_pcd.shape, (
        f"Shape mismatch: {expanded.shape} vs {gripper_pcd.shape}"
    )
    return expanded


def compute_bspline_subgoal_gripper_pcd(
    gripper_pcd: np.ndarray,  # (T, 4, 3)
    eef_pos: np.ndarray,  # (T, 3)
    method: str = "bspline",
    max_error: float = 0.01,  # metres, Chebyshev reconstruction error budget
    degree: int = 3,  # cubic, matches bspline-policy's default
    return_switch_idxs: bool = False,
):
    """
    Compute a goal_gripper_pcd via B-spline knot decomposition: fit the
    sparsest 3-D B-spline to eef_pos meeting `max_error`, use its interior
    knots as subgoal boundaries.

    Returns:
        expanded_goal_gripper_pcd: (T, 4, 3) float32
        switch_indices (optional): (K,) int array (boundaries, ending at T-1)
    """
    if method not in VALID_METHODS:
        raise ValueError(f"Unknown method '{method}'. Valid: {VALID_METHODS}")

    num_frames = gripper_pcd.shape[0]
    pos = np.asarray(eef_pos, dtype=np.float64)
    u = np.arange(num_frames, dtype=np.float64)

    if num_frames > degree:
        knots = _fit_sparsest_bspline(pos, u, degree, max_error)
        idxs = _bspline_knot_indices(knots, degree)
    else:
        # Too few frames to fit a degree-k spline.
        idxs = np.array([], dtype=int)

    switch_indices = _finalize_indices(idxs, num_frames)
    expanded = _expand_to_goal(gripper_pcd, switch_indices).astype(np.float32)

    if return_switch_idxs:
        return expanded, switch_indices
    return expanded
