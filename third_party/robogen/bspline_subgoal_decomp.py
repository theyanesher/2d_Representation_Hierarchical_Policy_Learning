"""
B-spline subgoal decomposition for trajectory keypoint extraction, ported
from the knot-fitting logic in the sibling `bspline-policy` repo
(bspline_policy/bspline_policy/common/bspline_action.py's
ScipyBSplineCompression / extract_unique_knots).

Companion to subgoal_decomp.py (curvature/jerk), bayesian_subgoal_decomp.py
(BOCPD), rdp_subgoal_decomp.py (RDP-family), and awe_subgoal_decomp.py (AWE).
Same public contract: compute_bspline_subgoal_gripper_pcd returns an expanded
goal_gripper_pcd of shape (T, 4, 3) plus the switch-index array, so the output
is a drop-in interchangeable replacement for the other methods' goal_gripper_pcd.

Method: fit a 3-D B-spline to the EEF position path (eef_pos), find the
SPARSEST knot vector whose reconstruction stays within `max_error` (Chebyshev
/ max-abs-error, same criterion bspline-policy's ScipyBSplineCompression.compress
uses), and treat each interior knot as a subgoal boundary -- exactly the same
"proprioceptive-only, purely geometric" flavor as RDP, just with the spline's
own adaptive knot placement standing in for RDP's perpendicular-distance mask.
Each frame's subgoal is the gripper keypoints at the next knot it hasn't
passed yet (inclusive) -- goals repeat until the demo passes each knot, the
same convention every sibling module uses for goal_gripper_pcd.

Deviation from bspline-policy's own fitting code (deliberate, documented):
bspline-policy's ScipyBSplineCompression.compress() searches over
scipy.interpolate.generate_knots(), a knot-refinement generator that only
exists in SciPy >= 1.15. This repo's pinned SciPy is 1.10.1 (no
generate_knots). Rather than bump SciPy (a fairly invasive, blast-radius-y
change to a core numerical dependency shared by the whole codebase), this
module reproduces the same "find the sparsest spline meeting an error budget"
search using the classic FITPACK API that IS available in 1.10.1:
scipy.interpolate.splprep, which fits an N-D parametric B-spline with
automatic interior-knot placement controlled by a smoothing factor `s` (s=0
interpolates every sample exactly; larger s -> fewer knots -> looser fit).
_fit_sparsest_bspline exponentially grows s from 0, then log-scale
bisects, to find the largest s (sparsest knot set) whose max-abs
reconstruction error is still <= max_error -- the same stopping criterion as
ScipyBSplineCompression.compress(max_error=...), just reached via a different
(older, stable) SciPy API. Verified empirically against real trajectory data:
splprep's knot values land exactly on integer frame indices (the u values
passed in), so int(round(...)) recovers exact frame indices, not an
approximation.

extract_unique_knots in bspline-policy strips `degree` (not degree+1) knots
from each end of the FITPACK full knot vector -- appropriate for THEIR
downstream chunking logic, which wants one repeated boundary knot to remain.
Here we strip degree+1 knots from each end, the mathematically correct count
for FITPACK's k+1-fold-repeated clamped boundary convention, to get the true
*interior* (non-boundary-duplicated) knots -- verified empirically (a fit
with 64 total knots at degree=3 has exactly 64 - 2*4 = 56 interior knots when
stripping degree+1=4 per side).
"""

import numpy as np
from scipy.interpolate import splev, splprep

VALID_METHODS = ("bspline",)


def _fit_sparsest_bspline(pos: np.ndarray, u: np.ndarray, degree: int, max_error: float):
    """Find the largest smoothing factor `s` (sparsest knot vector) for
    scipy.interpolate.splprep such that the max-abs reconstruction error over
    all dimensions stays within max_error. See module docstring for why this
    substitutes for bspline-policy's generate_knots-based search.
    """

    def fit(s):
        tck, _ = splprep([pos[:, i] for i in range(pos.shape[1])], u=u, k=degree, s=s)
        recon = np.stack(splev(u, tck), axis=1)
        err = float(np.max(np.abs(recon - pos)))
        return tck, err

    tck_lo, err_lo = fit(0.0)  # s=0: interpolates every sample, error ~0 (floor)
    if err_lo > max_error:
        # Can't do better than exact interpolation and still miss the budget --
        # shouldn't happen in practice; return the finest fit we have.
        return tck_lo

    s_lo = 0.0
    s_hi = 1e-8
    tck_hi, err_hi = fit(s_hi)
    while err_hi <= max_error and s_hi < 1e8:
        s_lo, tck_lo = s_hi, tck_hi
        s_hi *= 10.0
        tck_hi, err_hi = fit(s_hi)

    if err_hi <= max_error:
        # Never violated the budget even at the search cap -- coarsest fit found.
        return tck_hi

    # Binary search in log(s) between the last-good s_lo and first-bad s_hi.
    for _ in range(30):
        s_mid = (s_lo * s_hi) ** 0.5  # geometric mean = bisection in log-space
        tck_mid, err_mid = fit(s_mid)
        if err_mid <= max_error:
            s_lo, tck_lo = s_mid, tck_mid
        else:
            s_hi = s_mid
    return tck_lo


def _bspline_knot_indices(tck, degree: int) -> np.ndarray:
    """Interior (non-boundary) knot positions -> nearest integer frame index."""
    t_full = np.asarray(tck[0], dtype=np.float64)
    interior = t_full[degree + 1 : -(degree + 1)]
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
        tck = _fit_sparsest_bspline(pos, u, degree, max_error)
        idxs = _bspline_knot_indices(tck, degree)
    else:
        # Too few frames to fit a degree-k spline (splprep requires T > k).
        idxs = np.array([], dtype=int)

    switch_indices = _finalize_indices(idxs, num_frames)
    expanded = _expand_to_goal(gripper_pcd, switch_indices).astype(np.float32)

    if return_switch_idxs:
        return expanded, switch_indices
    return expanded
