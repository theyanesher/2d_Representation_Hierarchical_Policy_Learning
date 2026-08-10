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
within `max_error` (Chebyshev / max-abs-error). Two ways to turn that fit
into subgoal boundaries:

  - "bspline": treat each interior knot (via bspline-policy's own
    extract_unique_knots) as a subgoal boundary -- exactly the same
    "proprioceptive-only, purely geometric" flavor as RDP, just with the
    spline's own adaptive knot placement standing in for RDP's
    perpendicular-distance mask.
  - "bspline_greville": treat each HIGH-INFLUENCE control point as a subgoal
    boundary instead of every knot. A control point's Greville abscissa
    (de Boor, "A Practical Guide to Splines", ch. XI -- the knot-average
    g_i = mean(t[i+1 .. i+degree])) is its associated parameter/frame index;
    "influence" is how sharply the control polygon bends at that point
    (perpendicular distance from the control point to the chord joining its
    immediate neighbors -- an RDP-style test applied to the control polygon
    instead of the raw path). Only control points bending at least
    `influence_threshold` (default: max_error) become subgoals, so this is
    typically SPARSER than the knot count and picks out the control points
    doing the most shaping work, rather than every breakpoint.

Each frame's subgoal is the gripper keypoints at the next boundary it hasn't
passed yet (inclusive) -- goals repeat until the demo passes each boundary,
the same convention every sibling module uses for goal_gripper_pcd.
"""

import numpy as np
from scipy.interpolate import generate_knots, make_lsq_spline

VALID_METHODS = ("bspline", "bspline_greville")


def _fit_sparsest_bspline(pos: np.ndarray, u: np.ndarray, degree: int, max_error: float, s: float = 1e-12):
    """Port of bspline-policy's ScipyBSplineCompression.compress(): walk
    generate_knots's refinement sequence (for smoothing factor `s`) and stop
    at the first (sparsest) knot vector whose max-abs reconstruction error
    over all dimensions drops below max_error. Returns (knots, spl) -- the
    fitted BSpline is kept (not just its knots) so callers that need control
    points (e.g. the "bspline_greville" method) don't have to refit."""
    last_knots, last_spl, last_err = None, None, None
    for knots in generate_knots(u, pos, k=degree, s=s):
        spl = make_lsq_spline(u, pos, knots, k=degree)
        err = float(np.max(np.abs(spl(u) - pos)))
        last_knots, last_spl, last_err = knots, spl, err
        if err < max_error:
            return knots, spl
    # Search cap reached without hitting the budget -- return the finest
    # (highest knot count) fit found, matching compress()'s own fallback.
    return last_knots, last_spl


def _bspline_knot_indices(t_full: np.ndarray, degree: int) -> np.ndarray:
    """bspline-policy's extract_unique_knots: strip `degree` (not degree+1)
    repeated boundary knots per side. This leaves one boundary knot at each
    end (0 and T-1) in the result, but _finalize_indices below drops index 0
    and force-appends T-1 regardless, so the extra boundary copies are
    inert -- kept identical to the source repo rather than the mathematically
    "purer" degree+1 strip for exact fidelity."""
    interior = t_full[degree:-degree]
    return np.round(interior).astype(int)


def _greville_abscissae(knots: np.ndarray, degree: int) -> np.ndarray:
    """Greville abscissae a.k.a. knot averages: g_i = mean(t[i+1 .. i+degree]),
    one per control point (de Boor, "A Practical Guide to Splines", ch. XI).
    This is the parameter/frame index each control point is associated with
    -- NOT a point on the curve itself, just where along u it "sits"."""
    n_ctrl = len(knots) - degree - 1
    return np.array(
        [np.mean(knots[i + 1 : i + degree + 1]) for i in range(n_ctrl)]
    )


def _control_polygon_deviation(ctrl_pts: np.ndarray) -> np.ndarray:
    """Perpendicular distance of each interior control point from the chord
    joining its immediate neighbors -- an RDP-style bend test applied to the
    control polygon instead of the raw trajectory. Large deviation means the
    fit needed that control point to turn sharply, i.e. it's doing real
    shaping work ("maximum influence" on the curve); a control point sitting
    on the chord between its neighbors is redundant with them and shouldn't
    become its own subgoal. Endpoints get 0 (no two-sided neighbor pair;
    already covered by _finalize_indices' forced start/end boundaries)."""
    n = ctrl_pts.shape[0]
    dev = np.zeros(n, dtype=np.float64)
    if n < 3:
        return dev
    a, b, c = ctrl_pts[:-2], ctrl_pts[1:-1], ctrl_pts[2:]
    chord = c - a
    chord_norm = np.linalg.norm(chord, axis=1)
    cross = np.cross(b - a, chord)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.linalg.norm(cross, axis=1) / chord_norm
    degenerate = chord_norm < 1e-12
    d = np.where(degenerate, np.linalg.norm(b - a, axis=1), d)
    dev[1:-1] = d
    return dev


def _greville_influence_indices(
    knots: np.ndarray, ctrl_pts: np.ndarray, degree: int, influence_threshold: float
) -> np.ndarray:
    """Frame indices of the high-influence control points: Greville abscissa
    (rounded to the nearest frame) of every control point whose control-polygon
    deviation clears `influence_threshold`. Sparser than the full knot set
    returned by _bspline_knot_indices -- only the control points actually
    doing shaping work, not every breakpoint."""
    greville = _greville_abscissae(knots, degree)
    deviation = _control_polygon_deviation(ctrl_pts)
    keep = deviation >= influence_threshold
    return np.round(greville[keep]).astype(int)


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
    influence_threshold: float = None,  # bspline_greville only; metres, default: max_error
    return_switch_idxs: bool = False,
):
    """
    Compute a goal_gripper_pcd via B-spline decomposition. Both methods share
    the same underlying fit (sparsest spline meeting `max_error`); they only
    differ in how boundaries are read off it:

      - "bspline": every interior knot is a subgoal boundary.
      - "bspline_greville": only the control points whose control-polygon
        deviation clears `influence_threshold` (default: max_error) become
        boundaries, located via their Greville abscissa -- sparser, biased
        toward the control points doing the most shaping work.

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
        knots, spl = _fit_sparsest_bspline(pos, u, degree, max_error)
        if method == "bspline_greville":
            thresh = max_error if influence_threshold is None else influence_threshold
            idxs = _greville_influence_indices(knots, spl.c, degree, thresh)
        else:
            idxs = _bspline_knot_indices(knots, degree)
    else:
        # Too few frames to fit a degree-k spline.
        idxs = np.array([], dtype=int)

    switch_indices = _finalize_indices(idxs, num_frames)
    expanded = _expand_to_goal(gripper_pcd, switch_indices).astype(np.float32)

    if return_switch_idxs:
        return expanded, switch_indices
    return expanded
