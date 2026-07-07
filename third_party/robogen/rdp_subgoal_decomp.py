"""
RDP / geometric subgoal decomposition for trajectory keypoint extraction.

Companion to subgoal_decomp.py (curvature/jerk) and bayesian_subgoal_decomp.py
(BOCPD). Same public contract: each method returns an expanded
goal_gripper_pcd of shape (T, 4, 3) plus the switch-index array, so the output
is a drop-in interchangeable replacement for the BOCPD `goal_gripper_pcd`.

Methods (ported from external/mimicgen/mimicgen/scripts/RDP.py, minus the
`heuristic` method which needs joint velocities via _is_stopped):

  - rdp            : Ramer-Douglas-Peucker simplification of the EEF position path.
  - rdp_gripper    : RDP keypoints snapped to gripper open/close transitions.
  - random         : K random keypoints (seeded for reproducibility).
  - fixed_interval : a keypoint every `interval` timesteps.

All signals are proprioceptive (EEF position + gripper state) — no velocity,
no object state, no simulator — so this runs offline on the existing per-step
.npz dataset.

Gripper open/close events reuse gripper_switch_indices() from subgoal_decomp.py
(the same source BOCPD uses) so method comparisons are apples-to-apples.
"""

import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from subgoal_decomp import gripper_switch_indices


# ---------------------------------------------------------------------------
# Ramer-Douglas-Peucker (iterative, N-D, mask output)
# ---------------------------------------------------------------------------

def rdp_mask(points: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Ramer-Douglas-Peucker polyline simplification.

    Returns a boolean mask (T,) marking the points kept to approximate the
    polyline within `epsilon`. Iterative (explicit stack) to avoid Python
    recursion limits on long (~1000-step) trajectories.

    Perpendicular distance in N-D: |(p - a) x (b - a)| / |b - a| for the 3-D
    cross product; for non-3-D inputs falls back to the projection residual.

    Args:
        points:  (T, D) trajectory (here D=3 EEF positions, metres)
        epsilon: tolerance in the same units as `points`
    """
    T = len(points)
    mask = np.zeros(T, dtype=bool)
    if T == 0:
        return mask
    mask[0] = True
    mask[-1] = True
    if T <= 2:
        return mask

    stack = [(0, T - 1)]
    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        a = points[start]
        b = points[end]
        ab = b - a
        ab_norm = np.linalg.norm(ab)
        seg = points[start + 1:end]
        if ab_norm < 1e-12:
            # degenerate segment: distance to the anchor point
            dists = np.linalg.norm(seg - a, axis=1)
        elif points.shape[1] == 3:
            cross = np.cross(seg - a, ab)
            dists = np.linalg.norm(cross, axis=1) / ab_norm
        else:
            # general N-D: residual after projecting onto the segment direction
            u = ab / ab_norm
            diff = seg - a
            proj = np.outer(diff @ u, u)
            dists = np.linalg.norm(diff - proj, axis=1)
        idx_local = int(np.argmax(dists))
        if dists[idx_local] > epsilon:
            idx = start + 1 + idx_local
            mask[idx] = True
            stack.append((start, idx))
            stack.append((idx, end))
    return mask


# ---------------------------------------------------------------------------
# Per-method keypoint index extraction
# ---------------------------------------------------------------------------

def _rdp_keypoints(eef_pos: np.ndarray, epsilon: float) -> np.ndarray:
    mask = rdp_mask(eef_pos.astype(np.float64), epsilon)
    return np.where(mask)[0].astype(int)


def _gripper_keypoints(eef_qpos: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Gripper open/close transitions — the same source BOCPD uses.

    The stored npz `action` has its gripper channel sign-flipped and scaled
    (gripper *= -0.01 in convert_dataset.py). gripper_switch_indices keys only
    on the SIGN of actions[:, -1], so we un-flip by negating that channel to
    recover the raw (+1=close, -1=open) convention the function expects.
    """
    actions_raw = np.asarray(actions, dtype=np.float64).copy()
    actions_raw[:, -1] *= -1.0
    return gripper_switch_indices(eef_qpos, actions_raw)


def _rdp_gripper_keypoints(eef_pos, eef_qpos, actions, epsilon, snap_window):
    """RDP keypoints with each gripper transition snapped onto the nearest RDP
    keypoint within `snap_window`, else kept as its own keypoint."""
    rdp_kps = _rdp_keypoints(eef_pos, epsilon).tolist()
    grip_kps = _gripper_keypoints(eef_qpos, actions).tolist()

    snapped = rdp_kps.copy()
    for gkp in grip_kps:
        if not snapped:
            snapped.append(gkp)
            continue
        dists = [abs(r - gkp) for r in snapped]
        j = int(np.argmin(dists))
        if dists[j] <= snap_window:
            snapped[j] = gkp        # snap: align corner to the grasp/release
        else:
            snapped.append(gkp)     # keep transition as its own keypoint
    return np.array(sorted(set(snapped)), dtype=int)


def _random_keypoints(T: int, k: int, seed: int) -> np.ndarray:
    k = min(k, T)
    rng = np.random.default_rng(seed)
    idxs = rng.choice(T, size=k, replace=False)
    return np.sort(idxs).astype(int)


def _fixed_interval_keypoints(T: int, interval: int) -> np.ndarray:
    if interval is None or interval <= 0:
        interval = max(1, T // 20)
    return np.arange(0, T, interval, dtype=int)


# ---------------------------------------------------------------------------
# Expansion to per-timestep goal (shared contract with BOCPD/curvature paths)
# ---------------------------------------------------------------------------

def _expand_to_goal(gripper_pcd: np.ndarray, switch_indices: np.ndarray):
    """
    Build (T, 4, 3) goal_gripper_pcd: at each step the gripper PCD at the next
    subgoal boundary. Mirrors the expansion in subgoal_decomp.py exactly.

    switch_indices must be sorted, strictly within (0, T), and END at T-1.
    Index 0 is excluded by construction (a boundary at 0 would collapse the
    first segment under np.repeat).
    """
    T = gripper_pcd.shape[0]
    switch_indices = np.asarray(switch_indices, dtype=int)

    repeat_count = np.insert(np.diff(switch_indices), 0, switch_indices[0])
    repeat_count[-1] += 1

    goal = gripper_pcd[switch_indices]
    expanded = np.repeat(goal, repeat_count, axis=0)
    assert expanded.shape == gripper_pcd.shape, \
        f"Shape mismatch: {expanded.shape} vs {gripper_pcd.shape}"
    return expanded


def _finalize_indices(idxs: np.ndarray, T: int) -> np.ndarray:
    """Drop index 0 and any out-of-range, dedup, sort, force a boundary at T-1."""
    idxs = np.asarray(idxs, dtype=int)
    idxs = idxs[(idxs > 0) & (idxs < T)]
    idxs = np.unique(idxs)
    if len(idxs) == 0 or idxs[-1] != T - 1:
        idxs = np.append(idxs, T - 1)
    return idxs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

VALID_METHODS = ("rdp", "rdp_gripper", "random", "fixed_interval")


def compute_rdp_subgoal_gripper_pcd(
    gripper_pcd: np.ndarray,        # (T, 4, 3)
    eef_pos:     np.ndarray,        # (T, 3)
    method:      str,
    eef_qpos:    np.ndarray = None, # (T, 2)  required for rdp_gripper
    actions:     np.ndarray = None, # (T, D)  required for rdp_gripper (last dim = gripper)
    epsilon:     float = 0.02,
    interval:    int = None,
    n_random:    int = 20,
    seed:        int = 0,
    snap_window: int = 5,
    return_switch_idxs: bool = False,
):
    """
    Compute a goal_gripper_pcd via a velocity-free keypoint method.

    Returns:
        expanded_goal_gripper_pcd: (T, 4, 3) float32
        switch_indices (optional): (K,) int array (boundaries, ending at T-1)
    """
    T = gripper_pcd.shape[0]

    if method == "rdp":
        idxs = _rdp_keypoints(eef_pos, epsilon)
    elif method == "rdp_gripper":
        if eef_qpos is None or actions is None:
            raise ValueError("rdp_gripper requires eef_qpos and actions.")
        idxs = _rdp_gripper_keypoints(eef_pos, eef_qpos, actions, epsilon, snap_window)
    elif method == "random":
        idxs = _random_keypoints(T, n_random, seed)
    elif method == "fixed_interval":
        idxs = _fixed_interval_keypoints(T, interval)
    else:
        raise ValueError(f"Unknown method '{method}'. Valid: {VALID_METHODS}")

    switch_indices = _finalize_indices(idxs, T)
    expanded = _expand_to_goal(gripper_pcd, switch_indices).astype(np.float32)

    if return_switch_idxs:
        return expanded, switch_indices
    return expanded
