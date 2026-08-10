"""
Subgoal decomposition combining gripper open/close transitions and
trajectory curvature peaks.

A subgoal boundary is any timestep where EITHER:
  - The gripper transitions open↔close  (same logic as compute_new_goal_gripper_pcd)
  - The EEF trajectory curvature exceeds a threshold  (inflection points)

The goal_gripper_pcd at each timestep is the gripper PCD at the next subgoal boundary.

Usage as a drop-in replacement for compute_new_goal_gripper_pcd:

    from subgoal_decomp import compute_subgoal_gripper_pcd

    goal_gripper_pcd, switch_idxs = compute_subgoal_gripper_pcd(
        gripper_pcd=...,        # (T, 4, 3)
        eef_qpos=...,           # (T, 2)
        actions=...,            # (T, action_dim)
        eef_vel_lin=...,        # (T, 3)
        curvature_threshold=0.5,
        min_segment_len=10,
        return_switch_idxs=True,
    )
"""

import numpy as np
# scipy.signal is imported LAZILY (see curvature_switch_indices below), not
# here: scipy.signal -> scipy.special eagerly pulls in scipy's special-
# function gufuncs (sph_legendre_p & co., added in scipy 1.15), which crash
# on import ("ValueError: All ufuncs must have type numpy.ufunc") under this
# repo's conda-forge Python build regardless of numpy version -- an upstream
# scipy/conda-forge-Python ABI issue, not something this repo can pin its way
# out of. Every OTHER caller of this module (rdp_subgoal_decomp.py wants only
# gripper_switch_indices; bspline_subgoal_decomp.py only needs
# scipy.interpolate, never scipy.signal/scipy.special) doesn't need
# find_peaks, so paying that cost only when curvature_switch_indices is
# actually called unblocks them without touching any pin.


# ---------------------------------------------------------------------------
# Gripper open/close switch indices  (ported from robogen_utils.py)
# ---------------------------------------------------------------------------

def gripper_switch_indices(eef_qpos: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Return timesteps where gripper transitions open↔close."""
    gripper_actions = actions[:, -1]

    derivative = np.gradient(np.abs(eef_qpos), axis=0)
    deriv_right = derivative[:, 0]
    deriv_left  = derivative[:, 1]
    is_closing_right = deriv_right < -1e-3; is_closing_right[:20] = False
    is_closing_left  = deriv_left  < -1e-3; is_closing_left[:20]  = False
    is_closing = np.logical_and(is_closing_left, is_closing_right).astype(int)

    closing_last_indices  = np.where((is_closing[1:] - is_closing[:-1]) == -1)[0] + 1
    opening_first_indices = np.where(
        np.logical_and(
            np.sign(gripper_actions[:-1]) != np.sign(gripper_actions[1:]),
            np.sign(gripper_actions[1:]) == -1,
        )
    )[0] + 1

    switches = []
    last_t   = -1
    mode     = 'close'
    closes   = closing_last_indices.tolist()
    opens    = opening_first_indices.tolist()

    while True:
        if mode == 'close':
            closes = [i for i in closes if i > last_t]
            if not closes:
                break
            t = closes.pop(0)
        else:
            opens = [i for i in opens if i > last_t]
            if not opens:
                break
            t = opens.pop(0)
        switches.append(t)
        last_t = t
        mode = 'open' if mode == 'close' else 'close'

    return np.array(switches, dtype=int)


# ---------------------------------------------------------------------------
# Curvature-based switch indices
# ---------------------------------------------------------------------------

def curvature_switch_indices(
    eef_vel_lin: np.ndarray,
    threshold: float = 0.5,
    min_segment_len: int = 10,
) -> np.ndarray:
    """
    Return timesteps of high curvature in the EEF linear velocity trajectory.

    Curvature = |v × a| / |v|^3  where v = velocity, a = d(velocity)/dt.
    Peaks above `threshold` (after normalisation) are subgoal candidates.
    `min_segment_len` suppresses peaks that are too close together.
    """
    vel = eef_vel_lin.astype(np.float64)          # (T, 3)
    acc = np.gradient(vel, axis=0)                 # (T, 3)

    cross     = np.cross(vel, acc)                 # (T, 3)
    cross_mag = np.linalg.norm(cross, axis=1)      # (T,)
    vel_mag   = np.linalg.norm(vel,   axis=1)      # (T,)

    # avoid division by zero at near-zero velocity
    eps       = 1e-6
    curvature = cross_mag / (vel_mag ** 3 + eps)   # (T,)

    # normalise to [0, 1] for threshold comparison
    c_max = curvature.max()
    if c_max > 0:
        curvature_norm = curvature / c_max
    else:
        return np.array([], dtype=int)

    from scipy.signal import find_peaks  # noqa: PLC0415 -- see the lazy-import note at the top of this file

    peaks, _ = find_peaks(curvature_norm, height=threshold,
                          distance=min_segment_len)
    return peaks.astype(int)


# ---------------------------------------------------------------------------
# Combined decomposition
# ---------------------------------------------------------------------------

def compute_subgoal_gripper_pcd(
    gripper_pcd:        np.ndarray,
    eef_qpos:           np.ndarray,
    actions:            np.ndarray,
    eef_vel_lin:        np.ndarray,
    curvature_threshold: float = 0.5,
    min_segment_len:    int   = 10,
    warmup_steps:       int   = 20,
    return_switch_idxs: bool  = False,
):
    """
    Compute goal_gripper_pcd using gripper transitions + curvature peaks.

    Args:
        gripper_pcd:          (T, 4, 3)
        eef_qpos:             (T, 2)  gripper finger joint positions
        actions:              (T, D)  last dim is gripper action
        eef_vel_lin:          (T, 3)  EEF linear velocity
        curvature_threshold:  fraction of max curvature to use as peak threshold
        min_segment_len:      minimum timesteps between two curvature peaks
        return_switch_idxs:   if True, also return the switch index array

    Returns:
        expanded_goal_gripper_pcd:  (T, 4, 3)
        switch_indices (optional):  (K,) int array
    """
    T = gripper_pcd.shape[0]

    grip_idxs = gripper_switch_indices(eef_qpos, actions)
    curv_idxs = curvature_switch_indices(eef_vel_lin, curvature_threshold,
                                         min_segment_len)

    # merge, sort, deduplicate — no subgoal in first warmup_steps steps
    all_idxs = np.unique(np.concatenate([grip_idxs, curv_idxs])).astype(int)
    all_idxs = all_idxs[(all_idxs >= warmup_steps) & (all_idxs < T)]

    # always end at last timestep
    if len(all_idxs) == 0 or all_idxs[-1] != T - 1:
        all_idxs = np.append(all_idxs, T - 1)

    switch_indices = all_idxs
    repeat_count   = np.insert(np.diff(switch_indices), 0, switch_indices[0])
    repeat_count[-1] += 1

    goal_gripper_pcd          = gripper_pcd[switch_indices]
    expanded_goal_gripper_pcd = np.repeat(goal_gripper_pcd, repeat_count, axis=0)
    assert expanded_goal_gripper_pcd.shape == gripper_pcd.shape, \
        f"Shape mismatch: {expanded_goal_gripper_pcd.shape} vs {gripper_pcd.shape}"

    if return_switch_idxs:
        return expanded_goal_gripper_pcd, switch_indices
    return expanded_goal_gripper_pcd


# ---------------------------------------------------------------------------
# Quick visualisation / debugging
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse, os

    parser = argparse.ArgumentParser()
    parser.add_argument('--demo_dir', type=str, required=True,
                        help='path to demo_0/ directory with per-timestep .npz files')
    parser.add_argument('--curvature_threshold', type=float, default=0.5)
    parser.add_argument('--min_segment_len', type=int, default=10)
    args = parser.parse_args()

    # load trajectory
    T = 0
    while os.path.exists(os.path.join(args.demo_dir, f'{T}.npz')):
        T += 1
    print(f'Loaded {T} timesteps')

    gripper_pcd = np.stack([np.load(os.path.join(args.demo_dir, f'{t}.npz'))['gripper_pcd'][0] for t in range(T)])
    eef_qpos    = np.stack([np.load(os.path.join(args.demo_dir, f'{t}.npz'))['gripper_qpos'][0] for t in range(T)])
    actions     = np.stack([np.load(os.path.join(args.demo_dir, f'{t}.npz'))['action'][0] for t in range(T)])
    eef_vel_lin = np.stack([np.load(os.path.join(args.demo_dir, f'{t}.npz'))['eef_pos'][0] for t in range(T)])
    # use eef_pos finite differences as velocity proxy if vel not saved
    eef_vel_lin = np.gradient(eef_vel_lin, axis=0)

    grip_idxs = gripper_switch_indices(eef_qpos, actions)
    curv_idxs = curvature_switch_indices(eef_vel_lin, args.curvature_threshold,
                                          args.min_segment_len)

    print(f'Gripper switch indices ({len(grip_idxs)}): {grip_idxs}')
    print(f'Curvature peak indices ({len(curv_idxs)}): {curv_idxs}')

    all_idxs = np.unique(np.concatenate([grip_idxs, curv_idxs]))
    print(f'Combined subgoals ({len(all_idxs)}): {all_idxs}')
