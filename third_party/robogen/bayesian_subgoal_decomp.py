"""
Bayesian Online Changepoint Detection (BOCPD) for trajectory subgoal decomposition.

Drop-in companion to subgoal_decomp.py. The public entry point is:

    compute_bayesian_subgoal_gripper_pcd(
        gripper_pcd, eef_qpos, actions, eef_pos, eef_quat, eef_vel_lin,
        config, return_switch_idxs=False
    )

Algorithm:
  1. Extract a 4D kinematic feature vector at each timestep:
       [curvature, velocity magnitude, jerk, angular velocity magnitude]
  2. Z-score each feature dimension over the full trajectory.
  3. Run BOCPD with a Normal-Inverse-Wishart conjugate prior to get
     P(changepoint at t) for every t.
  4. Extract subgoal indices via peak detection on the changepoint probabilities.
  5. OR with gripper open/close transitions (same logic as subgoal_decomp.py).

All signals are proprioceptive — no object state needed — so this is
sim-to-real transferable.

Reference: Adams & MacKay (2007), "Bayesian Online Changepoint Detection".
Hyperparameters live in bocpd_config.yaml (loaded by the caller).
"""

import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from subgoal_decomp import gripper_switch_indices


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _compute_features(eef_pos: np.ndarray,
                      eef_quat: np.ndarray,
                      eef_vel_lin: np.ndarray,
                      config: dict) -> np.ndarray:
    """
    Compute kinematic feature matrix with shape (T, d), where d is the number
    of enabled features (controlled by use_* flags in config).

    Feature order (when enabled): curvature, velocity_mag, jerk, angular_vel_mag.

    Args:
        eef_pos:     (T, 3)
        eef_quat:    (T, 4)  xyzw (scipy convention)
        eef_vel_lin: (T, 3)
        config:      dict from bocpd_config.yaml

    Returns:
        features: (T, d)
    """
    use_curvature    = bool(config.get('use_curvature',    True))
    use_velocity_mag = bool(config.get('use_velocity_mag', True))
    use_jerk         = bool(config.get('use_jerk',         True))
    use_angular_vel  = bool(config.get('use_angular_vel',  True))

    vel = eef_vel_lin.astype(np.float64)
    acc = np.gradient(vel, axis=0)
    vel_mag = np.linalg.norm(vel, axis=1)

    cols = []

    if use_curvature:
        eps = 1e-6
        cross = np.cross(vel, acc)
        cross_mag = np.linalg.norm(cross, axis=1)
        curvature = cross_mag / (vel_mag ** 3 + eps)
        curvature[vel_mag < eps] = 0.0
        cols.append(curvature)

    if use_velocity_mag:
        cols.append(vel_mag)

    if use_jerk:
        jerk_vec = np.gradient(acc, axis=0)
        jerk = np.linalg.norm(jerk_vec, axis=1)
        cols.append(jerk)

    if use_angular_vel:
        T = len(eef_quat)
        ang_vel = np.zeros(T)
        for t in range(T - 1):
            r_curr = Rotation.from_quat(eef_quat[t])
            r_next = Rotation.from_quat(eef_quat[t + 1])
            delta_r = r_next * r_curr.inv()
            ang_vel[t] = delta_r.magnitude()
        ang_vel[-1] = ang_vel[-2]
        cols.append(ang_vel)

    if not cols:
        raise ValueError("bocpd_config: at least one feature must be enabled.")

    return np.stack(cols, axis=1)  # (T, d)


def _standardize(features: np.ndarray) -> np.ndarray:
    """Z-score each feature dimension over the full trajectory."""
    mu = features.mean(axis=0)
    sigma = features.std(axis=0)
    sigma[sigma < 1e-8] = 1.0  # avoid division by zero for constant features
    return (features - mu) / sigma


# ---------------------------------------------------------------------------
# NIW predictive (multivariate Student-t)
# ---------------------------------------------------------------------------

def _log_student_t(x: np.ndarray, mu: np.ndarray,
                   Sigma: np.ndarray, nu: float) -> float:
    """Log density of the multivariate Student-t distribution."""
    d = len(x)
    diff = x - mu
    sign, logdet = np.linalg.slogdet(Sigma)
    if sign <= 0:
        return -np.inf
    maha = float(diff @ np.linalg.solve(Sigma, diff))
    return (gammaln((nu + d) / 2.0)
            - gammaln(nu / 2.0)
            - (d / 2.0) * np.log(nu * np.pi)
            - 0.5 * logdet
            - ((nu + d) / 2.0) * np.log(1.0 + maha / nu))


def _predictive_log_prob(x: np.ndarray,
                         n: int, mean: np.ndarray, S: np.ndarray,
                         kappa_0: float, nu_0: float,
                         m_0: np.ndarray, Psi_0: np.ndarray) -> float:
    """
    NIW posterior predictive log P(x | x_{1:n}).
    When n=0 uses the prior directly.
    """
    d = len(x)
    kappa_n = kappa_0 + n
    m_n = (kappa_0 * m_0 + n * mean) / kappa_n if n > 0 else m_0.copy()
    nu_n = nu_0 + n

    if n > 0:
        diff_m = mean - m_0
        Psi_n = Psi_0 + S + (kappa_0 * n / kappa_n) * np.outer(diff_m, diff_m)
    else:
        Psi_n = Psi_0.copy()

    nu_pred = nu_n - d + 1
    if nu_pred <= 0:
        return -np.inf

    Sigma_pred = Psi_n * (kappa_n + 1) / (kappa_n * nu_pred)
    Sigma_pred += np.eye(d) * 1e-6  # ridge for numerical stability

    return _log_student_t(x, m_n, Sigma_pred, nu_pred)


# ---------------------------------------------------------------------------
# Welford online update for sufficient statistics
# ---------------------------------------------------------------------------

def _welford_update(n: int, mean: np.ndarray, S: np.ndarray,
                    x: np.ndarray):
    """Add one observation x to the running sufficient statistics."""
    n_new = n + 1
    delta = x - mean
    mean_new = mean + delta / n_new
    S_new = S + (n / n_new) * np.outer(delta, delta)
    return n_new, mean_new, S_new


# ---------------------------------------------------------------------------
# BOCPD core
# ---------------------------------------------------------------------------

def _bocpd(features: np.ndarray, config: dict) -> np.ndarray:
    """
    Run BOCPD on a standardized feature sequence.

    Args:
        features: (T, d) standardized feature array
        config:   dict loaded from bocpd_config.yaml

    Returns:
        changepoint_probs: (T,) — P(r_t = 0 | x_{1:t}) for each t
    """
    T, d = features.shape

    lambda_       = float(config['lambda_'])
    tau_min       = int(config['tau_min'])
    kappa_0       = float(config['kappa_0'])
    nu_0          = float(config['nu_0'])
    psi_0_scale   = float(config['psi_0_scale'])
    r_max         = int(config['r_max'])
    log_prob_floor = float(config['log_prob_floor'])

    m_0   = np.zeros(d)
    Psi_0 = np.eye(d) * psi_0_scale

    log_H    = -np.log(lambda_)                  # log(1/lambda)
    log_1mH  = np.log(1.0 - 1.0 / lambda_)       # log(1 - 1/lambda)

    max_r = min(T, r_max)

    # log P(r_t = r, x_{1:t}) for each run length r
    log_R = np.full(max_r + 1, -np.inf)
    log_R[0] = 0.0   # P(r_0 = 0) = 1

    # Sufficient statistics per run length
    ns    = np.zeros(max_r + 1, dtype=int)
    means = np.zeros((max_r + 1, d))
    Ss    = np.zeros((max_r + 1, d, d))

    changepoint_probs = np.zeros(T)

    for t in range(T):
        x = features[t]
        n_active = min(t, max_r)

        # --- Step 1: predictive likelihoods ---
        log_preds = np.full(n_active + 1, -np.inf)
        for r in range(n_active + 1):
            if r == 0 or log_R[r] > log_prob_floor:
                log_preds[r] = _predictive_log_prob(
                    x, ns[r], means[r], Ss[r], kappa_0, nu_0, m_0, Psi_0)

        # --- Step 2: growth and changepoint probabilities ---
        log_R_new = np.full(max_r + 1, -np.inf)

        # Growth: run length r -> r+1
        # Option 2 hazard: gate on absolute time t, not run length r.
        # Before warmup: H_t = 0 (no changepoint possible at all).
        # After warmup:  H_t = 1/lambda regardless of run length.
        lh = log_1mH if t >= tau_min else 0.0   # log(1 - H_t)
        for r in range(n_active + 1):
            if r + 1 > max_r:
                continue
            if r > 0 and log_R[r] <= log_prob_floor:
                continue
            log_R_new[r + 1] = log_R[r] + log_preds[r] + lh

        # Changepoint: all -> 0  (impossible before warmup)
        cp_terms = []
        if t >= tau_min:
            for r in range(n_active + 1):
                if r > 0 and log_R[r] <= log_prob_floor:
                    continue
                cp_terms.append(log_R[r] + log_preds[r] + log_H)
        if cp_terms:
            log_R_new[0] = logsumexp(cp_terms)

        # --- Step 3: normalize ---
        valid = log_R_new > log_prob_floor
        if valid.any():
            log_norm = logsumexp(log_R_new[valid])
            log_R_new -= log_norm

        changepoint_probs[t] = np.exp(np.clip(log_R_new[0], -50.0, 0.0))

        # --- Step 4: update sufficient statistics ---
        ns_new    = np.zeros(max_r + 1, dtype=int)
        means_new = np.zeros((max_r + 1, d))
        Ss_new    = np.zeros((max_r + 1, d, d))

        # r -> r+1: incorporate x into run r's stats
        for r in range(n_active + 1):
            if r + 1 > max_r:
                continue
            n_new, mean_new, S_new = _welford_update(ns[r], means[r], Ss[r], x)
            ns_new[r + 1]    = n_new
            means_new[r + 1] = mean_new
            Ss_new[r + 1]    = S_new
        # r=0: fresh stats — already zeros by construction

        log_R = log_R_new
        ns    = ns_new
        means = means_new
        Ss    = Ss_new

    return changepoint_probs


# ---------------------------------------------------------------------------
# Keypoint extraction
# ---------------------------------------------------------------------------

def _extract_bocpd_keypoints(changepoint_probs: np.ndarray,
                              config: dict, T: int) -> np.ndarray:
    """
    Peak detection on changepoint_probs -> subgoal indices.
    Uses Method 3 from the spec to avoid clusters of nearby detections.
    """
    peaks, _ = find_peaks(
        changepoint_probs,
        height=float(config['peak_min_height']),
        distance=int(config['peak_min_distance']),
    )
    # Enforce warmup
    peaks = peaks[peaks >= int(config['tau_min'])]
    return peaks.astype(int)


# ---------------------------------------------------------------------------
# Public API — drop-in replacement for compute_subgoal_gripper_pcd
# ---------------------------------------------------------------------------

def compute_bayesian_subgoal_gripper_pcd(
    gripper_pcd:  np.ndarray,   # (T, 4, 3)
    eef_qpos:     np.ndarray,   # (T, 2)  gripper finger joint positions
    actions:      np.ndarray,   # (T, D)  last dim is gripper action
    eef_pos:      np.ndarray,   # (T, 3)
    eef_quat:     np.ndarray,   # (T, 4)  xyzw (scipy convention)
    eef_vel_lin:  np.ndarray,   # (T, 3)
    config:       dict,
    return_switch_idxs: bool = False,
):
    """
    Compute goal_gripper_pcd using BOCPD changepoints OR-ed with gripper transitions.

    Args:
        gripper_pcd:    (T, 4, 3)
        eef_qpos:       (T, 2)   gripper finger positions
        actions:        (T, D)   last dim = gripper action
        eef_pos:        (T, 3)
        eef_quat:       (T, 4)   xyzw
        eef_vel_lin:    (T, 3)
        config:         dict from bocpd_config.yaml
        return_switch_idxs: if True, also return the index array

    Returns:
        expanded_goal_gripper_pcd: (T, 4, 3)
        switch_indices (optional): (K,) int array
    """
    T = gripper_pcd.shape[0]

    # BOCPD changepoints from kinematic features
    features = _compute_features(eef_pos, eef_quat, eef_vel_lin, config)
    features = _standardize(features)
    changepoint_probs = _bocpd(features, config)
    bocpd_idxs = _extract_bocpd_keypoints(changepoint_probs, config, T)

    # Gripper open/close transitions (same as subgoal_decomp.py)
    grip_idxs = gripper_switch_indices(eef_qpos, actions)

    # OR: merge, deduplicate, apply warmup
    warmup = int(config['tau_min'])
    all_idxs = np.unique(np.concatenate([bocpd_idxs, grip_idxs])).astype(int)
    all_idxs = all_idxs[(all_idxs >= warmup) & (all_idxs < T)]

    # Always end at last timestep
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
