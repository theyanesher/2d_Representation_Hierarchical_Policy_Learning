# import torch
import numpy as np

from manipulation.utils import rotation_transfer_6D_to_matrix, \
    rotation_transfer_matrix_to_6D,  \
    rotation_transfer_matrix_to_6D_batch, rotation_transfer_6D_to_matrix_batch_mino
# from mino_utils.math import rotation_transfer_6D_to_matrix_batch as mino_6d_to_mat_batch

def hybrid_relative_to_hybrid_delta_actions(relative_actions):
    """
    BxTxA -> BxTxA or TxA -> TxA. 
    Converts relative actions back to delta actions.
    """
    is_2d = relative_actions.ndim == 2
    if is_2d:
        relative_actions = relative_actions[np.newaxis, ...]
    
    B, T, D = relative_actions.shape
    rel_xyz = relative_actions[..., :3]
    rel_6d = relative_actions[..., 3:9]
    rel_grip = relative_actions[..., 9:10]

    delta_xyz = np.diff(rel_xyz, axis=1, prepend=0)
    delta_grip = np.diff(rel_grip, axis=1, prepend=0)
    delta_xyz[:, 0] = rel_xyz[:, 0] 
    delta_grip[:, 0] = rel_grip[:, 0]

    raw_mats = rotation_transfer_6D_to_matrix_batch_mino(rel_6d.reshape(-1, 6))
    rel_mats = raw_mats.reshape(B, T, 3, 3)
    delta_mats = np.zeros_like(rel_mats)
    delta_mats[:, 0] = rel_mats[:, 0]
    for t in range(1, T):
        prev_rel_T = rel_mats[:, t-1].transpose(0, 2, 1) 
        delta_mats[:, t] = prev_rel_T @ rel_mats[:, t]
    delta_6d = rotation_transfer_matrix_to_6D_batch(
        delta_mats.reshape(-1, 3, 3).transpose(0, 2, 1)
    ).reshape(B,T,6)
    delta_actions = np.concatenate([delta_xyz, delta_6d, delta_grip], axis=-1)
    return delta_actions[0] if is_2d else delta_actions

def hybrid_delta_to_hybrid_relative_actions(delta_actions):
    """BxTxA -> BxTxA or TxA -> TxA. Converts delta actions to relative actions."""
    is_2d = delta_actions.ndim == 2
    if is_2d:
        delta_actions = delta_actions[np.newaxis, ...]
    
    B, T, D = delta_actions.shape
    delta_xyz = delta_actions[..., :3]
    delta_6d = delta_actions[..., 3:9]
    delta_grip = delta_actions[..., 9:10]

    relative_xyz = np.cumsum(delta_xyz, axis=1)
    relative_grip = np.cumsum(delta_grip, axis=1)
    delta_rot_matrices = rotation_transfer_6D_to_matrix_batch_mino(delta_6d.reshape(-1, 6))
    delta_rot_matrices = delta_rot_matrices.reshape(B, T, 3, 3)

    relative_rots = np.zeros_like(delta_rot_matrices)
    relative_rots[:, 0] = delta_rot_matrices[:, 0]
    for t in range(1, T):
        relative_rots[:, t] = relative_rots[:, t-1] @ delta_rot_matrices[:, t]
    relative_6d = rotation_transfer_matrix_to_6D_batch(
        relative_rots.reshape(-1, 3, 3).transpose(0, 2, 1)
    ).reshape(B, T, 6)

    relative_actions = np.concatenate([relative_xyz, relative_6d, relative_grip], axis=-1)
    return relative_actions[0] if is_2d else relative_actions

def _get_R0_from_s0(s0):
    """Extract and convert initial orientation from 10D state (xyz + rot6d + gripper)."""
    s0_rot_6d = np.atleast_2d(s0)[..., 3:9].reshape(-1, 6)  # (B, 6)
    R_0 = rotation_transfer_6D_to_matrix_batch_mino(s0_rot_6d)
    return R_0

def hybrid_delta_to_delta_actions(hybrid_delta_actions, s0):
    """
    BxTxA, Bx10 -> BxTxA or TxA, (10,) -> TxA.
    Converts hybrid delta actions (world-frame Δxyz, body-frame Δrot) to
    pure EE-frame delta actions (EE-frame Δxyz, body-frame Δrot).
    s0: initial state in world frame, shape (10,) or (B, 10).
    """
    is_2d = hybrid_delta_actions.ndim == 2
    if is_2d:
        hybrid_delta_actions = hybrid_delta_actions[np.newaxis, ...]

    B, T, D = hybrid_delta_actions.shape
    delta_xyz_world = hybrid_delta_actions[..., :3]
    delta_6d = hybrid_delta_actions[..., 3:9]
    delta_grip = hybrid_delta_actions[..., 9:10]

    delta_rot_mats = rotation_transfer_6D_to_matrix_batch_mino(delta_6d.reshape(-1, 6))
    delta_rot_mats = delta_rot_mats.reshape(B, T, 3, 3)

    R_0 = _get_R0_from_s0(s0)  # (B, 3, 3)

    # R_abs[:, t] = world-frame orientation at the START of timestep t (before applying delta[t])
    R_abs = np.zeros((B, T, 3, 3))
    R_abs[:, 0] = R_0
    for t in range(1, T):
        R_abs[:, t] = R_abs[:, t-1] @ delta_rot_mats[:, t-1]

    # delta_xyz_ee[t] = R_abs[:, t]^T @ delta_xyz_world[t]
    delta_xyz_ee = np.einsum('btij,btj->bti', R_abs.transpose(0, 1, 3, 2), delta_xyz_world)

    delta_actions = np.concatenate([delta_xyz_ee, delta_6d, delta_grip], axis=-1)
    return delta_actions[0] if is_2d else delta_actions


def delta_to_hybrid_delta_actions(delta_actions, s0):
    """
    BxTxA, Bx10 -> BxTxA or TxA, (10,) -> TxA.
    Converts pure EE-frame delta actions (EE-frame Δxyz, body-frame Δrot) to
    hybrid delta actions (world-frame Δxyz, body-frame Δrot).
    s0: initial state in world frame, shape (10,) or (B, 10).
    """
    is_2d = delta_actions.ndim == 2
    if is_2d:
        delta_actions = delta_actions[np.newaxis, ...]

    B, T, D = delta_actions.shape
    delta_xyz_ee = delta_actions[..., :3]
    delta_6d = delta_actions[..., 3:9]
    delta_grip = delta_actions[..., 9:10]

    delta_rot_mats = rotation_transfer_6D_to_matrix_batch_mino(delta_6d.reshape(-1, 6))
    delta_rot_mats = delta_rot_mats.reshape(B, T, 3, 3)

    R_0 = _get_R0_from_s0(s0)  # (B, 3, 3)

    R_abs = np.zeros((B, T, 3, 3))
    R_abs[:, 0] = R_0
    for t in range(1, T):
        R_abs[:, t] = R_abs[:, t-1] @ delta_rot_mats[:, t-1]

    # delta_xyz_world[t] = R_abs[:, t] @ delta_xyz_ee[t]
    delta_xyz_world = np.einsum('btij,btj->bti', R_abs, delta_xyz_ee)

    hybrid_delta_actions = np.concatenate([delta_xyz_world, delta_6d, delta_grip], axis=-1)
    return hybrid_delta_actions[0] if is_2d else hybrid_delta_actions


def delta_to_relative_actions(delta_actions):
    """
    BxTxA -> BxTxA or TxA -> TxA.
    Converts EE-frame delta actions to EE-frame relative actions (cumulative).
    """
    is_2d = delta_actions.ndim == 2
    if is_2d:
        delta_actions = delta_actions[np.newaxis, ...]

    B, T, D = delta_actions.shape
    delta_xyz_ee = delta_actions[..., :3]
    delta_6d = delta_actions[..., 3:9]
    delta_grip = delta_actions[..., 9:10]

    relative_grip = np.cumsum(delta_grip, axis=1)

    delta_rot_mats = rotation_transfer_6D_to_matrix_batch_mino(delta_6d.reshape(-1, 6))
    delta_rot_mats = delta_rot_mats.reshape(B, T, 3, 3)

    relative_rots = np.zeros_like(delta_rot_mats)
    relative_rots[:, 0] = delta_rot_mats[:, 0]
    for t in range(1, T):
        relative_rots[:, t] = relative_rots[:, t-1] @ delta_rot_mats[:, t]

    # Each delta_xyz_ee[t] is in the EE frame at timestep t, so we must rotate
    # each delta into the t=0 EE reference frame before accumulating.
    relative_xyz = np.zeros_like(delta_xyz_ee)
    relative_xyz[:, 0] = delta_xyz_ee[:, 0]
    for t in range(1, T):
        relative_xyz[:, t] = relative_xyz[:, t-1] + np.einsum('bij,bj->bi', relative_rots[:, t-1], delta_xyz_ee[:, t])

    relative_6d = rotation_transfer_matrix_to_6D_batch(
        relative_rots.reshape(-1, 3, 3).transpose(0, 2, 1)
    ).reshape(B, T, 6)

    relative_actions = np.concatenate([relative_xyz, relative_6d, relative_grip], axis=-1)
    return relative_actions[0] if is_2d else relative_actions


def relative_to_delta_actions(relative_actions):
    """
    BxTxA -> BxTxA or TxA -> TxA.
    Converts EE-frame relative actions (cumulative) back to EE-frame delta actions.
    """
    is_2d = relative_actions.ndim == 2
    if is_2d:
        relative_actions = relative_actions[np.newaxis, ...]

    B, T, D = relative_actions.shape
    rel_xyz = relative_actions[..., :3]
    rel_6d = relative_actions[..., 3:9]
    rel_grip = relative_actions[..., 9:10]

    delta_grip = np.diff(rel_grip, axis=1, prepend=0)
    delta_grip[:, 0] = rel_grip[:, 0]

    rel_mats = rotation_transfer_6D_to_matrix_batch_mino(rel_6d.reshape(-1, 6))
    rel_mats = rel_mats.reshape(B, T, 3, 3)

    delta_mats = np.zeros_like(rel_mats)
    delta_mats[:, 0] = rel_mats[:, 0]
    for t in range(1, T):
        delta_mats[:, t] = rel_mats[:, t-1].transpose(0, 2, 1) @ rel_mats[:, t]

    # Inverse of the forward accumulation: differences are in the t=0 reference frame,
    # so rotate each back into the EE frame at timestep t via rel_mats[:, t-1]^T.
    delta_xyz = np.zeros_like(rel_xyz)
    delta_xyz[:, 0] = rel_xyz[:, 0]
    for t in range(1, T):
        delta_xyz[:, t] = np.einsum('bij,bj->bi', rel_mats[:, t-1].transpose(0, 2, 1), rel_xyz[:, t] - rel_xyz[:, t-1])

    delta_6d = rotation_transfer_matrix_to_6D_batch(
        delta_mats.reshape(-1, 3, 3).transpose(0, 2, 1)
    ).reshape(B, T, 6)

    delta_actions = np.concatenate([delta_xyz, delta_6d, delta_grip], axis=-1)
    return delta_actions[0] if is_2d else delta_actions


if __name__ == "__main__":
    import h5py
    from matplotlib import pyplot as plt
    import numpy as np

    f = h5py.File('data/rgb/41510/2025-10-30-21-05-53.h5')

    states = f['obs/state'][:]
    actions = f['action/hybrid'][:]
    s0 = np.asarray(states[0:1], dtype=np.float32)
    delta_actions = np.asarray(actions[:16], dtype=np.float32)
    
    relative_actions = hybrid_delta_to_hybrid_relative_actions(delta_actions)
    delta_actions_recovered = hybrid_relative_to_hybrid_delta_actions(relative_actions)
    assert np.allclose(delta_actions, delta_actions_recovered)
   
    def plot_actions(original_delta, relative, recovered_delta):
        T = original_delta.shape[0]
        time_steps = np.arange(T)
        labels = [
            'X', 'Y', 'Z', 
            'Rot_1', 'Rot_2', 'Rot_3', 'Rot_4', 'Rot_5', 'Rot_6', 
            'Grip'
        ]
        fig, axes = plt.subplots(4, 3, figsize=(18, 12), constrained_layout=True)
        fig.suptitle('Action Conversion Verification: Original vs. Relative vs. Recovered', fontsize=16)
        axes = axes.flatten()
        for i in range(10):
            ax = axes[i]
            ax_rel = ax.twinx()
            ln1 = ax.plot(time_steps, original_delta[:, i], 'g-', linewidth=3, label='Orig Delta', alpha=0.6)
            ln2 = ax.plot(time_steps, recovered_delta[:, i], 'r--', linewidth=2, label='Recov Delta')
            ln3 = ax_rel.plot(time_steps, relative[:, i], 'b:', label='Relative', alpha=0.8)
            ax.set_title(labels[i])
            ax.set_xlabel('Step')
            if i == 0: # Add legend once
                lines = ln1 + ln2 + ln3
                labs = [l.get_label() for l in lines]
                ax.legend(lines, labs, loc='upper left')
        for j in range(10, 12):
            axes[j].axis('off')
        plt.show()

    # plot_actions(delta_actions, relative_actions, delta_actions_recovered)

    s16 = np.asarray(states[16:17], dtype=np.float32)
    # Verify state transitions
    s16_from_s0 = s0.copy()
    s16_from_s0[:, :3] = relative_actions[-1, :3] + s0[:, :3]
    rel_rot_mat = rotation_transfer_6D_to_matrix(
        relative_actions[-1, 3:9]
    )
    s0_rot_mat = rotation_transfer_6D_to_matrix(
        s0[0, 3:9]
    )
    s16_from_s0_rot_mat = s0_rot_mat @ rel_rot_mat
    s16_from_s0_6d = rotation_transfer_matrix_to_6D(s16_from_s0_rot_mat)
    s16_from_s0[:, 3:9] = s16_from_s0_6d
    s16_gt = s16

    assert np.allclose(s16_from_s0, s16_gt)
    print("State transition verified successfully.")

    true_delta = hybrid_delta_to_delta_actions(delta_actions, s0)
    recovered_hybrid_delta = delta_to_hybrid_delta_actions(true_delta, s0)
    assert np.allclose(delta_actions, recovered_hybrid_delta)
    print("Hybrid <-> Pure delta conversion verified successfully.")

    true_relative = delta_to_relative_actions(true_delta)
    recovered_delta_from_relative = relative_to_delta_actions(true_relative)
    assert np.allclose(true_delta, recovered_delta_from_relative)
    print("Delta <-> Relative conversion verified successfully.")

    a0 = true_delta[0]
    a0_t = a0[:3]
    a0_r = rotation_transfer_6D_to_matrix(a0[3:9])
    a0_g = a0[9:10]

    s0_t = s0[0, :3]
    s0_r = rotation_transfer_6D_to_matrix(s0[0, 3:9])
    s0_g = s0[0, 9:10]

    s1 = np.asarray(states[1:2], dtype=np.float32)
    s1_t = s1[0, :3]
    s1_r = rotation_transfer_6D_to_matrix(s1[0, 3:9])
    s1_g = s1[0, 9:10]

    pred_s1_t = s0_t + s0_r @ a0_t
    pred_s1_r = s0_r @ a0_r
    pred_s1_g = s0_g + a0_g

    print("Predicted s1_t:", pred_s1_t)
    print("Actual s1_t:", s1_t)
    print("Predicted s1_r:", pred_s1_r)
    print("Actual s1_r:", s1_r)
    print("Predicted s1_g:", pred_s1_g)
    print("Actual s1_g:", s1_g)
