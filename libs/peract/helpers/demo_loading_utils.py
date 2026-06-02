import os
import logging
from typing import List

import numpy as np
from rdp import rdp
from rlbench.demo import Demo

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

rdp_keypoint_counts = []
os.makedirs("stats", exist_ok=True)

def visualize_rdp_trajectory(ee_traj, keypoints, save_path="rdp_vis.png"):
    ee_traj = np.asarray(ee_traj)
    keypoints = np.asarray(keypoints)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 1. Plot full trajectory (blue line)
    ax.plot(
        ee_traj[:, 0],
        ee_traj[:, 1],
        ee_traj[:, 2],
        color='blue',
        linewidth=2,
        label='Trajectory'
    )

    # 2. Plot keypoints (orange dots)
    kp = ee_traj[keypoints]
    ax.scatter(
        kp[:, 0],
        kp[:, 1],
        kp[:, 2],
        color='orange',
        s=50,
        label='RDP Keypoints'
    )

    # 3. Start point (green)
    ax.scatter(
        ee_traj[0, 0],
        ee_traj[0, 1],
        ee_traj[0, 2],
        color='green',
        s=100,
        label='Start'
    )

    # 4. End point (red)
    ax.scatter(
        ee_traj[-1, 0],
        ee_traj[-1, 1],
        ee_traj[-1, 2],
        color='red',
        s=100,
        label='End'
    )

    # Labels for interpretability
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.legend()
    plt.title("RDP Keypoint Selection on EE Trajectory")

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # Important: free memory (especially inside loops)
    plt.close(fig)


'''
def visualize_rdp_trajectory(
    ee_traj,
    rdp_keypoints,
    gripper_keypoints,
    save_path="rdp_vis.png"
):
    ee_traj = np.asarray(ee_traj)

    rdp_keypoints = np.asarray(rdp_keypoints)
    gripper_keypoints = np.asarray(gripper_keypoints)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # ============================================================
    # 1. Full trajectory
    # ============================================================
    ax.plot(
        ee_traj[:, 0],
        ee_traj[:, 1],
        ee_traj[:, 2],
        color='blue',
        linewidth=2,
        label='Trajectory'
    )

    # ============================================================
    # 2. RDP keypoints (orange)
    # ============================================================
    if len(rdp_keypoints) > 0:
        kp = ee_traj[rdp_keypoints]

        ax.scatter(
            kp[:, 0],
            kp[:, 1],
            kp[:, 2],
            color='orange',
            s=50,
            label='RDP Keypoints'
        )

    # ============================================================
    # 3. Gripper-only keypoints (red)
    #    Only visualize if NOT already in RDP
    # ============================================================
    extra_gripper_keypoints = [
        kp for kp in gripper_keypoints
        if kp not in set(rdp_keypoints.tolist())
    ]

    if len(extra_gripper_keypoints) > 0:
        gkp = ee_traj[extra_gripper_keypoints]

        ax.scatter(
            gkp[:, 0],
            gkp[:, 1],
            gkp[:, 2],
            color='red',
            s=80,
            label='Gripper Transition Keypoints'
        )

    # ============================================================
    # 4. Start point (green)
    # ============================================================
    ax.scatter(
        ee_traj[0, 0],
        ee_traj[0, 1],
        ee_traj[0, 2],
        color='green',
        s=120,
        label='Start'
    )

    # ============================================================
    # 5. End point (black)
    # ============================================================
    ax.scatter(
        ee_traj[-1, 0],
        ee_traj[-1, 1],
        ee_traj[-1, 2],
        color='black',
        s=120,
        label='End'
    )

    # ============================================================
    # Labels
    # ============================================================
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.legend()
    plt.title("RDP + Gripper Transition Keypoints")

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # Free memory
    plt.close(fig)
'''

def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = (
            i < (len(demo) - 2) and
            (obs.gripper_open == demo[i + 1].gripper_open and
             obs.gripper_open == demo[i - 1].gripper_open and
             demo[i - 2].gripper_open == demo[i - 1].gripper_open))
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    return stopped


def keypoint_discovery(demo: Demo, 
                       stopping_delta=0.1,
                       method='heuristic',
                       episode_idx=0) -> List[int]:
    episode_keypoints = []
    if method == 'heuristic':
        prev_gripper_open = demo[0].gripper_open
        stopped_buffer = 0
        for i, obs in enumerate(demo):
            stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
            stopped_buffer = 4 if stopped else stopped_buffer - 1
            # If change in gripper, or end of episode.
            last = i == (len(demo) - 1)
            if i != 0 and (obs.gripper_open != prev_gripper_open or
                           last or stopped):
                episode_keypoints.append(i)
            prev_gripper_open = obs.gripper_open
        if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
                episode_keypoints[-2]:
            episode_keypoints.pop(-2)
        logging.debug('Found %d keypoints.' % len(episode_keypoints),
                      episode_keypoints)
        return episode_keypoints

    elif method == 'random':
        # Randomly select keypoints.
        episode_keypoints = np.random.choice(
            range(len(demo)),
            size=20,
            replace=False)
        episode_keypoints.sort()
        return episode_keypoints

    elif method == 'fixed_interval':
        # Fixed interval.
        episode_keypoints = []
        segment_length = len(demo) // 20
        for i in range(0, len(demo), segment_length):
            episode_keypoints.append(i)
        return episode_keypoints

    elif method == "rdp": 
        epsilon = 0.02
        ee_traj = np.array([obs.gripper_pose[:3] for obs in demo])
        mask = rdp(ee_traj, epsilon=epsilon, return_mask=True)
        episode_keypoints = np.where(mask)[0].tolist()
        episode_keypoints.sort()
        
        rdp_keypoint_counts.append(len(episode_keypoints))
        print("number of keypoints: ", len(episode_keypoints))
        visualize_rdp_trajectory(ee_traj, episode_keypoints, save_path=f"visualizations/demo_{episode_idx}.png")
        
        with open("stats/rdp_keypoint_counts.csv", "a") as f:
            f.write(f"{episode_idx},{len(episode_keypoints)}\n")

        return episode_keypoints

    elif method == "rdp_gripper":
        epsilon = 0.02
        snap_window = 5   # frames; adjust this

        # 1. RDP keypoints from end-effector geometry
        ee_traj = np.array([obs.gripper_pose[:3] for obs in demo])
        mask = rdp(ee_traj, epsilon=epsilon, return_mask=True)
        rdp_keypoints = np.where(mask)[0].tolist()

        # 2. Detect gripper open/close transition keypoints
        gripper_keypoints = []
        prev_gripper_open = demo[0].gripper_open

        for i, obs in enumerate(demo):
            if i != 0 and obs.gripper_open != prev_gripper_open:
                gripper_keypoints.append(i)
            prev_gripper_open = obs.gripper_open

        # 3. Snap nearby RDP keypoints to gripper transition keypoints
        snapped_rdp_keypoints = rdp_keypoints.copy()

        for gkp in gripper_keypoints:
            if len(snapped_rdp_keypoints) == 0:
                snapped_rdp_keypoints.append(gkp)
                continue

            distances = [abs(rkp - gkp) for rkp in snapped_rdp_keypoints]
            closest_idx = int(np.argmin(distances))
            closest_dist = distances[closest_idx]

            if closest_dist <= snap_window:
                old_kp = snapped_rdp_keypoints[closest_idx]

                print(
                    f"[RDP SNAP] demo={episode_idx} "
                    f"replacing RDP keypoint {old_kp} "
                    f"with gripper transition {gkp} "
                    f"(distance={closest_dist})"
                )

                snapped_rdp_keypoints[closest_idx] = gkp
            else:
                # If no nearby RDP keypoint exists, keep gripper transition separately
                snapped_rdp_keypoints.append(gkp)

        # 4. Always include final frame
        last_idx = len(demo) - 1

        # 5. Combine snapped RDP keypoints + final frame
        episode_keypoints = sorted(set(snapped_rdp_keypoints + [last_idx]))

        visualize_rdp_trajectory(ee_traj, episode_keypoints, save_path=f"visualizations/demo_{episode_idx}.png")

        with open("stats/rdp_keypoint_counts.csv", "a") as f:
            f.write(f"{episode_idx},{len(episode_keypoints)}\n")

        return episode_keypoints

    else:
        raise NotImplementedError


# find minimum difference between any two elements in list
def find_minimum_difference(lst):
    minimum = lst[-1]
    for i in range(1, len(lst)):
        if lst[i] - lst[i - 1] < minimum:
            minimum = lst[i] - lst[i - 1]
    return minimum