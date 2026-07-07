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