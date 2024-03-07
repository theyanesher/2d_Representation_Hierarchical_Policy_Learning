import numpy as np
import pybullet_ompl.pb_ompl as pb_ompl
import pybullet as p
import copy
import os, pickle

PLANNER = "BITstar" # BITstar

def motion_planning(env, target_pos, target_orientation, planner=None, obstacles=[], allow_collision_links=[], save_path=None, target_joint_angle=None):
    if planner is None:
        planner = PLANNER
    current_joint_angles = copy.deepcopy(env.robot.get_joint_angles(indices=env.robot.right_arm_joint_indices))
    ompl_robot = pb_ompl.PbOMPLRobot(env.robot.body, control_joint_idx=env.robot.right_arm_joint_indices)
    ompl_robot.set_state(current_joint_angles)

    allow_collision_robot_link_pairs = []
    if env.robot_name == "sawyer":
        allow_collision_robot_link_pairs.append((5, 8))
    if env.robot_name == 'fetch':
        allow_collision_robot_link_pairs.append((3, 19))
    pb_ompl_interface = pb_ompl.PbOMPL(ompl_robot, obstacles, allow_collision_links, 
                                       allow_collision_robot_link_pairs=allow_collision_robot_link_pairs)
    pb_ompl_interface.set_planner(planner)

    # first need to compute a collision-free IK solution
    ik_lower_limits = env.robot.ik_lower_limits 
    ik_upper_limits = env.robot.ik_upper_limits 
    ik_joint_ranges = ik_upper_limits - ik_lower_limits

    if target_joint_angle is None:
        it = 0
        while True:
            if it % 10 == 0:
                print("sampling target ik it: ", it)

            ik_rest_poses = np.random.uniform(ik_lower_limits, ik_upper_limits)
            p.addUserDebugPoints([target_pos], [[1, 0, 0]], 25, 0)
        
            ik_start_pose = np.random.uniform(ik_lower_limits, ik_upper_limits)
            ompl_robot.set_state(ik_start_pose[env.robot.right_arm_joint_indices])

            target_joint_angle = np.array(p.calculateInverseKinematics(
                env.robot.body, env.robot.right_end_effector, 
                targetPosition=target_pos, targetOrientation=target_orientation, 
                # lowerLimits=ik_lower_limits.tolist(), upperLimits=ik_upper_limits.tolist(), jointRanges=ik_joint_ranges.tolist(), 
                # restPoses=ik_rest_poses.tolist(), 
                maxNumIterations=10000,
                residualThreshold=1e-4
            ))
            
            ompl_robot.set_state(target_joint_angle)
            
            eef_pos, eef_orient = env.robot.get_pos_orient(env.robot.right_end_effector)
            ik_error = np.linalg.norm(eef_pos - target_pos)
            print("within lower limit: ", np.all(target_joint_angle[env.robot.right_arm_joint_indices] >= ik_lower_limits[env.robot.right_arm_joint_indices]))
            print("within upper limit: ", np.all(target_joint_angle[env.robot.right_arm_joint_indices] <= ik_upper_limits[env.robot.right_arm_joint_indices]))
            print("is state valid: ", pb_ompl_interface.is_state_valid(target_joint_angle))
            print("ik_error: ", ik_error)
            if np.all(target_joint_angle[env.robot.right_arm_joint_indices] >= ik_lower_limits[env.robot.right_arm_joint_indices]) \
                    and np.all(target_joint_angle[env.robot.right_arm_joint_indices] <= ik_upper_limits[env.robot.right_arm_joint_indices]) \
                    and pb_ompl_interface.is_state_valid(target_joint_angle) \
                    and ik_error < 0.1:
                break
            
            it += 1

            if it > 1000:
                ompl_robot.set_state(current_joint_angles)
                print("failed to find a valid IK solution")
                return False, None
        
    
        # then plan using ompl
        target_joint_angle = target_joint_angle[env.robot.right_arm_joint_indices]    
        assert len(target_joint_angle) == ompl_robot.num_dim
    assert pb_ompl_interface.is_state_valid(target_joint_angle)

    ompl_robot.set_state(current_joint_angles)
    res, path = pb_ompl_interface.plan(target_joint_angle)
    ompl_robot.set_state(current_joint_angles)
    
    if not res:
        print("motion planning failed to find a path")
    else:
        if save_path is not None:
            with open(os.path.join(save_path, "target_joint_angle.pkl"), "wb") as f:
                pickle.dump(target_joint_angle, f)
            with open(os.path.join(save_path, "current_joint_angle.pkl"), "wb") as f:
                pickle.dump(current_joint_angles, f)

    return res, path

def motion_planning_joint_angle(env, target_joint_angle, planner="BITstar", obstacles=[], allow_collision_links=[]):
    current_joint_angles = copy.deepcopy(env.robot.get_joint_angles(indices=env.robot.right_arm_joint_indices))
    ompl_robot = pb_ompl.PbOMPLRobot(env.robot.body, control_joint_idx=env.robot.right_arm_joint_indices)
    # ompl_robot = pb_ompl.PbOMPLRobot(env.robot.body)
    ompl_robot.set_state(current_joint_angles)
    pb_ompl_interface = pb_ompl.PbOMPL(ompl_robot, obstacles, allow_collision_links)
    pb_ompl_interface.set_planner(planner)
        
    #  plan using ompl
    assert len(target_joint_angle) == ompl_robot.num_dim
    for idx in range(ompl_robot.num_dim):
        print("joint: ", idx, " lower limit: ", ompl_robot.joint_bounds[idx][0], " upper limit: ", ompl_robot.joint_bounds[idx][1], " target: ", target_joint_angle[idx])
        assert (ompl_robot.joint_bounds[idx][0] <= target_joint_angle[idx]) & (target_joint_angle[idx] <= ompl_robot.joint_bounds[idx][1])

    res, path = pb_ompl_interface.plan(target_joint_angle)
    
    if not res:
        print("motion planning failed to find a path")

    return res, path