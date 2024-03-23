import pybullet as p
import os
import numpy as np
import open3d as o3d
from manipulation.motion_planning_utils import motion_planning
from manipulation.grasping_utils import get_pc_and_normal, align_gripper_z_with_normal, align_gripper_x_with_normal
from manipulation.gpt_reward_api import (
    get_link_pc, get_bounding_box, get_link_id_from_name, get_handle_pos, get_link_pose,
)
from manipulation.utils import save_env, load_env
import scipy
import time
import copy

MOTION_PLANNING_TRY_TIMES=100

def get_save_path(simulator):
    state_save_path = os.path.join(simulator.primitive_save_path, "states")
    if not os.path.exists(state_save_path):
        os.makedirs(state_save_path)
    return simulator.primitive_save_path


def release_grasp(simulator):
    # simulator.deactivate_suction()
    save_path = get_save_path(simulator)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    rgbs = []
    states = []
    for t in range(20):
        p.stepSimulation()
        rgbs.append(simulator.render())
        state_save_path = os.path.join(save_path, "states", "state_{}.pkl".format(t))
        save_env(simulator, state_save_path)
        states.append(state_save_path)

    return rgbs, states

def grasp_object(simulator, object_name):
    ori_state = save_env(simulator, None)
    p.stepSimulation()
    object_name = object_name.lower()
    save_path = get_save_path(simulator)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # if the target object is already grasped.  
    points = p.getContactPoints(bodyA=simulator.robot.body, linkIndexA=simulator.suction_id, physicsClientId=simulator.id)
    if points:
        for point in points:
            obj_id, contact_link = point[2], point[4]
            if obj_id == simulator.urdf_ids[object_name]:
                # simulator.activate_suction()
                rgbs = []
                states = []
                for t in range(10):
                    p.stepSimulation()
                    rgbs.append(simulator.render())
                    state_save_path = os.path.join(save_path, "states", "state_{}.pkl".format(t))
                    save_env(simulator, state_save_path)
                    states.append(state_save_path)
                return rgbs, states

    rgbs, states = approach_object(simulator, object_name)
    base_t = len(rgbs)
    if base_t > 1:
        for t in range(10):
            # simulator.activate_suction()
            p.stepSimulation()
            rgbs.append(simulator.render())
            state_save_path = os.path.join(save_path, "states", "state_{}.pkl".format(t + base_t))
            save_env(simulator, state_save_path)
            states.append(state_save_path)
    else:
        # directy reset the state
        load_env(simulator, state=ori_state)

    return rgbs, states

def grasp_object_link(simulator, object_name, link_name):
    return approach_object_link(simulator, object_name, link_name)

    # NOTE: old code
    # ori_state = save_env(simulator, None)
    # p.stepSimulation()
    # object_name = object_name.lower()
    # save_path = get_save_path(simulator)
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    
    # # if the target object link is already grasped.  
    # points = p.getContactPoints(bodyA=simulator.robot.body, linkIndexA=simulator.suction_id, physicsClientId=simulator.id)
    # if points:
    #     for point in points:
    #         obj_id, contact_link = point[2], point[4]
    #         if obj_id == simulator.urdf_ids[object_name] and contact_link == get_link_id_from_name(simulator, object_name, link_name):
    #             # simulator.activate_suction()
    #             rgbs = []
    #             states = []
    #             for t in range(10):
    #                 p.stepSimulation()
    #                 rgbs.append(simulator.render())
    #                 state_save_path = os.path.join(save_path, "states", "state_{}.pkl".format(t))
    #                 save_env(simulator, state_save_path)
    #                 states.append(state_save_path)
    #             return rgbs, states


    # rgbs, states = approach_object_link(simulator, object_name, link_name)
    # base_t = len(rgbs)
    # if base_t > 1:
    #     # simulator.activate_suction()
    #     for t in range(10):
    #         p.stepSimulation()
    #         rgbs.append(simulator.render())
    #         state_save_path = os.path.join(save_path, "states", "state_{}.pkl".format(t + base_t))
    #         save_env(simulator, state_save_path)
    #         states.append(state_save_path)
    # else:
    #     # directy reset the state
    #     load_env(simulator, state=ori_state)

    # return rgbs, states

def approach_object(simulator, object_name, dynamics=False):
    save_path = get_save_path(simulator)
    ori_state = save_env(simulator, None)
    # simulator.deactivate_suction()
    release_rgbs = []
    release_states = []
    release_steps = 20
    for t in range(release_steps):
        p.stepSimulation()
        rgb = simulator.render()
        release_rgbs.append(rgb)
        state_save_path = os.path.join(save_path, "states", "state_{}.pkl".format(t))
        save_env(simulator, state_save_path)
        release_states.append(state_save_path)

    object_name = object_name.lower()
    it = 0
    object_name = object_name.lower()
    object_pc, object_normal = get_pc_and_normal(simulator, object_name)
    low, high = get_bounding_box(simulator, object_name)
    com = (low + high) / 2
    current_joint_angles = simulator.robot.get_joint_angles(indices=simulator.robot.right_arm_joint_indices)
    

    while True:
        random_point = object_pc[np.random.randint(0, object_pc.shape[0])]
        random_normal = object_normal[np.random.randint(0, object_normal.shape[0])]

        ### adjust the normal such that it points outwards the object.
        line = com - random_point
        if np.dot(line, random_normal) > 0:
            random_normal = -random_normal
            
        for normal in [random_normal, -random_normal]:
            simulator.robot.set_joint_angles(simulator.robot.right_arm_joint_indices, current_joint_angles)

            target_pos = random_point
            real_target_pos = target_pos + normal * 0
            if simulator.robot_name in ["panda", "sawyer"]:
                target_orientation = align_gripper_z_with_normal(-normal).as_quat()
                mp_target_pos = target_pos + normal * 0.03
            elif simulator.robot_name in ['ur5', 'fetch']:
                target_orientation = align_gripper_x_with_normal(-normal).as_quat()
                if simulator.robot_name == 'ur5':
                    mp_target_pos = target_pos + normal * 0.07
                elif simulator.robot_name == 'fetch':
                    mp_target_pos = target_pos + normal * 0.07

            all_objects = list(simulator.urdf_ids.keys())
            all_objects.remove("robot")
            obstacles = [simulator.urdf_ids[x] for x in all_objects]
            allow_collision_links = []
            res, path = motion_planning(simulator, mp_target_pos, target_orientation, obstacles=obstacles, allow_collision_links=allow_collision_links)

            if res:
                rgbs = release_rgbs
                intermediate_states = release_states
                for idx, q in enumerate(path):
                    if not dynamics:
                        simulator.robot.set_joint_angles(simulator.robot.right_arm_joint_indices, q)
                        p.stepSimulation()
                    else:
                        for _ in range(3):
                            simulator.robot.control(simulator.robot.right_arm_joint_indices, q, simulator.robot.motor_gains, forces=5 * 240.)
                            p.stepSimulation()

                    rgb = simulator.render()
                    rgbs.append(rgb)
                    save_state_path = os.path.join(save_path,  "states", "state_{}.pkl".format(idx + release_steps))
                    save_env(simulator, save_state_path)
                    intermediate_states.append(save_state_path)

                base_idx = len(intermediate_states)
                for t in range(20):
                    ik_indices = [_ for _ in range(len(simulator.robot.right_arm_joint_indices))]
                    ik_joints = simulator.robot.ik(simulator.robot.right_end_effector, 
                                                    real_target_pos, target_orientation, 
                                                    ik_indices=ik_indices)
                    p.setJointMotorControlArray(simulator.robot.body, jointIndices=simulator.robot.right_arm_joint_indices, 
                                                controlMode=p.POSITION_CONTROL, targetPositions=ik_joints,
                                                forces=[5*240] * len(simulator.robot.right_arm_joint_indices), physicsClientId=simulator.id)
                    p.stepSimulation()
                    rgb = simulator.render()
                    rgbs.append(rgb)
                    save_state_path = os.path.join(save_path, "states" , "state_{}.pkl".format(base_idx + t))
                    save_env(simulator, save_state_path)
                    intermediate_states.append(save_state_path)

                    # TODO: check if there is already a collision. if so, break.
                    collision = False
                    points = p.getContactPoints(bodyA=simulator.robot.body, linkIndexA=simulator.suction_id, physicsClientId=simulator.id)
                    if points:
                        # Handle contact between suction with a rigid object.
                        for point in points:
                            obj_id, contact_link, contact_position_on_obj = point[2], point[4], point[6]
                            
                            if obj_id == simulator.urdf_ids['plane'] or obj_id == simulator.robot.body:
                                pass
                            else:
                                collision = True
                                break
                    if collision:
                        break

                return rgbs, intermediate_states
        
            it += 1
            if it > MOTION_PLANNING_TRY_TIMES:
                print("failed to execute the primitive")
                load_env(simulator, state=ori_state)
                save_env(simulator, os.path.join(save_path,  "state_{}.pkl".format(0)))
                rgbs = [simulator.render()]
                state_files = [os.path.join(save_path,  "state_{}.pkl".format(0))]
                return rgbs, state_files

def approach_object_link(simulator, object_name, link_name, dynamics=False, grasp_handle=True, 
                         execute_opening_primitive=True):
    save_path = get_save_path(simulator)
    ori_simulator_state = save_env(simulator, None)

    it = 0
    object_name = object_name.lower()
    link_pc = get_link_pc(simulator, object_name, link_name) 
    object_pc = link_pc
    pcd = o3d.geometry.PointCloud() 
    pcd.points = o3d.utility.Vector3dVector(object_pc)
    pcd.estimate_normals()
    object_normal = np.asarray(pcd.normals)

    handle_grasp_scores = []
    env_states = []
    rgb_images = []
    if grasp_handle is not None:
        all_handle_pos, handle_joint_id = get_handle_pos(simulator, object_name, return_median=False)
        median_point = np.median(all_handle_pos, axis=0, keepdims=True)
        pc_to_handle_distance = scipy.spatial.distance.cdist(object_pc, all_handle_pos).min(axis=1)
        threshold = 0.01
        handle_pc = object_pc[pc_to_handle_distance < threshold]
        debug_idx_1 = p.addUserDebugPoints(handle_pc, [[0, 1, 0] for _ in range(len(handle_pc))], 10, 0)
        import pdb; pdb.set_trace()
        pc_to_median_distance = np.linalg.norm(handle_pc - median_point, axis=1)
        sorted_idx = np.argsort(pc_to_median_distance)
        available_pc = [1 for _ in range(len(handle_pc))]
        
    for it in range(MOTION_PLANNING_TRY_TIMES):
        # import pdb; pdb.set_trace()
        
        object_name = object_name.lower()
        
        num_working_configs = np.sum(np.array(handle_grasp_scores) > 0)
        if num_working_configs > 5:
            break
          
        if grasp_handle is not None:
            # if we have tried all the handle points to grasp, break
            if np.sum(available_pc) == 0:
                break
            # if we have already tried to grasp the handle points, continue
            if not available_pc[sorted_idx[it]]:
                continue

            target_pos = handle_pc[sorted_idx[it]]
            
            # we omit all other points that are very close to the current grasping points to accelerate the search
            threshold = 0.01
            object_pc_within_target_pos = np.linalg.norm(handle_pc - target_pos.reshape(1, 3), axis=1) < threshold
            available_pc = np.logical_and(available_pc, ~object_pc_within_target_pos)
            
        else:
            target_pos = link_pc[np.random.randint(0, link_pc.shape[0])]
        nearest_point_idx = np.argmin(np.linalg.norm(object_pc - target_pos.reshape(1, 3), axis=1))
        align_normal = object_normal[nearest_point_idx]
        
        ### adjust the normal such that it points outwards the object.
        low, high = get_bounding_box(simulator, object_name)
        com = (low + high) / 2
        line = com - target_pos
        if np.dot(line, align_normal) > 0:
            align_normal = -align_normal

        for normal in [align_normal, -align_normal]:
            intermediate_states = []
            simulator.reset(ori_simulator_state)

            real_target_pos = target_pos + normal * -0.02
            debug_id = p.addUserDebugLine(target_pos, target_pos + normal, [1, 0, 0], 5)
            # TODO: if there is a handle, might want to align the finger direction with the handle horizontal direction
            if grasp_handle is not None:
                handle_orientation = get_handle_orient(handle_pc)
                horizontal_grasp = True if handle_orientation == 'vertical' else False
                target_orientation = align_gripper_z_with_normal(-normal, horizontal=horizontal_grasp).as_quat()
            else:
                target_orientation = align_gripper_z_with_normal(-normal).as_quat()
            mp_target_pos = target_pos + normal * 0.04

            all_objects = list(simulator.urdf_ids.keys())
            all_objects.remove("robot")
            obstacles = [simulator.urdf_ids[x] for x in all_objects]
            allow_collision_links = []
            res, path = motion_planning(
                simulator, mp_target_pos, target_orientation, obstacles=obstacles, allow_collision_links=allow_collision_links, save_path=save_path)
                
            p.removeUserDebugItem(debug_id)

            if res:
                with open(os.path.join(save_path, "motion_planning_target.pkl"), "wb") as f:
                    import pickle
                    pickle.dump([mp_target_pos, target_orientation], f) 
                
                rgbs = []
                for idx, q in enumerate(path):
                    if not dynamics:
                        simulator.robot.set_joint_angles(simulator.robot.right_arm_joint_indices, q)

                    else:
                        for _ in range(3):
                            simulator.robot.control(simulator.robot.right_arm_joint_indices, q)
                            p.stepSimulation()

                    rgb = simulator.render()
                    rgbs.append(rgb)
                    state = save_env(simulator)
                    intermediate_states.append(state)
                
                
                # first just open the gripper
                steps = 20
                for t in range(steps):
                    new_joint_angle = (t + 1) / steps * 0.04
                    agent = simulator.robot
                    agent.set_gripper_open_position(agent.right_gripper_indices, [new_joint_angle, new_joint_angle], set_instantly=False)
                    p.stepSimulation()
                    state = save_env(simulator)
                    intermediate_states.append(state)
                    rgb = simulator.render()
                    rgbs.append(rgb)
                                        
                # reach till contact is made, and get the number of handle points between the two fingers
                steps = 30
                for t in range(steps):
                    ik_indices = [_ for _ in range(len(simulator.robot.right_arm_joint_indices))]
                    # decompose the translation and rotation into small delta actions
                    cur_eef_pos, cur_eef_orient = simulator.robot.get_pos_orient(simulator.robot.right_end_effector)
                    delta_pos = real_target_pos - cur_eef_pos
                    delta_pos = delta_pos / np.linalg.norm(delta_pos)
                    cur_eef_orient_matrix = np.array(p.getMatrixFromQuaternion(cur_eef_orient)).reshape(3, 3)
                    new_rotation_matrix = np.array(p.getMatrixFromQuaternion(target_orientation)).reshape(3, 3)
                    delta_rotation_matrix = cur_eef_orient_matrix.T @ new_rotation_matrix
                    delta_axis_angle = scipy.spatial.transform.Rotation.from_matrix(delta_rotation_matrix).as_rotvec()
                    delta_axis_angle = delta_axis_angle / np.linalg.norm(delta_axis_angle)
                    simulator.step(np.array([*delta_pos, *delta_axis_angle, 1]))
                    rgb = simulator.render()
                    rgbs.append(rgb)
                    state = save_env(simulator)
                    intermediate_states.append(state)
                    
                    collision = False
                    points_left_finger = p.getContactPoints(bodyA=simulator.robot.body, linkIndexA=simulator.robot.right_gripper_indices[0], physicsClientId=simulator.id)
                    points_right_finger = p.getContactPoints(bodyA=simulator.robot.body, linkIndexA=simulator.robot.right_gripper_indices[1], physicsClientId=simulator.id)
                    points_hand = p.getContactPoints(bodyA=simulator.robot.body, linkIndexA=8, physicsClientId=simulator.id)
                    points = points_left_finger + points_right_finger + points_hand
                    collision_points_a = [points[_][5] for _ in range(len(points))]
                    if len(collision_points_a) > 0:
                        p.addUserDebugPoints(collision_points_a, [[0, 1, 0] for _ in range(len(collision_points_a))], 12, 0.55, physicsClientId=simulator.id)
                    if points:
                        # Handle contact between suction with a rigid object.
                        for point in points:
                            obj_id, contact_link, contact_position_on_obj = point[2], point[4], point[6]
                            if obj_id == simulator.urdf_ids['plane'] or obj_id == simulator.robot.body or (simulator.use_table and obj_id == simulator.table):
                                pass
                            else:
                                # print("collision detected")
                                collision = True
                                break
                    if collision:
                        break
                
                # get a score for this grasping pose, which is the number of handle points between the two fingers
                cur_eef_pos, cur_eef_orient = simulator.robot.get_pos_orient(simulator.robot.right_end_effector)
                score = get_pc_num_within_gripper(cur_eef_pos, cur_eef_orient, handle_pc)
                # import pdb; pdb.set_trace()
                print("points within gripper: ", score)
               
                # close gripper
                close_steps = 40
                left_collision = False
                right_collision = False
                for t in range(close_steps):
                    new_joint_angle = 0.
                    agent = simulator.robot
                    agent.set_gripper_open_position(agent.right_gripper_indices, [new_joint_angle, new_joint_angle], set_instantly=False)
                    p.stepSimulation()
                    state = save_env(simulator)
                    intermediate_states.append(state)
                    rgb = simulator.render()
                    rgbs.append(rgb)
                    
                    # NOTE: update the score such that after closing, both gripper is in contact with the handle itself.
                    points_left_finger = p.getContactPoints(bodyA=simulator.robot.body, linkIndexA=simulator.robot.right_gripper_indices[0], physicsClientId=simulator.id)
                    points_right_finger = p.getContactPoints(bodyA=simulator.robot.body, linkIndexA=simulator.robot.right_gripper_indices[1], physicsClientId=simulator.id)

                    if points_left_finger:
                        # collision_points_b = [points_left_finger[_][6] for _ in range(len(points_left_finger))]
                        collision_points_b = [points_left_finger[_][5] for _ in range(len(points_left_finger))]
                        dist_collision_to_handle = scipy.spatial.distance.cdist(collision_points_b, handle_pc).min(axis=1)
                        if np.sum(dist_collision_to_handle < 0.01) > 0:
                            left_collision = True
                    if points_right_finger:
                        # collision_points_b = [points_right_finger[_][6] for _ in range(len(points_right_finger))]
                        collision_points_b = [points_right_finger[_][5] for _ in range(len(points_right_finger))]
                        dist_collision_to_handle = scipy.spatial.distance.cdist(collision_points_b, handle_pc).min(axis=1)
                        if np.sum(dist_collision_to_handle < 0.01) > 0:
                            right_collision = True
                            
                    if left_collision and right_collision and t > 20:
                        break

                if not (left_collision and right_collision):
                    score = 0
                
                handle_grasp_scores.append(score)    
                print("iteration {} score {}".format(it, score))
                   
                # let's not test this for now
                # pull out following the rotation axis
                if execute_opening_primitive:
                    eef_pos, eef_orient = simulator.robot.get_pos_orient(simulator.robot.right_end_effector)
                    link_pos, link_orient = get_link_pose(simulator, object_name, link_name)
                    world_to_link = p.invertTransform(link_pos, link_orient)
                    # EEf in link frame remains the same as the link frame rotates
                    eef_in_link = p.multiplyTransforms(world_to_link[0], world_to_link[1], eef_pos, eef_orient) 

                    joint_limit = p.getJointInfo(simulator.urdf_ids[object_name], handle_joint_id)[8:10]
                    ori_joint_angle = p.getJointState(simulator.urdf_ids[object_name], handle_joint_id)[0]
                    eef_poses = []
                    timesteps = 250
                    for t in range(1, timesteps):
                        joint_angle = joint_limit[0] + (joint_limit[1] - joint_limit[0]) * t / timesteps
                        p.resetJointState(simulator.urdf_ids[object_name], handle_joint_id, joint_angle)
                        new_link_pos, new_link_orient = get_link_pose(simulator, object_name, link_name)
                        # new_link_pos, new_link_orient is the transformation from link coordinate to world coordinate
                        new_eef_pos, new_eef_orient = p.multiplyTransforms(new_link_pos, new_link_orient, eef_in_link[0], eef_in_link[1])
                        eef_poses.append([new_eef_pos, new_eef_orient])
                    
                    p.resetJointState(simulator.urdf_ids[object_name], handle_joint_id, ori_joint_angle)
                    for t in range(len(eef_poses)):
                        pos, orient = eef_poses[t]
                        ik_indices = [_ for _ in range(len(simulator.robot.right_arm_joint_indices))]
                        ik_joints = simulator.robot.ik(simulator.robot.right_end_effector, 
                                                        pos, orient, 
                                                        ik_indices=ik_indices)
                        agent = simulator.robot
                        for _ in range(2):
                            new_joint_angle = 0
                            agent.set_gripper_open_position(agent.right_gripper_indices, [new_joint_angle, new_joint_angle], set_instantly=False)
                            p.setJointMotorControlArray(simulator.robot.body, jointIndices=simulator.robot.right_arm_joint_indices, 
                                                        controlMode=p.POSITION_CONTROL, targetPositions=ik_joints, physicsClientId=simulator.id)
                            p.stepSimulation()
                            
                        rgb = simulator.render()
                        rgbs.append(rgb)
                        state = save_env(simulator)
                        intermediate_states.append(state)
                        
                    env_states.append(intermediate_states)
                    rgb_images.append(rgbs)
                 
            # no need to try the other normal direction
            break
                        
    if len(handle_grasp_scores) > 0 and np.max(handle_grasp_scores) > 0:
        best_idx = np.argmax(handle_grasp_scores)
        best_score = handle_grasp_scores[best_idx]
        with open(os.path.join(save_path, "best_score.txt"), "w") as f:
            f.write(str(best_score))
            
        # store the best env states
        state_files = []
        for t_idx, state in enumerate(env_states[best_idx]):
            save_state_path = os.path.join(save_path, "states",  "state_{}.pkl".format(t_idx))
            state_files.append(save_state_path)
            with open(save_state_path, 'wb') as f:
                pickle.dump(state, f, pickle.HIGHEST_PROTOCOL)
        
        # get the opened angle of the last state
        load_env(simulator, state=env_states[best_idx][-1])
        joint_angle = p.getJointState(simulator.urdf_ids[object_name], handle_joint_id)[0]
        joint_limit_low, joint_limit_high = p.getJointInfo(simulator.urdf_ids[object_name], handle_joint_id)[8:10]
        with open(os.path.join(save_path, "opened_angle.txt"), "w") as f:
            f.write(str(joint_angle) + "\n")
            f.write(str(joint_limit_low) + "\n")
            f.write(str(joint_limit_high) + "\n")
        simulator.reset(ori_simulator_state)
                
        # p.removeUserDebugItem(debug_id)
        return rgb_images[best_idx], state_files
    
    with open(os.path.join(save_path, "best_score.txt"), "w") as f:
        f.write(str(0))
            
    joint_limit_low, joint_limit_high = p.getJointInfo(simulator.urdf_ids[object_name], handle_joint_id)[8:10]
    with open(os.path.join(save_path, "opened_angle.txt"), "w") as f:
        f.write(str(0) + "\n")
        f.write(str(joint_limit_low) + "\n")
        f.write(str(joint_limit_high) + "\n")
            
    load_env(simulator, state=ori_simulator_state)
    save_env(simulator, os.path.join(save_path,  "state_{}.pkl".format(0)))
    rgbs = [simulator.render()]
    state_files = [os.path.join(save_path,  "state_{}.pkl".format(0))]
    # p.removeUserDebugItem(debug_id)
    return rgbs, state_files
            
            
def get_pc_num_within_gripper(cur_eef_pos, cur_eef_orient, pc_points):
    
    cur_pos, cur_orient = cur_eef_pos, cur_eef_orient

    X_GW = p.invertTransform(cur_pos, cur_orient)
    translation = np.array(X_GW[0])
    rotation = np.array(p.getMatrixFromQuaternion(X_GW[1])).reshape(3, 3)
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation ### this is the transformation from world frame to gripper frame

    pc_homogeneous = np.hstack((pc_points, np.ones((pc_points.shape[0], 1))))  # Convert to homogeneous coordinates Nx4
    pc_transformed_homogeneous = T @ pc_homogeneous.T # 4x4 @ 4xN = 4xN
    p_GC = pc_transformed_homogeneous[:3, :] # 3xN

    ### Crop to a region inside of the finger box.
    crop_min = [-0.02, -0.06, -0.01] 
    crop_max = [0.02, 0.06, 0.01]
    indices = np.all(
        (
            crop_min[0] <= p_GC[0, :],
            p_GC[0, :] <= crop_max[0],
            crop_min[1] <= p_GC[1, :],
            p_GC[1, :] <= crop_max[1],
            crop_min[2] <= p_GC[2, :],
            p_GC[2, :] <= crop_max[2],
        ),
        axis=0,
    )
    
    within_bbox_handle_pc = pc_points[indices]
    if len(within_bbox_handle_pc) == 0:
        # print("no points are within the gripper")
        return 0
    debug_id = p.addUserDebugPoints(within_bbox_handle_pc, [[0, 1, 0] for _ in range(len(within_bbox_handle_pc))], 10, 0)
    score = np.sum(indices) 
    # print("score is: ", score)
    p.removeUserDebugItem(debug_id)
    return score

def get_handle_orient(handle_pc):
    # get axis aligned bounding box of the handle pc
    min_xyz = np.min(handle_pc, axis=0)
    max_xyz = np.max(handle_pc, axis=0)
    x_range = max_xyz[0] - min_xyz[0]
    y_range = max_xyz[1] - min_xyz[1]
    z_range = max_xyz[2] - min_xyz[2]
    horizontal_range = np.max([x_range, y_range])
    vertical_range = z_range
    if horizontal_range > vertical_range:
        handle_orient = "horizontal"
    else:
        handle_orient = "vertical"
    
    return handle_orient