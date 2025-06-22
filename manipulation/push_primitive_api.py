import pybullet as p
import numpy as np
import open3d as o3d
from manipulation.motion_planning_utils import motion_planning
from manipulation.grasping_utils import align_gripper_x_and_z

from manipulation.primitive_api import get_save_path, close_gripper, open_door
from manipulation.utils import *
import time
from termcolor import cprint
import fpsample
from multiprocessing import Pool

MOTION_PLANNING_TRY_TIMES=100
SAMPLE_ORIENTATION_NUM=3
PARALLEL_POOL_NUM=40
HANDLE_FPS_NUM_POINT=15

handle_clip = {
    'faucet': (0.4, 0.8, 0.0, 1.0, 0.0),
    'bucket': (0.9, 1.0, 0.0, 1.0, 0.25),
    'foldingchair': (0.9, 1.0, 0.0, 1.0, 0.25),
    'stapler': (0.5, 0.9, 0.7, 1.0, 0.0),
    'laptop': (0.9, 1.0, 0.0, 1.0, 0.25),
    'toilet': (0.95, 1.0, 0.0, 1.0, 0.25),
}


def push_object_link_parallel(simulator, object_name, link_name, debug=False):
    save_path = get_save_path(simulator)
    ori_simulator_state = save_env(simulator, None)
    object_name = object_name.lower()
    link_name = link_name.lower()

    if object_name not in ['foldingchair', 'laptop', 'toilet']:
        print(f"Object {object_name} is not supported for push primitive.")
        raise NotImplementedError

    # get the point cloud of the handle
    link_pc, view, projection, img = simulator.get_link_pc(object_name, link_name)
    object_pc = link_pc
    all_handle_pos, handle_joint_id, axis_world, axis_end_world = simulator.get_handle_pos(return_median=False, custom_joint_name=simulator.handle_name)
    if simulator.robot_name == 'panda':
        threshold = 0.02
    elif simulator.robot_name == 'xarm':
        threshold = 0.005
    handle_pc, handle_joint_id, handle_median, _ = get_link_handle(all_handle_pos, handle_joint_id, link_pc, threshold=threshold)
    print(handle_pc.shape)

    # filter based on the distance to the screw axis
    axis = axis_world[0]
    axis_end = axis_end_world[0]
    distance_pc = pc_to_line_distance(handle_pc, axis, axis_end)
    max_distance = np.max(distance_pc)
    handle_clip_factor = handle_clip[object_name] if object_name in handle_clip else (0.4, 0.8, 0.0, 1.0, 0.0)
    selected_idx_screw = np.where((distance_pc > max_distance * handle_clip_factor[0]) & (distance_pc < max_distance * handle_clip_factor[1]))[0]

    # filter based on height
    max_z = np.max(handle_pc[:, 2])
    min_z = np.min(handle_pc[:, 2])
    selected_idx_height = np.where((handle_pc[:, 2] > min_z + (max_z - min_z) * handle_clip_factor[2]) & (handle_pc[:, 2] < min_z + (max_z - min_z) * handle_clip_factor[3]))[0]

    # get the intersection of the two selections
    selected_idx = np.intersect1d(selected_idx_screw, selected_idx_height)
    handle_pc = handle_pc[selected_idx]

    # for foldingchair, there may be too opposite parts, we need to filter them out
    if object_name == 'foldingchair':
        handle_pc = filter_pointcloud_by_line(handle_pc, axis, axis_end)

    # clip points that is too far from the handle center
    handle_dir = estimate_line_direction(handle_pc) # use PCA to estimate the handle direction
    handle_pc_project = np.dot(handle_pc, handle_dir)
    min_handle_pc_project = np.min(handle_pc_project)
    max_handle_pc_project = np.max(handle_pc_project)
    selected_idx = np.where((handle_pc_project > min_handle_pc_project + (max_handle_pc_project - min_handle_pc_project) * handle_clip_factor[4]) & (handle_pc_project < min_handle_pc_project + (max_handle_pc_project - min_handle_pc_project) * (1-handle_clip_factor[4])))[0]
    handle_pc = handle_pc[selected_idx]

    handle_median = np.median(handle_pc, axis=0)

    # use fps to get a bunch of trying points
    fps_point = HANDLE_FPS_NUM_POINT if simulator.robot_name == 'panda' else 20
    handle_fps_num_point = min(fps_point, len(handle_pc))

    if handle_fps_num_point < 3:
        # maybe use the wrong link_id or handle_name
        raise ValueError(f"Not enough handle points to sample, got {handle_fps_num_point} points, need at least 3.")

    h = min(3, int(np.log2(handle_fps_num_point)))
    kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(handle_pc, handle_fps_num_point, h=h)
    to_try_handle_points = handle_pc[kdline_fps_samples_idx] 
    to_try_handle_points = np.concatenate([to_try_handle_points, handle_median.reshape(1, 3)], axis=0)

    args = []

    # parallel motion planning to search each fps handle point

    env_kwargs = {
        "task_config": simulator.config_path, 
        "env_name": "articulated",
        "task_name": simulator.task_name, 
        "restore_state_file": simulator.restore_state_file, 
        "render": False if not debug else True, 
        "randomize": False, 
        "obj_id": simulator.obj_id, 
    }

    mp_target_poses = []
    real_target_poses = []
    target_orientations = []

    for target_pos in to_try_handle_points:
        # compute the dir vec from screw to the point
        screw_dir = axis_end - axis
        screw_dir = screw_dir / np.linalg.norm(screw_dir)
        target_pos_displacement = target_pos - axis
        target_pos_projection = np.dot(target_pos_displacement, screw_dir) 
        vertical_dir = target_pos_displacement - target_pos_projection * screw_dir
        vertical_dir = vertical_dir / np.linalg.norm(vertical_dir)
        normal_dir = np.cross(screw_dir, vertical_dir)
        # normal dir point to positive side of x
        if normal_dir[0] < 0:
            normal_dir = -normal_dir
        
        # gripper parallel to the handle
        if object_name in ['bucket', 'foldingchair', 'laptop', 'toilet']:
            mp_target_pos = target_pos + normal_dir * 0.04
            real_target_pos = target_pos

            for orientation_idx in range(SAMPLE_ORIENTATION_NUM):
                target_orientation = align_gripper_x_and_z(normal_dir, -vertical_dir, randomize=True).as_quat()
                mp_target_poses.append(mp_target_pos)
                real_target_poses.append(real_target_pos)
                target_orientations.append(target_orientation)

            target_orientation_1 = align_gripper_x_and_z(normal_dir, -vertical_dir, randomize=False, flip=False).as_quat()
            target_orientation_2 = align_gripper_x_and_z(normal_dir, -vertical_dir, randomize=False, flip=True).as_quat()
            mp_target_poses.append(mp_target_pos)
            mp_target_poses.append(mp_target_pos)
            real_target_poses.append(real_target_pos)
            real_target_poses.append(real_target_pos)
            target_orientations.append(target_orientation_1)
            target_orientations.append(target_orientation_2)

        # gripper vertical to the handle
        if object_name in ['foldingchair', 'laptop', 'toilet', 'stapler']:
            mp_target_pos = target_pos + normal_dir * 0.04 - vertical_dir * 0.02
            if simulator.robot_name == 'panda':
                real_target_pos = target_pos + normal_dir * -0.02 - vertical_dir * 0.02
            elif simulator.robot_name == 'xarm':
                real_target_pos = target_pos + normal_dir * -0.01 - vertical_dir * 0.02

            for orientation_idx in range(SAMPLE_ORIENTATION_NUM):
                target_orientation = align_gripper_x_and_z(vertical_dir, -normal_dir, randomize=True).as_quat()
                mp_target_poses.append(mp_target_pos)
                real_target_poses.append(real_target_pos)
                target_orientations.append(target_orientation)

            target_orientation_1 = align_gripper_x_and_z(vertical_dir, -normal_dir, randomize=False, flip=False).as_quat()
            target_orientation_2 = align_gripper_x_and_z(vertical_dir, -normal_dir, randomize=False, flip=True).as_quat()
            mp_target_poses.append(mp_target_pos)
            mp_target_poses.append(mp_target_pos)
            real_target_poses.append(real_target_pos)
            real_target_poses.append(real_target_pos)
            target_orientations.append(target_orientation_1)
            target_orientations.append(target_orientation_2)
    
    args =  [[env_kwargs, object_name, real_target_poses[it], mp_target_poses[it], target_orientations[it],\
            handle_pc, handle_joint_id, save_path, ori_simulator_state, it, link_name] for it in range(len(target_orientations))]
    
    if debug:
        results = parallel_motion_planning(args[0])
        results = [results]
    else:
        with Pool(processes=PARALLEL_POOL_NUM) as pool:
            results = pool.map(parallel_motion_planning, args)

    door_opened_ratios = np.array([x[0][0] for x in results])
    door_opened_angles = np.array([x[0][1] for x in results])
    grasp_scores = [x[1] for x in results]
    all_traj_states = [x[2] for x in results]
    all_traj_rgbs = [x[3] for x in results]
    all_stage_lengths = [x[4] for x in results]
    all_motion_planning_path_translation_lengths = [x[5] for x in results]
    all_motion_planning_path_rotation_lengths = [x[6] for x in results]
    ratio_threshold = 0.7
    if len(door_opened_ratios) > 0 and np.max(door_opened_ratios) > 0.1:
        best_idx = None
        if not np.sum(door_opened_ratios > ratio_threshold) > 0:
            best_idx = np.argmax(door_opened_ratios)
        else:
            # TODO: optimize orientation length as well. 
            best_rank = 100000
            path_translation_length_rank = np.argsort(all_motion_planning_path_translation_lengths)
            path_rotation_length_rank = np.argsort(all_motion_planning_path_rotation_lengths)
            grasping_score_rank = np.argsort(-np.array(grasp_scores))
            for idx, score in enumerate(door_opened_ratios):
                # if score > 0.8 and path_translation_length_rank[idx] + path_rotation_length_rank[idx] + grasping_score_rank[idx] < best_rank:
                if score > ratio_threshold and path_translation_length_rank[idx] + grasping_score_rank[idx] < best_rank:
                    best_idx = idx
                    # best_rank = path_translation_length_rank[idx] + path_rotation_length_rank[idx] + grasping_score_rank[idx]
                    best_rank = path_translation_length_rank[idx] + grasping_score_rank[idx]
            
        best_score = grasp_scores[best_idx]
        with open(os.path.join(save_path, "best_score.txt"), "w") as f:
            f.write(str(best_score))
            
        # store the best env states
        state_files = []
        for t_idx, state in enumerate(all_traj_states[best_idx]):
            save_state_path = os.path.join(save_path, "states",  "state_{}.pkl".format(t_idx))
            state_files.append(save_state_path)
            with open(save_state_path, 'wb') as f:
                pickle.dump(state, f, pickle.HIGHEST_PROTOCOL)
        
        # get the opened angle of the last state
        joint_limit_low, joint_limit_high = p.getJointInfo(simulator.urdf_ids[object_name], handle_joint_id, physicsClientId=simulator.id)[8:10]
        best_opened_angle = door_opened_angles[best_idx]
        with open(os.path.join(save_path, "opened_angle.txt"), "w") as f:
            f.write(str(best_opened_angle) + "\n")
            f.write(str(joint_limit_low) + "\n")
            f.write(str(joint_limit_high) + "\n")
        simulator.reset(ori_simulator_state)
        
        best_stage_length = all_stage_lengths[best_idx]
        with open(os.path.join(save_path, "stage_lengths.json"), "w") as f:
            json.dump(best_stage_length, f, indent=4)
                
        return all_traj_rgbs[best_idx], state_files
    
    with open(os.path.join(save_path, "best_score.txt"), "w") as f:
        f.write(str(0))
    
    # print("handle joint id: ", handle_joint_id)
    joint_limit_low, joint_limit_high = p.getJointInfo(simulator.urdf_ids[object_name], handle_joint_id, physicsClientId=simulator.id)[8:10]
    with open(os.path.join(save_path, "opened_angle.txt"), "w") as f:
        f.write(str(0) + "\n")
        f.write(str(joint_limit_low) + "\n")
        f.write(str(joint_limit_high) + "\n")
            
    load_env(simulator, state=ori_simulator_state)
    save_env(simulator, os.path.join(save_path,  "state_{}.pkl".format(0)))
    rgbs = [simulator.render()]
    state_files = [os.path.join(save_path,  "state_{}.pkl".format(0))]
    return rgbs, state_files
        

def reach_till_contact(simulator, real_target_pos, target_orientation):
    intermediate_states = []
    rgbs = []
    cur_eef_pos, _ = simulator.robot.get_pos_orient(simulator.robot.right_end_effector)
    moving_vector = real_target_pos - cur_eef_pos
    delta_movement = 0.005
    movement_steps = int(np.linalg.norm(moving_vector) / delta_movement) + 1
    moving_direction = moving_vector / np.linalg.norm(moving_vector)
    target_orient_euler = p.getEulerFromQuaternion(target_orientation)
    for t in range(movement_steps):
        target_pos = cur_eef_pos + moving_direction * delta_movement * (t + 1)
        simulator.take_direct_action(np.array([*target_pos, *target_orient_euler, 0.0]))
        rgb = simulator.render()
        rgbs.append(rgb)
        state = save_env(simulator)
        intermediate_states.append(state)
        
            
    if simulator.robot_name == 'panda' and len(intermediate_states) >= 3:
        return intermediate_states[:-2], rgbs[:-2]
    else:
        return intermediate_states, rgbs

def parallel_motion_planning(args):
    debug = False
    np.random.seed(time.time_ns() % 2**32)
    
    env_kwargs, object_name, real_target_pos, mp_target_pos, target_orientation, \
        handle_pc, handle_joint_id, save_path, ori_simulator_state, \
        it, link_name = args
        
    stage_length = {}
    object_name = object_name.lower()
    
    simulator, _ = build_up_env(
        **env_kwargs
    )
    # load_env(simulator, state=ori_simulator_state)
    simulator.reset(ori_simulator_state)

    # add coordinate system for debugging
    p.addUserDebugLine([0, 0, 0], [1, 0, 0], [1, 0, 0], lineWidth=2, lifeTime=0, physicsClientId=simulator.id)
    p.addUserDebugLine([0, 0, 0], [0, 1, 0], [0, 1, 0], lineWidth=2, lifeTime=0, physicsClientId=simulator.id)
    p.addUserDebugLine([0, 0, 0], [0, 0, 1], [0, 0, 1], lineWidth=2, lifeTime=0, physicsClientId=simulator.id)

    intermediate_states = []
    rgbs = []

    # close the gripper
    close_states, close_rgbs, _, _ = close_gripper(simulator, handle_pc)
    intermediate_states += close_states
    rgbs += close_rgbs
    stage_length['close_gripper'] = len(close_states)

    # approach the handle using motion planning
    all_objects = list(simulator.urdf_ids.keys())
    all_objects.remove("robot")
    obstacles = [simulator.urdf_ids[x] for x in all_objects]
    allow_collision_links = []
    cur_eef_pos, cur_eef_orient = simulator.robot.get_pos_orient(simulator.robot.right_end_effector)
    translation_length = np.linalg.norm(mp_target_pos - cur_eef_pos)
    rotation_length = 2 * np.arccos(np.abs(np.dot(target_orientation, cur_eef_orient)))
    rotation_length = np.rad2deg(rotation_length)
    translation_steps = int(translation_length / 0.004) + 1
    rotation_steps = int(rotation_length / 1.8) + 1
    interpolation_steps = max(translation_steps, rotation_steps)
    
    if debug:
        p.addUserDebugPoints([mp_target_pos], [[1, 0, 0]], 12, 0, physicsClientId=simulator.id)
        p.addUserDebugPoints([real_target_pos], [[1, 0, 0]], 12, 0, physicsClientId=simulator.id)
    
    res, path, path_translation_length, path_rotation_length = motion_planning(
        simulator, mp_target_pos, target_orientation, obstacles=obstacles, allow_collision_links=allow_collision_links, save_path=save_path, 
        smooth_path=True, interpolation_num=interpolation_steps)
    if not res:
        print(f"Motion planning failed for {object_name} at target position {mp_target_pos} and orientation {target_orientation}.")
        simulator.close()
        return (-1, -1), -1, [], [], {}, np.inf, np.inf

    stage_length['reach_handle'] = len(path) + len(close_states)

    if debug:
        import pdb; pdb.set_trace()

    for idx, q in enumerate(path):
        simulator.robot.set_joint_angles(simulator.robot.right_arm_joint_indices, q)
        rgb = simulator.render()
        rgbs.append(rgb)
        state = save_env(simulator)
        intermediate_states.append(state)

    if debug:
        import pdb; pdb.set_trace()

    # reach till contact
    reach_to_concatc_states, reach_to_contact_rgbs = reach_till_contact(simulator, real_target_pos, target_orientation)
    intermediate_states += reach_to_concatc_states
    rgbs += reach_to_contact_rgbs
    stage_length['reach_to_contact'] = len(reach_to_contact_rgbs)

    open_door_states, open_door_rgbs, final_joint_angle_ratio = open_door(simulator, object_name, link_name, handle_joint_id, handle_pc, invert=True)

    intermediate_states += open_door_states
    rgbs += open_door_rgbs
    stage_length['open_door'] = len(open_door_states)
    cprint(f"final joint angle ratio: {final_joint_angle_ratio}", "green")
    # save_numpy_as_gif(np.array(rgbs), "tmp.gif")
    simulator.close()
    score = 10
    return final_joint_angle_ratio, score, intermediate_states, rgbs, stage_length, path_translation_length, path_rotation_length
