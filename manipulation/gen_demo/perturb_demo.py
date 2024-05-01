import os
from manipulation.utils import build_up_env, load_env, save_env, save_numpy_as_gif
import json
import yaml
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R
from manipulation.motion_planning_utils import motion_planning
import pickle
from gpt_primitive_api import get_link_pose
import scipy
from manipulation.gpt_reward_api import get_link_pc, get_handle_pos
from manipulation.gpt_primitive_api import open_gripper, close_gripper, reach_till_contact, open_door, get_link_handle
from multiprocessing import set_start_method
from multiprocessing import Pool

def generate_and_execute_perturbed_actions(env, demo_path, step_name, perturb_t, end_step, noise_ratio=0.2, perturb_finger=False):
    eef_poses = []
    eef_quats = []
    finger_joints = []
    for t in range(perturb_t, end_step):
        state_path = os.path.join(demo_path, step_name, "states", "state_{}.pkl".format(t))
        load_env(env, load_path=state_path)
        eef_pos, eef_quat = env.robot.get_pos_orient(env.robot.right_end_effector)
        left_finger_joint_angle = p.getJointState(env.robot.body, env.robot.right_gripper_indices[0], physicsClientId=env.id)[0]
        eef_poses.append(eef_pos)
        eef_quats.append(eef_quat)
        finger_joints.append(left_finger_joint_angle)

    perturbed_translations = []
    perturbed_rotations = []
    perturbed_fingers = []
    noise_ratio = noise_ratio
    for t in range(len(eef_poses) - 1):
        ori_pos = eef_poses[t]
        ori_quat = eef_quats[t]
        ori_matrix = np.array(p.getMatrixFromQuaternion(ori_quat)).reshape(3, 3)
        
        next_pos = eef_poses[t + 1]
        next_quat = eef_quats[t + 1]
        next_matrix = np.array(p.getMatrixFromQuaternion(next_quat)).reshape(3, 3)
        
        delta_pos = next_pos - ori_pos
        delta_matrix = ori_matrix.T @ next_matrix
        
        translation_scale = np.linalg.norm(delta_pos)
        noise_level = translation_scale * noise_ratio
        # add gaussian noise to the translation
        noise = np.random.normal(0, noise_level, 3)
        perturbed_translations.append(delta_pos + noise)
        
        # generate small rotation perturbation along each axis
        delta_orient = R.from_matrix(delta_matrix)
        delta_orient_rotvec = delta_orient.as_rotvec()
        noise_level = np.linalg.norm(delta_orient_rotvec) * noise_ratio
        # generate small noise along each axis
        noise = np.random.normal(0, noise_level, 3)
        perturb_rotation_x = np.array([noise[0], 0, 0])
        perturb_rotation_y = np.array([0, noise[1], 0])
        perturb_rotation_z = np.array([0, 0, noise[2]])
        perturbed_rotation_x = R.from_rotvec(perturb_rotation_x)
        perturbed_rotation_y = R.from_rotvec(perturb_rotation_y)
        perturbed_rotation_z = R.from_rotvec(perturb_rotation_z)
        perturbed_rotation = delta_orient * perturbed_rotation_x * perturbed_rotation_y * perturbed_rotation_z
        perturbed_rotations.append(perturbed_rotation)
        
        if perturb_finger:
            ori_finger_joint_angle = finger_joints[t]
            next_finger_joint_angle = finger_joints[t + 1]
            delta_finger = next_finger_joint_angle - ori_finger_joint_angle
            noise_level = np.abs(delta_finger) * noise_ratio
            noise = np.random.normal(0, noise_level)
            perturbed_fingers.append(delta_finger + noise)

    # execute the actions
    start_state_file = os.path.join(demo_path, step_name, "states", "state_{}.pkl".format(perturb_t))
    reset_state = pickle.load(open(start_state_file, "rb"))
    env.reset(reset_state=reset_state)
    for idx in range(len(perturbed_translations)):
        # execute the perturbed action
        translation = perturbed_translations[idx]
        rotation = perturbed_rotations[idx]
        cur_eef_pos, cur_eef_quat = env.robot.get_pos_orient(env.robot.right_end_effector)
        target_pos = cur_eef_pos + translation
        target_orient_matrix = np.array(p.getMatrixFromQuaternion(cur_eef_quat)).reshape(3, 3) @ rotation.as_matrix()
        target_quat = R.from_matrix(target_orient_matrix).as_quat()
        target_euler = p.getEulerFromQuaternion(target_quat)
        if not perturb_finger:
            env.take_direct_action([*target_pos, *target_euler, 0])
        else:
            delta_finger_angle = perturbed_fingers[idx]
            cur_finger_angle = p.getJointState(env.robot.body, env.robot.right_gripper_indices[0], physicsClientId=env.id)[0]
            target_finger_angle = cur_finger_angle + delta_finger_angle
            target_finger_angle = np.clip(target_finger_angle, 0, 0.04)
            env.take_direct_action([*target_pos, *target_euler, target_finger_angle])
            
    return save_env(env)
            
def recover_and_save(env, start_state, demo_path, step_name, perturb_t, recover_eef_pos, recover_eef_quat, save_name, 
                     recover_finger_joint=False, object_name=None, link_name=None, 
                     reach_contact_target_pos=None, reach_contact_target_orient=None):
    
    env.reset(reset_state=start_state)
    recover_states = []
    rgbs = []
    stage_lengths = {}
    
    all_objects = list(env.urdf_ids.keys())
    all_objects.remove("robot")
    obstacles = [env.urdf_ids[x] for x in all_objects]
    allow_collision_links = []
    # NOTE: determine the right interpolation num; this can be based on eef translation position change
    current_eef_pos, current_eef_quat = env.robot.get_pos_orient(env.robot.right_end_effector)
    translation_distance = np.linalg.norm(np.array(recover_eef_pos) - np.array(current_eef_pos))
    delta_translation = 0.003
    interpolation_num = int(translation_distance / delta_translation) + 1
    res, path, path_length = motion_planning(
        env, recover_eef_pos, recover_eef_quat, obstacles=obstacles, allow_collision_links=allow_collision_links, 
        smooth_path=True, interpolation_num=interpolation_num)
            
    if res:    
        for idx, q in enumerate(path):
            env.robot.set_joint_angles(env.robot.right_arm_joint_indices, q)
            rgb = env.render()
            rgbs.append(rgb)
            state = save_env(env)
            recover_states.append(state)
        stage_lengths['reach_handle'] = len(path)
            
        link_pc = get_link_pc(env, object_name, link_name)
        all_handle_pos, handle_joint_id = get_handle_pos(env, object_name, return_median=False)
        handle_pc, handle_joint_id, handle_median = get_link_handle(all_handle_pos, handle_joint_id, link_pc)
        # if recover_finger_joint:
        
        # import pdb; pdb.set_trace()
        open_states, open_rgbs = open_gripper(env)
        recover_states += open_states
        rgbs += open_rgbs
        stage_lengths['open_gripper'] = len(open_states)
        
        # reach till contact again
        # import pdb; pdb.set_trace()
        reach_to_contact_states, reach_to_contact_rgbs = reach_till_contact(env, reach_contact_target_pos, reach_contact_target_orient)
        recover_states += reach_to_contact_states
        rgbs += reach_to_contact_rgbs
        stage_lengths['reach_to_contact'] = len(reach_to_contact_states)
        
        # close gripper
        # import pdb; pdb.set_trace()
        close_states, close_rgbs, left_collision, right_collision = close_gripper(env, handle_pc)
        # TODO: handle the case where this fails
        recover_states += close_states
        rgbs += close_rgbs
        stage_lengths['close_gripper'] = len(close_states)
            
                
        # # NOTE: test if after this recovery, we can still correctly open the door
        # import pdb; pdb.set_trace()
        open_states, open_rgbs, final_opened_angle = open_door(env, object_name, link_name, handle_joint_id)
        recover_states += open_states
        rgbs += open_rgbs
        stage_lengths['open_door'] = len(open_states)

        new_demo_path = os.path.join(demo_path + "_perturbed_{}".format(save_name))
        # os.system("rsync -avrz --exclude='*.pkl' {} {}".format(demo_path, new_demo_path))
        os.makedirs(os.path.join(new_demo_path, step_name, "states"), exist_ok=True)
        perturbed_meta_info = {
            "perturbed_stage": save_name,
            "perturbed_t": perturb_t,
            "original_demo_path": demo_path,
        }
        with open(os.path.join(new_demo_path, step_name, "meta_info.json"), "w") as f:
            json.dump(perturbed_meta_info, f)
            
        with open(os.path.join(new_demo_path, step_name, "opened_angle.txt"), "w") as f:
            f.writelines(str(final_opened_angle))
            
        with open(os.path.join(new_demo_path, step_name, "stage_lengths.json"), "w") as f:
            json.dump(stage_lengths, f)
            
        for idx, state in enumerate(recover_states):
            state_path = os.path.join(new_demo_path, step_name, "states", "state_{}.pkl".format(idx))
            with open(state_path, "wb") as f:
                pickle.dump(state, f)
        save_numpy_as_gif(np.array(rgbs), os.path.join(new_demo_path, step_name, "all.gif"))


def perturb_demo(args):
    exp_path, task_name, ts = args
    demo_path = os.path.join(exp_path, ts)
    config_path = os.path.join(demo_path, "config.yaml")
    if not os.path.exists(config_path):
        meta_info_path = os.path.join(demo_path, "meta_info.json")
        with open(meta_info_path, "r") as f:
            meta_info = json.load(f)
        config_path = meta_info["config_path"]
        config = yaml.safe_load(open(config_path, "r"))
        solution_path = [x['solution_path'] for x in config if 'solution_path' in x][0]
        
    env, _ = build_up_env(config_path, solution_path, task_name, None, render=False)
    env.reset()

    step_name = "grasp_the_door_handle_primitive"
    label_file = os.path.join(demo_path, step_name, "label.json")
    if os.path.exists(label_file):
        with open(label_file, "r") as f:
            label = json.load(f)
            if not label["good_traj"]:
                return 

    opened_angle_file = os.path.join(demo_path, step_name, "opened_angle.txt")
    if os.path.exists(opened_angle_file): # for some perturbed trajectories, we did not really continue openeing the handle. 
        with open(opened_angle_file, "r") as f:
            opened_angle = f.readlines()
            opened_angle = float(opened_angle[0].lstrip().rstrip())
        if opened_angle < 0.5:
            return

    stage_lengths_json = os.path.join(demo_path, step_name, "stage_lengths.json")
    with open(stage_lengths_json, "r") as f:
        stage_lengths = json.load(f)

    perturb_reaching = True
    perturb_grasping = False
    perturb_opening = False

    # for getting the relative pose we still use the pose that is during the reaching to make contact
    object_name = "storagefurniture" # TODO: change to env.target_object
    link_name = "link_0" # TODO: change to env.target_link    
    recover_idx = stage_lengths["reach_handle"] + stage_lengths["open_gripper"]
    recover_state_file = os.path.join(demo_path, step_name, "states", "state_{}.pkl".format(recover_idx))
    reset_state = pickle.load(open(recover_state_file, "rb"))
    env.reset(reset_state=reset_state)
    eef_pos, eef_orient = env.robot.get_pos_orient(env.robot.right_end_effector)
    link_pos, link_orient = get_link_pose(env, object_name, link_name)
    world_to_link = p.invertTransform(link_pos, link_orient)
    # EEf in link frame remains the same as the link frame rotates
    eef_in_link = p.multiplyTransforms(world_to_link[0], world_to_link[1], eef_pos, eef_orient)

    # for getting the relative pose we still use the pose that is right after contact
    contact_state_file = os.path.join(demo_path, step_name, "states", "state_{}.pkl".format(
        stage_lengths["reach_handle"] + stage_lengths["open_gripper"] + stage_lengths["reach_to_contact"]))
    reset_state = pickle.load(open(contact_state_file, "rb"))
    env.reset(reset_state=reset_state)
    contact_eef_pos, contact_eef_orient = env.robot.get_pos_orient(env.robot.right_end_effector)
    link_pos, link_orient = get_link_pose(env, object_name, link_name)
    world_to_link = p.invertTransform(link_pos, link_orient)
    contact_eef_in_link = p.multiplyTransforms(world_to_link[0], world_to_link[1], contact_eef_pos, contact_eef_orient)


    if perturb_reaching:
        # for reaching stage, find a few time steps to perturb
        # should prioterize in finding the later time steps
        # extract the actions after that time step
        # add random noise to the actions
        # execute those perturbed actions
        # replan to the pregrasp pose
        new_demo_path = os.path.join(demo_path + "_perturbed_reach_handle")
        if os.path.exists(os.path.join(new_demo_path, step_name, "all.gif")): 
            pass
        else:
            start_step = 0
            end_step = stage_lengths["reach_handle"]

            perturb_t = np.random.randint(0, end_step // 3)
            # NOTE: I should really directly store the eef pose and orient in the state file instead of recovering it by resetting the env and using the FK to get the pose
            start_state = generate_and_execute_perturbed_actions(env, demo_path, step_name, perturb_t, end_step, noise_ratio=0.8)
            new_link_pos, new_link_orient = get_link_pose(env, object_name, link_name)
            # new_link_pos, new_link_orient is the transformation from link coordinate to world coordinate
            recover_eef_pos, recover_eef_quat = p.multiplyTransforms(new_link_pos, new_link_orient, eef_in_link[0], eef_in_link[1])
            recover_contact_eef_pos, recover_contact_eef_quat = p.multiplyTransforms(new_link_pos, new_link_orient, contact_eef_in_link[0], contact_eef_in_link[1])
            recover_and_save(env, start_state, demo_path, step_name, perturb_t, recover_eef_pos, recover_eef_quat, "reach_handle", 
                            object_name=object_name, link_name=link_name, 
                            reach_contact_target_pos=recover_contact_eef_pos, reach_contact_target_orient=recover_contact_eef_quat)
        
    # for the opening stage
    # add small noise during opening; could be both position/orientation noise, also finger open/close
    # replan to the pregrasp pose and open the finger

    # for the reaching till contact stage
    # add noise for position/orientation, close finger
    # replan to the after contact pose

    # for the close stage
    # add noise for position/orientation, open finger
    # replan to the grasp contact pose, close finger
    if perturb_grasping:
        new_demo_path = os.path.join(demo_path + "_perturbed_grasp")
        if os.path.exists(os.path.join(new_demo_path, step_name, "all.gif")): 
            pass
        else:
            start_step = stage_lengths["reach_handle"]
            end_step = stage_lengths["reach_handle"] + stage_lengths["open_gripper"] + stage_lengths["reach_to_contact"] 
            perturb_t = start_step
            start_state = generate_and_execute_perturbed_actions(env, demo_path, step_name, perturb_t, end_step, noise_ratio=0.8, perturb_finger=True)
            new_link_pos, new_link_orient = get_link_pose(env, object_name, link_name)
            # new_link_pos, new_link_orient is the transformation from link coordinate to world coordinate
            recover_eef_pos, recover_eef_quat = p.multiplyTransforms(new_link_pos, new_link_orient, eef_in_link[0], eef_in_link[1])
            recover_contact_eef_pos, recover_contact_eef_quat = p.multiplyTransforms(new_link_pos, new_link_orient, contact_eef_in_link[0], contact_eef_in_link[1])
            recover_and_save(env, start_state, demo_path, step_name, perturb_t, recover_eef_pos, recover_eef_quat, "grasp", 
                            object_name=object_name, link_name=link_name, 
                            reach_contact_target_pos=recover_contact_eef_pos, reach_contact_target_orient=recover_contact_eef_quat)

    # for the opening stage
    # add noise to the position / orientation, open finger
    # replan to the grasped pose 
    if perturb_opening:
        new_demo_path = os.path.join(demo_path + "_perturbed_open")
        if os.path.exists(os.path.join(new_demo_path, step_name, "all.gif")): 
            pass
        else:
            start_step = stage_lengths["reach_handle"] + stage_lengths["open_gripper"] + stage_lengths["reach_to_contact"] + stage_lengths['close_gripper']
            end_step = start_step + stage_lengths["open_door"]

            perturb_t = np.random.randint(start_step, start_step + (end_step - start_step) // 2)
            start_state = generate_and_execute_perturbed_actions(env, demo_path, step_name, perturb_t, end_step, noise_ratio=0.2, perturb_finger=True)
            new_link_pos, new_link_orient = get_link_pose(env, object_name, link_name)
            # new_link_pos, new_link_orient is the transformation from link coordinate to world coordinate
            recover_eef_pos, recover_eef_quat = p.multiplyTransforms(new_link_pos, new_link_orient, eef_in_link[0], eef_in_link[1])
            recover_contact_eef_pos, recover_contact_eef_quat = p.multiplyTransforms(new_link_pos, new_link_orient, contact_eef_in_link[0], contact_eef_in_link[1])
            recover_and_save(env, start_state, demo_path, step_name, perturb_t, recover_eef_pos, recover_eef_quat, "open", 
                            object_name=object_name, link_name=link_name, 
                            reach_contact_target_pos=recover_contact_eef_pos, reach_contact_target_orient=recover_contact_eef_quat)
            
    env.close()

if __name__ == "__main__":
    # build the env according to the stored config
    task_name = "grasp_the_door_handle"
    exp_name = "vary_robot_init_joint_near_handle"
    exp_path = "data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/{}".format(exp_name)
    all_timesteps = os.listdir(exp_path)
    all_timesteps = sorted(all_timesteps)
    all_timesteps = [x for x in all_timesteps if "perturbed" not in x]
    
    set_start_method('spawn', force=True)
    num_worker = 80
    pool = Pool(processes=num_worker)
    
    all_args = [
        [exp_path, task_name, ts] for ts in all_timesteps
    ]
    
    for _ in range(5):
        pool.map(perturb_demo, all_args)
    # perturb_demo(exp_path, task_name, all_timesteps[1])
    