import argparse
import os
import yaml
import json
import copy
import numpy as np
import pybullet as p
import time
import datetime
import multiprocessing as mp
from termcolor import cprint
from manipulation.utils import build_up_env, save_numpy_as_gif,  parse_center

def get_all_test_configs(root_dir='data/temp', extract_name=None):
    all_tasks = os.listdir(root_dir)
    all_tasks = sorted(all_tasks)
    yaml_configs = []
    solution_paths = []
    obj_ids = []
    if extract_name is not None:
        all_tasks = [x for x in all_tasks if extract_name in x]
        
    for task in all_tasks:
        path = os.path.join(root_dir, task)
        config_path = os.path.join(path, 'base_config.yaml')
        if not os.path.exists(config_path):
            print(f"Config file {config_path} does not exist.")
            continue
        config = yaml.safe_load(open(config_path, "r"))
        solution_path = [x['solution_path'] for x in config if 'solution_path' in x][0]
        obj_ids.append([x['reward_asset_path'] for x in config if 'reward_asset_path' in x][0])
            
        yaml_configs.append(config_path)
        solution_paths.append(solution_path)
        
    return yaml_configs, solution_paths, obj_ids

def get_current_num_demos(path):
    num = 0
    if os.path.exists(path):
        all_experiments = os.listdir(path)
        all_experiments = sorted(all_experiments)
        for exp in all_experiments:
            state_path = os.path.join(path, exp, "states")
            if os.path.exists(state_path):
                if len(os.listdir(state_path)) > 1 and os.path.exists(os.path.join(path, exp, "all.gif")):
                    num += 1
    return num

def extract_handle_mesh(obj_id):
    from manipulation.scripts.extract_handle_mesh import render
    cur_shape_dir = "data/dataset/{}".format(obj_id)
    cur_result_json = os.path.join(cur_shape_dir, 'result.json')
    with open(cur_result_json, 'r') as fin:
        tree_hier = yaml.safe_load(fin)[0]
    data = tree_hier
    render(data, cur_shape_dir)

def create_config_variant(config_path):
    config_variant_paths = os.path.join("/".join(config_path.split("/")[:-1]), "configs")
    if not os.path.exists(config_variant_paths):
        os.makedirs(config_variant_paths)

    # create config variant
    if args.robot_name == 'panda':
        if not args.use_augmented_handle:
            new_config_path = os.path.join(config_variant_paths, f"config_larger_randomization_{try_times+500}.yaml")
        else:
            new_config_path = os.path.join(config_variant_paths, f"config_handle_augmentation_{i}_{try_times}.yaml")
    elif args.robot_name == 'xarm':
        new_config_path = os.path.join(config_variant_paths, f"config_xarm_{try_times}.yaml")
        
    base_config = yaml.safe_load(open(config_path, "r"))
    new_config = copy.deepcopy(base_config)

    for config_dict in new_config:
        if 'size' in config_dict:
            if args.robot_name == 'xarm':
                size_ratio = np.random.uniform(0.65, 0.9)  ## xarm cannot handle too big objects
            else:
                size_ratio = np.random.uniform(0.7, 1.2) # make our refrigerator smaller    
            config_dict['size'] = size_ratio * config_dict['size']
            config_dict['is_crop_size'] = False
        if 'reward_asset_path' in config_dict and args.use_augmented_handle:
            config_dict['reward_asset_path'] = str(config_dict['reward_asset_path']) + f"_{i}"

    with open(new_config_path, 'w') as f:
        yaml.dump(new_config, f, indent=4)

    return new_config_path

def _gen_init_state(q, config_path, env_name, render, mobile=False, step_stone=False, far_distance=0.7, near_distance=0.3, invert=False):
    # get table bbox
    config = yaml.safe_load(open(config_path, "r"))
    on_table = False
    link_name = 'link_0'
    for config_dict in config:
        if 'use_table' in config_dict:
            on_table = config_dict['use_table']
        if 'name' in config_dict:
            object_name = config_dict['name'].lower()
        if 'link_name' in config_dict:
            link_name = config_dict['link_name']
        if 'euler' in config_dict:
            base_euler = parse_center(config_dict['euler'])
        if 'center' in config_dict:
            base_pos = parse_center(config_dict['center'])
            print("base pos: ", base_pos)

    # create env
    
    env, _ =  build_up_env(config_path, env_name, render=render)
    env.reset()

    if on_table:
        table_bbox_min, table_bbox_max = env.table_bbox_min, env.table_bbox_max
    else:
        table_bbox_min, table_bbox_max = [0,0,0], [0,0,0]

    # get the joint limits for the robot arm, using a smaller range
    if env.robot_name == 'panda':
        initial_joint_angles = [0 for _ in range(7)]
        low = [-2.9, -1.8, -2.9, -3.1, -2.9, -0.0, -2.9]
        high = [2.9, 1.8, 2.9, 0.0, 2.9, 3.8, 2.9]
        for i in range(7):
            joint_range = high[i] - low[i]
            low[i] += joint_range * 0.2
            high[i] -= joint_range * 0.2    
    elif env.robot_name == 'xarm':
        initial_joint_angles = [0 for _ in range(len(env.robot.right_arm_joint_indices))]
        low = [env.robot.joints[i].lowerLimit for i in env.robot.right_arm_joint_indices]
        high = [env.robot.joints[i].upperLimit for i in env.robot.right_arm_joint_indices]
        for i in range(7):
            joint_range = high[i] - low[i]
            low[i] += joint_range * 0.2
            high[i] -= joint_range * 0.2

     # get the initial position and orientation of the object

    object_id = env.urdf_ids[object_name]   
    init_pos, init_orient = p.getBasePositionAndOrientation(object_id, physicsClientId=env.id)
    init_euler = p.getEulerFromQuaternion(init_orient)

    good_init_pos = False
    while not good_init_pos:
        height = 0

        # randomly sample the position and orientation of the object
        new_pos = np.array([0, 0, init_pos[2]])
        if env.robot_name == 'panda':
            new_pos[0] = np.random.uniform(-0.1, 0.1) + base_pos[0]
        elif env.robot_name == 'xarm':
            new_pos[0] = np.random.uniform(0.65, 1.0) # xarm has a lower reachability
        new_pos[1] = np.random.uniform(-0.1, 0.1) + base_pos[1]
        new_pos[2] = height + init_pos[2]

        init_angle = None

        if invert:
            info = env._get_info()
            handle_joint_id = env.handle_joint
            print("handle joint id: ", handle_joint_id)
            joint_limit_max = p.getJointInfo(env.urdf_ids[object_name], handle_joint_id, physicsClientId=env.id)[9]
            joint_limit_min = p.getJointInfo(env.urdf_ids[object_name], handle_joint_id, physicsClientId=env.id)[8]
            joint_limit = joint_limit_max - joint_limit_min
            if object_name == 'bucket':
                init_angle = np.random.uniform(-np.pi / 6, np.pi / 6) + np.pi / 6 * 5
            elif object_name == 'laptop':
                init_angle = np.random.uniform(np.pi / 3, min(np.pi * 2 / 3, joint_limit - np.pi / 6))
            elif object_name == 'toilet':
                init_angle = np.random.uniform(np.pi / 6, min(np.pi / 3, joint_limit - np.pi / 6))
            elif object_name == 'faucet':
                init_angle = joint_limit - np.random.uniform(0, np.pi / 3)
            elif object_name == 'foldingchair':
                init_angle = np.random.uniform(np.pi / 4, min(np.pi / 3, joint_limit - np.pi / 6))
            elif object_name == 'stapler':
                init_angle = np.random.uniform(np.pi / 3, min(np.pi * 2 / 3, joint_limit - np.pi / 6))
            elif object_name == 'storagefurniture':
                init_angle = np.random.uniform(0.5, 1.0) * joint_limit 
            
            
        else:
            if object_name == 'bucket' or object_name == 'laptop':
                init_angle = np.random.uniform(-np.pi / 12, np.pi / 12) + np.pi / 6
            elif object_name == 'toilet':
                init_angle = np.random.uniform(-np.pi / 16, np.pi / 16) + np.pi / 8
            elif object_name == 'storagefurniture':
                init_angle = np.random.uniform(0, 0.8) * joint_limit

        if env.robot_name == 'panda':
            new_orient = p.getQuaternionFromEuler([init_euler[0], init_euler[1], base_euler[2] + np.random.uniform(-np.pi / 6, np.pi / 6)])
        elif env.robot_name == 'xarm':
            new_orient = p.getQuaternionFromEuler([init_euler[0], init_euler[1], base_euler[2] + np.random.uniform(-np.pi / 7, np.pi / 7)])

        # load step stone
        if step_stone:
            stepping_obj_pos = [new_pos[0], new_pos[1], table_bbox_max[2]]
            stepping_stone_id = p.loadURDF("objaverse_utils/data/obj/f9a7942ee5894152b73b72ce83ac9ee5/material.urdf", stepping_obj_pos, globalScaling=0.34313793694655315, physicsClientId=env.id)
            min_aabb, _ = p.getAABB(stepping_stone_id, physicsClientId=env.id) 
            stepping_obj_pos[2] = stepping_obj_pos[2] + table_bbox_max[2] - min_aabb[2] + 0.001 
            p.resetBasePositionAndOrientation(stepping_stone_id, stepping_obj_pos, [0, 0, 0, 1], physicsClientId=env.id) 
            new_pos[2] = init_pos[2] + 0.42

            # reload the object
            p.removeBody(object_id, physicsClientId=env.id)
            object_id = p.loadURDF(env.urdf_paths[object_name], basePosition=new_pos, baseOrientation=new_orient, useFixedBase=False, globalScaling=env.simulator_sizes[object_name], physicsClientId=env.id)
            for _ in range(10):
                p.stepSimulation()

        else:
            print("new pos: ", new_pos)
            print("new orient: ", p.getEulerFromQuaternion(new_orient))
            p.resetBasePositionAndOrientation(object_id, new_pos, new_orient, physicsClientId=env.id)


        
        # ensure the handle is not too far away from the robot base
        info = env._get_info()
        handle_joint_id = env.handle_joint
        handle_pos = info['handle_pos']
        robot_base = p.getBasePositionAndOrientation(env.robot.body, physicsClientId=env.id)[0]
        distance_base_to_handle = np.linalg.norm(handle_pos - np.array(robot_base))
        if env.robot_name == 'xarm' and distance_base_to_handle > 1.0:
            continue

        # randomly set the joint angle of the handle
        joint_limit_low, joint_limit_high = p.getJointInfo(object_id, handle_joint_id, physicsClientId=env.id)[8:10]
        max_opened_joint = joint_limit_low + 0.2 * (joint_limit_high - joint_limit_low)
        random_joint = np.random.uniform(joint_limit_low, max_opened_joint)
        p.resetJointState(object_id, handle_joint_id, random_joint, physicsClientId=env.id)

        for test_time in range(100):
            # randomly sample the joint angles of the robot arm
            for i in range(7):
                initial_joint_angles[i] = np.random.uniform(low[i], high[i])   
            if not mobile:
                env.robot.set_joint_angles(env.robot.right_arm_joint_indices, initial_joint_angles)
            else:
                initial_joint_angles = [0 for _ in range(3)] + initial_joint_angles
                env.robot.set_joint_angles(env.robot.right_arm_joint_indices, initial_joint_angles)

            # add a few steps to let the robot arm settle down
            for _ in range(5):
                p.stepSimulation()
                
            # check if the robot arm is in contact with the object
            contact_points = p.getContactPoints(env.robot.body, object_id, physicsClientId=env.id)
            closest_points = p.getClosestPoints(env.robot.body, object_id, distance=0.01, physicsClientId=env.id)
            if len(contact_points) > 0 or len(closest_points) > 0:
                continue

            # check if the robot arm is in contact with the plane
            link_contact = False
            num_links = p.getNumJoints(env.robot.body, physicsClientId=env.id)
            for link_idx in range(1, num_links):
                contact_points = p.getClosestPoints(bodyA=env.robot.body, linkIndexA=link_idx, bodyB=env.urdf_ids['plane'], distance=0.01, physicsClientId=env.id)
                if len(contact_points) > 0:
                    link_contact = True
                    break
            if link_contact: 
                continue
            
            # check if the robot end effector is in a good position to grasp the handle
            robot_eef_pos, _ = env.robot.get_pos_orient(env.robot.right_end_effector)
            distance = np.linalg.norm(handle_pos - robot_eef_pos)
            if distance < far_distance and distance > near_distance:
                # print("good init pos")
                good_init_pos = True
                new_pos = p.getBasePositionAndOrientation(object_id, physicsClientId=env.id)[0]
                break
            # print("bad init pos")
        
    if not good_init_pos:
        # fail to find a good initial position
        q.put(False)
        return False

    # found a good initial position
    # randomly sample the initial finger angle of the robot gripper
    initial_finger_angle = np.random.uniform(env.robot.finger_fully_close_joint_angle, env.robot.finger_fully_open_joint_angle)
    
    
    # add a few steps to let the robot gripper settle down
    env.robot.set_gripper_open_position(env.robot.right_gripper_indices, [initial_finger_angle, initial_finger_angle], set_instantly=True)
    p.stepSimulation()

    # save the initial state
    config = yaml.safe_load(open(config_path, "r"))
    for config_dict in config:
        if 'center' in config_dict:
            if on_table:
                table_xy_range = table_bbox_max[:2] - table_bbox_min[:2]
                obj_x = (new_pos[0] - table_bbox_min[0]) / table_xy_range[0]
                obj_y = (new_pos[1] - table_bbox_min[1]) / table_xy_range[1]
                obj_z = 0.0 if not step_stone else new_pos[2]-init_pos[2]+0.01
                new_pos = [obj_x, obj_y, obj_z]
            else:
                new_pos = [new_pos[0], new_pos[1], height]
            config_dict['center'] = str(tuple(new_pos))
            config_dict['orientation'] = str(tuple(new_orient))
            config_dict['init_angle'] = init_angle
            config_dict['is_crop_size'] = False
            if 'initial_joint_angles' not in config_dict:
                config_dict['initial_joint_angles'] = str(tuple(initial_joint_angles))
            if 'initial_finger_angle' not in config_dict:
                config_dict['initial_finger_angle'] = initial_finger_angle
        if "set_joint_angle_object_name" in config_dict:
            config_dict['set_joint_angle_object_name'] = object_name
            config_dict['set_joint_angle_joint_id'] = handle_joint_id
            config_dict['set_joint_angle_joint_angle'] = random_joint
        if 'initial_joint_angles' in config_dict:
            config_dict['initial_joint_angles'] = str(tuple(initial_joint_angles))
        if 'initial_finger_angle' in config_dict:
            config_dict['initial_finger_angle'] = initial_finger_angle

    if step_stone:
        stepping_stone = {}
        stepping_stone['all_uid'] = ["f9a7942ee5894152b73b72ce83ac9ee5"]
        stepping_stone["center"] = str(tuple([obj_x, obj_y, 0.0]))
        stepping_stone['lang'] = "stepping stone"
        stepping_stone['movable'] = False
        stepping_stone['name'] = "stepping_stone"
        stepping_stone["on_table"] = True
        stepping_stone["path"] = "stepping_stone.obj"
        stepping_stone["size"] = 0.7
        stepping_stone["type"] = "mesh"
        stepping_stone["uid"] = ["f9a7942ee5894152b73b72ce83ac9ee5"]
        config.append(stepping_stone)

    with open(config_path, 'w') as f:
        yaml.dump(config, f, indent=4)
    env.close()
    q.put(True)
    return True
# handle memory leak             
def gen_init_state(config_path, env_name, render, mobile=False, step_stone=False, far_distance=0.7, near_distance=0.3, invert=False):
    q = mp.Queue()  
    p = mp.Process(target=_gen_init_state, args=(q, config_path, env_name, render, mobile, step_stone, far_distance, near_distance, invert))
    p.start()
    p.join()
    success = q.get()
    return success

def _execute(q, config_path, env_name, solution_path, experiment_path, time_string=None, invert=False):
    # get time string
    if time_string is None:
        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')

    if not os.path.exists(solution_path):
        os.makedirs(solution_path)

    experiment_path = os.path.join(experiment_path, time_string)
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    os.system("cp {} {}".format(config_path, os.path.join(experiment_path, "task_config.yaml")))

    config = yaml.safe_load(open(config_path, "r"))
    link_name = 'link_0'
    for config_dict in config:
        if 'name' in config_dict:
            object_name = config_dict['name'].lower()
        if 'link_name' in config_dict:
            link_name = config_dict['link_name']
    
    # execute primitive
    print("execute primitive")
    env, _ = build_up_env(config_path, env_name)
    env.primitive_save_path = experiment_path
    np.random.seed(time.time_ns() % 2**32)

    env.reset()
    rgbs, states = env.execute(invert=invert)
    p.disconnect(env.id)
    # execute the primitive from the environment to get the trajectory.
    if len(states) > 10:
        with open(os.path.join(experiment_path, "last_state_files.txt"), 'w') as f:
            f.write("\n".join(str(states[-1])))
        save_numpy_as_gif(np.array(rgbs), "{}/{}.gif".format(experiment_path, "all"))
        q.put(True)
        return True
    else:
        q.put(False)
        return False
# handle memory leak
def execute(config_path, env_name, solution_path, experiment_path, invert=False):
    q = mp.Queue()  
    p = mp.Process(target=_execute, args=(q, config_path, env_name, solution_path, experiment_path, None, invert))
    p.start()
    p.join()
    success = q.get()
    return success

parser = argparse.ArgumentParser(description="Generate demos.")
parser.add_argument("--exp_name", type=str, default="debug") # demo name: 0730-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
parser.add_argument("--env_name", type=str, default="articulated") # env name: storage_furniture
parser.add_argument("--extract_name", type=str, default=None) # exp folder: open_the_door_46537
parser.add_argument("--num_to_generate", type=int, default=100)
parser.add_argument("--max_try_times", type=int, default=500)
parser.add_argument("--root_dir", type=str, default=None) # folder name: data/diverse_objects_rest/
parser.add_argument("--robot_name", type=str, default="panda") # folder name: data/diverse_objects_rest/
parser.add_argument("--render", type=int, default=0) # folder name: data/diverse_objects_rest/

parser.add_argument("--use_augmented_handle", type=int, default=0)
parser.add_argument("--num_augmented_handle", type=int, default=0)
parser.add_argument("--log_path", type=str, default="./log.txt") 
parser.add_argument("--invert", action='store_true')


args = parser.parse_args()
root_dir = args.root_dir
extract_name = args.extract_name
exp_name = args.exp_name
log_path = args.log_path
env_name = args.env_name

near_distance = 0.15
far_distance = 0.35 if args.robot_name == "xarm" else 0.5

generate_times = 1 if not args.use_augmented_handle else args.num_augmented_handle

mobile = False

path = "{}/{}/experiment/{}".format(root_dir, extract_name, exp_name)

all_config_paths, all_solution_paths, all_obj_ids = get_all_test_configs(root_dir=root_dir, extract_name=extract_name)
num_demos = get_current_num_demos(path)

for i in range(generate_times):
    done = False
    try_times = 0
    while True:
        for config_path, solution_path, obj_id in zip(all_config_paths, all_solution_paths, all_obj_ids):
            if try_times >= 20 and num_demos == 0:
                done = True
                # too many tries, fail to generate demo
                break
                    
            if try_times >= args.max_try_times:
                done = True
                # too many tries, skip 
                break
                        
            if num_demos >= args.num_to_generate:
                print("Generated enough demos: ", num_demos)
                done = True
                # generated enough demos
                break
            
            
            if args.use_augmented_handle:
                obj_id = str(obj_id + f"_{i}")

            extract_handle_mesh(obj_id)
            # create experiment folder, meta info and config variant

            experiment_path = os.path.join(solution_path, "experiment", exp_name)
            if not os.path.exists(experiment_path):
                os.makedirs(experiment_path)
            
            meta_info_path = os.path.join(experiment_path, "meta_info.json")
            with open(meta_info_path, 'w') as f:
                json.dump(args.__dict__, f, indent=4)

            new_config_path = create_config_variant(config_path)

            # generate initial state
            success = gen_init_state(new_config_path, env_name, args.render, mobile=mobile, step_stone=False, invert=args.invert)

            if not success:
                continue

            sucess = execute(new_config_path, env_name, solution_path, experiment_path, invert=args.invert)
            try_times += 1
            if sucess:
                num_demos += 1
                cprint("generated demo: {}".format(num_demos), color="cyan")
            
        if done:
            break