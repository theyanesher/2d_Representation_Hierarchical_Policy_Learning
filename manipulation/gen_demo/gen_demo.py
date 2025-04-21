import numpy as np
import yaml
import os
import time
# from gpt_4.prompts.prompt_from_description import generate_from_task_name
import copy
# from gpt_4.prompts.utils import save_another_yaml
from manipulation.utils import build_up_env
# from manipulation.gpt_reward_api import get_handle_pos, get_link_pc
import scipy
import pybullet as p
import argparse
import json
import matplotlib.pyplot as plt

def get_folders_from_id(id):
    meta_path = "data/generated_task_from_description"
    all_tasks = os.listdir(meta_path)
    all_tasks = sorted(all_tasks)
    folder = [x for x in all_tasks if id in x][0]
    task_path = os.path.join(meta_path, folder)
    yaml_config = [x for x in os.listdir(task_path) if x.endswith(".yaml")]
    yaml_config = yaml_config[0]
    config_path = os.path.join(task_path, yaml_config)
    config = yaml.safe_load(open(config_path, "r"))
    solution_path = [x['solution_path'] for x in config if 'solution_path' in x][0]
    return config_path, solution_path

def get_all_test_configs(root_dir='data/temp', extract_name=None):
    all_tasks = os.listdir(root_dir)
    all_tasks = sorted(all_tasks)
    yaml_configs = []
    solution_paths = []
    reward_assets = []
    if extract_name is not None:
        all_tasks = [x for x in all_tasks if extract_name in x]
        
    for task in all_tasks:
        path = os.path.join(root_dir, task)
        yaml_config = [x for x in os.listdir(path) if x.endswith(".yaml")]
        yaml_config_lengths = [len(x) for x in yaml_config]
        least_length = np.argmin(yaml_config_lengths)
        yaml_config = yaml_config[least_length]
        # select the config path with least length
        config_path = os.path.join(path, yaml_config)
        config = yaml.safe_load(open(config_path, "r"))
        solution_path = [x['solution_path'] for x in config if 'solution_path' in x][0]
        reward_assets.append([x['reward_asset_path'] for x in config if 'reward_asset_path' in x][0])
        new_config = copy.deepcopy(config)
        for obj in new_config:
            if 'solution_path' in obj:
                obj['solution_path'] = obj['solution_path'].replace("data/sac_storagefurniture/", "data/temp/")
        for obj in new_config:
            random_center = np.random.uniform(0.6, 0.7)
            if 'center' in obj:
                obj['center'] = f"[{random_center}, 0, 0]"
        #  modify the solution path and center for each obj. Question: what is the solution path and center?
            
        with open(config_path, "w") as f:
            yaml.dump(new_config, f)    
        # save the modified config
        solution_path = solution_path.replace("data/sac_storagefurniture/", "data/temp/")
        yaml_configs.append(config_path)
        solution_paths.append(solution_path)
        
    return yaml_configs, solution_paths, reward_assets
    # what is the return value? 

USE_STEPPING_STONE = False

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default="debug") # demo name: 0730-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
parser.add_argument("--extract_name", type=str, default=None) # exp folder: open_the_door_46537
parser.add_argument("--near_distance", type=float, default=0.15)
parser.add_argument("--far_distance", type=float, default=0.5)
parser.add_argument("--num_to_generate", type=int, default=200)
parser.add_argument("--max_try_times", type=int, default=75)
parser.add_argument("--root_dir", type=str, default=None) # folder name: data/diverse_objects_rest/
parser.add_argument("--robot_name", type=str, default="panda") # folder name: data/diverse_objects_rest/
parser.add_argument("--render", type=int, default=0) # folder name: data/diverse_objects_rest/

parser.add_argument("--use_augmented_handle", type=int, default=0)
parser.add_argument("--num_augmented_handle", type=int, default=0)

args = parser.parse_args()

root_dir = args.root_dir
all_config_paths, all_solution_paths, reward_assets = get_all_test_configs(root_dir, args.extract_name)
exp_name = args.exp_name
mobile = False

if args.robot_name == 'xarm':
    args.far_distance = 0.35

generated_demo = 0
num_generated = 0
new = "data/{}/{}/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/{}".format(args.root_dir, args.extract_name, args.exp_name)
if os.path.exists(new):
    all_experiments = os.listdir(new)
    all_experiments = sorted(all_experiments)
    for experiment in all_experiments:
        state_path = os.path.join(new, experiment, "grasp_the_handle_of_the_storage_furniture_door_primitive/states")
        if not os.path.exists(state_path): continue
        if len(os.listdir(state_path)) > 1 and os.path.exists(os.path.join(new, experiment, "all.gif")):
            num_generated += 1
    generated_demo = num_generated
# compute the number of generated demos
generate_times = 1
if args.use_augmented_handle:
    generate_times = args.num_augmented_handle
# what does use_augmented_handle mean?
for iii in range(generate_times):
    finish_one = False
    try_times = 0
    while True: 
        for config_path, solution_path, obj_id in zip(all_config_paths, all_solution_paths, reward_assets):
            # why reward_assets is recognized as obj_id?
            
            if args.use_augmented_handle:
                obj_id = str(obj_id) + f"_{iii}"
                    
            object_name = "storagefurniture"
            os.system(f"python manipulation/scripts/extract_handle_mesh.py --category {object_name} --obj_id {obj_id}")
            # add the handle mesh to the config?
            
            all_substeps_path = os.path.join(solution_path, "substeps.txt")
            with open(all_substeps_path, "r") as f:
                all_substeps = f.readlines()
                first_step = all_substeps[0].lstrip().rstrip()
            num_sub_steps = len(all_substeps)
            # each substep for one line 
            skip_argument = "0 " + " ".join(["1" for i in range(1, num_sub_steps)])
            # 0 1 1 1 ...

            # run execute.py
            beg_time = time.time()
            
            trained = False
            experiment_path = os.path.join(solution_path, "experiment", exp_name)
            if not os.path.exists(experiment_path):
                os.makedirs(experiment_path)
                
            meta_info_path = os.path.join(experiment_path, "meta_info.json")
            with open(meta_info_path, "w") as f:
                json.dump(args.__dict__, f, indent=4)
            
            config_variant_paths = os.path.join("/".join(config_path.split("/")[:-1]), "configs")
            if not os.path.exists(config_variant_paths):
                os.makedirs(config_variant_paths)
            
            if args.robot_name == 'panda':
                if not args.use_augmented_handle:
                    new_config_path = os.path.join(config_variant_paths, f"config_larger_randomization_{try_times+500}.yaml")
                else:
                    new_config_path = os.path.join(config_variant_paths, f"config_handle_augmentation_{iii}_{try_times}.yaml")
            elif args.robot_name == 'xarm':
                new_config_path = os.path.join(config_variant_paths, f"config_xarm_{try_times}.yaml")
                
            base_config = yaml.safe_load(open(config_path, "r"))
            all_substeps_path = os.path.join(solution_path, "substeps.txt")
            with open(all_substeps_path, "r") as f:
                substeps = f.readlines()
                first_step = substeps[0].lstrip().rstrip()
            # why run this command twice?

            new_config = copy.deepcopy(base_config)
            for config_dict in new_config:
                if 'size' in config_dict:
                    if args.robot_name == 'xarm':
                        size_ratio = np.random.uniform(0.65, 0.9)  ## xarm cannot handle too big objects
                    else:
                        size_ratio = np.random.uniform(0.7, 1.2) # make our refrigerator smaller    
                    config_dict['size'] = size_ratio * config_dict['size']
                    config_dict['is_crop_size'] = False
                    print("size ratio", size_ratio, "size: ", config_dict['size'])
                if 'reward_asset_path' in config_dict and args.use_augmented_handle:
                    config_dict['reward_asset_path'] = str(config_dict['reward_asset_path']) + f"_{iii}"
                    # update object id?
            with open(new_config_path, 'w') as f:
                yaml.dump(new_config, f, indent=4)
            
            print(new_config_path, solution_path)
            env, _ = build_up_env(new_config_path, solution_path, first_step.replace(" ", "_"), None, 
                                render=args.render)
            env.reset()

            # create the env based on the updated config

            on_table = False
            for config_dict in base_config:
                if 'use_table' in config_dict:
                    on_table = config_dict['use_table']
            print("use table", on_table)

            if on_table:
                table_bbox_min, table_bbox_max = env.table_bbox_min, env.table_bbox_max
            else:
                table_bbox_min, table_bbox_max = [0,0,0], [0,0,0]

            # get table bbox
            
            object_name = 'storagefurniture'

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

            # get the joint limits for the robot arm, using a smaller range

            object_id = env.urdf_ids[object_name]   
            init_pos, init_orient = p.getBasePositionAndOrientation(object_id, physicsClientId=env.id)
            init_euler = p.getEulerFromQuaternion(init_orient)

            # get the initial position and orientation of the object

            step_stone = False
            if USE_STEPPING_STONE:
                step_stone = True if np.random.uniform() > 0.4 else False

            # USE_STEPPING_STONE: use a stepping stone to lift the object up for the robot to grasp easily?

            good_config = False
            while not good_config:
                # for microwave, lift it up
                height = 0
                # height = np.random.uniform(0, 0.3)
                # ================================
                new_pos = np.array([0, 0, init_pos[2]])
                if env.robot_name == 'panda':
                    new_pos[0] = np.random.uniform(-0.2, 0.1) + init_pos[0]
                elif env.robot_name == 'xarm':
                    # new_pos[0] = np.random.uniform(-0.2, 0.1) + init_pos[0] - 0.3 # xarm has a lower reachability
                    new_pos[0] = np.random.uniform(0.65, 1.0) # xarm has a lower reachability
                new_pos[1] = np.random.uniform(-0.1, 0.1) + init_pos[1]
                new_pos[2] = height + init_pos[2]
            
                if env.robot_name == 'panda':
                    new_orient = p.getQuaternionFromEuler([init_euler[0], init_euler[1], np.random.uniform(-np.pi / 6, np.pi / 6)])
                elif env.robot_name == 'xarm':
                    new_orient = p.getQuaternionFromEuler([init_euler[0], init_euler[1], np.random.uniform(-np.pi / 7, np.pi / 7)])

                # randomly sample the position and orientation of the object
                
                if USE_STEPPING_STONE:
                    stepping_obj_pos = [new_pos[0], new_pos[1], table_bbox_max[2]]
                    stepping_stone_id = p.loadURDF("objaverse_utils/data/obj/f9a7942ee5894152b73b72ce83ac9ee5/material.urdf", stepping_obj_pos, globalScaling=0.34313793694655315, physicsClientId=env.id)
                    min_aabb, _ = p.getAABB(stepping_stone_id, physicsClientId=env.id) # getAABB : get axis-aligned bounding box of the object, returning the min and max coordinates
                    stepping_obj_pos[2] = stepping_obj_pos[2] + table_bbox_max[2] - min_aabb[2] + 0.001 # put the stepping stone on the table
                    p.resetBasePositionAndOrientation(stepping_stone_id, stepping_obj_pos, [0, 0, 0, 1], physicsClientId=env.id) # reset the position and orientation of the stepping stone

                    new_pos[2] = init_pos[2] + 0.42 # seems like the object is lifted up by 0.42m to be above the stepping stone?
                    p.removeBody(object_id, physicsClientId=env.id)
                    object_id = p.loadURDF(env.urdf_paths[object_name], basePosition=new_pos, baseOrientation=new_orient, useFixedBase=False, globalScaling=env.simulator_sizes[object_name], physicsClientId=env.id)
                    # p.resetBasePositionAndOrientation(object_id, basePosition=new_pos, baseOrientation=new_orient, physicsClientId=env.id)

                    # remove the object and load it again with the new position and orientation
                    for _ in range(10):
                        p.stepSimulation()
                    # add a few steps to let the object settle down, drop it to the stepping stone?
                else:
                    print("new pos", new_pos)
                    p.resetBasePositionAndOrientation(object_id, new_pos, new_orient, physicsClientId=env.id)

                info = env._get_info()
                handle_joint_id = env.handle_joint
                handle_pos = info['handle_pos']
                
                # p.addUserDebugPoints([handle_pos], [[1, 0, 0]], 10, 0, physicsClientId=env.id)
                
                robot_base = p.getBasePositionAndOrientation(env.robot.body, physicsClientId=env.id)[0]
                distance_base_to_handle = np.linalg.norm(handle_pos - np.array(robot_base))
                print(new_pos)
                print("handle distance to robot base {}".format(distance_base_to_handle))
                if env.robot_name == 'xarm' and distance_base_to_handle > 1.0:
                    continue
                # guarantee the handle is not too far away from the robot base
                object_AABB = p.getAABB(object_id, physicsClientId=env.id)
                min_aabb, max_aabb = object_AABB[0], object_AABB[1]
                distance_object_to_robot_base = np.linalg.norm(np.array(robot_base) - np.array(min_aabb))
                if env.robot_name == 'xarm' and distance_object_to_robot_base < 0.5:
                    continue
                # guarantee the object is not too close to the robot base, pos of the object is larger than the robot base?
                
                # import pdb; pdb.set_trace()

                joint_limit_low, joint_limit_high = p.getJointInfo(object_id, handle_joint_id, physicsClientId=env.id)[8:10]
                max_opened_joint = joint_limit_low + 0.2 * (joint_limit_high - joint_limit_low)
                random_joint = np.random.uniform(joint_limit_low, max_opened_joint)
                p.resetJointState(object_id, handle_joint_id, random_joint, physicsClientId=env.id)
                # random_joint = 0.1
                # randomly set the joint angle of the handle
                for test_time in range(100):
                    for i in range(7):
                        initial_joint_angles[i] = np.random.uniform(low[i], high[i])   
                        # randomly sample the joint angles of the robot arm

                    if not mobile:
                        env.robot.set_joint_angles(env.robot.right_arm_joint_indices, initial_joint_angles)
                    else:
                        initial_joint_angles = [0 for _ in range(3)] + initial_joint_angles
                        env.robot.set_joint_angles(env.robot.right_arm_joint_indices, initial_joint_angles)
                        # add the mobile base joint angles to the robot arm joint angles
                    for _ in range(5):
                        p.stepSimulation()
                        # add a few steps to let the robot arm settle down
                    
                    
                    contact_points = p.getContactPoints(env.robot.body, object_id, physicsClientId=env.id)
                    closest_points = p.getClosestPoints(env.robot.body, object_id, distance=0.01, physicsClientId=env.id)
                    if len(contact_points) > 0 or len(closest_points) > 0:
                        continue
                    # check if the robot arm is in contact with the object
                    link_contact = False
                    num_links = p.getNumJoints(env.robot.body, physicsClientId=env.id)
                    for link_idx in range(1, num_links):
                        contact_points = p.getClosestPoints(bodyA=env.robot.body, linkIndexA=link_idx, bodyB=env.urdf_ids['plane'], distance=0.01, physicsClientId=env.id)
                        if len(contact_points) > 0:
                            link_contact = True
                            break
                    if link_contact: continue
                    # check if the robot arm is in contact with the plane
                    
                    robot_eef_pos, robot_eef_orient = env.robot.get_pos_orient(env.robot.right_end_effector)
                    distance = np.linalg.norm(handle_pos - robot_eef_pos)
                    if distance < args.far_distance and distance > args.near_distance:
                        print("distance to handle", distance)
                        good_config = True
                        new_pos = p.getBasePositionAndOrientation(object_id, physicsClientId=env.id)[0]
                        break
                    # check if the robot end effector is in a good position to grasp the handle
                    
            if not good_config:
                print("fail to find a good config")
                continue
                
            print("find good config")
            
            # if np.random.uniform() < 0.5:
            #     initial_finger_angle = 0.002
            # else:
            #     initial_finger_angle = np.random.uniform(0, 0.04)
            initial_finger_angle = np.random.uniform(env.robot.finger_fully_close_joint_angle, env.robot.finger_fully_open_joint_angle)
            # randomly sample the initial finger angle of the robot gripper
            # initial_finger_angle = env.robot.finger_fully_open_joint_angle
            env.robot.set_gripper_open_position(env.robot.right_gripper_indices, [initial_finger_angle, initial_finger_angle], set_instantly=True)
            p.stepSimulation()
            # add a few steps to let the robot gripper settle down

            # print("initial joint angles", initial_joint_angles)
            # import pdb; pdb.set_trace()
            for config_dict in new_config:
                if 'center' in config_dict:
                    if on_table:
                        table_xy_range = table_bbox_max[:2] - table_bbox_min[:2]
                        obj_x = (new_pos[0] - table_bbox_min[0]) / table_xy_range[0]
                        obj_y = (new_pos[1] - table_bbox_min[1]) / table_xy_range[1]
                        obj_z = 0.0 if not USE_STEPPING_STONE else new_pos[2]-init_pos[2]+0.01
                        new_pos = [obj_x, obj_y, obj_z]
                    else:
                        new_pos = [new_pos[0], new_pos[1], height]
                    config_dict['center'] = str(tuple(new_pos))
                    config_dict['orientation'] = str(tuple(new_orient))
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

            if USE_STEPPING_STONE:
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
                new_config.append(stepping_stone)

            with open(new_config_path, 'w') as f:
                yaml.dump(new_config, f, indent=4)
            env.close()
            # save the modified config
                
            ts = time.time()
            import datetime
            time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
            os.system("python execute.py --task_config_path {} --gui 0 --skip {} --exp_name {} --time_string {}".format(
                new_config_path, skip_argument, exp_name, time_string
            ))
        
            try_times += 1
            save_state_dir = os.path.join(experiment_path, time_string, "grasp_the_handle_of_the_storage_furniture_door_primitive", "states")
            if os.path.exists(save_state_dir):
                all_states = os.listdir(save_state_dir)
                if len(all_states) > 10:
                    generated_demo += 1
                    
            if try_times >= 20 and generated_demo == 0:
                with open('data/ziyuw2/gen_data.log', "a") as f:
                    f.write("failed to generate demo for object {}\n".format(obj_id))
                finish_one = True
                # too many tries, fail to generate demo
                break
                    
            if try_times >= args.max_try_times:
                finish_one = True
                # too many tries, skip 
                break
                        
            if generated_demo >= args.num_to_generate:
                finish_one = True
                # generated enough demos
                break
            
        if finish_one:
            break




    