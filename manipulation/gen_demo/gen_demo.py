import numpy as np
import yaml
import os
import time
from gpt_4.prompts.prompt_from_description import generate_from_task_name
import copy
from gpt_4.prompts.utils import save_another_yaml
from manipulation.utils import build_up_env
from manipulation.gpt_reward_api import get_handle_pos, get_link_pc
import scipy
import pybullet as p
import argparse
import json

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
            
        with open(config_path, "w") as f:
            yaml.dump(new_config, f)    
        solution_path = solution_path.replace("data/sac_storagefurniture/", "data/temp/")
        yaml_configs.append(config_path)
        solution_paths.append(solution_path)
        
    return yaml_configs, solution_paths, reward_assets

USE_STEPPING_STONE = False

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default="debug")
parser.add_argument("--extract_name", type=str, default=None)
parser.add_argument("--near_distance", type=float, default=0.15)
parser.add_argument("--far_distance", type=float, default=0.4)
parser.add_argument("--num_to_generate", type=int, default=5)
parser.add_argument("--max_try_times", type=int, default=10)

args = parser.parse_args()

root_dir = "data/diverse_objects_2"
all_config_paths, all_solution_paths, reward_assets = get_all_test_configs(root_dir, args.extract_name)

# args.exp_name = "0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
args.exp_name = "0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
args.exp_name = "debug"

exp_name = args.exp_name
mobile = False

generated_demo = 0
try_times = 0
while True: 
    for config_path, solution_path, obj_id in zip(all_config_paths, all_solution_paths, reward_assets):
                
        object_name = "storagefurniture"
        os.system(f"python manipulation/scripts/extract_handle_mesh.py --category {object_name} --obj_id {obj_id}")
        
        all_substeps_path = os.path.join(solution_path, "substeps.txt")
        with open(all_substeps_path, "r") as f:
            all_substeps = f.readlines()
            first_step = all_substeps[0].lstrip().rstrip()
        num_sub_steps = len(all_substeps)
        skip_argument = "0 " + " ".join(["1" for i in range(1, num_sub_steps)])

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
        new_config_path = os.path.join(config_variant_paths, f"config_{try_times}.yaml")
        base_config = yaml.safe_load(open(config_path, "r"))
        all_substeps_path = os.path.join(solution_path, "substeps.txt")
        with open(all_substeps_path, "r") as f:
            substeps = f.readlines()
            first_step = substeps[0].lstrip().rstrip()
        
        env, _ = build_up_env(config_path, solution_path, first_step.replace(" ", "_"), None, 
                            render=False)
        env.reset()

        on_table = False
        for config_dict in base_config:
            if 'use_table' in config_dict:
                on_table = config_dict['use_table']
        print("use table", on_table)
        
        if on_table:
            table_bbox_min, table_bbox_max = env.table_bbox_min, env.table_bbox_max
        
        object_name = 'storagefurniture'
        info = env._get_info()
        handle_pos = info['handle_pos']
        handle_joint_id = env.handle_joint

        initial_joint_angles = [0 for _ in range(7)]
        low = [-2.9, -1.8, -2.9, -3.1, -2.9, -0.0, -2.9]
        high = [2.9, 1.8, 2.9, 0.0, 2.9, 3.8, 2.9]
        for i in range(7):
            joint_range = high[i] - low[i]
            low[i] += joint_range * 0.2
            high[i] -= joint_range * 0.2    

        object_id = env.urdf_ids[object_name]   
        init_pos, init_orient = p.getBasePositionAndOrientation(object_id, physicsClientId=env.id)
        init_euler = p.getEulerFromQuaternion(init_orient)

        good_config = False
        while not good_config:
            new_pos = np.array([0, 0, init_pos[2]])
            new_pos[0] = np.random.uniform(-0.1, 0.1) + init_pos[0]
            new_pos[1] = np.random.uniform(-0.1, 0.1) + init_pos[1]
            new_orient = p.getQuaternionFromEuler([init_euler[0], init_euler[1], np.random.uniform(-np.pi / 6, np.pi / 6)])
            # new_orient = p.getQuaternionFromEuler([init_euler[0], init_euler[1], np.random.uniform(0, np.pi / 6)])
            if USE_STEPPING_STONE:
                stepping_obj_pos = [new_pos[0], new_pos[1], table_bbox_max[2]]
                stepping_stone_id = p.loadURDF("objaverse_utils/data/obj/f9a7942ee5894152b73b72ce83ac9ee5/material.urdf", stepping_obj_pos, globalScaling=0.34313793694655315, physicsClientId=env.id)
                min_aabb, _ = p.getAABB(stepping_stone_id, physicsClientId=env.id)
                stepping_obj_pos[2] = stepping_obj_pos[2] + table_bbox_max[2] - min_aabb[2] + 0.001
                p.resetBasePositionAndOrientation(stepping_stone_id, stepping_obj_pos, [0, 0, 0, 1], physicsClientId=env.id)

                new_pos[2] = init_pos[2] + 0.42
                p.removeBody(object_id, physicsClientId=env.id)
                object_id = p.loadURDF(env.urdf_paths[object_name], basePosition=new_pos, baseOrientation=new_orient, useFixedBase=False, globalScaling=env.simulator_sizes[object_name], physicsClientId=env.id)
                # p.resetBasePositionAndOrientation(object_id, basePosition=new_pos, baseOrientation=new_orient, physicsClientId=env.id)
                for _ in range(10):
                    p.stepSimulation()
            else:
                p.resetBasePositionAndOrientation(object_id, new_pos, new_orient, physicsClientId=env.id)


            joint_limit_low, joint_limit_high = p.getJointInfo(object_id, handle_joint_id, physicsClientId=env.id)[8:10]
            max_opened_joint = joint_limit_low + 0.2 * (joint_limit_high - joint_limit_low)
            random_joint = np.random.uniform(joint_limit_low, max_opened_joint)
            p.resetJointState(object_id, handle_joint_id, random_joint, physicsClientId=env.id)
            
            for test_time in range(100):
                for i in range(7):
                    initial_joint_angles[i] = np.random.uniform(low[i], high[i])

                if not mobile:
                    env.robot.set_joint_angles(env.robot.right_arm_joint_indices, initial_joint_angles)
                else:
                    initial_joint_angles = [0 for _ in range(3)] + initial_joint_angles
                    env.robot.set_joint_angles(env.robot.right_arm_joint_indices, initial_joint_angles)
                for _ in range(5):
                    p.stepSimulation()
                
                contact_points = p.getContactPoints(env.robot.id, object_id, physicsClientId=env.id)
                if len(contact_points) > 0:
                    print("fail due to contact")
                    continue
                
                robot_eef_pos, robot_eef_orient = env.robot.get_pos_orient(env.robot.right_end_effector)
                distance = np.linalg.norm(handle_pos - robot_eef_pos)
                if distance < args.far_distance and distance > args.near_distance:
                    good_config = True
                    new_pos = p.getBasePositionAndOrientation(object_id, physicsClientId=env.id)[0]
                    break
                
        if not good_config:
            print("fail to find a good config")
            continue
            
        print("find good config")
        new_config = copy.deepcopy(base_config)
        
        for config_dict in new_config:
            if 'center' in config_dict:
                if on_table:
                    table_xy_range = table_bbox_max[:2] - table_bbox_min[:2]
                    obj_x = (new_pos[0] - table_bbox_min[0]) / table_xy_range[0]
                    obj_y = (new_pos[1] - table_bbox_min[1]) / table_xy_range[1]
                    obj_z = 0.0 if not USE_STEPPING_STONE else new_pos[2]-init_pos[2]+0.01
                    new_pos = [obj_x, obj_y, obj_z]
                else:
                    new_pos = [new_pos[0], new_pos[1], 0]
                config_dict['center'] = str(tuple(new_pos))
                config_dict['orientation'] = str(tuple(new_orient))
                if 'initial_joint_angles' not in config_dict:
                    config_dict['initial_joint_angles'] = str(tuple(initial_joint_angles))
            if "set_joint_angle_object_name" in config_dict:
                config_dict['set_joint_angle_object_name'] = object_name
                config_dict['set_joint_angle_joint_id'] = handle_joint_id
                config_dict['set_joint_angle_joint_angle'] = random_joint
            if 'initial_joint_angles' in config_dict:
                config_dict['initial_joint_angles'] = str(tuple(initial_joint_angles))

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
            with open('/project_data/held/yufeiw2/RoboGen_sim2real/data/local/gen_data.log', "a") as f:
                f.write("failed to generate demo for object {}\n".format(obj_id))
            exit()
                
        if try_times >= args.max_try_times:
            exit()
                    
        if generated_demo >= args.num_to_generate:
            exit()



 