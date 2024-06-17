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

def get_all_test_configs():
    path = "data/temp"
    all_tasks = os.listdir(path)
    all_tasks = sorted(all_tasks)
    yaml_configs = []
    solution_paths = []
    reward_assets = []
    for task in all_tasks:
        path = os.path.join("data/temp", task)
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
            if 'center' in obj:
                obj['center'] = "[0.7, 0, 0]"
            
        with open(config_path, "w") as f:
            yaml.dump(new_config, f)    
        solution_path = solution_path.replace("data/sac_storagefurniture/", "data/temp/")
        yaml_configs.append(config_path)
        solution_paths.append(solution_path)
        
    return yaml_configs, solution_paths, reward_assets

temperature_dict = {
        "reward": 0,
        "yaml": 0,
        "size": 0,
        "joint": 0,
        "spatial_relationship": 0,
    }
    
model_dict = {
    "reward": "gpt-4",
    "yaml": "gpt-4",
    "size": "gpt-4",
    "joint": "gpt-4",
    "spatial_relationship": "gpt-4",
}

USE_STEPPING_STONE = False

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default="debug")
parser.add_argument("--beg_idx", type=int, default=12)
parser.add_argument("--end_idx", type=int, default=13)
parser.add_argument("--near_distance", type=float, default=0.15)
parser.add_argument("--far_distance", type=float, default=0.4)

args = parser.parse_args()

all_config_paths, all_solution_paths, reward_assets = get_all_test_configs()
beg_idx = args.beg_idx
end_idx = args.end_idx
all_config_paths = all_config_paths[beg_idx:end_idx]
all_solution_paths = all_solution_paths[beg_idx:end_idx]
reward_assets = reward_assets[beg_idx:end_idx]

# exp_name = "0502-vary-obj-init-angle-robot-init-joint-near-handle-larger"
# exp_name = "0504-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-100-demo"
# exp_name = "0505-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-100-demo"
# exp_name = "0509-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo"
# args.exp_name = "0511-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
args.exp_name = "0613-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-120-demo-0.4-0.15-translation-first"
args.exp_name = "eval_45410"

exp_name = args.exp_name
try_times_min = 0
try_times_max = 10

for try_idx in range(try_times_min, try_times_max):
    for config_path, solution_path, obj_id in zip(all_config_paths, all_solution_paths, reward_assets):
        
        object_name = "storagefurniture"
        os.system(f"python manipulation/scripts/extract_handle_mesh.py --category {object_name} --obj_id {obj_id}")
        
        # config_path, solution_path = generate_from_task_name(
        #             "open the door of the dishwasher", 
        #             "Dishwasher", 
        #             dishwasher_id, 
        #             temperature_dict,
        #             model_dict)
        # config_path, solution_path = get_folders_from_id(dishwasher_id)

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
        
        # if os.path.exists(experiment_path):
        #     all_experiments = os.listdir(experiment_path)
        #     all_experiments = sorted(all_experiments)
        #     newest_experiment = all_experiments[-1]
        #     newest_experiment_path = os.path.join(experiment_path, newest_experiment)
            
            
        #     all_substeps_type = os.path.join(solution_path, "substep_types.txt")
        #     with open(all_substeps_type, "r") as f:
        #         all_substeps_type = f.readlines()
        #         first_step_type = all_substeps_type[0].lstrip().rstrip()
        #     first_step_folder = first_step.replace(" ", "_") + "_" + first_step_type
        #     first_step_folder_path = os.path.join(newest_experiment_path, first_step_folder)
            
        #     score_file = os.path.join(first_step_folder_path, "best_score.txt")
        #     if os.path.exists(score_file):
        #         trained = True
        
        config_variant_paths = os.path.join("/".join(config_path.split("/")[:-1]), "configs")
        if not os.path.exists(config_variant_paths):
            os.makedirs(config_variant_paths)
        new_config_path = os.path.join(config_variant_paths, f"config_{try_idx}.yaml")
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
        # all_handle_pos, handle_joint_id = get_handle_pos(env, object_name, return_median=False)
        # handle_median_points = np.array([np.median(handle_pos, axis=0) for handle_pos in all_handle_pos]).reshape(-1, 3)
        # link_name = "link_0"
        # link_name = link_name.lower()
        # link_pc = get_link_pc(env, object_name, link_name)
        # distance_handle_median_to_link_pc = scipy.spatial.distance.cdist(handle_median_points, link_pc)
        # min_distance = np.min(distance_handle_median_to_link_pc, axis=1)
        # min_distance_handle_idx = np.argmin(min_distance)
        # handle_pos = handle_median_points[min_distance_handle_idx]
        # handle_joint_id = handle_joint_id[min_distance_handle_idx]

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
            # new_orient = p.getQuaternionFromEuler([init_euler[0], init_euler[1], np.random.uniform(-np.pi / 6, np.pi / 6)])
            new_orient = p.getQuaternionFromEuler([init_euler[0], init_euler[1], np.random.uniform(0, np.pi / 6)])
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

                env.robot.set_joint_angles(env.robot.right_arm_joint_indices, initial_joint_angles)
                for _ in range(5):
                    p.stepSimulation()
                
                contact_points = p.getContactPoints(env.robot.id, object_id, physicsClientId=env.id)
                if len(contact_points) > 0:
                    print("fail due to contact")
                    continue
                
                robot_eef_pos, robot_eef_orient = env.robot.get_pos_orient(env.robot.right_end_effector)
                distance = np.linalg.norm(handle_pos - robot_eef_pos)
                print("dsitance: ", distance)
                print("eef pos: ", robot_eef_pos)
                print("handle pos: ", handle_pos)
                if distance < args.far_distance and distance > args.near_distance:
                    good_config = True
                    new_pos = p.getBasePositionAndOrientation(object_id, physicsClientId=env.id)[0]
                    break
            
        new_config = copy.deepcopy(base_config)
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
            
        if not trained:
            os.system("python execute.py --task_config_path {} --gui 0 --skip {} --exp_name {}".format(
                new_config_path, skip_argument, exp_name
            ))
        
        end_time = time.time()
        
    #     all_experiments = os.listdir(experiment_path)
    #     all_experiments = sorted(all_experiments)
    #     newest_experiment = all_experiments[-1]
    #     newest_experiment_path = os.path.join(experiment_path, newest_experiment)
        
        
    #     all_substeps_type = os.path.join(solution_path, "substep_types.txt")
    #     with open(all_substeps_type, "r") as f:
    #         all_substeps_type = f.readlines()
    #         first_step_type = all_substeps_type[0].lstrip().rstrip()
    #     first_step_folder = first_step.replace(" ", "_") + "_" + first_step_type
    #     first_step_folder_path = os.path.join(newest_experiment_path, first_step_folder)
        
    #     score_file = os.path.join(first_step_folder_path, "best_score.txt")
    #     angle_file = os.path.join(first_step_folder_path, "opened_angle.txt")
    #     with open(score_file, "r") as f:
    #         score = f.readlines()
    #         score = float(score[0].lstrip().rstrip())
    #         # handle_grasping_scores[dishwasher_id] = (score)
    #         handle_grasping_scores.append(score)
    #     with open(angle_file, "r") as f:
    #         angle = f.readlines()
    #         opened_angle = float(angle[0].lstrip().rstrip())
    #         opened_angles.append(opened_angle)

    # print("=============== opened angles =============")
    # print(opened_angles)
    # with open("data/opened_angles.yaml", "w") as f:
    #     yaml.dump(opened_angles, f)
        



 