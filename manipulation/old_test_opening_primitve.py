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
                obj['center'] = "[0.5, 0.5, 0]"
            
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


all_time_costs = []
handle_grasping_scores = []
opened_angles = []

all_config_paths, all_solution_paths, reward_assets = get_all_test_configs()
beg_idx = 0
end_idx = 1
all_config_paths = all_config_paths[beg_idx:end_idx]
all_solution_paths = all_solution_paths[beg_idx:end_idx]
reward_assets = reward_assets[beg_idx:end_idx]

exp_name = "vary_robot_init_joint"
exp_name = "vary_robot_init_joint_near_handle"
exp_name = "0502-vary-obj-init-angle-robot-init-joint-near-handle-larger"
exp_name = "0504-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-100-demo"
try_times_min = 0
try_times_max = 100
for try_idx in range(try_times_min, try_times_max):
    for config_path, solution_path, obj_id in zip(all_config_paths, all_solution_paths, reward_assets):
        
        object_name = "Microwave"
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
        new_config_path = os.path.join(config_variant_paths, f"config_{try_idx}.yaml")
        base_config = yaml.safe_load(open(config_path, "r"))
        task_name = "grasp_the_microwave_door"
        env, _ = build_up_env(config_path, solution_path, task_name, None, 
                            render=False)
        env.reset()

        on_table = False
        for config_dict in base_config:
            if 'use_table' in config_dict:
                on_table = config_dict['use_table']
        print("use table", on_table)
        
        if on_table:
            table_bbox_min, table_bbox_max = env.table_bbox_min, env.table_bbox_max
        
        
        object_name = 'microwave'
        all_handle_pos, handle_joint_id = get_handle_pos(env, object_name, return_median=False)
        handle_median_points = np.array([np.median(handle_pos, axis=0) for handle_pos in all_handle_pos]).reshape(-1, 3)
        link_name = "link_0"
        link_name = link_name.lower()
        link_pc = get_link_pc(env, object_name, link_name)
        distance_handle_median_to_link_pc = scipy.spatial.distance.cdist(handle_median_points, link_pc)
        min_distance = np.min(distance_handle_median_to_link_pc, axis=1)
        min_distance_handle_idx = np.argmin(min_distance)
        handle_pos = all_handle_pos[min_distance_handle_idx]
        handle_joint_id = handle_joint_id[min_distance_handle_idx]

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
                distance = np.linalg.norm(np.mean(handle_pos, axis=0) - robot_eef_pos)
                print("distance", distance)
                # import pdb; pdb.set_trace()
                if distance < 0.7 and distance > 0.2:
                    good_config = True
                    break
            
        new_config = copy.deepcopy(base_config)

        
        for config_dict in new_config:
            if 'center' in config_dict:
                if on_table:
                    table_xy_range = table_bbox_max[:2] - table_bbox_min[:2]
                    obj_x = (new_pos[0] - table_bbox_min[0]) / table_xy_range[0]
                    obj_y = (new_pos[1] - table_bbox_min[1]) / table_xy_range[1]
                    new_pos = [obj_x, obj_y, 0]
                else:
                    new_pos = [new_pos[0], new_pos[1], 0]
                config_dict['center'] = str(tuple(new_pos))
                config_dict['orientation'] = str(tuple(new_orient))
                config_dict['initial_joint_angles'] = str(tuple(initial_joint_angles))
            if "set_joint_angle_object_name" in config_dict:
                config_dict['set_joint_angle_object_name'] = object_name
                config_dict['set_joint_angle_joint_id'] = handle_joint_id
                config_dict['set_joint_angle_joint_angle'] = random_joint

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
        



 