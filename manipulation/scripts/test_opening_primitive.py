import numpy as np
import yaml
import os
import time
from gpt_4.prompts.prompt_from_description import generate_from_task_name
import copy
from gpt_4.prompts.utils import save_another_yaml

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
        yaml_config = yaml_config[0]
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


all_time_costs = []
handle_grasping_scores = []
opened_angles = []

all_config_paths, all_solution_paths, reward_assets = get_all_test_configs()
beg_idx = 1
end_idx = 5
all_config_paths = all_config_paths[beg_idx:end_idx]
all_solution_paths = all_solution_paths[beg_idx:end_idx]
reward_assets = reward_assets[beg_idx:end_idx]

exp_name = "vary_robot_init_joint"
try_times_min = 0
try_times_max = 40
for try_idx in range(try_times_min, try_times_max):
    for config_path, solution_path, obj_id in zip(all_config_paths, all_solution_paths, reward_assets):
        
        object_name = "StorageFurniture"
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
        
        new_config_path = config_path.replace(".yaml", f"_{try_idx}.yaml")
        save_another_yaml(config_path, new_config_path,
                          randomize_orientation=False, randomize_position=False,
                          randomize_robot_joint_angle=True, randomize_size=False)
        if not trained:
            os.system("python execute.py --task_config_path {} --gui 0 --skip {} --exp_name {}".format(
                new_config_path, skip_argument, exp_name
            ))
        
        end_time = time.time()
                    
        all_experiments = os.listdir(experiment_path)
        all_experiments = sorted(all_experiments)
        newest_experiment = all_experiments[-1]
        newest_experiment_path = os.path.join(experiment_path, newest_experiment)
        
        
        all_substeps_type = os.path.join(solution_path, "substep_types.txt")
        with open(all_substeps_type, "r") as f:
            all_substeps_type = f.readlines()
            first_step_type = all_substeps_type[0].lstrip().rstrip()
        first_step_folder = first_step.replace(" ", "_") + "_" + first_step_type
        first_step_folder_path = os.path.join(newest_experiment_path, first_step_folder)
        
        score_file = os.path.join(first_step_folder_path, "best_score.txt")
        angle_file = os.path.join(first_step_folder_path, "opened_angle.txt")
        with open(score_file, "r") as f:
            score = f.readlines()
            score = float(score[0].lstrip().rstrip())
            # handle_grasping_scores[dishwasher_id] = (score)
            handle_grasping_scores.append(score)
        with open(angle_file, "r") as f:
            angle = f.readlines()
            opened_angle = float(angle[0].lstrip().rstrip())
            opened_angles.append(opened_angle)

    print("=============== opened angles =============")
    print(opened_angles)
    with open("data/opened_angles.yaml", "w") as f:
        yaml.dump(opened_angles, f)
        



 