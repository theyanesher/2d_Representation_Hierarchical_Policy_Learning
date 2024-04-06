import numpy as np
import yaml
import os
import time
from gpt_4.prompts.prompt_from_description import generate_from_task_name
from pprint import pprint
from execute import execute
# from all_obj_with_handle import all_obj_with_handle
from termcolor import cprint

import argparse
parser = argparse.ArgumentParser()

args = parser.parse_args([])

CUDA_VISIBLE_DEVICES=2


micrwave_with_handle = [
    # '7310', '7263', '7119', '7167'
    '7119'
]

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

# for i, id in enumerate(all_obj_with_handle):
for i, id in enumerate([0]):
    # if i > 11:
    #     break
    id = '48700'
    # config_path, solution_path = generate_from_task_name(
    #     "open the door of the storagefurniture by its handle",
    #     "StorageFurniture",
    #     id,
    #     temperature_dict,
    #     model_dict, 
    #     meta_path='storagefurniture_48700'
    # )
    # time.sleep(1)
    config_path = "data/storagefurniture_48700/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_48700_2024-04-03-20-53-11/open_the_door_of_the_storagefurniture_by_its_handle_The_robotic_arm_will_open_the_door_of_the_storage_furniture_by_its_handle.yaml"
    solution_path = "data/storagefurniture_48700/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_48700_2024-04-03-20-53-11/task_open_the_door_of_the_storagefurniture_by_its_handle"


    os.system(f"python manipulation/scripts/extract_handle_mesh.py --category StorageFurniture --obj_id {id}")
    all_substeps_path = os.path.join(solution_path, "substeps.txt")
    with open(all_substeps_path, "r") as f:
        substeps = f.readlines()
        first_step = substeps[0].lstrip().rstrip()
    num_sub_steps = len(substeps)
    skip_argument = "0 " + " ".join(["1" for _ in range(1, num_sub_steps)])
    os.system("CUDA_VISIBLE_DEVICES={} python execute.py --task_config_path {} --gui 1 --skip {}".format(CUDA_VISIBLE_DEVICES, config_path, skip_argument))

    run_opening = False
    if run_opening:
        experiment_path = os.path.join(solution_path, "experiment")
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
        if not os.path.exists(score_file):
            print("failed to grasp the handle")
            continue
        with open(score_file, "r") as f:
            score = f.readlines()
            score = float(score[0].lstrip().rstrip())
            if score < 0.5:
                print("failed to grasp the handle")
                continue

        state_folder = os.path.join(first_step_folder_path, "states")
        all_states = os.listdir(state_folder)
        all_states = sorted(all_states)
        # final_state = all_states[-1]
        # final_state_path = os.path.join(state_folder, final_state)
        max_n = -1  # Initialize with a value lower than any possible 'n'
        max_file = None

        for file_name in all_states:
            if file_name.startswith('state_') and file_name.endswith('.pkl'):
                try:
                    n = int(file_name.split('_')[1].split('.')[0])
                    if n > max_n:
                        max_n = n
                        max_file = file_name
                except ValueError:
                    # Handle cases where the filename doesn't match expected pattern
                    pass


        second_step = substeps[1].lstrip().rstrip()
        second_step = second_step.replace(" ", "_")
        final_state_path = os.path.join(state_folder, max_file)
        second_save_path = os.path.join(newest_experiment_path, second_step)
        if not os.path.exists(second_save_path):
            os.makedirs(second_save_path)

        time_limit = 60 * 60

        chechpoints_dir = os.path.join(second_save_path, "checkpoints")
        if not os.path.exists(chechpoints_dir):
            os.makedirs(chechpoints_dir)
        states_dir = os.path.join(second_save_path, "states")
        if not os.path.exists(states_dir):
            os.makedirs(states_dir)

        # generate open door demonstration using pytorch_sac
        cprint(f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} python pytorch_sac/train.py task_config_path={config_path}  solution_path={solution_path} substep={second_step} final_state_path={final_state_path} rl_save_path={second_save_path} time_limit={time_limit} seed={i}", "green")
        os.system(f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} python pytorch_sac/train.py task_config_path={config_path}  solution_path={solution_path} substep={second_step} final_state_path={final_state_path} rl_save_path={second_save_path} time_limit={time_limit} seed={i}")

        # generate open door demonstration using rl_games PPO
        # os.system(f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} python rl_games/rl_games/rl_game_learn.py is_train=True is_play=False \
        #             params.config.env_config.task_config_path={config_path} \
        #             params.config.env_config.solution_path={solution_path} \
        #             params.config.env_config.substep={second_step} \
        #             params.config.env_config.final_state_path={final_state_path} \
        #             params.config.env_config.rl_save_path={second_save_path} \
        #             params.config.env_config.time_limit={time_limit} \
        #             params.seed={i}") 
        # saved_checkpoint = os.path.join(second_save_path, "checkpoints", "rl_game_PPO.pth")
        # os.system(f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} python rl_games/rl_games/rl_game_learn.py is_train=False is_play=True checkpoint={saved_checkpoint} \
        #             params.config.env_config.task_config_path={config_path} \
        #             params.config.env_config.solution_path={solution_path} \
        #             params.config.env_config.substep={second_step} \
        #             params.config.env_config.final_state_path={final_state_path} \
        #             params.config.env_config.rl_save_path={second_save_path} \
        #             params.config.env_config.time_limit={time_limit}")
            
        # generate open door demonstration using CEM Policy
        # os.system(f"python CEM_policy.py \
        #             --task_config_path {config_path} \
        #             --solution_path {solution_path} \
        #             --substep {second_step} \
        #             --final_state_path {final_state_path} \
        #             --save_path {second_save_path}")