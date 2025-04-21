import numpy as np
import yaml
import os
import time
from gpt_4.prompts.prompt_from_description import generate_from_task_name
from pprint import pprint
from execute import execute
from all_obj_with_handle import all_obj_with_handle
from termcolor import cprint
from gpt_4.prompts.utils import save_another_yaml
import copy
import datetime


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

def main(index_min, index_max, object_name, run_times, train_minutes, visible_gpu):
    object_name_low = object_name.lower()

    for i, id in enumerate(all_obj_with_handle):
        if i < index_min or i > index_max:
            continue
        
        meta_path = f"demos/{object_name_low}_{id}"
        generated = False
        if os.path.exists('data/' + meta_path):
            dir = os.listdir("data/" + meta_path)[0]
            all_files = os.listdir("data/" + meta_path + "/" + dir)
            yaml_files = [f for f in all_files if f.endswith(".yaml")]
            num_generated = len(yaml_files)
            if len(yaml_files) > 0:
                generated = True
                config_path = "data/" + meta_path + "/" + dir + "/" + yaml_files[0]
                solution_path = None
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                    solution_path = [obj["solution_path"] for obj in config if 'solution_path' in obj][0]
        
        if not generated:
            config_path, solution_path = generate_from_task_name(
                f"open the door of the {object_name_low}",
                f"{object_name}",
                id,
                temperature_dict,
                model_dict, 
                meta_path=meta_path,
                random_initialization=True,
            )
            num_generated = 1
            
        import pdb; pdb.set_trace()
        
        os.system(f"python manipulation/scripts/extract_handle_mesh.py --category {object_name} --obj_id {id}")
        all_substeps_path = os.path.join(solution_path, "substeps.txt")
        with open(all_substeps_path, "r") as f:
            substeps = f.readlines()
            first_step = substeps[0].lstrip().rstrip()
        num_sub_steps = len(substeps)
        skip_argument = "0 " + " ".join(["1" for _ in range(1, num_sub_steps)])
        
        ori_config_path = copy.deepcopy(config_path)
        for run_idx in range(run_times):
            print(f"Running {run_idx}th time")
            new_config_path = ori_config_path.replace(".yaml", f"_{run_idx + num_generated}.yaml")
            import pdb; pdb.set_trace()
            save_another_yaml(config_path, new_config_path)
            config_path = new_config_path
            
            ts = time.time()
            time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')            
            os.system("CUDA_VISIBLE_DEVICES={} python execute.py --task_config_path {} --gui 0 --skip {} --time_string {}".format(
                visible_gpu, config_path, skip_argument, time_string))

            experiment_path = os.path.join(solution_path, "experiment")
            # all_experiments = os.listdir(experiment_path)
            # all_experiments = sorted(all_experiments)
            # newest_experiment = all_experiments[-1]
            # newest_experiment_path = os.path.join(experiment_path, newest_experiment)
            newest_experiment_path = os.path.join(experiment_path, time_string)

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

            time_limit = train_minutes * 60

            chechpoints_dir = os.path.join(second_save_path, "checkpoints")
            if not os.path.exists(chechpoints_dir):
                os.makedirs(chechpoints_dir)
            states_dir = os.path.join(second_save_path, "states")
            if not os.path.exists(states_dir):
                os.makedirs(states_dir)
            import pdb; pdb.set_trace()

            # generate open door demonstration using pytorch_sac
            # cprint(f"CUDA_VISIBLE_DEVICES={visible_gpu} python pytorch_sac/train.py task_config_path={config_path}  solution_path={solution_path} substep={second_step} final_state_path={final_state_path} rl_save_path={second_save_path} time_limit={time_limit} seed={i}", "green")
            # os.system(f"CUDA_VISIBLE_DEVICES={visible_gpu} python pytorch_sac/train.py task_config_path={config_path}  solution_path={solution_path} substep={second_step} final_state_path={final_state_path} rl_save_path={second_save_path} time_limit={time_limit} seed={i}")

            # generate open door demonstration using rl_games PPO
            os.system(f"CUDA_VISIBLE_DEVICES={visible_gpu} python rl_games/rl_games/rl_game_learn.py is_train=True is_play=False \
                        params.config.env_config.task_config_path={config_path} \
                        params.config.env_config.solution_path={solution_path} \
                        params.config.env_config.substep={second_step} \
                        params.config.env_config.final_state_path={final_state_path} \
                        params.config.env_config.rl_save_path={second_save_path} \
                        params.config.env_config.time_limit={time_limit} \
                        params.seed={i}") 
            saved_checkpoint = os.path.join(second_save_path, "checkpoints", "rl_game_PPO.pth")
            os.system(f"CUDA_VISIBLE_DEVICES={visible_gpu} python rl_games/rl_games/rl_game_learn.py is_train=False is_play=True checkpoint={saved_checkpoint} \
                        params.config.env_config.task_config_path={config_path} \
                        params.config.env_config.solution_path={solution_path} \
                        params.config.env_config.substep={second_step} \
                        params.config.env_config.final_state_path={final_state_path} \
                        params.config.env_config.rl_save_path={second_save_path} \
                        params.config.env_config.time_limit={time_limit}")
                
            # generate open door demonstration using CEM Policy
            # os.system(f"python CEM_policy.py \
            #             --task_config_path {config_path} \
            #             --solution_path {solution_path} \
            #             --substep {second_step} \
            #             --final_state_path {final_state_path} \
            #             --save_path {second_save_path}")

from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--object_name", type=str, default="StorageFurniture")
    parser.add_argument("--index_min", type=int, default=0)
    parser.add_argument("--index_max", type=int, default=1)
    parser.add_argument("--run_times", type=int, default=1)
    parser.add_argument("--train_minutes", type=int, default=45)
    parser.add_argument("--visible_gpu", type=str, default=0)
    args = parser.parse_args()
    main(args.index_min, args.index_max, args.object_name, args.run_times, 
         args.train_minutes, args.visible_gpu,)