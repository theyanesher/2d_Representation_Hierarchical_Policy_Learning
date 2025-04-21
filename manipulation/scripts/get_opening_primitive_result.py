import os
import numpy as np
import yaml
from pprint import pprint

grasping_scores = {}
opening_scores = {}

meta_path = "data/generated_task_from_description"
all_tasks = os.listdir(meta_path)
all_tasks = sorted(all_tasks)
for task in all_tasks:
    print("processing task: ", task)
    task_path = os.path.join(meta_path, task)
    # import pdb; pdb.set_trace()
    yaml_config = [x for x in os.listdir(task_path) if x.endswith(".yaml")]
    if len(yaml_config) == 0:
        continue
    yaml_config = yaml_config[0]
    config_path = os.path.join(task_path, yaml_config)
    config = yaml.safe_load(open(config_path, "r"))
    solution_path = [x['solution_path'] for x in config if 'solution_path' in x][0]
    reward_asset_path = [x['reward_asset_path'] for x in config if 'reward_asset_path' in x][0]
    
    experiment_path = os.path.join(solution_path, "experiment")
    all_experiments = os.listdir(experiment_path)
    all_experiments = sorted(all_experiments)
    newest_experiment = all_experiments[-1]
    newest_experiment_path = os.path.join(experiment_path, newest_experiment)
    
    all_substeps_path = os.path.join(solution_path, "substeps.txt")
    with open(all_substeps_path, "r") as f:
        all_substeps = f.readlines()
        first_step = all_substeps[0].lstrip().rstrip()
    
    all_substeps_type = os.path.join(solution_path, "substep_types.txt")
    with open(all_substeps_type, "r") as f:
        all_substeps_type = f.readlines()
        first_step_type = all_substeps_type[0].lstrip().rstrip()
    first_step_folder = first_step.replace(" ", "_") + "_" + first_step_type
    first_step_folder_path = os.path.join(newest_experiment_path, first_step_folder)
    
    score_file = os.path.join(first_step_folder_path, "best_score.txt")
    angle_file = os.path.join(first_step_folder_path, "opened_angle.txt")
    if os.path.exists(score_file) == False or os.path.exists(angle_file) == False:
        continue
    
    with open(score_file, "r") as f:
        score = f.readlines()
        score = float(score[0].lstrip().rstrip())
        grasping_scores[reward_asset_path] = (score)
    with open(angle_file, "r") as f:
        angle = f.readlines()
        opened_angle = float(angle[0].lstrip().rstrip())
        angle_low_limit = float(angle[1].lstrip().rstrip())
        angle_high_limit = float(angle[2].lstrip().rstrip())
        opening_scores[reward_asset_path] = ((opened_angle - angle_low_limit) / (angle_high_limit - angle_low_limit))   
        
# pprint(grasping_scores)
# pprint(opening_scores)

print("=============== handle_grasping_scores =============")
for key in grasping_scores.keys():
    if grasping_scores[key] == 0:
        print(key, grasping_scores[key], opening_scores[key])
        
print("=============== opened_angles =============")
for key in opening_scores.keys():
    if opening_scores[key] == 0:
        print(key, grasping_scores[key], opening_scores[key])