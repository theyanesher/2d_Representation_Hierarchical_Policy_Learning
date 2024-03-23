import numpy as np
import yaml
import os
import time
from gpt_4.prompts.prompt_from_description import generate_from_task_name
from pprint import pprint

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

# run execute.py 
# get all dishwasher with a handle
all_dishwasher_ids = [
#  '12596',
#  '12565',
#  '12594',
#  '12480',
#  '12597',
#  '12606',
#  '12428',
 '12592',
#  '11700',
#  '12085',
#  '12590',
#  '12614',
#  '11622',
#  '12563',
#  '12579',
#  '12540', # data/dataset/12540/textured_objs/original-8.obj
#  '12605',
#  '12259',
#  '12587',
#  '12536',
#  '12562',
#  '12553',
#  '12559',
#  '12583',
#  '11661',
#  '12065',
#  '12552',
#  '12561',
#  '12531',
#  '12414',
#  '12580',
#  '12543',
#  '12530', # weird handle
#  '12560', # error in handle processing
#  '12484',
#  '12092',
#  '11826',
#  '12617'
]

all_time_costs = {}
handle_grasping_scores = {}
opened_angles = {}
for dishwasher_id in all_dishwasher_ids:
    
    print("=" * 50)
    print("running for dishwasher: ", dishwasher_id)
    print("=" * 50)
    
    # config_path, solution_path = generate_from_task_name(
    #             "open the door of the dishwasher", 
    #             "Dishwasher", 
    #             dishwasher_id, 
    #             temperature_dict,
    #             model_dict)
    
    # config_path = "data/generated_task_from_description/open_the_door_of_the_dishwasher_Dishwasher_12085_2024-03-19-01-27-53/open_the_door_of_the_dishwasher_The_robot_arm_opens_the_door_of_the_dishwasher.yaml"
    # solution_path = "data/generated_task_from_description/open_the_door_of_the_dishwasher_Dishwasher_12085_2024-03-19-01-27-53/task_open_the_door_of_the_dishwasher"

    config_path, solution_path = get_folders_from_id(dishwasher_id)

    all_substeps_path = os.path.join(solution_path, "substeps.txt")
    with open(all_substeps_path, "r") as f:
        all_substeps = f.readlines()
        first_step = all_substeps[0].lstrip().rstrip()
    num_sub_steps = len(all_substeps)
    skip_argument = "0 " + " ".join(["1" for i in range(1, num_sub_steps)])
    # skip_argument = [0] + [1 for i in range(1, num_sub_steps)]
    # import pdb; pdb.set_trace()

    # run execute.py
    beg_time = time.time()
    
    os.system("python execute.py --task_config_path {} --gui 1 --skip {}".format(config_path, skip_argument))
    # execute(config_path, 
    #         resume=args.resume, 
    #         training_algo=args.training_algo, 
    #         time_string=args.time_string, 
    #         gui=1, 
    #         randomize=args.randomize,
    #         use_bard=args.use_bard,
    #         use_gpt_size=args.use_gpt_size,
    #         use_gpt_joint_angle=args.use_gpt_joint_angle,
    #         use_gpt_spatial_relationship=args.use_gpt_spatial_relationship,
    #         run_training=args.run_training,
    #         obj_id=args.obj_id,
    #         use_motion_planning=args.use_motion_planning,
    #         use_distractor=args.use_distractor,
    #         skip=skip_argument,
    #         move_robot=args.move_robot,
    #         only_learn_substep=args.only_learn_substep,
    #         reward_learning_save_path=args.reward_learning_save_path,
    #         last_restore_state_file=args.last_restore_state_file
    # )
    
    end_time = time.time()
        
    all_time_costs[dishwasher_id] = (end_time - beg_time)
    
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
    angle_file = os.path.join(first_step_folder_path, "opened_angle.txt")
    with open(score_file, "r") as f:
        score = f.readlines()
        score = float(score[0].lstrip().rstrip())
        handle_grasping_scores[dishwasher_id] = (score)
    with open(angle_file, "r") as f:
        angle = f.readlines()
        opened_angle = float(angle[0].lstrip().rstrip())
        angle_low_limit = float(angle[1].lstrip().rstrip())
        angle_high_limit = float(angle[2].lstrip().rstrip())
        opened_angles[dishwasher_id] = ((opened_angle - angle_low_limit) / (angle_high_limit - angle_low_limit))   

print("=============== time cost =============")
pprint(all_time_costs)
print("=============== handle_grasping_scores =============")
pprint(handle_grasping_scores)
print("=============== opened_angles =============")
pprint(opened_angles)

with open("data/opened_angles.yaml", "w") as f:
    yaml.dump(opened_angles, f)
with open("data/handle_grasping_scores.yaml", "w") as f:
    yaml.dump(handle_grasping_scores, f)
with open("data/all_time_costs.yaml", "w") as f:
    yaml.dump(all_time_costs, f)
 