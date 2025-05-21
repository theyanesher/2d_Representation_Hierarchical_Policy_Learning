import json
import os

eval_result_path = 'data/0519_low_level_110-obj_finetune/'

bucket_tasks = [100435, 100441]
faucet_tasks = [149, 960, 991]
foldingchair_tasks = [100520, 100526]
laptop_tasks = [9748, 9912, 9960]
stapler_tasks = [102990, 103095]
toilet_tasks = [101320, 102620, 102621]
storagefurniture_tasks = [41510, 45448, 46462, 46732, 46801]
bucket_tasks = [str(i) for i in bucket_tasks]
faucet_tasks = [str(i) for i in faucet_tasks]
foldingchair_tasks = [str(i) for i in foldingchair_tasks]
laptop_tasks = [str(i) for i in laptop_tasks]
stapler_tasks = [str(i) for i in stapler_tasks]
toilet_tasks = [str(i) for i in toilet_tasks]
storagefurniture_tasks = [str(i) for i in storagefurniture_tasks]

all_tasks = [bucket_tasks, faucet_tasks, foldingchair_tasks, laptop_tasks, stapler_tasks, toilet_tasks, storagefurniture_tasks]
all_task_names = ["bucket", "faucet", "foldingchair", "laptop", "stapler", "toilet", "storagefurniture"]
all_open_ratios = []
for i, task_list in enumerate(all_tasks):
    avg_open_ratios = []
    for task in task_list:
        json_path = os.path.join(eval_result_path, task, "opened_joint_angles.json")
        with open(json_path, 'r') as f:
            data = json.load(f)
        open_ratios = []
        for entry in data.values():
            final_door_joint_angle = entry["final_door_joint_angle"]
            expert_door_joint_angle = entry["expert_door_joint_angle"]
            initial_joint_angle = entry["initial_joint_angle"]
            if (expert_door_joint_angle - initial_joint_angle) == 0:
                print("task: " + task + " has zero denominator")
                continue
            open_ratio = (final_door_joint_angle - initial_joint_angle) / (expert_door_joint_angle - initial_joint_angle)
            open_ratio = max(0, min(1, open_ratio))
            open_ratios.append(open_ratio)
        avg_open_ratio = sum(open_ratios) / len(open_ratios)
        avg_open_ratios.append(avg_open_ratio)
    print("Average open ratio for category "+ all_task_names[i] + ": " + str(sum(avg_open_ratios) / len(avg_open_ratios)))
    all_open_ratios.append(sum(avg_open_ratios) / len(avg_open_ratios))
print("Average open ratio for all categories: " + str(sum(all_open_ratios) / len(all_open_ratios)))

        
        