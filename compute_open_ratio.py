import json
import os

eval_result_path = 'data/0505_one_hot/'

bucket_tasks = [100444, 100452, 100454, 100460, 100461, 100462, 100469, 100472, 102352, 102365]
faucet_tasks = [148, 149, 152, 153, 154, 168, 811, 857, 960, 991]
foldingchair_tasks = [100520, 100521, 100526, 100562, 100586, 100590, 100599, 102263, 102269, 102314]
laptop_tasks = [9748, 9912, 9960, 9968, 9992, 9996, 10040, 10098, 10101, 10238]
stapler_tasks = [103095, 103099, 103100, 103104, 103111, 103292, 103293, 103297, 103299, 103301]
toilet_tasks = [101320, 102621, 102622, 102630, 102634, 102645, 102648, 102651, 102652, 102658]
storagefurniture_tasks = [41510, 45448, 46462, 46732, 46801, 46874, 46922, 46966, 47570, 47578]
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

        
        