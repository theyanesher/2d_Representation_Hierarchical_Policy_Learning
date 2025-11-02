import json
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--articubot_only', type=int, default=1)
parser.add_argument('--invert', type=int, default=0)
args = parser.parse_args()

eval_result_path = 'data_yufei/eval_ptv3_articubot_50_cgn/'
eval_result_path = 'data_yufei/eval_3dfa_50/'
eval_result_path = 'data_yufei/eval_articubot_both_450_dagger/'
eval_result_path = 'data_yufei/eval_3dfa_50_more_epoch/'
eval_result_path = "data_yufei/eval_3dfa_50_120k/"
# eval_result_path = "data_yufei/eval_3dfa_50_300k/"
eval_result_path = "data_yufei/eval_ptv3_articubot_50_single/"
# eval_result_path = "data_yufei/eval_ptv3_articubot_50_single_2/"
# eval_result_path = "data_yufei/eval_m2t2_cgn_articubot_50_gmm"
eval_result_path = "data_yufei/eval_3dfa_new"
# eval_result_path = "data_yufei/eval_3dfa_small_pn_correct"
# eval_result_path = "data_yufei/eval_3dfa_even_larger_pn"
# # eval_result_path = "data_yufei/eval_3dfa_even_larger_pn_64_not_128"
# # eval_result_path = "data_yufei/eval_3dfa_articulated_larger"
# eval_result_path = "data_yufei/eval_3dfa_articulated_large"
# eval_result_path = "data_yufei/eval_pointnext_50"
# eval_result_path = "data_yufei/eval_pointnet_multigmm_50"
# eval_result_path = "data_yufei/eval_pointnext_new_cat_camera_random_close_w_pick_place_new_low_level"
# eval_result_path = "data_yufei/eval_articubot-pointnext-fp-50"
# eval_result_path = "data_yufei/eval_pointnext-full-pick-and-place"
eval_result_path = "data_yufei/eval_3dfa_articulated_large_2"
eval_result_path = "data_yufei/eval_baseline_ours_articulated"
eval_result_path = "data_yufei/eval_baseline_ptv3_articulated"

storagefurniture_tasks = os.listdir(os.path.join(eval_result_path, "diverse_objects"))
if not args.articubot_only:
    if not args.invert:
        bucket_tasks = os.listdir(os.path.join(eval_result_path, "bucket"))
        faucet_tasks = os.listdir(os.path.join(eval_result_path, "faucet"))
        foldingchair_tasks = os.listdir(os.path.join(eval_result_path, "foldingchair"))
        laptop_tasks = os.listdir(os.path.join(eval_result_path, "laptop"))
        stapler_tasks = os.listdir(os.path.join(eval_result_path, "stapler"))
        toilet_tasks = os.listdir(os.path.join(eval_result_path, "toilet"))
        all_tasks = [bucket_tasks, faucet_tasks, foldingchair_tasks, laptop_tasks, stapler_tasks, toilet_tasks, storagefurniture_tasks]
        all_task_names = ["bucket", "faucet", "foldingchair", "laptop", "stapler", "toilet", "diverse_objects"]
    else:
        foldingchair_tasks = os.listdir(os.path.join(eval_result_path, "foldingchair"))
        laptop_tasks = os.listdir(os.path.join(eval_result_path, "laptop"))
        stapler_tasks = os.listdir(os.path.join(eval_result_path, "stapler"))
        toilet_tasks = os.listdir(os.path.join(eval_result_path, "toilet"))
        all_tasks = [foldingchair_tasks, laptop_tasks, stapler_tasks, toilet_tasks, storagefurniture_tasks]
        all_task_names = ["foldingchair", "laptop", "stapler", "toilet", "diverse_objects"]
else:
    all_tasks = [storagefurniture_tasks]
    all_task_names = ["diverse_objects"]
    
all_open_ratios = []
all_ik_failures = []
all_oversized_ratios = []
all_grasp_ratios = []
all_open_ratios_per_object = []
for i, task_list in enumerate(all_tasks):
    avg_open_ratios = []
    avg_ik_failures = []
    avg_oversized_ratios = []
    avg_grasp_ratios = []
    for task in task_list:
        json_path = os.path.join(eval_result_path, all_task_names[i], task, "opened_joint_angles.json")
        with open(json_path, 'r') as f:
            data = json.load(f)
        open_ratios = []
        ik_failures = []
        oversized_ratios = []
        grasp_ratios = []
        for entry in data.values():
            final_door_joint_angle = entry["final_door_joint_angle"]
            expert_door_joint_angle = entry["expert_door_joint_angle"]
            initial_joint_angle = entry["initial_joint_angle"]

            if (expert_door_joint_angle - initial_joint_angle) == 0:
                print("task: " + task + " has zero denominator")
                continue
            open_ratio = (final_door_joint_angle - initial_joint_angle) / (expert_door_joint_angle - initial_joint_angle)
            open_ratio = max(0, min(1, open_ratio))
            ik_failure = entry["ik_failure"]
            oversized_ratio = entry["oversized_joint_distance"]
            grasp_ratio = entry["grasped_handle"]
            open_ratios.append(open_ratio)
            ik_failures.append(ik_failure)
            oversized_ratios.append(oversized_ratio)
            grasp_ratios.append(grasp_ratio)
        avg_open_ratio = sum(open_ratios) / len(open_ratios)
        avg_ik_failure = sum(ik_failures) / len(ik_failures)
        avg_oversized_ratio = sum(oversized_ratios) / len(oversized_ratios)
        avg_grasp_ratio = sum(grasp_ratios) / len(grasp_ratios)
        avg_open_ratios.append(avg_open_ratio)
        avg_ik_failures.append(avg_ik_failure)
        avg_oversized_ratios.append(avg_oversized_ratio)
        avg_grasp_ratios.append(avg_grasp_ratio)
    print("Average open ratio for category "+ all_task_names[i] + ": " + str(sum(avg_open_ratios) / len(avg_open_ratios)))
    # print("Average ik failure for category "+ all_task_names[i] + ": " + str(sum(avg_ik_failures) / len(avg_ik_failures)))
    # print("Average oversized ratio for category "+ all_task_names[i] + ": " + str(sum(avg_oversized_ratios) / len(avg_oversized_ratios)))
    # print("Average grasp ratio for category "+ all_task_names[i] + ": " + str(sum(avg_grasp_ratios) / len(avg_grasp_ratios)))
    all_open_ratios.append(sum(avg_open_ratios) / len(avg_open_ratios))
    all_open_ratios_per_object.extend(avg_open_ratios)
    # all_ik_failures.append(sum(avg_ik_failures) / len(avg_ik_failures))
    # all_oversized_ratios.append(sum(avg_oversized_ratios) / len(avg_oversized_ratios))
    # all_grasp_ratios.append(sum(avg_grasp_ratios) / len(avg_grasp_ratios))
print("Average open ratio for all categories: " + str(sum(all_open_ratios) / len(all_open_ratios)))
print("Average open ratio for all objects: ", np.mean(all_open_ratios_per_object))
# print("Average ik failure for all categories: " + str(sum(all_ik_failures) / len(all_ik_failures)))
# print("Average oversized ratio for all categories: " + str(sum(all_oversized_ratios) / len(all_oversized_ratios)))
# print("Average grasp ratio for all categories: " + str(sum(all_grasp_ratios) / len(all_grasp_ratios)))

        
        