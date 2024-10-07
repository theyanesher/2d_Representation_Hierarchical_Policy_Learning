from matplotlib import pyplot as plt
import json
import os
import numpy as np

folder = "3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/eval_low_50_pcd_200_high_level_weighted_diffusion_1/09092207-dp3_goal_gripper_dense-horizon-8-num_load_episodes-75_2024.09.09_22.07.41_train_dp3_robogen_open_door_checkpoints_epoch-10.ckpt"

all_subfolders = os.listdir(folder)
all_obj_runs = []
for idx in range(10):
    performance_runs = []
    grasped_handles_runs = []
    for subfolder in all_subfolders:
        json_results = os.path.join(folder, subfolder, "opened_joint_angles_{}.json".format(idx))
        with open(json_results) as f:
            data = json.load(f)
            
        opened_joint_angles = []
        expert_angles = []
        initial_angles = []
        grasped_handles = []
        string = "open_the_door_" 
        # string = "StorageFurniture_"
        obj_id = list(data.keys())[0].find(string)
        obj_id = list(data.keys())[0][obj_id+len(string):obj_id+len(string)+5]
        keys = list(data.keys())
        keys = sorted(keys)
        for key in keys[:25]:
            if data[key]['expert_door_joint_angle'] == data[key]['initial_joint_angle']:
                continue
            opened_joint_angles.append(data[key]["final_door_joint_angle"])
            expert_angles.append(data[key]["expert_door_joint_angle"] if "46462" not in key else 0.27)
            initial_angles.append(data[key]['initial_joint_angle'])
            grasped_handles.append(data[key]["grasped_handle"])

        normalized_performance = (np.array(opened_joint_angles) - np.array(initial_angles)) / (np.array(expert_angles) - np.array(initial_angles))
        normalized_performance[normalized_performance > 1] = 1
        normalized_performance = normalized_performance[~np.isnan(normalized_performance)]
        normalized_performance = np.mean(normalized_performance)
        
        performance_runs.append(normalized_performance)
        grasped_handles_runs.append(np.mean(grasped_handles))
    
    all_obj_runs.extend(performance_runs)

    # print(performance_runs)
    print("obj id {} {:.3f} {:.3f}".format(obj_id, np.mean(performance_runs), np.std(performance_runs)))
    # print("obj id grasped handles {} {} {}".format(obj_id, np.mean(grasped_handles_runs), np.std(grasped_handles_runs)))
    
print("all obj average: {:.3f} {:.3f}".format(np.mean(all_obj_runs), np.std(all_obj_runs)))