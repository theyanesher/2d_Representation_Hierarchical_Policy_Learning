import os
import argparse
from plot.plot_utils import read_and_group_data, tolerant_mean, tolerant_max
import numpy as np
from matplotlib import pyplot as plt
from termcolor import cprint

parser = argparse.ArgumentParser()
parser.add_argument("data_paths", type=str, nargs='*')
parser.add_argument("--read_train", type=int, default=1)
parser.add_argument("--return_dict", type=int, default=0)
args = parser.parse_args()


data_dirs = args.data_paths

def label_func(variant):
    # return variant.get("policy.encoder_type", "mlp")
    return variant.get("task.dataset.zarr_path")#.split("/")[-1]

all_res = read_and_group_data(data_dirs, read_train=args.read_train, label_function=label_func, mean=True, return_dict=args.return_dict)

# obj_ids = [41510, 45448, 46462]
obj_id = 46462

# print(all_res)
for l_idx, label in enumerate(all_res):
    values = all_res[label]
    all_seed_max_values = []
    for seed_value in values:
        print(seed_value)
        seed_max_value = -1
        for epoch_value in seed_value:
            if type(epoch_value) == list:
                seed_max_value = max(seed_max_value, np.mean(epoch_value))
            elif type(epoch_value) == dict:
                obj_values = []
                for idx, config_path in enumerate(epoch_value):
                    if str(obj_id) not in config_path:
                        continue
                    res = epoch_value[config_path]
                    policy_angle = float(res[0])
                    max_angle = 0.27 if obj_id == 46462 else None
                    expert_angle = float(res[1]) if max_angle is None else max_angle
                    initial_angle = float(res[2])
                    policy_angle = min(policy_angle, expert_angle)
                    normalized_performance = (policy_angle - initial_angle) / (expert_angle - initial_angle)
                    binary = 1 if normalized_performance > 0.1 else 0
                    obj_values.append(normalized_performance)
                seed_max_value = max(seed_max_value, np.mean(obj_values))
            else:
                seed_max_value = max(seed_max_value, epoch_value)    
        
        all_seed_max_values.append(seed_max_value)
        
    # mean = np.mean(all_seed_max_values)
        
    print(f"{label} mean: {all_seed_max_values}")