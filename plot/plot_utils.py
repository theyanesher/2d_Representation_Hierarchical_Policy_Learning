import pandas
import os.path as osp
import os
import json
import numpy as np
from collections import defaultdict
import json
import yaml
import glob
import os

def find_video_directories(base_directory):
    # Use glob to find directories containing the word 'video'
    pattern = os.path.join(base_directory, '**', '*video*')
    directories = [d for d in glob.glob(pattern, recursive=True) if os.path.isdir(d)]
    return directories

def read_json(path, use_binary=False, return_dict=False, max_angle=None):
    values = []
    results = path
    with open(results, 'r') as f:
        data = json.load(f)
    
    if return_dict:
        return data
        
    for idx, config_path in enumerate(data):
        # if idx == 30:
        #     break
        res = data[config_path]
        if type(res) == list:
            policy_angle = float(res[0])
            expert_angle = float(res[1]) if "46462" not in config_path else 0.27
            initial_angle = float(res[2])
            policy_angle = min(policy_angle, expert_angle)
            normalized_performance = (policy_angle - initial_angle) / (expert_angle - initial_angle)
            binary = 1 if normalized_performance > 0.1 else 0
            if use_binary:
                values.append(binary)
            else:
                values.append(normalized_performance)
    return values

def read_data(data_dirs, filter_function=None, label_function=None, read_train=False, mean=False, return_dict=False):
    
    all_subdirs = []
    for data_dir in data_dirs:
        # subdirs = os.listdir(data_dir)
        # for sd in subdirs:
        #     all_subsubdirs = os.listdir(osp.join(data_dir, sd))
        #     for ssd in all_subsubdirs:
        #         all_subdirs.append(osp.join(data_dir, sd, ssd))
        base_directory = data_dir  # Replace with your base directory
        video_directories = find_video_directories(base_directory)

        # for directory in video_directories:
        #     print(directory)
        
        all_subdirs.extend(video_directories)
        
    all_subdirs = sorted(all_subdirs)
    print(all_subdirs)

    all_results = []
    for video_dir in all_subdirs:
        if not os.path.exists(video_dir):
            continue
        all_eval_video_paths = os.listdir(video_dir)
        if len(all_eval_video_paths) == 0:
            continue
        all_eval_video_paths = sorted(all_eval_video_paths)
        
        all_values = []
        for eval_video_path in all_eval_video_paths:
            if (read_train and "trainset" in eval_video_path) or (not read_train and "valset" in eval_video_path):
                results = os.path.join(video_dir, eval_video_path, "opened_joint_angles.json")
                if os.path.exists(results):
                    max_angle = 0.27 if "45462" in video_dir else None
                    values = read_json(results, return_dict=return_dict, max_angle=max_angle)
                    # import pdb; pdb.set_trace()
                    if not return_dict:
                        if mean:
                            mean_values = np.mean(values) if len(values) > 0 else 0
                            all_values.append(mean_values)
                        else:
                            all_values.append(values)
                    else:
                        all_values.append(values)

        # if len(all_values) < 3:
        #     continue
        
        subdir = os.path.dirname(video_dir)
        variant_path = osp.join(subdir, '.hydra', "overrides.yaml")
        if not osp.exists(variant_path): 
            continue
        variant = yaml.safe_load(open(variant_path, 'r'))
        real_variant = {}
        for key_val in variant:
            key, val = key_val.split("=")
            real_variant[key] = val
        variant = real_variant
        # import pdb; pdb.set_trace()

        if filter_function is not None:
            if filter_function(variant):
                continue

        label = label_function(variant)
        # print(subdir, label)
        all_results.append((all_values, label))

    # result is a two level dict, first key is plot key, second key is group key
    return all_results

def group_data(results, return_eval_freq=False):
    if type(results) == list: # sinlge plot key
        result_dict = defaultdict(list)
        eval_freq_dict = defaultdict(list)
        for res in results:
            # import pdb; pdb.set_trace()
            res_values, group_key = res
            result_dict[group_key].append(res_values)
            if return_eval_freq:
                eval_freq_dict[group_key].append(eval_freq)
        if return_eval_freq:
            return result_dict, eval_freq_dict
        return result_dict
    else: # multiple plot keys
        all_result_dict = {} 
        all_eval_freq_dict = {}
        for plot_key, result in results.items():
            result_dict = defaultdict(list)
            eval_dict = defaultdict(list)
            for res in result:
                res_values, eval_freq, group_key = res
                result_dict[group_key].append(res_values)
                eval_dict[group_key].append(eval_freq)
            all_result_dict[plot_key] = result_dict
            all_eval_freq_dict[plot_key] = eval_dict
        if return_eval_freq:
            return all_result_dict, all_eval_freq_dict
        return all_result_dict


def read_and_group_data(data_dirs, filter_function=None, label_function=None, return_eval_freq=False, read_train=False, mean=True, return_dict=False):

    results = read_data(data_dirs, filter_function, label_function, read_train=read_train, mean=mean, return_dict=return_dict)
    if not return_eval_freq:
        result_dict = group_data(results, return_eval_freq=False)
        return result_dict
    else:
        result_dict, eval_freq_dict = group_data(results, return_eval_freq=True)
        return result_dict, eval_freq_dict

def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)

def tolerant_max(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.max(axis = -1), arr.std(axis=-1)

def read_json_log(path):
    filtered_data = []
    with open(path, 'r') as handle:
        json_data = [json.loads(line) for line in handle]
        for dict in json_data:
            if len(dict.keys()) > 1:
                filtered_data.append(dict)

    return filtered_data