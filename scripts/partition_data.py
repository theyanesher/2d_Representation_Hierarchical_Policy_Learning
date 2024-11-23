import os

### TODO: 
# gather all objects that are rendered under perfect camera poses
# partition them to different objects randomly: 10, 50, 100, 200, 300
# count the associate # of trajectories
# find a decent set of test objects that cover all categories: storagefurniture, microwave, dishwasher, fridge, oven

# Test set: 
storagefurniture_test_objs = [
    'data/diverse_objects/open_the_door_40147/task_open_the_door_of_the_storagefurniture_by_its_handle', # pull right, vertical handle
    'data/diverse_objects/open_the_door_44817/task_open_the_door_of_the_storagefurniture_by_its_handle', # drawer, horizontal handle
    'data/diverse_objects/open_the_door_44962/task_open_the_door_of_the_storagefurniture_by_its_handle', # drawer, horizontal hanlde
    'data/diverse_objects/open_the_door_45132/task_open_the_door_of_the_storagefurniture_by_its_handle', # drawer, knob
    'data/diverse_objects/open_the_door_45219/task_open_the_door_of_the_storagefurniture_by_its_handle', # pull right, very small handle
    'data/diverse_objects/open_the_door_45332/task_open_the_door_of_the_storagefurniture_by_its_handle', # pull right, horizontal handle
    "data/diverse_objects_2/open_the_door_45249/task_open_the_door_of_the_storagefurniture_by_its_handle", # pull left, vertical handle
    "data/diverse_objects/open_the_door_46417/task_open_the_door_of_the_storagefurniture_by_its_handle" #, top down, horizontoal handle
    "data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_45448_2024-03-27-22-40-39/task_open_the_door_of_the_storagefurniture_by_its_handle", # pull left horizontal
    "data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle", # pull left, knob
]

storagefurniture_test_experiment_names = [
    "0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first",
    "0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first",
    "0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first",
    "0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first",
    "0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first",
    "0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first",
    "0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first",
    "0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first",
    "0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-200-demo-0.4-0.15-translation-first",
    "0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-200-demo-0.4-0.15-translation-first"
]

other_test_objs = [
    "data/diverse_objects_other/open_the_door_7290/task_open_the_door_of_the_storagefurniture_by_its_handlee", # oven
    "data/diverse_objects_other/open_the_door_7310/task_open_the_door_of_the_storagefurniture_by_its_handle", # microwave
    "data/diverse_objects_other/open_the_door_10867/task_open_the_door_of_the_storagefurniture_by_its_handle", # fridge
    "data/diverse_objects_other/open_the_door_12092/task_open_the_door_of_the_storagefurniture_by_its_handle", # dishwahser
    "data/diverse_objects_other/open_the_door_12606/task_open_the_door_of_the_storagefurniture_by_its_handle", # diswasher
]

all_test_obj_ids = [
    "40147",
    "44817",
    "44962",
    "45132",
    "45219",
    "45332",
    "45249",
    "46417",
    "45448",
    "41510",
    "7290",
    "7310",
    "10867",
    "12092",
    "12606",
]

import glob
import os

# Specify the base directory to search
base_directory = '/scratch/yufeiw2/dp3_demo/'

# Use glob.glob to find all matching directories recursively
matching_directories = glob.glob(os.path.join(base_directory, '**/*-obj-*'), recursive=True)

# Filter only directories
matching_directories = [d for d in matching_directories if os.path.isdir(d)]
matching_directories = sorted(matching_directories)

# Print the results
all_training_directories = []
for directory in matching_directories:
    if '0622' in directory or '0624' in directory or '0626' in directory or '0628' in directory or '0705' in directory or '0725' in directory or '0730' in directory or '1121' in directory:
        is_test = False
        for test_obj_id in all_test_obj_ids:
            if test_obj_id in directory:
                is_test = True
                break
        if not is_test:
            all_training_directories.append(directory)

import random
random.shuffle(all_training_directories)

num_train_objs = [10, 50, 100, 200, len(all_training_directories)]
for num_train_obj in num_train_objs:
    train_directories = all_training_directories[:num_train_obj]
    with open(f"scripts/train_dataset_{num_train_obj}.sh", 'w') as f:
        for idx, directory in enumerate(train_directories):
            f.write(f"save_data_name_{idx}='{directory}'\n")
            

            

            

