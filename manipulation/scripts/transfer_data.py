import os
import json

demo_data = "data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_45448_2024-03-27-22-40-39/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-2-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
all_subdirs = os.listdir(demo_data)
all_subdirs = sorted(all_subdirs)

task = "grasp_the_door_handle_primitive"
for dir in all_subdirs:
    stage_length = os.path.join(demo_data, dir, task, "stage_lengths.json")
    if not os.path.exists(stage_length):
        continue
    with open(stage_length, 'r') as fin:
        stage_lengths = json.load(fin)
    open_begin_t_idx = stage_lengths['reach_handle'] + stage_lengths['reach_to_contact'] + stage_lengths['close_gripper']
    all_time_steps = stage_lengths['reach_handle'] + stage_lengths['reach_to_contact'] + stage_lengths['close_gripper'] + stage_lengths['open_door'] - 1
    autobot_folder = "autobot:/project_data/held/yufeiw2/RoboGen_sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_45448_2024-03-27-22-40-39/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-2-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first/"
    local_folder = "{}/{}/{}/states/".format(demo_data, dir, task)
    scp_cmd_1 = "scp {}/{}/{}/states/state_{}.pkl {}".format(autobot_folder, dir, task, open_begin_t_idx, local_folder)
    scp_cmd_2 = "scp {}/{}/{}/states/state_{}.pkl {}".format(autobot_folder, dir, task, all_time_steps, local_folder)
    # print(scp_cmd_1)
    # print(scp_cmd_2)
    os.system(scp_cmd_1)
    os.system(scp_cmd_2)
    
    