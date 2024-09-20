import os
import json
import numpy as np
from termcolor import cprint
import yaml
import copy

generated_obj_ids = [
    41510,
    45397,
    45448,
    46236,
    46462,
    46732,
    46801,
    46874,
    46922,
    46966,
    47570,
    47578,
    48700
]
def traverse(data):
    cur_names = [data['name'], data['text']]
    if 'children' in data.keys():
        for child in data['children']:
            child_names = traverse(child)
            cur_names.extend(child_names)
    
    return cur_names
all_objs_diverse_obj = os.listdir("data/diverse_objects/")
all_objs_diverse_obj = [x.split("_")[-1] for x in all_objs_diverse_obj]
generated_obj_ids.extend([int(x) for x in all_objs_diverse_obj])

all_objs_diverse_obj_2 = os.listdir("data/diverse_objects_2/")
all_objs_diverse_obj_2 = [x.split("_")[-1] for x in all_objs_diverse_obj_2]
generated_obj_ids.extend([int(x) for x in all_objs_diverse_obj_2])


all_objs_diverse_obj_rest = os.listdir("data/diverse_objects_rest/")
all_objs_diverse_obj_rest = [x.split("_")[-1] for x in all_objs_diverse_obj_rest]
generated_obj_ids.extend([int(x) for x in all_objs_diverse_obj_rest])



partnet_obj_dict = "data/partnet_mobility_dict.json"
with open(partnet_obj_dict, 'r') as f:
    partnet_obj_dict = json.load(f)
    
all_storagefurniture = partnet_obj_dict['StorageFurniture']
already_generated = set([str(obj_id) for obj_id in generated_obj_ids])

num_already_generated = len(already_generated)
num_no_handle = 0
all_obj_num = len(all_storagefurniture)
for obj in all_storagefurniture:
    if obj in already_generated: 
        # print("Already generated")
        continue
    
    cur_result_json = os.path.join('data/dataset', obj, 'result.json')
    with open(cur_result_json, 'r') as fin:
        tree_hier = json.load(fin)[0]
    data = tree_hier
    all_names = traverse(data)
    if "handle" not in all_names:
        # print(f"{obj} does not have a handle")
        num_no_handle += 1
        continue
    
    random_obj_id = obj
    cprint(f"Generating object with ID: {random_obj_id}", "green")
    
current_training_obj = [
41510,
45448,
46462,
46732,
46801,
46874,
46922,
46966,
47570,
47578,
48700,
45526,
45661,
45694,
45780,
45910,
45961,
46408,
46417,
46440,
46490,
46762,
46825,
46893,
47235,
47281,
47315,
47529,
47669,
47944,
48063,
48177,
48356,
48623,
48876,
49025,
49062,
49132,
49133,
40417,
41085,
41452,
45162,
45176,
45194,
45203,
45248,
45271,
45290,
45305,
45427,
45620,
45623,
45636,
45689,
45696,
45749,
45759,
45936,
45984,
46130,
46197,
46481,
46544,
47178,
47182,
47227,
47577,
47648,
47747,
47808,
47976,
48010,
48258,
48379,
48797,
48855,
48859,
49188,
35059,
41004,
41083,
44781,
44826,
44853,
45092,
45130,
45135,
45146,
45164,
45168,
45173,
45212,
45213,
45372,
45374,
45387,
45415,
45419,
45423,
45503,
45505,
45524,
45573,
45575,
45606,
45612,
45621,
45622,
45632,
45638,
45645,
45662,
45671,
45676,
45677,
45687,
45699,
45710,
45746,
45756,
45783,
45784,
45790,
45801,
45822,
45853,
45855,
45915,
45948,
45949,
45963,
45964,
46019,
46029,
46033,
46037,
46044,
46045,
46060,
46084,
46108,
46117,
46120,
46123,
46145,
46179,
46180,
46199,
46380,
46427,
46430,
46439,
46537,
46549,
46556,
46598,
46616,
46699,
46700,
46741,
46744,
46847,
46856,
46859,
46889,
46906,
46944,
46955,
46981,
47024,
47089,
47183,
47207,
47233,
47252,
47278,
47290,
47296,
47438,
47514,
47595,
47601,
47632,
47701,
47729,
47853,
47926,
48413,
48452,
48467,
48490,
48513,
48517,
48721,
48746,
48878,
]
    
# print(f"Already generated: {num_already_generated}")
# print(f"Num no handle: {num_no_handle}")
# print(f"Total: {all_obj_num}")
# print(f"Num current training obj: {len(current_training_obj)}")

eval_objs = [
    40147,
    44817,
    44962,
    45132,
    45219,
    45243,
    45332,
    45378,
    45384,
    45463,
]

discard_objs = [
    48243,
    45600, # physically impossible
    45693, # physically impossible
    46127, # physically impossible
    46172, # too many drawers
    48253, # too weird
    41086,
]

discard_due_to_fixed_arm = [
    46787,
    46768,
    45850,
    45297,
    48167,
    48243,
]

to_regenerate = [
    46839,
    47168,
    48169,
    48497,
    49038,
]

already_generated = sorted(already_generated)
for obj in already_generated:
    if int(obj) in eval_objs: continue
    if int(obj) in discard_objs: continue
    if int(obj) in discard_due_to_fixed_arm: continue
    if int(obj) in to_regenerate: continue
    
    if int(obj) not in current_training_obj:
        if obj in all_objs_diverse_obj:
            path = f"/project_data/held/yufeiw2/RoboGen_sim2real/data/diverse_objects/open_the_door_{obj}/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
            all_subfolders = os.listdir(path)
            all_subfolders = sorted(all_subfolders)
            all_subfolders = [x for x in all_subfolders if os.path.isdir(os.path.join(path, x))]
            num_demo = 0
            for ts in all_subfolders:
                state_folder = os.path.join(path, ts, "grasp_the_handle_of_the_storage_furniture_door_primitive/states")
                if not os.path.exists(state_folder):
                    continue
                num_states = len(os.listdir(state_folder))
                if num_states > 20:
                    opened_angle = os.path.join(path, ts, "grasp_the_handle_of_the_storage_furniture_door_primitive/opened_angle.txt")
                    with open(opened_angle, 'r') as f:
                        lines = f.readlines()
                        opened_angle = float(lines[0])
                        max_angle = float(lines[2])
                        ratio = opened_angle / max_angle
                        if ratio > 0.35:
                            num_demo += 1

            if num_demo > 15:
                print(f"{obj}: diverse_objects, has actual demo: {num_demo}")
            else:
                print(f"diverse_objects :", obj)
        if obj in all_objs_diverse_obj_2:
            # if int(obj) >= 45413: continue
            path = f"/project_data/held/yufeiw2/RoboGen_sim2real/data/diverse_objects_2/open_the_door_{obj}/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
            if not os.path.exists(path): continue
            all_subfolders = os.listdir(path)
            all_subfolders = sorted(all_subfolders)
            all_subfolders = [x for x in all_subfolders if os.path.isdir(os.path.join(path, x))]
            num_demo = 0
            for ts in all_subfolders:
                state_folder = os.path.join(path, ts, "grasp_the_handle_of_the_storage_furniture_door_primitive/states")
                # if not os.path.exists(state_folder):
                #     continue
                num_states = len(os.listdir(state_folder))
                if num_states > 20:
                    opened_angle = os.path.join(path, ts, "grasp_the_handle_of_the_storage_furniture_door_primitive/opened_angle.txt")
                    with open(opened_angle, 'r') as f:
                        lines = f.readlines()
                        opened_angle = float(lines[0])
                        max_angle = float(lines[2])
                        ratio = opened_angle / max_angle
                        if ratio > 0.35:
                            num_demo += 1

            if num_demo > 15:
                print(f"{obj}: diverse_objects_2, has actual demo: {num_demo}")
            else:
                print(f"diverse_objects_2: obj")
        if obj in all_objs_diverse_obj_rest:
            continue
            # print(f"{obj}: diverse_objects_rest")