import os
import json
import numpy as np
from termcolor import cprint
import yaml
import copy

### bad objects 41086

# generated_obj_ids = [
#     41510,
#     45397,
#     45448,
#     46236,
#     46462,
#     46732,
#     46801,
#     46874,
#     46922,
#     46966,
#     47570,
#     47578,
#     48700
# ]
def traverse(data):
    cur_names = [data['name'], data['text']]
    if 'children' in data.keys():
        for child in data['children']:
            child_names = traverse(child)
            cur_names.extend(child_names)
    
    return cur_names
# all_objs = os.listdir("data/diverse_objects/")
# all_objs = [x.split("_")[-1] for x in all_objs]
# generated_obj_ids.extend([int(x) for x in all_objs])

# all_objs = os.listdir("data/diverse_objects_2/")
# all_objs = [x.split("_")[-1] for x in all_objs]
# generated_obj_ids.extend([int(x) for x in all_objs])


# 10 testing objects
# generated_obj_ids = [
#     40147,
#     44817,
#     44962,
#     45132,
#     45219,
#     45243,
#     45332,
#     45378,
#     45384,
#     45463,
# ]
generated_obj_ids = []

save_dir = "diverse_objects_other"
partnet_obj_dict = "data/partnet_mobility_dict.json"
with open(partnet_obj_dict, 'r') as f:
    partnet_obj_dict = json.load(f)
    
all_storagefurniture = partnet_obj_dict['StorageFurniture']
all_storagefurniture = partnet_obj_dict['Refrigerator']
already_generated = set([str(obj_id) for obj_id in generated_obj_ids])

for obj in all_storagefurniture:
    if obj in already_generated: 
        print(f"{obj} already generated")
        continue
    
    cur_result_json = os.path.join('data/dataset', obj, 'result.json')
    with open(cur_result_json, 'r') as fin:
        tree_hier = json.load(fin)[0]
    data = tree_hier
    all_names = traverse(data)
    if "handle" not in all_names:
        print(f"{obj} does not have a handle")
        continue
    
    random_obj_id = obj
    cprint(f"Generating object with ID: {random_obj_id}", "green")
    
    if not os.path.exists(f"data/{save_dir}"):
        os.makedirs(f"data/{save_dir}")
    command = "cp -r data/temp/template/ data/{}/open_the_door_{}/".format(save_dir, random_obj_id)
    print(command)
    os.system(command)

    path = "data/{}/open_the_door_{}/".format(save_dir, random_obj_id)
    yaml_config = [x for x in os.listdir(path) if x.endswith(".yaml")][0]
    config_path = os.path.join(path, yaml_config)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    solution_path = [x['solution_path'] for x in config if 'solution_path' in x][0]
    reward_assets = [x['reward_asset_path'] for x in config if 'reward_asset_path' in x][0]
    new_config = copy.deepcopy(config)
    for obj in new_config:
        if 'solution_path' in obj:
            obj['solution_path'] = obj['solution_path'].replace("data/diverse_objects_2/open_the_door_40417", "data/{}/open_the_door_{}".format(save_dir, random_obj_id))
    for obj in new_config:
        random_center = np.random.uniform(0.6, 0.7)
        if 'center' in obj:
            obj['center'] = f"[{random_center}, 0, 0]"
    for obj in new_config:
        if "reward_asset_path" in obj:
            obj["reward_asset_path"] = str(random_obj_id)

    for obj in new_config:
        if 'size' in obj:
            obj['size'] = 1.50  # storagefurniture
            obj['size'] = 0.65  # microwave
            obj['size'] = 1.00  # dishwasher
            obj['size'] = 0.80  # oven
            obj['size'] = 1.50  # refrigerator
        
    with open(config_path, "w") as f:
        yaml.dump(new_config, f)   
        
    semantics_file = os.path.join("data/dataset", random_obj_id, "semantics.txt")
    with open(semantics_file, 'r') as f:
        semantics = f.readlines()
    door_link = None    
    for line in semantics:
        if 'door' in line or 'drawer' in line:
            door_link = line.split()[0]
            break
        
    path = "data/{}/open_the_door_{}/".format(save_dir, random_obj_id)
    if door_link is not None and door_link != 'link_0':
        with open(os.path.join(path, "task_open_the_door_of_the_storagefurniture_by_its_handle/grasp_the_handle_of_the_storage_furniture_door.py"), 'r') as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            if 'link_0' in line:
                new_lines.append(line.replace('link_0', door_link))
            else:
                new_lines.append(line)
        with open(os.path.join(path, "task_open_the_door_of_the_storagefurniture_by_its_handle/grasp_the_handle_of_the_storage_furniture_door.py"), 'w') as f:
            f.writelines(new_lines)
        
        print(f"{obj} door link is not link 0")
        
    import pdb; pdb.set_trace()