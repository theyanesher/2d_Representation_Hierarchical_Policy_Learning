import os
import json
import numpy as np
from termcolor import cprint
import yaml
import copy

### bad objects 41086

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

all_objs = os.listdir("data/diverse_objects/")
all_objs = [x.split("_")[-1] for x in all_objs]
generated_obj_ids.extend([int(x) for x in all_objs])


save_dir = "diverse_objects_2"
partnet_obj_dict = "data/partnet_mobility_dict.json"
with open(partnet_obj_dict, 'r') as f:
    partnet_obj_dict = json.load(f)
    
all_storagefurniture = partnet_obj_dict['StorageFurniture']
already_generated = set([str(obj_id) for obj_id in generated_obj_ids])

generate_num = 100
for gen_idx in range(generate_num):
    find_new_obj = False
    while not find_new_obj:
        random_obj_id = all_storagefurniture[np.random.randint(0, len(all_storagefurniture))]
        if str(random_obj_id) not in already_generated:
            find_new_obj = True
            already_generated.add(str(random_obj_id))
    
    cprint(f"Generating object with ID: {random_obj_id}", "green")
    
    if not os.path.exists(f"data/{save_dir}"):
        os.makedirs(f"data/{save_dir}")
    command = "rsync -avrz --exclude 'experiment' --exclude '__pycache__' --exclude 'configs' data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_48700_2024-03-27-12-59-58/ data/{}/open_the_door_{}/".format(save_dir, random_obj_id)
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
            obj['solution_path'] = obj['solution_path'].replace("data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_48700_2024-03-27-12-59-58", "data/{}/open_the_door_{}".format(save_dir, random_obj_id))
    for obj in new_config:
        random_center = np.random.uniform(0.6, 0.7)
        if 'center' in obj:
            obj['center'] = f"[{random_center}, 0, 0]"
    for obj in new_config:
        if "reward_asset_path" in obj:
            obj["reward_asset_path"] = str(random_obj_id)
        
    with open(config_path, "w") as f:
        yaml.dump(new_config, f)   