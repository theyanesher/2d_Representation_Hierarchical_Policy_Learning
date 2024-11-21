import os
import json

def traverse(data):
    cur_names = [data['name'], data['text']]
    if 'children' in data.keys():
        for child in data['children']:
            child_names = traverse(child)
            cur_names.extend(child_names)
    
    return cur_names

path = 'data/diverse_objects/'
all_directories = os.listdir(path)
all_directories = sorted(all_directories)

for dir in all_directories:
    obj = dir.split('_')[-1]
    
    # check if handle exists
    cur_result_json = os.path.join('data/dataset', obj, 'result.json')
    with open(cur_result_json, 'r') as fin:
        tree_hier = json.load(fin)[0]
    data = tree_hier
    all_names = traverse(data)
    if "handle" not in all_names:
        print(f"{obj} does not have a handle")
        # os.system(f"rm -r {os.path.join(path, dir)}")
        continue
    
    semantics_file = os.path.join("data/dataset", obj, "semantics.txt")
    with open(semantics_file, 'r') as f:
        semantics = f.readlines()
    door_link = None    
    for line in semantics:
        if 'door' in line or 'drawer' in line:
            door_link = line.split()[0]
            break
    
    if door_link is not None and door_link != 'link_0':
        with open(os.path.join(path, dir, "task_open_the_door_of_the_storagefurniture_by_its_handle/grasp_the_handle_of_the_storage_furniture_door.py"), 'r') as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            if 'link_0' in line:
                new_lines.append(line.replace('link_0', door_link))
            else:
                new_lines.append(line)
        with open(os.path.join(path, dir, "task_open_the_door_of_the_storagefurniture_by_its_handle/grasp_the_handle_of_the_storage_furniture_door.py"), 'w') as f:
            f.writelines(new_lines)
        
        print(f"{obj} door link is not link 0")
        
        
    if door_link is None:
        print(f"{obj} does not have a door link")
        # os.system(f"rm -r {os.path.join(path, dir)}")
