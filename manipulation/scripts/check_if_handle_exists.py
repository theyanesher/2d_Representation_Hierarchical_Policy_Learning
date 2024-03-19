import json
import os
# to import pprint
from pprint import pprint


meta_dict = "data/partnet_mobility_dict.json"
with open(meta_dict, 'r') as fin:
    meta_dict = json.load(fin)
    
def traverse(data):
    cur_names = [data['name'], data['text']]
    if 'children' in data.keys():
        for child in data['children']:
            child_names = traverse(child)
            cur_names.extend(child_names)
    
    return cur_names

name = 'Dishwasher'
# name = 'Microwave'
# name = 'Safe'
# name = 'TrashCan'
all_objects = meta_dict[name]
handle_num = 0
with_handle_obj_ids = []
for obj_id in all_objects:
    cur_shape_dir = "data/dataset/{}".format(obj_id)
    cur_result_json = os.path.join(cur_shape_dir, 'result.json')
    with open(cur_result_json, 'r') as fin:
        tree_hier = json.load(fin)[0]
    data = tree_hier
    
    all_names = traverse(data)
    # print(obj_id, all_names)
    # print("obj_id: ", obj_id, "has handles: ", "handle" in all_names)   
    print(obj_id, all_names)
    # import pdb; pdb.set_trace()
    if "handle" in all_names:
        # print(obj_id) 
        handle_num += 1
        with_handle_obj_ids.append(obj_id)
        
# print(f"{name} all objects num: {len(all_objects)}, number of objects with handle: {handle_num}")    
pprint(with_handle_obj_ids)
print(len(with_handle_obj_ids))
    