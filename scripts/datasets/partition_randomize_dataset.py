import os
from test_PointNet2.all_data import *


all_objs = os.listdir("/scratch/yufeiw2/dp3_demo")
all_objs = sorted(all_objs)

non_randomize_objs_dict = {}
randomize_objs_dict = {}

for obj in all_objs:
    if obj.startswith("062") or obj.startswith("0705") or obj.startswith("0712") or obj.startswith("0725") or obj.startswith("0730"):
        if 'act3d' in obj:
            obj_id = obj.split("-")[3]
        else:
            obj_id = obj.split("-")[2]
        non_randomize_objs_dict[obj_id] = obj
#     elif obj.startswith("0822") or obj.startswith("0815") or obj.startswith("0826"):
#         obj_id = obj.split("-")[2]
#         randomize_objs_dict[obj_id] = obj
#         randomize_objs.append(obj_id)


randomize_obj_dirs = []
for idx in range(287, 287 + 150):
    obj_dir = globals()[f"save_data_name_{idx}"]
    obj_id = obj_dir.split("-")[-1]
    randomize_objs_dict[obj_id] = obj_dir
    randomize_obj_dirs.append(obj_dir)
# print(len(randomize_objs_dict))
all_randomize_obj_ids = sorted(list(randomize_objs_dict.keys()))

### 200 partition: all above randomzied and pick 200 unrandomized    
# idx = 0
# for obj_id in all_randomize_obj_ids:
#     if obj_id not in non_randomize_objs_dict: 
#         continue # print("************************ not in: ", obj_id)
#     else: 
#         print(f"new_partition_save_data_name_{idx}='{non_randomize_objs_dict[obj_id]}'")
#         idx += 1
        
# for idx2 in range(246):
#     old_save_data_name = globals()[f"save_data_name_{idx2}"]
#     obj_id = old_save_data_name.split("-")[-1]
#     if obj_id not in all_randomize_obj_ids:
#         print(f"new_partition_save_data_name_{idx}='{old_save_data_name}'")
#         idx += 1
#     if idx == 200:
#         break
        
# for obj_dir in randomize_obj_dirs:
#     print(f"new_partition_save_data_name_{idx}='{obj_dir}'")
#     idx += 1
    
    
### 100 partition
# idx = 0
# for obj_id in all_randomize_obj_ids[:75]:
#     if obj_id not in non_randomize_objs_dict: 
#         continue # print("************************ not in: ", obj_id)
#     else: 
#         print(f"new_partition_save_data_name_{idx}='{non_randomize_objs_dict[obj_id]}'")
#         idx += 1
        
# for idx2 in range(246):
#     old_save_data_name = globals()[f"save_data_name_{idx2}"]
#     obj_id = old_save_data_name.split("-")[-1]
#     if obj_id not in all_randomize_obj_ids:
#         print(f"new_partition_save_data_name_{idx}='{old_save_data_name}'")
#         idx += 1
#     if idx == 100:
#         break
        
# for obj_id in all_randomize_obj_ids[:75]:
#     print(f"new_partition_save_data_name_{idx}='{randomize_objs_dict[obj_id]}'")
#     idx += 1
    
    
### 50 partition
idx = 0
for obj_id in all_randomize_obj_ids[:37]:
    if obj_id not in non_randomize_objs_dict: 
        continue # print("************************ not in: ", obj_id)
    else: 
        print(f"new_partition_save_data_name_{idx}='{non_randomize_objs_dict[obj_id]}'")
        idx += 1
        
for idx2 in range(246):
    old_save_data_name = globals()[f"save_data_name_{idx2}"]
    obj_id = old_save_data_name.split("-")[-1]
    if obj_id not in all_randomize_obj_ids:
        print(f"new_partition_save_data_name_{idx}='{old_save_data_name}'")
        idx += 1
    if idx == 50:
        break
        
for obj_id in all_randomize_obj_ids[:37]:
    print(f"new_partition_save_data_name_{idx}='{randomize_objs_dict[obj_id]}'")
    idx += 1
    





    
        