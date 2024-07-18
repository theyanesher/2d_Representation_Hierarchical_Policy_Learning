import os, zarr, json
from termcolor import cprint
import numpy as np
from tqdm import tqdm

def fix_goal(demo_path, zarr_path):

    all_subfolder = os.listdir(zarr_path)
    for string in ["action_dist", "demo_rgbs", "all_demo_path.txt", "meta_info.json", 'example_pointcloud']:
        if string in all_subfolder:
            cprint(f'{string} has been removed', 'green')
            all_subfolder.remove(string)
            
    all_subfolder = sorted(all_subfolder)

    keys = ['state', 'action', 'point_cloud', ]
    keys += ['gripper_pcd', 'displacement_gripper_to_object', 'goal_gripper_pcd']
    combine_action_steps = 2

    # # read each step of each episode
    # all_trajs = os.listdir(zarr_path)
    # all_trajs = sorted(all_trajs)

    for traj in tqdm(all_subfolder, desc='processing'):

        zarr_path_traj = os.path.join(zarr_path, traj)
        exp_name = traj

        if not os.path.exists(os.path.join(demo_path, exp_name)):
            cprint('{} not exists'.format(os.path.join(demo_path, exp_name)), 'yellow')
            cmd = "rm -r " + zarr_path_traj
            os.system(cmd)
            cprint(f'{zarr_path_traj} deleted', 'red')
            continue

        all_subfolder = os.listdir(os.path.join(demo_path, exp_name))
        all_subfolder = [d for d in all_subfolder if os.path.isdir(os.path.join(demo_path, exp_name, d))]
        d = all_subfolder[0]
        stage_lengths_json_file = os.path.join(demo_path, exp_name, d, 'stage_lengths.json')
        with open(stage_lengths_json_file, 'r') as f:
            stage_lengths = json.load(f)
        new_opening_start_idx = (stage_lengths['reach_handle'] + stage_lengths["reach_to_contact"] + stage_lengths["close_gripper"]) // combine_action_steps    
        
        all_steps = os.listdir(zarr_path_traj)
        all_steps = sorted(all_steps, key=lambda x: int(x))
        
        last_step = all_steps[-1]
        zarr_path_step = os.path.join(zarr_path_traj, last_step)
        group = zarr.open(zarr_path_step, 'r')
        src_store = group.store
        src_root = zarr.group(src_store)
        goal_gripper_pcd_arr = src_root['data']["gripper_pcd"][:]

        # for every step after opening, change the goal gripper pcd
        for step in all_steps[new_opening_start_idx:]:
            zarr_path_step = os.path.join(zarr_path_traj, step)
            group = zarr.open(zarr_path_step, 'r+')
            src_store = group.store

            # numpy backend
            src_root = zarr.group(src_store)
            # if keys is None:
            #     keys = src_root['data'].keys()
            # data = dict()
            # for key in keys:
            #     arr = src_root['data'][key]
            #     data[key] = arr[:]
            
            if np.linalg.norm(src_root['data']['goal_gripper_pcd'] - goal_gripper_pcd_arr) > 0.001:
                cprint(f'{zarr_path_step} has been modified', 'green')
                src_root['data']['goal_gripper_pcd'] = goal_gripper_pcd_arr
        
            # # remove the old data
            # cmd = "rm -r " + zarr_path_step
            # os.system(cmd)
        
            # # save new data
            # new_data_save_dir = zarr_path_step
            # # print("Saving new data to: ", new_data_save_dir)
            # save_data(
            #         data['point_cloud'], 
            #         data['state'], 
            #         data['gripper_pcd'], 
            #         data['action'], 
            #         data['goal_gripper_pcd'], 
            #         data['displacement_gripper_to_object'],
            #         new_data_save_dir
            # )


exp_dirs = [

        '/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_45448_2024-03-27-22-40-39/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-2-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46462_2024-03-27-23-35-10/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-4-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46732_2024-03-27-18-46-00/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46801_2024-03-27-20-37-05/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46874_2024-03-27-13-57-49/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46922_2024-03-27-19-42-45/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46966_2024-03-27-16-55-33/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_47570_2024-03-27-21-36-50/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_47578_2024-03-27-14-56-07/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_48700_2024-03-27-12-59-58/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
    
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_45526/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_45661/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first', # 2024-07-06-16-58-56
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_45694/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_45780/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_45910/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_45961/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_46408/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_46417/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_46440/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_46490/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_46762/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_46825/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_46893/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_47235/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_47281/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_47315/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_47529/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_47669/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first', # 2024-07-06-23-48-11
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_47944/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_48063/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_48177/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first', # 2024-07-07-02-20-38
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_48356/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_48623/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_48876/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_49025/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_49062/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_49132/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_49133/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects_2/open_the_door_40417/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects_2/open_the_door_41085/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects_2/open_the_door_41452/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects_2/open_the_door_45162/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects_2/open_the_door_45176/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects_2/open_the_door_45194/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects_2/open_the_door_45203/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects_2/open_the_door_45248/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects_2/open_the_door_45271/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first', # 2024-07-12-02-53-33
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects_2/open_the_door_45290/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects_2/open_the_door_45305/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
]

output_dirs = [

        '/scratch/chialiang/dp3_demo/0705-obj-41510',
        '/scratch/chialiang/dp3_demo/0705-obj-45448',
        '/scratch/chialiang/dp3_demo/0705-obj-46462',
        '/scratch/chialiang/dp3_demo/0705-obj-46732',
        '/scratch/chialiang/dp3_demo/0705-obj-46801',
        '/scratch/chialiang/dp3_demo/0705-obj-46874',
        '/scratch/chialiang/dp3_demo/0705-obj-46922',
        '/scratch/chialiang/dp3_demo/0705-obj-46966',
        '/scratch/chialiang/dp3_demo/0705-obj-47570',
        '/scratch/chialiang/dp3_demo/0705-obj-47578',
        '/scratch/chialiang/dp3_demo/0705-obj-48700',
        
        # g1
        '/scratch/chialiang/dp3_demo/0705-obj-45526',
        '/scratch/chialiang/dp3_demo/0705-obj-45661', # 2024-07-06-16-58-56
        '/scratch/chialiang/dp3_demo/0705-obj-45694',
        '/scratch/chialiang/dp3_demo/0705-obj-45780',
        '/scratch/chialiang/dp3_demo/0705-obj-45910',

        '/scratch/chialiang/dp3_demo/0705-obj-45961',
        '/scratch/chialiang/dp3_demo/0705-obj-46408',
        '/scratch/chialiang/dp3_demo/0705-obj-46417',
        '/scratch/chialiang/dp3_demo/0705-obj-46440',
        '/scratch/chialiang/dp3_demo/0705-obj-46490',

        # g2
        '/scratch/chialiang/dp3_demo/0705-obj-46762',
        '/scratch/chialiang/dp3_demo/0705-obj-46825',
        '/scratch/chialiang/dp3_demo/0705-obj-46893',
        '/scratch/chialiang/dp3_demo/0705-obj-47235',
        '/scratch/chialiang/dp3_demo/0705-obj-47281',

        '/scratch/chialiang/dp3_demo/0705-obj-47315',
        '/scratch/chialiang/dp3_demo/0705-obj-47529',
        '/scratch/chialiang/dp3_demo/0705-obj-47669', # 2024-07-06-23-48-11
        '/scratch/chialiang/dp3_demo/0705-obj-47944',
        '/scratch/chialiang/dp3_demo/0705-obj-48063',
        
        # g3
        '/scratch/chialiang/dp3_demo/0705-obj-48177', # 2024-07-07-02-20-38
        '/scratch/chialiang/dp3_demo/0705-obj-48356',
        '/scratch/chialiang/dp3_demo/0705-obj-48623',
        '/scratch/chialiang/dp3_demo/0705-obj-48876',
        '/scratch/chialiang/dp3_demo/0705-obj-49025',
        
        '/scratch/chialiang/dp3_demo/0705-obj-49062',
        '/scratch/chialiang/dp3_demo/0705-obj-49132',
        '/scratch/chialiang/dp3_demo/0705-obj-49133',
        '/scratch/chialiang/dp3_demo/0712-obj-40417',
        '/scratch/chialiang/dp3_demo/0712-obj-41085',

        # # g4
        '/scratch/chialiang/dp3_demo/0712-obj-41452',
        '/scratch/chialiang/dp3_demo/0712-obj-45162',
        '/scratch/chialiang/dp3_demo/0712-obj-45176',
        '/scratch/chialiang/dp3_demo/0712-obj-45194',
        '/scratch/chialiang/dp3_demo/0712-obj-45203',

        '/scratch/chialiang/dp3_demo/0712-obj-45248',
        '/scratch/chialiang/dp3_demo/0712-obj-45271', # 2024-07-12-02-53-33
        '/scratch/chialiang/dp3_demo/0712-obj-45290',
        '/scratch/chialiang/dp3_demo/0712-obj-45305',
]

for (exp_dir, output_dir) in zip(exp_dirs, output_dirs):
    fix_goal(exp_dir, output_dir)