import yaml
import os
import json
from scipy.spatial.transform import Rotation as R
from manipulation.utils import build_up_env, load_env, down_load_single_object
from copy import deepcopy
import pybullet as p
import numpy as np
from termcolor import cprint
np.random.seed(0)


def add_distractors_around_target(yaml_path, solution_path, task_name, save_path='local_exps/add_distractors/temp.yaml'):
    current_config = yaml.load(open(yaml_path, 'r'), Loader=yaml.FullLoader)
    temp_config = deepcopy(current_config)

    target_object_dir = None

    for dir in temp_config:
        if 'orientation' in dir.keys():
            true_orientation = dir['orientation']
            dir['orientation'] = str((0, 0, 0, 1))
        if 'type' in dir.keys() and dir['type'] == 'urdf':
            target_object_dir = dir
    
    if target_object_dir is None:
        raise ValueError('No target object found in the yaml file')

    # save the new configuration
    with open(save_path, 'w') as f:
        yaml.dump(temp_config, f)

    # build up the environment
    env, _ = build_up_env(
        save_path,
        solution_path,
        task_name,
        None,
        render=False,
        randomize=False,
        obj_id=0        
    )
    

    target_obj_name = target_object_dir['name'].lower()
    # get the bounding box of the target object
    target_obj_id = env.urdf_ids[target_obj_name]
    min_aabb, max_aabb = env.get_aabb(target_obj_id)
    x_range, y_range, z_range = max_aabb - min_aabb

    # import pdb; pdb.set_trace()

    env.close()

    # search for some distractors that are similar size to the target object
    left_obj_info, right_obj_info = search_distractor_side(x_range, z_range)

    if left_obj_info is not None:
        # add the left distractor
        bouding_box = left_obj_info['boundingBox']
        x_size = bouding_box['x']
        y_size = bouding_box['z']
        z_size = bouding_box['y']
        position_x_1 = min_aabb[0] + np.random.uniform(-0.1, 0.1) * x_size
        position_y_1 = max_aabb[1] + y_size + 0.1
        position_z_1 = 0

        left_obj_dir = {}
        left_obj_dir['all_uid'] = [left_obj_info['name']]
        left_obj_dir['uid'] = [left_obj_info['name']]
        left_obj_dir['type'] = 'mesh'
        left_obj_dir['lang'] = left_obj_info['name']
        left_obj_dir['name'] = 'left_distractor'
        left_obj_dir['center'] = str(tuple([position_x_1, position_y_1, position_z_1]))
        left_obj_dir['orientation'] = str((0, 0, 0, 1))
        left_obj_dir['size'] = float(np.sqrt(x_size**2 + y_size**2 + z_size**2))
        left_obj_dir['movable'] = False
        left_obj_dir['path'] = left_obj_info['name'] + '.obj'
        temp_config.append(left_obj_dir)


    if right_obj_info is not None:
        # add the right distractor
        bouding_box = right_obj_info['boundingBox']
        x_size = bouding_box['x']
        y_size = bouding_box['z']
        z_size = bouding_box['y']
        position_x_2 = min_aabb[0] + np.random.uniform(-0.1, 0.1) * x_range
        position_y_2 = min_aabb[1] - y_size - 0.1
        position_z_2 = 0

        right_obj_dir = {}
        right_obj_dir['all_uid'] = [right_obj_info['name']]
        right_obj_dir['uid'] = [right_obj_info['name']]
        right_obj_dir['type'] = 'mesh'
        right_obj_dir['lang'] = right_obj_info['name']
        right_obj_dir['name'] = 'right_distractor'
        right_obj_dir['center'] = str(tuple([position_x_2, position_y_2, position_z_2]))
        right_obj_dir['orientation'] = str((0, 0, 0, 1))
        right_obj_dir['size'] = float(np.sqrt(x_size**2 + y_size**2 + z_size**2))
        right_obj_dir['movable'] = False
        right_obj_dir['path'] = right_obj_info['name'] + '.obj'
        temp_config.append(right_obj_dir)

    # top_obj_info_list = search_distractor_top(x_range, y_range)
    # for top_obj_info in top_obj_info_list:
    #     position_x = min_aabb[0] + top_obj_info['relative_position_on_top'][0] * x_range
    #     position_y = min_aabb[1] + top_obj_info['relative_position_on_top'][1] * y_range
    #     position_z = max_aabb[2] + 0.05

    #     x_size = top_obj_info['boundingBox']['x']
    #     y_size = top_obj_info['boundingBox']['z']
    #     z_size = top_obj_info['boundingBox']['y']

    #     top_obj_dir = {}
    #     top_obj_dir['all_uid'] = [top_obj_info['name']]
    #     top_obj_dir['uid'] = [top_obj_info['name']]
    #     top_obj_dir['type'] = 'mesh'
    #     top_obj_dir['lang'] = top_obj_info['name']
    #     top_obj_dir['name'] = 'top_distractor'
    #     top_obj_dir['center'] = str(tuple([position_x, position_y, position_z]))
    #     top_obj_dir['orientation'] = str((0, 0, 0, 1))
    #     top_obj_dir['size'] = float(np.sqrt(x_size**2 + y_size**2 + z_size**2))
    #     top_obj_dir['movable'] = False
    #     top_obj_dir['path'] = top_obj_info['name'] + '.obj'
    #     temp_config.append(top_obj_dir)

    # save the new configuration at save_path
    with open(save_path, 'w') as f:
        yaml.dump(temp_config, f)

    env, _ = build_up_env(
        save_path,
        solution_path,
        task_name,
        None,
        render=False,
        randomize=False,
        obj_id=0, 
    )
    max_x = -100
    for obj_id in env.urdf_ids.values():
        if obj_id < 2: # skip robot and plane
            continue
        min_aabb_, max_aabb_ = env.get_aabb(obj_id)
        max_x = max(max_x, max_aabb_[0])
    
    # get the bounding box of left and right distractors
    if left_obj_info is not None:
        left_obj_id = env.urdf_ids[left_obj_dir['name']]
        min_aabb_, max_aabb_ = env.get_aabb(left_obj_id)
        y_min_ = min_aabb_[1]
        add_residual = y_min_ - max_aabb[1] - 0.05 + np.random.uniform(-0.02, 0.02)
        left_obj_dir['center'] = str(tuple([position_x_1, position_y_1 - add_residual, position_z_1]))
        current_config.append(left_obj_dir)

    if right_obj_info is not None:
        right_obj_id = env.urdf_ids[right_obj_dir['name']]
        min_aabb_, max_aabb_ = env.get_aabb(right_obj_id)
        y_max_ = max_aabb_[1]
        add_residual = min_aabb[1] - y_max_ - 0.05 + np.random.uniform(-0.02, 0.02)
        right_obj_dir['center'] = str(tuple([position_x_2, position_y_2 + add_residual, position_z_2]))
        current_config.append(right_obj_dir)

    temp_config = deepcopy(current_config)

    env.close()

    # save the new configuration at save_path
    with open(save_path, 'w') as f:
        yaml.dump(temp_config, f)
    
    # add wall to the background
    position_x = max_x + 0.7
    position_y = min_aabb[1] + y_range / 2
    position_z = 0
    background_wall = {}
    background_wall['all_uid'] = ['eeb27b0a4eb740c69607314971b88f70']
    background_wall['uid'] = ['eeb27b0a4eb740c69607314971b88f70']
    background_wall['type'] = 'mesh'
    background_wall['lang'] = 'wall'
    background_wall['name'] = 'background_wall'
    background_wall['center'] = str(tuple([position_x, position_y, position_z]))
    # rotate around x axis by 90 degree
    background_wall['orientation'] = str((0, 0, 0.7071067811865476, 0.7071067811865476))
    background_wall['size'] = 10
    background_wall['movable'] = False
    background_wall['path'] = 'wall.obj'
    temp_config.append(background_wall)

    obj_inside = search_distractor_inside(target_obj_id, save_path, solution_path, task_name)
    if obj_inside is not None:
        position_x, position_y, position_z = obj_inside['position']
        x_size = obj_inside['boundingBox']['x']
        y_size = obj_inside['boundingBox']['z']
        z_size = obj_inside['boundingBox']['y']

        inside_obj_dir = {}
        inside_obj_dir['all_uid'] = [obj_inside['name']]
        inside_obj_dir['uid'] = [obj_inside['name']]
        inside_obj_dir['type'] = 'mesh'
        inside_obj_dir['lang'] = obj_inside['name']
        inside_obj_dir['name'] = 'inside_distractor'
        inside_obj_dir['center'] = str(tuple([position_x, position_y, position_z]))
        inside_obj_dir['orientation'] = str((0, 0, 0, 1))
        inside_obj_dir['size'] = float(np.sqrt(x_size**2 + y_size**2 + z_size**2))
        inside_obj_dir['movable'] = False
        inside_obj_dir['path'] = obj_inside['name'] + '.obj'
        temp_config.append(inside_obj_dir)

    # save the new configuration at save_path
    with open(save_path, 'w') as f:
        yaml.dump(temp_config, f)

    # env, _ = build_up_env(
    #     save_path,
    #     solution_path,
    #     task_name,
    #     None,
    #     render=True,
    #     randomize=False,
    #     obj_id=0        
    # )
    # import pdb; pdb.set_trace()
    # env.close()
    cprint("Distractors added successfully!", "green")
    return save_path


def search_distractor_side(x_range, z_range, search_file_path='data/data_holodeck/09_23_combine_scale/objaverse_holodeck_database.json'):
    # load the json file that contains the object information
    with open(search_file_path, 'r') as f:
        search_file = json.load(f)

    all_objs = list(search_file.keys())[:50091]

    left_obj_info = None
    right_obj_info = None

    try_times = 0
    while try_times < 500:
        try_times += 1
        obj_id = np.random.randint(0, len(all_objs))
        obj_name = all_objs[obj_id]
        obj_info = search_file[obj_name]['assetMetadata']

        bouding_box = obj_info['boundingBox']
        x_size = bouding_box['x']
        y_size = bouding_box['z']
        z_size = bouding_box['y']

        if x_size < x_range * 1.5 and x_size > x_range * 0.5 and z_size < z_range * 1.5 and z_size > z_range * 0.5:
            if down_load_single_object(name=obj_info['name'], uids=[obj_info['name']], debug=True):
                if left_obj_info is None:
                    left_obj_info = obj_info
                elif right_obj_info is None:
                    right_obj_info = obj_info
                    break

    return left_obj_info, right_obj_info

def search_distractor_top(x_range, y_range, search_file_path='data/data_holodeck/09_23_combine_scale/objaverse_holodeck_database.json', return_one_obj=True, sample_times=500):
    # load the json file that contains the object information
    with open(search_file_path, 'r') as f:
        search_file = json.load(f)

    all_objs = list(search_file.keys())[:50091]

    top_obj_info_list = []
    try_times = 0
    first_obj = False
    while try_times < sample_times:
        try_times += 1
        obj_id = np.random.randint(0, len(all_objs))
        obj_name = all_objs[obj_id]
        obj_info = search_file[obj_name]['assetMetadata']

        bouding_box = obj_info['boundingBox']
        x_size = bouding_box['x']
        y_size = bouding_box['z']
        z_size = bouding_box['y']

        if x_size < x_range * 0.6 and x_size > x_range * 0.2 and y_size < y_range * 0.6 and y_size > y_range * 0.2 and z_size < 0.4:
            position = np.random.uniform(0.15, 0.85, 2)
            # clip the position to the range of the target object
            position[0] = np.clip(position[0], x_size / x_range / 2, 1 - x_size / x_range / 2)
            position[1] = np.clip(position[1], y_size / y_range / 2, 1 - y_size / y_range / 2)
            obj_info['relative_position_on_top'] = position
            top_obj_info_list.append(obj_info)
            first_obj = True
            break
    if first_obj == False or return_one_obj:
        return top_obj_info_list
    
    return []
    


def search_distractor_inside(target_obj_id, save_path, solution_path, task_name, search_file_path='data/data_holodeck/09_23_combine_scale/objaverse_holodeck_database.json'):
    with open(search_file_path, 'r') as f:
        search_file = json.load(f)
    # np.random.seed(0)
    all_objs = list(search_file.keys())[:50091]
    env, _ = build_up_env(
        save_path,
        solution_path,
        task_name,
        None,
        render=False,
        randomize=False,
        obj_id=0        
    )
    min_aabb, max_aabb = env.get_aabb(target_obj_id)
    x_range, y_range, z_range = max_aabb - min_aabb
    try_times = 0
    found_obj = False
    while try_times < 500:
        try_times += 1
        obj_id = np.random.randint(0, len(all_objs))
        obj_name = all_objs[obj_id]
        obj_info = search_file[obj_name]['assetMetadata']

        bouding_box = obj_info['boundingBox']
        x_size = bouding_box['x']
        y_size = bouding_box['z']
        z_size = bouding_box['y']
        if x_size < x_range * 0.4 and y_size < y_range * 0.4 and z_size < z_range * 0.4:
            # try to put the object inside the target object
            virtual_box = p.createCollisionShape(p.GEOM_BOX, halfExtents=[x_size, y_size, z_size], physicsClientId=env.id)
            virtual_vision_box = p.createVisualShape(p.GEOM_BOX, halfExtents=[x_size, y_size, z_size], rgbaColor=[1, 0, 0, 1], physicsClientId=env.id)
            box_id = p.createMultiBody(baseCollisionShapeIndex=virtual_box, baseVisualShapeIndex=virtual_vision_box, basePosition=[5, 5, 5], physicsClientId=env.id)
            
            # try to put the virtual box inside the target object
            for _ in range(100):
                position_x = np.random.uniform(min_aabb[0] + x_size + 0.1 * x_range, max_aabb[0] - x_size - 0.1 * x_range)
                position_y = np.random.uniform(min_aabb[1] + y_size + 0.1 * y_range, max_aabb[1] - y_size - 0.1 * y_range)
                position_z = np.random.uniform(min_aabb[2] + z_size + 0.1 * z_range, max_aabb[2] - z_size - 0.1 * z_range)
                p.resetBasePositionAndOrientation(box_id, [position_x, position_y, position_z], [0, 0, 0, 1], physicsClientId=env.id)
                
                if not p.getClosestPoints(box_id, target_obj_id, distance=0, physicsClientId=env.id):
                    if down_load_single_object(name=obj_info['name'], uids=[obj_info['name']], debug=True):
                        found_obj = True
                        break

            p.removeBody(box_id, physicsClientId=env.id)

        if found_obj:
            break

    env.close()
    if not found_obj:
        return None

    obj_info['position'] = [position_x - x_size/2, position_y, position_z - z_size/2]
    obj_info['orientation'] = [0, 0, 0, 1]

    return obj_info
        


if __name__ == '__main__':
    yaml_path = '/media/ziyu/Elements/workspace/RoboGen-sim2real/data/temp/nihao/2024-05-11-00-56-52/task_config.yaml'
    solution_path = 'data/temp/nihao'
    task_name = 'grasp_the_door_handle'
    save_path = 'local_exps/add_distractors/temp.yaml'
    # env, _ = build_up_env(
    #     yaml_path,
    #     solution_path,
    #     task_name,
    #     None,
    #     render=True,
    #     randomize=False,
    #     obj_id=0        
    # )
    # load_env(env, load_path="/media/ziyu/Elements/workspace/RoboGen-sim2real/data/temp/nihao/2024-05-11-00-56-52/grasp_the_door_handle_primitive/states/state_0.pkl")
    add_distractors_around_target(yaml_path, solution_path, task_name, save_path=save_path)

    env, _ = build_up_env(
        save_path,
        solution_path,
        task_name,
        None,
        render=True,
        randomize=False,
        obj_id=0        
    )
    from manipulation.robogen_wrapper import RobogenPointCloudWrapper
    rpy = [[0, 0, -45], [0, 0, -135]]
    simulator = RobogenPointCloudWrapper(env, 
        'storagefurniture', rpy_mean_list=rpy, seed=0, in_gripper_frame=0, 
        gripper_num_points=0, add_contact=0, num_points=4500,
        use_joint_angle=0, use_segmask=0, only_handle_points=0, 
        observation_mode='act3d')
    obs = simulator._get_observation(only_object=False)
    import pdb; pdb.set_trace()