
from test_PointNet2.all_data import *
from scripts.datasets.randomize_partition_10_obj import *
from scripts.datasets.randomize_partition_50_obj import *
from scripts.datasets.randomize_partition_100_obj import *
from scripts.datasets.randomize_partition_200_obj import *
from termcolor import cprint
import os
from tqdm import tqdm
import pickle
import numpy as np

dataset_prefix = '/scratch/yufeiw2/dp3_demo'

def get_statistics(all_obj_paths, high_level=True):
    all_zarr_paths = []
    for obj_path in all_obj_paths:
        all_subfolder = os.listdir(obj_path)
        for s in ['action_dist', 'demo_rgbs', 'all_demo_path.txt', 'meta_info.json', 'example_pointcloud']:
            if s in all_subfolder:
                all_subfolder.remove(s)
        all_subfolder = sorted(all_subfolder)
        beg_ratio = 0
        end_ratio = 1
        beg = int(beg_ratio * len(all_subfolder))
        end = int(end_ratio * len(all_subfolder))
        # high level
        if high_level:
            end = min(end, 75)
        else:
            end = min(end, 1000)
        all_subfolder = all_subfolder[beg:end]
        all_zarr_paths += [os.path.join(obj_path, s) for s in all_subfolder]

    # cprint('Preparing all zarr paths', 'green')
    episode_lengths = []

    for idx, zarr_path in enumerate((all_zarr_paths)):
        all_substeps = os.listdir(zarr_path)
        # all_substeps = sorted(all_substeps, key=lambda x: int(x.split('.')[0]))
        episode_lengths.append(len(all_substeps))


    episode_lengths = np.array(episode_lengths)
    num_episodes = len(episode_lengths)
    num_datapoints = np.sum(episode_lengths)
    cprint(f'Finished preparing all zarr paths with total trajectories {num_episodes} total datapoints {num_datapoints}', 'white')


def get_all_objs(num_train_objects):
    if num_train_objects == 'debug':
        all_obj_paths = [f'{dataset_prefix}/{save_data_name_0}']
    elif num_train_objects == '10':
        all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(10)]
    elif num_train_objects == '50':
        all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(50)]
    elif num_train_objects == '100':
        all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(100)]
    elif num_train_objects == '200':
        all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(200)]
    elif num_train_objects == '300':
        all_zarr_paths_part_1 = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(246)]
        all_subfolders = sorted(os.listdir(dataset_prefix))
        object_other_categories_no_cam_rand = [x for x in all_subfolders if "1121-other-cat-no-cam-rand" in x]
        all_zarr_paths_part_2 = [f"{dataset_prefix}/{x}" for x in object_other_categories_no_cam_rand]
        all_obj_paths = all_zarr_paths_part_1 + all_zarr_paths_part_2
    
    elif num_train_objects == 'camera_random_10_obj_high_level':
        all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["camera_random_10_save_data_name_{}".format(i)]) for i in range(20)]
    elif num_train_objects == 'camera_random_50_obj_high_level':
        all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["camera_random_50_save_data_name_{}".format(i)]) for i in range(87)]
    elif num_train_objects == 'camera_random_100_obj_high_level':
        all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["camera_random_100_save_data_name_{}".format(i)]) for i in range(175)]
    elif num_train_objects == 'camera_random_200_obj_high_level':
        all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["camera_random_200_save_data_name_{}".format(i)]) for i in range(350)]
    elif num_train_objects == 'camera_random_500_obj_high_level' or num_train_objects == "camera_random_300_obj_high_level":
        all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(462)]

    return all_obj_paths

### no camera randomization
# for num_train_objects in ["10", "50", "100", "200", "300"]:
#     print(f"================================== {num_train_objects} ======================================")
#     all_obj_path = get_all_objs(num_train_objects)
#     get_statistics(all_obj_path)
    
### with camera randomization
for num_train_objects in ["camera_random_10_obj_high_level", "camera_random_50_obj_high_level", "camera_random_100_obj_high_level", "camera_random_200_obj_high_level", "camera_random_300_obj_high_level"]:
    print(f"================================== {num_train_objects} ======================================")
    all_obj_path = get_all_objs(num_train_objects)
    get_statistics(all_obj_path)

### for the real world dataset
all_obj_path_part_1 = get_all_objs("camera_random_300_obj_high_level")
real_world_camera_500_paths = os.listdir("/scratch/yufeiw2/dp3_demo_real_world_noise_pcd_clean_distorted_goal")
real_world_camera_500_paths = sorted(real_world_camera_500_paths)
real_world_camera_500_paths = [os.path.join("/scratch/yufeiw2/dp3_demo_real_world_noise_pcd_clean_distorted_goal", x) for x in real_world_camera_500_paths]
all_obj_path = all_obj_path_part_1 + real_world_camera_500_paths
get_statistics(all_obj_path)

    
    
### result
# ================================== 10 ======================================
# Finished preparing all zarr paths with total trajectories 750 total datapoints 79809
# ================================== 50 ======================================
# Finished preparing all zarr paths with total trajectories 3669 total datapoints 402986
# ================================== 100 ======================================
# Finished preparing all zarr paths with total trajectories 6444 total datapoints 701875
# ================================== 200 ======================================
# Finished preparing all zarr paths with total trajectories 11795 total datapoints 1285050
# ================================== 300 ======================================
# Finished preparing all zarr paths with total trajectories 15998 total datapoints 1760077
# ================================== camera_random_50_obj_high_level ======================================
# Finished preparing all zarr paths with total trajectories 4656 total datapoints 501980
# ================================== camera_random_100_obj_high_level ======================================
# Finished preparing all zarr paths with total trajectories 8749 total datapoints 958008
# ================================== camera_random_200_obj_high_level ======================================
# Finished preparing all zarr paths with total trajectories 17893 total datapoints 1977401
# ================================== camera_random_300_obj_high_level ======================================
# Finished preparing all zarr paths with total trajectories 22918 total datapoints 2546438
