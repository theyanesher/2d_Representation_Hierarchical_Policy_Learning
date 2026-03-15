import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
from termcolor import cprint
import numpy as np
from tqdm import tqdm
import pickle
import random
from diffuser_actor_3d.robogen_utils import get_gripper_pos_orient_from_4_points
from scipy.spatial.transform import Rotation as R

def get_4_points_from_gripper_pos_orient(gripper_pos, gripper_orn, cur_joint_angle):
    original_gripper_pcd = np.array([[ 0.5648266,   0.05482348,  0.34434554],
        [ 0.5642125,   0.02702148,  0.2877661 ],
        [ 0.53906703,  0.01263776,  0.38347825],
        [ 0.54250515, -0.00441092,  0.32957944]]
    )
    original_gripper_orn = np.array([0.21120763,  0.75430543, -0.61925177, -0.05423936])
    
    gripper_pcd_right_finger_closed = np.array([ 0.55415434,  0.02126799,  0.32605097])
    gripper_pcd_left_finger_closed = np.array([ 0.54912525,  0.01839125,  0.3451934 ])
    gripper_pcd_closed_finger_angle = 2.6652539383870777e-05
 
    original_gripper_pcd[1] = gripper_pcd_right_finger_closed + (original_gripper_pcd[1] - gripper_pcd_right_finger_closed) / (0.04 - gripper_pcd_closed_finger_angle) * (cur_joint_angle - gripper_pcd_closed_finger_angle)
    original_gripper_pcd[2] = gripper_pcd_left_finger_closed + (original_gripper_pcd[2] - gripper_pcd_left_finger_closed) / (0.04 - gripper_pcd_closed_finger_angle) * (cur_joint_angle - gripper_pcd_closed_finger_angle)
 
    goal_R = R.from_quat(gripper_orn)
    original_R = R.from_quat(original_gripper_orn)
    rotation_transfer = goal_R * original_R.inv()
    original_pcd = original_gripper_pcd - original_gripper_pcd[3]
    rotated_pcd = rotation_transfer.apply(original_pcd)
    gripper_pcd = rotated_pcd + gripper_pos
    return gripper_pcd.astype(np.float32)

def change_goal_gripper_pcd_to_open(goal_gripper_pcd):
    goal_pos, goal_orient = get_gripper_pos_orient_from_4_points(goal_gripper_pcd)
    open_joint_angle = 0.04
    open_gripper_pcd = get_4_points_from_gripper_pos_orient(goal_pos, goal_orient, open_joint_angle)
    return open_gripper_pcd

categories = ['bucket', 'faucet', 'foldingchair', 'laptop', 'stapler', 'toilet']
num_cats = 15

articulated_new = [
    # Bucket
    "bucket_100443", "bucket_100444", "bucket_100452", "bucket_100454", "bucket_100460", "bucket_100461",
    "bucket_100462", "bucket_100469", "bucket_100472", "bucket_102352", "bucket_102358", "bucket_102365",

    # Faucet
    "faucet_148", "faucet_152", "faucet_153", "faucet_154", "faucet_168", "faucet_811", "faucet_822",
    "faucet_857", "faucet_908", "faucet_929", "faucet_1028", "faucet_1052", "faucet_1053", "faucet_1288",
    "faucet_1343", "faucet_1370", "faucet_1466", "faucet_1492", "faucet_1528", "faucet_1626", "faucet_1633",
    "faucet_1646", "faucet_1668", "faucet_1741", "faucet_1794", "faucet_1795", "faucet_1802", "faucet_1885",
    "faucet_1901", "faucet_1903", "faucet_1925", "faucet_1961", "faucet_1986", "faucet_2054",

    # Foldingchair
    "foldingchair_100531", "foldingchair_100532", "foldingchair_100557", "foldingchair_100561",
    "foldingchair_100562", "foldingchair_100568", "foldingchair_100579", "foldingchair_100586",
    "foldingchair_100590", "foldingchair_100599", "foldingchair_100600", "foldingchair_100608",
    "foldingchair_100609", "foldingchair_100611", "foldingchair_100616", "foldingchair_102255",
    "foldingchair_102263", "foldingchair_102269", "foldingchair_102314",

    # Laptop
    "laptop_9968", "laptop_9992", "laptop_9996", "laptop_10040", "laptop_10098", "laptop_10101",
    "laptop_10238", "laptop_10243", "laptop_10248", "laptop_10269", "laptop_10270", "laptop_10280",
    "laptop_10289", "laptop_10305", "laptop_10306", "laptop_10383", "laptop_10626", "laptop_10697",
    "laptop_10885", "laptop_10915", "laptop_11075", "laptop_11156", "laptop_11242", "laptop_11248",
    "laptop_11395", "laptop_11405", "laptop_11406", "laptop_11429", "laptop_11477", "laptop_11581",
    "laptop_11586", "laptop_11691", "laptop_11778", "laptop_11876", "laptop_11888", "laptop_11945",
    "laptop_12073",

    # Stapler
    "stapler_103099", "stapler_103100", "stapler_103104", "stapler_103111", "stapler_103113",
    "stapler_103271", "stapler_103275", "stapler_103276", "stapler_103280", "stapler_103292",
    "stapler_103293", "stapler_103297", "stapler_103299", "stapler_103301", "stapler_103303",
    "stapler_103305", "stapler_103789", "stapler_103792",

    # Toilet
    "toilet_102622", "toilet_102630", "toilet_102634", "toilet_102645", "toilet_102648",
    "toilet_102651", "toilet_102652", "toilet_102654", "toilet_102658", "toilet_102663",
    "toilet_102666", "toilet_102667", "toilet_102668", "toilet_102669", "toilet_102670",
    "toilet_102675", "toilet_102676", "toilet_102677", "toilet_102687", "toilet_102689",
    "toilet_102692", "toilet_102694", "toilet_102697", "toilet_102699", "toilet_102701",
    "toilet_102703", "toilet_102707", "toilet_102708", "toilet_103234"
    ]

articulated_new_replace = [
    "laptop_10040", "laptop_10248", "laptop_10305", "laptop_10885", "laptop_11248", "laptop_11477", "laptop_11876", "laptop_9992", "toilet_102692",
    "laptop_10098", "laptop_10269", "laptop_10306", "laptop_10915", "laptop_11395", "laptop_11581", "laptop_11888", "laptop_9996", "toilet_102694",
    "laptop_10101", "laptop_10270", "laptop_10383", "laptop_11075", "laptop_11405", "laptop_11586", "laptop_11945", "toilet_102630",
    "laptop_10238", "laptop_10280", "laptop_10626", "laptop_11156", "laptop_11406", "laptop_11691", "laptop_12073", "toilet_102667",
    "laptop_10243", "laptop_10289", "laptop_10697", "laptop_11242", "laptop_11429", "laptop_11778", "laptop_9968", "toilet_102668"
]

class PointNetDatasetFromDisk(torch.utils.data.Dataset):
    def __init__(self, all_obj_paths, beg_ratio=0, end_ratio=0.9, eval_episode=None, only_first_stage=False, is_pickle=False, use_all_data=False, 
                 conditioning_on_demo=False, camera_frame=False, goal_always_open=False, use_rgb=False, use_dino=False, pred_gripper_width=False, gripper_width_scale_factor=1.0):
        self.all_obj_paths = all_obj_paths
        self.beg_ratio = beg_ratio
        self.end_ratio = end_ratio
        self.is_pickle = is_pickle
        self.cat_counts = np.zeros(num_cats)  # +1 for the background category
        self.use_all_data = use_all_data
        self.conditioning_on_demo = conditioning_on_demo
        self.goal_always_open = goal_always_open 
        self.use_rgb = use_rgb
        self.use_dino = use_dino
        self.pred_gripper_width = pred_gripper_width
        self.gripper_width_scale_factor = gripper_width_scale_factor
        
        self.camera_frame = camera_frame
        if self.camera_frame:
            project_dir = os.environ['PROJECT_DIR']
            with open(os.path.join(project_dir, "data/world_to_camera_T.pkl"), "rb") as f:
                self.world_to_camera_T = pickle.load(f)
        
        if only_first_stage:
            cprint('======= ONLY FIRST STAGE =======', 'red')

        if eval_episode is not None:
            cprint('======= EVAL MODE =======', 'red')
            cprint(f'Only evaluating the first observation of {eval_episode} episodes', 'red')
            
        # TODO for conditioning
        # for each trajectory of the object, record the grasping pose and opening pose. Store as a dictionary maybe, key is object_traj-id

        self.all_zarr_paths = []
        self.all_zarr_categories = []
        self.episode_idx_to_obj_id = {}
        self.obj_id_to_all_episodes_indices = {}
        episode_idx = 0
        for obj_path in all_obj_paths:
            all_subfolder = os.listdir(obj_path)
            cat_idx = 0
            for i, cat in enumerate(categories):
                if cat in obj_path:
                    cat_idx = i + 1
                    break
            if 'invert' in obj_path:
                if cat_idx == 0:
                    cat_idx = 7
                else:
                    cat_idx += 5
            if cat_idx == 0:
                if 'grasp' in obj_path:
                    cat_idx = 12
                if 'top' in obj_path:
                    cat_idx = 13
                if 'inside' in obj_path:
                    cat_idx = 14
                if 'grasp_for_pap' in obj_path:
                    cat_idx = random.choice([13, 14])
            
            if 'aloha' in obj_path and 'plate' in obj_path:
                print("using sriram plate category")
                cat_idx = 13 ### put plate on top of the category 13 (top)
            if 'aloha' in obj_path and 'towel' in obj_path:
                print("using sriram towel category")
                cat_idx = 0 ### use open 
            if 'mimicgen' in obj_path:
                print("using mimicgen task")
                cat_idx = 0
                    
            # storage furniture, bucket, faucet, foldingchair, laptop, stapler, toilet, invert storage furniture, invert foldingchair, invert laptop, invert stapler, invert toilet
            for s in ['action_dist', 'demo_rgbs', 'all_demo_path.txt', 'meta_info.json', 'example_pointcloud']:
                if s in all_subfolder:
                    all_subfolder.remove(s)
            all_subfolder = sorted(all_subfolder)
            beg = int(beg_ratio * len(all_subfolder))
            end = int(end_ratio * len(all_subfolder))
            if not self.use_all_data:
                end = min(end, 75)
            if eval_episode is not None:
                end = beg + eval_episode
            all_subfolder = all_subfolder[beg:end]
            self.all_zarr_paths += [os.path.join(obj_path, s) for s in all_subfolder]
            self.all_zarr_categories += [cat_idx for s in all_subfolder]
            this_obj_episode_beg = episode_idx
            for s in all_subfolder:
                self.episode_idx_to_obj_id[episode_idx] = obj_path
                episode_idx += 1
            this_obj_episode_end = episode_idx
            self.obj_id_to_all_episodes_indices[obj_path] = [i for i in range(this_obj_episode_beg, this_obj_episode_end)]            

        cprint('Preparing all zarr paths', 'green')
        self.episode_lengths = []
        self.episode_idx_to_grasp_frame_idx = {}
        self.episode_idx_to_open_frame_idx = {}
        for idx, zarr_path in enumerate(tqdm(self.all_zarr_paths)):
            cat_idx = self.all_zarr_categories[idx]
            all_substeps = os.listdir(zarr_path)
            if is_pickle:
                all_substeps = [s for s in all_substeps if s.endswith('.pkl')]
            else:
                all_substeps = [s for s in all_substeps if s.endswith('.npz')]
            all_substeps = sorted(all_substeps, key=lambda x: int(x.split('.')[0]))
                
            first_goal = None

            # for i, substep in enumerate(all_substeps):
            #     if eval_episode is not None and i >=1:
            #         self.episode_lengths.append(i)
            #         self.cat_counts[cat_idx] += i
            #         break

            #     substep_path = os.path.join(zarr_path, substep)
            #     with open(substep_path, 'rb') as f:
            #         try:
            #             data = pickle.load(f)
            #         except:
            #             print(substep_path)
            #     action = data['action'][:]

            #     current_goal = data['goal_gripper_pcd'][:]
            #     if first_goal is None:
            #         first_goal = current_goal
            #     elif only_first_stage and not np.allclose(first_goal, current_goal):
            #         self.episode_lengths.append(i)
            #         self.cat_counts[cat_idx] += i
            #         break
                
            #     if not np.allclose(first_goal, current_goal):
            #         self.episode_idx_to_grasp_frame_idx[idx] = i
            
            # # assume -10 erases all the distorted goal. This is just an approximation. 
            # self.episode_idx_to_open_frame_idx[idx] = len(all_substeps) - 1 #- 10

            
            if not only_first_stage and eval_episode is None:
                self.episode_lengths.append(len(all_substeps))
                self.cat_counts[cat_idx] += len(all_substeps)
                
        self.episode_lengths = np.array(self.episode_lengths)
        self.accumulated_episode_lengths = np.cumsum(self.episode_lengths)
        cprint(f'Finished preparing all zarr paths with total datapoints: {self.accumulated_episode_lengths[-1]}', 'green')
        self.class_weights = [1.0 / count if count > 0 else 0.0 for count in self.cat_counts]
        self.class_weights = np.array(self.class_weights)
        num_existing_classes = np.sum(self.class_weights > 0)
        self.class_weights *= np.sum(self.cat_counts) / num_existing_classes  # normalize to have sum of weights equal to number of classes
        print(f'Class weights: {self.class_weights}')
        print(f'Cat_counts: {self.cat_counts}')

        self.cat_idxs = np.repeat(self.all_zarr_categories, self.episode_lengths)
        self.all_weights = self.class_weights[self.cat_idxs]
        self.all_weights /= np.sum(self.all_weights)  # normalize weights to sum to 1

    def __len__(self):
        return self.accumulated_episode_lengths[-1]
    
    def transform_pcd_to_camera_frame(self, pcd):
        pcd_homo = np.concatenate([pcd, np.ones((pcd.shape[0], 1))], axis=1)
        pcd_cam = self.world_to_camera_T @ pcd_homo.T 
        pcd_cam = pcd_cam.T
        pcd_cam = pcd_cam[:, :3]
        # change to cgn coordinate system
        pcd_cam[:, 0] = -pcd_cam[:, 0]
        pcd_cam[:, 2] = -pcd_cam[:, 2]
        
        return pcd_cam.astype(np.float32)
    
    def read_pickle_data(self, episode_idx, step_idx):
        step_path = os.path.join(self.all_zarr_paths[episode_idx], str(step_idx) + '.pkl')
        cat_idx = self.all_zarr_categories[episode_idx]
        weight = self.class_weights[cat_idx]
        with open(step_path, 'rb') as f:
            data = pickle.load(f)
        pointcloud = data['point_cloud'][:][0].astype(np.float32)
        gripper_pcd = data['gripper_pcd'][:][0].astype(np.float32)
        goal_gripper_pcd = data['goal_gripper_pcd'][:][0].astype(np.float32)
        
        if self.camera_frame:
            pointcloud = self.transform_pcd_to_camera_frame(pointcloud)
            gripper_pcd = self.transform_pcd_to_camera_frame(gripper_pcd)
            goal_gripper_pcd = self.transform_pcd_to_camera_frame(goal_gripper_pcd)
        
        return pointcloud, gripper_pcd, goal_gripper_pcd, cat_idx, weight, {}  
    
    def read_numpy_data(self, episode_idx, step_idx):    
        step_path = os.path.join(self.all_zarr_paths[episode_idx], str(step_idx) + '.npz')
        cat_idx = self.all_zarr_categories[episode_idx]
        weight = self.class_weights[cat_idx]
        data = np.load(step_path, allow_pickle=True)
        pointcloud = data['point_cloud'][:][0].astype(np.float32)
        gripper_pcd = data['gripper_pcd'][:][0].astype(np.float32)
        goal_gripper_pcd = data['goal_gripper_pcd'][:][0].astype(np.float32)
        
        extra = {}
        if self.use_rgb:
            rgb = data['rgb_values'][:][0].astype(np.float32)
            rgb_gripper = np.ones((gripper_pcd.shape[0], 3), dtype=np.float32)
            extra['rgb'] = rgb
            extra['rgb_gripper'] = rgb_gripper
        if self.use_dino:
            dino_features = data['rgb_features'][:][0].astype(np.float32)
            dino_features_gripper = np.ones((gripper_pcd.shape[0], dino_features.shape[1]), dtype=np.float32)
            extra['dino_features'] = dino_features
            extra['dino_features_gripper'] = dino_features_gripper
            
        return pointcloud, gripper_pcd, goal_gripper_pcd, cat_idx, weight, extra  

    def __getitem__(self, idx):
        # TODO for conditioning:
        # after we gete episode_idx, figure out which object this episode is from, and randomly sample another episode from this same object.
        # return additionally the grasping and opening pose of this other trajectory for this object. 
        
        episode_idx = np.searchsorted(self.accumulated_episode_lengths, idx, side='right')
        start_idx = idx - self.accumulated_episode_lengths[episode_idx]

        if start_idx < 0:
            start_idx += self.episode_lengths[episode_idx]
            
        if self.is_pickle:
            pointcloud, gripper_pcd, goal_gripper_pcd, cat_idx, weight, extra = self.read_pickle_data(episode_idx, start_idx)
        else:
            pointcloud, gripper_pcd, goal_gripper_pcd, cat_idx, weight, extra = self.read_numpy_data(episode_idx, start_idx)
            
        if self.pred_gripper_width:
            goal_gripper_width = np.linalg.norm(goal_gripper_pcd[1] - goal_gripper_pcd[2]) * self.gripper_width_scale_factor
        else:
            goal_gripper_width = 0
            
        extra['goal_gripper_width'] = goal_gripper_width
        
        if self.goal_always_open or self.pred_gripper_width:
            # print("change gripper to be fully open")
            goal_gripper_pcd = change_goal_gripper_pcd_to_open(goal_gripper_pcd)
            
        return pointcloud, gripper_pcd, goal_gripper_pcd, cat_idx, weight, extra
    
    
from test_PointNet2.all_data import *
from scripts.datasets.randomize_partition_10_obj import *
from scripts.datasets.randomize_partition_50_obj import *
from scripts.datasets.randomize_partition_100_obj import *
from scripts.datasets.randomize_partition_200_obj import *

def get_dataset_from_pickle(all_obj_paths=None, beg_ratio=0, end_ratio=0.9, eval_episode=None, only_first_stage=False, 
                            use_all_data=False, use_combined_action=False, dataset_prefix=None, num_train_objects=200, 
                            is_pickle=True,
                            predict_two_goals=False, conditioning_on_demo=False, camera_frame=False, goal_always_open=False, 
                            use_rgb=False, use_dino=False, pred_gripper_width=False, gripper_width_scale_factor=1.0):
    
    if dataset_prefix is None:
        dataset_prefix='/scratch/chialiang/dp3_demo'
        if use_combined_action:
            dataset_prefix='/scratch/chialiang/dp3_demo_combine_2_new'
    
    if all_obj_paths is None:
        num_train_objects = str(num_train_objects)
        print(" ", num_train_objects)
        print("num_train_objects: ", num_train_objects)
        print("num_train_objects: ", num_train_objects)
        print("num_train_objects: ", num_train_objects)
        print("num_train_objects: ", num_train_objects)
        if num_train_objects == 'test':
            dataset_prefix = '/tmp/new_7_category_real_cam'
            articulated_real_cam = sorted(os.listdir(dataset_prefix))
            ### only use folders not starting with digit
            articulated_real_cam = [f for f in articulated_real_cam if not f[0].isdigit()]
            articulated_real_cam = [os.path.join(dataset_prefix, x) for x in articulated_real_cam]
            all_obj_paths = [articulated_real_cam[0]]
            
        elif num_train_objects == 'articubot_and_reset':
            dataset_prefix = "/tmp/dp3_demo_clean_distorted_goal"
            non_real_world_camera_500_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(463)]
            
            dataset_prefix = "/tmp/articubot_all_reset_1203"
            reset_path = sorted(os.listdir(dataset_prefix))
            reset_path = [os.path.join(dataset_prefix, x) for x in reset_path]
            
            all_obj_paths = non_real_world_camera_500_paths + reset_path        
            
        elif num_train_objects == 'aritucbot_new_cat_camera_random_close':
            ### articubot with camera randomization
            dataset_prefix = "/tmp/dp3_demo_clean_distorted_goal"
            # non_real_world_camera_500_paths = sorted(os.listdir(dataset_prefix))
            # non_real_world_camera_500_paths = [os.path.join(dataset_prefix, x) for x in non_real_world_camera_500_paths]
            non_real_world_camera_500_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(463)]

            ### articubot with real camera randomization
            dataset_prefix = "/tmp/dp3_demo_real_world_noise_pcd_clean_distorted_goal"
            real_world_camera_500_paths = sorted(os.listdir(dataset_prefix))
            real_world_camera_500_paths = [os.path.join(dataset_prefix, x) for x in real_world_camera_500_paths]
            
            ### new category
            dataset_prefix = '/tmp/articulated'
            articulated = sorted(os.listdir(dataset_prefix))
            # articulated = [os.path.join(dataset_prefix, x) for x in articulated]
            articulated = ["{}/{}".format(dataset_prefix, name) for name in articulated_new]

            ### new category with camera randomization
            dataset_prefix = '/tmp/new_7_category_random_cam'
            articulated_random_cam = sorted(os.listdir(dataset_prefix))
            ### only use folders not starting with digit
            articulated_random_cam = [f for f in articulated_random_cam if not f[0].isdigit()]
            articulated_random_cam = [os.path.join(dataset_prefix, x) for x in articulated_random_cam]
            
            ### new category with real world randomization
            dataset_prefix = '/tmp/new_7_category_real_cam'
            articulated_real_cam = sorted(os.listdir(dataset_prefix))
            ### only use folders not starting with digit
            articulated_real_cam = [f for f in articulated_real_cam if not f[0].isdigit()]
            articulated_real_cam = [os.path.join(dataset_prefix, x) for x in articulated_real_cam]
            
            ### dagger on new categories
            dataset_prefix = '/tmp/dp3_demo_weighted_full_dagger'
            articulated_dagger = sorted(os.listdir(dataset_prefix))
            ### only use folders not starting with digit
            articulated_dagger = [f for f in articulated_dagger if not f[0].isdigit()]
            articulated_dagger = [os.path.join(dataset_prefix, x) for x in articulated_dagger]
            
            ### close data
            dataset_prefix = '/tmp/invert_push'
            close_data = sorted(os.listdir(dataset_prefix))
            close_data = [os.path.join(dataset_prefix, x) for x in close_data]
            # close_data = []

            all_obj_paths = non_real_world_camera_500_paths + real_world_camera_500_paths + articulated + \
                articulated_random_cam + articulated_real_cam + articulated_dagger + close_data
                
            print("all obj paths: ============================================")
            print(all_obj_paths)
            print("all obj paths: ============================================")
            
        elif num_train_objects == 'grasping':
            dataset_prefix = '/tmp/grasping/gen_grasp_1017'
            all_obj_paths = sorted(os.listdir(dataset_prefix))
            all_obj_paths = [os.path.join(dataset_prefix, x) for x in all_obj_paths]
            print("all_obj_paths: ", all_obj_paths)
            
        elif num_train_objects in ['pick_and_place', 'pick_and_place_new_1024', 'pick_and_place_grasp', "pick_and_place_grasp_plus_grasp_pap", 
                                   "pick_and_place_new_grasp_and_old_place"]:
            if num_train_objects == 'pick_and_place':
                dataset_prefix = ["top", "inside_whole_1", "inside_whole", "inside_link_2", "inside_link_1", "inside_link"]
            elif num_train_objects == 'pick_and_place_new_1024':
                dataset_prefix = ["top_cgn_1204", "inside_link_cgn_1204", "inside_whole_cgn_1204"]
            elif num_train_objects == 'pick_and_place_grasp':
                dataset_prefix = ["top_cgn_grasp", "inside_link_cgn_grasp", "inside_whole_cgn_grasp"]
            elif num_train_objects == 'pick_and_place_grasp_plus_grasp_pap':
                dataset_prefix = ["top_cgn_grasp", "inside_link_cgn_grasp", "inside_whole_cgn_grasp", "grasp_for_pap"]
            elif num_train_objects == 'pick_and_place_new_grasp_and_old_place':
                dataset_prefix = ["top_cgn_grasp_grasp_only", "inside_link_cgn_grasp_grasp_only", "inside_whole_cgn_grasp_grasp_only", "top_cgn_1204_place_only", "inside_link_cgn_1204_place_only", "inside_whole_cgn_1204_place_only"]
                
            all_pick_place_data = []
            for name in dataset_prefix:
                path = f"/tmp/pick_and_place/{name}"
                all_data = sorted(os.listdir(path))
                all_data = [os.path.join(path, x) for x in all_data]
                all_pick_place_data.extend(all_data)
                
            all_obj_paths = all_pick_place_data
            print("all_obj_paths: ", all_obj_paths)
            
        elif "pick_and_place" in num_train_objects and len(num_train_objects) > len("pick_and_place"):
                     
            if "25_percent" in num_train_objects:
                import random
                random.seed(0)
                print("using 25 percent of all data!!!!")
                print("using 25 percent of all data!!!!")
                print("using 25 percent of all data!!!!")
                print("using 25 percent of all data!!!!")
                print("using 25 percent of all data!!!!")
            if "50_percent" in num_train_objects:
                import random
                random.seed(0)
                print("using 50 percent of all data!!!!")
                print("using 50 percent of all data!!!!")
                print("using 50 percent of all data!!!!")
                print("using 50 percent of all data!!!!")
                print("using 50 percent of all data!!!!")
   
            ### articubot with camera randomization
            dataset_prefix = "/project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/data/multitask_all_training_data/dp3_demo_clean_distorted_goal"
            non_real_world_camera_500_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(463)]
            if "25_percent" in num_train_objects:
                num_non_real = int(0.25 * len(non_real_world_camera_500_paths))
                random_indices = random.sample(range(len(non_real_world_camera_500_paths)), num_non_real)
                non_real_world_camera_500_paths = [non_real_world_camera_500_paths[index] for index in random_indices]
            if "50_percent" in num_train_objects:
                num_non_real = int(0.5 * len(non_real_world_camera_500_paths))
                random_indices = random.sample(range(len(non_real_world_camera_500_paths)), num_non_real)
                non_real_world_camera_500_paths = [non_real_world_camera_500_paths[index] for index in random_indices]

            ### articubot with real camera randomization
            dataset_prefix = "/tmp/dp3_demo_real_world_noise_pcd_clean_distorted_goal"
            real_world_camera_500_paths = sorted(os.listdir(dataset_prefix))
            real_world_camera_500_paths = [os.path.join(dataset_prefix, x) for x in real_world_camera_500_paths]
            if "25_percent" in num_train_objects:
                num_real = int(0.25 * len(real_world_camera_500_paths))
                random_indices = random.sample(range(len(real_world_camera_500_paths)), num_real)
                real_world_camera_500_paths = [real_world_camera_500_paths[index] for index in random_indices]
            if "50_percent" in num_train_objects:
                num_real = int(0.5 * len(real_world_camera_500_paths))
                random_indices = random.sample(range(len(real_world_camera_500_paths)), num_real)
                real_world_camera_500_paths = [real_world_camera_500_paths[index] for index in random_indices]
            
            ### new category
            dataset_prefix = '/tmp/articulated'
            articulated = sorted(os.listdir(dataset_prefix))
            # articulated = [os.path.join(dataset_prefix, x) for x in articulated]
            articulated = ["{}/{}".format(dataset_prefix, name) for name in articulated_new]
            if "25_percent" in num_train_objects:
                num_articulated = int(0.25 * len(articulated))
                random_indices = random.sample(range(len(articulated)), num_articulated)
                articulated = [articulated[index] for index in random_indices]
            if "50_percent" in num_train_objects:
                num_articulated = int(0.5 * len(articulated))
                random_indices = random.sample(range(len(articulated)), num_articulated)
                articulated = [articulated[index] for index in random_indices]

            ### new category with camera randomization
            dataset_prefix = '/tmp/new_7_category_random_cam'
            articulated_random_cam = sorted(os.listdir(dataset_prefix))
            ### only use folders not starting with digit
            articulated_random_cam = [f for f in articulated_random_cam if not f[0].isdigit()]
            articulated_random_cam = [os.path.join(dataset_prefix, x) for x in articulated_random_cam]
            if "25_percent" in num_train_objects:
                num_articulated_random = int(0.25 * len(articulated_random_cam))
                random_indices = random.sample(range(len(articulated_random_cam)), num_articulated_random)
                articulated_random_cam = [articulated_random_cam[index] for index in random_indices]
            if "50_percent" in num_train_objects:
                num_articulated_random = int(0.5 * len(articulated_random_cam))
                random_indices = random.sample(range(len(articulated_random_cam)), num_articulated_random)
                articulated_random_cam = [articulated_random_cam[index] for index in random_indices]
            
            ### new category with real world randomization
            dataset_prefix = '/tmp/new_7_category_real_cam'
            articulated_real_cam = sorted(os.listdir(dataset_prefix))
            ### only use folders not starting with digit
            articulated_real_cam = [f for f in articulated_real_cam if not f[0].isdigit()]
            articulated_real_cam = [os.path.join(dataset_prefix, x) for x in articulated_real_cam]
            if "25_percent" in num_train_objects:
                num_articulated_real = int(0.25 * len(articulated_real_cam))
                random_indices = random.sample(range(len(articulated_real_cam)), num_articulated_real)
                articulated_real_cam = [articulated_real_cam[index] for index in random_indices]
            if "50_percent" in num_train_objects:
                num_articulated_real = int(0.5 * len(articulated_real_cam))
                random_indices = random.sample(range(len(articulated_real_cam)), num_articulated_real)
                articulated_real_cam = [articulated_real_cam[index] for index in random_indices]
            
            ### dagger on new categories
            dataset_prefix = '/tmp/dp3_demo_weighted_full_dagger'
            articulated_dagger = sorted(os.listdir(dataset_prefix))
            ### only use folders not starting with digit
            articulated_dagger = [f for f in articulated_dagger if not f[0].isdigit()]
            articulated_dagger = [os.path.join(dataset_prefix, x) for x in articulated_dagger]
            if "25_percent" in num_train_objects:
                num_articulated_dagger = int(0.25 * len(articulated_dagger))
                random_indices = random.sample(range(len(articulated_dagger)), num_articulated_dagger)
                articulated_dagger = [articulated_dagger[index] for index in random_indices]
            if "50_percent" in num_train_objects:
                num_articulated_dagger = int(0.5 * len(articulated_dagger))
                random_indices = random.sample(range(len(articulated_dagger)), num_articulated_dagger)
                articulated_dagger = [articulated_dagger[index] for index in random_indices]
            
            ### close data
            dataset_prefix = '/tmp/invert_push'
            close_data = sorted(os.listdir(dataset_prefix))
            close_data = [os.path.join(dataset_prefix, x) for x in close_data]
            if "25_percent" in num_train_objects:
                num_close = int(0.25 * len(close_data))
                random_indices = random.sample(range(len(close_data)), num_close)
                close_data = [close_data[index] for index in random_indices]
            if "50_percent" in num_train_objects:
                num_close = int(0.5 * len(close_data))
                random_indices = random.sample(range(len(close_data)), num_close)
                close_data = [close_data[index] for index in random_indices]
            
            ### pick and place data
            all_pick_place_data = []
            if num_train_objects == 'aritucbot_new_cat_camera_random_close_pick_and_place':
                dataset_prefix = ["top", "inside_whole_1", "inside_whole", "inside_link_2", "inside_link_1", "inside_link"]
            elif "pick_and_place_more_1005" in num_train_objects:
                dataset_prefix = ["inside_whole_1005", "inside_link_1005", "top_1005"]
            elif "pick_and_place_new_1204" in num_train_objects:
                dataset_prefix = ["top_cgn_1204", "inside_link_cgn_1204", "inside_whole_cgn_1204"]
            elif "pick_and_place_0101" in num_train_objects:
                dataset_prefix = ["inside_link_cgn_grasp_0101_grasp_only", "inside_whole_cgn_grasp_0101_grasp_only", "top_cgn_grasp_0101_grasp_only", 
                                  "top_cgn_place_0101", "inside_link_cgn_place_0101", "inside_whole_cgn_place_0101"]
            elif "pick_and_place_0103" in num_train_objects:
                dataset_prefix = ["inside_link_cgn_grasp_0101_grasp_only", "inside_whole_cgn_grasp_0101_grasp_only", "top_cgn_grasp_0101_grasp_only", 
                                  "top_cgn_place_0101", "inside_link_cgn_place_0101", "inside_whole_cgn_place_0101",
                                  "inside_whole_cgn_place_0103", "inside_link_cgn_place_0103"
                                  ]
                
                
            for name in dataset_prefix:
                path = f"/tmp/pick_and_place/{name}"
                all_data = sorted(os.listdir(path))
                all_data = [os.path.join(path, x) for x in all_data]
                all_pick_place_data.extend(all_data)
            if "25_percent" in num_train_objects:
                num_pick_place = int(0.25 * len(all_pick_place_data))
                random_indices = random.sample(range(len(all_pick_place_data)), num_pick_place)
                all_pick_place_data = [all_pick_place_data[index] for index in random_indices]
            if "50_percent" in num_train_objects:
                num_pick_place = int(0.5 * len(all_pick_place_data))
                random_indices = random.sample(range(len(all_pick_place_data)), num_pick_place)
                all_pick_place_data = [all_pick_place_data[index] for index in random_indices]
            
            all_grasping_data = []
            if "grasping_1009" in num_train_objects:
                name = "gen_grasp_1009"
                path = f"/project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/data/multitask_all_training_data/grasping/{name}"
                all_data = sorted(os.listdir(path))
                all_data = [os.path.join(path, x) for x in all_data]
                all_grasping_data.extend(all_data)
            elif "grasping_1017" in num_train_objects:
                name = "gen_grasp_1017"
                path = f"/project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/data/multitask_all_training_data/grasping/{name}"
                all_data = sorted(os.listdir(path))
                all_data = [os.path.join(path, x) for x in all_data]
                all_grasping_data.extend(all_data)
            if "25_percent" in num_train_objects:
                num_grasping = int(0.25 * len(all_grasping_data))
                random_indices = random.sample(range(len(all_grasping_data)), num_grasping)
                all_grasping_data = [all_grasping_data[index] for index in random_indices]
            if "50_percent" in num_train_objects:
                num_grasping = int(0.5 * len(all_grasping_data))
                random_indices = random.sample(range(len(all_grasping_data)), num_grasping)
                all_grasping_data = [all_grasping_data[index] for index in random_indices]

            all_obj_paths = non_real_world_camera_500_paths + real_world_camera_500_paths + articulated + \
                articulated_random_cam + articulated_real_cam + articulated_dagger + close_data + all_pick_place_data + \
                    all_grasping_data
                
            print("all obj paths: ============================================")
            print(all_obj_paths)
            print("all obj paths: ============================================")
            
        elif num_train_objects == 'articulated':
            data_name = [
                save_data_name_5, save_data_name_6, save_data_name_7, save_data_name_8, save_data_name_9,
                save_data_name_10, save_data_name_11, save_data_name_12, save_data_name_13, save_data_name_14, save_data_name_15, save_data_name_16, save_data_name_17, save_data_name_18, save_data_name_19,
                save_data_name_20, save_data_name_21, save_data_name_22, save_data_name_23, save_data_name_24, save_data_name_25, save_data_name_26, save_data_name_27, save_data_name_28, save_data_name_29,
                save_data_name_30, save_data_name_31, save_data_name_32, save_data_name_33, save_data_name_34, save_data_name_35, save_data_name_36, save_data_name_37, save_data_name_38, save_data_name_39,
                save_data_name_40, save_data_name_41, save_data_name_42, save_data_name_43, save_data_name_44, save_data_name_45, save_data_name_46, save_data_name_47, save_data_name_48, save_data_name_49,
            ] + articulated_new
            all_obj_paths = [
                "{}/{}".format(dataset_prefix, data_name[i]) for i in range(len(data_name))
            ]
        elif num_train_objects == 'articulated_250':
            data_name = [
                save_data_name_5, save_data_name_6, save_data_name_7, save_data_name_8, save_data_name_9,
                save_data_name_10, save_data_name_11, save_data_name_12, save_data_name_13, save_data_name_14, save_data_name_15, save_data_name_16, save_data_name_17, save_data_name_18, save_data_name_19,
                save_data_name_20, save_data_name_21, save_data_name_22, save_data_name_23, save_data_name_24, save_data_name_25, save_data_name_26, save_data_name_27, save_data_name_28, save_data_name_29,
                save_data_name_30, save_data_name_31, save_data_name_32, save_data_name_33, save_data_name_34, save_data_name_35, save_data_name_36, save_data_name_37, save_data_name_38, save_data_name_39,
                save_data_name_40, save_data_name_41, save_data_name_42, save_data_name_43, save_data_name_44, save_data_name_45, save_data_name_46, save_data_name_47, save_data_name_48, save_data_name_49,
                save_data_name_50, save_data_name_51, save_data_name_52, save_data_name_53, save_data_name_54, save_data_name_55, save_data_name_56, save_data_name_57, save_data_name_58, save_data_name_59,
                save_data_name_60, save_data_name_61, save_data_name_62, save_data_name_63, save_data_name_64, save_data_name_65, save_data_name_66, save_data_name_67, save_data_name_68, save_data_name_69,
                save_data_name_70, save_data_name_71, save_data_name_72, save_data_name_73, save_data_name_74, save_data_name_75, save_data_name_76, save_data_name_77, save_data_name_78, save_data_name_79,
                save_data_name_80, save_data_name_81, save_data_name_82, save_data_name_83, save_data_name_84, save_data_name_85, save_data_name_86, save_data_name_87, save_data_name_88, save_data_name_89,
                save_data_name_90, save_data_name_91, save_data_name_92, save_data_name_93, save_data_name_94, save_data_name_95, save_data_name_96, save_data_name_97, save_data_name_98, save_data_name_99,
            ] 
            
            articulated_new_not_replace = [name for name in articulated_new if name not in articulated_new_replace]
            data_name += articulated_new_not_replace
            all_obj_paths = [
                "{}/{}".format(dataset_prefix, data_name[i]) for i in range(len(data_name))
            ]
            
            dataset_prefix = "/project_data/held/chenyuah/RoboGen-sim2real/data/dp3_demo/165-obj_1219"
            replaced_articulated_paths = [
                "{}/{}".format(dataset_prefix, name) for name in articulated_new_replace
            ]
            all_obj_paths += replaced_articulated_paths
            
            print("total articulated 250 objects: ", len(all_obj_paths))
            print("all_obj_paths: ", all_obj_paths)
            
        elif num_train_objects == 'articulated_full':
            all_zarr_paths_part_1 = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(246)]
            all_subfolders = sorted(os.listdir(dataset_prefix))
            object_other_categories_no_cam_rand = [x for x in all_subfolders if "1121-other-cat-no-cam-rand" in x]
            all_zarr_paths_part_2 = [f"{dataset_prefix}/{x}" for x in object_other_categories_no_cam_rand]
            all_zarr_paths_part_3 = articulated_new
            all_zarr_paths_part_3 = ["{}/{}".format(dataset_prefix, name) for name in all_zarr_paths_part_3]
            all_obj_paths = all_zarr_paths_part_1 + all_zarr_paths_part_2 + all_zarr_paths_part_3
        elif num_train_objects == 'full_and_close':
            all_zarr_paths_part_1 = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(246)]
            all_subfolders = sorted(os.listdir(dataset_prefix))
            object_other_categories_no_cam_rand = [x for x in all_subfolders if "1121-other-cat-no-cam-rand" in x]
            all_zarr_paths_part_2 = [f"{dataset_prefix}/{x}" for x in object_other_categories_no_cam_rand]
            all_zarr_paths_part_3 = articulated_new
            all_zarr_paths_part_3 = ["{}/{}".format(dataset_prefix, name) for name in all_zarr_paths_part_3]
            all_obj_paths = all_zarr_paths_part_1 + all_zarr_paths_part_2 + all_zarr_paths_part_3
            close_prefix = '/mnt/RoboGen_sim2real/data/dp3_demo/invert'
            close_prefix_2 = '/mnt/RoboGen_sim2real/data/dp3_demo/invert_new'
            close_names = os.listdir(close_prefix)
            close_names = [name for name in close_names if name[0].isalpha()]
            close_names_2 = os.listdir(close_prefix_2)
            close_obj_paths = [
                "{}/{}".format(close_prefix, close_names[i]) for i in range(len(close_names))
            ] + [
                "{}/{}".format(close_prefix_2, close_names_2[i]) for i in range(len(close_names_2))
            ]
            all_obj_paths += close_obj_paths
        elif num_train_objects == 'articulated_full_dagger':
            all_zarr_paths_part_1 = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(246)]
            all_subfolders = sorted(os.listdir(dataset_prefix))
            object_other_categories_no_cam_rand = [x for x in all_subfolders if "1121-other-cat-no-cam-rand" in x]
            all_zarr_paths_part_2 = [f"{dataset_prefix}/{x}" for x in object_other_categories_no_cam_rand]
            all_zarr_paths_part_3 = articulated_new
            all_zarr_paths_part_3 = ["{}/{}".format(dataset_prefix, name) for name in all_zarr_paths_part_3]
            all_obj_paths = all_zarr_paths_part_1 + all_zarr_paths_part_2 + all_zarr_paths_part_3
            dagger_prefix = f'{dataset_prefix}/weighted_full_dagger'
            dagger_names = os.listdir(dagger_prefix)
            dagger_obj_paths = [
                "{}/{}".format(dagger_prefix, dagger_names[i]) for i in range(len(dagger_names))
            ]
            print(len(all_obj_paths))
            all_obj_paths += dagger_obj_paths
            print(len(all_obj_paths))
            print(all_obj_paths)
        elif num_train_objects == 'dagger':
            data_name = [
                save_data_name_5, save_data_name_6, save_data_name_7, save_data_name_8, save_data_name_9,
                save_data_name_10, save_data_name_11, save_data_name_12, save_data_name_13, save_data_name_14, save_data_name_15, save_data_name_16, save_data_name_17, save_data_name_18, save_data_name_19,
                save_data_name_20, save_data_name_21, save_data_name_22, save_data_name_23, save_data_name_24, save_data_name_25, save_data_name_26, save_data_name_27, save_data_name_28, save_data_name_29,
                save_data_name_30, save_data_name_31, save_data_name_32, save_data_name_33, save_data_name_34, save_data_name_35, save_data_name_36, save_data_name_37, save_data_name_38, save_data_name_39,
                save_data_name_40, save_data_name_41, save_data_name_42, save_data_name_43, save_data_name_44, save_data_name_45, save_data_name_46, save_data_name_47, save_data_name_48, save_data_name_49,
            ] + articulated_new
            all_obj_paths = [
                "{}/{}".format(dataset_prefix, data_name[i]) for i in range(len(data_name))
            ]
            dagger_prefix = '/mnt/RoboGen_sim2real/data/dagger'
            dagger_names = os.listdir(dagger_prefix)
            dagger_obj_paths = [
                "{}/{}".format(dagger_prefix, dagger_names[i]) for i in range(len(dagger_names))
            ]
            all_obj_paths += dagger_obj_paths
        elif num_train_objects == 'bucket':
            data_name = [
                "bucket_100444", "bucket_100452", "bucket_100454", "bucket_100460", "bucket_100461",
                "bucket_100462", "bucket_100469", "bucket_100472", "bucket_102352", "bucket_102365",
            ]
            all_obj_paths = [
                "{}/{}".format(dataset_prefix, data_name[i]) for i in range(len(data_name))
            ]
        elif num_train_objects == 'faucet':
            data_name = [
                "faucet_148", "faucet_149", "faucet_152", "faucet_153", "faucet_154",
                "faucet_168", "faucet_811", "faucet_857", "faucet_960", "faucet_991",
            ]
            all_obj_paths = [
                "{}/{}".format(dataset_prefix, data_name[i]) for i in range(len(data_name))
            ]
        elif num_train_objects == 'foldingchair':
            data_name = [
                "foldingchair_100520", "foldingchair_100521", "foldingchair_100526", "foldingchair_100562", "foldingchair_100586",
                "foldingchair_100590", "foldingchair_100599", "foldingchair_102263", "foldingchair_102269", "foldingchair_102314",
            ]
            all_obj_paths = [
                "{}/{}".format(dataset_prefix, data_name[i]) for i in range(len(data_name))
            ]
        elif num_train_objects == 'laptop':
            data_name = [
                "laptop_9748", "laptop_9912", "laptop_9960", "laptop_9968", "laptop_9992",
                "laptop_9996", "laptop_10040", "laptop_10098", "laptop_10101", "laptop_10238",
            ]
            all_obj_paths = [
                "{}/{}".format(dataset_prefix, data_name[i]) for i in range(len(data_name))
            ]
        elif num_train_objects == 'stapler':
            data_name = [
                "stapler_103095", "stapler_103099", "stapler_103100", "stapler_103104", "stapler_103111",
                "stapler_103292", "stapler_103293", "stapler_103297", "stapler_103299", "stapler_103301",
            ]
            all_obj_paths = [
                "{}/{}".format(dataset_prefix, data_name[i]) for i in range(len(data_name))
            ]
        elif num_train_objects == 'toilet':
            data_name = [
                "toilet_101320", "toilet_102621", "toilet_102622", "toilet_102630", "toilet_102634",
                "toilet_102645", "toilet_102648", "toilet_102651", "toilet_102652", "toilet_102658",
            ]
            all_obj_paths = [
                "{}/{}".format(dataset_prefix, data_name[i]) for i in range(len(data_name))
            ]
        elif num_train_objects == 'debug':
            # all_obj_paths = [f'{dataset_prefix}/0622-act3d-obj-45448-remove-reaching-collision-resize-2-full-per-step-gripper-goal-displacement-to-closest-obj-point']
            all_obj_paths = ["/tmp/pick_and_place/top_cgn_1204/"]
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
            
        elif num_train_objects == "camera_random_10_obj_high_level":
            all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["camera_random_10_save_data_name_{}".format(i)]) for i in range(20)]
        elif num_train_objects == 'camera_random_50_obj_high_level':
            all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["camera_random_50_save_data_name_{}".format(i)]) for i in range(87)]
        elif num_train_objects == 'camera_random_100_obj_high_level':
            all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["camera_random_100_save_data_name_{}".format(i)]) for i in range(175)]
        elif num_train_objects == 'camera_random_200_obj_high_level':
            all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["camera_random_200_save_data_name_{}".format(i)]) for i in range(350)]
        elif num_train_objects == 'camera_random_500_obj_high_level' or num_train_objects == "500_object_high_level":
            all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(462)]
            
        elif num_train_objects == '300_old':
            
            all_obj_paths = [f'{dataset_prefix}/{save_data_name_0}', f'{dataset_prefix}/{save_data_name_1}', f'{dataset_prefix}/{save_data_name_2}', f'{dataset_prefix}/{save_data_name_3}', f'{dataset_prefix}/{save_data_name_4}', f'{dataset_prefix}/{save_data_name_5}', f'{dataset_prefix}/{save_data_name_6}', f'{dataset_prefix}/{save_data_name_7}', f'{dataset_prefix}/{save_data_name_8}', f'{dataset_prefix}/{save_data_name_9}', 
            f'{dataset_prefix}/{save_data_name_10}', f'{dataset_prefix}/{save_data_name_11}', f'{dataset_prefix}/{save_data_name_12}', f'{dataset_prefix}/{save_data_name_13}', f'{dataset_prefix}/{save_data_name_14}', f'{dataset_prefix}/{save_data_name_15}', f'{dataset_prefix}/{save_data_name_16}', f'{dataset_prefix}/{save_data_name_17}', f'{dataset_prefix}/{save_data_name_18}', f'{dataset_prefix}/{save_data_name_19}', 
            f'{dataset_prefix}/{save_data_name_20}', f'{dataset_prefix}/{save_data_name_21}', f'{dataset_prefix}/{save_data_name_22}', f'{dataset_prefix}/{save_data_name_23}', f'{dataset_prefix}/{save_data_name_24}', f'{dataset_prefix}/{save_data_name_25}', f'{dataset_prefix}/{save_data_name_26}', f'{dataset_prefix}/{save_data_name_27}', f'{dataset_prefix}/{save_data_name_28}', f'{dataset_prefix}/{save_data_name_29}', 
            f'{dataset_prefix}/{save_data_name_30}', f'{dataset_prefix}/{save_data_name_31}', f'{dataset_prefix}/{save_data_name_32}', f'{dataset_prefix}/{save_data_name_33}', f'{dataset_prefix}/{save_data_name_34}', f'{dataset_prefix}/{save_data_name_35}', f'{dataset_prefix}/{save_data_name_36}', f'{dataset_prefix}/{save_data_name_37}', f'{dataset_prefix}/{save_data_name_38}', f'{dataset_prefix}/{save_data_name_39}', 
            f'{dataset_prefix}/{save_data_name_40}', f'{dataset_prefix}/{save_data_name_41}', f'{dataset_prefix}/{save_data_name_42}', f'{dataset_prefix}/{save_data_name_43}', f'{dataset_prefix}/{save_data_name_44}', f'{dataset_prefix}/{save_data_name_45}', f'{dataset_prefix}/{save_data_name_46}', f'{dataset_prefix}/{save_data_name_47}', f'{dataset_prefix}/{save_data_name_48}', f'{dataset_prefix}/{save_data_name_49}',
            f'{dataset_prefix}/{save_data_name_50}', f'{dataset_prefix}/{save_data_name_51}', f'{dataset_prefix}/{save_data_name_52}', f'{dataset_prefix}/{save_data_name_53}', f'{dataset_prefix}/{save_data_name_54}', f'{dataset_prefix}/{save_data_name_55}', f'{dataset_prefix}/{save_data_name_56}', f'{dataset_prefix}/{save_data_name_57}', f'{dataset_prefix}/{save_data_name_58}', f'{dataset_prefix}/{save_data_name_59}',
            f'{dataset_prefix}/{save_data_name_60}', f'{dataset_prefix}/{save_data_name_61}', f'{dataset_prefix}/{save_data_name_62}', f'{dataset_prefix}/{save_data_name_63}', f'{dataset_prefix}/{save_data_name_64}', f'{dataset_prefix}/{save_data_name_65}', f'{dataset_prefix}/{save_data_name_66}', f'{dataset_prefix}/{save_data_name_67}', f'{dataset_prefix}/{save_data_name_68}', f'{dataset_prefix}/{save_data_name_69}',
            f'{dataset_prefix}/{save_data_name_70}', f'{dataset_prefix}/{save_data_name_71}', f'{dataset_prefix}/{save_data_name_72}', f'{dataset_prefix}/{save_data_name_73}', f'{dataset_prefix}/{save_data_name_74}', f'{dataset_prefix}/{save_data_name_75}', f'{dataset_prefix}/{save_data_name_76}', f'{dataset_prefix}/{save_data_name_77}', f'{dataset_prefix}/{save_data_name_78}', f'{dataset_prefix}/{save_data_name_79}',
            f'{dataset_prefix}/{save_data_name_80}', f'{dataset_prefix}/{save_data_name_81}', f'{dataset_prefix}/{save_data_name_82}', f'{dataset_prefix}/{save_data_name_83}', f'{dataset_prefix}/{save_data_name_84}', f'{dataset_prefix}/{save_data_name_85}', f'{dataset_prefix}/{save_data_name_86}', f'{dataset_prefix}/{save_data_name_87}', f'{dataset_prefix}/{save_data_name_88}', f'{dataset_prefix}/{save_data_name_89}',
            f'{dataset_prefix}/{save_data_name_90}', f'{dataset_prefix}/{save_data_name_91}', f'{dataset_prefix}/{save_data_name_92}', f'{dataset_prefix}/{save_data_name_93}', f'{dataset_prefix}/{save_data_name_94}', f'{dataset_prefix}/{save_data_name_95}', f'{dataset_prefix}/{save_data_name_96}', f'{dataset_prefix}/{save_data_name_97}', f'{dataset_prefix}/{save_data_name_98}', f'{dataset_prefix}/{save_data_name_99}',
            f'{dataset_prefix}/{save_data_name_100}', f'{dataset_prefix}/{save_data_name_101}', f'{dataset_prefix}/{save_data_name_102}', f'{dataset_prefix}/{save_data_name_103}', f'{dataset_prefix}/{save_data_name_104}', f'{dataset_prefix}/{save_data_name_105}', f'{dataset_prefix}/{save_data_name_106}', f'{dataset_prefix}/{save_data_name_107}', f'{dataset_prefix}/{save_data_name_108}', f'{dataset_prefix}/{save_data_name_109}',
            f'{dataset_prefix}/{save_data_name_110}', f'{dataset_prefix}/{save_data_name_111}', f'{dataset_prefix}/{save_data_name_112}', f'{dataset_prefix}/{save_data_name_113}', f'{dataset_prefix}/{save_data_name_114}', f'{dataset_prefix}/{save_data_name_115}', f'{dataset_prefix}/{save_data_name_116}', f'{dataset_prefix}/{save_data_name_117}', f'{dataset_prefix}/{save_data_name_118}', f'{dataset_prefix}/{save_data_name_119}',
            f'{dataset_prefix}/{save_data_name_120}', f'{dataset_prefix}/{save_data_name_121}', f'{dataset_prefix}/{save_data_name_122}', f'{dataset_prefix}/{save_data_name_123}', f'{dataset_prefix}/{save_data_name_124}', f'{dataset_prefix}/{save_data_name_125}', f'{dataset_prefix}/{save_data_name_126}', f'{dataset_prefix}/{save_data_name_127}', f'{dataset_prefix}/{save_data_name_128}', f'{dataset_prefix}/{save_data_name_129}',
            f'{dataset_prefix}/{save_data_name_130}', f'{dataset_prefix}/{save_data_name_131}', f'{dataset_prefix}/{save_data_name_132}', f'{dataset_prefix}/{save_data_name_133}', f'{dataset_prefix}/{save_data_name_134}', f'{dataset_prefix}/{save_data_name_135}', f'{dataset_prefix}/{save_data_name_136}', f'{dataset_prefix}/{save_data_name_137}', f'{dataset_prefix}/{save_data_name_138}', f'{dataset_prefix}/{save_data_name_139}',
            f'{dataset_prefix}/{save_data_name_140}', f'{dataset_prefix}/{save_data_name_141}', f'{dataset_prefix}/{save_data_name_142}', f'{dataset_prefix}/{save_data_name_143}', f'{dataset_prefix}/{save_data_name_144}', f'{dataset_prefix}/{save_data_name_145}', f'{dataset_prefix}/{save_data_name_146}', f'{dataset_prefix}/{save_data_name_147}', f'{dataset_prefix}/{save_data_name_148}', f'{dataset_prefix}/{save_data_name_149}',
            f'{dataset_prefix}/{save_data_name_150}', f'{dataset_prefix}/{save_data_name_151}', f'{dataset_prefix}/{save_data_name_152}', f'{dataset_prefix}/{save_data_name_153}', f'{dataset_prefix}/{save_data_name_154}', f'{dataset_prefix}/{save_data_name_155}', f'{dataset_prefix}/{save_data_name_156}', f'{dataset_prefix}/{save_data_name_157}', f'{dataset_prefix}/{save_data_name_158}', f'{dataset_prefix}/{save_data_name_159}',
            f'{dataset_prefix}/{save_data_name_160}', f'{dataset_prefix}/{save_data_name_161}', f'{dataset_prefix}/{save_data_name_162}', f'{dataset_prefix}/{save_data_name_163}', f'{dataset_prefix}/{save_data_name_164}', f'{dataset_prefix}/{save_data_name_165}', f'{dataset_prefix}/{save_data_name_166}', f'{dataset_prefix}/{save_data_name_167}', f'{dataset_prefix}/{save_data_name_168}', f'{dataset_prefix}/{save_data_name_169}',
            f'{dataset_prefix}/{save_data_name_170}', f'{dataset_prefix}/{save_data_name_171}', f'{dataset_prefix}/{save_data_name_172}', f'{dataset_prefix}/{save_data_name_173}', f'{dataset_prefix}/{save_data_name_174}', f'{dataset_prefix}/{save_data_name_175}', f'{dataset_prefix}/{save_data_name_176}', f'{dataset_prefix}/{save_data_name_177}', f'{dataset_prefix}/{save_data_name_178}', f'{dataset_prefix}/{save_data_name_179}',
            f'{dataset_prefix}/{save_data_name_180}', f'{dataset_prefix}/{save_data_name_181}', f'{dataset_prefix}/{save_data_name_182}', f'{dataset_prefix}/{save_data_name_183}', f'{dataset_prefix}/{save_data_name_184}', f'{dataset_prefix}/{save_data_name_185}', f'{dataset_prefix}/{save_data_name_186}', f'{dataset_prefix}/{save_data_name_187}', f'{dataset_prefix}/{save_data_name_188}', f'{dataset_prefix}/{save_data_name_189}',
            f'{dataset_prefix}/{save_data_name_190}', f'{dataset_prefix}/{save_data_name_191}', f'{dataset_prefix}/{save_data_name_192}', f'{dataset_prefix}/{save_data_name_193}', f'{dataset_prefix}/{save_data_name_194}', f'{dataset_prefix}/{save_data_name_195}', f'{dataset_prefix}/{save_data_name_196}', f'{dataset_prefix}/{save_data_name_197}', f'{dataset_prefix}/{save_data_name_198}', f'{dataset_prefix}/{save_data_name_199}',
            f'{dataset_prefix}/{save_data_name_200}', f'{dataset_prefix}/{save_data_name_201}', f'{dataset_prefix}/{save_data_name_202}', f'{dataset_prefix}/{save_data_name_203}', f'{dataset_prefix}/{save_data_name_204}', f'{dataset_prefix}/{save_data_name_205}', f'{dataset_prefix}/{save_data_name_206}', f'{dataset_prefix}/{save_data_name_207}', f'{dataset_prefix}/{save_data_name_208}', f'{dataset_prefix}/{save_data_name_209}',
            f'{dataset_prefix}/{save_data_name_210}', f'{dataset_prefix}/{save_data_name_211}', f'{dataset_prefix}/{save_data_name_212}', f'{dataset_prefix}/{save_data_name_213}', f'{dataset_prefix}/{save_data_name_214}', f'{dataset_prefix}/{save_data_name_215}', f'{dataset_prefix}/{save_data_name_216}', f'{dataset_prefix}/{save_data_name_217}', f'{dataset_prefix}/{save_data_name_218}', f'{dataset_prefix}/{save_data_name_219}',
            f'{dataset_prefix}/{save_data_name_220}', f'{dataset_prefix}/{save_data_name_221}', f'{dataset_prefix}/{save_data_name_222}', f'{dataset_prefix}/{save_data_name_223}', f'{dataset_prefix}/{save_data_name_224}', f'{dataset_prefix}/{save_data_name_225}', f'{dataset_prefix}/{save_data_name_226}', f'{dataset_prefix}/{save_data_name_227}', f'{dataset_prefix}/{save_data_name_228}', f'{dataset_prefix}/{save_data_name_229}',
            f'{dataset_prefix}/{save_data_name_230}', f'{dataset_prefix}/{save_data_name_231}', f'{dataset_prefix}/{save_data_name_232}', f'{dataset_prefix}/{save_data_name_233}', f'{dataset_prefix}/{save_data_name_234}', f'{dataset_prefix}/{save_data_name_235}', f'{dataset_prefix}/{save_data_name_236}', f'{dataset_prefix}/{save_data_name_237}', f'{dataset_prefix}/{save_data_name_238}', f'{dataset_prefix}/{save_data_name_239}',
            f'{dataset_prefix}/{save_data_name_240}', f'{dataset_prefix}/{save_data_name_241}', f'{dataset_prefix}/{save_data_name_242}', f'{dataset_prefix}/{save_data_name_243}', f'{dataset_prefix}/{save_data_name_244}', f'{dataset_prefix}/{save_data_name_245}', f'{dataset_prefix}/{save_data_name_246}', f'{dataset_prefix}/{save_data_name_247}', f'{dataset_prefix}/{save_data_name_248}', f'{dataset_prefix}/{save_data_name_249}',
            f'{dataset_prefix}/{save_data_name_250}', f'{dataset_prefix}/{save_data_name_251}', f'{dataset_prefix}/{save_data_name_252}', f'{dataset_prefix}/{save_data_name_253}', f'{dataset_prefix}/{save_data_name_254}', f'{dataset_prefix}/{save_data_name_255}', f'{dataset_prefix}/{save_data_name_256}', f'{dataset_prefix}/{save_data_name_257}', f'{dataset_prefix}/{save_data_name_258}', f'{dataset_prefix}/{save_data_name_259}',
            f'{dataset_prefix}/{save_data_name_260}', f'{dataset_prefix}/{save_data_name_261}', f'{dataset_prefix}/{save_data_name_262}', f'{dataset_prefix}/{save_data_name_263}', f'{dataset_prefix}/{save_data_name_264}', f'{dataset_prefix}/{save_data_name_265}', f'{dataset_prefix}/{save_data_name_266}', f'{dataset_prefix}/{save_data_name_267}', f'{dataset_prefix}/{save_data_name_268}', f'{dataset_prefix}/{save_data_name_269}',
            f'{dataset_prefix}/{save_data_name_270}', f'{dataset_prefix}/{save_data_name_271}', f'{dataset_prefix}/{save_data_name_272}', f'{dataset_prefix}/{save_data_name_273}', f'{dataset_prefix}/{save_data_name_274}', f'{dataset_prefix}/{save_data_name_275}', f'{dataset_prefix}/{save_data_name_276}', f'{dataset_prefix}/{save_data_name_277}', f'{dataset_prefix}/{save_data_name_278}', f'{dataset_prefix}/{save_data_name_279}',
            f'{dataset_prefix}/{save_data_name_280}', f'{dataset_prefix}/{save_data_name_281}', f'{dataset_prefix}/{save_data_name_282}', f'{dataset_prefix}/{save_data_name_283}', f'{dataset_prefix}/{save_data_name_284}', f'{dataset_prefix}/{save_data_name_285}', f'{dataset_prefix}/{save_data_name_286}']
        elif num_train_objects == '500':
            all_obj_paths = [f'{dataset_prefix}/{save_data_name_0}', f'{dataset_prefix}/{save_data_name_1}', f'{dataset_prefix}/{save_data_name_2}', f'{dataset_prefix}/{save_data_name_3}', f'{dataset_prefix}/{save_data_name_4}', f'{dataset_prefix}/{save_data_name_5}', f'{dataset_prefix}/{save_data_name_6}', f'{dataset_prefix}/{save_data_name_7}', f'{dataset_prefix}/{save_data_name_8}', f'{dataset_prefix}/{save_data_name_9}', 
            f'{dataset_prefix}/{save_data_name_10}', f'{dataset_prefix}/{save_data_name_11}', f'{dataset_prefix}/{save_data_name_12}', f'{dataset_prefix}/{save_data_name_13}', f'{dataset_prefix}/{save_data_name_14}', f'{dataset_prefix}/{save_data_name_15}', f'{dataset_prefix}/{save_data_name_16}', f'{dataset_prefix}/{save_data_name_17}', f'{dataset_prefix}/{save_data_name_18}', f'{dataset_prefix}/{save_data_name_19}', 
            f'{dataset_prefix}/{save_data_name_20}', f'{dataset_prefix}/{save_data_name_21}', f'{dataset_prefix}/{save_data_name_22}', f'{dataset_prefix}/{save_data_name_23}', f'{dataset_prefix}/{save_data_name_24}', f'{dataset_prefix}/{save_data_name_25}', f'{dataset_prefix}/{save_data_name_26}', f'{dataset_prefix}/{save_data_name_27}', f'{dataset_prefix}/{save_data_name_28}', f'{dataset_prefix}/{save_data_name_29}', 
            f'{dataset_prefix}/{save_data_name_30}', f'{dataset_prefix}/{save_data_name_31}', f'{dataset_prefix}/{save_data_name_32}', f'{dataset_prefix}/{save_data_name_33}', f'{dataset_prefix}/{save_data_name_34}', f'{dataset_prefix}/{save_data_name_35}', f'{dataset_prefix}/{save_data_name_36}', f'{dataset_prefix}/{save_data_name_37}', f'{dataset_prefix}/{save_data_name_38}', f'{dataset_prefix}/{save_data_name_39}', 
            f'{dataset_prefix}/{save_data_name_40}', f'{dataset_prefix}/{save_data_name_41}', f'{dataset_prefix}/{save_data_name_42}', f'{dataset_prefix}/{save_data_name_43}', f'{dataset_prefix}/{save_data_name_44}', f'{dataset_prefix}/{save_data_name_45}', f'{dataset_prefix}/{save_data_name_46}', f'{dataset_prefix}/{save_data_name_47}', f'{dataset_prefix}/{save_data_name_48}', f'{dataset_prefix}/{save_data_name_49}',
            f'{dataset_prefix}/{save_data_name_50}', f'{dataset_prefix}/{save_data_name_51}', f'{dataset_prefix}/{save_data_name_52}', f'{dataset_prefix}/{save_data_name_53}', f'{dataset_prefix}/{save_data_name_54}', f'{dataset_prefix}/{save_data_name_55}', f'{dataset_prefix}/{save_data_name_56}', f'{dataset_prefix}/{save_data_name_57}', f'{dataset_prefix}/{save_data_name_58}', f'{dataset_prefix}/{save_data_name_59}',
            f'{dataset_prefix}/{save_data_name_60}', f'{dataset_prefix}/{save_data_name_61}', f'{dataset_prefix}/{save_data_name_62}', f'{dataset_prefix}/{save_data_name_63}', f'{dataset_prefix}/{save_data_name_64}', f'{dataset_prefix}/{save_data_name_65}', f'{dataset_prefix}/{save_data_name_66}', f'{dataset_prefix}/{save_data_name_67}', f'{dataset_prefix}/{save_data_name_68}', f'{dataset_prefix}/{save_data_name_69}',
            f'{dataset_prefix}/{save_data_name_70}', f'{dataset_prefix}/{save_data_name_71}', f'{dataset_prefix}/{save_data_name_72}', f'{dataset_prefix}/{save_data_name_73}', f'{dataset_prefix}/{save_data_name_74}', f'{dataset_prefix}/{save_data_name_75}', f'{dataset_prefix}/{save_data_name_76}', f'{dataset_prefix}/{save_data_name_77}', f'{dataset_prefix}/{save_data_name_78}', f'{dataset_prefix}/{save_data_name_79}',
            f'{dataset_prefix}/{save_data_name_80}', f'{dataset_prefix}/{save_data_name_81}', f'{dataset_prefix}/{save_data_name_82}', f'{dataset_prefix}/{save_data_name_83}', f'{dataset_prefix}/{save_data_name_84}', f'{dataset_prefix}/{save_data_name_85}', f'{dataset_prefix}/{save_data_name_86}', f'{dataset_prefix}/{save_data_name_87}', f'{dataset_prefix}/{save_data_name_88}', f'{dataset_prefix}/{save_data_name_89}',
            f'{dataset_prefix}/{save_data_name_90}', f'{dataset_prefix}/{save_data_name_91}', f'{dataset_prefix}/{save_data_name_92}', f'{dataset_prefix}/{save_data_name_93}', f'{dataset_prefix}/{save_data_name_94}', f'{dataset_prefix}/{save_data_name_95}', f'{dataset_prefix}/{save_data_name_96}', f'{dataset_prefix}/{save_data_name_97}', f'{dataset_prefix}/{save_data_name_98}', f'{dataset_prefix}/{save_data_name_99}',
            f'{dataset_prefix}/{save_data_name_100}', f'{dataset_prefix}/{save_data_name_101}', f'{dataset_prefix}/{save_data_name_102}', f'{dataset_prefix}/{save_data_name_103}', f'{dataset_prefix}/{save_data_name_104}', f'{dataset_prefix}/{save_data_name_105}', f'{dataset_prefix}/{save_data_name_106}', f'{dataset_prefix}/{save_data_name_107}', f'{dataset_prefix}/{save_data_name_108}', f'{dataset_prefix}/{save_data_name_109}',
            f'{dataset_prefix}/{save_data_name_110}', f'{dataset_prefix}/{save_data_name_111}', f'{dataset_prefix}/{save_data_name_112}', f'{dataset_prefix}/{save_data_name_113}', f'{dataset_prefix}/{save_data_name_114}', f'{dataset_prefix}/{save_data_name_115}', f'{dataset_prefix}/{save_data_name_116}', f'{dataset_prefix}/{save_data_name_117}', f'{dataset_prefix}/{save_data_name_118}', f'{dataset_prefix}/{save_data_name_119}',
            f'{dataset_prefix}/{save_data_name_120}', f'{dataset_prefix}/{save_data_name_121}', f'{dataset_prefix}/{save_data_name_122}', f'{dataset_prefix}/{save_data_name_123}', f'{dataset_prefix}/{save_data_name_124}', f'{dataset_prefix}/{save_data_name_125}', f'{dataset_prefix}/{save_data_name_126}', f'{dataset_prefix}/{save_data_name_127}', f'{dataset_prefix}/{save_data_name_128}', f'{dataset_prefix}/{save_data_name_129}',
            f'{dataset_prefix}/{save_data_name_130}', f'{dataset_prefix}/{save_data_name_131}', f'{dataset_prefix}/{save_data_name_132}', f'{dataset_prefix}/{save_data_name_133}', f'{dataset_prefix}/{save_data_name_134}', f'{dataset_prefix}/{save_data_name_135}', f'{dataset_prefix}/{save_data_name_136}', f'{dataset_prefix}/{save_data_name_137}', f'{dataset_prefix}/{save_data_name_138}', f'{dataset_prefix}/{save_data_name_139}',
            f'{dataset_prefix}/{save_data_name_140}', f'{dataset_prefix}/{save_data_name_141}', f'{dataset_prefix}/{save_data_name_142}', f'{dataset_prefix}/{save_data_name_143}', f'{dataset_prefix}/{save_data_name_144}', f'{dataset_prefix}/{save_data_name_145}', f'{dataset_prefix}/{save_data_name_146}', f'{dataset_prefix}/{save_data_name_147}', f'{dataset_prefix}/{save_data_name_148}', f'{dataset_prefix}/{save_data_name_149}',
            f'{dataset_prefix}/{save_data_name_150}', f'{dataset_prefix}/{save_data_name_151}', f'{dataset_prefix}/{save_data_name_152}', f'{dataset_prefix}/{save_data_name_153}', f'{dataset_prefix}/{save_data_name_154}', f'{dataset_prefix}/{save_data_name_155}', f'{dataset_prefix}/{save_data_name_156}', f'{dataset_prefix}/{save_data_name_157}', f'{dataset_prefix}/{save_data_name_158}', f'{dataset_prefix}/{save_data_name_159}',
            f'{dataset_prefix}/{save_data_name_160}', f'{dataset_prefix}/{save_data_name_161}', f'{dataset_prefix}/{save_data_name_162}', f'{dataset_prefix}/{save_data_name_163}', f'{dataset_prefix}/{save_data_name_164}', f'{dataset_prefix}/{save_data_name_165}', f'{dataset_prefix}/{save_data_name_166}', f'{dataset_prefix}/{save_data_name_167}', f'{dataset_prefix}/{save_data_name_168}', f'{dataset_prefix}/{save_data_name_169}',
            f'{dataset_prefix}/{save_data_name_170}', f'{dataset_prefix}/{save_data_name_171}', f'{dataset_prefix}/{save_data_name_172}', f'{dataset_prefix}/{save_data_name_173}', f'{dataset_prefix}/{save_data_name_174}', f'{dataset_prefix}/{save_data_name_175}', f'{dataset_prefix}/{save_data_name_176}', f'{dataset_prefix}/{save_data_name_177}', f'{dataset_prefix}/{save_data_name_178}', f'{dataset_prefix}/{save_data_name_179}',
            f'{dataset_prefix}/{save_data_name_180}', f'{dataset_prefix}/{save_data_name_181}', f'{dataset_prefix}/{save_data_name_182}', f'{dataset_prefix}/{save_data_name_183}', f'{dataset_prefix}/{save_data_name_184}', f'{dataset_prefix}/{save_data_name_185}', f'{dataset_prefix}/{save_data_name_186}', f'{dataset_prefix}/{save_data_name_187}', f'{dataset_prefix}/{save_data_name_188}', f'{dataset_prefix}/{save_data_name_189}',
            f'{dataset_prefix}/{save_data_name_190}', f'{dataset_prefix}/{save_data_name_191}', f'{dataset_prefix}/{save_data_name_192}', f'{dataset_prefix}/{save_data_name_193}', f'{dataset_prefix}/{save_data_name_194}', f'{dataset_prefix}/{save_data_name_195}', f'{dataset_prefix}/{save_data_name_196}', f'{dataset_prefix}/{save_data_name_197}', f'{dataset_prefix}/{save_data_name_198}', f'{dataset_prefix}/{save_data_name_199}',
            f'{dataset_prefix}/{save_data_name_200}', f'{dataset_prefix}/{save_data_name_201}', f'{dataset_prefix}/{save_data_name_202}', f'{dataset_prefix}/{save_data_name_203}', f'{dataset_prefix}/{save_data_name_204}', f'{dataset_prefix}/{save_data_name_205}', f'{dataset_prefix}/{save_data_name_206}', f'{dataset_prefix}/{save_data_name_207}', f'{dataset_prefix}/{save_data_name_208}', f'{dataset_prefix}/{save_data_name_209}',
            f'{dataset_prefix}/{save_data_name_210}', f'{dataset_prefix}/{save_data_name_211}', f'{dataset_prefix}/{save_data_name_212}', f'{dataset_prefix}/{save_data_name_213}', f'{dataset_prefix}/{save_data_name_214}', f'{dataset_prefix}/{save_data_name_215}', f'{dataset_prefix}/{save_data_name_216}', f'{dataset_prefix}/{save_data_name_217}', f'{dataset_prefix}/{save_data_name_218}', f'{dataset_prefix}/{save_data_name_219}',
            f'{dataset_prefix}/{save_data_name_220}', f'{dataset_prefix}/{save_data_name_221}', f'{dataset_prefix}/{save_data_name_222}', f'{dataset_prefix}/{save_data_name_223}', f'{dataset_prefix}/{save_data_name_224}', f'{dataset_prefix}/{save_data_name_225}', f'{dataset_prefix}/{save_data_name_226}', f'{dataset_prefix}/{save_data_name_227}', f'{dataset_prefix}/{save_data_name_228}', f'{dataset_prefix}/{save_data_name_229}',
            f'{dataset_prefix}/{save_data_name_230}', f'{dataset_prefix}/{save_data_name_231}', f'{dataset_prefix}/{save_data_name_232}', f'{dataset_prefix}/{save_data_name_233}', f'{dataset_prefix}/{save_data_name_234}', f'{dataset_prefix}/{save_data_name_235}', f'{dataset_prefix}/{save_data_name_236}', f'{dataset_prefix}/{save_data_name_237}', f'{dataset_prefix}/{save_data_name_238}', f'{dataset_prefix}/{save_data_name_239}',
            f'{dataset_prefix}/{save_data_name_240}', f'{dataset_prefix}/{save_data_name_241}', f'{dataset_prefix}/{save_data_name_242}', f'{dataset_prefix}/{save_data_name_243}', f'{dataset_prefix}/{save_data_name_244}', f'{dataset_prefix}/{save_data_name_245}', f'{dataset_prefix}/{save_data_name_246}', f'{dataset_prefix}/{save_data_name_247}', f'{dataset_prefix}/{save_data_name_248}', f'{dataset_prefix}/{save_data_name_249}',
            f'{dataset_prefix}/{save_data_name_250}', f'{dataset_prefix}/{save_data_name_251}', f'{dataset_prefix}/{save_data_name_252}', f'{dataset_prefix}/{save_data_name_253}', f'{dataset_prefix}/{save_data_name_254}', f'{dataset_prefix}/{save_data_name_255}', f'{dataset_prefix}/{save_data_name_256}', f'{dataset_prefix}/{save_data_name_257}', f'{dataset_prefix}/{save_data_name_258}', f'{dataset_prefix}/{save_data_name_259}',
            f'{dataset_prefix}/{save_data_name_260}', f'{dataset_prefix}/{save_data_name_261}', f'{dataset_prefix}/{save_data_name_262}', f'{dataset_prefix}/{save_data_name_263}', f'{dataset_prefix}/{save_data_name_264}', f'{dataset_prefix}/{save_data_name_265}', f'{dataset_prefix}/{save_data_name_266}', f'{dataset_prefix}/{save_data_name_267}', f'{dataset_prefix}/{save_data_name_268}', f'{dataset_prefix}/{save_data_name_269}',
            f'{dataset_prefix}/{save_data_name_270}', f'{dataset_prefix}/{save_data_name_271}', f'{dataset_prefix}/{save_data_name_272}', f'{dataset_prefix}/{save_data_name_273}', f'{dataset_prefix}/{save_data_name_274}', f'{dataset_prefix}/{save_data_name_275}', f'{dataset_prefix}/{save_data_name_276}', f'{dataset_prefix}/{save_data_name_277}', f'{dataset_prefix}/{save_data_name_278}', f'{dataset_prefix}/{save_data_name_279}',
            f'{dataset_prefix}/{save_data_name_280}', f'{dataset_prefix}/{save_data_name_281}', f'{dataset_prefix}/{save_data_name_282}', f'{dataset_prefix}/{save_data_name_283}', f'{dataset_prefix}/{save_data_name_284}', f'{dataset_prefix}/{save_data_name_285}', f'{dataset_prefix}/{save_data_name_286}',
            f'{dataset_prefix}/{save_data_name_287}', f'{dataset_prefix}/{save_data_name_288}', f'{dataset_prefix}/{save_data_name_289}', f'{dataset_prefix}/{save_data_name_290}', f'{dataset_prefix}/{save_data_name_291}', f'{dataset_prefix}/{save_data_name_292}', f'{dataset_prefix}/{save_data_name_293}', f'{dataset_prefix}/{save_data_name_294}', f'{dataset_prefix}/{save_data_name_295}', f'{dataset_prefix}/{save_data_name_296}', f'{dataset_prefix}/{save_data_name_297}', f'{dataset_prefix}/{save_data_name_298}', f'{dataset_prefix}/{save_data_name_299}', f'{dataset_prefix}/{save_data_name_300}', f'{dataset_prefix}/{save_data_name_301}', f'{dataset_prefix}/{save_data_name_302}', f'{dataset_prefix}/{save_data_name_303}', f'{dataset_prefix}/{save_data_name_304}', f'{dataset_prefix}/{save_data_name_305}', f'{dataset_prefix}/{save_data_name_306}', f'{dataset_prefix}/{save_data_name_307}', f'{dataset_prefix}/{save_data_name_308}', f'{dataset_prefix}/{save_data_name_309}', f'{dataset_prefix}/{save_data_name_310}', f'{dataset_prefix}/{save_data_name_311}', f'{dataset_prefix}/{save_data_name_312}', f'{dataset_prefix}/{save_data_name_313}', f'{dataset_prefix}/{save_data_name_314}', f'{dataset_prefix}/{save_data_name_315}', f'{dataset_prefix}/{save_data_name_316}', f'{dataset_prefix}/{save_data_name_317}', f'{dataset_prefix}/{save_data_name_318}', f'{dataset_prefix}/{save_data_name_319}', f'{dataset_prefix}/{save_data_name_320}', f'{dataset_prefix}/{save_data_name_321}', f'{dataset_prefix}/{save_data_name_322}', f'{dataset_prefix}/{save_data_name_323}', f'{dataset_prefix}/{save_data_name_324}', f'{dataset_prefix}/{save_data_name_325}', f'{dataset_prefix}/{save_data_name_326}', f'{dataset_prefix}/{save_data_name_327}', f'{dataset_prefix}/{save_data_name_328}', f'{dataset_prefix}/{save_data_name_329}', f'{dataset_prefix}/{save_data_name_330}', f'{dataset_prefix}/{save_data_name_331}', f'{dataset_prefix}/{save_data_name_332}', f'{dataset_prefix}/{save_data_name_333}', f'{dataset_prefix}/{save_data_name_334}', f'{dataset_prefix}/{save_data_name_335}', f'{dataset_prefix}/{save_data_name_336}', f'{dataset_prefix}/{save_data_name_337}', f'{dataset_prefix}/{save_data_name_338}', f'{dataset_prefix}/{save_data_name_339}', f'{dataset_prefix}/{save_data_name_340}', f'{dataset_prefix}/{save_data_name_341}', f'{dataset_prefix}/{save_data_name_342}', f'{dataset_prefix}/{save_data_name_343}', f'{dataset_prefix}/{save_data_name_344}', f'{dataset_prefix}/{save_data_name_345}', f'{dataset_prefix}/{save_data_name_346}', f'{dataset_prefix}/{save_data_name_347}', f'{dataset_prefix}/{save_data_name_348}', f'{dataset_prefix}/{save_data_name_349}', f'{dataset_prefix}/{save_data_name_350}', f'{dataset_prefix}/{save_data_name_351}', f'{dataset_prefix}/{save_data_name_352}', f'{dataset_prefix}/{save_data_name_353}', f'{dataset_prefix}/{save_data_name_354}', f'{dataset_prefix}/{save_data_name_355}', f'{dataset_prefix}/{save_data_name_356}', f'{dataset_prefix}/{save_data_name_357}', f'{dataset_prefix}/{save_data_name_358}', f'{dataset_prefix}/{save_data_name_359}', f'{dataset_prefix}/{save_data_name_360}', f'{dataset_prefix}/{save_data_name_361}', f'{dataset_prefix}/{save_data_name_362}', f'{dataset_prefix}/{save_data_name_363}', f'{dataset_prefix}/{save_data_name_364}', f'{dataset_prefix}/{save_data_name_365}', f'{dataset_prefix}/{save_data_name_366}', f'{dataset_prefix}/{save_data_name_367}', f'{dataset_prefix}/{save_data_name_368}', f'{dataset_prefix}/{save_data_name_369}', f'{dataset_prefix}/{save_data_name_370}', f'{dataset_prefix}/{save_data_name_371}', f'{dataset_prefix}/{save_data_name_372}', f'{dataset_prefix}/{save_data_name_373}', f'{dataset_prefix}/{save_data_name_374}', f'{dataset_prefix}/{save_data_name_375}', f'{dataset_prefix}/{save_data_name_376}', f'{dataset_prefix}/{save_data_name_377}', f'{dataset_prefix}/{save_data_name_378}', f'{dataset_prefix}/{save_data_name_379}', f'{dataset_prefix}/{save_data_name_380}', f'{dataset_prefix}/{save_data_name_381}', f'{dataset_prefix}/{save_data_name_382}', f'{dataset_prefix}/{save_data_name_383}', f'{dataset_prefix}/{save_data_name_384}', f'{dataset_prefix}/{save_data_name_385}', f'{dataset_prefix}/{save_data_name_386}', f'{dataset_prefix}/{save_data_name_387}', f'{dataset_prefix}/{save_data_name_388}', f'{dataset_prefix}/{save_data_name_389}', f'{dataset_prefix}/{save_data_name_390}', f'{dataset_prefix}/{save_data_name_391}', f'{dataset_prefix}/{save_data_name_392}', f'{dataset_prefix}/{save_data_name_393}', f'{dataset_prefix}/{save_data_name_394}', f'{dataset_prefix}/{save_data_name_395}', f'{dataset_prefix}/{save_data_name_396}', f'{dataset_prefix}/{save_data_name_397}', f'{dataset_prefix}/{save_data_name_398}', f'{dataset_prefix}/{save_data_name_399}', f'{dataset_prefix}/{save_data_name_400}', f'{dataset_prefix}/{save_data_name_401}', f'{dataset_prefix}/{save_data_name_402}', f'{dataset_prefix}/{save_data_name_403}', f'{dataset_prefix}/{save_data_name_404}', f'{dataset_prefix}/{save_data_name_405}', f'{dataset_prefix}/{save_data_name_406}', f'{dataset_prefix}/{save_data_name_407}', f'{dataset_prefix}/{save_data_name_408}', f'{dataset_prefix}/{save_data_name_409}', f'{dataset_prefix}/{save_data_name_410}', f'{dataset_prefix}/{save_data_name_411}', f'{dataset_prefix}/{save_data_name_412}', f'{dataset_prefix}/{save_data_name_413}', f'{dataset_prefix}/{save_data_name_414}', f'{dataset_prefix}/{save_data_name_415}', f'{dataset_prefix}/{save_data_name_416}', f'{dataset_prefix}/{save_data_name_417}', f'{dataset_prefix}/{save_data_name_418}', f'{dataset_prefix}/{save_data_name_419}', f'{dataset_prefix}/{save_data_name_420}', f'{dataset_prefix}/{save_data_name_421}', f'{dataset_prefix}/{save_data_name_422}', f'{dataset_prefix}/{save_data_name_423}', f'{dataset_prefix}/{save_data_name_424}', f'{dataset_prefix}/{save_data_name_425}', f'{dataset_prefix}/{save_data_name_426}', f'{dataset_prefix}/{save_data_name_427}', f'{dataset_prefix}/{save_data_name_428}', f'{dataset_prefix}/{save_data_name_429}', f'{dataset_prefix}/{save_data_name_430}', f'{dataset_prefix}/{save_data_name_431}', f'{dataset_prefix}/{save_data_name_432}', f'{dataset_prefix}/{save_data_name_433}', f'{dataset_prefix}/{save_data_name_434}', f'{dataset_prefix}/{save_data_name_435}', f'{dataset_prefix}/{save_data_name_436}', f'{dataset_prefix}/{save_data_name_437}', f'{dataset_prefix}/{save_data_name_438}', f'{dataset_prefix}/{save_data_name_439}', f'{dataset_prefix}/{save_data_name_440}', f'{dataset_prefix}/{save_data_name_441}', f'{dataset_prefix}/{save_data_name_442}', f'{dataset_prefix}/{save_data_name_443}', f'{dataset_prefix}/{save_data_name_444}', f'{dataset_prefix}/{save_data_name_445}', f'{dataset_prefix}/{save_data_name_446}', f'{dataset_prefix}/{save_data_name_447}', f'{dataset_prefix}/{save_data_name_448}', f'{dataset_prefix}/{save_data_name_449}', f'{dataset_prefix}/{save_data_name_450}', f'{dataset_prefix}/{save_data_name_451}', f'{dataset_prefix}/{save_data_name_452}', f'{dataset_prefix}/{save_data_name_453}', f'{dataset_prefix}/{save_data_name_454}', f'{dataset_prefix}/{save_data_name_455}', f'{dataset_prefix}/{save_data_name_456}', f'{dataset_prefix}/{save_data_name_457}', f'{dataset_prefix}/{save_data_name_458}', f'{dataset_prefix}/{save_data_name_459}', f'{dataset_prefix}/{save_data_name_460}', f'{dataset_prefix}/{save_data_name_461}', f'{dataset_prefix}/{save_data_name_462}',
            ]
        elif num_train_objects == '600':
            all_obj_paths = [f'{dataset_prefix}/{save_data_name_0}', f'{dataset_prefix}/{save_data_name_1}', f'{dataset_prefix}/{save_data_name_2}', f'{dataset_prefix}/{save_data_name_3}', f'{dataset_prefix}/{save_data_name_4}', f'{dataset_prefix}/{save_data_name_5}', f'{dataset_prefix}/{save_data_name_6}', f'{dataset_prefix}/{save_data_name_7}', f'{dataset_prefix}/{save_data_name_8}', f'{dataset_prefix}/{save_data_name_9}', 
            f'{dataset_prefix}/{save_data_name_10}', f'{dataset_prefix}/{save_data_name_11}', f'{dataset_prefix}/{save_data_name_12}', f'{dataset_prefix}/{save_data_name_13}', f'{dataset_prefix}/{save_data_name_14}', f'{dataset_prefix}/{save_data_name_15}', f'{dataset_prefix}/{save_data_name_16}', f'{dataset_prefix}/{save_data_name_17}', f'{dataset_prefix}/{save_data_name_18}', f'{dataset_prefix}/{save_data_name_19}', 
            f'{dataset_prefix}/{save_data_name_20}', f'{dataset_prefix}/{save_data_name_21}', f'{dataset_prefix}/{save_data_name_22}', f'{dataset_prefix}/{save_data_name_23}', f'{dataset_prefix}/{save_data_name_24}', f'{dataset_prefix}/{save_data_name_25}', f'{dataset_prefix}/{save_data_name_26}', f'{dataset_prefix}/{save_data_name_27}', f'{dataset_prefix}/{save_data_name_28}', f'{dataset_prefix}/{save_data_name_29}', 
            f'{dataset_prefix}/{save_data_name_30}', f'{dataset_prefix}/{save_data_name_31}', f'{dataset_prefix}/{save_data_name_32}', f'{dataset_prefix}/{save_data_name_33}', f'{dataset_prefix}/{save_data_name_34}', f'{dataset_prefix}/{save_data_name_35}', f'{dataset_prefix}/{save_data_name_36}', f'{dataset_prefix}/{save_data_name_37}', f'{dataset_prefix}/{save_data_name_38}', f'{dataset_prefix}/{save_data_name_39}', 
            f'{dataset_prefix}/{save_data_name_40}', f'{dataset_prefix}/{save_data_name_41}', f'{dataset_prefix}/{save_data_name_42}', f'{dataset_prefix}/{save_data_name_43}', f'{dataset_prefix}/{save_data_name_44}', f'{dataset_prefix}/{save_data_name_45}', f'{dataset_prefix}/{save_data_name_46}', f'{dataset_prefix}/{save_data_name_47}', f'{dataset_prefix}/{save_data_name_48}', f'{dataset_prefix}/{save_data_name_49}',
            f'{dataset_prefix}/{save_data_name_50}', f'{dataset_prefix}/{save_data_name_51}', f'{dataset_prefix}/{save_data_name_52}', f'{dataset_prefix}/{save_data_name_53}', f'{dataset_prefix}/{save_data_name_54}', f'{dataset_prefix}/{save_data_name_55}', f'{dataset_prefix}/{save_data_name_56}', f'{dataset_prefix}/{save_data_name_57}', f'{dataset_prefix}/{save_data_name_58}', f'{dataset_prefix}/{save_data_name_59}',
            f'{dataset_prefix}/{save_data_name_60}', f'{dataset_prefix}/{save_data_name_61}', f'{dataset_prefix}/{save_data_name_62}', f'{dataset_prefix}/{save_data_name_63}', f'{dataset_prefix}/{save_data_name_64}', f'{dataset_prefix}/{save_data_name_65}', f'{dataset_prefix}/{save_data_name_66}', f'{dataset_prefix}/{save_data_name_67}', f'{dataset_prefix}/{save_data_name_68}', f'{dataset_prefix}/{save_data_name_69}',
            f'{dataset_prefix}/{save_data_name_70}', f'{dataset_prefix}/{save_data_name_71}', f'{dataset_prefix}/{save_data_name_72}', f'{dataset_prefix}/{save_data_name_73}', f'{dataset_prefix}/{save_data_name_74}', f'{dataset_prefix}/{save_data_name_75}', f'{dataset_prefix}/{save_data_name_76}', f'{dataset_prefix}/{save_data_name_77}', f'{dataset_prefix}/{save_data_name_78}', f'{dataset_prefix}/{save_data_name_79}',
            f'{dataset_prefix}/{save_data_name_80}', f'{dataset_prefix}/{save_data_name_81}', f'{dataset_prefix}/{save_data_name_82}', f'{dataset_prefix}/{save_data_name_83}', f'{dataset_prefix}/{save_data_name_84}', f'{dataset_prefix}/{save_data_name_85}', f'{dataset_prefix}/{save_data_name_86}', f'{dataset_prefix}/{save_data_name_87}', f'{dataset_prefix}/{save_data_name_88}', f'{dataset_prefix}/{save_data_name_89}',
            f'{dataset_prefix}/{save_data_name_90}', f'{dataset_prefix}/{save_data_name_91}', f'{dataset_prefix}/{save_data_name_92}', f'{dataset_prefix}/{save_data_name_93}', f'{dataset_prefix}/{save_data_name_94}', f'{dataset_prefix}/{save_data_name_95}', f'{dataset_prefix}/{save_data_name_96}', f'{dataset_prefix}/{save_data_name_97}', f'{dataset_prefix}/{save_data_name_98}', f'{dataset_prefix}/{save_data_name_99}',
            f'{dataset_prefix}/{save_data_name_100}', f'{dataset_prefix}/{save_data_name_101}', f'{dataset_prefix}/{save_data_name_102}', f'{dataset_prefix}/{save_data_name_103}', f'{dataset_prefix}/{save_data_name_104}', f'{dataset_prefix}/{save_data_name_105}', f'{dataset_prefix}/{save_data_name_106}', f'{dataset_prefix}/{save_data_name_107}', f'{dataset_prefix}/{save_data_name_108}', f'{dataset_prefix}/{save_data_name_109}',
            f'{dataset_prefix}/{save_data_name_110}', f'{dataset_prefix}/{save_data_name_111}', f'{dataset_prefix}/{save_data_name_112}', f'{dataset_prefix}/{save_data_name_113}', f'{dataset_prefix}/{save_data_name_114}', f'{dataset_prefix}/{save_data_name_115}', f'{dataset_prefix}/{save_data_name_116}', f'{dataset_prefix}/{save_data_name_117}', f'{dataset_prefix}/{save_data_name_118}', f'{dataset_prefix}/{save_data_name_119}',
            f'{dataset_prefix}/{save_data_name_120}', f'{dataset_prefix}/{save_data_name_121}', f'{dataset_prefix}/{save_data_name_122}', f'{dataset_prefix}/{save_data_name_123}', f'{dataset_prefix}/{save_data_name_124}', f'{dataset_prefix}/{save_data_name_125}', f'{dataset_prefix}/{save_data_name_126}', f'{dataset_prefix}/{save_data_name_127}', f'{dataset_prefix}/{save_data_name_128}', f'{dataset_prefix}/{save_data_name_129}',
            f'{dataset_prefix}/{save_data_name_130}', f'{dataset_prefix}/{save_data_name_131}', f'{dataset_prefix}/{save_data_name_132}', f'{dataset_prefix}/{save_data_name_133}', f'{dataset_prefix}/{save_data_name_134}', f'{dataset_prefix}/{save_data_name_135}', f'{dataset_prefix}/{save_data_name_136}', f'{dataset_prefix}/{save_data_name_137}', f'{dataset_prefix}/{save_data_name_138}', f'{dataset_prefix}/{save_data_name_139}',
            f'{dataset_prefix}/{save_data_name_140}', f'{dataset_prefix}/{save_data_name_141}', f'{dataset_prefix}/{save_data_name_142}', f'{dataset_prefix}/{save_data_name_143}', f'{dataset_prefix}/{save_data_name_144}', f'{dataset_prefix}/{save_data_name_145}', f'{dataset_prefix}/{save_data_name_146}', f'{dataset_prefix}/{save_data_name_147}', f'{dataset_prefix}/{save_data_name_148}', f'{dataset_prefix}/{save_data_name_149}',
            f'{dataset_prefix}/{save_data_name_150}', f'{dataset_prefix}/{save_data_name_151}', f'{dataset_prefix}/{save_data_name_152}', f'{dataset_prefix}/{save_data_name_153}', f'{dataset_prefix}/{save_data_name_154}', f'{dataset_prefix}/{save_data_name_155}', f'{dataset_prefix}/{save_data_name_156}', f'{dataset_prefix}/{save_data_name_157}', f'{dataset_prefix}/{save_data_name_158}', f'{dataset_prefix}/{save_data_name_159}',
            f'{dataset_prefix}/{save_data_name_160}', f'{dataset_prefix}/{save_data_name_161}', f'{dataset_prefix}/{save_data_name_162}', f'{dataset_prefix}/{save_data_name_163}', f'{dataset_prefix}/{save_data_name_164}', f'{dataset_prefix}/{save_data_name_165}', f'{dataset_prefix}/{save_data_name_166}', f'{dataset_prefix}/{save_data_name_167}', f'{dataset_prefix}/{save_data_name_168}', f'{dataset_prefix}/{save_data_name_169}',
            f'{dataset_prefix}/{save_data_name_170}', f'{dataset_prefix}/{save_data_name_171}', f'{dataset_prefix}/{save_data_name_172}', f'{dataset_prefix}/{save_data_name_173}', f'{dataset_prefix}/{save_data_name_174}', f'{dataset_prefix}/{save_data_name_175}', f'{dataset_prefix}/{save_data_name_176}', f'{dataset_prefix}/{save_data_name_177}', f'{dataset_prefix}/{save_data_name_178}', f'{dataset_prefix}/{save_data_name_179}',
            f'{dataset_prefix}/{save_data_name_180}', f'{dataset_prefix}/{save_data_name_181}', f'{dataset_prefix}/{save_data_name_182}', f'{dataset_prefix}/{save_data_name_183}', f'{dataset_prefix}/{save_data_name_184}', f'{dataset_prefix}/{save_data_name_185}', f'{dataset_prefix}/{save_data_name_186}', f'{dataset_prefix}/{save_data_name_187}', f'{dataset_prefix}/{save_data_name_188}', f'{dataset_prefix}/{save_data_name_189}',
            f'{dataset_prefix}/{save_data_name_190}', f'{dataset_prefix}/{save_data_name_191}', f'{dataset_prefix}/{save_data_name_192}', f'{dataset_prefix}/{save_data_name_193}', f'{dataset_prefix}/{save_data_name_194}', f'{dataset_prefix}/{save_data_name_195}', f'{dataset_prefix}/{save_data_name_196}', f'{dataset_prefix}/{save_data_name_197}', f'{dataset_prefix}/{save_data_name_198}', f'{dataset_prefix}/{save_data_name_199}',
            f'{dataset_prefix}/{save_data_name_200}', f'{dataset_prefix}/{save_data_name_201}', f'{dataset_prefix}/{save_data_name_202}', f'{dataset_prefix}/{save_data_name_203}', f'{dataset_prefix}/{save_data_name_204}', f'{dataset_prefix}/{save_data_name_205}', f'{dataset_prefix}/{save_data_name_206}', f'{dataset_prefix}/{save_data_name_207}', f'{dataset_prefix}/{save_data_name_208}', f'{dataset_prefix}/{save_data_name_209}',
            f'{dataset_prefix}/{save_data_name_210}', f'{dataset_prefix}/{save_data_name_211}', f'{dataset_prefix}/{save_data_name_212}', f'{dataset_prefix}/{save_data_name_213}', f'{dataset_prefix}/{save_data_name_214}', f'{dataset_prefix}/{save_data_name_215}', f'{dataset_prefix}/{save_data_name_216}', f'{dataset_prefix}/{save_data_name_217}', f'{dataset_prefix}/{save_data_name_218}', f'{dataset_prefix}/{save_data_name_219}',
            f'{dataset_prefix}/{save_data_name_220}', f'{dataset_prefix}/{save_data_name_221}', f'{dataset_prefix}/{save_data_name_222}', f'{dataset_prefix}/{save_data_name_223}', f'{dataset_prefix}/{save_data_name_224}', f'{dataset_prefix}/{save_data_name_225}', f'{dataset_prefix}/{save_data_name_226}', f'{dataset_prefix}/{save_data_name_227}', f'{dataset_prefix}/{save_data_name_228}', f'{dataset_prefix}/{save_data_name_229}',
            f'{dataset_prefix}/{save_data_name_230}', f'{dataset_prefix}/{save_data_name_231}', f'{dataset_prefix}/{save_data_name_232}', f'{dataset_prefix}/{save_data_name_233}', f'{dataset_prefix}/{save_data_name_234}', f'{dataset_prefix}/{save_data_name_235}', f'{dataset_prefix}/{save_data_name_236}', f'{dataset_prefix}/{save_data_name_237}', f'{dataset_prefix}/{save_data_name_238}', f'{dataset_prefix}/{save_data_name_239}',
            f'{dataset_prefix}/{save_data_name_240}', f'{dataset_prefix}/{save_data_name_241}', f'{dataset_prefix}/{save_data_name_242}', f'{dataset_prefix}/{save_data_name_243}', f'{dataset_prefix}/{save_data_name_244}', f'{dataset_prefix}/{save_data_name_245}', f'{dataset_prefix}/{save_data_name_246}', f'{dataset_prefix}/{save_data_name_247}', f'{dataset_prefix}/{save_data_name_248}', f'{dataset_prefix}/{save_data_name_249}',
            f'{dataset_prefix}/{save_data_name_250}', f'{dataset_prefix}/{save_data_name_251}', f'{dataset_prefix}/{save_data_name_252}', f'{dataset_prefix}/{save_data_name_253}', f'{dataset_prefix}/{save_data_name_254}', f'{dataset_prefix}/{save_data_name_255}', f'{dataset_prefix}/{save_data_name_256}', f'{dataset_prefix}/{save_data_name_257}', f'{dataset_prefix}/{save_data_name_258}', f'{dataset_prefix}/{save_data_name_259}',
            f'{dataset_prefix}/{save_data_name_260}', f'{dataset_prefix}/{save_data_name_261}', f'{dataset_prefix}/{save_data_name_262}', f'{dataset_prefix}/{save_data_name_263}', f'{dataset_prefix}/{save_data_name_264}', f'{dataset_prefix}/{save_data_name_265}', f'{dataset_prefix}/{save_data_name_266}', f'{dataset_prefix}/{save_data_name_267}', f'{dataset_prefix}/{save_data_name_268}', f'{dataset_prefix}/{save_data_name_269}',
            f'{dataset_prefix}/{save_data_name_270}', f'{dataset_prefix}/{save_data_name_271}', f'{dataset_prefix}/{save_data_name_272}', f'{dataset_prefix}/{save_data_name_273}', f'{dataset_prefix}/{save_data_name_274}', f'{dataset_prefix}/{save_data_name_275}', f'{dataset_prefix}/{save_data_name_276}', f'{dataset_prefix}/{save_data_name_277}', f'{dataset_prefix}/{save_data_name_278}', f'{dataset_prefix}/{save_data_name_279}',
            f'{dataset_prefix}/{save_data_name_280}', f'{dataset_prefix}/{save_data_name_281}', f'{dataset_prefix}/{save_data_name_282}', f'{dataset_prefix}/{save_data_name_283}', f'{dataset_prefix}/{save_data_name_284}', f'{dataset_prefix}/{save_data_name_285}', f'{dataset_prefix}/{save_data_name_286}',
            f'{dataset_prefix}/{save_data_name_287}', f'{dataset_prefix}/{save_data_name_288}', f'{dataset_prefix}/{save_data_name_289}', f'{dataset_prefix}/{save_data_name_290}', f'{dataset_prefix}/{save_data_name_291}', f'{dataset_prefix}/{save_data_name_292}', f'{dataset_prefix}/{save_data_name_293}', f'{dataset_prefix}/{save_data_name_294}', f'{dataset_prefix}/{save_data_name_295}', f'{dataset_prefix}/{save_data_name_296}', f'{dataset_prefix}/{save_data_name_297}', f'{dataset_prefix}/{save_data_name_298}', f'{dataset_prefix}/{save_data_name_299}', f'{dataset_prefix}/{save_data_name_300}', f'{dataset_prefix}/{save_data_name_301}', f'{dataset_prefix}/{save_data_name_302}', f'{dataset_prefix}/{save_data_name_303}', f'{dataset_prefix}/{save_data_name_304}', f'{dataset_prefix}/{save_data_name_305}', f'{dataset_prefix}/{save_data_name_306}', f'{dataset_prefix}/{save_data_name_307}', f'{dataset_prefix}/{save_data_name_308}', f'{dataset_prefix}/{save_data_name_309}', f'{dataset_prefix}/{save_data_name_310}', f'{dataset_prefix}/{save_data_name_311}', f'{dataset_prefix}/{save_data_name_312}', f'{dataset_prefix}/{save_data_name_313}', f'{dataset_prefix}/{save_data_name_314}', f'{dataset_prefix}/{save_data_name_315}', f'{dataset_prefix}/{save_data_name_316}', f'{dataset_prefix}/{save_data_name_317}', f'{dataset_prefix}/{save_data_name_318}', f'{dataset_prefix}/{save_data_name_319}', f'{dataset_prefix}/{save_data_name_320}', f'{dataset_prefix}/{save_data_name_321}', f'{dataset_prefix}/{save_data_name_322}', f'{dataset_prefix}/{save_data_name_323}', f'{dataset_prefix}/{save_data_name_324}', f'{dataset_prefix}/{save_data_name_325}', f'{dataset_prefix}/{save_data_name_326}', f'{dataset_prefix}/{save_data_name_327}', f'{dataset_prefix}/{save_data_name_328}', f'{dataset_prefix}/{save_data_name_329}', f'{dataset_prefix}/{save_data_name_330}', f'{dataset_prefix}/{save_data_name_331}', f'{dataset_prefix}/{save_data_name_332}', f'{dataset_prefix}/{save_data_name_333}', f'{dataset_prefix}/{save_data_name_334}', f'{dataset_prefix}/{save_data_name_335}', f'{dataset_prefix}/{save_data_name_336}', f'{dataset_prefix}/{save_data_name_337}', f'{dataset_prefix}/{save_data_name_338}', f'{dataset_prefix}/{save_data_name_339}', f'{dataset_prefix}/{save_data_name_340}', f'{dataset_prefix}/{save_data_name_341}', f'{dataset_prefix}/{save_data_name_342}', f'{dataset_prefix}/{save_data_name_343}', f'{dataset_prefix}/{save_data_name_344}', f'{dataset_prefix}/{save_data_name_345}', f'{dataset_prefix}/{save_data_name_346}', f'{dataset_prefix}/{save_data_name_347}', f'{dataset_prefix}/{save_data_name_348}', f'{dataset_prefix}/{save_data_name_349}', f'{dataset_prefix}/{save_data_name_350}', f'{dataset_prefix}/{save_data_name_351}', f'{dataset_prefix}/{save_data_name_352}', f'{dataset_prefix}/{save_data_name_353}', f'{dataset_prefix}/{save_data_name_354}', f'{dataset_prefix}/{save_data_name_355}', f'{dataset_prefix}/{save_data_name_356}', f'{dataset_prefix}/{save_data_name_357}', f'{dataset_prefix}/{save_data_name_358}', f'{dataset_prefix}/{save_data_name_359}', f'{dataset_prefix}/{save_data_name_360}', f'{dataset_prefix}/{save_data_name_361}', f'{dataset_prefix}/{save_data_name_362}', f'{dataset_prefix}/{save_data_name_363}', f'{dataset_prefix}/{save_data_name_364}', f'{dataset_prefix}/{save_data_name_365}', f'{dataset_prefix}/{save_data_name_366}', f'{dataset_prefix}/{save_data_name_367}', f'{dataset_prefix}/{save_data_name_368}', f'{dataset_prefix}/{save_data_name_369}', f'{dataset_prefix}/{save_data_name_370}', f'{dataset_prefix}/{save_data_name_371}', f'{dataset_prefix}/{save_data_name_372}', f'{dataset_prefix}/{save_data_name_373}', f'{dataset_prefix}/{save_data_name_374}', f'{dataset_prefix}/{save_data_name_375}', f'{dataset_prefix}/{save_data_name_376}', f'{dataset_prefix}/{save_data_name_377}', f'{dataset_prefix}/{save_data_name_378}', f'{dataset_prefix}/{save_data_name_379}', f'{dataset_prefix}/{save_data_name_380}', f'{dataset_prefix}/{save_data_name_381}', f'{dataset_prefix}/{save_data_name_382}', f'{dataset_prefix}/{save_data_name_383}', f'{dataset_prefix}/{save_data_name_384}', f'{dataset_prefix}/{save_data_name_385}', f'{dataset_prefix}/{save_data_name_386}', f'{dataset_prefix}/{save_data_name_387}', f'{dataset_prefix}/{save_data_name_388}', f'{dataset_prefix}/{save_data_name_389}', f'{dataset_prefix}/{save_data_name_390}', f'{dataset_prefix}/{save_data_name_391}', f'{dataset_prefix}/{save_data_name_392}', f'{dataset_prefix}/{save_data_name_393}', f'{dataset_prefix}/{save_data_name_394}', f'{dataset_prefix}/{save_data_name_395}', f'{dataset_prefix}/{save_data_name_396}', f'{dataset_prefix}/{save_data_name_397}', f'{dataset_prefix}/{save_data_name_398}', f'{dataset_prefix}/{save_data_name_399}', f'{dataset_prefix}/{save_data_name_400}', f'{dataset_prefix}/{save_data_name_401}', f'{dataset_prefix}/{save_data_name_402}', f'{dataset_prefix}/{save_data_name_403}', f'{dataset_prefix}/{save_data_name_404}', f'{dataset_prefix}/{save_data_name_405}', f'{dataset_prefix}/{save_data_name_406}', f'{dataset_prefix}/{save_data_name_407}', f'{dataset_prefix}/{save_data_name_408}', f'{dataset_prefix}/{save_data_name_409}', f'{dataset_prefix}/{save_data_name_410}', f'{dataset_prefix}/{save_data_name_411}', f'{dataset_prefix}/{save_data_name_412}', f'{dataset_prefix}/{save_data_name_413}', f'{dataset_prefix}/{save_data_name_414}', f'{dataset_prefix}/{save_data_name_415}', f'{dataset_prefix}/{save_data_name_416}', f'{dataset_prefix}/{save_data_name_417}', f'{dataset_prefix}/{save_data_name_418}', f'{dataset_prefix}/{save_data_name_419}', f'{dataset_prefix}/{save_data_name_420}', f'{dataset_prefix}/{save_data_name_421}', f'{dataset_prefix}/{save_data_name_422}', f'{dataset_prefix}/{save_data_name_423}', f'{dataset_prefix}/{save_data_name_424}', f'{dataset_prefix}/{save_data_name_425}', f'{dataset_prefix}/{save_data_name_426}', f'{dataset_prefix}/{save_data_name_427}', f'{dataset_prefix}/{save_data_name_428}', f'{dataset_prefix}/{save_data_name_429}', f'{dataset_prefix}/{save_data_name_430}', f'{dataset_prefix}/{save_data_name_431}', f'{dataset_prefix}/{save_data_name_432}', f'{dataset_prefix}/{save_data_name_433}', f'{dataset_prefix}/{save_data_name_434}', f'{dataset_prefix}/{save_data_name_435}', f'{dataset_prefix}/{save_data_name_436}', f'{dataset_prefix}/{save_data_name_437}', f'{dataset_prefix}/{save_data_name_438}', f'{dataset_prefix}/{save_data_name_439}', f'{dataset_prefix}/{save_data_name_440}', f'{dataset_prefix}/{save_data_name_441}', f'{dataset_prefix}/{save_data_name_442}', f'{dataset_prefix}/{save_data_name_443}', f'{dataset_prefix}/{save_data_name_444}', f'{dataset_prefix}/{save_data_name_445}', f'{dataset_prefix}/{save_data_name_446}', f'{dataset_prefix}/{save_data_name_447}', f'{dataset_prefix}/{save_data_name_448}', f'{dataset_prefix}/{save_data_name_449}', f'{dataset_prefix}/{save_data_name_450}', f'{dataset_prefix}/{save_data_name_451}', f'{dataset_prefix}/{save_data_name_452}', f'{dataset_prefix}/{save_data_name_453}', f'{dataset_prefix}/{save_data_name_454}', f'{dataset_prefix}/{save_data_name_455}', f'{dataset_prefix}/{save_data_name_456}', f'{dataset_prefix}/{save_data_name_457}', f'{dataset_prefix}/{save_data_name_458}', f'{dataset_prefix}/{save_data_name_459}', f'{dataset_prefix}/{save_data_name_460}', f'{dataset_prefix}/{save_data_name_461}', f'{dataset_prefix}/{save_data_name_462}',
            f'{dataset_prefix}/{save_data_name_463}',f'{dataset_prefix}/{save_data_name_464}',f'{dataset_prefix}/{save_data_name_465}',f'{dataset_prefix}/{save_data_name_466}',f'{dataset_prefix}/{save_data_name_467}',f'{dataset_prefix}/{save_data_name_468}',f'{dataset_prefix}/{save_data_name_469}',f'{dataset_prefix}/{save_data_name_470}',f'{dataset_prefix}/{save_data_name_471}',f'{dataset_prefix}/{save_data_name_472}',f'{dataset_prefix}/{save_data_name_473}',f'{dataset_prefix}/{save_data_name_474}',f'{dataset_prefix}/{save_data_name_475}',f'{dataset_prefix}/{save_data_name_476}',f'{dataset_prefix}/{save_data_name_477}',f'{dataset_prefix}/{save_data_name_478}',f'{dataset_prefix}/{save_data_name_479}',f'{dataset_prefix}/{save_data_name_480}',f'{dataset_prefix}/{save_data_name_481}',f'{dataset_prefix}/{save_data_name_482}',f'{dataset_prefix}/{save_data_name_483}',f'{dataset_prefix}/{save_data_name_484}',f'{dataset_prefix}/{save_data_name_485}',f'{dataset_prefix}/{save_data_name_486}',f'{dataset_prefix}/{save_data_name_487}',f'{dataset_prefix}/{save_data_name_488}',f'{dataset_prefix}/{save_data_name_489}',f'{dataset_prefix}/{save_data_name_490}',f'{dataset_prefix}/{save_data_name_491}',f'{dataset_prefix}/{save_data_name_492}',f'{dataset_prefix}/{save_data_name_493}',f'{dataset_prefix}/{save_data_name_494}',f'{dataset_prefix}/{save_data_name_495}',f'{dataset_prefix}/{save_data_name_496}',f'{dataset_prefix}/{save_data_name_497}',f'{dataset_prefix}/{save_data_name_498}',f'{dataset_prefix}/{save_data_name_499}',f'{dataset_prefix}/{save_data_name_500}',f'{dataset_prefix}/{save_data_name_501}',f'{dataset_prefix}/{save_data_name_502}',f'{dataset_prefix}/{save_data_name_503}',f'{dataset_prefix}/{save_data_name_504}',f'{dataset_prefix}/{save_data_name_505}',f'{dataset_prefix}/{save_data_name_506}',f'{dataset_prefix}/{save_data_name_507}',f'{dataset_prefix}/{save_data_name_508}',f'{dataset_prefix}/{save_data_name_509}',f'{dataset_prefix}/{save_data_name_510}',f'{dataset_prefix}/{save_data_name_511}',f'{dataset_prefix}/{save_data_name_512}',f'{dataset_prefix}/{save_data_name_513}',f'{dataset_prefix}/{save_data_name_514}',f'{dataset_prefix}/{save_data_name_515}',f'{dataset_prefix}/{save_data_name_516}',f'{dataset_prefix}/{save_data_name_517}',f'{dataset_prefix}/{save_data_name_518}',f'{dataset_prefix}/{save_data_name_519}',f'{dataset_prefix}/{save_data_name_520}',f'{dataset_prefix}/{save_data_name_521}',f'{dataset_prefix}/{save_data_name_522}',f'{dataset_prefix}/{save_data_name_523}',f'{dataset_prefix}/{save_data_name_524}',f'{dataset_prefix}/{save_data_name_525}',f'{dataset_prefix}/{save_data_name_526}',f'{dataset_prefix}/{save_data_name_527}',f'{dataset_prefix}/{save_data_name_528}',f'{dataset_prefix}/{save_data_name_529}',f'{dataset_prefix}/{save_data_name_530}',f'{dataset_prefix}/{save_data_name_531}',f'{dataset_prefix}/{save_data_name_532}',f'{dataset_prefix}/{save_data_name_533}',f'{dataset_prefix}/{save_data_name_534}',f'{dataset_prefix}/{save_data_name_535}',f'{dataset_prefix}/{save_data_name_536}',f'{dataset_prefix}/{save_data_name_537}',f'{dataset_prefix}/{save_data_name_538}',f'{dataset_prefix}/{save_data_name_539}',f'{dataset_prefix}/{save_data_name_540}',f'{dataset_prefix}/{save_data_name_541}',f'{dataset_prefix}/{save_data_name_542}',f'{dataset_prefix}/{save_data_name_543}',f'{dataset_prefix}/{save_data_name_544}',f'{dataset_prefix}/{save_data_name_545}',f'{dataset_prefix}/{save_data_name_546}',f'{dataset_prefix}/{save_data_name_547}',f'{dataset_prefix}/{save_data_name_548}',f'{dataset_prefix}/{save_data_name_549}',f'{dataset_prefix}/{save_data_name_550}',f'{dataset_prefix}/{save_data_name_551}',f'{dataset_prefix}/{save_data_name_552}',f'{dataset_prefix}/{save_data_name_553}',f'{dataset_prefix}/{save_data_name_554}',f'{dataset_prefix}/{save_data_name_555}',f'{dataset_prefix}/{save_data_name_556}',f'{dataset_prefix}/{save_data_name_557}',f'{dataset_prefix}/{save_data_name_558}',f'{dataset_prefix}/{save_data_name_559}',f'{dataset_prefix}/{save_data_name_560}',f'{dataset_prefix}/{save_data_name_561}',f'{dataset_prefix}/{save_data_name_562}',f'{dataset_prefix}/{save_data_name_563}',f'{dataset_prefix}/{save_data_name_564}',f'{dataset_prefix}/{save_data_name_565}',f'{dataset_prefix}/{save_data_name_566}',f'{dataset_prefix}/{save_data_name_567}',f'{dataset_prefix}/{save_data_name_568}',f'{dataset_prefix}/{save_data_name_569}',
            ]
        elif num_train_objects == 'mixed_old_and_real_world_noisy_1119':
            dataset_prefix_1 = '/scratch/yufeiw2/dp3_demo'
            dataset_prefix_2 = '/scratch/yufeiw2/dp3_demo_real_world_noise_pcd'
            
            old_list = [i * 3 for i in range(150)]
            all_old_obj_paths = ["{}/{}".format(dataset_prefix_1, globals()["save_data_name_{}".format(i)]) for i in old_list]
            
            all_new_obj_paths = os.listdir(dataset_prefix_2)
            all_new_obj_paths = sorted(all_new_obj_paths)
            all_new_obj_paths = [os.path.join(dataset_prefix_2, x) for x in all_new_obj_paths]
            
            all_obj_paths = all_old_obj_paths + all_new_obj_paths
            
        elif num_train_objects == 'real_world_noisy_pcd_clean_distorted_goal_all':
            dataset_prefix = '/scratch/yufeiw2/dp3_demo_real_world_noise_pcd_clean_distorted_goal'
            all_obj_paths = os.listdir(dataset_prefix)
            all_obj_paths = sorted(all_obj_paths)
            all_obj_paths = [os.path.join(dataset_prefix, x) for x in all_obj_paths]
        
        elif num_train_objects == '500_plus_all_real_world':
            non_real_world_camera_500_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(463)]
            real_world_camera_500_paths = os.listdir("/scratch/yufeiw2/dp3_demo_real_world_noise_pcd")
            real_world_camera_500_paths = sorted(real_world_camera_500_paths)
            real_world_camera_500_paths = [os.path.join("/scratch/yufeiw2/dp3_demo_real_world_noise_pcd", x) for x in real_world_camera_500_paths]
            all_obj_paths = non_real_world_camera_500_paths + real_world_camera_500_paths
            # all_obj_paths = [os.path.join(dataset_prefix, x) for x in all_obj_paths]
            print(all_obj_paths)
            
        elif num_train_objects == '500_plus_all_real_world_clean_distorted_goal':
            dataset_prefix = "/scratch/yufeiw2/dp3_demo_clean_distorted_goal"
            non_real_world_camera_500_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(463)]
            real_world_camera_500_paths = os.listdir("/scratch/yufeiw2/dp3_demo_real_world_noise_pcd_clean_distorted_goal")
            real_world_camera_500_paths = sorted(real_world_camera_500_paths)
            real_world_camera_500_paths = [os.path.join("/scratch/yufeiw2/dp3_demo_real_world_noise_pcd_clean_distorted_goal", x) for x in real_world_camera_500_paths]
            all_obj_paths = non_real_world_camera_500_paths + real_world_camera_500_paths
            # all_obj_paths = [os.path.join(dataset_prefix, x) for x in all_obj_paths]
            print(all_obj_paths)
            
        elif num_train_objects == "sriam_plate":
            dataset_prefix = "/project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/data_yufei/aloha"
            all_obj_paths = os.listdir(dataset_prefix)
            all_obj_paths = sorted(all_obj_paths)
            all_obj_paths = [os.path.join(dataset_prefix, x) for x in all_obj_paths if 'new_rot' not in x]
            print(all_obj_paths)
            
        elif num_train_objects == "sriram_towel":
            dataset_prefix = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/data/aloha"
            all_obj_paths = os.listdir(dataset_prefix)
            all_obj_paths = sorted(all_obj_paths)
            all_obj_paths = [os.path.join(dataset_prefix, x) for x in all_obj_paths if 'towel' in x]
            print(all_obj_paths)

        elif num_train_objects == "sriram_plate_new_rot":
            dataset_prefix = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/data/aloha"
            all_obj_paths = os.listdir(dataset_prefix)
            all_obj_paths = sorted(all_obj_paths)
            all_obj_paths = [os.path.join(dataset_prefix, x) for x in all_obj_paths if 'new_rot' in x]
            print(all_obj_paths)
            
        elif num_train_objects == "sriram_plate_new_rot_rgb":
            dataset_prefix = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/data/aloha"
            all_obj_paths = os.listdir(dataset_prefix)
            all_obj_paths = sorted(all_obj_paths)
            all_obj_paths = [os.path.join(dataset_prefix, x) for x in all_obj_paths if 'rgb' in x]
            print(all_obj_paths)
            
        elif num_train_objects == "sriram_towel":
            dataset_prefix = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/data/aloha"
            all_obj_paths = os.listdir(dataset_prefix)
            all_obj_paths = sorted(all_obj_paths)
            all_obj_paths = [os.path.join(dataset_prefix, x) for x in all_obj_paths if 'folding-towel' in x]
            print(all_obj_paths)

        elif num_train_objects == "mimicgen_square_d2":
            dataset_prefix = "/project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/data_yufei/mimicgen/"
            all_obj_paths = os.listdir(dataset_prefix)
            all_obj_paths = sorted(all_obj_paths)
            all_obj_paths = [os.path.join(dataset_prefix, x) for x in all_obj_paths if 'square_d2' in x]
            print(all_obj_paths)

        elif num_train_objects == "mimicgen_three_piece_assembly_d2":
            dataset_prefix = "/project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/data_yufei/mimicgen/"
            all_obj_paths = os.listdir(dataset_prefix)
            all_obj_paths = sorted(all_obj_paths)
            all_obj_paths = [os.path.join(dataset_prefix, x) for x in all_obj_paths if 'three_piece_assembly_d2' in x]
            print(all_obj_paths)

        elif num_train_objects == "mimicgen_threading_d2":
            dataset_prefix = "/project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/data_yufei/mimicgen/"
            all_obj_paths = os.listdir(dataset_prefix)
            all_obj_paths = sorted(all_obj_paths)
            all_obj_paths = [os.path.join(dataset_prefix, x) for x in all_obj_paths if 'threading_d2' in x]
            print(all_obj_paths)

        elif num_train_objects == "mimicgen_mug_cleanup_d1":
            dataset_prefix = "/project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/data_yufei/mimicgen/"
            all_obj_paths = os.listdir(dataset_prefix)
            all_obj_paths = sorted(all_obj_paths)
            all_obj_paths = [os.path.join(dataset_prefix, x) for x in all_obj_paths if 'mug_cleanup_d1' in x]
            print(all_obj_paths)
                        
        else:
            raise ValueError('num_train_objects not supported')
    dataset = PointNetDatasetFromDisk(all_obj_paths, beg_ratio, end_ratio, eval_episode, only_first_stage, 
                                        is_pickle=is_pickle, use_all_data=use_all_data, conditioning_on_demo=conditioning_on_demo, camera_frame=camera_frame, 
                                        goal_always_open=goal_always_open, use_rgb=use_rgb, use_dino=use_dino, pred_gripper_width=pred_gripper_width, gripper_width_scale_factor=gripper_width_scale_factor)    
    return dataset