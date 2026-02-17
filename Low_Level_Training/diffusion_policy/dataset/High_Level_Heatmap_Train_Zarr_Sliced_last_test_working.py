# from typing import Dict, List
import torch
import zarr
import numpy as np
from PIL import Image
from torchvision import transforms
import re
import numba
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset, BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)
import pickle
import time
# from torch.utils.data._utils.pin_memory import pin_memory
import torch.nn.functional as F

EPISODE_FOLDER = 'episode%d'
LOW_DIM_PICKLE = 'low_dim_obs.pkl'
import os

# Point to your CoppeliaSim Player folder
os.environ['COPPELIASIM_ROOT'] = '/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/RVT2_GMM_Codebase/Coppeliasim/CoppeliaSim_Player_V4_1_0_Ubuntu18_04'

# Optional: adjust LD_LIBRARY_PATH in Python itself
ld_path = os.environ.get('LD_LIBRARY_PATH', '')
os.environ['LD_LIBRARY_PATH'] = f"{os.environ['COPPELIASIM_ROOT']}:{ld_path}"

# @numba.jit(nopython=True)
def create_indices(
    episode_ends:np.ndarray, sequence_length:int, 
    episode_mask: np.ndarray,
    pad_before: int=0, pad_after: int=0,
    debug:bool=True) -> np.ndarray:
    # import pdb; pdb.set_trace();
    episode_mask.shape == episode_ends.shape        
    pad_before = min(max(pad_before, 0), sequence_length-1)
    pad_after = min(max(pad_after, 0), sequence_length-1)

    indices = list()
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            # skip episode
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx
        
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        # import pdb; pdb.set_trace()
        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert(start_offset >= 0)
                assert(end_offset >= 0)
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
            indices.append([
                buffer_start_idx, buffer_end_idx, 
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


class HighLevelHeatmapTrain(BaseImageDataset):  # torch.utils.data.Dataset
    def __init__(self, episode_path, sequence_length, pad_before, pad_after, use_wrist_camera, episode_mask = None, num_cameras = 4, return_single_image = False): # image_path, heatmap_path, low_dim_path, depth_path
        super().__init__()
        self.return_single_image = return_single_image
        self.use_wrist_camera = use_wrist_camera
        self.num_cameras = num_cameras
        # self.low_dim_path = low_dim_path
        # self.image_path = image_path
        # self.heatmap_path = heatmap_path
        self.episode_path = episode_path
        self.sequence_length = sequence_length
        import pdb; pdb.set_trace()
        self.episode_paths = [os.path.join(self.episode_path, f) for f in os.listdir(self.episode_path)]
        self.episode_paths = sorted(self.episode_paths, key=self.extract_number)
        import pdb; pdb.set_trace()
        self.episode_lengths = []
        for ep_pa in self.episode_paths:
            ep = zarr.open(ep_pa)
            self.episode_lengths.append(len(ep["rgb/camera1"]))
        import pdb; pdb.set_trace()
        self.accumulated_episode_lengths = np.cumsum(self.episode_lengths)

        # self.camera_image_paths = [os.path.join(self.image_path, f) for f in os.listdir(self.image_path)]
        # self.camera_image_paths = sorted(self.camera_image_paths) #, key=self.extract_number

        # self.depth_path = depth_path
        # self.camera_depth_paths = [os.path.join(self.depth_path, f) for f in os.listdir(self.depth_path)]
        # self.camera_depth_paths = sorted(self.camera_depth_paths) #, key=self.extract_number
        # # import pdb; pdb.set_trace();
        # self.sequence_length = sequence_length
        # self.camera_heatmap_paths = [os.path.join(self.heatmap_path, f) for f in os.listdir(self.heatmap_path)]
        # self.camera_heatmap_paths = sorted(self.camera_heatmap_paths) # , key=self.extract_number
        # # import pdb; pdb.set_trace();
        # self.low_dim_paths = [os.path.join(low_dim_path, f) for f in os.listdir(low_dim_path)]
        # self.low_dim_paths = sorted(self.low_dim_paths) # , key=self.extract_number
        # # import pdb; pdb.set_trace();
        # self.episodes_images = []
        # for camera_path in self.camera_image_paths:
        #     single_camera_episode_paths = [os.path.join(camera_path, f) for f in os.listdir(camera_path)]
        #     single_camera_episode_paths = sorted(single_camera_episode_paths) # key=self.extract_number
        #     # import pdb; pdb.set_trace();
        #     self.episodes_images.append(single_camera_episode_paths)
        # self.episode_lengths = []
        # for i in range(len(self.camera_heatmap_paths)):
        #     for j, episode in enumerate(self.episodes_images[i]):
        #         # import pdb; pdb.set_trace();
        #         episode_images = [os.path.join(episode,f) for f in os.listdir(episode)]
        #         episode_images = sorted(episode_images, key=self.extract_number)
        #         if j==0:
        #             self.episode_lengths.append(len(episode_images))
        # self.accumulated_episode_lengths = np.cumsum(self.episode_lengths)
        import pdb; pdb.set_trace();
        if episode_mask is None:
            episode_mask = np.ones(self.accumulated_episode_lengths.shape, dtype=bool)

        if np.any(episode_mask):
            self.indices = create_indices(self.accumulated_episode_lengths, 
                sequence_length=sequence_length, 
                pad_before=pad_before, 
                pad_after=pad_after,
                episode_mask=episode_mask
                )
        else:
            self.indices = np.zeros((0,4), dtype=np.int64)
        import pdb; pdb.set_trace()
            
    def extract_number(self, path):
        # Match digits before .png, .pkl, or .npy at the end of the filename
        match = re.search(r'(\d+)\.(?:png|pkl|npy|zarr)$', path)
        return int(match.group(1)) if match else -1


    


    def __len__(self):
        return len(self.indices) #self.accumulated_episode_lengths[-1]

    def read_pickle_data(self, proprio_data, start_idx, end_idx): #file_name
        # import pdb; pdb.set_trace();
        gripper_pose = torch.from_numpy(proprio_data["pose"][start_idx:end_idx]) #.clone().detach().cpu() #torch.tensor(data["gripper_pose"].detach().cpu())
        gripper_open_close = torch.from_numpy(proprio_data["open_close"][start_idx:end_idx])  #.clone().detach().cpu() #torch.tensor(data["gripper_open_close"].detach().cpu())
        proprioceprion = torch.cat((gripper_pose, gripper_open_close), dim = -1).squeeze(0)
        gripper_action = torch.from_numpy(proprio_data["action"][start_idx:end_idx])    # .clone().detach().cpu().squeeze(0) # torch.tensor(data["gripper_action"].detach().cpu()).squeeze(0)
        return proprioceprion, gripper_open_close # proprioceprion, #step_path # gripper_action


    def read_language_instructions(self, episode_of_interest):
        start_idx = 0 
        end_idx = 8
        # import pdb; pdb.set_trace();
        lang_emb = episode_of_interest["language"]["pooled_text_features"][start_idx:end_idx]
        return lang_emb


    def read_images(self, episode_of_interest, start_idx, end_idx): #step_idx image_path, heatmap_path, depth_path, episode_idx, file_name
        # import pdb; pdb.set_trace()
        self.use_wrist_camera = False
        if self.use_wrist_camera:
            total_cams = self.num_cameras + 1
        else:
            total_cams = self.num_cameras
        rgb_step_idx = []
        for cam_num in range(1, total_cams):
            # import pdb; pdb.set_trace()
            rgb = episode_of_interest["rgb"][f"camera{cam_num}"][start_idx:end_idx]  # (H, W, 3), uint8

            # rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0   # (3, H, W)
            # rgb = torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255.0
            # rgb = torch.from_numpy(rgb).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
            # rgb = F.interpolate(
            #     rgb,          # (1, 3, H, W)
            #     size=(128, 128),
            #     mode="bilinear",
            #     align_corners=False
            # )                   # (3, 128, 128)

            rgb_step_idx.append(torch.from_numpy(rgb))
        # import pdb; pdb.set_trace()
        rgb_step_idx = torch.cat(rgb_step_idx, dim=1)

        heatmap_step_idx = []
        for cam_num in range(1, self.num_cameras):
            # import pdb; pdb.set_trace()
            heatmap = episode_of_interest["heatmaps"]["camera"+str(cam_num)][start_idx:end_idx]
            # heatmap = np.expand_dims(heatmap.mean(axis=3), axis=3)
            # import pdb; pdb.set_trace()
            # heatmap = torch.from_numpy(heatmap).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
            # heatmap = F.interpolate(
            #     heatmap,          # (1, 3, H, W)
            #     size=(128, 128),
            #     mode="bilinear",
            #     align_corners=False
            # )                   # (3, 128, 128)

            heatmap_step_idx.append(torch.from_numpy(heatmap))
        # import pdb; pdb.set_trace()
        heatmap_step_idx = torch.cat(heatmap_step_idx, dim=1)

        depth_step_idx = []
        for cam_num in range(1, self.num_cameras):
            # import pdb; pdb.set_trace()
            depth = episode_of_interest["depth"]["camera"+str(cam_num)][start_idx:end_idx]
            # depth = np.expand_dims(depth.mean(axis=3), axis=3)
            # import pdb; pdb.set_trace()
            # depth = (torch.from_numpy(depth).permute(0, 3, 1, 2).contiguous().float().div_(255.0))

            # depth = F.interpolate(
            #     depth,          # (1, 3, H, W)
            #     size=(128, 128),
            #     mode="bilinear",
            #     align_corners=False
            # )  

            depth_step_idx.append(torch.from_numpy(depth))
        # import pdb; pdb.set_trace()
        depth_step_idx = torch.cat(depth_step_idx, dim=1)

        return rgb_step_idx, heatmap_step_idx, depth_step_idx #, image_path
    
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        # import pdb; pdb.set_trace()
        normalizer = LinearNormalizer()
        
        actions_all = []
        agent_pos_all = []
        # for path in self.low_dim_paths:

            # file_names = os.listdir(path)
        for idx, episode_of_interest_path in enumerate(self.episode_paths):
            # import pdb; pdb.set_trace()
            episode_of_interest = zarr.open(episode_of_interest_path)
            proprioception_data = episode_of_interest["gripper"]
            length_proprio_data = len(proprioception_data["pose"])
            temp_pro_for_norm, _ = self.read_pickle_data(proprio_data = proprioception_data, start_idx = 0, end_idx = length_proprio_data) # path, file_name
            # import pdb; pdb.set_trace()
            agent_pos_all.append(temp_pro_for_norm)
            actions_all.append(temp_pro_for_norm)
        # import pdb; pdb.set_trace();
        actions_all = torch.cat(actions_all, axis=0).squeeze(1)
        agent_pos_all = torch.cat(agent_pos_all, axis=0).squeeze(1)
        # print("BEFORE EVERYTHINGGGGGGGGGGG", actions_all.shape, agent_pos_all.shape)
        # import pdb; pdb.set_trace();
        # print("STAT ACTION POS")
        stat_action_pos = array_to_stats(actions_all[:,:3].numpy())
        # print("Array to Stats STAT ACTION POS", stat_action_pos)
        # print("STAT QUATERNION ACTION")
        stat_action_quat = array_to_stats(actions_all[:,3:7].numpy())
        # print("Array to Stats STAT QUATERNION ACTION", stat_action_quat)
        # import pdb; pdb.set_trace();
        this_normalizer_pos = get_range_normalizer_from_stat(stat_action_pos)
        this_normalizer_quat = get_identity_normalizer_from_stat(stat_action_quat)
        # import pdb; pdb.set_trace();
        this_normalizer = [this_normalizer_pos, this_normalizer_quat]
        normalizer['action_pos'] = this_normalizer_pos
        normalizer['action_quat'] = this_normalizer_quat

        # print("STAT AGENT POS")
        stat_action_pos_agent_pos = array_to_stats(agent_pos_all[:,:3].numpy())
        # print("Array to Stats STAT AGENT POS", stat_action_pos_agent_pos)
        # print("STAT AGENT QUARTERNION POS")
        stat_action_quat_agent_pos = array_to_stats(agent_pos_all[:,3:7].numpy()) 
        # print("Array to Stats STAT AGENT QUARTERNION POS", stat_action_quat_agent_pos)
        this_normalizer_pos_agent_pos = get_range_normalizer_from_stat(stat_action_pos_agent_pos)
        this_normalizer_quat_agent_pos = get_identity_normalizer_from_stat(stat_action_quat_agent_pos)
        this_normalizer_agent_pos = [this_normalizer_pos_agent_pos, this_normalizer_quat_agent_pos]
        normalizer['agent_pos_pos'] = this_normalizer_pos_agent_pos
        normalizer['agent_pos_quat'] = this_normalizer_quat_agent_pos

        normalizer["image_cam1"] = get_image_range_normalizer()
        normalizer["image_cam2"] = get_image_range_normalizer()
        normalizer["image_cam3"] = get_image_range_normalizer()
        return normalizer
    
    
    def __getitem__(self,idx):
        # import pdb; pdb.set_trace();
        # print("idxxxxxx", idx)
        # import pdb; pdb.set_trace()
        # t0 = time.time()
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx \
            = self.indices[idx]
        episode_idx = np.searchsorted(self.accumulated_episode_lengths, buffer_start_idx, side='right') # idx
        start_idx = buffer_start_idx - self.accumulated_episode_lengths[episode_idx] # idx
        if start_idx < 0:
            start_idx += self.episode_lengths[episode_idx]
        end_idx = start_idx + (sample_end_idx - sample_start_idx)
        sample_proprioception = []
        sample_image = []
        sample_heatmap = []
        sample_depth = []
        goal_sample_proprioception = []
        # import pdb; pdb.set_trace();
        # print(start_idx, end_idx)
        # for indices in range(start_idx, end_idx):
            # print(indices)
            # import pdb; pdb.set_trace()
            # local_episode_files = os.listdir(self.low_dim_paths[episode_idx])
            # local_episode_files = sorted(local_episode_files, key=self.extract_number)
            # import pdb; pdb.set_trace();

        ###################### LOAD THE PROPRIO DATA #############################
        episode_of_interest_path = self.episode_paths[episode_idx]
        episode_of_interest = zarr.open(episode_of_interest_path) #local_episode_files[indices]
        proprioception_data = episode_of_interest["gripper"]
        sample_proprioception, action_temp_pro = self.read_pickle_data(proprioception_data, start_idx=start_idx, end_idx = end_idx)
        # goal_sample_proprioception.append(action_temp_pro)
        # sample_proprioception.append(temp_pro)

        ###################### LOAD THE IMAGE AND HEATMAP DATA #####################
        # import pdb; pdb.set_trace();
        sample_image, sample_heatmap, sample_depth = self.read_images(episode_of_interest, start_idx=start_idx, end_idx = end_idx)


        ###################### LOAD THE LANGUAGE DATA ##############################
        lang_emb = self.read_language_instructions(episode_of_interest)

        #temp_pro, _, action_temp_pro, step_path_CHECK = self.read_pickle_data(self.low_dim_paths[episode_idx], the_file_used)
        
        # the_file_used = the_file_used.replace('.pkl', '.png')
        # import pdb; pdb.set_trace()
        # this_episode_idx = os.path.basename(self.low_dim_paths[episode_idx])
        # temp_image, temp_heatmap, temp_depth , image_path_CHECK = self.read_images(self.image_path, self.heatmap_path, self.depth_path, this_episode_idx, the_file_used)
        # print("PICKLE FILE", step_path_CHECK , "IMAGE FILE", image_path_CHECK)
        
        # sample_image.append(temp_image)
        # sample_heatmap.append(temp_heatmap)
        # sample_depth.append(temp_depth)





        # import pdb; pdb.set_trace()
        # goal_sample_proprioception = torch.stack(goal_sample_proprioception, dim=0)
        # sample_proprioception = torch.stack(sample_proprioception, dim=0)
        # sample_image = torch.stack(sample_image, dim=0)
        # sample_heatmap = torch.stack(sample_heatmap, dim=0)
        # sample_depth = torch.stack(sample_depth, dim=0)
        # import pdb; pdb.set_trace()
        gripper_proprioception = torch.zeros(
            (self.sequence_length,) + sample_proprioception.shape[1:], 
            dtype=sample_proprioception.dtype, 
            device=sample_proprioception.device
        )
        gripper_image = torch.zeros(
            (self.sequence_length,) + sample_image.shape[1:], 
            dtype=sample_proprioception.dtype, 
            device=sample_proprioception.device
        )
        gripper_heatmap = torch.zeros(
            (self.sequence_length,) + sample_heatmap.shape[1:], 
            dtype=sample_heatmap.dtype, 
            device=sample_heatmap.device
        )
        gripper_depth = torch.zeros(
            (self.sequence_length,) + sample_depth.shape[1:], 
            dtype=sample_depth.dtype, 
            device=sample_depth.device
        )
        # goal_gripper_proprioception = torch.zeros(
        #     (self.sequence_length,) + goal_sample_proprioception.shape[1:], 
        #     dtype=sample_proprioception.dtype, 
        #     device=sample_proprioception.device
        # )
        if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
            if sample_start_idx > 0:
                # import pdb; pdb.set_trace()
                # data[:sample_start_idx] = #sample[0]
                gripper_proprioception[:sample_start_idx] = sample_proprioception[0]
                gripper_image[:sample_start_idx] = sample_image[0]
                gripper_heatmap[:sample_start_idx] = sample_heatmap[0]
                gripper_depth[:sample_start_idx] = sample_depth[0]
                # goal_gripper_proprioception[:sample_start_idx] = goal_sample_proprioception[0]
            if sample_end_idx < self.sequence_length:
                # import pdb; pdb.set_trace()
                # data[sample_end_idx:] = sample[-1]
                gripper_proprioception[sample_end_idx:] = sample_proprioception[-1]
                gripper_image[:sample_start_idx] = sample_image[-1]
                gripper_heatmap[:sample_start_idx] = sample_heatmap[-1]
                gripper_depth[:sample_start_idx] = sample_depth[-1]
                # goal_gripper_proprioception[:sample_start_idx] = goal_sample_proprioception[-1]
        # import pdb; pdb.set_trace()
        gripper_proprioception[sample_start_idx:sample_end_idx] = sample_proprioception
        gripper_image[sample_start_idx:sample_end_idx] = sample_image
        gripper_heatmap[sample_start_idx:sample_end_idx] = sample_heatmap
        gripper_depth[sample_start_idx:sample_end_idx] = sample_depth
        # goal_gripper_proprioception[sample_start_idx:sample_end_idx] = goal_sample_proprioception
        # import pdb; pdb.set_trace();
        # if gripper_image.shape[0] == 0 or gripper_heatmap.shape[0] == 0 or gripper_proprioception is None or goal_gripper_proprioception is None:
            # print("NONEEEEEEEEE NOTEDDDDDDDDD")
        for x, name in zip(
        [gripper_image, gripper_heatmap, gripper_depth, gripper_proprioception, ], # goal_gripper_proprioception
        ["gripper_image", "gripper_heatmap", "gripper_depth", "gripper_proprioception", ] # "goal_gripper_proprioception"
        ):
            if not isinstance(x, torch.Tensor):
                print(f"Bad type detected: {name} is {type(x)} at idx {idx}")
        # print("IMAGEEEEEE", torch.cat((gripper_image, gripper_heatmap), dim = 1).shape)
        # print("AGENTTTTTTTTTTT POSSSSSS", gripper_proprioception.shape)
        # print("ACTIONNNNNNNNNNNNN", goal_gripper_proprioception.shape)
        # import pdb; pdb.set_trace();
        if lang_emb is None:
            print("NONEEEEEEEEEEEEEEEEEEEEE READ HEREEEEEE SUDDENLYYYYYY !!!!!!", lang_emb)
            import pdb; pdb.set_trace()
        if self.return_single_image:
            data = {
                'obs': {
                    'image': torch.cat((gripper_image, gripper_heatmap, gripper_depth), dim = 1).contiguous(), # T, 3, 96, 96
                    'agent_pos': gripper_proprioception.squeeze(1).contiguous(), # T, 2
                },
                'obs_lang_emb' : torch.tensor(lang_emb).squeeze(0),
                'action': gripper_proprioception.squeeze(1).contiguous() # T, 2 goal_gripper_proprioception
            }
        else:
            # import pdb; pdb.set_trace()
            data = {
                'obs': {
                    'image_cam1': torch.cat((gripper_image[:,:3,:,:], gripper_heatmap[:,:1,:,:], gripper_depth[:,:1,:,:]), dim = 1).contiguous(), # T, 3, 96, 96
                    'image_cam2': torch.cat((gripper_image[:,3:6,:,:], gripper_heatmap[:,1:2,:,:], gripper_depth[:,1:2,:,:]), dim = 1).contiguous(),
                    'image_cam3': torch.cat((gripper_image[:,6:9,:,:], gripper_heatmap[:,2:3,:,:], gripper_depth[:,2:3,:,:]), dim = 1).contiguous(),
                    'agent_pos': gripper_proprioception.squeeze(1).contiguous(), # T, 2
                },
                'obs_lang_emb' : torch.tensor(lang_emb).squeeze(0),
                'action': gripper_proprioception.squeeze(1).contiguous() # T, 2 goal_gripper_proprioception
            }
        # import pdb; pdb.set_trace();
        # import pdb; pdb.set_trace();
        # import pdb; pdb.set_trace();
        for x, name in zip(
            [gripper_image, gripper_heatmap, gripper_depth,
            gripper_proprioception, ], # goal_gripper_proprioception
            ["gripper_image", "gripper_heatmap", "gripper_depth",
            "gripper_proprioception",] #  "goal_gripper_proprioception"
        ):
            assert x.numel() > 0, f"{name} is empty at idx={idx}"
        assert gripper_image.is_contiguous(), f"image not contiguous at idx={idx}"
        # pin_memory(data, device=0)
        # t1 = time.time()
        # print("TIMEEEEEE", t1 - t0)
        return data

    # functions
    def get_stored_demo(self, data_path, index):
        episode_path = os.path.join(data_path, EPISODE_FOLDER % index)
        
        # low dim pickle file
        with open(os.path.join(episode_path, LOW_DIM_PICKLE), 'rb') as f:
            obs = pickle.load(f)

            
        return obs


# image_path = "/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/RVT2_GMM_Codebase/RVT2_Codebase/RVT/outputs_11th/unnormalized_rgb" 
# heatmap_path="/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/RVT2_GMM_Codebase/RVT2_Codebase/RVT/outputs_11th/unnormalized_heatmap_images/" 
# low_dim_path="/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/RVT2_GMM_Codebase/RVT2_Codebase/RVT/outputs_11th/gripper_pose/"
# depth_path = "/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/RVT2_GMM_Codebase/RVT2_Codebase/RVT/outputs_11th/depth/"
# episode_path = "/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/RVT2_GMM_Codebase/RVT2_Codebase/RVT_Backup_IDK/RVT/data_diffusion_policy/outputs_1st_insert_onto_square_peg/trajectories_data_FINAL_WITH_TEXT_EMBEDDING" #"/scratch/pbhowal/Diffusion_Policy_Training/Zarr_Trajectories/insert_onto_square_peg_WITH_LANGUAGE_EMBEDDINGS/" #"/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/RVT2_GMM_Codebase/RVT2_Codebase/RVT_Backup_IDK/RVT/data_diffusion_policy/outputs_1st_insert_onto_square_peg/trajectories_data_FINAL_WITH_TEXT_EMBEDDING" #"/scratch/pbhowal/Diffusion_Policy_Training/Zarr_Trajectories/insert_onto_square_peg/" #"/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/RVT2_GMM_Codebase/RVT2_Codebase/RVT_Backup_IDK/RVT/data_diffusion_policy/outputs_1st_insert_onto_square_peg/trajectories_data"
# dataset = HighLevelHeatmapTrain(episode_path=episode_path, sequence_length = 16, pad_before = 4, pad_after = 4, use_wrist_camera = True)
# # # dataset.get_normalizer()
# # # # # # # import pdb; pdb.set_trace()
# print(dataset[718])
# # print("DATASET LEN", len(dataset))
# for i in range(len(dataset)):
#     print(i, dataset[i]['obs_lang_emb'].shape) #dataset[i].keys
# import pdb; pdb.set_trace()
# print(dataset[2000])
# for i in range(36500, 159874):
#     dataset[i]
#     print(i)