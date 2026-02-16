# # from typing import Dict, List
# import torch
# import numpy as np
# from PIL import Image
# from torchvision import transforms
# import re
# # import numba
# from diffusion_policy.dataset.base_dataset import BaseLowdimDataset, BaseImageDataset
# from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
# from diffusion_policy.common.normalize_util import (
#     robomimic_abs_action_only_normalizer_from_stat,
#     robomimic_abs_action_only_dual_arm_normalizer_from_stat,
#     get_range_normalizer_from_stat,
#     get_image_range_normalizer,
#     get_identity_normalizer_from_stat,
#     array_to_stats
# )
# import pickle

# EPISODE_FOLDER = 'episode%d'
# LOW_DIM_PICKLE = 'low_dim_obs.pkl'
# import os

# # Point to your CoppeliaSim Player folder
# os.environ['COPPELIASIM_ROOT'] = '/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/RVT2_GMM_Codebase/Coppeliasim/CoppeliaSim_Player_V4_1_0_Ubuntu18_04'

# # Optional: adjust LD_LIBRARY_PATH in Python itself
# ld_path = os.environ.get('LD_LIBRARY_PATH', '')
# os.environ['LD_LIBRARY_PATH'] = f"{os.environ['COPPELIASIM_ROOT']}:{ld_path}"

# # @numba.jit(nopython=True)
# def create_indices(
#     episode_ends:np.ndarray, sequence_length:int, 
#     episode_mask: np.ndarray,
#     pad_before: int=0, pad_after: int=0,
#     debug:bool=True) -> np.ndarray:
#     # import pdb; pdb.set_trace();
#     episode_mask.shape == episode_ends.shape        
#     pad_before = min(max(pad_before, 0), sequence_length-1)
#     pad_after = min(max(pad_after, 0), sequence_length-1)

#     indices = list()
#     for i in range(len(episode_ends)):
#         if not episode_mask[i]:
#             # skip episode
#             continue
#         start_idx = 0
#         if i > 0:
#             start_idx = episode_ends[i-1]
#         end_idx = episode_ends[i]
#         episode_length = end_idx - start_idx
        
#         min_start = -pad_before
#         max_start = episode_length - sequence_length + pad_after
        
#         # range stops one idx before end
#         for idx in range(min_start, max_start+1):
#             buffer_start_idx = max(idx, 0) + start_idx
#             buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
#             start_offset = buffer_start_idx - (idx+start_idx)
#             end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
#             sample_start_idx = 0 + start_offset
#             sample_end_idx = sequence_length - end_offset
#             if debug:
#                 assert(start_offset >= 0)
#                 assert(end_offset >= 0)
#                 assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
#             indices.append([
#                 buffer_start_idx, buffer_end_idx, 
#                 sample_start_idx, sample_end_idx])
#     indices = np.array(indices)
#     return indices


# class HighLevelHeatmapTrain(BaseImageDataset):  # torch.utils.data.Dataset
#     def __init__(self, image_path, heatmap_path, low_dim_path, depth_path, sequence_length, pad_before, pad_after, episode_mask = None, num_cameras = 4):
#         super().__init__()
#         self.num_cameras = num_cameras
#         self.low_dim_path = low_dim_path
#         self.image_path = image_path
#         self.heatmap_path = heatmap_path
#         self.camera_image_paths = [os.path.join(self.image_path, f) for f in os.listdir(self.image_path)]
#         self.camera_image_paths = sorted(self.camera_image_paths) #, key=self.extract_number

#         self.depth_path = depth_path
#         self.camera_depth_paths = [os.path.join(self.depth_path, f) for f in os.listdir(self.depth_path)]
#         self.camera_depth_paths = sorted(self.camera_depth_paths) #, key=self.extract_number
#         # import pdb; pdb.set_trace();
#         self.sequence_length = sequence_length
#         self.camera_heatmap_paths = [os.path.join(self.heatmap_path, f) for f in os.listdir(self.heatmap_path)]
#         self.camera_heatmap_paths = sorted(self.camera_heatmap_paths) # , key=self.extract_number
#         # import pdb; pdb.set_trace();
#         self.low_dim_paths = [os.path.join(low_dim_path, f) for f in os.listdir(low_dim_path)]
#         self.low_dim_paths = sorted(self.low_dim_paths) # , key=self.extract_number
#         # import pdb; pdb.set_trace();
#         self.episodes_images = []
#         for camera_path in self.camera_image_paths:
#             single_camera_episode_paths = [os.path.join(camera_path, f) for f in os.listdir(camera_path)]
#             single_camera_episode_paths = sorted(single_camera_episode_paths) # key=self.extract_number
#             # import pdb; pdb.set_trace();
#             self.episodes_images.append(single_camera_episode_paths)
#         # import pdb; pdb.set_trace();
#         # self.episodes_images = os.listdir(self.camera_image_paths[0])
#         # self.episodes_heatmaps = []
#         # for camera_path in self.camera_heatmap_paths:
#         #     single_camera_episode_paths = [os.path.join(camera_path, f) for f in os.listdir(camera_path)]
#         #     single_camera_episode_paths = sorted(single_camera_episode_paths, key=self.extract_number)
#         #     self.episodes_heatmaps.append(single_camera_episode_paths)
#         # # self.episodes_heatmaps = os.listdir(self.camera_heatmap_paths[0])
#         # assert len(self.episodes_images) == len(self.episodes_heatmaps)
#         # for i in range(len(self.camera_heatmap_paths)):
#         #     assert len(self.episodes_images[i]) == len(self.episodes_heatmaps[i])
#         # import pdb; pdb.set_trace();
#         # self.all_image_paths = []
#         self.episode_lengths = []
#         for i in range(len(self.camera_heatmap_paths)):
#             # all_image_paths_single_camera = []
#             for j, episode in enumerate(self.episodes_images[i]):
#                 # import pdb; pdb.set_trace();
#                 episode_images = [os.path.join(episode,f) for f in os.listdir(episode)]
#                 episode_images = sorted(episode_images, key=self.extract_number)
#                 # import pdb; pdb.set_trace();
#                 # import pdb; pdb.set_trace();
#                 # all_image_paths_single_camera.append(episode_images)
#                 # import pdb; pdb.set_trace();
#                 if j==0:
#                     self.episode_lengths.append(len(episode_images))
#             # self.all_image_paths.append(all_image_paths_single_camera)
#         # import pdb; pdb.set_trace();
#         # self.all_heatmap_paths = []
#         # for i in range(len(self.camera_heatmap_paths)):
#         #     all_heatmap_paths_single_camera = []
#         #     for episode in self.episodes_heatmaps[i]:
#         #         episode_heatmaps = [os.path.join(episode,f) for f in os.listdir(episode)]
#         #         episode_heatmaps = sorted(episode_heatmaps, key=self.extract_number)
#         #         all_heatmap_paths_single_camera.append(episode_heatmaps)
#         #     self.all_heatmap_paths.append(all_heatmap_paths_single_camera)
#         # for i in range(len(self.camera_heatmap_paths)):
#         #     assert len(self.all_image_paths[i]) == len(self.all_heatmap_paths[i])
#         self.accumulated_episode_lengths = np.cumsum(self.episode_lengths)
#         # import pdb; pdb.set_trace();
#         if episode_mask is None:
#             episode_mask = np.ones(self.accumulated_episode_lengths.shape, dtype=bool)

#         if np.any(episode_mask):
#             self.indices = create_indices(self.accumulated_episode_lengths, 
#                 sequence_length=sequence_length, 
#                 pad_before=pad_before, 
#                 pad_after=pad_after,
#                 episode_mask=episode_mask
#                 )
#         else:
#             self.indices = np.zeros((0,4), dtype=np.int64)
#         # import pdb; pdb.set_trace();

#     # def extract_number(self,path):
#     #     # Finds digits before .png at the end
#     #     match = re.search(r'(\d+)\.png$', path)
#     #     return int(match.group(1)) if match else -1
#     def extract_number(self, path):
#         # Match digits before .png, .pkl, or .npy at the end of the filename
#         match = re.search(r'(\d+)\.(?:png|pkl|npy)$', path)
#         return int(match.group(1)) if match else -1



#     def __len__(self):
#         return len(self.indices) #self.accumulated_episode_lengths[-1]

#     def read_pickle_data(self, path, file_name):
#         # print("READ PICKLE")
#         # import pdb; pdb.set_trace();
#         step_path = os.path.join(path, file_name) #os.path.join(self.low_dim_paths[episode_idx], str(step_idx) + '.pkl')
#         with open(step_path, 'rb') as f:
#             # import pdb; pdb.set_trace();
#             data = pickle.load(f)
#         gripper_pose = data["gripper_pose"].clone().detach().cpu() #torch.tensor(data["gripper_pose"].detach().cpu())
#         gripper_open_close = data["gripper_open_close"].clone().detach().cpu() #torch.tensor(data["gripper_open_close"].detach().cpu())
#         proprioceprion = torch.cat((gripper_pose, gripper_open_close), dim = -1).squeeze(0)
#         gripper_action = data["gripper_action"].clone().detach().cpu().squeeze(0) # torch.tensor(data["gripper_action"].detach().cpu()).squeeze(0)
#         return proprioceprion, gripper_open_close, proprioceprion, step_path # gripper_action

    

#     def read_images(self, image_path, heatmap_path, depth_path, episode_idx, file_name): #step_idx
#         images_step_idx = []
#         heatmap_step_idx = []
#         depth_step_idx = []
#         transform = transforms.Compose([
#         #     transforms.Resize((128, 128)),
#             transforms.ToTensor(),
#         #     transforms.Normalize(mean=[0.5, 0.5, 0.5],
#         #                         std=[0.5, 0.5, 0.5])
#         ])
        
        
#         episode_path = os.path.join(image_path, episode_idx)
#         # print("EPISODE_PATH", episode_path)
#         for i in range(1, self.num_cameras): # self.all_image_paths[episode_idx]
#             # image_path = self.all_image_paths[episode_idx][i][step_idx]
#             camera_path = os.path.join(episode_path, "camera" + str(i))
#             image_path = os.path.join(camera_path, file_name)
#             # print("EPISODE INDEX", episode_idx, "image_path", image_path)
#             # print("THE IMAGE INSIDE CAMERA", i, "IS", step_idx, "PATH", image_path)
#             image = Image.open(image_path).convert('RGB')
#             image = transform(image)
#             images_step_idx.append(image)
#         # import pdb; pdb.set_trace();
#         images_step_idx = torch.cat(images_step_idx, dim=0)

#         heatmap_episode_path = os.path.join(heatmap_path, episode_idx)
#         for i in range(1, self.num_cameras):
#             heatmap_camera_path = os.path.join(heatmap_episode_path, "camera" + str(i))
#             # file_name = file_name.replace('.png', '.npy')
#             heatmap_path = os.path.join(heatmap_camera_path, file_name)
#             # import pdb; pdb.set_trace();
#             heatmap = Image.open(heatmap_path).convert('RGB')
#             # heatmap = np.load(heatmap_path)
#             heatmap = transform(heatmap)
#             heatmap = heatmap.mean(axis=0).unsqueeze(0)
#             heatmap_step_idx.append(heatmap)
#         heatmap_step_idx = torch.cat(heatmap_step_idx, dim=0)


#         depth_episode_path = os.path.join(depth_path, episode_idx)
#         for i in range(1, self.num_cameras):
#             depth_camera_path = os.path.join(depth_episode_path, "camera" + str(i))
#             # file_name = file_name.replace('.png', '.npy')
#             depth_path = os.path.join(depth_camera_path, file_name)
#             # import pdb; pdb.set_trace();
#             depth = Image.open(depth_path).convert('RGB')
#             # heatmap = np.load(heatmap_path)
#             depth = transform(depth)
#             depth = depth.mean(axis=0).unsqueeze(0)
#             depth_step_idx.append(depth)
#         depth_step_idx = torch.cat(depth_step_idx, dim=0)



#         # import pdb; pdb.set_trace();
#         return images_step_idx, heatmap_step_idx, depth_step_idx, image_path
    
#     def get_normalizer(self, **kwargs) -> LinearNormalizer:
#         normalizer = LinearNormalizer()
        
#         actions_all = []
#         agent_pos_all = []
#         for path in self.low_dim_paths:

#             file_names = os.listdir(path)
#             for idx, file_name in enumerate(file_names):
#                 temp_pro_for_norm, _, action_for_norm, _ = self.read_pickle_data(path, file_name)
#                 agent_pos_all.append(temp_pro_for_norm)
#                 actions_all.append(action_for_norm)
#         actions_all = torch.stack(actions_all, axis=0)
#         agent_pos_all = torch.stack(agent_pos_all, axis=0)
#         # print("BEFORE EVERYTHINGGGGGGGGGGG", actions_all.shape, agent_pos_all.shape)
#         # import pdb; pdb.set_trace();
#         # print("STAT ACTION POS")
#         stat_action_pos = array_to_stats(actions_all[:,:3].numpy())
#         # print("Array to Stats STAT ACTION POS", stat_action_pos)
#         # print("STAT QUATERNION ACTION")
#         stat_action_quat = array_to_stats(actions_all[:,3:7].numpy())
#         # print("Array to Stats STAT QUATERNION ACTION", stat_action_quat)
#         # import pdb; pdb.set_trace();
#         this_normalizer_pos = get_range_normalizer_from_stat(stat_action_pos)
#         this_normalizer_quat = get_identity_normalizer_from_stat(stat_action_quat)
#         # import pdb; pdb.set_trace();
#         this_normalizer = [this_normalizer_pos, this_normalizer_quat]
#         normalizer['action_pos'] = this_normalizer_pos
#         normalizer['action_quat'] = this_normalizer_quat

#         # print("STAT AGENT POS")
#         stat_action_pos_agent_pos = array_to_stats(agent_pos_all[:,:3].numpy())
#         # print("Array to Stats STAT AGENT POS", stat_action_pos_agent_pos)
#         # print("STAT AGENT QUARTERNION POS")
#         stat_action_quat_agent_pos = array_to_stats(agent_pos_all[:,3:7].numpy()) 
#         # print("Array to Stats STAT AGENT QUARTERNION POS", stat_action_quat_agent_pos)
#         this_normalizer_pos_agent_pos = get_range_normalizer_from_stat(stat_action_pos_agent_pos)
#         this_normalizer_quat_agent_pos = get_identity_normalizer_from_stat(stat_action_quat_agent_pos)
#         this_normalizer_agent_pos = [this_normalizer_pos_agent_pos, this_normalizer_quat_agent_pos]
#         normalizer['agent_pos_pos'] = this_normalizer_pos_agent_pos
#         normalizer['agent_pos_quat'] = this_normalizer_quat_agent_pos

#         normalizer["image"] = get_image_range_normalizer()
#         return normalizer
    
    
#     def __getitem__(self,idx):
#         # import pdb; pdb.set_trace();
#         print("idxxxxxx", idx)
#         buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx \
#             = self.indices[idx]
#         episode_idx = np.searchsorted(self.accumulated_episode_lengths, buffer_start_idx, side='right') # idx
#         start_idx = buffer_start_idx - self.accumulated_episode_lengths[episode_idx] # idx
#         if start_idx < 0:
#             start_idx += self.episode_lengths[episode_idx]
#         end_idx = start_idx + (sample_end_idx - sample_start_idx)
#         sample_proprioception = []
#         sample_image = []
#         sample_heatmap = []
#         sample_depth = []
#         goal_sample_proprioception = []
#         # import pdb; pdb.set_trace();
#         print(start_idx, end_idx)
#         for indices in range(start_idx, end_idx):
            
#             local_episode_files = os.listdir(self.low_dim_paths[episode_idx])
#             local_episode_files = sorted(local_episode_files, key=self.extract_number)
#             # import pdb; pdb.set_trace();
#             the_file_used = local_episode_files[indices]
#             temp_pro, _, action_temp_pro, step_path_CHECK = self.read_pickle_data(self.low_dim_paths[episode_idx], the_file_used)
#             goal_sample_proprioception.append(action_temp_pro)
#             sample_proprioception.append(temp_pro)
#             the_file_used = the_file_used.replace('.pkl', '.png')
#             # import pdb; pdb.set_trace()
#             this_episode_idx = os.path.basename(self.low_dim_paths[episode_idx])
#             temp_image, temp_heatmap, temp_depth , image_path_CHECK = self.read_images(self.image_path, self.heatmap_path, self.depth_path, this_episode_idx, the_file_used)
#             # print("PICKLE FILE", step_path_CHECK , "IMAGE FILE", image_path_CHECK)
#             sample_image.append(temp_image)
#             sample_heatmap.append(temp_heatmap)
#             sample_depth.append(temp_depth)
#         # import pdb; pdb.set_trace()
#         goal_sample_proprioception = torch.stack(goal_sample_proprioception, dim=0)
#         sample_proprioception = torch.stack(sample_proprioception, dim=0)
#         sample_image = torch.stack(sample_image, dim=0)
#         sample_heatmap = torch.stack(sample_heatmap, dim=0)
#         sample_depth = torch.stack(sample_depth, dim=0)
#         # import pdb; pdb.set_trace()
#         gripper_proprioception = torch.zeros(
#             (self.sequence_length,) + sample_proprioception.shape[1:], 
#             dtype=sample_proprioception.dtype, 
#             device=sample_proprioception.device
#         )
#         gripper_image = torch.zeros(
#             (self.sequence_length,) + sample_image.shape[1:], 
#             dtype=sample_proprioception.dtype, 
#             device=sample_proprioception.device
#         )
#         gripper_heatmap = torch.zeros(
#             (self.sequence_length,) + sample_heatmap.shape[1:], 
#             dtype=sample_heatmap.dtype, 
#             device=sample_heatmap.device
#         )
#         gripper_depth = torch.zeros(
#             (self.sequence_length,) + sample_depth.shape[1:], 
#             dtype=sample_depth.dtype, 
#             device=sample_depth.device
#         )
#         goal_gripper_proprioception = torch.zeros(
#             (self.sequence_length,) + goal_sample_proprioception.shape[1:], 
#             dtype=sample_proprioception.dtype, 
#             device=sample_proprioception.device
#         )
#         if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
#             if sample_start_idx > 0:
#                 # import pdb; pdb.set_trace()
#                 # data[:sample_start_idx] = #sample[0]
#                 gripper_proprioception[:sample_start_idx] = sample_proprioception[0]
#                 gripper_image[:sample_start_idx] = sample_image[0]
#                 gripper_heatmap[:sample_start_idx] = sample_heatmap[0]
#                 gripper_depth[:sample_start_idx] = sample_depth[0]
#                 goal_gripper_proprioception[:sample_start_idx] = goal_sample_proprioception[0]
#             if sample_end_idx < self.sequence_length:
#                 # import pdb; pdb.set_trace()
#                 # data[sample_end_idx:] = sample[-1]
#                 gripper_proprioception[sample_end_idx:] = sample_proprioception[-1]
#                 gripper_image[:sample_start_idx] = sample_image[-1]
#                 gripper_heatmap[:sample_start_idx] = sample_heatmap[-1]
#                 gripper_depth[:sample_start_idx] = sample_depth[-1]
#                 goal_gripper_proprioception[:sample_start_idx] = goal_sample_proprioception[-1]
#         # import pdb; pdb.set_trace()
#         gripper_proprioception[sample_start_idx:sample_end_idx] = sample_proprioception
#         gripper_image[sample_start_idx:sample_end_idx] = sample_image
#         gripper_heatmap[sample_start_idx:sample_end_idx] = sample_heatmap
#         gripper_depth[sample_start_idx:sample_end_idx] = sample_depth
#         goal_gripper_proprioception[sample_start_idx:sample_end_idx] = goal_sample_proprioception
#         # import pdb; pdb.set_trace();
#         # if gripper_image.shape[0] == 0 or gripper_heatmap.shape[0] == 0 or gripper_proprioception is None or goal_gripper_proprioception is None:
#             # print("NONEEEEEEEEE NOTEDDDDDDDDD")
#         for x, name in zip(
#         [gripper_image, gripper_heatmap, gripper_depth, gripper_proprioception, goal_gripper_proprioception],
#         ["gripper_image", "gripper_heatmap", "gripper_depth", "gripper_proprioception", "goal_gripper_proprioception"]
#         ):
#             if not isinstance(x, torch.Tensor):
#                 print(f"Bad type detected: {name} is {type(x)} at idx {idx}")
#         # print("IMAGEEEEEE", torch.cat((gripper_image, gripper_heatmap), dim = 1).shape)
#         # print("AGENTTTTTTTTTTT POSSSSSS", gripper_proprioception.shape)
#         # print("ACTIONNNNNNNNNNNNN", goal_gripper_proprioception.shape)
#         # import pdb; pdb.set_trace();
#         data = {
#             'obs': {
#                 'image': torch.cat((gripper_image, gripper_heatmap, gripper_depth), dim = 1), # T, 3, 96, 96
#                 'agent_pos': gripper_proprioception, # T, 2
#             },
#             'action': gripper_proprioception # T, 2 goal_gripper_proprioception
#         }
#         # import pdb; pdb.set_trace();
#         # import pdb; pdb.set_trace();
#         return data

#     # functions
#     def get_stored_demo(self, data_path, index):
#         episode_path = os.path.join(data_path, EPISODE_FOLDER % index)
        
#         # low dim pickle file
#         with open(os.path.join(episode_path, LOW_DIM_PICKLE), 'rb') as f:
#             obs = pickle.load(f)

            
#         return obs


# # image_path = "/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/RVT2_GMM_Codebase/RVT2_Codebase/RVT/outputs_7th/unnormalized_rgb" 
# # heatmap_path="/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/RVT2_GMM_Codebase/RVT2_Codebase/RVT/outputs_7th/unnormalized_heatmap_images/" 
# # low_dim_path="/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/RVT2_GMM_Codebase/RVT2_Codebase/RVT/outputs_7th/gripper_pose/"
# # depth_path = "/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/RVT2_GMM_Codebase/RVT2_Codebase/RVT/outputs_7th/depth/"
# # dataset = HighLevelHeatmapTrain(image_path=image_path, heatmap_path = heatmap_path, depth_path=depth_path, low_dim_path = low_dim_path, sequence_length = 16, pad_before = 4, pad_after = 4)
# # dataset[100]
# # print(dataset[2000])
# # for i in range(36500, 159874):
# #     dataset[i]
# #     print(i)



# from typing import Dict, List
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import re
# import numba
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
    def __init__(self, image_path, heatmap_path, low_dim_path, depth_path, sequence_length, pad_before, pad_after, use_wrist_camera, episode_mask = None, num_cameras = 4):
        super().__init__()
        self.use_wrist_camera = use_wrist_camera
        self.num_cameras = num_cameras
        self.low_dim_path = low_dim_path
        self.image_path = image_path
        self.heatmap_path = heatmap_path
        self.camera_image_paths = [os.path.join(self.image_path, f) for f in os.listdir(self.image_path)]
        self.camera_image_paths = sorted(self.camera_image_paths) #, key=self.extract_number

        self.depth_path = depth_path
        self.camera_depth_paths = [os.path.join(self.depth_path, f) for f in os.listdir(self.depth_path)]
        self.camera_depth_paths = sorted(self.camera_depth_paths) #, key=self.extract_number
        # import pdb; pdb.set_trace();
        self.sequence_length = sequence_length
        self.camera_heatmap_paths = [os.path.join(self.heatmap_path, f) for f in os.listdir(self.heatmap_path)]
        self.camera_heatmap_paths = sorted(self.camera_heatmap_paths) # , key=self.extract_number
        # import pdb; pdb.set_trace();
        self.low_dim_paths = [os.path.join(low_dim_path, f) for f in os.listdir(low_dim_path)]
        self.low_dim_paths = sorted(self.low_dim_paths) # , key=self.extract_number
        # import pdb; pdb.set_trace();
        self.episodes_images = []
        for camera_path in self.camera_image_paths:
            single_camera_episode_paths = [os.path.join(camera_path, f) for f in os.listdir(camera_path)]
            single_camera_episode_paths = sorted(single_camera_episode_paths) # key=self.extract_number
            # import pdb; pdb.set_trace();
            self.episodes_images.append(single_camera_episode_paths)
        # import pdb; pdb.set_trace();
        # self.episodes_images = os.listdir(self.camera_image_paths[0])
        # self.episodes_heatmaps = []
        # for camera_path in self.camera_heatmap_paths:
        #     single_camera_episode_paths = [os.path.join(camera_path, f) for f in os.listdir(camera_path)]
        #     single_camera_episode_paths = sorted(single_camera_episode_paths, key=self.extract_number)
        #     self.episodes_heatmaps.append(single_camera_episode_paths)
        # # self.episodes_heatmaps = os.listdir(self.camera_heatmap_paths[0])
        # assert len(self.episodes_images) == len(self.episodes_heatmaps)
        # for i in range(len(self.camera_heatmap_paths)):
        #     assert len(self.episodes_images[i]) == len(self.episodes_heatmaps[i])
        # import pdb; pdb.set_trace();
        # self.all_image_paths = []
        self.episode_lengths = []
        for i in range(len(self.camera_heatmap_paths)):
            # all_image_paths_single_camera = []
            for j, episode in enumerate(self.episodes_images[i]):
                # import pdb; pdb.set_trace();
                episode_images = [os.path.join(episode,f) for f in os.listdir(episode)]
                episode_images = sorted(episode_images, key=self.extract_number)
                # import pdb; pdb.set_trace();
                # import pdb; pdb.set_trace();
                # all_image_paths_single_camera.append(episode_images)
                # import pdb; pdb.set_trace();
                if j==0:
                    self.episode_lengths.append(len(episode_images))
            # self.all_image_paths.append(all_image_paths_single_camera)
        # import pdb; pdb.set_trace();
        # self.all_heatmap_paths = []
        # for i in range(len(self.camera_heatmap_paths)):
        #     all_heatmap_paths_single_camera = []
        #     for episode in self.episodes_heatmaps[i]:
        #         episode_heatmaps = [os.path.join(episode,f) for f in os.listdir(episode)]
        #         episode_heatmaps = sorted(episode_heatmaps, key=self.extract_number)
        #         all_heatmap_paths_single_camera.append(episode_heatmaps)
        #     self.all_heatmap_paths.append(all_heatmap_paths_single_camera)
        # for i in range(len(self.camera_heatmap_paths)):
        #     assert len(self.all_image_paths[i]) == len(self.all_heatmap_paths[i])
        self.accumulated_episode_lengths = np.cumsum(self.episode_lengths)
        # import pdb; pdb.set_trace();
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
        # import pdb; pdb.set_trace();

    # def extract_number(self,path):
    #     # Finds digits before .png at the end
    #     match = re.search(r'(\d+)\.png$', path)
    #     return int(match.group(1)) if match else -1
    def extract_number(self, path):
        # Match digits before .png, .pkl, or .npy at the end of the filename
        match = re.search(r'(\d+)\.(?:png|pkl|npy)$', path)
        return int(match.group(1)) if match else -1



    def __len__(self):
        return len(self.indices) #self.accumulated_episode_lengths[-1]

    def read_pickle_data(self, path, file_name):
        # print("READ PICKLE")
        # import pdb; pdb.set_trace();
        step_path = os.path.join(path, file_name) #os.path.join(self.low_dim_paths[episode_idx], str(step_idx) + '.pkl')
        with open(step_path, 'rb') as f:
            # import pdb; pdb.set_trace();
            data = pickle.load(f)

        ###### FOR CHECKING WITH ANY LOW DIM FILE !!!!! #########
        step_path_ACTION = os.path.join("/home/pratik_final/Downloads/Bimanual/Original_RVT2_Inference/RVT/outputs_9th/gripper_pose/episode0", file_name)
        with open(step_path_ACTION, "rb") as f:
            data_action = pickle.load(f)
        # import pdb; pdb.set_trace();
        gripper_pose = torch.from_numpy(data["gripper_pose"]).clone().detach().cpu().float() #torch.tensor(data["gripper_pose"].detach().cpu())
        gripper_open_close = torch.from_numpy(data["gripper_open_close"]).clone().detach().cpu().float() #torch.tensor(data["gripper_open_close"].detach().cpu())
        proprioceprion = torch.cat((gripper_pose, gripper_open_close), dim = -1).squeeze(0)
        gripper_action = data_action["gripper_action"].clone().detach().cpu().squeeze(0).float() #  gripper_action # torch.tensor(data["gripper_action"].detach().cpu()).squeeze(0)
        # REPLACING PROPRIO WITH GRIPPER ACTION JUST FOR ROLLOUT CHECKING NOT FOR TRAINING.
        return proprioceprion, gripper_open_close, gripper_action, step_path # gripper_action proprioceprion

    

    def read_images(self, image_path, heatmap_path, depth_path, episode_idx, file_name): #step_idx
        images_step_idx = []
        heatmap_step_idx = []
        depth_step_idx = []
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.5, 0.5, 0.5],
        #                         std=[0.5, 0.5, 0.5])
        ])
        
        
        episode_path = os.path.join(image_path, episode_idx)
        # print("EPISODE_PATH", episode_path)
        if self.use_wrist_camera:
            total_cams = self.num_cameras + 1
        else:
            total_cams = self.num_cameras
        for i in range(1, total_cams): # self.all_image_paths[episode_idx] self.num_cameras
            # image_path = self.all_image_paths[episode_idx][i][step_idx]
            # import pdb; pdb.set_trace();
            camera_path = os.path.join(episode_path, "camera" + str(i))
            image_path = os.path.join(camera_path, file_name)
            # print("EPISODE INDEX", episode_idx, "image_path", image_path)
            # print("THE IMAGE INSIDE CAMERA", i, "IS", step_idx, "PATH", image_path)
            image = Image.open(image_path).convert('RGB')
            image = transform(image)
            images_step_idx.append(image)
        # import pdb; pdb.set_trace();
        images_step_idx = torch.cat(images_step_idx, dim=0)

        heatmap_episode_path = os.path.join(heatmap_path, episode_idx)
        for i in range(1, self.num_cameras):
            heatmap_camera_path = os.path.join(heatmap_episode_path, "camera" + str(i))
            # file_name = file_name.replace('.png', '.npy')
            heatmap_path = os.path.join(heatmap_camera_path, file_name)
            # import pdb; pdb.set_trace();
            heatmap = Image.open(heatmap_path).convert('RGB')
            # heatmap = np.load(heatmap_path)
            heatmap = transform(heatmap)
            heatmap = heatmap.mean(axis=0).unsqueeze(0)
            heatmap_step_idx.append(heatmap)
        heatmap_step_idx = torch.cat(heatmap_step_idx, dim=0)

        if self.use_wrist_camera:
            total_cams = self.num_cameras + 1
        else:
            total_cams = self.num_cameras
        depth_episode_path = os.path.join(depth_path, episode_idx)
        for i in range(1, total_cams):
            depth_camera_path = os.path.join(depth_episode_path, "camera" + str(i))
            # file_name = file_name.replace('.png', '.npy')
            depth_path = os.path.join(depth_camera_path, file_name)
            # import pdb; pdb.set_trace();
            depth = Image.open(depth_path).convert('RGB')
            # heatmap = np.load(heatmap_path)
            depth = transform(depth)
            depth = depth.mean(axis=0).unsqueeze(0)
            depth_step_idx.append(depth)
        depth_step_idx = torch.cat(depth_step_idx, dim=0)



        # import pdb; pdb.set_trace();
        return images_step_idx, heatmap_step_idx, depth_step_idx, image_path
    
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        
        actions_all = []
        agent_pos_all = []
        for path in self.low_dim_paths:

            file_names = os.listdir(path)
            for idx, file_name in enumerate(file_names):
                temp_pro_for_norm, _, action_for_norm, _ = self.read_pickle_data(path, file_name)
                agent_pos_all.append(temp_pro_for_norm)
                actions_all.append(action_for_norm)
        actions_all = torch.stack(actions_all, axis=0)
        agent_pos_all = torch.stack(agent_pos_all, axis=0)
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

        normalizer["image"] = get_image_range_normalizer()
        return normalizer
    
    
    def __getitem__(self,idx):
        # import pdb; pdb.set_trace();
        # print("idxxxxxx", idx)
        # import pdb; pdb.set_trace()
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
        for indices in range(start_idx, end_idx):
            # print(indices)
            # import pdb; pdb.set_trace()
            local_episode_files = os.listdir(self.low_dim_paths[episode_idx])
            local_episode_files = sorted(local_episode_files, key=self.extract_number)
            # import pdb; pdb.set_trace();
            the_file_used = local_episode_files[indices]
            temp_pro, _, action_temp_pro, step_path_CHECK = self.read_pickle_data(self.low_dim_paths[episode_idx], the_file_used)
            goal_sample_proprioception.append(action_temp_pro)
            sample_proprioception.append(temp_pro)
            the_file_used = the_file_used.replace('.pkl', '.png')
            # import pdb; pdb.set_trace()
            this_episode_idx = os.path.basename(self.low_dim_paths[episode_idx])
            temp_image, temp_heatmap, temp_depth , image_path_CHECK = self.read_images(self.image_path, self.heatmap_path, self.depth_path, this_episode_idx, the_file_used)
            # print("PICKLE FILE", step_path_CHECK , "IMAGE FILE", image_path_CHECK)
            sample_image.append(temp_image)
            sample_heatmap.append(temp_heatmap)
            sample_depth.append(temp_depth)
        # import pdb; pdb.set_trace()
        goal_sample_proprioception = torch.stack(goal_sample_proprioception, dim=0)
        sample_proprioception = torch.stack(sample_proprioception, dim=0)
        sample_image = torch.stack(sample_image, dim=0)
        sample_heatmap = torch.stack(sample_heatmap, dim=0)
        sample_depth = torch.stack(sample_depth, dim=0)
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
        goal_gripper_proprioception = torch.zeros(
            (self.sequence_length,) + goal_sample_proprioception.shape[1:], 
            dtype=sample_proprioception.dtype, 
            device=sample_proprioception.device
        )
        if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
            if sample_start_idx > 0:
                # import pdb; pdb.set_trace()
                # data[:sample_start_idx] = #sample[0]
                gripper_proprioception[:sample_start_idx] = sample_proprioception[0]
                gripper_image[:sample_start_idx] = sample_image[0]
                gripper_heatmap[:sample_start_idx] = sample_heatmap[0]
                gripper_depth[:sample_start_idx] = sample_depth[0]
                goal_gripper_proprioception[:sample_start_idx] = goal_sample_proprioception[0]
            if sample_end_idx < self.sequence_length:
                # import pdb; pdb.set_trace()
                # data[sample_end_idx:] = sample[-1]
                gripper_proprioception[sample_end_idx:] = sample_proprioception[-1]
                gripper_image[:sample_start_idx] = sample_image[-1]
                gripper_heatmap[:sample_start_idx] = sample_heatmap[-1]
                gripper_depth[:sample_start_idx] = sample_depth[-1]
                goal_gripper_proprioception[:sample_start_idx] = goal_sample_proprioception[-1]
        # import pdb; pdb.set_trace()
        gripper_proprioception[sample_start_idx:sample_end_idx] = sample_proprioception
        gripper_image[sample_start_idx:sample_end_idx] = sample_image
        gripper_heatmap[sample_start_idx:sample_end_idx] = sample_heatmap
        gripper_depth[sample_start_idx:sample_end_idx] = sample_depth
        goal_gripper_proprioception[sample_start_idx:sample_end_idx] = goal_sample_proprioception
        # import pdb; pdb.set_trace();
        # if gripper_image.shape[0] == 0 or gripper_heatmap.shape[0] == 0 or gripper_proprioception is None or goal_gripper_proprioception is None:
            # print("NONEEEEEEEEE NOTEDDDDDDDDD")
        for x, name in zip(
        [gripper_image, gripper_heatmap, gripper_depth, gripper_proprioception, goal_gripper_proprioception],
        ["gripper_image", "gripper_heatmap", "gripper_depth", "gripper_proprioception", "goal_gripper_proprioception"]
        ):
            if not isinstance(x, torch.Tensor):
                print(f"Bad type detected: {name} is {type(x)} at idx {idx}")
        # print("IMAGEEEEEE", torch.cat((gripper_image, gripper_heatmap), dim = 1).shape)
        # print("AGENTTTTTTTTTTT POSSSSSS", gripper_proprioception.shape)
        # print("ACTIONNNNNNNNNNNNN", goal_gripper_proprioception.shape)
        # import pdb; pdb.set_trace();
        data = {
            'obs': {
                'image': torch.cat((gripper_image, gripper_heatmap, gripper_depth), dim = 1), # T, 3, 96, 96
                'agent_pos': gripper_proprioception, # T, 2
            },
            'action': gripper_proprioception # T, 2 goal_gripper_proprioception
        }
        # import pdb; pdb.set_trace();
        # import pdb; pdb.set_trace();
        return data

    # functions
    def get_stored_demo(self, data_path, index):
        episode_path = os.path.join(data_path, EPISODE_FOLDER % index)
        
        # low dim pickle file
        with open(os.path.join(episode_path, LOW_DIM_PICKLE), 'rb') as f:
            obs = pickle.load(f)

            
        return obs