import torch
import torch.nn as nn
from diffusion_policy_3d.model.vision.layers import RelativeCrossAttentionModule
from diffusion_policy_3d.common.network_helper import replace_bn_with_gn
from diffusion_policy_3d.model.vision.position_encodings import RotaryPositionEncoding3D
from diffusion_policy_3d.model.vision.pointnet_extractor import create_mlp
import segmentation_models_pytorch as smp
from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint
import einops
import copy

class Act3dEncoder(nn.Module):
    def __init__(self, 
                #  in_channels=3, 
                 in_channels=5, 
                 encoder_output_dim=256, 
                 num_gripper_points=4, 
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 observation_space=None,
                 goal_mode=None,
                 mode=None,
                 **kwargs
                 ):
        super(Act3dEncoder, self).__init__()
        
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.feature_map_key = 'feature_map'
        self.gripper_pcd_key = 'gripper_pcd'
        self.num_gripper_points = num_gripper_points
        self.encoder_output_dim = encoder_output_dim
        self.state_shape = observation_space[self.state_key]
        self.goal_mode = goal_mode
        
        self.mode = mode
        if self.mode == 'keep_position_feature_in_attention_feature':
            vision_output_dim = encoder_output_dim // 3 * 2
        else:
            vision_output_dim = encoder_output_dim
        
        vision_encoder = smp.Unet(
            encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=vision_output_dim,                      # model output channels (number of classes in your dataset)
        )
        attn_layers = RelativeCrossAttentionModule(encoder_output_dim, 4, 2)
        attn_layers = replace_bn_with_gn(attn_layers)
        vision_encoder = replace_bn_with_gn(vision_encoder)
        self.nets = nn.ModuleDict({
            'vision_encoder': vision_encoder,
            'relative_pe_layer': RotaryPositionEncoding3D(encoder_output_dim),
            'attn_layers': attn_layers,
        })
        
        if self.mode == 'keep_position_feature_in_attention_feature':
            input_dim = 6 if self.goal_mode == 'cross_attention_to_goal' else 3
            position_embedding_mlp = nn.Sequential(
                nn.Linear(input_dim, 128), nn.ReLU(),
                nn.Linear(128, 256), nn.ReLU(),
                nn.Linear(256, encoder_output_dim // 3),
            )
            object_pcd_position_embedding_mlp = nn.Sequential(
                nn.Linear(3, 128), nn.ReLU(),
                nn.Linear(128, 256), nn.ReLU(),
                nn.Linear(256, encoder_output_dim // 3),
            )
            self.nets['object_pcd_position_embedding_mlp'] = object_pcd_position_embedding_mlp
            self.nets['gripper_pcd_position_embedding_mlp'] = position_embedding_mlp
            self.nets['embed'] = nn.Embedding(1, encoder_output_dim // 3 * 2)
        else:
            self.nets['embed'] = nn.Embedding(1, encoder_output_dim)

        if self.goal_mode == 'cross_attention_to_goal':
            goal_attn_layers = RelativeCrossAttentionModule(encoder_output_dim, 4, 2)
            goal_attn_layers = replace_bn_with_gn(goal_attn_layers)
            self.nets['goal_attn_layers'] = goal_attn_layers
            if self.mode == 'keep_position_feature_in_attention_feature':
                self.nets['goal_pcd_position_embedding_mlp'] = copy.deepcopy(position_embedding_mlp)
                self.nets['goal_embed'] = nn.Embedding(1, encoder_output_dim // 3 * 2)
            else:
                self.nets['goal_embed'] = nn.Embedding(1, encoder_output_dim)
        
        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels = encoder_output_dim * self.num_gripper_points
        self.n_output_channels += output_dim
        if self.goal_mode == 'cross_attention_to_goal':
            self.n_output_channels += encoder_output_dim * self.num_gripper_points
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

    def forward(self, observation: Dict) -> torch.Tensor:
        # TODO: the things passed in is already flattend from B, T, ... -> B*T, ...
        
        nets = self.nets
        
        # TODO: check the input shape
        agent_pos = observation[self.state_key]
        B = agent_pos.shape[0] #  B = batch_size * obs_horizon

        # NOTE: rgb_obs should actually be segmentation mask + depth, or segmentation mask + point position
        rgb_obs = observation[self.feature_map_key]
        B, n_cam, h, w, c = rgb_obs.shape
        rgb_obs = einops.rearrange(rgb_obs, "B n h w c -> B n c h w") # NOTE: our rgb comes in as B n_camera H W C
        rgb_obs = einops.rearrange(rgb_obs, "B n c h w -> (B n) c h w") # NOTE: our rgb comes in as B n_camera H W C
        rgb_features = nets['vision_encoder'](rgb_obs)
        rgb_features = einops.rearrange(rgb_features, "(B n_cam) c h w -> (n_cam h w) B c", n_cam=n_cam) # shape N=image_size B encoder_output_dim

        
        # NOTE: extract rgb features corresponding to the fpsed points
        pcd_mask = observation['pcd_mask'] # B * (n * h * w)
        pcd_mask = einops.rearrange(pcd_mask, "B N -> N B")
        vision_output_dim = rgb_features.shape[-1]
        rgb_features = rgb_features[pcd_mask == 1].reshape(-1, B, vision_output_dim) # shape (num_points, B, encoder_output_dim)
        if self.mode == 'keep_position_feature_in_attention_feature':
            obj_pcd = observation[self.point_cloud_key]
            _, n_obj, _ = obj_pcd.shape
            obj_pcd = einops.rearrange(obj_pcd, "B N c -> (B N) c", B=B, N=n_obj)
            obj_pcd_position_embedding = nets['object_pcd_position_embedding_mlp'](obj_pcd) # shape B*N encoder_output_dim // 3
            obj_pcd_position_embedding = einops.rearrange(obj_pcd_position_embedding, "(B N) encoder_output_dim -> N B encoder_output_dim", B=B, N=n_obj)
            rgb_features = torch.cat([rgb_features, obj_pcd_position_embedding], dim=-1)
        
            
        point_cloud = observation[self.point_cloud_key]
        point_cloud_rel_pos_embedding = nets['relative_pe_layer'](point_cloud) # shape B N encoder_output_dim
                       
        num_gripper_points = observation['gripper_pcd'].shape[1] # gripper pcd is B num_gripper_points 3
        assert num_gripper_points == self.num_gripper_points, f"Expected {self.num_gripper_points} gripper points, got {num_gripper_points}"
        gripper_pcd = observation[self.gripper_pcd_key]
        gripper_pcd_rel_pos_embedding = nets['relative_pe_layer'](gripper_pcd) # shape B num_gripper_points encoder_output_dim
        gripper_pcd_features = nets['embed'].weight.unsqueeze(0).repeat(num_gripper_points, B, 1) # shape (num_gripper_points, B, encoder_output_dim)
        if self.mode == 'keep_position_feature_in_attention_feature':
            if self.goal_mode == 'cross_attention_to_goal':
                displacement_to_goal = observation['goal_gripper_pcd'] - observation['gripper_pcd']
                input_to_position_embedding = torch.cat([gripper_pcd, displacement_to_goal], dim=-1)
                input_to_position_embedding = einops.rearrange(input_to_position_embedding, "B num_gripper_points c -> (B num_gripper_points) c", B=B, num_gripper_points=self.num_gripper_points)
                gripper_pcd_position_embedding = nets['gripper_pcd_position_embedding_mlp'](input_to_position_embedding)
            else:
                gripper_pcd_position = einops.rearrange(gripper_pcd, "B num_gripper_points c -> (B num_gripper_points) c", B=B, num_gripper_points=self.num_gripper_points)
                gripper_pcd_position_embedding = nets['gripper_pcd_position_embedding_mlp'](gripper_pcd_position) # shape B*num_gripper_points encoder_output_dim//3

            gripper_pcd_position_embedding = einops.rearrange(gripper_pcd_position_embedding, "(B num_gripper_points) encoder_output_dim -> num_gripper_points B encoder_output_dim", B=B, num_gripper_points=self.num_gripper_points)
            gripper_pcd_features = torch.cat([gripper_pcd_features, gripper_pcd_position_embedding], dim=-1)

        attn_output = nets['attn_layers'](
            query=gripper_pcd_features, value=rgb_features,
            query_pos=gripper_pcd_rel_pos_embedding, value_pos=point_cloud_rel_pos_embedding,
        )[-1]
        
        rgb_features = einops.rearrange(
            attn_output, "num_gripper_points B embed_dim -> B num_gripper_points embed_dim").flatten(start_dim=1) # shape B (num_gripper_points * encoder_output_dim)

        if self.goal_mode == 'cross_attention_to_goal':
            # print("Cross attention to goal")
            goal_gripper_pcd_rel_pos_embedding = nets['relative_pe_layer'](observation['goal_gripper_pcd']) # shape B num_gripper_points encoder_output_dim
            goal_gripper_pcd_features = nets['goal_embed'].weight.unsqueeze(0).repeat(num_gripper_points, B, 1) # shape (num_gripper_points, B, encoder_output_dim)
            if self.mode == 'keep_position_feature_in_attention_feature':
                displacement_to_goal = observation['goal_gripper_pcd'] - observation['gripper_pcd']
                input_to_position_embedding = torch.cat([observation['goal_gripper_pcd'], displacement_to_goal], dim=-1)
                goal_gripper_pcd_position = einops.rearrange(input_to_position_embedding, "B num_gripper_points c -> (B num_gripper_points) c", B=B, num_gripper_points=self.num_gripper_points)
                goal_gripper_pcd_position_embedding = nets['goal_pcd_position_embedding_mlp'](goal_gripper_pcd_position)
                goal_gripper_pcd_position_embedding = einops.rearrange(goal_gripper_pcd_position_embedding, "(B num_gripper_points) encoder_output_dim -> num_gripper_points B encoder_output_dim", B=B, num_gripper_points=self.num_gripper_points)
                goal_gripper_pcd_features = torch.cat([goal_gripper_pcd_features, goal_gripper_pcd_position_embedding], dim=-1)
            
            goal_attn_output = nets['goal_attn_layers'](query=gripper_pcd_features, value=goal_gripper_pcd_features,
                query_pos=gripper_pcd_rel_pos_embedding, value_pos=goal_gripper_pcd_rel_pos_embedding,
            )[-1]
            
            goal_features = einops.rearrange(
                goal_attn_output, "num_gripper_points B embed_dim -> B num_gripper_points embed_dim").flatten(start_dim=1)
        
        
        state_feat = self.state_mlp(agent_pos)  # B * 64
        
        obs_features = torch.cat([rgb_features, state_feat], dim=-1)
        if self.goal_mode == 'cross_attention_to_goal':
            obs_features = torch.cat([obs_features, goal_features], dim=-1)
        return obs_features
    
    def output_shape(self):
        return self.n_output_channels

        