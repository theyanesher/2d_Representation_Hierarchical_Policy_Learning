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

class Act3dEncoder(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 encoder_output_dim=256, 
                 num_gripper_points=4, 
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 observation_space=None,
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
        
        vision_encoder = smp.Unet(
            encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=encoder_output_dim,                      # model output channels (number of classes in your dataset)
        )
        attn_layers = RelativeCrossAttentionModule(encoder_output_dim, 4, 2)
        attn_layers = replace_bn_with_gn(attn_layers)
        vision_encoder = replace_bn_with_gn(vision_encoder)

        self.nets = nn.ModuleDict({
            'vision_encoder': vision_encoder,
            'relative_pe_layer': RotaryPositionEncoding3D(encoder_output_dim),
            'embed': nn.Embedding(1, encoder_output_dim),
            'attn_layers': attn_layers,
        })
        
        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels = encoder_output_dim * self.num_gripper_points
        self.n_output_channels += output_dim
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

    def forward(self, observation: Dict) -> torch.Tensor:
        # TODO: the things passed in is already flattend from B, T, ... -> B*T, ...
        
        nets = self.nets
        
        # TODO: check the input shape
        agent_pos = observation[self.state_key]
        B = agent_pos.shape[0] #  B = batch_size * obs_horizon

        # NOTE: rgb_obs should actually be segmentation mask + depth, or segmentation mask + point position
        rgb_obs = observation[self.feature_map_key]
        rgb_obs = einops.rearrange(rgb_obs, "B h w c -> B c h w") # NOTE: our rgb comes in as B H W C
        rgb_features = nets['vision_encoder'](rgb_obs)
        rgb_features = einops.rearrange(rgb_features, "B c h w -> (h w) B c") # shape N=image_size B encoder_output_dim
            
        point_cloud = observation[self.point_cloud_key]
        # TODO: this can be done when retrieving the point cloud
        # point_cloud = einops.rearrange(point_cloud, "B c h w -> B (h w) c", B=B) # NOTE: our pcd comes in as B N 3, where N = h*w is the image size
        point_cloud_rel_pos_embedding = nets['relative_pe_layer'](point_cloud) # shape B N encoder_output_dim
                       
        num_gripper_points = observation['gripper_pcd'].shape[1] # gripper pcd is B num_gripper_points 3
        assert num_gripper_points == self.num_gripper_points, f"Expected {self.num_gripper_points} gripper points, got {num_gripper_points}"
        gripper_pcd = observation[self.gripper_pcd_key]
        gripper_pcd_rel_pos_embedding = nets['relative_pe_layer'](gripper_pcd) # shape B num_gripper_points encoder_output_dim
        gripper_pcd_features = nets['embed'].weight.unsqueeze(0).repeat(num_gripper_points, B, 1) # shape (num_gripper_points, B, encoder_output_dim)

        # TODO: we can further modify this such that the gripper only attends to the object point cloud
        attn_output = nets['attn_layers'](
            query=gripper_pcd_features, value=rgb_features,
            query_pos=gripper_pcd_rel_pos_embedding, value_pos=point_cloud_rel_pos_embedding,
        )[-1]
        
        rgb_features = einops.rearrange(
            attn_output, "num_gripper_points B embed_dim -> B num_gripper_points embed_dim").flatten(start_dim=1) # shape B (num_gripper_points * encoder_output_dim)

        state_feat = self.state_mlp(agent_pos)  # B * 64
        print('rgb_features ', rgb_features.shape)
        print('agent_pos ', state_feat.shape)
        
        obs_features = torch.cat([rgb_features, state_feat], dim=-1)
        return obs_features
    
    def output_shape(self):
        return self.n_output_channels

        