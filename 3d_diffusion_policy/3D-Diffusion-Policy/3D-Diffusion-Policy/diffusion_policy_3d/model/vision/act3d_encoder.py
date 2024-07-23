import torch
import torch.nn as nn
from diffusion_policy_3d.model.vision.layers import RelativeCrossAttentionModule
from diffusion_policy_3d.common.network_helper import replace_bn_with_gn
from diffusion_policy_3d.model.vision.position_encodings import RotaryPositionEncoding3D
from diffusion_policy_3d.model.vision.pointnet_extractor import create_mlp
import segmentation_models_pytorch as smp
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.models import mobilenet_v3_small
from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint
import einops
import copy

class ModifiedMobileNetV3Small(nn.Module):
    def __init__(self, in_channels=5, out_channels=60):
        super(ModifiedMobileNetV3Small, self).__init__()
        self.mobilenet_v3_small = mobilenet_v3_small(pretrained=True)

        #######################################
        # Modify the input feature dimensions #
        #######################################

        # Create a new convolutional layer with "in_channels" input channels and the same number of output channels
        orig_conv = self.mobilenet_v3_small.features[0][0]
        new_conv = nn.Conv2d(in_channels=in_channels, out_channels=orig_conv.out_channels,
                            kernel_size=orig_conv.kernel_size, stride=orig_conv.stride,
                            padding=orig_conv.padding, bias=orig_conv.bias is not None)

        # Copy the weights from the original convolutional layer for the first 3 channels
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = orig_conv.weight
            if new_conv.weight.size(1) > 3:
                nn.init.kaiming_normal_(new_conv.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')

        # Replace the original convolutional layer with the new one
        self.mobilenet_v3_small.features[0][0] = new_conv

        ##############################################################
        # Modify the classification head to per-pixel classification #
        ##############################################################

        self.mobilenet_v3_small.classifier = nn.Identity()  # Remove the classifier
        # Add a new convolutional layer to adjust the number of channels if needed
        self.conv = nn.Conv2d(576, out_channels, kernel_size=1)  # Example: 576 to output_dim channels
        # Add an upsample layer to ensure the output is 256x256
        self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.mobilenet_v3_small(x)
        x = self.conv(x)
        x = self.upsample(x)
        return x

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
                 use_mlp=False,
                 use_lightweight_unet=False,
                 self_attention=False,
                 final_attention=False,
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

        # [Chialiang]
        self.use_mlp = use_mlp
        self.use_lightweight_unet = use_lightweight_unet
        
        self.self_attention = self_attention
        self.final_attention = final_attention
        self.mode = mode
        if self.mode in ['keep_position_feature_in_attention_feature']:
            vision_output_dim = encoder_output_dim // 3 * 2
        else:
            vision_output_dim = encoder_output_dim
        
        # [Chialiang]
        vision_encoder = None
        if self.use_mlp:
            hidden_layer_dim = encoder_output_dim
            vision_encoder = nn.Sequential(
                nn.Linear(in_channels, hidden_layer_dim),
                nn.ReLU(),
                nn.Linear(hidden_layer_dim, hidden_layer_dim),
                nn.ReLU(),
                nn.Linear(hidden_layer_dim, encoder_output_dim)
            )
        elif self.use_lightweight_unet:

            cprint('lightweight UNet', 'green')

            vision_encoder = ModifiedMobileNetV3Small(in_channels=in_channels, out_channels=vision_output_dim)
            cprint(vision_encoder, 'green')

            # ###################################################################################################################

            # vision_encoder = deeplabv3_mobilenet_v3_large(
            #     progress=True, # (bool, optional) – If True, displays a progress bar of the download to stderr. Default is True.
            #     num_classes=vision_output_dim, # (int, optional) – number of output classes of the model (including the background)
            # )

            # # Get the original first convolutional layer
            # orig_conv = vision_encoder.backbone['0'][0]

            # # Create a new convolutional layer with 5 input channels and the same number of output channels
            # new_conv = nn.Conv2d(in_channels=in_channels, out_channels=orig_conv.out_channels,
            #                     kernel_size=orig_conv.kernel_size, stride=orig_conv.stride,
            #                     padding=orig_conv.padding, bias=orig_conv.bias is not None)

            # # Copy the weights from the original convolutional layer for the first 3 channels
            # with torch.no_grad():
            #     new_conv.weight[:, :3, :, :] = orig_conv.weight
            #     if new_conv.weight.size(1) > 3:
            #         nn.init.kaiming_normal_(new_conv.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')

            # # Replace the original convolutional layer with the new one
            # vision_encoder.backbone['0'][0] = new_conv
            # cprint(vision_encoder, 'green')

        else :

            vision_encoder = smp.Unet(
                encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=vision_output_dim,                      # model output channels (number of classes in your dataset)
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)

        # attn_layers = RelativeCrossAttentionModule(encoder_output_dim, 4, 2)  # [Debug] make it deeper
        attn_layers = RelativeCrossAttentionModule(encoder_output_dim, 4, 4) 
        attn_layers = replace_bn_with_gn(attn_layers)
        self.nets = nn.ModuleDict({
            'vision_encoder': vision_encoder,
            'relative_pe_layer': RotaryPositionEncoding3D(encoder_output_dim),
            'attn_layers': attn_layers,
        })
        
        if self.self_attention:
            # self.nets['self_attn_layers'] = RelativeCrossAttentionModule(encoder_output_dim, 4, 2) # [Debug] make it deeper
            self.nets['self_attn_layers'] = RelativeCrossAttentionModule(encoder_output_dim, 4, 4) 
            self.nets['self_attn_layers'] = replace_bn_with_gn(self.nets['self_attn_layers'])
        
        if self.mode in ['keep_position_feature_in_attention_feature', 
                         "keep_position_feature_in_attention_feature_with_gripper_displacement_to_closest_object"]:
            input_dim = 3
            if self.goal_mode is not None:
                input_dim += 3
            if self.mode == "keep_position_feature_in_attention_feature_with_gripper_displacement_to_closest_object":
                input_dim += 3
                
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

        # NOTE: 
        # cross attention to goal means:
        # cross attention between current gripper and object -> vec1
        # cross attention between current gripper and target gripper -> vec2
        # concatenate vectors -> diffusion -> output
        
        # cross attention to goal not concat and self attention means:
        # cross attention between current gripper and object
        # cross attention between current gripper and target gripper, keep using the feature obtained from the first cross attention
        # current gripper self-attention

        if self.goal_mode in ['cross_attention_to_goal', 'cross_attention_to_goal_not_concat_and_self_attention']:
            # goal_attn_layers = RelativeCrossAttentionModule(encoder_output_dim, 4, 2)  # [Debug] make it deeper
            goal_attn_layers = RelativeCrossAttentionModule(encoder_output_dim, 4, 4)  
            goal_attn_layers = replace_bn_with_gn(goal_attn_layers)
            self.nets['goal_attn_layers'] = goal_attn_layers
            if self.mode in ['keep_position_feature_in_attention_feature', "keep_position_feature_in_attention_feature_with_gripper_displacement_to_closest_object"]:
                self.nets['goal_pcd_position_embedding_mlp'] = copy.deepcopy(position_embedding_mlp)
                self.nets['goal_embed'] = nn.Embedding(1, encoder_output_dim // 3 * 2)
            else:
                self.nets['goal_embed'] = nn.Embedding(1, encoder_output_dim)
            if self.goal_mode == 'cross_attention_to_goal_not_concat_and_self_attention' or self.self_attention:
                # self.nets['goal_self_attn_layers'] = RelativeCrossAttentionModule(encoder_output_dim, 4, 2)   # [Debug] make it deeper
                self.nets['goal_self_attn_layers'] = RelativeCrossAttentionModule(encoder_output_dim, 4, 4)
                self.nets['goal_self_attn_layers'] = replace_bn_with_gn(self.nets['goal_self_attn_layers'])

            if self.final_attention:
                self.nets['final_attn_layers'] = RelativeCrossAttentionModule(encoder_output_dim, 4, 4)
                self.nets['final_attn_layers'] = replace_bn_with_gn(self.nets['final_attn_layers'])
                self.nets['final_slef_attn_layers'] = RelativeCrossAttentionModule(encoder_output_dim, 4, 4)
                self.nets['final_slef_attn_layers'] = replace_bn_with_gn(self.nets['final_slef_attn_layers'])
        
        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels = encoder_output_dim * self.num_gripper_points
        self.n_output_channels += output_dim
        if self.goal_mode == 'cross_attention_to_goal' and not self.final_attention: # [Debug] [Chialiang] 
            self.n_output_channels += encoder_output_dim * self.num_gripper_points
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

    def forward(self, observation: Dict) -> torch.Tensor:
        # NOTE: the things passed in is already flattend from B, T, ... -> B*T, ...
        
        nets = self.nets
        
        agent_pos = observation[self.state_key]
        B = agent_pos.shape[0] #  B = batch_size * obs_horizon

        if self.use_mlp:
            rgb_obs_feat = observation[self.point_cloud_key]
            B, N, C = rgb_obs_feat.shape
            rgb_obs_flatten = rgb_obs_feat.reshape(-1, C)
            rgb_features_flatten = nets['vision_encoder'](rgb_obs_flatten)
            rgb_features = rgb_features_flatten.reshape(B, N, -1) # shape B N encoder_output_dim
            rgb_features = einops.rearrange(rgb_features, "B N encoder_output_dim -> N B encoder_output_dim") # shape N B encoder_output_dim
            point_cloud = observation[self.point_cloud_key]
        else:
            # NOTE: rgb_obs should actually be segmentation mask + depth, or segmentation mask + point position
            rgb_obs = observation[self.feature_map_key]
            B, n_cam, h, w, c = rgb_obs.shape
            rgb_obs = einops.rearrange(rgb_obs, "B n h w c -> B n c h w") # NOTE: our rgb comes in as B n_camera H W C
            rgb_obs = einops.rearrange(rgb_obs, "B n c h w -> (B n) c h w") # NOTE: our rgb comes in as B n_camera H W C
            rgb_features = nets['vision_encoder'](rgb_obs)

            # if self.use_lightweight_unet:
            #     rgb_features = rgb_features['out']

            cprint(rgb_features.shape, 'green')

            rgb_features = einops.rearrange(rgb_features, "(B n_cam) c h w -> (n_cam h w) B c", n_cam=n_cam) # shape N=image_size B encoder_output_dim
        
            # NOTE: extract rgb features corresponding to the fpsed points
            pcd_mask = observation['pcd_mask'] # B * (n * h * w)
            pcd_mask = einops.rearrange(pcd_mask, "B N -> N B")
            vision_output_dim = rgb_features.shape[-1]
            rgb_features = rgb_features[pcd_mask == 1].reshape(-1, B, vision_output_dim) # shape (num_points, B, encoder_output_dim)
            if self.mode in ['keep_position_feature_in_attention_feature']:
                obj_pcd = observation[self.point_cloud_key]
                _, n_obj, _ = obj_pcd.shape
                obj_pcd = einops.rearrange(obj_pcd, "B N c -> (B N) c", B=B, N=n_obj)
                obj_pcd_position_embedding = nets['object_pcd_position_embedding_mlp'](obj_pcd) # shape B*N encoder_output_dim // 3
                obj_pcd_position_embedding = einops.rearrange(obj_pcd_position_embedding, "(B N) encoder_output_dim -> N B encoder_output_dim", B=B, N=n_obj)
                rgb_features = torch.cat([rgb_features, obj_pcd_position_embedding], dim=-1)
        
            # BUG: before 2024/7/1, the order of the point cloud stored in observation['point_cloud'] does not match the order of the point cloud stored in osbervation['feature_map'] and then sampled using "pcd_mask".
            # for consistentcy should just use the positiones from feature map and then downsampled using pcd_mask.
            # point_cloud = observation[self.point_cloud_key] 
            all_point_positions = observation[self.feature_map_key][:, :, :, :, 2:5]
            all_point_positions = einops.rearrange(all_point_positions, "B n h w c -> B (n h w) c")
            object_point_positions = all_point_positions[observation['pcd_mask'] == 1] # shape B num_points 3
            point_cloud = object_point_positions.reshape(B, -1, 3) # shape (B, num_points, 3)

        point_cloud_rel_pos_embedding = nets['relative_pe_layer'](point_cloud) # shape B N encoder_output_dim
                       
        num_gripper_points = observation['gripper_pcd'].shape[1] # gripper pcd is B num_gripper_points 3
        assert num_gripper_points == self.num_gripper_points, f"Expected {self.num_gripper_points} gripper points, got {num_gripper_points}"
        gripper_pcd = observation[self.gripper_pcd_key]
        gripper_pcd_rel_pos_embedding = nets['relative_pe_layer'](gripper_pcd) # shape B num_gripper_points encoder_output_dim
        gripper_pcd_features = nets['embed'].weight.unsqueeze(0).repeat(num_gripper_points, B, 1) # shape (num_gripper_points, B, encoder_output_dim)
        if self.mode in ['keep_position_feature_in_attention_feature', "keep_position_feature_in_attention_feature_with_gripper_displacement_to_closest_object"]:
            if self.goal_mode is not None:
                displacement_to_goal = observation['goal_gripper_pcd'] - observation['gripper_pcd']
                input_to_position_embedding = torch.cat([gripper_pcd, displacement_to_goal], dim=-1)
                if self.mode == 'keep_position_feature_in_attention_feature_with_gripper_displacement_to_closest_object':
                    # print("Keep position feature in attention feature with gripper displacement to closest object")
                    displacement_to_closest_object = observation['displacement_gripper_to_object']
                    input_to_position_embedding = torch.cat([input_to_position_embedding, displacement_to_closest_object], dim=-1)
                input_to_position_embedding = einops.rearrange(input_to_position_embedding, "B num_gripper_points c -> (B num_gripper_points) c", B=B, num_gripper_points=self.num_gripper_points)
                gripper_pcd_position_embedding = nets['gripper_pcd_position_embedding_mlp'](input_to_position_embedding)
            else:
                gripper_pcd_position = einops.rearrange(gripper_pcd, "B num_gripper_points c -> (B num_gripper_points) c", B=B, num_gripper_points=self.num_gripper_points)
                if self.mode == 'keep_position_feature_in_attention_feature_with_gripper_displacement_to_closest_object':
                    displacement_to_closest_object = observation['displacement_gripper_to_object']
                    displacement_to_closest_object = einops.rearrange(displacement_to_closest_object, "B num_gripper_points c -> (B num_gripper_points) c", B=B, num_gripper_points=self.num_gripper_points)
                    gripper_pcd_position = torch.cat([gripper_pcd_position, displacement_to_closest_object], dim=-1)
                gripper_pcd_position_embedding = nets['gripper_pcd_position_embedding_mlp'](gripper_pcd_position) # shape B*num_gripper_points encoder_output_dim//3

            gripper_pcd_position_embedding = einops.rearrange(gripper_pcd_position_embedding, "(B num_gripper_points) encoder_output_dim -> num_gripper_points B encoder_output_dim", B=B, num_gripper_points=self.num_gripper_points)
            gripper_pcd_features = torch.cat([gripper_pcd_features, gripper_pcd_position_embedding], dim=-1)

        attn_output = nets['attn_layers'](
            query=gripper_pcd_features, value=rgb_features,
            query_pos=gripper_pcd_rel_pos_embedding, value_pos=point_cloud_rel_pos_embedding,
        )[-1]

        if self.self_attention:
            attn_output = nets['self_attn_layers'](
                query=attn_output, value=attn_output,
                query_pos=gripper_pcd_rel_pos_embedding, value_pos=gripper_pcd_rel_pos_embedding,
            )[-1]

        state_feat = self.state_mlp(agent_pos)  # B * 64
        if not self.final_attention:
            rgb_features = einops.rearrange(
                attn_output, "num_gripper_points B embed_dim -> B num_gripper_points embed_dim").flatten(start_dim=1)
            obs_features = torch.cat([rgb_features, state_feat], dim=-1)

        # if not self.self_attention:
        #     rgb_features = einops.rearrange(
        #         attn_output, "num_gripper_points B embed_dim -> B num_gripper_points embed_dim").flatten(start_dim=1) # shape B (num_gripper_points * encoder_output_dim)
        # else:
        #     self_attn_output = nets['self_attn_layers'](
        #         query=attn_output, value=attn_output,
        #         query_pos=gripper_pcd_rel_pos_embedding, value_pos=gripper_pcd_rel_pos_embedding,
        #     )[-1]
        #     rgb_features = einops.rearrange(
        #         self_attn_output, "num_gripper_points B embed_dim -> B num_gripper_points embed_dim").flatten(start_dim=1)

        if self.goal_mode in ['cross_attention_to_goal', "cross_attention_to_goal_not_concat_and_self_attention"]:
            goal_gripper_pcd_rel_pos_embedding = nets['relative_pe_layer'](observation['goal_gripper_pcd']) # shape B num_gripper_points encoder_output_dim
            goal_gripper_pcd_features = nets['goal_embed'].weight.unsqueeze(0).repeat(num_gripper_points, B, 1) # shape (num_gripper_points, B, encoder_output_dim)
            if self.mode in ['keep_position_feature_in_attention_feature', "keep_position_feature_in_attention_feature_with_gripper_displacement_to_closest_object"]:
                displacement_to_goal = observation['goal_gripper_pcd'] - observation['gripper_pcd']
                input_to_position_embedding = torch.cat([observation['goal_gripper_pcd'], displacement_to_goal], dim=-1)
                if self.mode == 'keep_position_feature_in_attention_feature_with_gripper_displacement_to_closest_object':
                    displacement_to_closest_object = observation['displacement_gripper_to_object']
                    input_to_position_embedding = torch.cat([input_to_position_embedding, displacement_to_closest_object], dim=-1)
                goal_gripper_pcd_position = einops.rearrange(input_to_position_embedding, "B num_gripper_points c -> (B num_gripper_points) c", B=B, num_gripper_points=self.num_gripper_points)
                goal_gripper_pcd_position_embedding = nets['goal_pcd_position_embedding_mlp'](goal_gripper_pcd_position)
                goal_gripper_pcd_position_embedding = einops.rearrange(goal_gripper_pcd_position_embedding, "(B num_gripper_points) encoder_output_dim -> num_gripper_points B encoder_output_dim", B=B, num_gripper_points=self.num_gripper_points)
                goal_gripper_pcd_features = torch.cat([goal_gripper_pcd_features, goal_gripper_pcd_position_embedding], dim=-1)
            
            if self.goal_mode == 'cross_attention_to_goal': # using original gripper pcd features for cross attention to goal and concat it with features from gripper object cross attention
                goal_attn_output = nets['goal_attn_layers'](query=gripper_pcd_features, value=goal_gripper_pcd_features,
                    query_pos=gripper_pcd_rel_pos_embedding, value_pos=goal_gripper_pcd_rel_pos_embedding,
                )[-1]
                
                if self.self_attention:
                    goal_attn_output = nets['goal_self_attn_layers'](query=goal_attn_output, value=goal_attn_output,
                        query_pos=gripper_pcd_rel_pos_embedding, value_pos=gripper_pcd_rel_pos_embedding,
                    )[-1]

                # [Chialiang] more layers
                if self.final_attention:
                    final_attn_output = nets['final_attn_layers'](query=attn_output, value=goal_attn_output,
                        query_pos=gripper_pcd_rel_pos_embedding, value_pos=goal_gripper_pcd_rel_pos_embedding,
                    )[-1]
                    final_attn_output = nets['final_slef_attn_layers'](query=final_attn_output, value=final_attn_output,
                        query_pos=gripper_pcd_rel_pos_embedding, value_pos=gripper_pcd_rel_pos_embedding,
                    )[-1]
                    obs_features = einops.rearrange(
                        final_attn_output, "num_gripper_points B embed_dim -> B num_gripper_points embed_dim").flatten(start_dim=1)
                        
                    obs_features = torch.cat([obs_features, state_feat], dim=-1)     

                else :
                    goal_features = einops.rearrange(
                        goal_attn_output, "num_gripper_points B embed_dim -> B num_gripper_points embed_dim").flatten(start_dim=1)

                    obs_features = torch.cat([obs_features, goal_features], dim=-1)     

            # # [Chialiang] not used
            # elif self.goal_mode == 'cross_attention_to_goal_not_concat_and_self_attention': # using gripper features obtained from cross attention to object, and then do self attention
            #     goal_attn_output = nets['goal_attn_layers'](query=attn_output, value=goal_gripper_pcd_features,
            #         query_pos=gripper_pcd_rel_pos_embedding, value_pos=goal_gripper_pcd_rel_pos_embedding,
            #     )[-1]
                
            #     gripper_self_attn_output = nets['goal_self_attn_layers'](query=goal_attn_output, value=goal_attn_output,
            #         query_pos=gripper_pcd_rel_pos_embedding, value_pos=gripper_pcd_rel_pos_embedding,
            #     )[-1]
                
            #     goal_features = einops.rearrange(
            #         gripper_self_attn_output, "num_gripper_points B embed_dim -> B num_gripper_points embed_dim").flatten(start_dim=1)
            
            #     obs_features = torch.cat([goal_features, state_feat], dim=-1)
        
        
        return obs_features
    
    def output_shape(self):
        return self.n_output_channels

        