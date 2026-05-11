import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy

from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint

from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import einops
from equi_diffpo.model.vision.layers import RelativeCrossAttentionModule, RotaryPositionEncoding3D
from third_party.robogen.test_PointNet2.model_invariant import PointNet2_small2

def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d) or isinstance(x, nn.BatchNorm1d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module


class PN2Act3dEncoder(nn.Module):
    def __init__(self, 
                 in_channels=6, 
                 encoder_output_dim=60, 
                 num_gripper_points=4, 
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 observation_space=None,
                 goal_mode=None,
                 mode=None,
                 use_mlp=False,
                 self_attention=False,
                 use_attn_for_point_features=False,
                 pointcloud_backbone='mlp',
                 use_lightweight_unet=False,
                 final_attention=False,
                 attention_num_heads=3,
                 attention_num_layers=2,
                 use_repr_10d=False,
                 goal_conditioning=True,
                 **kwargs
                 ):
        super(PN2Act3dEncoder, self).__init__()
        hidden_layer_dim = encoder_output_dim
        vision_encoder = PointNet2_small2(num_classes=hidden_layer_dim, in_channels=in_channels)
        vision_encoder = replace_bn_with_gn(vision_encoder)

        attn_layers = RelativeCrossAttentionModule(encoder_output_dim, attention_num_heads, attention_num_layers)
        attn_layers = replace_bn_with_gn(attn_layers)

        self.nets = nn.ModuleDict({
            'vision_encoder': vision_encoder,
            'relative_pe_layer': RotaryPositionEncoding3D(encoder_output_dim),
            'attn_layers': attn_layers,
        })

        position_embedding_input_size = 9 if goal_conditioning else 6
        position_embedding_mlp = nn.Sequential(
            nn.Linear(position_embedding_input_size, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, encoder_output_dim // 3),
        )
        
        self.nets['gripper_pcd_position_embedding_mlp'] = position_embedding_mlp
        self.nets['embed'] = nn.Embedding(1, encoder_output_dim // 3 * 2)

        goal_attn_layers = RelativeCrossAttentionModule(encoder_output_dim, attention_num_heads, attention_num_layers)
        goal_attn_layers = replace_bn_with_gn(goal_attn_layers)
        self.nets['goal_attn_layers'] = goal_attn_layers
        self.nets['goal_pcd_position_embedding_mlp'] = copy.deepcopy(position_embedding_mlp)
        self.nets['goal_embed'] = nn.Embedding(1, encoder_output_dim // 3 * 2)
        self.goal_conditioning = goal_conditioning

    def forward(self, x):
        # x shape: [B, hor, N, input_dim]
        # scene point cloud
        num_scene_points = x.shape[1] - 8
        point_cloud = x[..., :num_scene_points, :]

        B, N, C = point_cloud.shape
        point_cloud_features = self.nets['vision_encoder'](point_cloud)
        # point_cloud_flatten = point_cloud.reshape(-1, C)
        # point_cloud_features_flatten = self.nets['vision_encoder'](point_cloud_flatten)
        # point_cloud_features = point_cloud_features_flatten.reshape(B, N, -1)
        point_cloud_features = einops.rearrange(point_cloud_features, "B N encoder_output_dim -> N B encoder_output_dim")  # N B encoder_output_dim
        point_cloud_rel_pos_embedding = self.nets['relative_pe_layer'](point_cloud)  # B N encoder_output_dim

        # attention between gripper pcd and scene pcd
        # gripper_pcd = x[..., 1024 + chosen_four_point_idx, :]
        gripper_pcd = x[:, num_scene_points:num_scene_points + 4, :]
        gripper_pcd_rel_pos_embedding = self.nets['relative_pe_layer'](gripper_pcd)  # B num_gripper_points encoder_output_dim
        gripper_pcd_features = self.nets['embed'].weight.unsqueeze(0).repeat(4, B, 1)  # num_gripper_points B encoder_output_dim


        if self.goal_conditioning:
            goal_gripper_pcd = x[:, -4:, :]
            displacement_to_goal = goal_gripper_pcd - gripper_pcd
            input_to_position_embedding = torch.cat([gripper_pcd, displacement_to_goal[:, :, :3]], dim=-1)  # B num_gripper_points 9
            gripper_pcd_position_embedding = self.nets['gripper_pcd_position_embedding_mlp'](input_to_position_embedding)
            gripper_pcd_position_embedding = einops.rearrange(gripper_pcd_position_embedding, "B N encoder_output_dim -> N B encoder_output_dim")  # N B encoder_output_dim

            gripper_pcd_features = torch.cat([gripper_pcd_features, gripper_pcd_position_embedding], dim=-1)

            # import pdb; pdb.set_trace()
            attn_output = self.nets['attn_layers'](query=gripper_pcd_features, value=point_cloud_features,query_pos=gripper_pcd_rel_pos_embedding, value_pos=point_cloud_rel_pos_embedding,)[-1]  # N B encoder_output_dim

            # goal gripper
            goal_gripper_pcd_rel_pos_embedding = self.nets['relative_pe_layer'](goal_gripper_pcd)
            goal_gripper_pcd_features = self.nets['goal_embed'].weight.unsqueeze(0).repeat(4, B, 1)

            displacement_to_goal = goal_gripper_pcd[..., :3] - gripper_pcd[..., :3]
            input_to_position_embedding = torch.cat([goal_gripper_pcd, displacement_to_goal], dim=-1)
            goal_gripper_pcd_position_embedding = self.nets['goal_pcd_position_embedding_mlp'](input_to_position_embedding)
            goal_gripper_pcd_position_embedding = einops.rearrange(goal_gripper_pcd_position_embedding, "B N encoder_output_dim -> N B encoder_output_dim")  # N B encoder_output_dim

            goal_gripper_pcd_features = torch.cat([goal_gripper_pcd_features, goal_gripper_pcd_position_embedding], dim=-1)

            goal_attn_output = self.nets['goal_attn_layers'](query=gripper_pcd_features, value=goal_gripper_pcd_features,query_pos=gripper_pcd_rel_pos_embedding, value_pos=goal_gripper_pcd_rel_pos_embedding,)[-1]

            obs_feature = torch.cat([attn_output, goal_attn_output], dim=-1)
            obs_feature = einops.rearrange(obs_feature, "N B encoder_output_dim -> B N encoder_output_dim")
        else:
            gripper_pcd_position_embedding = self.nets['gripper_pcd_position_embedding_mlp'](gripper_pcd) # should be 6
            gripper_pcd_position_embedding = einops.rearrange(gripper_pcd_position_embedding, "B N encoder_output_dim -> N B encoder_output_dim")  # N B encoder_output_dim

            gripper_pcd_features = torch.cat([gripper_pcd_features, gripper_pcd_position_embedding], dim=-1)

            # import pdb; pdb.set_trace()
            attn_output = self.nets['attn_layers'](query=gripper_pcd_features, value=point_cloud_features,query_pos=gripper_pcd_rel_pos_embedding, value_pos=point_cloud_rel_pos_embedding,)[-1]  # N B encoder_output_dim

            obs_feature = einops.rearrange(attn_output, "N B encoder_output_dim -> B N encoder_output_dim")

        return obs_feature.flatten(start_dim=1)
