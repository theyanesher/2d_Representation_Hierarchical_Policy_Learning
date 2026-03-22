import torch
import torch.nn as nn
from robomimic.models.obs_nets import ObservationEncoder
from diffusion_policy.model.vision.prope import PropeDotProductAttention
import robomimic.utils.tensor_utils as TensorUtils
import torch.nn.functional as F
from diffusion_policy.model.common.mlp import create_mlp

class PropeCrossAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int, patches_x: int, 
                 patches_y: int, img_w: int, img_h: int):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        assert self.head_dim % 4 == 0, "Head dim must be divisible by 4 for PRoPE"

        # Projections to separate the roles of Q, K, and V
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        
        # The final projection after attention
        self.o_proj = nn.Linear(channels, channels)

        self.attn_src = PropeDotProductAttention(
            head_dim=self.head_dim,
            patches_x=patches_x, patches_y=patches_y,
            image_width=img_w, image_height=img_h
        )
        self.attn_tgt = PropeDotProductAttention(
            head_dim=self.head_dim,
            patches_x=patches_x, patches_y=patches_y,
            image_width=img_w, image_height=img_h
        )

    def forward(self, 
                shoulder_feat: torch.Tensor,
                wrist_feat: torch.Tensor,
                viewmats_shoulder: torch.Tensor, 
                Ks_shoulder: torch.Tensor,
                viewmats_wrist: torch.Tensor,
                Ks_wrist: torch.Tensor):
        
        B, N, C = shoulder_feat.shape
        H, D = self.num_heads, self.head_dim

        # 1. Reshape to (B, Heads, Tokens, Head_Dim)
        q = self.q_proj(shoulder_feat).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(wrist_feat).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(wrist_feat).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Precompute and cache for BOTH cameras separately
        self.attn_src._precompute_and_cache_apply_fns(viewmats_shoulder, Ks_shoulder)
        self.attn_tgt._precompute_and_cache_apply_fns(viewmats_wrist, Ks_wrist)

        # 3. Apply specific transforms to each component
        # Source (Shoulder) provides the Query
        q_src = self.attn_src._apply_to_q(q)
        
        # Target (Wrist) provides the Key and Value
        k_tgt = self.attn_tgt._apply_to_kv(k)
        v_tgt = self.attn_tgt._apply_to_kv(v)

        # 4. Attention in shared geometric space
        o_src = F.scaled_dot_product_attention(q_src, k_tgt, v_tgt)

        # 5. Project output back into Source (Shoulder) frame
        o_src = self.attn_src._apply_to_o(o_src)

        # 6. Recombine Heads: (B, H, N, D) -> (B, N, C)
        out = o_src.transpose(1, 2).contiguous().view(B, N, C)
        return self.o_proj(out)

class PRoPEObsEncoder(nn.Module):

    def __init__(self, obs_encoder: ObservationEncoder):
        super(PRoPEObsEncoder, self).__init__()
        self.obs_encoder = obs_encoder
        self.prope_cross_attention_0_2 = PropeCrossAttention(
            channels=512,
            num_heads=16,
            patches_x=7,
            patches_y=7,
            img_w=256,
            img_h=256
        )
        self.prope_cross_attention_1_2 = PropeCrossAttention(
            channels=512,
            num_heads=16,
            patches_x=7,
            patches_y=7,
            img_w=256,
            img_h=256
        )

        self.mlp =  nn.Sequential(*create_mlp(512 * 2,
                                              self.obs_encoder.output_shape()[0] - 10,
                                              [512,256]))

    def output_shape(self):
        return self.obs_encoder.output_shape()

    def forward(self, obs_dict):
        """
        hack to override ObservationEncoder forward to add prope attention
        """
        assert self.obs_encoder._locked, "ObservationEncoder: @make has not been called yet"

        assert set(self.obs_encoder.obs_shapes.keys()).issubset(obs_dict), "ObservationEncoder: {} does not contain all modalities {}".format(
            list(obs_dict.keys()), list(self.obs_encoder.obs_shapes.keys())
        )
        # process modalities by order given by @self.obs_shapes
        feats = {}
        for k in self.obs_encoder.obs_shapes:
            x = obs_dict[k]
            # maybe process encoder input with randomizer
            if self.obs_encoder.obs_randomizers[k] is not None:
                x = self.obs_encoder.obs_randomizers[k].forward_in(x)
            # maybe process with obs net
            if self.obs_encoder.obs_nets[k] is not None:
                x = self.obs_encoder.obs_nets[k].nets[:1](x) # only use resnet
                if self.obs_encoder.activation is not None:
                    x = self.obs_encoder.activation(x)
            # maybe process encoder output with randomizer
            if self.obs_encoder.obs_randomizers[k] is not None:
                x = self.obs_encoder.obs_randomizers[k].forward_out(x)
            feats[k] = x

        # grab a spatial softmax from the networks
        B, C, H, W = feats['cam0_image'].shape
        cam_02_feat = self.prope_cross_attention_0_2(
            shoulder_feat=feats['cam0_image'].view(B, C, -1).permute(0, 2, 1),
            wrist_feat=feats['cam2_image'].view(B, C, -1).permute(0, 2, 1),
            viewmats_shoulder=obs_dict['cam0_extrinsics'].unsqueeze(1),
            Ks_shoulder=obs_dict['cam0_intrinsics'].unsqueeze(1),
            viewmats_wrist=obs_dict['cam2_extrinsics'].unsqueeze(1),
            Ks_wrist=obs_dict['cam2_intrinsics'].unsqueeze(1)
        )

        cam_02_feat = self.obs_encoder.obs_nets['cam0_image'].nets[1:](cam_02_feat.permute(0, 2, 1).view(B, C, H, W))
        # cam_02_feat = cam_02_feat.mean(dim=1) 

        cam_12_feat = self.prope_cross_attention_1_2(
            shoulder_feat=feats['cam1_image'].view(B, C, -1).permute(0, 2, 1),
            wrist_feat=feats['cam2_image'].view(B, C, -1).permute(0, 2, 1),
            viewmats_shoulder=obs_dict['cam1_extrinsics'].unsqueeze(1),
            Ks_shoulder=obs_dict['cam1_intrinsics'].unsqueeze(1),
            viewmats_wrist=obs_dict['cam2_extrinsics'].unsqueeze(1),
            Ks_wrist=obs_dict['cam2_intrinsics'].unsqueeze(1)
        )

        cam_12_feat = self.obs_encoder.obs_nets['cam1_image'].nets[1:](cam_12_feat.permute(0, 2, 1).view(B, C, H, W))
        # cam_12_feat = cam_12_feat.mean(dim=1)

        #  cam2_feat = self.obs_encoder.obs_nets['cam2_image'].nets[1:](feats['cam2_image'])
        # feats = torch.cat([cam_02_feat, cam_12_feat], dim=-1)
        # feats = self.mlp(feats)
        # feats = torch.cat([feats, cam2_feat, obs_dict['state']], dim=-1)

        cam2_feat = self.obs_encoder.obs_nets['cam2_image'].nets[1:](feats['cam2_image'])
        feats = torch.cat([cam_02_feat, cam_12_feat, cam2_feat, feats['state']], dim=-1)
        return feats