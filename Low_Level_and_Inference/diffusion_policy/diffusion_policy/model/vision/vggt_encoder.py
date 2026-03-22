from torch import nn
import torch
from torchvision import models as vision_models
from vggt.models.aggregator import Aggregator
import robomimic.models.base_nets as rmbn

class VGGTEncoder(nn.Module):
    def __init__(self, obs_key_shapes, crop_shape):
        super(VGGTEncoder, self).__init__()
        embed_dim = 384
        self.vggt_encoder = Aggregator(crop_shape[0],patch_size=14,depth=4,embed_dim=embed_dim,patch_embed='dinov2_vits14_reg')
        # self.vggt_encoder = Aggregator(crop_shape[0],patch_size=28,depth=4,embed_dim=embed_dim,patch_embed='dinov2_vits14_reg')
        # embed_dim = 768
        # self.vggt_encoder = Aggregator(crop_shape[0],patch_size=20,depth=2,embed_dim=embed_dim,patch_embed='dinov2_vitb14_reg')
        # self.vggt_encoder = Aggregator(crop_shape[0],patch_size=20,depth=4,embed_dim=embed_dim,patch_embed='dinov2_vitl14_reg')

        self.obs_key_shapes = obs_key_shapes

        vggt_params_count = sum(p.numel() for p in self.vggt_encoder.parameters())
        dino_params_count = sum(p.numel() for p in self.vggt_encoder.patch_embed.parameters())


        self.crop_randomizers = nn.ModuleDict()
        for key in obs_key_shapes.keys():
            if len(obs_key_shapes[key]) == 3:  # Only create crop randomizers for image observations
                self.crop_randomizers[key] = rmbn.CropRandomizer(
                    input_shape=obs_key_shapes[key],
                    crop_height=crop_shape[0],
                    crop_width=crop_shape[1],
                    num_crops=1,
                    pos_enc=False
                )
        self.token_norm = nn.LayerNorm(embed_dim*2)
        self.feature_proj = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

        print("VGGT params: %e" % vggt_params_count)
        print("DINO params: %e" % dino_params_count)
        print("Attn params: %e" % (vggt_params_count - dino_params_count))

    def forward(self, obs_dict):
        images = []

        for key in self.crop_randomizers.keys():
            obs_dict[key] = self.crop_randomizers[key].forward_in(obs_dict[key])

        # start with cam2_image since we eventually want to predict actions is in gripper frame
        for key in ['cam2_image', 'cam0_image', 'cam1_image']:
            images.append(obs_dict[key])
        x = torch.stack(images, dim=1)
        output_list, patch_start_idx = self.vggt_encoder(x)
        tokens = output_list[-1]
        cls_tokens = tokens[:, :, 0, :]
        cls_tokens = self.token_norm(cls_tokens)
        features = self.feature_proj(cls_tokens)
        features = features.view(features.size(0), -1)
        features = torch.cat([features, obs_dict['state']], dim=-1)
        return features

    def output_shape(self):
        return [128 * 3 + 10]