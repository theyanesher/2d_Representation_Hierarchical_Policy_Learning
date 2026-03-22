import torch 
import torch.nn as nn
import diffusion_policy.model.vision.crop_randomizer as dmvc
import torchvision.transforms.functional as ttf
import robomimic.models.base_nets as rmbn
from robomimic.models.obs_nets import ObservationEncoder
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules

class IdentityRandomizer(nn.Module):
    def __init__(self, crop_height, crop_width):
        super().__init__()
        self.crop_height = crop_height
        self.crop_width = crop_width

    def forward_in(self, x): return x
    def forward_out(self, x): return x
    def forward(self, x): return x

    def output_shape_in(self, input_shape):
        return [input_shape[0], self.crop_height, self.crop_width]

    def output_shape_out(self, input_shape):
        return list(input_shape)

class SharedCropModule(nn.Module):
    def __init__(self, groups, crop_height, crop_width, num_crops=1):
        super().__init__()
        self.groups = groups 
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.num_crops = num_crops

    def forward(self, obs_dict):
        for prefix, keys in self.groups.items():
            lead_key = next((k for k in keys if k in obs_dict), None)
            if lead_key is None: continue
            
            lead_img = obs_dict[lead_key]
            leading_dims = lead_img.shape[:-3] 
            img_shape = lead_img.shape[-3:]

            if self.training:
                flat_img = lead_img.reshape(-1, *img_shape)
                
                _, crop_inds = dmvc.sample_random_image_crops(
                    flat_img, 
                    self.crop_height, self.crop_width, self.num_crops
                )
                
                for key in keys:
                    if key not in obs_dict: continue
                    val = obs_dict[key]
                    C = val.shape[-3]
                    
                    val_flat = val.reshape(-1, C, *val.shape[-2:])
                    cropped = dmvc.crop_image_from_indices(
                        val_flat, 
                        crop_inds, 
                        self.crop_height, self.crop_width
                    )
                    
                    target_shape = (*leading_dims, self.num_crops, C, self.crop_height, self.crop_width)
                    cropped = cropped.reshape(target_shape)
                    if self.num_crops > 1:
                        obs_dict[key] = cropped.mean(dim=len(leading_dims))
                    else:
                        obs_dict[key] = cropped.squeeze(len(leading_dims))
            else:
                for key in keys:
                    if key not in obs_dict: continue
                    obs_dict[key] = ttf.center_crop(obs_dict[key], (self.crop_height, self.crop_width))
                    
        return obs_dict

class SharedCropObsEncoder(nn.Module):
    # wrapper class to share cropping information between resnets

    def __init__(self, obs_encoder: ObservationEncoder, obs_key_shapes, crop_shape):
        super(SharedCropObsEncoder, self).__init__()
        self.obs_encoder = obs_encoder
        self.crop_shape = crop_shape
        self.obs_key_shapes = obs_key_shapes
        self.camera_groups = {
            'cam0': [k for k in obs_key_shapes if k.startswith('cam0')],
            'cam1': [k for k in obs_key_shapes if k.startswith('cam1')],
            'cam2': [k for k in obs_key_shapes if k.startswith('cam2')]
        }
        assert len(self.camera_groups['cam0']) > 1

        replace_submodules(
            root_module=obs_encoder,
            predicate=lambda x: isinstance(x, rmbn.CropRandomizer) or isinstance(x, dmvc.CropRandomizer),
            func=lambda x: IdentityRandomizer(
                crop_height=x.crop_height,
                crop_width=x.crop_width
            )
        )
        self.shared_crop_module = SharedCropModule(
            groups=self.camera_groups,
            crop_height=crop_shape[0],
            crop_width=crop_shape[1],
            num_crops=1
        )


    def output_shape(self):
        return self.obs_encoder.output_shape()

    def forward(self, obs_dict):
        obs_dict = self.shared_crop_module.forward(obs_dict)
        obs_features = self.obs_encoder(obs_dict)
        return obs_features


class SinglePointMapSharedCropObsEncoder(nn.Module):
    # wrapper class to share cropping information between resnets

    def __init__(self, obs_encoder: ObservationEncoder, obs_key_shapes, crop_shape):
        super(SinglePointMapSharedCropObsEncoder, self).__init__()
        self.obs_encoder = obs_encoder
        self.crop_shape = crop_shape
        self.obs_key_shapes = obs_key_shapes
        self.camera_groups = {
            'cam0': [k for k in obs_key_shapes if k.startswith('cam0')],
            'cam1': [k for k in obs_key_shapes if k.startswith('cam1')],
            'cam2': [k for k in obs_key_shapes if k.startswith('cam2')]
        }
        assert len(self.camera_groups['cam0']) > 1

        replace_submodules(
            root_module=obs_encoder,
            predicate=lambda x: isinstance(x, rmbn.CropRandomizer) or isinstance(x, dmvc.CropRandomizer),
            func=lambda x: IdentityRandomizer(
                crop_height=x.crop_height,
                crop_width=x.crop_width
            )
        )
        self.shared_crop_module = SharedCropModule(
            groups=self.camera_groups,
            crop_height=crop_shape[0],
            crop_width=crop_shape[1],
            num_crops=1
        )

        # share encoder with cam0 pointmap for cam1 and cam2 pointmaps
        for key in ['cam1_pointmap', 'cam2_pointmap']:
            net = obs_encoder.obs_nets[key]
            obs_encoder.obs_nets[key] = obs_encoder.obs_nets['cam0_pointmap']
            del net

    def output_shape(self):
        return self.obs_encoder.output_shape()

    def forward(self, obs_dict):
        obs_dict = self.shared_crop_module.forward(obs_dict)
        obs_features = self.obs_encoder(obs_dict)
        return obs_features