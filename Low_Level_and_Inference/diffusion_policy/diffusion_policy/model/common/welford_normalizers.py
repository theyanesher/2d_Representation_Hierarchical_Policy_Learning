from typing import Union, Dict

import unittest
import zarr
import numpy as np
import torch
import torch.nn as nn
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.dict_of_tensor_mixin import DictOfTensorMixin


class WelfordOnlineRobotStateNormalizer(DictOfTensorMixin):
    """
    Online Welford normalizer for a robot state vector [pos(3), rot6d(6), gripper(1)].
    Indices 3:9 (rotation) are never normalized — scale=1, offset=0 for those dims.
    """

    @classmethod
    def create_manual(cls, state_dim: int = 10, max_samples: int = None, sigma_normalization: float = 3.0):
        no_norm_mask = torch.zeros(state_dim, dtype=torch.float32)
        no_norm_mask[3:9] = 1.0
        s_norm = torch.ones(state_dim, dtype=torch.float32) * sigma_normalization
        s_norm[no_norm_mask.bool()] = 1.0

        params_dict = nn.ParameterDict({
            'scale': nn.Parameter(torch.ones(state_dim), requires_grad=False),
            'offset': nn.Parameter(torch.zeros(state_dim), requires_grad=False),
            'no_norm_mask': nn.Parameter(no_norm_mask, requires_grad=False),
            'sigma_norm': nn.Parameter(s_norm, requires_grad=False),
            'welford_type': nn.Parameter(torch.tensor(3.0, dtype=torch.float32), requires_grad=False),
            'input_stats': nn.ParameterDict({
                'mean': nn.Parameter(torch.zeros(state_dim), requires_grad=False),
                'std': nn.Parameter(torch.ones(state_dim), requires_grad=False),
                'm2': nn.Parameter(torch.zeros(state_dim), requires_grad=False),
                'samples_seen': nn.Parameter(torch.tensor(0.0, dtype=torch.float64), requires_grad=False),
                'max_samples': nn.Parameter(torch.tensor(max_samples or -1.0, dtype=torch.float64), requires_grad=False),
            }),
        })
        return cls(params_dict=params_dict)

    @torch.no_grad()
    def _update_welford(self, x: torch.Tensor):
        stats = self.params_dict['input_stats']
        n_old = stats['samples_seen'].item()
        max_s = stats['max_samples'].item()
        if max_s != -1 and n_old >= max_s:
            return

        state_dim = self.params_dict['scale'].shape[0]
        x_flat = x.reshape(-1, state_dim).to(stats['mean'].device)
        batch_size = x_flat.shape[0]

        if max_s != -1 and (n_old + batch_size) > max_s:
            batch_size = int(max_s - n_old)
            x_flat = x_flat[:batch_size]

        n_new = n_old + batch_size
        batch_mean = x_flat.mean(dim=0)
        batch_m2 = ((x_flat - batch_mean) ** 2).sum(dim=0)

        delta = batch_mean - stats['mean']
        stats['mean'].add_(delta * (batch_size / n_new))
        stats['m2'].add_(batch_m2 + (delta ** 2) * (n_old * batch_size / n_new))
        stats['samples_seen'].add_(float(batch_size))

        if n_new > 1:
            stats['std'].copy_(torch.sqrt(stats['m2'] / n_new))
            new_scale = 1.0 / stats['std'].clamp(min=1e-4)
            new_offset = -stats['mean'] * new_scale
            mask = self.params_dict['no_norm_mask'].bool()
            new_scale[mask] = 1.0
            new_offset[mask] = 0.0
            self.params_dict['scale'].copy_(new_scale)
            self.params_dict['offset'].copy_(new_offset)

    def normalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if self.training:
            self._update_welford(x)
        scale = self.params_dict['scale']
        offset = self.params_dict['offset']
        s_norm = self.params_dict['sigma_norm']
        x = x.to(device=scale.device, dtype=scale.dtype)
        return (x * scale + offset) / s_norm

    def unnormalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        scale = self.params_dict['scale']
        offset = self.params_dict['offset']
        s_norm = self.params_dict['sigma_norm']
        x = x.to(device=scale.device, dtype=scale.dtype)
        return (x * s_norm - offset) / scale

    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)

class OnlineRangeRobotStateNormalizer(DictOfTensorMixin):
    """
    Online min-max normalizer for a robot state vector [pos(3), rot6d(6), gripper(1)].
    Tracks running per-dimension min/max and maps the observed range to [output_min, output_max].
    Indices 3:9 (rotation) are never normalized — scale=1, offset=0 for those dims.
    """

    @classmethod
    def create_manual(cls, state_dim: int = 10, max_samples: int = None,
                      output_max: float = 1.0, output_min: float = -1.0, range_eps: float = 1e-7):
        no_norm_mask = torch.zeros(state_dim, dtype=torch.float32)
        no_norm_mask[3:9] = 1.0

        params_dict = nn.ParameterDict({
            'scale': nn.Parameter(torch.ones(state_dim), requires_grad=False),
            'offset': nn.Parameter(torch.zeros(state_dim), requires_grad=False),
            'no_norm_mask': nn.Parameter(no_norm_mask, requires_grad=False),
            'output_max': nn.Parameter(torch.tensor(output_max, dtype=torch.float32), requires_grad=False),
            'output_min': nn.Parameter(torch.tensor(output_min, dtype=torch.float32), requires_grad=False),
            'range_eps': nn.Parameter(torch.tensor(range_eps, dtype=torch.float32), requires_grad=False),
            'welford_type': nn.Parameter(torch.tensor(4.0, dtype=torch.float32), requires_grad=False),
            'input_stats': nn.ParameterDict({
                'min': nn.Parameter(torch.full((state_dim,), float('inf')), requires_grad=False),
                'max': nn.Parameter(torch.full((state_dim,), float('-inf')), requires_grad=False),
                'samples_seen': nn.Parameter(torch.tensor(0.0, dtype=torch.float64), requires_grad=False),
                'max_samples': nn.Parameter(torch.tensor(max_samples or -1.0, dtype=torch.float64), requires_grad=False),
            }),
        })
        return cls(params_dict=params_dict)

    @torch.no_grad()
    def _update_range(self, x: torch.Tensor):
        stats = self.params_dict['input_stats']
        n_old = stats['samples_seen'].item()
        max_s = stats['max_samples'].item()
        if max_s != -1 and n_old >= max_s:
            return

        state_dim = self.params_dict['scale'].shape[0]
        x_flat = x.reshape(-1, state_dim).to(stats['min'].device)
        batch_size = x_flat.shape[0]

        if max_s != -1 and (n_old + batch_size) > max_s:
            batch_size = int(max_s - n_old)
            x_flat = x_flat[:batch_size]

        # Initialised to ±inf so this is correct even on the first batch.
        stats['min'].copy_(torch.minimum(stats['min'], x_flat.min(dim=0).values))
        stats['max'].copy_(torch.maximum(stats['max'], x_flat.max(dim=0).values))
        stats['samples_seen'].add_(float(batch_size))

        # Recompute scale/offset from the updated range (mirrors get_range_normalizer_from_stat).
        output_max = self.params_dict['output_max'].item()
        output_min = self.params_dict['output_min'].item()
        range_eps = self.params_dict['range_eps'].item()

        input_range = stats['max'] - stats['min']
        ignore_dim = input_range < range_eps

        effective_range = input_range.clone()
        effective_range[ignore_dim] = output_max - output_min  # => scale = 1 for constant dims

        new_scale = (output_max - output_min) / effective_range
        new_offset = output_min - new_scale * stats['min']
        new_offset[ignore_dim] = (output_max + output_min) / 2 - stats['min'][ignore_dim]

        mask = self.params_dict['no_norm_mask'].bool()
        new_scale[mask] = 1.0
        new_offset[mask] = 0.0

        self.params_dict['scale'].copy_(new_scale)
        self.params_dict['offset'].copy_(new_offset)

    def normalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if self.training:
            self._update_range(x)
        scale = self.params_dict['scale']
        offset = self.params_dict['offset']
        x = x.to(device=scale.device, dtype=scale.dtype)
        return x * scale + offset

    def unnormalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        scale = self.params_dict['scale']
        offset = self.params_dict['offset']
        x = x.to(device=scale.device, dtype=scale.dtype)
        return (x - offset) / scale

    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)


class WelfordImageNormalizer(DictOfTensorMixin):
    @classmethod
    def create_manual(cls, num_channels: int, max_samples: int = None, sigma_normalization: float = 3.0):
        params_dict = nn.ParameterDict({
            'scale': nn.Parameter(torch.ones(num_channels), requires_grad=False),
            'offset': nn.Parameter(torch.zeros(num_channels), requires_grad=False),
            'sigma_norm': nn.Parameter(torch.ones(num_channels) * sigma_normalization, requires_grad=False),
            'welford_type': nn.Parameter(torch.tensor(2.0, dtype=torch.float32), requires_grad=False),
            'input_stats': nn.ParameterDict({
                'mean': nn.Parameter(torch.zeros(num_channels), requires_grad=False),
                'm2': nn.Parameter(torch.zeros(num_channels), requires_grad=False),
                'std': nn.Parameter(torch.ones(num_channels), requires_grad=False),
                'samples_seen': nn.Parameter(torch.tensor(0.0, dtype=torch.float64), requires_grad=False),
                'max_samples': nn.Parameter(torch.tensor(max_samples or -1.0, dtype=torch.float64), requires_grad=False)
            })
        })
        return cls(params_dict=params_dict)

    @torch.no_grad()
    def _update_welford(self, x: torch.Tensor):
        stats = self.params_dict['input_stats']
        n_old = stats['samples_seen'].item()
        max_s = stats['max_samples'].item()
        if max_s != -1 and n_old >= max_s: return

        x_flat = x.permute(0, 1, 3, 4, 2).reshape(-1, x.shape[2]).to(stats['mean'].device)
        batch_size = x_flat.shape[0]

        if max_s != -1 and (n_old + batch_size) > max_s:
            batch_size = int(max_s - n_old)
            x_flat = x_flat[:batch_size]

        n_new = n_old + batch_size
        batch_mean = x_flat.mean(dim=0)
        batch_m2 = ((x_flat - batch_mean) ** 2).sum(dim=0)

        delta = batch_mean - stats['mean']
        stats['mean'].add_(delta * (batch_size / n_new))
        stats['m2'].add_(batch_m2 + (delta ** 2) * (n_old * batch_size / n_new))
        stats['samples_seen'].add_(float(batch_size))

        if n_new > 1:
            stats['std'].copy_(torch.sqrt(stats['m2'] / n_new))
            new_scale = 1.0 / stats['std'].clamp(min=1e-4)
            new_offset = -stats['mean'] * new_scale
            self.params_dict['scale'].copy_(new_scale)
            self.params_dict['offset'].copy_(new_offset)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.training: 
            self._update_welford(x)
        scale = self.params_dict['scale'].view(1, 1, -1, 1, 1)
        offset = self.params_dict['offset'].view(1, 1, -1, 1, 1)
        sigma = self.params_dict['sigma_norm'].view(1, 1, -1, 1, 1)
        
        return (x.to(scale.device) * scale + offset) / sigma

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.params_dict['scale'].view(1, 1, -1, 1, 1)
        offset = self.params_dict['offset'].view(1, 1, -1, 1, 1)
        sigma = self.params_dict['sigma_norm'].view(1, 1, -1, 1, 1)
        
        return (x.to(scale.device) * sigma - offset) / scale