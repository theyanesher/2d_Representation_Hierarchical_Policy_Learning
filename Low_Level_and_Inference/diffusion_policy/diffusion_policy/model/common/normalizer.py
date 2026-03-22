from typing import Union, Dict

import unittest
import zarr
import numpy as np
import torch
import torch.nn as nn
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.dict_of_tensor_mixin import DictOfTensorMixin
from diffusion_policy.model.common.welford_normalizers import WelfordImageNormalizer, WelfordOnlineRobotStateNormalizer, OnlineRangeRobotStateNormalizer

class LinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ['limits', 'gaussian']
    
    @torch.no_grad()
    def fit(self,
        data: Union[Dict, torch.Tensor, np.ndarray, zarr.Array],
        last_n_dims=1,
        dtype=torch.float32,
        mode='limits',
        output_max=1.,
        output_min=-1.,
        range_eps=1e-4,
        fit_offset=True):
        if isinstance(data, dict):
            for key, value in data.items():
                self.params_dict[key] =  _fit(value, 
                    last_n_dims=last_n_dims,
                    dtype=dtype,
                    mode=mode,
                    output_max=output_max,
                    output_min=output_min,
                    range_eps=range_eps,
                    fit_offset=fit_offset)
        else:
            self.params_dict['_default'] = _fit(data, 
                    last_n_dims=last_n_dims,
                    dtype=dtype,
                    mode=mode,
                    output_max=output_max,
                    output_min=output_min,
                    range_eps=range_eps,
                    fit_offset=fit_offset)
    
    def __call__(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)
    
    def __getitem__(self, key: str):
        params = self.params_dict[key]
        if 'welford_type' in params:
            w_type = params['welford_type'].item()
            if np.isclose(w_type, 1.0):
                return WelfordOnlineActionNormalizer(params_dict=params)
            elif np.isclose(w_type, 2.0):
                return WelfordImageNormalizer(params_dict=params)
            elif np.isclose(w_type, 3.0):
                return WelfordOnlineRobotStateNormalizer(params_dict=params)
            elif np.isclose(w_type, 4.0):
                return OnlineRangeRobotStateNormalizer(params_dict=params)
        elif 'sigma_norm' in params:
            # temporary hack for older policies without welford type
            return WelfordOnlineActionNormalizer(params_dict=params)
        return SingleFieldLinearNormalizer(params_dict=params)

    def __setitem__(self, key: str , value: 'SingleFieldLinearNormalizer'):
        self.params_dict[key] = value.params_dict

    def _normalize_impl(self, x, forward=True):
        if isinstance(x, dict):
            result = dict()
            for key, value in x.items():
                params = self.params_dict[key]
                if 'samples_seen' in params.get('input_stats', {}):
                    normalizer = self[key]
                    result[key] = normalizer.normalize(value) if forward else normalizer.unnormalize(value)
                else:
                    result[key] = _normalize(value, params, forward=forward)
            return result
        else:
            if '_default' not in self.params_dict:
                raise RuntimeError("Not initialized")
            params = self.params_dict['_default']
            return _normalize(x, params, forward=forward)

    def normalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._normalize_impl(x, forward=True)

    def unnormalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._normalize_impl(x, forward=False)

    def get_input_stats(self) -> Dict:
        if len(self.params_dict) == 0:
            raise RuntimeError("Not initialized")
        if len(self.params_dict) == 1 and '_default' in self.params_dict:
            return self.params_dict['_default']['input_stats']
        
        result = dict()
        for key, value in self.params_dict.items():
            if key != '_default':
                result[key] = value['input_stats']
        return result


    def get_output_stats(self, key='_default'):
        input_stats = self.get_input_stats()
        if 'min' in input_stats:
            # no dict
            return dict_apply(input_stats, self.normalize)
        
        result = dict()
        for key, group in input_stats.items():
            this_dict = dict()
            for name, value in group.items():
                this_dict[name] = self.normalize({key:value})[key]
            result[key] = this_dict
        return result


class SingleFieldLinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ['limits', 'gaussian']
    
    @torch.no_grad()
    def fit(self,
            data: Union[torch.Tensor, np.ndarray, zarr.Array],
            last_n_dims=1,
            dtype=torch.float32,
            mode='limits',
            output_max=1.,
            output_min=-1.,
            range_eps=1e-4,
            fit_offset=True):
        self.params_dict = _fit(data, 
            last_n_dims=last_n_dims,
            dtype=dtype,
            mode=mode,
            output_max=output_max,
            output_min=output_min,
            range_eps=range_eps,
            fit_offset=fit_offset)
    
    @classmethod
    def create_fit(cls, data: Union[torch.Tensor, np.ndarray, zarr.Array], **kwargs):
        obj = cls()
        obj.fit(data, **kwargs)
        return obj
    
    @classmethod
    def create_manual(cls, 
            scale: Union[torch.Tensor, np.ndarray], 
            offset: Union[torch.Tensor, np.ndarray],
            input_stats_dict: Dict[str, Union[torch.Tensor, np.ndarray]]):
        def to_tensor(x):
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            x = x.flatten()
            return x
        
        # check
        for x in [offset] + list(input_stats_dict.values()):
            assert x.shape == scale.shape
            assert x.dtype == scale.dtype
        
        params_dict = nn.ParameterDict({
            'scale': to_tensor(scale),
            'offset': to_tensor(offset),
            'input_stats': nn.ParameterDict(
                dict_apply(input_stats_dict, to_tensor))
        })
        return cls(params_dict)

    @classmethod
    def create_identity(cls, dtype=torch.float32):
        scale = torch.tensor([1], dtype=dtype)
        offset = torch.tensor([0], dtype=dtype)
        input_stats_dict = {
            'min': torch.tensor([-1], dtype=dtype),
            'max': torch.tensor([1], dtype=dtype),
            'mean': torch.tensor([0], dtype=dtype),
            'std': torch.tensor([1], dtype=dtype)
        }
        return cls.create_manual(scale, offset, input_stats_dict)

    def normalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=True)

    def unnormalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=False)

    def get_input_stats(self):
        return self.params_dict['input_stats']

    def get_output_stats(self):
        return dict_apply(self.params_dict['input_stats'], self.normalize)

    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)

class WelfordOnlineActionNormalizer(DictOfTensorMixin):

    @classmethod
    def create_manual(cls, horizon: int, action_dim: int, max_samples: int = None, sigma_normalization=3.0):
        d_out = horizon * action_dim
        no_norm_mask = torch.zeros((horizon, action_dim), dtype=torch.float32)
        no_norm_mask[:, 3:9] = 1.0
        no_norm_mask = no_norm_mask.flatten()
        s_norm = torch.ones(d_out, dtype=torch.float32) * sigma_normalization
        s_norm[no_norm_mask.bool()] = 1.0

        params_dict = nn.ParameterDict({
            'scale': nn.Parameter(torch.ones(d_out), requires_grad=False),
            'offset': nn.Parameter(torch.zeros(d_out), requires_grad=False),
            'no_norm_mask': nn.Parameter(no_norm_mask, requires_grad=False),
            'sigma_norm': nn.Parameter(s_norm, requires_grad=False),
            'welford_type': nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=False),
            'input_stats': nn.ParameterDict({
                'mean': nn.Parameter(torch.zeros(d_out), requires_grad=False),
                'std': nn.Parameter(torch.ones(d_out), requires_grad=False),
                # Online buffers
                'm2': nn.Parameter(torch.zeros(d_out), requires_grad=False),
                # floats to get around some annoying PyTorch dtype issues
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
        if max_s != -1 and n_old >= max_s:
            return
        x_flat = x.reshape(x.shape[0], -1).to(stats['mean'].device)
        batch_size = x_flat.shape[0]

        if max_s != -1 and (n_old + batch_size) > max_s:
            batch_size = max_s - n_old
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
        x_norm = _normalize(x, self.params_dict, forward=True)
        s_norm = self.params_dict['sigma_norm'].reshape(x_norm.shape[1:])
    
        return x_norm / s_norm

    def unnormalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        s_norm = self.params_dict['sigma_norm'].reshape(x.shape[1:])
        return _normalize(x * s_norm, self.params_dict, forward=False)


class RobotStateNormalizer(DictOfTensorMixin):
    """
    Normalizes a 10D robot state: [pos(3), rot6d(6), gripper(1)].
    The rotation part (indices 3 to 8) is not normalized (identity)."""
    def __init__(self, params_dict: nn.ParameterDict = None):
        super().__init__(params_dict)

    @classmethod
    def create_manual(cls, 
            scale: Union[torch.Tensor, np.ndarray], 
            offset: Union[torch.Tensor, np.ndarray],
            input_stats_dict: Dict[str, Union[torch.Tensor, np.ndarray]]):
        def to_tensor(x):
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            return x.flatten().float()
        scale = to_tensor(scale)
        offset = to_tensor(offset)
        scale[3:9] = 1.0
        offset[3:9] = 0.0
        
        params_dict = nn.ParameterDict({
            'scale': nn.Parameter(scale, requires_grad=False),
            'offset': nn.Parameter(offset, requires_grad=False),
            'input_stats': nn.ParameterDict(
                dict_apply(input_stats_dict, to_tensor))
        })
        return cls(params_dict)

    def normalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=True)

    def unnormalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=False)

    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)

def _fit(data: Union[torch.Tensor, np.ndarray, zarr.Array],
        last_n_dims=1,
        dtype=torch.float32,
        mode='limits',
        output_max=1.,
        output_min=-1.,
        range_eps=1e-4,
        fit_offset=True):
    assert mode in ['limits', 'gaussian']
    assert last_n_dims >= 0
    assert output_max > output_min

    # convert data to torch and type
    if isinstance(data, zarr.Array):
        data = data[:]
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if dtype is not None:
        data = data.type(dtype)

    # convert shape
    dim = 1
    if last_n_dims > 0:
        dim = np.prod(data.shape[-last_n_dims:])
    data = data.reshape(-1,dim)

    # compute input stats min max mean std
    input_min, _ = data.min(axis=0)
    input_max, _ = data.max(axis=0)
    input_mean = data.mean(axis=0)
    input_std = data.std(axis=0)

    # compute scale and offset
    if mode == 'limits':
        if fit_offset:
            # unit scale
            input_range = input_max - input_min
            ignore_dim = input_range < range_eps
            input_range[ignore_dim] = output_max - output_min
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * input_min
            offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
            # ignore dims scaled to mean of output max and min
        else:
            # use this when data is pre-zero-centered.
            assert output_max > 0
            assert output_min < 0
            # unit abs
            output_abs = min(abs(output_min), abs(output_max))
            input_abs = torch.maximum(torch.abs(input_min), torch.abs(input_max))
            ignore_dim = input_abs < range_eps
            input_abs[ignore_dim] = output_abs
            # don't scale constant channels 
            scale = output_abs / input_abs
            offset = torch.zeros_like(input_mean)
    elif mode == 'gaussian':
        ignore_dim = input_std < range_eps
        scale = input_std.clone()
        scale[ignore_dim] = 1
        scale = 1 / scale

        if fit_offset:
            offset = - input_mean * scale
        else:
            offset = torch.zeros_like(input_mean)
    
    # save
    this_params = nn.ParameterDict({
        'scale': scale,
        'offset': offset,
        'input_stats': nn.ParameterDict({
            'min': input_min,
            'max': input_max,
            'mean': input_mean,
            'std': input_std
        })
    })
    for p in this_params.parameters():
        p.requires_grad_(False)
    return this_params


def _normalize(x, params, forward=True):
    assert 'scale' in params
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    scale = params['scale']
    offset = params['offset']
    x = x.to(device=scale.device, dtype=scale.dtype)
    src_shape = x.shape
    x = x.reshape(-1, scale.shape[0])
    if forward:
        x = x * scale + offset
    else:
        x = (x - offset) / scale
    x = x.reshape(src_shape)
    return x


def test():
    data = torch.zeros((100,10,9,2)).uniform_()
    data[...,0,0] = 0

    normalizer = SingleFieldLinearNormalizer()
    normalizer.fit(data, mode='limits', last_n_dims=2)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.max(), 1.)
    assert np.allclose(datan.min(), -1.)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    input_stats = normalizer.get_input_stats()
    output_stats = normalizer.get_output_stats()

    normalizer = SingleFieldLinearNormalizer()
    normalizer.fit(data, mode='limits', last_n_dims=1, fit_offset=False)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.max(), 1., atol=1e-3)
    assert np.allclose(datan.min(), 0., atol=1e-3)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    data = torch.zeros((100,10,9,2)).uniform_()
    normalizer = SingleFieldLinearNormalizer()
    normalizer.fit(data, mode='gaussian', last_n_dims=0)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.mean(), 0., atol=1e-3)
    assert np.allclose(datan.std(), 1., atol=1e-3)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)


    # dict
    data = torch.zeros((100,10,9,2)).uniform_()
    data[...,0,0] = 0

    normalizer = LinearNormalizer()
    normalizer.fit(data, mode='limits', last_n_dims=2)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.max(), 1.)
    assert np.allclose(datan.min(), -1.)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    input_stats = normalizer.get_input_stats()
    output_stats = normalizer.get_output_stats()

    data = {
        'obs': torch.zeros((1000,128,9,2)).uniform_() * 512,
        'action': torch.zeros((1000,128,2)).uniform_() * 512
    }
    normalizer = LinearNormalizer()
    normalizer.fit(data)
    datan = normalizer.normalize(data)
    dataun = normalizer.unnormalize(datan)
    for key in data:
        assert torch.allclose(data[key], dataun[key], atol=1e-4)
    
    input_stats = normalizer.get_input_stats()
    output_stats = normalizer.get_output_stats()

    state_dict = normalizer.state_dict()
    n = LinearNormalizer()
    n.load_state_dict(state_dict)
    datan = n.normalize(data)
    dataun = n.unnormalize(datan)
    for key in data:
        assert torch.allclose(data[key], dataun[key], atol=1e-4)
