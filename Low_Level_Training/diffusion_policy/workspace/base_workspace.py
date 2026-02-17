from typing import Optional
import os
import pathlib
import hydra
import copy
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import dill
import torch
import threading


class BaseWorkspace:
    include_keys = tuple()
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir: Optional[str]=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir
    
    def run(self, rank, world_size):
        """
        Create any resource shouldn't be serialized as local variables
        """
        pass

    def save_checkpoint(self, path=None, tag='latest', 
            exclude_keys=None,
            include_keys=None,
            use_thread=True):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        } 

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    def get_checkpoint_path(self, tag='latest'):
        return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')

    # def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
    #     if exclude_keys is None:
    #         exclude_keys = tuple()
    #     if include_keys is None:
    #         include_keys = payload['pickles'].keys()

    #     for key, value in payload['state_dicts'].items():
    #         if key not in exclude_keys:
    #             self.__dict__[key].load_state_dict(value, **kwargs)
    #     for key in include_keys:
    #         if key in payload['pickles']:
    #             self.__dict__[key] = dill.loads(payload['pickles'][key])

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, state_dict in payload['state_dicts'].items():
            if key not in exclude_keys:
                # Strip "module." from each key
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_k = k[len("module."):] if k.startswith("module.") else k
                    # Optionally remove _dummy_variable
                    if "_dummy_variable" not in new_k:
                        new_state_dict[new_k] = v

                obj = self.__dict__[key]

                # If the object is a Module, ensure dummy buffers exist
                if isinstance(obj, torch.nn.Module):
                    if not hasattr(obj, "_dummy_variable"):
                        obj.register_buffer("_dummy_variable", torch.zeros(1))
                    if hasattr(obj, "mask_generator") and not hasattr(obj.mask_generator, "_dummy_variable"):
                        obj.mask_generator.register_buffer("_dummy_variable", torch.zeros(1))

                    # Load state dict safely with strict=False
                    obj.load_state_dict(new_state_dict, strict=False)
                else:
                    # fallback for non-module objects
                    try:
                        obj.load_state_dict(new_state_dict)
                    except AttributeError:
                        pass  # ignore objects without load_state_dict

        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
    
    # def load_checkpoint(self, path=None, tag='latest',
    #         exclude_keys=None, 
    #         include_keys=None, 
    #         **kwargs):
    #     if path is None:
    #         path = self.get_checkpoint_path(tag=tag)
    #     else:
    #         path = pathlib.Path(path)
    #     payload = torch.load(path.open('rb'), pickle_module=dill, **kwargs)
    #     self.load_payload(payload, 
    #         exclude_keys=exclude_keys, 
    #         include_keys=include_keys)
    #     return payload

    def load_checkpoint(self, path=None, tag='latest',
                    exclude_keys=None, 
                    include_keys=None, 
                    **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        
        # Load checkpoint payload
        payload = torch.load(path.open('rb'), pickle_module=dill, **kwargs)

        # Handle state_dict prefix "module." automatically
        state_dict = payload.get("state_dict", payload)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_key = k[len("module."):]  # remove prefix
            else:
                new_key = k
            new_state_dict[new_key] = v

        # Put back modified state_dict in payload
        if "state_dict" in payload:
            payload["state_dict"] = new_state_dict
        else:
            payload = new_state_dict

        # Load into model
        self.load_payload(payload, 
                        exclude_keys=exclude_keys, 
                        include_keys=include_keys)

        return payload
    
    @classmethod
    def create_from_checkpoint(cls, path, 
            exclude_keys=None, 
            include_keys=None,
            **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload, 
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance

    def save_snapshot(self, tag='latest'):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)


def _copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to('cpu')
    elif isinstance(x, dict):
        result = dict()
        for k, v in x.items():
            result[k] = _copy_to_cpu(v)
        return result
    elif isinstance(x, list):
        return [_copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)
