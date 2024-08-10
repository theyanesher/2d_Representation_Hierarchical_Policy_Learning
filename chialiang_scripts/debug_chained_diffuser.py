from diffusion_policy_3d.policy.chained_diffusor import DiffusionPlanner
from diffusion_policy_3d.dataset.dataset_engine import ChainedDiffusorDataset
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel

from tqdm import tqdm
from typing import Dict, Callable, List
import collections
import torch
import copy
import torch.nn as nn

from diffusers.optimization import (
    Union, SchedulerType, Optional,
    Optimizer, TYPE_TO_SCHEDULER_FUNCTION
)

def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    **kwargs
):
    """
    Added kwargs vs diffuser's original implementation

    Unified API to get any scheduler from its name.

    Args:
        name (`str` or `SchedulerType`):
            The name of the scheduler to use.
        optimizer (`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (`int`, *optional*):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (`int``, *optional*):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer, **kwargs)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **kwargs)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, **kwargs)


def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result

from torch.utils.data import DataLoader

if __name__=="__main__":

    dataset = ChainedDiffusorDataset()# Testing with DataLoader
    dataloader = DataLoader(
                    dataset,
                    batch_size=64,
                    num_workers=15,
                    shuffle=True,
                    pin_memory=True,
                    persistent_workers=False,
                )

    model = DiffusionPlanner(rotation_parametrization='quat')
    # model = DiffusionPlanner(rotation_parametrization='6D')
    model.to('cuda')

    ema_model = copy.deepcopy(model)

    ema = EMAModel(
        update_after_step=0,
        inv_gamma=1.0,
        power=0.75,
        min_value=0.0,
        max_value=0.9999,
        model=ema_model
    )

    optimizer = torch.optim.AdamW(model.parameters(), 
                                    lr=1.0e-4,
                                    betas=[0.95, 0.999],
                                    eps=1.0e-8,
                                    weight_decay=1.0e-6
                                )

    # lr_scheduler = get_scheduler(
    #         'cosine',
    #         optimizer=optimizer,
    #         num_warmup_steps=500,
    #         num_training_steps=(
    #             len(dataloader) * 100) \
    #                 // 1,
    #         # pytorch assumes stepping LRScheduler every epoch
    #         # however huggingface diffusers steps it every batch
    #         last_epoch=-1
    #     )
    
    for i in range(100):
        for batch in tqdm(dataloader):

            batch = dict_apply(batch, lambda x: x.to('cuda', non_blocking=True))

            total_loss = model.compute_loss(batch)
            total_loss.backward()
            optimizer.step()
            # optimizer.zero_grad()
            # lr_scheduler.step()
            ema.step(model)

            print(total_loss)