# # """
# # Usage:
# # Training:
# # python train.py --config-name=train_diffusion_lowdim_workspace
# # """

# # import sys
# # # use line-buffering for both stdout and stderr
# # sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
# # sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)
# # import torch.multiprocessing as mp
# # import hydra
# # from omegaconf import OmegaConf
# # import pathlib
# # from diffusion_policy.workspace.base_workspace import BaseWorkspace

# # # allows arbitrary python code execution in configs using the ${eval:''} resolver
# # OmegaConf.register_new_resolver("eval", eval, replace=True)

# # @hydra.main(
# #     version_base=None,
# #     config_path=str(pathlib.Path(__file__).parent.joinpath(
# #         'diffusion_policy','config'))
# # )
# # def main(cfg: OmegaConf):
# #     # resolve immediately so all the ${now:} resolvers
# #     # will use the same time.
# #     # import pdb; pdb.set_trace();
# #     OmegaConf.resolve(cfg)

# #     cls = hydra.utils.get_class(cfg._target_)
# #     workspace: BaseWorkspace = cls(cfg)
# #     workspace.run()

# # if __name__ == "__main__":
# #     mp.set_start_method("spawn", force=True)
# #     main()



# """
# Usage:
# Single GPU:
#     python train.py --config-name=train_diffusion_lowdim_workspace

# Multi-GPU:
#     torchrun --standalone --nproc_per_node=<NUM_GPUS> train.py --config-name=train_diffusion_lowdim_workspace
# """

# import sys
# import torch
# import torch.multiprocessing as mp
# import torch.distributed as dist
# import hydra
# from omegaconf import OmegaConf
# import pathlib
# from diffusion_policy.workspace.base_workspace import BaseWorkspace
# import os

# # Line-buffering for live logs
# sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
# sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

# OmegaConf.register_new_resolver("eval", eval, replace=True)


# def setup_distributed():
#     """Initialize distributed training environment."""
#     dist.init_process_group(backend="nccl")
#     torch.cuda.set_device(dist.get_rank())
#     print(f"Process {dist.get_rank()} initialized on GPU {dist.get_rank()}")


# def cleanup_distributed():
#     dist.destroy_process_group()


# @hydra.main(
#     version_base=None,
#     config_path=str(pathlib.Path(__file__).parent.joinpath("diffusion_policy", "config"))
# )
# def main(cfg: OmegaConf):
#     OmegaConf.resolve(cfg)

#     # If launched with torchrun, initialize distributed
#     if dist.is_available() and dist.is_initialized() is False and "LOCAL_RANK" in os.environ:
#         setup_distributed()

#     cls = hydra.utils.get_class(cfg._target_)
#     workspace: BaseWorkspace = cls(cfg)
#     workspace.run(rank, world_size)

#     if dist.is_initialized():
#         cleanup_distributed()


# if __name__ == "__main__":
#     mp.set_start_method("spawn", force=True)
#     main()


"""
Usage:
Single GPU:
    python train.py --config-name=train_diffusion_lowdim_workspace

Multi-GPU:
    torchrun --standalone --nproc_per_node=<NUM_GPUS> train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
import torch
import torch.distributed as dist
import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import os

# Line-buffering for live logs
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

OmegaConf.register_new_resolver("eval", eval, replace=True)


def setup_distributed():
    """Initialize distributed training environment from torchrun env variables."""
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"[Rank {rank}/{world_size}] initialized on GPU {rank}")
    return rank, world_size


def cleanup_distributed():
    dist.destroy_process_group()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("diffusion_policy", "config"))
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    # import pdb; pdb.set_trace();
    # Check if we're running distributed
    is_distributed = "LOCAL_RANK" in os.environ
    if is_distributed:
        rank, world_size = setup_distributed()
    else:
        rank, world_size = 0, 1

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)

    # Pass rank and world_size to workspace.run()
    workspace.run(rank=rank, world_size=world_size)

    if is_distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()
