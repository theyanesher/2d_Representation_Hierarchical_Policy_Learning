"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
pixi run CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 torchrun --standalone --nproc_per_node=1 train_ddp.py --config-name=train_ddp_diffusion_unet_hybrid_workspace.yaml task=rgb_heatmap_articubot.yaml \task.dataset.data_dir=/scratch/pbhowal/Articubot_Data_For_DP_and_Groot/Heatmap_Articubot_Dataset/ task.dataset.max_train_episodes=1
CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 torchrun --standalone --nproc_per_node=1 train.py --config-dir=diffusion_policy/config/ --config-name=train_diffusion_unet_hybrid_workspace_zarr_dataloader_EARLY policy.use_min_snr_strategy=True
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os 
import torch
import datetime
import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

def ddp_setup():
    os.environ["NCCL_P2P_LEVEL"] = "NVL"
    init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=5400))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    ddp_setup()
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()
    destroy_process_group()

if __name__ == "__main__":
    main()
