"""Main script for trajectory optimization."""

import io
import os
from pathlib import Path
import random
from typing import Tuple, Optional

# import cv2
from matplotlib import pyplot as plt
import numpy as np
import tap
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.data import DataLoader, default_collate
from torch.utils.data.distributed import DistributedSampler
from main_trajectory import TrajectoryCriterion

# from datasets.dataset_engine import RLBenchDataset
from datasets.dataset_robogen import RobogenDataset

from engine import BaseTrainTester
from diffuser_actor import DiffuserActor
from termcolor import cprint

from utils.common_utils import (
    load_instructions, count_parameters, get_gripper_loc_bounds
)
from robogen_utils import get_gripper_pos_orient_from_4_points, get_4_points_from_gripper_pos_orient
from torch.distributed import init_process_group


class Arguments(tap.Tap):
    cameras: Tuple[str, ...] = ("left_shoulder", "right_shoulder")
    image_size: str = "256,256"
    max_episodes_per_task: int = 100
    instructions: Optional[Path] = None
    seed: int = 0
    tasks: Tuple[str, ...]
    variations: Tuple[int, ...] = (0,)
    checkpoint: Optional[Path] = None
    accumulate_grad_batches: int = 1
    val_freq: int = 500
    gripper_loc_bounds: Optional[str] = None
    gripper_loc_bounds_buffer: float = 0.04
    eval_only: int = 0

    # Training and validation datasets
    dataset: Path
    valset: Path = None
    dense_interpolation: int = 0
    interpolation_length: int = 100

    # Logging to base_log_dir/exp_log_dir/run_log_dir
    base_log_dir: Path = Path(__file__).parent / "train_logs"
    exp_log_dir: str = "exp"
    run_log_dir: str = "run"
    log_dir: str = "debug"

    # Main training parameters
    num_workers: int = 1
    batch_size: int = 16
    batch_size_val: int = 4
    cache_size: int = 100
    cache_size_val: int = 100
    lr: float = 1e-4
    wd: float = 5e-3  # used only for CALVIN
    train_iters: int = 200_000
    val_iters: int = -1  # -1 means heuristically-defined
    max_episode_length: int = 5  # -1 for no limit

    # Data augmentations
    image_rescale: str = "0.75,1.25"  # (min, max), "1.0,1.0" for no rescaling

    # Model
    backbone: str = "clip"  # one of "resnet", "clip"
    embedding_dim: int = 120
    num_vis_ins_attn_layers: int = 2
    use_instruction: int = 0
    rotation_parametrization: str = 'quat'
    quaternion_format: str = 'xyzw'
    diffusion_timesteps: int = 100
    keypose_only: int = 0
    num_history: int = 0
    relative_action: int = 0
    lang_enhanced: int = 0
    fps_subsampling_factor: int = 5
    num_load_episodes: int = 1000000
    local_rank: int = 0

class TrainTester(BaseTrainTester):
    def __init__(self, args):
        super().__init__(args)

    def get_datasets(self):
        cprint("Using Robogen Dataset, no instructions", "yellow")
        cprint("Notice Here, you need to modify the code for the dataset!!!", "red")
        train_dataset = RobogenDataset(
            root=self.args.dataset, 
            max_episode_length=self.args.max_episode_length,
            cache_size=self.args.cache_size,
            max_episodes_per_task=self.args.max_episodes_per_task,
            num_iters=self.args.train_iters,
            cameras=self.args.cameras,
            training=True,
            image_rescale=tuple(
                float(x) for x in self.args.image_rescale.split(",")
            ),
            return_low_lvl_trajectory=True,
            dense_interpolation=bool(self.args.dense_interpolation),
            interpolation_length=self.args.interpolation_length, 
            start_episode_idx=0,
            end_episode_idx=args.num_load_episodes,
        )
        test_dataset = RobogenDataset(
            root=self.args.dataset, 
            max_episode_length=self.args.max_episode_length,
            cache_size=self.args.cache_size,
            max_episodes_per_task=self.args.max_episodes_per_task,
            cameras=self.args.cameras,
            training=False,
            image_rescale=tuple(
                float(x) for x in self.args.image_rescale.split(",")
            ),
            return_low_lvl_trajectory=True,
            dense_interpolation=bool(self.args.dense_interpolation),
            interpolation_length=self.args.interpolation_length, 
            start_episode_idx=args.num_load_episodes,
        )
        return train_dataset, test_dataset
    
    def get_model(self):
        _model = DiffuserActor(
            backbone="robogen_resnet18",
            image_size=tuple(int(x) for x in self.args.image_size.split(",")),
            embedding_dim=self.args.embedding_dim,
            num_vis_ins_attn_layers=self.args.num_vis_ins_attn_layers,
            use_instruction=bool(self.args.use_instruction),
            fps_subsampling_factor=self.args.fps_subsampling_factor,
            gripper_loc_bounds=self.args.gripper_loc_bounds,
            rotation_parametrization=self.args.rotation_parametrization,
            quaternion_format=self.args.quaternion_format,
            diffusion_timesteps=self.args.diffusion_timesteps,
            nhist=self.args.num_history,
            relative=bool(self.args.relative_action),
            lang_enhanced=bool(self.args.lang_enhanced)
        )
        print(f"Model has {count_parameters(_model)} parameters.")
        return _model
    

    def get_loaders(self, collate_fn=default_collate):
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            np.random.seed(np.random.get_state()[1][0] + worker_id)
        # Datasets
        train_dataset, test_dataset = self.get_datasets()
        g = torch.Generator()
        g.manual_seed(0)
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            worker_init_fn=seed_worker,
            collate_fn=collate_fn,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,
            generator=g
        )
        test_sampler = DistributedSampler(test_dataset, shuffle=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            worker_init_fn=seed_worker,
            collate_fn=collate_fn,
            pin_memory=True,
            sampler=test_sampler,
            drop_last=True,
            generator=g
        )
        return train_loader, test_loader

    @staticmethod
    def get_criterion():
        return TrajectoryCriterion()
    
    def train_one_step(self, model, criterion, optimizer, step_id, sample):
        # for key, value in sample.items():
        #     if key != 'task':
        #         cprint(f"{key}: {value.shape}", "red")
        import time 
        start = time.time()

        if step_id % self.args.accumulate_grad_batches == 0:
            optimizer.zero_grad()

        if self.args.keypose_only:
            sample["trajectory"] = sample["trajectory"][:, [-1]] # 10, 1, 8
            sample["trajectory_mask"] = sample["trajectory_mask"][:, [-1]] # 10, 1
        else:
            sample["trajectory"] = sample["trajectory"][:, 1:]
            sample["trajectory_mask"] = sample["trajectory_mask"][:, 1:]

        # Forward pass
        curr_gripper = (
            sample["curr_gripper"] if self.args.num_history < 1
            else sample["curr_gripper_history"][:, -self.args.num_history:]
        ) # 10, 3, 8
        out = model(
            sample["trajectory"],
            sample["trajectory_mask"],
            sample["rgbs"],
            sample["pcds"],
            sample["instr"],
            curr_gripper
        )
        loss = criterion.compute_loss(out)
        loss.backward()

        # Update
        if step_id % self.args.accumulate_grad_batches == self.args.accumulate_grad_batches - 1:
            optimizer.step()

        # Log
        if dist.get_rank() == 0 and (step_id + 1) % self.args.val_freq == 0:
            self.writer.add_scalar("lr", self.args.lr, step_id)
            self.writer.add_scalar("train-loss/noise_mse", loss, step_id)

        end = time.time()
        # cprint(f"Step {step_id} took {end - start} seconds", "yellow")

    @torch.no_grad()
    def evaluate_nsteps(self, model, criterion, loader, step_id, val_iters,
                        split='val'):
        # return None
        val_iters = self.args.val_iters
        if split == 'train':
            val_iters = (val_iters // self.args.batch_size) + 1

        save_dir = self.args.log_dir / f"viz_{step_id}"
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        
        device = next(model.parameters()).device
        model.eval()
        for i, sample in enumerate(loader):
            # for key, value in sample.items():
            #     if key != 'task':
            #         cprint(f"{key}: {value.shape}", "red")
            if i == val_iters:
                break

            if self.args.keypose_only:
                sample["trajectory"] = sample["trajectory"][:, [-1]]
                sample["trajectory_mask"] = sample["trajectory_mask"][:, [-1]]
            else:
                sample["trajectory"] = sample["trajectory"][:, 1:]
                sample["trajectory_mask"] = sample["trajectory_mask"][:, 1:]

            curr_gripper = (
                sample["curr_gripper"] if self.args.num_history < 1
                else sample["curr_gripper_history"][:, -self.args.num_history:]
            )

            action = model(
                sample["trajectory"].to(device), # 10, 1, 8
                sample["trajectory_mask"].to(device), # 10, 1
                sample["rgbs"].to(device),
                sample["pcds"].to(device), # 10, 4, 3, 256, 256
                sample["instr"].to(device),
                curr_gripper.to(device), # 10, 3, 8
                run_inference=True
            )
            for j in range(action.shape[0]):
                pcd = sample['pcds'][j]
                pcd = pcd.permute(0, 2, 3, 1).detach().cpu().numpy()
                pcd = pcd.reshape(-1, 3)
                action_ = action[j][0].detach().cpu().numpy()
                curr_gripper_ = curr_gripper[j][0].detach().cpu().numpy()
                curr_gripper_pcd = get_4_points_from_gripper_pos_orient(curr_gripper_[:3], curr_gripper_[3:7])
                action_pcd = get_4_points_from_gripper_pos_orient(action_[:3], action_[3:7])
                # visualize those points
                fig = plt.figure(figsize=(10, 10))
                ax = plt.axes(projection='3d')
                ax.scatter3D(
                    pcd[:, 0], pcd[:, 1], pcd[:, 2],
                    color='blue', label='pcd', s=0.1
                )
                ax.scatter3D(
                    curr_gripper_pcd[:, 0], curr_gripper_pcd[:, 1], curr_gripper_pcd[:, 2],
                    color='red', label='curr_gripper'
                )
                ax.scatter3D(
                    action_pcd[:, 0], action_pcd[:, 1], action_pcd[:, 2],
                    color='green', label='action'
                )

                ax.set_xlim(0.5, 2)
                ax.set_ylim(-0.5, 0.5)
                ax.set_zlim(0, 1)
                ax.view_init(elev=30, azim=120)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_zticklabels([])
                plt.legend()
                fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
                # plt.savefig(f"output_{split}_{i}_{j}.png")
                plt.savefig(save_dir / f"output_{split}_{i}_{j}.png")
                plt.close()
            # for j in range(sample['pcds'].shape[0]):
            #     plt.imshow(sample['pcds'][j][0].permute(1, 2, 0).detach().cpu().numpy())
            #     plt.savefig(f"pcd_{split}_{i}_{j}.png")
            #     plt.close()

        model.train()
        return None



def traj_collate_fn(batch):
    keys = [
        "trajectory", "trajectory_mask",
        "rgbs", "pcds",
        "curr_gripper", "curr_gripper_history", "action", "instr"
    ]
    ret_dict = {
        key: torch.cat([
            item[key].float() if key != 'trajectory_mask' else item[key]
            for item in batch
        ]) for key in keys
    }

    ret_dict["task"] = []
    for item in batch:
        ret_dict["task"] += item['task']
    return ret_dict

def fig_to_numpy(fig, dpi=60):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    return img

def generate_visualizations(pred, gt, mask, box_size=0.3):
    batch_idx = 0
    pred = pred[batch_idx].detach().cpu().numpy()
    gt = gt[batch_idx].detach().cpu().numpy()
    mask = mask[batch_idx].detach().cpu().numpy()

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.scatter3D(
        pred[~mask][:, 0], pred[~mask][:, 1], pred[~mask][:, 2],
        color='red', label='pred'
    )
    ax.scatter3D(
        gt[~mask][:, 0], gt[~mask][:, 1], gt[~mask][:, 2],
        color='blue', label='gt'
    )

    center = gt[~mask].mean(0)
    ax.set_xlim(center[0] - box_size, center[0] + box_size)
    ax.set_ylim(center[1] - box_size, center[1] + box_size)
    ax.set_zlim(center[2] - box_size, center[2] + box_size)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    plt.legend()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    img = fig_to_numpy(fig, dpi=120)
    plt.close()
    return img.transpose(2, 0, 1)

def ddp_setup():
    os.environ["NCCL_P2P_LEVEL"] = "NVL"
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = Arguments().parse_args()
    print("Arguments:")
    print(args)
    print("-" * 100)
    if args.gripper_loc_bounds is None:
        args.gripper_loc_bounds = np.array([[-5, -5, -5], [5, 5, 5]]) * 1.0
    log_dir = args.base_log_dir / args.exp_log_dir / args.run_log_dir
    args.log_dir = log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    print("Logging to", log_dir)
    print(
        "Available devices (CUDA_VISIBLE_DEVICES):",
        os.environ.get("CUDA_VISIBLE_DEVICES")
    )

    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.cuda.set_device(args.local_rank)
    # torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    ddp_setup()
    


    train_tester = TrainTester(args)
    train_tester.main(collate_fn=traj_collate_fn)