import datetime
import os
import pathlib
import random

import hydra
import numpy as np
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from termcolor import cprint
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from ptv3.highlevel_ptv3 import HighlevelPTv3
from test_PointNet2.dataset_from_disk import get_dataset_from_pickle

OmegaConf.register_new_resolver("eval", eval, replace=True)


def ddp_setup():
    os.environ["NCCL_P2P_LEVEL"] = "NVL"
    init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=5400))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


class TrainHighlevelPTv3Workspace:
    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg
        self.rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self._setup_model()
        self._setup_optimizer()

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def _setup_model(self):
        model: HighlevelPTv3 = hydra.utils.instantiate(self.cfg.model)
        model = model.to(device=torch.device(self.rank))
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        cprint(f"Num params: {n_parameters}", "blue")
        if "LOCAL_RANK" in os.environ:
            model = DDP(model, device_ids=[self.rank])
        self.model = model

    def _setup_dataloader(self):
        beg_ratio = self.cfg.data_beg_ratio
        end_ratio = self.cfg.data_end_ratio
        val_ratio = self.cfg.data_val_ratio * (end_ratio - beg_ratio)
        dataset = get_dataset_from_pickle(
            beg_ratio=beg_ratio,
            end_ratio=end_ratio - val_ratio,
            **self.cfg.dataset,
        )
        dataset_val = get_dataset_from_pickle(
            beg_ratio=end_ratio - val_ratio,
            end_ratio=end_ratio,
            **self.cfg.dataset,
        )
        self.dataloader = DataLoader(
            dataset,
            sampler=DistributedSampler(dataset),
            **self.cfg.dataloader,
        )
        self.dataloader_val = DataLoader(
            dataset_val,
            sampler=DistributedSampler(dataset_val),
            **self.cfg.dataloader,
        )

    def _setup_scheduler(self):
        ctn = OmegaConf.to_container(self.cfg.scheduler)
        s_type = ""
        s_kwargs = {}
        for k, v in ctn.items():
            if k == "type":
                s_type = v
            else:
                s_kwargs[k] = v
        self.scheduler: torch.optim.lr_scheduler.LRScheduler = None
        if s_type == "OneCycleLR":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                steps_per_epoch=len(self.dataloader),
                epochs=self.cfg.epoch,
                optimizer=self.optimizer,
                **s_kwargs,
            )
        else:
            raise ValueError(f"Unsupported Scheduler Type {s_type}")

    def _setup_optimizer(self):
        # process different configs for params
        if self.cfg.param_dicts is None:
            params = self.model.parameters()
        else:
            params = [dict(names=[], params=[], lr=self.cfg.optimizer.lr)]
            for i in range(len(self.cfg.param_dicts)):
                param_group = dict(names=[], params=[])
                if "lr" in self.cfg.param_dicts[i].keys():
                    param_group["lr"] = self.cfg.param_dicts[i].lr
                if "momentum" in self.cfg.param_dicts[i].keys():
                    param_group["momentum"] = self.cfg.param_dicts[i].momentum
                if "weight_decay" in self.cfg.param_dicts[i].keys():
                    param_group["weight_decay"] = self.cfg.param_dicts[i].weight_decay
                params.append(param_group)

            for n, p in self.model.named_parameters():
                flag = False
                for i in range(len(self.cfg.param_dicts)):
                    if self.cfg.param_dicts[i].keyword in n:
                        params[i + 1]["names"].append(n)
                        params[i + 1]["params"].append(p)
                        flag = True
                        break
                if not flag:
                    params[0]["names"].append(n)
                    params[0]["params"].append(p)

            for i in range(len(params)):
                param_names = params[i].pop("names")
                message = ""
                for key in params[i].keys():
                    if key != "params":
                        message += f" {key}: {params[i][key]};"
                # if self.rank == 0:
                #     cprint(
                #         f"Params Group {i+1} -{message} Params: {param_names}.",
                #         "yellow",
                #     )
        # manually build optimizer since hydra cant do this passthrough
        opt_type = ""
        opt_kwargs = {}
        for k, v in self.cfg.optimizer.items():
            if k == "type":
                opt_type = v
            else:
                opt_kwargs[k] = v
        self.optimizer: torch.optim.Optimizer = None
        if opt_type == "AdamW":
            self.optimizer = torch.optim.AdamW(params=params, **opt_kwargs)
        else:
            raise ValueError(f"Unsupported Optimizer Type {opt_type}")

    def state_dict(self):
        ret = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
        }
        if hasattr(self, "scheduler"):
            ret["scheduler"] = self.scheduler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        model_dict = state_dict["model"]
        if list(model_dict.keys())[0].startswith("module.") and not isinstance(
            self.model, DDP
        ):
            # when saving DDP but not loading DDP keys mismatch
            model_dict = {k[7:]: v for k, v in model_dict.items()}
        self.model.load_state_dict(model_dict)
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.epoch = state_dict["epoch"]
        self.global_step = state_dict["global_step"]
        if "scheduler" in state_dict and hasattr(self, "scheduler"):
            self.scheduler.load_state_dict(state_dict["scheduler"])

    def _setup_wandb(self):
        if os.environ["LOCAL_RANK"] == "0":
            self.output_dir = HydraConfig.get().runtime.output_dir
            self.wandb_run = wandb.init(
                name=self.cfg.exp_name,
                project="pointnet-weighted-displacement",
                dir=str(self.output_dir),
                config=OmegaConf.to_container(self.cfg, resolve=True),
                save_code=True,
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                }
            )

    def train(self):
        self._setup_dataloader()
        self._setup_scheduler()
        self._setup_wandb()
        for self.epoch in range(self.epoch, self.cfg.epoch):
            self.dataloader.sampler.set_epoch(self.epoch)
            self.model.train()
            self.data_iterator = enumerate(tqdm(self.dataloader))
            # self.before_epoch()
            for self.batch_idx, self.batch_data in self.data_iterator:
                self.run_step()
                self.global_step += 1
            self.after_epoch()

    def compute_loss(self, batch):
        criterion = torch.nn.functional.mse_loss
        device = torch.device(self.rank)
        args = self.cfg.training

        pointcloud, gripper_pcd, goal_gripper_pcd, _ = batch
        # inputs: B, N, 3
        # gripper_pcd: B, 4, 3
        # goal_gripper_points: B, 4, 3
        # calculate the displacement from every point to the gripper to get the labels with shape B, N, 4, 3
        gripper_points = goal_gripper_pcd

        if args.pc_with_onehot_foreground:
            # for pointcloud, we add (0) to make fg=(1,0,0) or bg=(0,1,0)
            # for gripper_pcd, we add (0,0,1) to make gripper=(0,0,1)
            pointcloud_one_hot = torch.zeros(
                pointcloud.shape[0], pointcloud.shape[1], 1
            )
            # pointcloud_one_hot[:, :, 0] = 0
            pointcloud_ = torch.cat([pointcloud, pointcloud_one_hot], dim=2)
            gripper_pcd_one_hot = torch.zeros(
                gripper_pcd.shape[0], gripper_pcd.shape[1], 3
            )
            gripper_pcd_one_hot[:, :, 2] = 1
            gripper_pcd_ = torch.cat([gripper_pcd, gripper_pcd_one_hot], dim=2)
            inputs = torch.cat([pointcloud_, gripper_pcd_], dim=1)  # B, N+4, 6
        elif args.add_one_hot_encoding:
            # for pointcloud, we add (1, 0)
            # for gripper_pcd, we add (0, 1)
            pointcloud_one_hot = torch.zeros(
                pointcloud.shape[0], pointcloud.shape[1], 2
            )
            pointcloud_one_hot[:, :, 0] = 1
            pointcloud_ = torch.cat([pointcloud, pointcloud_one_hot], dim=2)
            gripper_pcd_one_hot = torch.zeros(
                gripper_pcd.shape[0], gripper_pcd.shape[1], 2
            )
            gripper_pcd_one_hot[:, :, 1] = 1
            gripper_pcd_ = torch.cat([gripper_pcd, gripper_pcd_one_hot], dim=2)
            inputs = torch.cat([pointcloud_, gripper_pcd_], dim=1)  # B, N+4, 5
        else:
            inputs = torch.cat([pointcloud, gripper_pcd], dim=1)  # B, N+4, 3

        labels = gripper_points.unsqueeze(1) - inputs[:, :, :3].unsqueeze(2)
        B, N, _, _ = labels.shape
        labels = labels.view(B, N, -1)  # B, N, 12

        inputs, labels = inputs.to(device), labels.to(device)
        outputs = self.model(inputs)  # B, N, 13
        weights = outputs[:, :, -1]  # B, N
        outputs = outputs[:, :, :-1]  # B, N, 12
        if args.output_obj_pcd_only:
            weights = weights[:, :-4]
            outputs = outputs[:, :-4, :]
            labels = labels[:, :-4, :]
            inputs = inputs[:, :-4, :]
            N = N - 4
        displacement_loss = criterion(outputs, labels)

        # inputs = inputs.permute(0, 2, 1)
        outputs = outputs.view(B, N, 4, 3)
        outputs = outputs + inputs[:, :, :3].unsqueeze(2)  # B, N, 4, 3

        # softmax the weights
        weights = torch.nn.functional.softmax(weights, dim=1)

        # sum the displacement of the predicted gripper point cloud according to the weights
        outputs = outputs * weights.unsqueeze(-1).unsqueeze(-1)
        outputs = outputs.sum(dim=1)
        weight_loss = criterion(outputs, gripper_points.to(device))

        total_loss = displacement_loss + weight_loss * args.weight_loss_weight
        return {
            "displacement_loss": displacement_loss,
            "weight_loss": weight_loss,
            "total_loss": total_loss,
        }

    @torch.inference_mode()
    def predict_goal(self, data_dict):
        args = self.cfg.training

        pointcloud = data_dict["pointcloud"]
        gripper_pcd = data_dict["gripper_pcd"]
        device = pointcloud.device
        # inputs: B, N, 3
        # gripper_pcd: B, 4, 3
        # goal_gripper_points: B, 4, 3
        # calculate the displacement from every point to the gripper to get the labels with shape B, N, 4, 3
        # gripper_points = goal_gripper_pcd

        if args.pc_with_onehot_foreground:
            # for pointcloud, we add (0) to make fg=(1,0,0) or bg=(0,1,0)
            # for gripper_pcd, we add (0,0,1) to make gripper=(0,0,1)
            pointcloud_one_hot = torch.zeros(
                pointcloud.shape[0], pointcloud.shape[1], 1, device=device
            )
            # pointcloud_one_hot[:, :, 0] = 0
            pointcloud_ = torch.cat([pointcloud, pointcloud_one_hot], dim=2)
            gripper_pcd_one_hot = torch.zeros(
                gripper_pcd.shape[0], gripper_pcd.shape[1], 3, device=device
            )
            gripper_pcd_one_hot[:, :, 2] = 1
            gripper_pcd_ = torch.cat([gripper_pcd, gripper_pcd_one_hot], dim=2)
            inputs = torch.cat([pointcloud_, gripper_pcd_], dim=1)  # B, N+4, 6
        elif args.add_one_hot_encoding:
            # for pointcloud, we add (1, 0)
            # for gripper_pcd, we add (0, 1)
            pointcloud_one_hot = torch.zeros(
                pointcloud.shape[0], pointcloud.shape[1], 2, device=device
            )
            pointcloud_one_hot[:, :, 0] = 1
            pointcloud_ = torch.cat([pointcloud, pointcloud_one_hot], dim=2)
            gripper_pcd_one_hot = torch.zeros(
                gripper_pcd.shape[0], gripper_pcd.shape[1], 2, device=device
            )
            gripper_pcd_one_hot[:, :, 1] = 1
            gripper_pcd_ = torch.cat([gripper_pcd, gripper_pcd_one_hot], dim=2)
            inputs = torch.cat([pointcloud_, gripper_pcd_], dim=1)  # B, N+4, 5
        else:
            inputs = torch.cat([pointcloud, gripper_pcd], dim=1)  # B, N+4, 3

        inputs = inputs.to(device)
        outputs = self.model(inputs)  # B, N, 13
        weights = outputs[:, :, -1]  # B, N
        outputs = outputs[:, :, :-1]  # B, N, 12
        if args.output_obj_pcd_only:
            weights = weights[:, :-4]
            outputs = outputs[:, :-4, :]
            inputs = inputs[:, :-4, :]
            N = N - 4
        # displacement_loss = criterion(outputs, labels)

        # inputs = inputs.permute(0, 2, 1)
        B, N, C = inputs.shape
        outputs = outputs.view(B, N, 4, 3)
        outputs = outputs + inputs[:, :, :3].unsqueeze(2)  # B, N, 4, 3

        # softmax the weights
        weights = torch.nn.functional.softmax(weights, dim=1)

        # sum the displacement of the predicted gripper point cloud according to the weights
        outputs = outputs * weights.unsqueeze(-1).unsqueeze(-1)
        outputs = outputs.sum(dim=1)

        return {"weights": weights, "predicted_goal": outputs}

    def run_step(self):
        self.optimizer.zero_grad()
        loss_dict = self.compute_loss(self.batch_data)
        loss_dict["total_loss"].backward()
        self.optimizer.step()
        self.scheduler.step()
        # logging
        self.log_info = {
            "epoch": self.epoch + 1,
            "global_step": self.global_step,
            "lr": self.scheduler.get_last_lr()[0],
        }
        for k, v in loss_dict.items():
            self.log_info[k] = v.item()

        if (
            (self.batch_idx + 1) % self.cfg.training.log_every_batches == 0
            and (self.batch_idx + 1) != len(self.dataloader)
            and os.environ["LOCAL_RANK"] == "0"
        ):
            tqdm.write(
                f"Epoch {self.epoch + 1}, iter {self.batch_idx + 1}, loss: {loss_dict['total_loss'].item()}"
            )

            self.wandb_run.log(self.log_info, step=self.global_step)

    def after_epoch(self):
        # val
        self.model.eval()
        val_dicts = []
        with torch.no_grad():
            for val_batch in tqdm(self.dataloader_val):
                loss_dict = self.compute_loss(self.batch_data)
                val_dicts.append(loss_dict)
        self.model.train()
        for k in val_dicts[0].keys():
            kname = "val_" + k
            kval = np.mean([d[k].item() for d in val_dicts])
            self.log_info[kname] = kval
        if os.environ["LOCAL_RANK"] == "0":
            print(
                f"Epoch {self.epoch + 1}, val_loss: {self.log_info['val_total_loss']}"
            )
            self.wandb_run.log(self.log_info, step=self.global_step)

        if (
            self.epoch + 1
        ) % self.cfg.training.checkpoint_every_epoch == 0 and os.environ[
            "LOCAL_RANK"
        ] == "0":
            save_path = f"{self.output_dir}/model_{self.epoch + 1}.pth"
            torch.save(self.state_dict(), save_path)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("configs")),
    config_name="highlevel_ptv3",
)
def main(cfg):
    ddp_setup()
    workspace = TrainHighlevelPTv3Workspace(cfg)
    workspace.train()
    destroy_process_group()


if __name__ == "__main__":
    main()
