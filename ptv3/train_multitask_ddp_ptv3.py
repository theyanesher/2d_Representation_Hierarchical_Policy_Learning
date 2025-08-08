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

### cgn imports
from test_PointNet2.cgn.acronym_dataloader import AcryonymDataset
from test_PointNet2.cgn import utils as cgn_utils
from test_PointNet2.cgn.cgn_loss import ContactGraspnetLoss

OmegaConf.register_new_resolver("eval", eval, replace=True)


def ddp_setup():
    os.environ["NCCL_P2P_LEVEL"] = "NVL"
    init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=5400))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def infinite_loader(dl):
    while True:
        for batch in dl:
            yield batch

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
        if self.cfg.general.load_model_path is not None:
            state_dict = torch.load(self.cfg.general.load_model_path)['model']
            model.load_state_dict(state_dict)
            print("loading pretrained model from {}".format(self.cfg.general.load_model_path))
            
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        cprint(f"Num params: {n_parameters}", "blue")
        if "LOCAL_RANK" in os.environ:
            model = DDP(model, device_ids=[self.rank])
        self.model = model
        

    def _setup_articubot_dataloader(self, articbot_cfg):
        beg_ratio = articbot_cfg.data_beg_ratio
        end_ratio = articbot_cfg.data_end_ratio
        val_ratio = articbot_cfg.data_val_ratio * (end_ratio - beg_ratio)
        
        ### TODO: setup different dataloader for different tasks here
        dataset = get_dataset_from_pickle(
            beg_ratio=beg_ratio,
            end_ratio=end_ratio - val_ratio,
            **articbot_cfg.dataset,
        )
        # dataset_val = get_dataset_from_pickle(
        #     beg_ratio=end_ratio - val_ratio,
        #     end_ratio=end_ratio,
        #     **self.cfg.dataset,
        # )
        self.articubot_dataloader = DataLoader(
            dataset,
            sampler=DistributedSampler(dataset),
            **articbot_cfg.dataloader,
        )
        # self.dataloader_val = DataLoader(
        #     dataset_val,
        #     sampler=DistributedSampler(dataset_val),
        #     **self.cfg.dataloader,
        # )
        
    def _setup_cgn_dataloader(self, global_config):
        batch_size = global_config['OPTIMIZER']['batch_size']
        num_workers = 6  # Increase after debug
        device = torch.device(self.rank)
        train_dataset = AcryonymDataset(global_config, train=True, device=device, use_saved_renders=True)
        # test_dataset = AcryonymDataset(global_config, train=False, device=device, use_saved_renders=True)

        self.cgn_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    num_workers=num_workers,
                                                    sampler=DistributedSampler(train_dataset)
                                                    )
        # test_dataloader = torch.utils.data.DataLoader(test_dataset,
        #                                                 batch_size=batch_size,
        #                                                 shuffle=False,
        #                                                 num_workers=num_workers,
        #                                                 sampler=DistributedSampler(test_dataset)
        #                                                 )
        

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
                steps_per_epoch=sum([len(dataloader) for dataloader in self.dataloaders]),
                epochs=self.cfg.general.num_iterations // len(self.articubot_dataloader),
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
                name=self.cfg.general.exp_name,
                project="articubot_multitask",
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
        self._setup_articubot_dataloader(self.cfg.articubot)
        self._setup_cgn_dataloader(self.cfg.cgn)
        self.dataloaders = [self.articubot_dataloader, self.cgn_dataloader]
        self._setup_scheduler()
        self._setup_wandb()
        self.model.train()
        
        ### setup all dataloaders
        all_tasks = self.cfg.general.tasks
        
        
        all_task_dataloaders = {
            "articubot": self.articubot_dataloader,
            "cgn": self.cgn_dataloader, 
        }
        all_dataloaders = [all_task_dataloaders[task] for task in all_tasks]
        all_dataloaders = [x for x in all_dataloaders if x is not None]
        dataloader_iters = [infinite_loader(loader) for loader in all_dataloaders]
        
        forward_functions = {
            "articubot": self.compute_articubot_loss,
            "cgn": self.compute_cgn_loss,  
        }
        
        args = self.cfg
        general_args = self.cfg.general
        train_frequency = {
            "articubot": args.articubot.train_frequency,
            'cgn': args.cgn.train_frequency,
        }
        
        if general_args.category_embedding_type == "siglip":
            self.siglip_text_features = torch.load("../siglip_text_features.pt")
        else:
            self.siglip_text_features = None
        
        device = torch.device(self.rank)
        self.cgn_loss_fn = ContactGraspnetLoss(args.cgn, device).to(device) if 'cgn' in args.general.tasks else None
        
        num_iterations = args.general.num_iterations
        for global_step in range(num_iterations):  
            
            samples = [next(it) for it in dataloader_iters]
            all_logs = {}
            for task_idx in range(len(all_tasks)):
                task = all_tasks[task_idx]
                forward_func = forward_functions[all_tasks[task_idx]]
                if global_step % train_frequency[task] == 0:
                    log = forward_func(samples[task_idx])
                    for key in log:
                        assert not torch.is_tensor(log[key])
                        all_logs[f"{task}_{key}"] = log[key]
            
            if os.environ['LOCAL_RANK'] == '0':
                ### TODO: log the losses here
                for task in all_tasks:
                    dataloader_length = len(all_task_dataloaders[task])
                    epoch = global_step / dataloader_length
                    all_logs[f"{task}_epoch"] = epoch
                    
                self.wandb_run.log(all_logs, step=global_step)
                
                print(f"{global_step} {all_logs}")
                
                ### TODO: save the model here
                if (global_step + 1) % args.general.save_freq == 0:
                    save_path = f"{self.output_dir}/model_{global_step + 1}.pth"
                    save_dict = {
                        "model": self.model.module.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        
                    }
                    torch.save(save_dict, save_path)
            
            torch.cuda.empty_cache()        
            torch.cuda.ipc_collect()      
        
    def compute_cgn_loss(self, data):    
        device = torch.device(self.rank)
        loss_fn = self.cgn_loss_fn
        siglip_features = self.siglip_text_features
        model = self.model
        global_config = self.cfg['cgn']
        optimizer = self.optimizer
        
        cgn_utils.send_dict_to_device(data, device)
        # Target contains input and target values
        pc_cam = data['pc_cam']
        # import pdb; pdb.set_trace()
        # pc_cam = pc_cam.permute(0, 2, 1)

        if siglip_features is None:
            pred = model(pc_cam)
        else:
            embedding = siglip_features[-1].float().unsqueeze(0).repeat(pc_cam.shape[0], 1)
            # pred = model(pc_cam, embedding)
            # import pdb; pdb.set_trace()
            pred = model(pc_cam, embedding)
            
        loss, loss_info = loss_fn(pred, data)
        loss = loss * global_config.loss_scale
        keys = list(loss_info.keys())
        for key in keys:
            loss_info[key + "_scaled"] = loss_info[key] * global_config.loss_scale
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss_info

    def compute_articubot_loss(self, batch):
        criterion = torch.nn.functional.mse_loss
        device = torch.device(self.rank)
        args = self.cfg.articubot

        pointcloud, gripper_pcd, goal_gripper_pcd, cat_idx, class_weight = batch
        # inputs: B, N, 3
        # gripper_pcd: B, 4, 3
        # goal_gripper_points: B, 4, 3
        # calculate the displacement from every point to the gripper to get the labels with shape B, N, 4, 3
        gripper_points = goal_gripper_pcd.to(device)

        if args.add_one_hot_encoding:
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

        B, N, _ = inputs.shape

        inputs = inputs.to(device)
        if self.siglip_text_features is not None:
            cat_embedding = self.siglip_text_features[cat_idx].float()
            # outputs = self.model(inputs, embedding=cat_embedding)  # B, N, 13
            # import pdb; pdb.set_trace()
            pred_dict = self.model(inputs, cat_embedding)  # B, N, 13
        else:
            pred_dict = self.model(inputs)  # B, N, 13
            
        # weights = outputs[:, :, -1]  # B, N
        # outputs = outputs[:, :, :-1]  # B, N, 12
        
        B, N, _, _ = pred_dict['pred_offsets'].shape
        outputs = pred_dict['pred_offsets'].view(B, N, -1)
        pred_points = pred_dict['pred_points'] 
        weights = pred_dict['pred_scores'].squeeze(-1)
        
        labels = gripper_points.unsqueeze(1) - pred_points.unsqueeze(2)
        labels = labels.view(B, N, -1)  # B, N, 12
        
        if args.output_obj_pcd_only:
            weights = weights[:, :-4]
            outputs = outputs[:, :-4, :]
            labels = labels[:, :-4, :]
            inputs = inputs[:, :-4, :]
            N = N - 4
        displacement_loss = criterion(outputs, labels)

        # inputs = inputs.permute(0, 2, 1)
        outputs = outputs.view(B, N, 4, 3)
        outputs = outputs + pred_points[:, :, :3].unsqueeze(2)  # B, N, 4, 3

        # softmax the weights
        weights = torch.nn.functional.softmax(weights, dim=1)

        # sum the displacement of the predicted gripper point cloud according to the weights
        outputs = outputs * weights.unsqueeze(-1).unsqueeze(-1)
        outputs = outputs.sum(dim=1)
        weight_loss = criterion(outputs, gripper_points.to(device))

        total_loss = displacement_loss + weight_loss * args.weight_loss_weight
        
        total_loss = total_loss * args.loss_scale
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "perpoint_loss": displacement_loss.item(),
            "weighted_average_loss": weight_loss.item(),
            "loss": total_loss.item(),
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


    def after_epoch(self):
        # val
        self.model.eval()
        val_dicts = []
        with torch.no_grad():
            for val_batch in tqdm(self.dataloader_val):
                loss_dict = self.compute_articubot_loss(self.batch_data)
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
    config_name="multitask_ptv3",
)
def main(cfg):
    ddp_setup()
    workspace = TrainHighlevelPTv3Workspace(cfg)
    workspace.train()
    destroy_process_group()


if __name__ == "__main__":
    main()
