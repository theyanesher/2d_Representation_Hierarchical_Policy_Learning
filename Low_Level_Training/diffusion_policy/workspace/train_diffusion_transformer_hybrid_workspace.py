if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import DiffusionTransformerHybridImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)



import torch.distributed as dist
import wandb
import tempfile


# def init_wandb_safe(cfg, output_dir):

#     # force safe temp dir
#     safe_tmp = os.path.join(str(output_dir), "wandb_tmp")
#     os.makedirs(safe_tmp, exist_ok=True)
#     os.environ["WANDB_DIR"] = safe_tmp
#     os.environ["TMPDIR"] = safe_tmp
#     tempfile.tempdir = safe_tmp

#     # determine rank
#     rank = 0
#     try:
#         if dist.is_available() and dist.is_initialized():
#             rank = dist.get_rank()
#     except Exception:
#         rank = 0

#     if rank != 0:
#         return None

#     # Only rank 0 runs wandb
#     try:
#         run = wandb.init(
#             dir=str(output_dir),
#             config=OmegaConf.to_container(cfg, resolve=True),
#             **cfg.logging
#         )
#         wandb.config.update({"output_dir": output_dir})
#         return run
#     except Exception as e:
#         print("W&B init failed on rank 0:", e)
#         print("Falling back to WANDB_MODE=offline")
#         os.environ["WANDB_MODE"] = "offline"
#         try:
#             run = wandb.init(
#                 dir=str(output_dir),
#                 config=OmegaConf.to_container(cfg, resolve=True),
#                 **cfg.logging
#             )
#             return run
#         except Exception as e2:
#             print("W&B still failed; continuing without wandb. Error:", e2)
#             return None
        

def init_wandb_safe(cfg, output_dir):
    import time

    os.makedirs(output_dir, exist_ok=True)  # ensure permanent logs exist

    # Local temp dir
    local_tmp = os.path.join("/tmp", f"wandb_tmp_{int(time.time())}")
    os.makedirs(local_tmp, exist_ok=True)
    os.environ["WANDB_DIR"] = local_tmp
    os.environ["TMPDIR"] = local_tmp
    tempfile.tempdir = local_tmp

    # Determine rank
    rank = 0
    try:
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
    except Exception:
        rank = 0

    if rank != 0:
        return None

    # Init W&B
    try:
        run = wandb.init(
            dir=str(output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update({"output_dir": output_dir})
        return run
    except Exception as e:
        print("W&B init failed on rank 0:", e)
        os.environ["WANDB_MODE"] = "offline"
        try:
            run = wandb.init(
                dir=str(output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            return run
        except Exception as e2:
            print("W&B still failed; continuing without wandb. Error:", e2)
            return None


        
# def init_wandb_safe(cfg, output_dir):
#     # determine rank (works if dist initialized)
#     rank = 0
#     try:
#         if dist.is_available() and dist.is_initialized():
#             rank = dist.get_rank()
#     except Exception:
#         rank = 0

#     if rank != 0:
#         return None

#     # Only rank 0 reaches here
#     try:
#         run = wandb.init(
#             dir=str(output_dir),
#             config=OmegaConf.to_container(cfg, resolve=True),
#             **cfg.logging
#         )
#         wandb.config.update({"output_dir": output_dir})
#         return run
#     except Exception as e:
#         # fallback: run offline and continue
#         print("W&B init failed on rank 0: ", e)
#         print("Falling back to WANDB_MODE=offline")
#         os.environ["WANDB_MODE"] = "offline"
#         try:
#             run = wandb.init(
#                 dir=str(output_dir),
#                 config=OmegaConf.to_container(cfg, resolve=True),
#                 **cfg.logging
#             )
#             return run
#         except Exception as e2:
#             print("W&B still failed; continuing without wandb. Error:", e2)
#             return None



class TrainDiffusionTransformerHybridWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir="Primary_Training_Outputs"): # None
        super().__init__(cfg, output_dir=output_dir)
        self.no_validation = True
        # set seed  
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionTransformerHybridImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionTransformerHybridImagePolicy = None
        # if cfg.training.use_ema:
        #     self.ema_model = copy.deepcopy(self.model)
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except: # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)

        

        # configure training state
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        # configure training state
        self.global_step = 0
        self.epoch = 0




    def run(self, rank, world_size):
        import sys, torch, tqdm, os, contextlib, copy
        import torch.distributed as dist
        from torch.utils.data import DataLoader, DistributedSampler
        from diffusion_policy.common.pytorch_util import optimizer_to, dict_apply
        import numpy as np
        import hydra

        cfg = copy.deepcopy(self.cfg)
        self.rank = rank
        self.world_size = world_size

        # ---------------- DEVICE SETUP ----------------
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        # ---------------- RESUME ----------------
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                if rank == 0:
                    print(f"Resuming from checkpoint {lastest_ckpt_path}")
                    # import pdb; pdb.set_trace();
                self.load_checkpoint(path=lastest_ckpt_path)

        # ---------------- DATASET ----------------
        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)

        train_sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        ) if world_size > 1 else None

        train_dataloader = DataLoader(
            dataset,
            sampler=train_sampler,
            batch_size=cfg.dataloader.batch_size,
            shuffle=False,
            num_workers=cfg.dataloader.num_workers,
            pin_memory=cfg.dataloader.pin_memory,
            persistent_workers=cfg.dataloader.persistent_workers,
            drop_last=True
        )

        val_dataloader = None
        if not self.no_validation:
            val_dataset = dataset.get_validation_dataset()
            val_sampler = DistributedSampler(
                val_dataset, num_replicas=world_size, rank=rank, shuffle=False
            ) if world_size > 1 else None
            val_dataloader = DataLoader(
                val_dataset,
                sampler=val_sampler,
                batch_size=cfg.val_dataloader.batch_size,
                shuffle=False,
                num_workers=cfg.val_dataloader.num_workers,
                pin_memory=cfg.val_dataloader.pin_memory,
                persistent_workers=cfg.val_dataloader.persistent_workers,
                drop_last=False
            )

        # ---------------- NORMALIZER ----------------
        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema and self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)


        # ---------------- SCHEDULER ----------------
        total_steps = (len(train_dataloader) * cfg.training.num_epochs) // cfg.training.gradient_accumulate_every
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=total_steps,
            last_epoch=self.global_step - 1
        )

        # ema = None
        # if cfg.training.use_ema:
        #     ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)
            

        # ---------------- MODEL & DDP ----------------
        self.model.to(device)

        # if cfg.training.use_ema:
        #     self.ema_model = copy.deepcopy(self.model)

        
        if self.ema_model is not None:
            self.ema_model.to(device)

        # import pdb; pdb.set_trace();
        if world_size > 1:
            print(f"[Rank {self.rank}] wrapping model with DDP")
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.rank], output_device=self.rank, find_unused_parameters=False
            )

        optimizer_to(self.optimizer, device)

        

        wandb_run = init_wandb_safe(cfg, self.output_dir) if rank == 0 else None
        log_path = os.path.join(self.output_dir, "logs.json.txt") if rank == 0 else None
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"), **cfg.checkpoint.topk
        )

        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        if rank == 0:
            print("Starting training loop...")
            sys.stdout.flush()

        # ---------------- TRAIN LOOP ----------------
        with JsonLogger(log_path) if rank == 0 else contextlib.nullcontext() as json_logger:
            mse = None
            mse_trans = None
            mse_quat = None
            for epoch_idx in range(cfg.training.num_epochs):
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch_idx)

                tepoch = (
                    tqdm.tqdm(train_dataloader, desc=f"Training epoch {epoch_idx}", leave=False,
                            mininterval=cfg.training.tqdm_interval_sec)
                    if rank == 0 else train_dataloader
                )

                for batch_idx, batch in enumerate(tepoch):
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch
                    if self.world_size > 1:
                        raw_loss = self.model.module.compute_loss(batch) #self.model.compute_loss(batch)
                    else:
                        raw_loss = self.model.compute_loss(batch) 
                    if not torch.is_tensor(raw_loss):
                        raw_loss = torch.tensor(raw_loss, device=device)
                    raw_loss = raw_loss.to(device)

                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()

                    if (self.global_step % cfg.training.gradient_accumulate_every) == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()
                    # import pdb; pdb.set_trace()
                    # if not dist.is_initialized() or dist.get_rank() == 1:
                    #     import pdb; pdb.set_trace()
                    # print("RANKKKKKKKKKKKKKKKKKKKKKKKK", dist.get_rank())
                    if cfg.training.use_ema:
                        if self.world_size > 1:
                            ema.step(self.model.module)
                        else:
                            ema.step(self.model)

                    # sync loss across ranks
                    if world_size > 1:
                        with torch.no_grad():
                            loss_tensor = raw_loss.detach().clone().to(device)
                            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                            loss_tensor /= float(world_size)
                            logged_loss = loss_tensor
                    else:
                        logged_loss = raw_loss.detach()
                    is_last_batch = (batch_idx == (len(train_dataloader)-1))
                    self.global_step += 1
                    if (self.epoch % cfg.training.sample_every) == 0: #and is_last_batch:
                        with torch.no_grad():
                            # sample trajectory from training set, and evaluate difference
                            batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                            # import pdb; pdb.set_trace();
                            obs_dict = batch['obs']#{'obs': batch['obs']}
                            gt_action = batch['action']
                            
                            result = self.ema_model.predict_action(obs_dict)
                            # import pdb; pdb.set_trace();
                            if cfg.pred_action_steps_only:
                                pred_action = result['action']
                                start = cfg.n_obs_steps - 1
                                end = start + cfg.n_action_steps
                                gt_action = gt_action[:,start:end]
                            else:
                                pred_action = result['action_pred']
                            mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                            import pdb; pdb.set_trace();
                            mse_trans = torch.nn.functional.mse_loss(pred_action[:,:,:3], gt_action[:,:,:3])
                            mse_quat = torch.nn.functional.mse_loss(pred_action[:,:,3:7], gt_action[:,:,3:7])
                            # step_log['train_action_mse_error'] = mse.item()
                            del batch
                            del obs_dict
                            del gt_action
                            del result
                            del pred_action
                            # del mse
                    if rank == 0:
                        tepoch.set_postfix(loss=logged_loss.item(), refresh=False)
                        step_log = {
                            "train_loss": logged_loss.item(),
                            "global_step": self.global_step,
                            "epoch": epoch_idx,
                            "lr": lr_scheduler.get_last_lr()[0],
                            "mse": mse,
                            "mse_trans": mse_trans, 
                            "mse_quat": mse_quat
                        }
                        # if wandb_run:
                        #     wandb_run.log(step_log, step=self.global_step)
                        #     json_logger.log(step_log)

                        if rank == 0 and wandb_run is not None:
                            try:
                                wandb_run.log(step_log, step=self.global_step)
                                json_logger.log(step_log)
                            except Exception as e:
                                print("Warning: W&B log failed:", e)

                    if cfg.training.max_train_steps is not None and batch_idx >= cfg.training.max_train_steps - 1:
                        break

                # ---- end of epoch, only rank 0 does checkpointing ----
                if rank == 0:
                    self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()
                    print(f"[Rank 0] Finished epoch {epoch_idx}")

        # ---------------- CLEANUP ----------------
        if world_size > 1 and dist.is_initialized():
            dist.barrier()



@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionTransformerHybridWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
