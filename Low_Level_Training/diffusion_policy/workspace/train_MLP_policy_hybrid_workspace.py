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
import torch.nn.functional as F
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
from diffusion_policy.policy.MLP_hybrid_image_policy import MLPHybridImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from torch.utils.data import DistributedSampler
import torch.distributed as dist

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainMLPHybridWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: MLPHybridImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: MLPHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        self.global_step = 0
        self.epoch = 0

    def run(self, rank=0, world_size=1):

        # -------- device --------
        if world_size > 1:
            torch.cuda.set_device(rank)
            device = torch.device(f"cuda:{rank}")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cfg = copy.deepcopy(self.cfg)

        # -------- optional resume --------
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # -------- dataset --------
        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)

        is_main = (rank == 0)
        if is_main:
            print("Dataset length:", len(dataset))

        if world_size > 1:
            train_sampler = DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=True)
        else:
            train_sampler = None

        train_dataloader = DataLoader(dataset, sampler=train_sampler, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        val_dataset = dataset.get_validation_dataset()
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        ) if world_size > 1 else None
        val_dataloader = DataLoader(val_dataset, sampler=val_sampler, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # -------- lr scheduler --------
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs
            ) // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1
        )

        # -------- EMA --------
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # -------- logging --------
        if is_main:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            wandb.config.update({"output_dir": self.output_dir})
        else:
            wandb_run = None

        # -------- checkpoint manager --------
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # -------- device transfer --------
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # -------- DDP wrap --------
        if world_size > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[rank], output_device=rank,
                find_unused_parameters=False)

        # -------- full-dataset eval dataloader (rank 0 only) --------
        if is_main:
            sample_eval_dataloader = DataLoader(
                dataset, batch_size=cfg.dataloader.batch_size, shuffle=False,
                num_workers=cfg.dataloader.num_workers, pin_memory=False
            )

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # -------- committed-steps slice (same formula as diffusion policy) --------
        n_obs_steps = cfg.n_obs_steps
        n_action_steps = cfg.n_action_steps
        commit_start = n_obs_steps - 1
        commit_end = commit_start + n_action_steps

        # ====================================================================
        # Training loop
        # ====================================================================
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                if train_sampler is not None:
                    train_sampler.set_epoch(self.epoch)

                step_log = dict()
                train_losses = list()

                with tqdm.tqdm(train_dataloader,
                        desc=f"Training epoch {self.epoch}",
                        leave=False,
                        mininterval=cfg.training.tqdm_interval_sec,
                        disable=not is_main) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                            raw_loss = self.model.module.compute_loss(batch)
                        else:
                            raw_loss = self.model.compute_loss(batch)

                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                            if cfg.training.use_ema:
                                ema.step(self.model.module)
                        else:
                            if cfg.training.use_ema:
                                ema.step(self.model)

                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader) - 1))
                        if not is_last_batch:
                            if is_main:
                                wandb_run.log(step_log, step=self.global_step)
                                json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                                and batch_idx >= (cfg.training.max_train_steps - 1):
                            break

                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # -------- eval policy --------
                policy = self.ema_model if cfg.training.use_ema else self.model
                # unwrap DDP — predict_action lives on the underlying module
                if isinstance(policy, torch.nn.parallel.DistributedDataParallel):
                    policy = policy.module
                policy.eval()

                # -------- validation loss --------
                if (self.epoch % cfg.training.val_every) == 0 and not cfg.training.no_validation:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader,
                                desc=f"Validation epoch {self.epoch}",
                                leave=False,
                                mininterval=cfg.training.tqdm_interval_sec,
                                disable=not is_main) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                                    loss = self.model.module.compute_loss(batch)
                                else:
                                    loss = self.model.compute_loss(batch)
                                val_losses.append(loss.item())
                                if (cfg.training.max_val_steps is not None) \
                                        and batch_idx >= (cfg.training.max_val_steps - 1):
                                    break
                        if len(val_losses) > 0:
                            step_log['val_loss'] = np.mean(val_losses)

                # -------- MSE on committed steps over full dataset --------
                # Deterministic (no DDPM stochasticity) — should go to near-zero when overfitting.
                if is_main and (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        all_mse = []
                        all_mse_trans = []
                        all_mse_quat = []
                        all_mse_open_close = []
                        for eval_batch in tqdm.tqdm(sample_eval_dataloader,
                                desc=f"Sample eval epoch {self.epoch}",
                                leave=False,
                                mininterval=cfg.training.tqdm_interval_sec):
                            eval_batch = dict_apply(eval_batch, lambda x: x.to(device, non_blocking=True))
                            obs_dict = eval_batch['obs']
                            # GT committed steps: shape (B, n_action_steps, action_dim)
                            gt_action = eval_batch['action'][:, commit_start:commit_end, :]

                            result = policy.predict_action(obs_dict, eval_batch['obs_lang_emb'])
                            pred_action = result['action']  # (B, n_action_steps, action_dim)

                            all_mse.append(F.mse_loss(pred_action, gt_action).item())
                            all_mse_trans.append(F.mse_loss(pred_action[:, :, :3], gt_action[:, :, :3]).item())
                            all_mse_quat.append(F.mse_loss(pred_action[:, :, 3:7], gt_action[:, :, 3:7]).item())
                            all_mse_open_close.append(F.mse_loss(pred_action[:, :, 7:], gt_action[:, :, 7:]).item())

                        mse = np.mean(all_mse)
                        mse_trans = np.mean(all_mse_trans)
                        mse_quat = np.mean(all_mse_quat)
                        mse_oc = np.mean(all_mse_open_close)
                        print(f"MSE [trans={mse_trans:.6f}  quat={mse_quat:.6f}  open_close={mse_oc:.6f}  total={mse:.6f}]")
                        step_log['train_action_mse_error'] = mse
                        step_log['train_trans_action_mse_error'] = mse_trans
                        step_log['train_quat_action_mse_error'] = mse_quat
                        step_log['train_open_close_action_mse_error'] = mse_oc

                # -------- checkpoint --------
                if is_main and (self.epoch % cfg.training.checkpoint_every) == 0:
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    metric_dict = {k.replace('/', '_'): v for k, v in step_log.items()}
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                policy.train()

                if is_main:
                    wandb_run.log(step_log, step=self.global_step)
                    json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainMLPHybridWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
