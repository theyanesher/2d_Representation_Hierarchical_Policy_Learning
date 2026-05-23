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
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
# from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionUnetHybridWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetHybridImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    # --- attention diagnostics (no-op if model has no DiT transformer_blocks) -----
    def _get_dit_blocks(self):
        return getattr(getattr(self.model, 'model', None), 'transformer_blocks', None)

    def _collect_attn_grad_norms(self, accumulator):
        """Push the latest per-block goal/vis grad norms into the accumulator.
        Called once per backward; reads buffers set by hooks in BasicTransformerBlock."""
        blocks = self._get_dit_blocks()
        if blocks is None:
            return
        for i, blk in enumerate(blocks):
            g = getattr(blk, '_last_goal_grad_norm', None)
            v = getattr(blk, '_last_vis_grad_norm',  None)
            if g is not None:
                accumulator.setdefault(f'grad_norm_wca/block_{i}',    []).append(g)
            if v is not None:
                accumulator.setdefault(f'grad_norm_vis_ca/block_{i}', []).append(v)

    def _log_attn_diagnostics(self, step_log, accumulator):
        """Add per-block α_goal + per-block grad norm means + cross-block aggregates."""
        blocks = self._get_dit_blocks()
        if blocks is None:
            return
        # Per-block α_goal — current value (gates change slowly; no need to average)
        for i, blk in enumerate(blocks):
            if hasattr(blk, 'goal_residual_gate'):
                step_log[f'alpha_goal/block_{i}'] = blk.goal_residual_gate.detach().item()
        # Per-block grad norm — mean across the epoch's steps
        for key, vals in accumulator.items():
            if vals:
                step_log[key] = float(np.mean(vals))
        # Aggregates: mean across blocks, and the WCA/CA ratio
        wca = [v for k, v in step_log.items() if k.startswith('grad_norm_wca/block_')]
        vca = [v for k, v in step_log.items() if k.startswith('grad_norm_vis_ca/block_')]
        if wca:
            step_log['grad_norm_wca/mean']    = float(np.mean(wca))
        if vca:
            step_log['grad_norm_vis_ca/mean'] = float(np.mean(vca))
        if wca and vca:
            step_log['grad_norm_ratio_wca_over_vis_ca'] = (
                step_log['grad_norm_wca/mean'] / (step_log['grad_norm_vis_ca/mean'] + 1e-12)
            )

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = cfg.training.get('resume_ckpt_path', None)
            if lastest_ckpt_path is None:
                raise ValueError(
                    "training.resume=true but training.resume_ckpt_path is not set. "
                    "Pass +training.resume_ckpt_path=/path/to/ckpt on the CLI."
                )
            print(f"Resuming from checkpoint {lastest_ckpt_path}")
            self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        # env_runner: BaseImageRunner
        # env_runner = hydra.utils.instantiate(
        #     cfg.task.env_runner,
        #     output_dir=self.output_dir)
        # assert isinstance(env_runner, BaseImageRunner)

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        # topk_manager = TopKCheckpointManager(
        #     save_dir=os.path.join(self.output_dir, 'checkpoints'),
        #     **cfg.checkpoint.topk
        # )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            print("TOTAL EPOCHSSSSSSSSSSSS", cfg.training.num_epochs)
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # Per-epoch accumulator for attention-block grad norms.
                # Populated by _collect_attn_grad_norms() after each loss.backward().
                attn_grad_norm_accum = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        # import pdb; pdb.set_trace();
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        # plucker_0 = np.load('media/plucker_map_0.npy')
                        # plucker_1 = np.load('media/plucker_map_1.npy')
                        # plucker_2 = np.load('media/plucker_map_2.npy')
                        # print(np.allclose(batch['obs']['cam0_plucker'][0,0].permute(1,2,0).cpu().numpy(), plucker_0))
                        # print(np.allclose(batch['obs']['cam1_plucker'][0,0].permute(1,2,0).cpu().numpy(), plucker_1))
                        # print(np.allclose(batch['obs']['cam2_plucker'][0,0].permute(1,2,0).cpu().numpy(), plucker_2))
                        # breakpoint()
                        # import pdb; pdb.set_trace()
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # Read backward-hook buffers (populated during loss.backward()
                        # if log_attention_grad_norms=true). No-op otherwise.
                        self._collect_attn_grad_norms(attn_grad_norm_accum)

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                # if (self.epoch % cfg.training.rollout_every) == 0:
                #     runner_log = env_runner.run(policy)
                #     # log all
                #     step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint(tag=f'epoch_{self.epoch}')
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    # topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    # if topk_ckpt_path is not None:
                    #     self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # Attention diagnostics: per-block α_goal current values +
                # per-block grad norm means over this epoch + cross-block aggregates.
                # No-op when the model has no DiT transformer_blocks / gates / hooks.
                self._log_attn_diagnostics(step_log, attn_grad_norm_accum)
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetHybridWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
