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
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from torch.utils.data import DistributedSampler
import torch.distributed as dist
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

    def run(self, rank, world_size):

        ####### DEVICE SELECTION FOR DDP ##########
        if world_size > 1:
            torch.cuda.set_device(rank)
            device = torch.device(f"cuda:{rank}")
            # dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            # import pdb; pdb.set_trace();
            # lastest_ckpt_path = "/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/Diffusion_Policy_Paper/Diff_Policy/diffusion_policy/data/outputs/2025.12.04/21.30.20_train_diffusion_unet_hybrid_high_level_heatmap_cnn/checkpoints/latest.ckpt" #self.get_checkpoint_path()
            # lastest_ckpt_path = "/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/Diffusion_Policy_Paper/Diff_Policy/diffusion_policy/data/outputs/2025.11.24/23.34.37_train_diffusion_unet_hybrid_high_level_heatmap_cnn_NO_INPAINT_TRAIN/checkpoints/latest.ckpt"
            lastest_ckpt_path = "/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/Diffusion_Policy_Paper/Diff_Policy/diffusion_policy/data/outputs/2026.02.22/21.23.31_RVT_Heatmap_ALL_Tasks_LATER_high_level_heatmap_cnn_zarr_dataloader_LATER_RVT_Heatmap/checkpoints/latest.ckpt"#"/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/Diffusion_Policy_Paper/Diff_Policy/diffusion_policy/data/outputs/2025.12.13/17.20.42_train_diffusion_unet_hybrid_WHOLE_DATASET_WRIST_high_level_heatmap_cnn/checkpoints/latest.ckpt" #"/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/Diffusion_Policy_Paper/Diff_Policy/diffusion_policy/data/outputs/2025.12.11/19.43.56_train_diffusion_unet_hybrid_WHOLE_DATASET_WRIST_high_level_heatmap_cnn/checkpoints/latest.ckpt" #"/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/Diffusion_Policy_Paper/Diff_Policy/diffusion_policy/data/outputs/2025.12.08/22.41.02_train_diffusion_unet_hybrid_WHOLE_DATASET_WRIST_high_level_heatmap_cnn/checkpoints/latest.ckpt"
            # if lastest_ckpt_path.is_file():
            print(f"Resuming from checkpoint {lastest_ckpt_path}")
            self.load_checkpoint(path=lastest_ckpt_path)

        # ========= test_model mode: open-loop MSE evaluation =========
        if cfg.training.get('test_model', False):
            if rank == 0:
                print("[test_model] Building full-dataset dataloader (batch_size=1, shuffle=False)...")
            dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
            assert isinstance(dataset, BaseImageDataset)
            if rank == 0:
                print(f"[test_model] Dataset length: {len(dataset)}")
            test_dataloader = DataLoader(
                dataset, batch_size=1, shuffle=False,
                num_workers=cfg.dataloader.num_workers, pin_memory=False
            )
            normalizer = dataset.get_normalizer()
            self.model.set_normalizer(normalizer)
            if cfg.training.use_ema:
                self.ema_model.set_normalizer(normalizer)
            policy = self.ema_model if (cfg.training.use_ema and self.ema_model is not None) else self.model
            policy.to(device)
            policy.eval()

            all_mse, all_mse_trans, all_mse_quat, all_mse_open_close = [], [], [], []
            with torch.no_grad():
                for batch in tqdm.tqdm(test_dataloader, desc="[test_model] Evaluating", disable=(rank != 0)):
                    batch = dict_apply(batch, lambda x: x.to(device))
                    obs_dict = batch['obs']
                    gt_action = batch['action']
                    result = policy.predict_action(obs_dict, batch['obs_lang_emb'])
                    pred_action = result['action_pred']['action']
                    all_mse.append(torch.nn.functional.mse_loss(pred_action, gt_action).item())
                    all_mse_trans.append(torch.nn.functional.mse_loss(pred_action[:, :, :3], gt_action[:, :, :3]).item())
                    all_mse_quat.append(torch.nn.functional.mse_loss(pred_action[:, :, 3:7], gt_action[:, :, 3:7]).item())
                    all_mse_open_close.append(torch.nn.functional.mse_loss(pred_action[:, :, 7], gt_action[:, :, 7]).item())

            if rank == 0:
                n = len(all_mse)
                print(f"\n===== TEST MODEL RESULTS (N={n} samples) =====")
                print(f"Avg MSE (full action):  {np.mean(all_mse):.6f}  ± {np.std(all_mse):.6f}")
                print(f"Avg MSE (translation):  {np.mean(all_mse_trans):.6f}  ± {np.std(all_mse_trans):.6f}")
                print(f"Avg MSE (quaternion):   {np.mean(all_mse_quat):.6f}  ± {np.std(all_mse_quat):.6f}")
                print(f"Avg MSE (open/close):   {np.mean(all_mse_open_close):.6f}  ± {np.std(all_mse_open_close):.6f}")
            return
        # ========= end test_model mode =========

        # # configure dataset
        # dataset: BaseImageDataset
        # dataset = hydra.utils.instantiate(cfg.task.dataset)
        # assert isinstance(dataset, BaseImageDataset)
        # train_dataloader = DataLoader(dataset, **cfg.dataloader)
        # normalizer = dataset.get_normalizer()

        # # configure validation dataset
        # val_dataset = dataset.get_validation_dataset()
        # val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)



        # configure dataset
        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        if rank == 0:
            print("LENGTH OF THE DATASET = ", len(dataset))
        # distributed sampler for multi-GPU
        if world_size > 1:
            train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        else:
            train_sampler = None

        train_dataloader = DataLoader(
            dataset,
            sampler=train_sampler,
            **cfg.dataloader
        )
        # import pdb; pdb.set_trace();
        normalizer = dataset.get_normalizer()

        # validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
        val_dataloader = DataLoader(val_dataset, sampler=val_sampler, **cfg.val_dataloader)

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
        ########### for distributed training ###########
        is_main = (rank == 0)
        if is_main:
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
        else:
            wandb_run = None

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        ############### FOR DISTRIBUTED ####################
        # -----------------------------
        # 3. DDP wrap
        # -----------------------------
        if world_size > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=False  # change only if needed
            )
        # full-dataset eval dataloader for sample_every MSE (rank 0 only, no DDP sampler)
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
            cfg.training.checkpoint_every = 20
            cfg.training.val_every = 1
            cfg.training.sample_every = 20

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                if train_sampler is not None:
                    train_sampler.set_epoch(self.epoch)
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                        leave=False, mininterval=cfg.training.tqdm_interval_sec,
                        disable=not is_main) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        # import pdb; pdb.set_trace();
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                        # compute loss
                        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                            if cfg.policy.seperate_open_close:
                                raw_loss, bce_loss = self.model.module.compute_loss(batch)
                                # avg_bce_loss = bce_loss / cfg.training.gradient_accumulate_every
                            else:
                                raw_loss = self.model.module.compute_loss(batch)
                        else:
                            if cfg.policy.seperate_open_close:
                                raw_loss, bce_loss = self.model.compute_loss(batch)
                                # avg_bce_loss = bce_loss / cfg.training.gradient_accumulate_every
                            else:
                                raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                            if cfg.training.use_ema:
                                ema.step(self.model.module)
                        else:
                            if cfg.training.use_ema:
                                ema.step(self.model)

                        # logging
                        if cfg.policy.seperate_open_close:
                            bce_loss_cpu = bce_loss.item()
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        if not cfg.policy.seperate_open_close:
                            step_log = {
                                'train_loss': raw_loss_cpu,
                                'global_step': self.global_step,
                                'epoch': self.epoch,
                                'lr': lr_scheduler.get_last_lr()[0]
                            }
                        else:
                            step_log = {
                                'train_loss': raw_loss_cpu,
                                'train_bce_loss': bce_loss_cpu,
                                'global_step': self.global_step,
                                'epoch': self.epoch,
                                'lr': lr_scheduler.get_last_lr()[0]
                            }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            if is_main:
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
                if (self.epoch % cfg.training.val_every) == 0 and not cfg.training.no_validation:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                leave=False, mininterval=cfg.training.tqdm_interval_sec,
                                disable=not is_main) as tepoch:
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

                # run diffusion sampling on the full dataset
                if is_main and (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        all_mse, all_mse_trans, all_mse_quat, all_mse_open_close = [], [], [], []
                        for eval_batch in tqdm.tqdm(sample_eval_dataloader,
                                desc=f"Sample eval epoch {self.epoch}",
                                leave=False, mininterval=cfg.training.tqdm_interval_sec):
                            eval_batch = dict_apply(eval_batch, lambda x: x.to(device, non_blocking=True))
                            obs_dict = eval_batch['obs']
                            gt_action = eval_batch['action']
                            result = policy.predict_action(obs_dict, eval_batch['obs_lang_emb'])
                            pred_action = result['action_pred']['action']
                            all_mse.append(torch.nn.functional.mse_loss(pred_action, gt_action).item())
                            all_mse_trans.append(torch.nn.functional.mse_loss(pred_action[:,:,:3], gt_action[:,:,:3]).item())
                            all_mse_quat.append(torch.nn.functional.mse_loss(pred_action[:,:,3:7], gt_action[:,:,3:7]).item())
                            all_mse_open_close.append(torch.nn.functional.mse_loss(pred_action[:,:,7], gt_action[:,:,7]).item())
                        mse = np.mean(all_mse)
                        mse_trans = np.mean(all_mse_trans)
                        mse_quat = np.mean(all_mse_quat)
                        mse_open_close = np.mean(all_mse_open_close)
                        print("MSEEEEEEEE", mse, mse_trans, mse_quat)
                        step_log['train_action_mse_error'] = mse
                        step_log['train_trans_action_mse_error'] = mse_trans
                        step_log['train_quat_action_mse_error'] = mse_quat
                        step_log['train_open_close_action_mse_error'] = mse_open_close
                
                # checkpoint
                if is_main and (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
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
                    # import pdb; pdb.set_trace();
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
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
    workspace = TrainDiffusionUnetHybridWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
