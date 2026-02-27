"""
Open-loop inference script for the MLP behavioral-cloning policy.

Usage:
    python infer_MLP_hybrid_workspace.py

Edit the CONFIG_PATH, CONFIG_NAME, and CKPT_PATH constants below before running.
Predictions and GT actions are saved as per-sample .pkl files under SAVE_DIR.

Output per file: {"pred_action": np.ndarray, "gt_action": np.ndarray}
  pred_action  shape: (1, n_action_steps, action_dim)   — committed steps only
  gt_action    shape: (1, horizon, action_dim)           — full GT window
"""

import os
import copy
import pickle
from typing import Dict, Callable

import torch
import hydra
import pathlib
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.workspace.train_MLP_policy_hybrid_workspace import TrainMLPHybridWorkspace
from diffusion_policy.policy.MLP_hybrid_image_policy import MLPHybridImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset

# ──────────────────────────────────────────────────────────────────────────────
# Edit these three constants before running
# ──────────────────────────────────────────────────────────────────────────────
CONFIG_PATH = "/path/to/diffusion_policy/config"   # absolute path to config dir
CONFIG_NAME = "train_MLP_policy_hybrid_workspace_zarr_dataloader_EARLY"
CKPT_PATH   = "/path/to/checkpoints/latest.ckpt"
SAVE_DIR    = "predicted_actions_MLP"
# ──────────────────────────────────────────────────────────────────────────────

OmegaConf.register_new_resolver("eval", eval, replace=True)


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


@hydra.main(
    version_base=None,
    config_path=CONFIG_PATH,
    config_name=CONFIG_NAME,
)
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── load workspace + checkpoint ──────────────────────────────────────────
    workspace = TrainMLPHybridWorkspace(cfg)
    print(f"Loading checkpoint: {CKPT_PATH}")
    workspace.load_checkpoint(path=CKPT_PATH)

    # ── pick the policy to run ───────────────────────────────────────────────
    # If EMA was enabled during training, use ema_model; otherwise use model.
    if cfg.training.use_ema and workspace.ema_model is not None:
        try:
            policy: MLPHybridImagePolicy = copy.deepcopy(workspace.ema_model)
        except Exception:
            policy = hydra.utils.instantiate(cfg.policy)
        print("Using EMA model for inference.")
    else:
        policy: MLPHybridImagePolicy = workspace.model
        print("Using regular model for inference (EMA disabled).")

    policy.to(device)
    policy.eval()

    # ── dataset & dataloader ─────────────────────────────────────────────────
    dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
    assert isinstance(dataset, BaseImageDataset)
    print(f"Dataset length: {len(dataset)} samples")

    inference_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── committed-step slice (same as training) ───────────────────────────────
    n_obs_steps   = cfg.n_obs_steps
    n_action_steps = cfg.n_action_steps
    commit_start  = n_obs_steps - 1
    commit_end    = commit_start + n_action_steps

    # ── inference loop ────────────────────────────────────────────────────────
    mse_trans_sum      = 0.0
    mse_trans_l2_sum   = 0.0
    mse_quat_sum       = 0.0
    mse_open_close_sum = 0.0
    mse_total_sum      = 0.0

    with torch.no_grad():
        for i, batch in enumerate(inference_loader):
            batch    = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            obs_dict = batch['obs']
            lang_emb = batch['obs_lang_emb']
            gt_action = batch['action']   # (1, horizon, action_dim)

            result = policy.predict_action(obs_dict, lang_emb=lang_emb)

            # committed predicted steps: (1, n_action_steps, action_dim)
            pred_action = result['action']

            # matching GT committed window
            gt_committed = gt_action[:, commit_start:commit_end, :]  # (1, n_action_steps, action_dim)

            # ── metrics (on committed steps) ─────────────────────────────────
            mse_trans_l2   = torch.norm(pred_action[:, :, :3] - gt_committed[:, :, :3], dim=2).mean()
            mse_trans      = torch.nn.functional.mse_loss(pred_action[:, :, :3], gt_committed[:, :, :3])
            mse_quat       = torch.nn.functional.mse_loss(pred_action[:, :, 3:7], gt_committed[:, :, 3:7])
            mse_open_close = torch.nn.functional.mse_loss(pred_action[:, :, 7:], gt_committed[:, :, 7:])
            mse_total      = torch.nn.functional.mse_loss(pred_action, gt_committed)

            mse_trans_sum      += mse_trans.item()
            mse_trans_l2_sum   += mse_trans_l2.item()
            mse_quat_sum       += mse_quat.item()
            mse_open_close_sum += mse_open_close.item()
            mse_total_sum      += mse_total.item()

            print(
                f"[{i:05d}]  "
                f"trans_L2={mse_trans_l2:.5f}  "
                f"trans_MSE={mse_trans:.5f}  "
                f"quat={mse_quat:.5f}  "
                f"open_close={mse_open_close:.5f}  "
                f"total={mse_total:.5f}"
            )

            # ── save ─────────────────────────────────────────────────────────
            pred_dict = {
                "pred_action": pred_action.cpu().numpy(),   # (1, n_action_steps, action_dim) committed
                "gt_action":   gt_action.cpu().numpy(),     # (1, horizon, action_dim) full window
            }
            out_path = os.path.join(SAVE_DIR, f"{i:06d}.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(pred_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    n = len(dataset)
    print("\n══════════════ AVERAGE OVER DATASET ══════════════")
    print(f"  trans_L2    : {mse_trans_l2_sum   / n:.6f}")
    print(f"  trans_MSE   : {mse_trans_sum       / n:.6f}")
    print(f"  quat_MSE    : {mse_quat_sum        / n:.6f}")
    print(f"  open_close  : {mse_open_close_sum  / n:.6f}")
    print(f"  total_MSE   : {mse_total_sum       / n:.6f}")
    print(f"  Saved {n} files → {SAVE_DIR}/")


if __name__ == "__main__":
    main()
