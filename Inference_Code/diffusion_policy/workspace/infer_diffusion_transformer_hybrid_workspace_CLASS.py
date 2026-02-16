import torch
import hydra
import pathlib
import copy
import os
from omegaconf import OmegaConf
from typing import Dict, Callable
import pickle
from hydra import initialize, compose, initialize_config_dir
from diffusion_policy.diffusion_policy.workspace.train_diffusion_transformer_hybrid_workspace import (
    TrainDiffusionTransformerHybridWorkspace
)
from diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace import TrainDiffusionUnetHybridWorkspace
from diffusion_policy.diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import (
    DiffusionTransformerHybridImagePolicy
)
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.diffusion_policy.dataset.base_dataset import BaseImageDataset


def dict_apply(x: Dict, func: Callable):
    out = {}
    for k, v in x.items():
        if isinstance(v, dict):
            out[k] = dict_apply(v, func)
        else:
            out[k] = func(v)
    return out


# -----------------------------------------------------------
#               CLASS FOR INFERENCE
# -----------------------------------------------------------
class DiffusionHybridInference:
    """
    Wrapper for loading the workspace, model, EMA, and running inference
    from an obs_dict only.
    """

    def __init__(self, config_name: str):
        """
        Args:
            config_name: YAML config filename inside diffusion_policy/config/
                         e.g., "train_diffusion_transformer_lowdim_highlevel_heatmaps_workspace.yaml"
        """

        # --------------------------
        # Resolve Hydra config path
        # --------------------------
        # import pdb; pdb.set_trace();
        config_path = pathlib.Path(__file__).parent.parent.joinpath("config")
        config_path = str(config_path)
        self.use_unet = True
        # Load config manually (no @hydra.main needed)
        # self.cfg = hydra.compose(
        #     config_name=config_name,
        #     overrides=[],
        #     return_hydra_config=False
        # )
        with initialize_config_dir(config_path):    self.cfg = compose(        config_name=config_name,        overrides=[]    )
        # --------------------------
        # Setup workspace
        # --------------------------
        # import pdb; pdb.set_trace();
        if not self.use_unet:
            self.workspace = TrainDiffusionTransformerHybridWorkspace(self.cfg)
        else:
            self.workspace = TrainDiffusionUnetHybridWorkspace(self.cfg)

        # --------------------------
        # Load the checkpoint
        # --------------------------
        import pdb; pdb.set_trace();
        ckpt_path = "/home/pratik_final/Downloads/Bimanual/Original_RVT2_Inference/RVT/runs/diffusion_policy_full_dataset_train/rsync_copy/latest.ckpt" #"/home/pratik_final/Downloads/Bimanual/Original_RVT2_Inference/RVT/diffusion_policy/Primary_Training_Outputs/checkpoints/epoch=2130-train_loss=0.004_WHOLE_DATASET_WRIST_CAM.ckpt" #self.workspace.get_checkpoint_path()
        print(f"Loading checkpoint: {ckpt_path}")
        self.workspace.load_checkpoint(ckpt_path)

        # --------------------------
        # Build EMA model
        # --------------------------
        self.ema_model: DiffusionTransformerHybridImagePolicy = None
        if self.cfg.training.use_ema:
            try:
                if not self.use_unet:
                    self.ema_model: DiffusionTransformerHybridImagePolicy = \
                        copy.deepcopy(self.workspace.model)
                else:
                    self.ema_model: DiffusionUnetHybridImagePolicy = \
                        copy.deepcopy(self.workspace.model)
            except Exception:
                print("Deep copy failed (e.g. Minkowski Engine), reinstantiating.")
                self.ema_model = hydra.utils.instantiate(self.cfg.policy)
        else:
            self.model = self.workspace.model

        # Move model to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ema_model.to(self.device)
        self.ema_model.eval()

        print("Inference model ready.")
        # import pdb; pdb.set_trace();
    # -------------------------------------------------------
    #                 RUN INFERENCE
    # -------------------------------------------------------
    @torch.no_grad()
    def predict(self, obs_dict: Dict) -> Dict:
        """
        Run inference on a single obs_dict.

        Args:
            obs_dict: dictionary of observations (same format your dataset returns)
        Returns:
            result dict -> from policy.predict_action()
        """
        import pdb; pdb.set_trace()
        obs = dict_apply(obs_dict, lambda x: x.to(self.device, non_blocking=True))
        import pdb; pdb.set_trace()
        result = self.ema_model.predict_action(obs["obs"], obs["obs_lang_emb"])
        return result

    # -------------------------------------------------------
    #         OPTIONAL: Evaluate full dataset
    # -------------------------------------------------------
    @torch.no_grad()
    def evaluate_dataset(self, dataset: BaseImageDataset, save_dir="predicted_actions"):

        os.makedirs(save_dir, exist_ok=True)

        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        mse_trans_sum = 0
        mse_quat_sum = 0

        print(f"Evaluating {len(dataset)} samples...")

        for i, batch in enumerate(loader):

            batch = dict_apply(batch, lambda x: x.to(self.device))
            obs_dict = batch["obs"]
            gt_action = batch["action"]

            result = self.model.predict_action(obs_dict)
            pred_action = result["action_pred"]

            mse_trans = torch.nn.functional.mse_loss(
                pred_action[:, :, :3], gt_action[:, :, :3]
            )
            mse_quat = torch.nn.functional.mse_loss(
                pred_action[:, :, 3:7], gt_action[:, :, 3:7]
            )

            mse_trans_sum += mse_trans.item()
            mse_quat_sum += mse_quat.item()

            out_path = os.path.join(save_dir, f"{i:06d}.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(
                    {
                        "pred_action": pred_action.cpu().numpy(),
                        "gt_action": gt_action.cpu().numpy(),
                    },
                    f,
                )

        print(f"Done. Average translation MSE: {mse_trans_sum/len(dataset)}")
        print(f"Average quaternion MSE: {mse_quat_sum/len(dataset)}")

