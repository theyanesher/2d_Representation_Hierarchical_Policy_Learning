import torch
import hydra
import pathlib
from torch.utils.data import DataLoader
import copy
import os
from diffusion_policy.workspace.train_diffusion_transformer_hybrid_workspace import TrainDiffusionTransformerHybridWorkspace
from diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace import TrainDiffusionUnetHybridWorkspace
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import DiffusionTransformerHybridImagePolicy
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from typing import Dict, Callable, List
import pickle

def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        # print("KEYYYYYYYY", key, value.shape)
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            # print("KEYYYYYYYY", key, value.shape)
            result[key] = func(value)
    # import pdb; pdb.set_trace();
    return result
use_unet = True

config_name = (
    "train_diffusion_unet_hybrid_workspace.yaml"
    if use_unet
    else "train_diffusion_transformer_lowdim_highlevel_heatmaps_workspace.yaml"
)

@hydra.main(
    version_base=None,
    config_path= "/home/pratik_final/Downloads/Bimanual/Original_RVT2_Inference/RVT/diffusion_policy/diffusion_policy/config",      #str(pathlib.Path(__file__).parent.parent / "config"),
    config_name=config_name,
)
def main(cfg):
    # ---------------------------------------------------------
    # Setup workspace and device
    # ---------------------------------------------------------
    if not use_unet:
        workspace = TrainDiffusionTransformerHybridWorkspace(cfg)
    else:
        workspace = TrainDiffusionUnetHybridWorkspace(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = workspace.model
    lastest_ckpt_path = "/home/pratik_final/Downloads/Bimanual/Original_RVT2_Inference/RVT/diffusion_policy/Primary_Training_Outputs/checkpoints/latest_WITH_WRIST_CAMERA.ckpt"  #"/home/pratik_final/Downloads/Bimanual/Original_RVT2_Inference/RVT/diffusion_policy/Primary_Training_Outputs/checkpoints/latest_WHOLE_DATASET_NO_WRIST.ckpt" # workspace.get_checkpoint_path() "/home/pratik_final/Downloads/Bimanual/Original_RVT2_Inference/RVT/diffusion_policy/Primary_Training_Outputs/checkpoints/latest_WITH_WRIST_CAMERA.ckpt"
    workspace.load_checkpoint(path=lastest_ckpt_path)
    # if not use_unet:
    #     ema_model: DiffusionTransformerHybridImagePolicy = None
    # else:
    ema_model: DiffusionUnetHybridImagePolicy = None
    if cfg.training.use_ema:
        try:
            ema_model = copy.deepcopy(workspace.model)
        except: # minkowski engine could not be copied. recreate it
            ema_model = hydra.utils.instantiate(cfg.policy)
    
        # if cfg.training.use_ema:
        #     self.ema_model = copy.deepcopy(self.model)
    policy = ema_model
    policy.to(device)
    # import pdb; pdb.set_trace();
    dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
    assert isinstance(dataset, BaseImageDataset)

    inference_loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False
    )
    save_dir = "predicted_actions_CONVOLUTION"
    os.makedirs(save_dir, exist_ok=True)
    # import pdb; pdb.set_trace();
    print(f"✅ Loaded dataset: {len(dataset)} samples")

    # ---------------------------------------------------------
    # Inference loop — use predict_action() on data
    # ---------------------------------------------------------
    mse_trans_sum = 0
    mse_quat_sum = 0
    mse_open_close_sum = 0
    abs_mse_sum = 0
    mse_trans_MSELoss_sum = 0
    with torch.no_grad():
        for i, batch in enumerate(inference_loader):
            # batch is a dictionary of observation tensors
            # obs_dict = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            obs_dict = batch['obs']
            gt_action = batch["action"]
            # import pdb; pdb.set_trace();
            result = ema_model.predict_action(obs_dict)
            if not use_unet:
                pred_action = result["action_pred"]
            else:
                pred_action = result["action_pred"]["action"]
            print(f"[Batch {i}] Predicted action shape: {pred_action.shape}")
            mse_trans_MSELoss = torch.nn.functional.mse_loss(pred_action[:,:,:3], gt_action[:,:,:3])
            # mse_quat = torch.nn.functional.mse_loss(pred_action[:,:,3:7], gt_action[:,:,3:7])
            # mse_open_close = torch.nn.functional.mse_loss(pred_action[:,:,7], gt_action[:,:,7])
            # import pdb; pdb.set_trace();
            mse_trans = torch.norm(pred_action[:,:,:3] - gt_action[:,:,:3], dim=2).mean()
            mse_quat = torch.nn.functional.mse_loss(pred_action[:,:,3:7], gt_action[:,:,3:7])
            mse_open_close = torch.nn.functional.mse_loss(pred_action[:,:,7], gt_action[:,:,7])
            abs_mse = torch.nn.functional.mse_loss(pred_action, gt_action)
            mse_trans_MSELoss_sum += mse_trans_MSELoss
            mse_trans_sum += mse_trans 
            mse_quat_sum += mse_quat
            mse_open_close_sum  += mse_open_close
            abs_mse_sum += abs_mse
            pred_dict = {"pred_action":pred_action.cpu().numpy(), "gt_action": gt_action.cpu().numpy()}
            print("trans_norm",mse_trans, "trans_MSELoss", mse_trans_MSELoss, "quat", mse_quat, "open_close", mse_open_close, "abs", abs_mse)
            # import pdb; pdb.set_trace();
            # Save as: predicted_actions/000000.pkl, 000001.pkl, ...
            out_path = os.path.join(save_dir, f"{i}.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(pred_dict , f, protocol=pickle.HIGHEST_PROTOCOL)

            # print(mse_trans, mse_quat)
            # import pdb; pdb.set_trace();
            # # Stop after a few batches for demonstration
            # if i >= 4:
            #     break
    print("AVERAGE", "trans_norm", mse_trans_sum/len(dataset), "trans_MSE", mse_trans_MSELoss_sum/len(dataset), "quat", mse_quat_sum/len(dataset), "open_close", mse_open_close_sum/len(dataset), "total_mse", abs_mse_sum/len(dataset))


if __name__ == "__main__":
    main()
