"""
Articubot-only DDP trainer (GMM / displacement policy).

Same CLI and config.yaml shape as train_multitask_ddp_weighted_displacement_gmm.py:
- `cgn` and `general.tasks` entries are ignored except that `articubot` must appear in tasks.
- Iterates the dataloader with a normal for-loop (epochs until num_iterations), no infinite_loader.
"""
import torch
import argparse
import datetime
import json
import os
import random
import time

import wandb
from omegaconf import OmegaConf
from termcolor import cprint
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from test_PointNet2.dataset_from_disk import get_dataset_from_pickle

mse_loss = torch.nn.MSELoss()

def upload_file(local_folder):
    base = "gs://cmu-gpucloud-yufeiw2/articubot_exps"
    folder_name = os.path.basename(local_folder.rstrip("/"))
    destination = f"{base}/{folder_name}"
    
    try:
        cmd = ["gcloud", "storage", "rsync", "-r", local_folder, destination]
        # print(cmd)
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"[Success] Uploaded: {local_folder} -> {destination}")
    except subprocess.CalledProcessError as e:
        print(f"[Failure] Failed to upload {local_folder}: {e.stderr.strip()}")



def is_main_process():
    return dist.is_available() and dist.is_initialized() and dist.get_rank() == 0 or int(
        os.getenv("RANK", "0")
    ) == 0


def setup_articubot_dataloader(args):
    dataset = get_dataset_from_pickle(
        all_obj_paths=args.all_zarr_path,
        beg_ratio=args.beg_ratio,
        end_ratio=args.end_ratio,
        use_all_data=args.use_all_data,
        dataset_prefix=args.dataset_prefix,
        num_train_objects=args.num_train_objects,
        camera_frame=args.camera_frame,
        goal_always_open=args.goal_always_open,
        is_pickle=args.is_pickle,
        use_rgb=args.use_rgb,
        pred_gripper_width=args.pred_gripper_width,
        gripper_width_scale_factor=args.gripper_width_scale_factor,
        use_dino=args.use_dino,
    )
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        sampler=DistributedSampler(dataset),
        batch_size=args.batch_size,
        num_workers=3,
        pin_memory=False,
    )
    return dataloader


def compute_articubot_loss(
    data, model, optimizer, device, articubot_args, siglip_features=None, scheduler=None
):
    pointcloud, gripper_pcd, goal_gripper_pcd, cat_idx, class_weight, extra = data
    class_weight = class_weight.to(device, non_blocking=True)
    gripper_points = goal_gripper_pcd.to(device, non_blocking=True)

    if "rgb" in extra and articubot_args.use_rgb:
        rgb = extra["rgb"]
        pointcloud = torch.cat([pointcloud, rgb], dim=2)
        gripper_rgb = extra["rgb_gripper"]
        gripper_pcd = torch.cat([gripper_pcd, gripper_rgb], dim=2)
    if "dino_features" in extra and articubot_args.use_dino:
        dino_features = extra["dino_features"]
        pointcloud = torch.cat([pointcloud, dino_features], dim=2)
        gripper_dino = extra["dino_features_gripper"]
        gripper_pcd = torch.cat([gripper_pcd, gripper_dino], dim=2)

    if articubot_args.add_one_hot_encoding:
        pointcloud_one_hot = torch.zeros(pointcloud.shape[0], pointcloud.shape[1], 2)
        pointcloud_one_hot[:, :, 0] = 1
        pointcloud_ = torch.cat([pointcloud, pointcloud_one_hot], dim=2)
        gripper_pcd_one_hot = torch.zeros(gripper_pcd.shape[0], gripper_pcd.shape[1], 2)
        gripper_pcd_one_hot[:, :, 1] = 1
        gripper_pcd_ = torch.cat([gripper_pcd, gripper_pcd_one_hot], dim=2)
        inputs = torch.cat([pointcloud_, gripper_pcd_], dim=1)
        inputs = inputs.contiguous().float()
    else:
        inputs = torch.cat([pointcloud, gripper_pcd], dim=1)

    inputs = inputs.to(device, non_blocking=True)
    inputs = inputs.permute(0, 2, 1)

    if siglip_features is None:
        pred_dict = model(inputs)
    else:
        cat_embedding = siglip_features[cat_idx].float()
        pred_dict = model(inputs, embedding=cat_embedding)

    outputs = pred_dict["pred_offsets"]
    pred_points = pred_dict["pred_points"]
    weights = pred_dict["pred_scores"].squeeze(-1)

    labels = gripper_points.unsqueeze(1) - pred_points.unsqueeze(2)
    B, N, _, _ = labels.shape
    labels = labels.view(B, N, -1)

    if articubot_args.output_obj_pcd_only:
        weights = weights[:, :-4]
        outputs = outputs[:, :-4, :]
        labels = labels[:, :-4, :]
        inputs = inputs[:, :, :-4]
        N = N - 4

    outputs = outputs.view(B, N, -1)

    if articubot_args.gmm:
        diff = outputs - labels
        log_info = {}
        loss = 0
        for fixed_variance, variance_loss_scale in zip(
            articubot_args.fixed_variance, articubot_args.variance_loss_scale
        ):
            exponent = -0.5 * torch.sum((diff**2) / fixed_variance, dim=2)
            log_gaussians = exponent
            log_mixing_coeffs = torch.log_softmax(weights, dim=1)
            log_mixing_coeffs = torch.clamp(log_mixing_coeffs, min=-20)
            max_log = torch.max(log_gaussians + log_mixing_coeffs, dim=1, keepdim=True).values
            log_probs = max_log.squeeze(1) + torch.logsumexp(
                log_gaussians + log_mixing_coeffs - max_log, dim=1
            )
            this_loss = -torch.mean(log_probs * class_weight)
            loss += this_loss * variance_loss_scale
            log_info["gmm_" + str(fixed_variance)] = this_loss.item()
            log_info["gmm_" + str(fixed_variance) + "_scaled"] = (
                this_loss * variance_loss_scale
            ).item()
    else:
        per_point_loss = mse_loss(outputs, labels)
        inputs = pred_points
        outputs = outputs.view(B, N, 4, 3)
        outputs = outputs + inputs[:, :, :3].unsqueeze(2)
        weights = torch.nn.functional.softmax(weights, dim=1)
        outputs = outputs * weights.unsqueeze(-1).unsqueeze(-1)
        outputs = outputs.sum(dim=1)
        avg_loss = mse_loss(outputs, gripper_points)
        loss = per_point_loss + avg_loss * articubot_args.weight_loss_weight
        loss = loss * articubot_args.loss_scale
        log_info = {"loss": loss.item()}
        log_info["perpoint_loss"] = per_point_loss.item()
        log_info["weighted_average_loss"] = avg_loss.item()

    if articubot_args.pred_gripper_width:
        pred_gripper_width = pred_dict["gripper_width"]
        gt_gripper_width = extra["goal_gripper_width"].to(device, non_blocking=True).view(B, 1).float()
        loss_gripper_width = mse_loss(pred_gripper_width, gt_gripper_width)
        loss += loss_gripper_width * articubot_args.gripper_width_loss_weight
        log_info["gripper_width_loss"] = loss_gripper_width.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if scheduler is not None:
        scheduler.step()
        log_info["lr"] = scheduler.get_last_lr()[0]

    del pred_dict
    return log_info


def ddp_setup():
    os.environ["NCCL_P2P_LEVEL"] = "NVL"
    init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=5400))
    print("Local rank: ", os.environ["LOCAL_RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def train(args):
    gpu_id = int(os.environ["LOCAL_RANK"])
    device = torch.device(gpu_id)
    general_args = args.general

    if "articubot" not in list(general_args.tasks):
        raise ValueError(
            "This script only trains articubot. Include 'articubot' in general.tasks "
            f"(got {list(general_args.tasks)})."
        )

    input_channel = 5 if general_args.add_one_hot_encoding else 3
    if general_args.use_rgb:
        input_channel += 3
    if general_args.use_dino:
        input_channel += 1024

    output_dim = 13
    if general_args.policy_class == "pointnet2":
        from test_PointNet2.model_invariant import PointNet2_super_multitask

        policy_class = PointNet2_super_multitask
    elif general_args.policy_class == "pointnext":
        from test_PointNet2.model_invariant import PointNet2_super_next_multitask

        policy_class = PointNet2_super_next_multitask
    elif general_args.policy_class == "pointnext_fp":
        from test_PointNet2.model_invariant import PointNet2_super_next_fp_multitask

        policy_class = PointNet2_super_next_fp_multitask
    elif general_args.policy_class == "pointnet2_attn":
        from test_PointNet2.model_invariant import PointNet2_super_multitask_attn

        policy_class = PointNet2_super_multitask_attn
    else:
        raise ValueError(f"Unknown policy_class: {general_args.policy_class}")

    if general_args.category_embedding_type == "one_hot":
        embedding_dim = args.num_categories
    elif general_args.category_embedding_type == "siglip":
        embedding_dim = 768
    else:
        embedding_dim = None

    model = policy_class(
        num_classes=output_dim,
        keep_gripper_in_fps=general_args.keep_gripper_in_fps,
        input_channel=input_channel,
        first_sa_point=general_args.first_sa_point,
        fp_to_full=general_args.fp_to_full,
        replace_bn_w_gn=general_args.replace_bn_with_gn,
        replace_bn_w_in=general_args.replace_bn_with_in,
        embedding_dim=embedding_dim,
        film_in_sa_and_fp=general_args.film_in_sa_and_fp,
        embedding_as_input=general_args.embedding_as_input,
        replace_bn_w_ln=general_args.replace_bn_with_ln,
        pred_gripper_width=args.articubot.pred_gripper_width,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    cprint(f"model has parameters {total_params}", "red")

    if general_args.load_model_path is not None:
        load = torch.load(general_args.load_model_path, map_location=device)
        old_sd = load["model"]
        new_sd = model.state_dict()
        filtered_sd = {}
        for k, v in old_sd.items():
            if k in new_sd and v.shape == new_sd[k].shape:
                filtered_sd[k] = v
            else:
                print(
                    f"Skipping {k}: checkpoint shape {v.shape} != model shape "
                    f"{new_sd.get(k, None).shape if k in new_sd else 'N/A'}"
                )
        new_sd.update(filtered_sd)
        model.load_state_dict(new_sd, strict=False)
        print("Successfully load model and optimizer from: ", general_args.load_model_path)

    model.train()
    model = DDP(model, device_ids=[gpu_id])

    if general_args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=general_args.lr
        )
    elif general_args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=general_args.lr
        )
    else:
        raise ValueError(f"Unknown optimizer: {general_args.optimizer}")

    if general_args.get("use_lr_scheduler", False):
        total_steps = general_args.num_iterations
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    else:
        scheduler = None

    output_dir = str(datetime.date.today())
    output_dir += general_args.exp_name
    general_args.exp_path = os.path.join(general_args.exp_path, output_dir)
    wandb_run = None
    if is_main_process():
        if not os.path.exists(general_args.exp_path):
            os.makedirs(general_args.exp_path)
        wandb_run = wandb.init(
            project="articubot_multitask",
            name=str(output_dir),
            dir=str(general_args.exp_path),
        )
        cfg_dict = OmegaConf.to_container(args, resolve=True)
        wandb.config.update(cfg_dict)
        with open(os.path.join(general_args.exp_path, "config.json"), "w") as f:
            json.dump(cfg_dict, f, indent=4)

    articubot_dataloader = setup_articubot_dataloader(args.articubot)
    dataloader_len = len(articubot_dataloader)
    train_frequency = int(args.articubot.train_frequency)

    if general_args.category_embedding_type == "siglip":
        siglip_text_features = torch.load(
            "../siglip_text_features_w_pick_and_place_w_grasp_and_lift.pt"
        )
        siglip_text_features = siglip_text_features["values"]
    else:
        siglip_text_features = None

    num_iterations = int(general_args.num_iterations)
    global_step = 0
    epoch = 0

    while global_step < num_iterations:
        articubot_dataloader.sampler.set_epoch(epoch)
        for batch in articubot_dataloader:
            if global_step >= num_iterations:
                break
            if global_step % train_frequency == 0:
                beg = time.time()
                log = compute_articubot_loss(
                    batch,
                    model,
                    optimizer,
                    device,
                    args.articubot,
                    siglip_features=siglip_text_features,
                    scheduler=scheduler,
                )
                time_cost = time.time() - beg
                if is_main_process() and wandb_run is not None:
                    all_logs = {f"articubot_{k}": v for k, v in log.items()}
                    all_logs["articubot_time"] = time_cost
                    all_logs["articubot_epoch"] = global_step / max(dataloader_len, 1)
                    wandb_run.log(all_logs, step=global_step)

                if is_main_process() and global_step % general_args.save_freq == 0:
                    save_path = f"{general_args.exp_path}/model_{global_step + 1}.pth"
                    save_dict = {
                        "model": model.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                    torch.save(save_dict, save_path)

            global_step += 1
        epoch += 1

    print("Finished Training")


def load_and_parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arg_configs",
        nargs="*",
        type=str,
        default=[],
        help="overwrite config parameters",
    )
    args = parser.parse_args()

    this_file_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(this_file_path, "configs/config.yaml")
    cfg = OmegaConf.load(config_path)
    cli_cfg = OmegaConf.from_dotlist(args.arg_configs)
    cfg = OmegaConf.merge(cfg, cli_cfg)
    return cfg


if __name__ == "__main__":
    args = load_and_parse_config()
    ddp_setup()
    train(args)
    destroy_process_group()
