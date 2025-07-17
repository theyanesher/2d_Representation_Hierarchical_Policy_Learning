import torch
from test_PointNet2.model_attn import AttnModel
from tqdm import tqdm
import argparse
import einops
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime
import os
from torch.utils.data import DataLoader
from test_PointNet2.dataset_from_disk import get_dataset_from_pickle
import wandb
import numpy as np
from termcolor import cprint
from itertools import cycle
from omegaconf import OmegaConf
import json

### cgn imports
from test_PointNet2.cgn.acronym_dataloader import AcryonymDataset
from test_PointNet2.cgn import utils as cgn_utils
from test_PointNet2.cgn.cgn_loss import ContactGraspnetLoss

def setup_articubot_dataloader(args):
    dataset = get_dataset_from_pickle(all_obj_paths=args.all_zarr_path, beg_ratio=args.beg_ratio, end_ratio=args.end_ratio, 
                                      use_all_data=args.use_all_data, 
                                      dataset_prefix=args.dataset_prefix, 
                                      num_train_objects=args.num_train_objects)
    dataloader = DataLoader(dataset, 
                shuffle=False,
                sampler=DistributedSampler(dataset),
                batch_size=args.batch_size,
                num_workers=8, 
                pin_memory=True,
                )
    
    return dataloader

def setup_cgn_dataloader(global_config, device):
    batch_size = global_config['OPTIMIZER']['batch_size']
    num_workers = 6  # Increase after debug
    train_dataset = AcryonymDataset(global_config, train=True, device=device, use_saved_renders=True)
    test_dataset = AcryonymDataset(global_config, train=False, device=device, use_saved_renders=True)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
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
    
    return train_dataloader
    

def compute_articubot_loss(data, model, optimizer, device, args):
    pointcloud, gripper_pcd, goal_gripper_pcd, cat_idx, class_weight = data
    class_weight = class_weight.to(device)
    gripper_points = goal_gripper_pcd
    
    if args.add_one_hot_encoding:
        # for pointcloud, we add (1, 0)
        # for gripper_pcd, we add (0, 1)
        pointcloud_one_hot = torch.zeros(pointcloud.shape[0], pointcloud.shape[1], 2)
        pointcloud_one_hot[:, :, 0] = 1
        pointcloud_ = torch.cat([pointcloud, pointcloud_one_hot], dim=2)
        gripper_pcd_one_hot = torch.zeros(gripper_pcd.shape[0], gripper_pcd.shape[1], 2)
        gripper_pcd_one_hot[:, :, 1] = 1
        gripper_pcd_ = torch.cat([gripper_pcd, gripper_pcd_one_hot], dim=2)
        inputs = torch.cat([pointcloud_, gripper_pcd_], dim=1) # B, N+4, 5
    else:
        inputs = torch.cat([pointcloud, gripper_pcd], dim=1) # B, N+4, 3

    labels = gripper_points.unsqueeze(1) - inputs[:, :, :3].unsqueeze(2)
    B, N, _, _ = labels.shape
    labels = labels.view(B, N, -1) # B, N, 12

    inputs, labels = inputs.to(device), labels.to(device)
    inputs = inputs.permute(0, 2, 1)

    outputs = model(inputs) # B, N, 13
    weights = outputs[:, :, -1] # B, N
    outputs = outputs[:, :, :-1] # B, N, 12 ### now the outputs is a per-point Gaussian
    if args.output_obj_pcd_only:
        weights = weights[:, :-4]
        outputs = outputs[:, :-4, :]
        labels = labels[:, :-4, :]
        inputs = inputs[:, :, :-4]
        N = N - 4
        
    diff = outputs - labels  # Shape: (B, N, 12)
    fixed_variance = args.fixed_variance
    exponent = -0.5 * torch.sum((diff ** 2) / fixed_variance, dim=2)  # Shape: (B, N), sum over the guassian dimension
    log_gaussians = exponent 

    # Compute log mixing coefficients
    log_mixing_coeffs = torch.log_softmax(weights, dim=1) # softmax the weight along the per-point dimension, shape B, N
    log_mixing_coeffs = torch.clamp(log_mixing_coeffs, min=-20)  # Prevent extreme values

    max_log = torch.max(log_gaussians + log_mixing_coeffs, dim=1, keepdim=True).values # get the per-batch max log along all the points, B, 1
    log_probs = max_log.squeeze(1) + torch.logsumexp(log_gaussians + log_mixing_coeffs - max_log, dim=1) # B,
    
    loss = -torch.mean(log_probs * class_weight)  # B,
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
        
    log_info = {
        'loss': loss.item(),
    }

    return loss, log_info

def compute_cgn_loss(data, model, optimizer, device, global_config):
    loss_fn = ContactGraspnetLoss(global_config, device).to(device)
    
    cgn_utils.send_dict_to_device(data, device)
    # Target contains input and target values
    pc_cam = data['pc_cam']

    pred = model(pc_cam)
    loss, loss_info = loss_fn(pred, data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss, loss_info

def ddp_setup():
    os.environ["NCCL_P2P_LEVEL"] = "NVL"
    init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=5400))
    print("Local rank: ", os.environ["LOCAL_RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def train(args):    
    ### setup model
    gpu_id = int(os.environ["LOCAL_RANK"])
    device = torch.device(gpu_id)
    input_channel = 5 if args.add_one_hot_encoding else 3
    output_dim = 13 
    from test_PointNet2.model_invariant import PointNet2_super
    model = PointNet2_super(num_classes=output_dim, keep_gripper_in_fps=args.keep_gripper_in_fps, input_channel=input_channel).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    cprint(f"model has parameters {total_params}", "red")
    if args.load_model_path is not None:
        model.load_state_dict(torch.load(args.load_model_path, map_location=device))
        print("Successfully load model from: ", args.load_model_path)
    model.train()
    model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    ### setup logging
    output_dir = "GMM_" 
    output_dir = output_dir + "_" + str(datetime.date.today())
    output_dir += args.exp_name
    args.exp_path = os.path.join(args.exp_path, output_dir)
    if os.environ['LOCAL_RANK'] == '0':
        if not os.path.exists(args.exp_path):
            os.makedirs(args.exp_path)
        wandb_run = wandb.init(
                project="articubot_multitask",
                name=str(output_dir),
                dir=str(args.exp_path),
            )
        
        cfg_dict = OmegaConf.to_container(args, resolve=True)
        wandb.config.update(cfg_dict)

        # save the config file
        with open(os.path.join(args.exp_path, "config.json"), "w") as f:
            json.dump(cfg_dict, f, indent=4)
            
    ### setup all dataloaders
    articubot_dataloader = setup_articubot_dataloader(args.articubot)
    cgn_dataloader = setup_cgn_dataloader(args.cgn, device)
    
    all_task_dataloaders = {
        "articubot": articubot_dataloader,
        "cgn": cgn_dataloader, 
    }
    all_tasks = list(all_task_dataloaders.keys())
    all_dataloaders = [all_task_dataloaders[task] for task in all_tasks]
    dataloader_iters = [cycle(loader) for loader in all_dataloaders]
    
    forward_functions = {
        "articubot": compute_articubot_loss,
        "cgn": None,  # TODO: set up contact graspnet forward function
    }
    
    num_iterations = args.num_epochs * len(articubot_dataloader) // args.batch_size
    for global_step in range(num_iterations):  
        
        samples = [next(it) for it in dataloader_iters]
        all_logs = {}
        for task_idx in range(len(all_tasks)):
            task = all_tasks[task_idx]
            forward_func = forward_functions[all_tasks[task_idx]]
            loss, log = forward_func(samples[task_idx], model, optimizer, device, args)
            for key in log:
                all_logs[f"{task}_{key}"] = log[key]
        
        if os.environ['LOCAL_RANK'] == '0':
            ### TODO: log the losses here
            for task in all_tasks:
                dataloader_length = len(all_task_dataloaders[task])
                epoch = global_step * args.batch_size / dataloader_length
                all_logs[f"{task}_epoch"] = epoch
                
            wandb_run.log(all_logs, step=global_step)
            
            ### TODO: save the model here
            if (global_step + 1) % args.save_freq == 0:
                save_path = f"{args.exp_path}/model_{global_step + 1}.pth"
                torch.save(model.module.state_dict(), save_path)

    print('Finished Training')


def load_and_parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
    args = parser.parse_args()

    this_file_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(this_file_path, "configs/config.yaml")
    cfg = OmegaConf.load("config.yaml")
    cli_cfg = OmegaConf.from_dotlist(args.arg_configs)
    cfg = OmegaConf.merge(cfg, cli_cfg)
    return cfg

if __name__ == "__main__":
    args = load_and_parse_config()
    ddp_setup()
    train(args)
    destroy_process_group()