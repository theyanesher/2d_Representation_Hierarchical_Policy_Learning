import torch
from tqdm import tqdm
import argparse
import einops
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime
import os
from torch.utils.data import DataLoader
import wandb
import numpy as np
from termcolor import cprint
from omegaconf import OmegaConf
import json
import random

### articubot imports
from test_PointNet2.dataset_from_disk import get_dataset_from_pickle

### cgn imports
from test_PointNet2.cgn.acronym_dataloader import AcryonymDataset
from test_PointNet2.cgn import utils as cgn_utils
from test_PointNet2.cgn.cgn_loss import ContactGraspnetLoss

def infinite_loader(dl):
    while True:
        for batch in dl:
            yield batch

def setup_articubot_dataloader(args):
    dataset = get_dataset_from_pickle(all_obj_paths=args.all_zarr_path, beg_ratio=args.beg_ratio, end_ratio=args.end_ratio, 
                                      use_all_data=args.use_all_data, 
                                      dataset_prefix=args.dataset_prefix, 
                                      num_train_objects=args.num_train_objects,
                                      camera_frame=args.camera_frame)
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
    # test_dataset = AcryonymDataset(global_config, train=False, device=device, use_saved_renders=True)

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
    
mse_loss = torch.nn.MSELoss()
def compute_articubot_loss(data, model, optimizer, device, args, siglip_features=None, loss_fn=None):
    pointcloud, gripper_pcd, goal_gripper_pcd, cat_idx, class_weight = data
    class_weight = class_weight.to(device)
    gripper_points = goal_gripper_pcd.to(device)
    
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

    
    inputs = inputs.to(device)
    inputs = inputs.permute(0, 2, 1)

    if siglip_features is None:
        pred_dict = model(inputs) 
    else:
        # print("articubot using siglip embedding")
        cat_embedding = siglip_features[cat_idx].float()
        pred_dict = model(inputs, embedding=cat_embedding)
        
    outputs = pred_dict['pred_offsets']
    pred_points = pred_dict['pred_points'] 
    weights = pred_dict['pred_scores'].squeeze(-1)
        
    labels = gripper_points.unsqueeze(1) - pred_points.unsqueeze(2)
    B, N, _, _ = labels.shape
    labels = labels.view(B, N, -1) # B, N, 12
    
    # weights = outputs[:, :, -1] # B, N
    # outputs = outputs[:, :, :-1] # B, N, 12 ### now the outputs is a per-point Gaussian
    if args.output_obj_pcd_only:
        weights = weights[:, :-4]
        outputs = outputs[:, :-4, :]
        labels = labels[:, :-4, :]
        inputs = inputs[:, :, :-4]
        N = N - 4
    
    outputs = outputs.view(B, N, -1)
    # fixed_variance = args.fixed_variance

    if args.gmm:
        diff = outputs - labels  # Shape: (B, N, 12)
        fixed_variance = random.choice(args.fixed_variance)
        exponent = -0.5 * torch.sum((diff ** 2) / fixed_variance, dim=2)  # Shape: (B, N), sum over the guassian dimension
        log_gaussians = exponent 

        # Compute log mixing coefficients
        log_mixing_coeffs = torch.log_softmax(weights, dim=1) # softmax the weight along the per-point dimension, shape B, N
        log_mixing_coeffs = torch.clamp(log_mixing_coeffs, min=-20)  # Prevent extreme values

        max_log = torch.max(log_gaussians + log_mixing_coeffs, dim=1, keepdim=True).values # get the per-batch max log along all the points, B, 1
        log_probs = max_log.squeeze(1) + torch.logsumexp(log_gaussians + log_mixing_coeffs - max_log, dim=1) # B,
        
        loss = -torch.mean(log_probs * class_weight)  # B,
    else:
        per_point_loss = mse_loss(outputs, labels)
        inputs = pred_points
        outputs = outputs.view(B, N, 4, 3)
        outputs = outputs + inputs[:, :, :3].unsqueeze(2) # B, N, 4, 3

        # softmax the weights
        weights = torch.nn.functional.softmax(weights, dim=1)
        
        # sum the displacement of the predicted gripper point cloud according to the weights
        outputs = outputs * weights.unsqueeze(-1).unsqueeze(-1)
        outputs = outputs.sum(dim=1)
        avg_loss = mse_loss(outputs, gripper_points)
        loss = per_point_loss + avg_loss * args.weight_loss_weight
        
    loss = loss * args.loss_scale
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
        
    if args.gmm:
        log_info = {
            'loss_{}'.format(fixed_variance): loss.item(),
        }
    
    if not args.gmm:
        log_info = {"loss": loss.item()}
        log_info['perpoint_loss'] = per_point_loss.item()
        log_info['weighted_average_loss'] = avg_loss.item()

    del pred_dict
    return log_info

def compute_cgn_loss(data, model, optimizer, device, global_config, siglip_features=None, loss_fn=None):    
    cgn_utils.send_dict_to_device(data, device)
    # Target contains input and target values
    pc_cam = data['pc_cam']
    # import pdb; pdb.set_trace()
    pc_cam = pc_cam.permute(0, 2, 1)

    if siglip_features is None:
        pred = model(pc_cam)
    else:
        # print("cgn use siglip embedding")
        embedding = siglip_features[-1].float().unsqueeze(0).repeat(pc_cam.shape[0], 1)
        pred = model(pc_cam, embedding)
        
    loss, loss_info = loss_fn(pred, data)
    loss = loss * global_config.loss_scale
    keys = list(loss_info.keys())
    for key in keys:
        loss_info[key + "_scaled"] = loss_info[key] * global_config.loss_scale
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    del data, pred
    return loss_info

def ddp_setup():
    os.environ["NCCL_P2P_LEVEL"] = "NVL"
    init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=5400))
    print("Local rank: ", os.environ["LOCAL_RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def train(args):    
    ### setup model
    gpu_id = int(os.environ["LOCAL_RANK"])
    device = torch.device(gpu_id)
    general_args = args.general
    input_channel = 5 if general_args.add_one_hot_encoding else 3
    output_dim = 13 
    from test_PointNet2.model_invariant import PointNet2_super_multitask
    
    if general_args.category_embedding_type == "one_hot":
        embedding_dim = args.num_categories
    elif general_args.category_embedding_type == "siglip":
        embedding_dim = 768
    else:
        embedding_dim = None
    
    model = PointNet2_super_multitask(num_classes=output_dim, keep_gripper_in_fps=general_args.keep_gripper_in_fps, input_channel=input_channel,
                                      first_sa_point=general_args.first_sa_point,
                                      fp_to_full=general_args.fp_to_full,
                                      replace_bn_w_gn=general_args.replace_bn_with_gn,
                                      replace_bn_w_in=general_args.replace_bn_with_in,
                                      embedding_dim=embedding_dim,
                                      film_in_sa_and_fp=general_args.film_in_sa_and_fp,
                                      embedding_as_input=general_args.embedding_as_input,
                                      replace_bn_w_ln=general_args.replace_bn_with_ln,
                                      ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    cprint(f"model has parameters {total_params}", "red")
    
    if general_args.load_model_path is not None:
        load = torch.load(general_args.load_model_path, map_location=device)
        model.load_state_dict(load['model'])
        # optimizer.load_state_dict(load['optimizer'])
        print("Successfully load model and optimizer from: ", general_args.load_model_path)
        
    model.train()
    # print(model)
    # exit()
    # import pdb; pdb.set_trace()
    model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)
    if general_args.optimizer == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=general_args.lr)
    elif general_args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=general_args.lr)
    
    ### setup logging
    output_dir = "GMM_" 
    output_dir = output_dir + "_" + str(datetime.date.today())
    output_dir += general_args.exp_name
    general_args.exp_path = os.path.join(general_args.exp_path, output_dir)
    if os.environ['LOCAL_RANK'] == '0':
        if not os.path.exists(general_args.exp_path):
            os.makedirs(general_args.exp_path)
        wandb_run = wandb.init(
                project="articubot_multitask",
                name=str(output_dir),
                dir=str(general_args.exp_path),
            )
        
        cfg_dict = OmegaConf.to_container(args, resolve=True)
        wandb.config.update(cfg_dict)

        # save the config file
        with open(os.path.join(general_args.exp_path, "config.json"), "w") as f:
            json.dump(cfg_dict, f, indent=4)
            
    ### setup all dataloaders
    all_tasks = args.general.tasks
    
    articubot_dataloader = setup_articubot_dataloader(args.articubot) if "articubot" in all_tasks else None
    cgn_dataloader = setup_cgn_dataloader(args.cgn, device) if 'cgn' in all_tasks else None
    
    all_task_dataloaders = {
        "articubot": articubot_dataloader,
        "cgn": cgn_dataloader, 
    }
    all_dataloaders = [all_task_dataloaders[task] for task in all_tasks]
    all_dataloaders = [x for x in all_dataloaders if x is not None]
    dataloader_iters = [infinite_loader(loader) for loader in all_dataloaders]
    dataloader_lengths = [len(loader) for loader in all_dataloaders]
    
    forward_functions = {
        "articubot": compute_articubot_loss,
        "cgn": compute_cgn_loss,  # TODO: set up contact graspnet forward function
    }
    
    train_frequency = {
        "articubot": args.articubot.train_frequency,
        'cgn': args.cgn.train_frequency,
    }
    
    if general_args.category_embedding_type == "siglip":
        siglip_text_features = torch.load("../siglip_text_features.pt")
    else:
        siglip_text_features = None
    
    cgn_loss_fn = ContactGraspnetLoss(args.cgn, device).to(device) if 'cgn' in args.general.tasks else None
    loss_funcs = {
        'cgn': cgn_loss_fn,
        'articubot': None
    }
    
    num_iterations = args.general.num_iterations
    for global_step in range(num_iterations):  
        
        samples = [next(it) for it in dataloader_iters]
        all_logs = {}
        for task_idx in range(len(all_tasks)):
            task = all_tasks[task_idx]
            forward_func = forward_functions[all_tasks[task_idx]]
            if global_step % train_frequency[task] == 0:
                log = forward_func(samples[task_idx], model, optimizer, device, args[task], siglip_features=siglip_text_features, loss_fn=loss_funcs[task])
                for key in log:
                    assert not torch.is_tensor(log[key])
                    all_logs[f"{task}_{key}"] = log[key]
        
        if os.environ['LOCAL_RANK'] == '0':
            ### TODO: log the losses here
            for task in all_tasks:
                dataloader_length = len(all_task_dataloaders[task])
                epoch = global_step / dataloader_length
                all_logs[f"{task}_epoch"] = epoch
                
            wandb_run.log(all_logs, step=global_step)
            
            print(f"{global_step} {all_logs}")
            
            ### TODO: save the model here
            if (global_step + 1) % args.general.save_freq == 0:
                save_path = f"{general_args.exp_path}/model_{global_step + 1}.pth"
                save_dict = {
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    
                }
                torch.save(save_dict, save_path)
        
        torch.cuda.empty_cache()        
        torch.cuda.ipc_collect()         

    print('Finished Training')


def load_and_parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
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