from test_PointNet2.dataset_from_disk import get_dataloader, get_dataloader_from_pickle
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

def ddp_setup():
    os.environ["NCCL_P2P_LEVEL"] = "NVL"
    init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=5400))
    print("Local rank: ", os.environ["LOCAL_RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def train(args):
    gpu_id = int(os.environ["LOCAL_RANK"])
    device = torch.device(gpu_id)

    input_channel = 5 if args.add_one_hot_encoding else 3
    if args.conditioning_on_demo and not args.demo_cross_attn_bottleneck and not args.demo_hadamard_production and not args.demo_aligned_cross_attn and not args.bottleneck_film_cond:
        input_channel += args.demo_attn_embedding_dim
    if args.conditioning_on_demo and args.demo_just_use_pn:
        input_channel = 5 + 3

    output_dim = 13 

    if args.model_invariant:
        if args.conditioning_on_demo:
            from test_PointNet2.model_condition import PointNet2_super
            model = PointNet2_super(num_classes=output_dim, keep_gripper_in_fps=args.keep_gripper_in_fps, 
                                    input_channel=input_channel, 
                                    cross_attn_bottleneck=args.demo_cross_attn_bottleneck,
                                    use_hadamard_production=args.demo_hadamard_production,
                                    aligned_cross_attn=args.demo_aligned_cross_attn,
                                    attn_embedding_dim=args.demo_attn_embedding_dim,
                                    demo_use_attn=args.demo_use_attn,
                                    demo_pn_type=args.demo_pn_type,
                                    demo_use_cur_obs=args.demo_use_cur_obs,
                                    use_flow_in_demo=args.demo_use_flow,
                                    separate_demo_feature=args.separate_demo_feature,
                                    cross_attn_every_layer=args.cross_attn_every_layer,
                                    bottleneck_film_cond=args.bottleneck_film_cond,
                                    always_train_with_conditioning=args.demo_pretrained_pn_path is not None or args.always_train_with_conditioning,
                                    condition_set_to_false=args.condition_set_to_false,
                                    just_use_pn=args.demo_just_use_pn,
                                    condition_prob=args.demo_condition_prob,
                                    small_film=args.small_film,
                                    ).to(device)
            
            total_params = sum(p.numel() for p in model.parameters())
            cprint(f"model has parameters {total_params}", "red")
            # exit()
            
                
            if args.demo_pretrained_pn_path is not None:
                # import pdb; pdb.set_trace()
                cprint("load partially trained pointnet++ from {}".format(args.demo_pretrained_pn_path), "red")
                # loaded_submodules = {k.split(".")[0] for k in checkpoint.keys()}  # Extract top-level module names
                # model.load_state_dict(torch.load(args.demo_pretrained_pn_path, map_location=device), strict=False)
                # for name, module in model.named_children():
                #     if name != "demo_transformer":  # If this submodule was in the checkpoint
                #         for param in module.parameters():
                #             param.requires_grad = False  # Freeze it
                
                checkpoint = torch.load(args.demo_pretrained_pn_path, map_location=device)  # Replace with actual checkpoint

                # Get the submodules in the checkpoint
                loaded_submodules = {k.split(".")[0] for k in checkpoint.keys()}  # Extract top-level module names

                # Load only the checkpointed submodules
                filtered_state_dict = {k: v for k, v in checkpoint.items() if k.split(".")[0] in loaded_submodules}
                model.load_state_dict(filtered_state_dict, strict=False)  # Ignore missing keys
                for name, module in model.named_children():
                    cprint(f"checking module {name}", "red")
                    if name in loaded_submodules:  # If this submodule was in the checkpoint
                        cprint(f"freezing {name}", 'yellow')
                        for param in module.parameters():
                            param.requires_grad = False  # Freeze it
                # import pdb; pdb.set_trace()
                
        else:
            from test_PointNet2.model_invariant import PointNet2_super
            model = PointNet2_super(num_classes=output_dim, keep_gripper_in_fps=args.keep_gripper_in_fps, input_channel=input_channel,
                                    ).to(device)
            
            total_params = sum(p.numel() for p in model.parameters())
            cprint(f"model has parameters {total_params}", "red")
            # exit()

    else:
        from test_PointNet2.model import PointNet2_small2
        from test_PointNet2.model import PointNet2
        from test_PointNet2.model import PointNet2_super
        if args.model_type == 'pointnet2':
            model = PointNet2_small2(num_classes=output_dim).to(device)
        elif args.model_type == 'pointnet2_large':
            model = PointNet2(num_classes=output_dim).to(device)
        elif args.model_type == 'pointnet2_super':
            model = PointNet2_super(num_classes=output_dim).to(device)
        elif args.model_type == 'attn':
            model = AttnModel(num_classes=output_dim).to(device)
        else:
            raise ValueError(f"model_type {args.model_type} not recognized")
        
        
    if args.load_model_path is not None:
        model.load_state_dict(torch.load(args.load_model_path, map_location=device))
        print("Successfully load model from: ", args.load_model_path)
    
    model.train()

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = torch.nn.MSELoss()
    
    # for name, param in model.named_parameters():
    #     print(f"{name}: requires_grad={param.requires_grad}")  # Should be True only for `another_model`

    if args.gmm:
        output_dir = "GMM_" 
    else:
        output_dir = "Reg_"
    
    output_dir = output_dir + "_" + str(datetime.date.today())

    if args.conditioning_on_demo:
        output_dir += '_demo'

        if not args.separate_demo_feature:
            output_dir += '_not_separate'
        if args.demo_use_attn:
            output_dir += "_demo_attn"
        if args.demo_use_cur_obs:
            output_dir += "_demo_curobs"
        if args.demo_just_use_pn:
            output_dir += "_just_use_pn"
        output_dir += "_pn_" + args.demo_pn_type
        if args.demo_cross_attn_bottleneck:
            output_dir += "_attn_bottleneck"
        if args.demo_hadamard_production:
            output_dir += "_hadamard"
        if args.demo_aligned_cross_attn:
            output_dir += "_aligned_cross_attn"
        if args.bottleneck_film_cond:
            output_dir +="_Film_bottleneck"
        if args.cross_attn_every_layer:
            output_dir +="_attn_every_upsample"
        if args.demo_pretrained_pn_path is not None:
            output_dir += "_load_pretrained_pn"
        if args.always_train_with_conditioning:
            output_dir += "_always_condition"
        if args.condition_set_to_false:
            output_dir += "_half_cond_false"
        if args.small_film:
            output_dir += '_small_film'
        output_dir += f"_cond_prob_{args.demo_condition_prob}"
    
    # output_dir += args.model_type 
    # if args.model_invariant:
    #     output_dir = output_dir + "_model_invariant"
    # if args.use_all_data:
    #     output_dir = output_dir + "_use_all_data"
    # else:
    #     output_dir = output_dir + "_use_75_episodes"
    # if args.use_combined_action:
    #     output_dir = output_dir + "_use_combined_data"
    
    output_dir = output_dir + "_" + str(args.num_train_objects) + "-obj"
    
    if args.output_obj_pcd_only:
        output_dir = output_dir + "_output_obj_only"
        
    if args.only_first_stage:
        output_dir = output_dir + "_only_first_stage"
        
    if args.keep_gripper_in_fps:
        output_dir = output_dir + "_keep_gripper_in_fps"
        
    if args.add_one_hot_encoding:
        output_dir = output_dir + "_one_hot"
        
    # output_dir += args.optimizer

    
    if not args.using_weight:
        output_dir = output_dir + "_no_weight"
        
    output_dir += args.exp_name
    
    args.exp_path = os.path.join(args.exp_path, output_dir)


    gpu_id = int(os.environ["LOCAL_RANK"])
    model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)

    if os.environ['LOCAL_RANK'] == '0':
        if not os.path.exists(args.exp_path):
            os.makedirs(args.exp_path)
        wandb_run = wandb.init(
                project="pointnet-weighted-displacement",
                name=str(output_dir),
                dir=str(args.exp_path),
            )
        wandb.config.update(
            {
                "output_dir": args.exp_path,
                "model_type": args.model_type,
                "lr": args.lr,
                "weight_loss_weight": args.weight_loss_weight,
                "batch_size": args.batch_size
            }
        )
        
        config_dict = args.__dict__
        wandb.config.update(config_dict)

        # save the config file
        with open(os.path.join(args.exp_path, "config.txt"), "w") as f:
            for key, value in config_dict.items():
                f.write(f"{key}: {value}\n")

    print("trying to load dataset")
    dataset = get_dataset_from_pickle(all_obj_paths=args.all_zarr_path, beg_ratio=args.beg_ratio, end_ratio=args.end_ratio, 
                                      only_first_stage=args.only_first_stage, use_all_data=args.use_all_data, 
                                      use_combined_action=args.use_combined_action, dataset_prefix=args.dataset_prefix, 
                                      num_train_objects=args.num_train_objects, conditioning_on_demo=args.conditioning_on_demo)
    dataloader = DataLoader(dataset, 
                shuffle=False,
                sampler=DistributedSampler(dataset),
                batch_size=args.batch_size,
                num_workers=8, 
                pin_memory=True,
                )

    global_step = 0

    for epoch in range(args.num_epochs):
        running_loss = 0.0
        accumulated_displacement_loss = 0.0
        accumulated_weighting_loss = 0.0
        for i, data in enumerate(tqdm(dataloader)):
            if not args.conditioning_on_demo:
                pointcloud, gripper_pcd, goal_gripper_pcd = data
            else:
                #import pdb; pdb.set_trace();
                data = {k: v.to('cuda') for k, v in data.items()}
                pointcloud, gripper_pcd, goal_gripper_pcd = data['pointcloud'], data['gripper_pcd'], data['goal_gripper_pcd']
                #import pdb; pdb.set_trace();
                
            # inputs: B, N, 3
            # gripper_pcd: B, 4, 3
            # goal_gripper_points: B, 4, 3
            # calculate the displacement from every point to the gripper to get the labels with shape B, N, 4, 3
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

            optimizer.zero_grad()
            if not args.conditioning_on_demo:
                outputs = model(inputs) # B, N, 13
            else:
                outputs = model(inputs, data)
            weights = outputs[:, :, -1] # B, N
            outputs = outputs[:, :, :-1] # B, N, 12 ### now the outputs is a per-point Gaussian
            if args.output_obj_pcd_only:
                weights = weights[:, :-4]
                outputs = outputs[:, :-4, :]
                labels = labels[:, :-4, :]
                inputs = inputs[:, :, :-4]
                N = N - 4
                
            if args.gmm:
                diff = outputs - labels  # Shape: (B, N, 12)
                fixed_variance = args.fixed_variance
                exponent = -0.5 * torch.sum((diff ** 2) / fixed_variance, dim=2)  # Shape: (B, N), sum over the guassian dimension
                log_gaussians = exponent 

                # Compute log mixing coefficients
                log_mixing_coeffs = torch.log_softmax(weights, dim=1) # softmax the weight along the per-point dimension, shape B, N
                log_mixing_coeffs = torch.clamp(log_mixing_coeffs, min=-10)  # Prevent extreme values

                max_log = torch.max(log_gaussians + log_mixing_coeffs, dim=1, keepdim=True).values # get the per-batch max log along all the points, B, 1
                log_probs = max_log.squeeze(1) + torch.logsumexp(log_gaussians + log_mixing_coeffs - max_log, dim=1) # B,
                
                        
                loss = -torch.mean(log_probs) # mean of the negative log likelihood
                accumulated_displacement_loss += loss.item()

               
            else:
                loss = criterion(outputs, labels)
                accumulated_displacement_loss += loss.item()

                if args.using_weight:
                    inputs = inputs.permute(0, 2, 1)
                    outputs = outputs.view(B, N, 4, 3)
                    outputs = outputs + inputs[:, :, :3].unsqueeze(2) # B, N, 4, 3

                    # softmax the weights
                    weights = torch.nn.functional.softmax(weights, dim=1)
                    
                    # sum the displacement of the predicted gripper point cloud according to the weights
                    outputs = outputs * weights.unsqueeze(-1).unsqueeze(-1)
                    outputs = outputs.sum(dim=1)
                    avg_loss = criterion(outputs, gripper_points.to(device))

                    loss = loss + avg_loss * args.weight_loss_weight
                    accumulated_weighting_loss += (avg_loss * args.weight_loss_weight).item()

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            log_interval = 50 if args.gmm else 1000
            if (i+1) % log_interval == 0 and os.environ['LOCAL_RANK'] == '0':
                print(f"Epoch {epoch + 1}, iter {i + 1}, loss: {running_loss / log_interval}")
                
                log_info = {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "total_loss": running_loss / log_interval,
                    "displacement_loss": accumulated_displacement_loss / log_interval,
                    "weighting_loss": accumulated_weighting_loss / log_interval,
                }

                wandb_run.log(log_info, step=global_step)

                running_loss = 0.0
                accumulated_displacement_loss = 0.0

            global_step += 1

        if (epoch + 1) % args.save_freq == 0 and os.environ['LOCAL_RANK'] == '0':
            save_path = f"{args.exp_path}/model_{epoch + 1}.pth"
            torch.save(model.module.state_dict(), save_path)

    print('Finished Training')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all_zarr_path', type=str, default=None)
    parser.add_argument('--num_train_objects', type=str, default=200)
    parser.add_argument('--dataset_prefix', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--beg_ratio', type=float, default=0)
    parser.add_argument('--end_ratio', type=float, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--only_first_stage', action='store_true')
    parser.add_argument('--exp_path', type=str, default="/project_data/held/ziyuw2/Robogen-sim2real/test_PointNet2/exps")
    parser.add_argument('--model_type', type=str, default='pointnet2')
    parser.add_argument('--load_model_path', type=str, default=None)
    parser.add_argument('--output_obj_pcd_only', action='store_true')
    parser.add_argument('--weight_loss_weight', type=float, default=10)
    parser.add_argument('--use_all_data', action='store_true')
    parser.add_argument('--use_combined_action', action='store_true')
    parser.add_argument('--model_invariant', action='store_true')
    parser.add_argument('--keep_gripper_in_fps', type=int, default=0)
    parser.add_argument('--add_one_hot_encoding', type=int, default=0)
    parser.add_argument('--using_weight', type=int, default=1)
    parser.add_argument('--exp_name', type=str, default="")
    parser.add_argument('--fixed_variance', type=float, default=0.05)
    parser.add_argument('--conditioning_on_demo', type=int, default=0)
    parser.add_argument('--demo_attn_embedding_dim', type=int, default=240)
    parser.add_argument('--demo_use_attn', type=int, default=0)
    parser.add_argument('--demo_use_cur_obs', type=int, default=0)
    parser.add_argument('--demo_pn_type', type=str, default='large')
    parser.add_argument('--demo_cross_attn_bottleneck', type=int, default=0)
    parser.add_argument('--demo_hadamard_production', type=int, default=0)
    parser.add_argument('--demo_aligned_cross_attn', type=int, default=0)
    parser.add_argument('--separate_demo_feature', type=int, default=1)
    parser.add_argument('--demo_use_flow', type=int, default=1)
    parser.add_argument('--demo_pretrained_pn_path', type=str, default=None)
    parser.add_argument('--always_train_with_conditioning', type=int, default=0)
    parser.add_argument('--cross_attn_every_layer', type=int, default=0)
    parser.add_argument('--bottleneck_film_cond', type=int, default=0)
    parser.add_argument('--small_film', type=int, default=0)
    parser.add_argument('--condition_set_to_false', type=int, default=0)
    parser.add_argument('--demo_just_use_pn', type=int, default=0)
    parser.add_argument('--demo_condition_prob', type=float, default=0.5)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--gmm', type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ddp_setup()
    train(args)
    destroy_process_group()