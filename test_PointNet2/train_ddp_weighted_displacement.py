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

def ddp_setup():
    os.environ["NCCL_P2P_LEVEL"] = "NVL"
    init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=5400))
    print("Local rank: ", os.environ["LOCAL_RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def train(args):
    gpu_id = int(os.environ["LOCAL_RANK"])
    device = torch.device(gpu_id)
    if args.model_invariant:
        from test_PointNet2.model_invariant import PointNet2_small2
        from test_PointNet2.model_invariant import PointNet2
        from test_PointNet2.model_invariant import PointNet2_super
        if args.model_type == 'pointnet2':
            model = PointNet2_small2(num_classes=13).to(device)
        elif args.model_type == 'pointnet2_large':
            model = PointNet2(num_classes=13).to(device)
        elif args.model_type == 'pointnet2_super':
            model = PointNet2_super(num_classes=13).to(device)
        elif args.model_type == 'attn':
            model = AttnModel(num_classes=13).to(device)
        else:
            raise ValueError(f"model_type {args.model_type} not recognized")
    else:
        from test_PointNet2.model import PointNet2_small2
        from test_PointNet2.model import PointNet2
        from test_PointNet2.model import PointNet2_super
        if args.model_type == 'pointnet2':
            model = PointNet2_small2(num_classes=13).to(device)
        elif args.model_type == 'pointnet2_large':
            model = PointNet2(num_classes=13).to(device)
        elif args.model_type == 'pointnet2_super':
            model = PointNet2_super(num_classes=13).to(device)
        elif args.model_type == 'attn':
            model = AttnModel(num_classes=13).to(device)
        else:
            raise ValueError(f"model_type {args.model_type} not recognized")
    
    if args.load_model_path is not None:
        model.load_state_dict(torch.load(args.load_model_path, map_location=device))
        print("Successfully load model from: ", args.load_model_path)
    
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    # dataloader = get_dataloader(all_obj_paths=args.all_zarr_path, batch_size=args.batch_size, beg_ratio=args.beg_ratio, end_ratio=args.end_ratio, shuffle=True, only_first_stage=args.only_first_stage)
    # dataloader = get_dataloader_from_pickle(all_obj_paths=args.all_zarr_path, batch_size=args.batch_size, beg_ratio=args.beg_ratio, end_ratio=args.end_ratio, shuffle=True, only_first_stage=args.only_first_stage)
    
    output_dir = args.model_type 

    if args.model_invariant:
        output_dir = output_dir + "_model_invariant"
    
    output_dir = output_dir + "_" + str(datetime.date.today())

    if args.use_all_data:
        output_dir = output_dir + "_use_all_data"
    else:
        output_dir = output_dir + "_use_75_episodes"

    if args.use_combined_action:
        output_dir = output_dir + "_use_combined_data"
    
    output_dir = output_dir + "_" + str(args.num_train_objects) + "-obj"
    
    args.exp_path = os.path.join(args.exp_path, output_dir)


    gpu_id = int(os.environ["LOCAL_RANK"])
    model = DDP(model, device_ids=[gpu_id])

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

    print("trying to load dataset")
    dataset = get_dataset_from_pickle(all_obj_paths=args.all_zarr_path, beg_ratio=args.beg_ratio, end_ratio=args.end_ratio, only_first_stage=args.only_first_stage, use_all_data=args.use_all_data, use_combined_action=args.use_combined_action, dataset_prefix=args.dataset_prefix, num_train_objects=args.num_train_objects)
    dataloader = DataLoader(dataset, 
                shuffle=False,
                sampler=DistributedSampler(dataset),
                batch_size=args.batch_size,
                num_workers=4,
                pin_memory=True,
                )

    global_step = 0

    for epoch in range(args.num_epochs):
        running_loss = 0.0
        accumulated_displacement_loss = 0.0
        accumulated_weighting_loss = 0.0
        for i, data in enumerate(tqdm(dataloader)):
            pointcloud, gripper_pcd, goal_gripper_pcd = data
            # inputs: B, N, 3
            # gripper_pcd: B, 4, 3
            # goal_gripper_points: B, 4, 3
            # calculate the displacement from every point to the gripper to get the labels with shape B, N, 4, 3
            gripper_points = goal_gripper_pcd
            inputs = torch.cat([pointcloud, gripper_pcd], dim=1) # B, N+4, 3
            
            labels = gripper_points.unsqueeze(1) - inputs.unsqueeze(2)
            B, N, _, _ = labels.shape
            labels = labels.view(B, N, -1) # B, N, 12

            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.permute(0, 2, 1)
            optimizer.zero_grad()
            outputs = model(inputs) # B, N, 13
            weights = outputs[:, :, -1] # B, N
            outputs = outputs[:, :, :-1] # B, N, 12
            if args.output_obj_pcd_only:
                weights = weights[:, :-4]
                outputs = outputs[:, :-4, :]
                labels = labels[:, :-4, :]
                inputs = inputs[:, :, :-4]
                N = N - 4
            loss = criterion(outputs, labels)
            accumulated_displacement_loss += loss.item()

            # import pdb; pdb.set_trace()
            inputs = inputs.permute(0, 2, 1)
            outputs = outputs.view(B, N, 4, 3)
            outputs = outputs + inputs.unsqueeze(2) # B, N, 4, 3

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

            if (i+1) % 1000 == 0 and os.environ['LOCAL_RANK'] == '0':
                print(f"Epoch {epoch + 1}, iter {i + 1}, loss: {running_loss / 1000}")
                
                log_info = {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "total_loss": running_loss / 1000,
                    "displacement_loss": accumulated_displacement_loss / 1000,
                    "weighting_loss": accumulated_weighting_loss / 1000,
                }

                wandb_run.log(log_info, step=global_step)

                running_loss = 0.0
                accumulated_displacement_loss = 0.0
                accumulated_weighting_loss = 0.0

            global_step += 1

        if (epoch + 1) % args.save_freq == 0 and os.environ['LOCAL_RANK'] == '0':
            save_path = f"{args.exp_path}/model_{epoch + 1}.pth"
            torch.save(model.module.state_dict(), save_path)

    print('Finished Training')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all_zarr_path', type=str, default=None)
    parser.add_argument('--num_train_objects', type=int, default=200)
    parser.add_argument('--dataset_prefix', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--beg_ratio', type=float, default=0)
    parser.add_argument('--end_ratio', type=float, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=3)
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ddp_setup()
    train(args)
    destroy_process_group()