from test_PointNet2.dataset_from_disk import get_dataloader, get_dataloader_from_pickle
import torch
from test_PointNet2.model import PointNet2_small2
from test_PointNet2.model_attn import AttnModel
from tqdm import tqdm
import argparse
import einops

def train(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.model_type == 'pointnet2':
        model = PointNet2_small2(num_classes=13).to(device)
    elif args.model_type == 'attn':
        model = AttnModel(num_classes=13).to(device)
    else:
        raise ValueError(f"model_type {args.model_type} not recognized")
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    # dataloader = get_dataloader(all_obj_paths=args.all_zarr_path, batch_size=args.batch_size, beg_ratio=args.beg_ratio, end_ratio=args.end_ratio, shuffle=True, only_first_stage=args.only_first_stage)
    dataloader = get_dataloader_from_pickle(all_obj_paths=args.all_zarr_path, batch_size=args.batch_size, beg_ratio=args.beg_ratio, end_ratio=args.end_ratio, shuffle=True, only_first_stage=args.only_first_stage)

    for epoch in range(args.num_epochs):
        running_loss = 0.0
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
            loss = criterion(outputs, labels)

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

            loss = loss + avg_loss * 10

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (i+1) % 1000 == 0:
                print(f"Epoch {epoch + 1}, iter {i + 1}, loss: {running_loss / 1000}")
                running_loss = 0.0

        if (epoch + 1) % args.save_freq == 0:
            save_path = f"{args.exp_path}/model_{epoch + 1}.pth"
            torch.save(model.state_dict(), save_path)

    print('Finished Training')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all_zarr_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--beg_ratio', type=float, default=0)
    parser.add_argument('--end_ratio', type=float, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--only_first_stage', action='store_true')
    parser.add_argument('--exp_path', type=str, default="/project_data/held/ziyuw2/Robogen-sim2real/test_PointNet2/exps")
    parser.add_argument('--model_type', type=str, default='pointnet2')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)