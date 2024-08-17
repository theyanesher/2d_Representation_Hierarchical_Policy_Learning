from test_PointNet2.dataset import get_dataloader
import torch
from test_PointNet2.model import PointNet2_small2
from tqdm import tqdm
import argparse
import einops

def train(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PointNet2_small2(num_classes=12).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    dataloader = get_dataloader(zarr_path=args.zarr_path, batch_size=args.batch_size, beg_ratio=args.beg_ratio, end_ratio=args.end_ratio, shuffle=True)

    for epoch in range(args.num_epochs):
        running_loss = 0.0
        for i, data in enumerate(tqdm(dataloader)):
            inputs, _, gripper_points = data
            # inputs: B, N, 3
            # gripper_points: B, 4, 3
            # calculate the displacement from every point to the gripper to get the labels with shape B, N, 4, 3
            labels = gripper_points.unsqueeze(1) - inputs.unsqueeze(2)
            B, N, _, _ = labels.shape
            labels = labels.view(B, N, -1) # B, N, 12

            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.permute(0, 2, 1)
            optimizer.zero_grad()
            outputs = model(inputs) # B, N, 12
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, loss: {running_loss / len(dataloader)}")

        if (epoch + 1) % args.save_freq == 0:
            save_path = f"{args.exp_path}/model_{epoch + 1}.pth"
            torch.save(model.state_dict(), save_path)

    print('Finished Training')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zarr_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--beg_ratio', type=float, default=0)
    parser.add_argument('--end_ratio', type=float, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--exp_path', type=str, default="/project_data/held/ziyuw2/Robogen-sim2real/test_PointNet2/exps")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)