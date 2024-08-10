from test_PointNet2.dataset import get_dataloader
import torch
from test_PointNet2.model import PointNet2_small2
from tqdm import tqdm
import argparse

def train(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PointNet2_small2(num_classes=1).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    dataloader = get_dataloader(zarr_path=args.zarr_path, batch_size=args.batch_size, beg_ratio=args.beg_ratio, end_ratio=args.end_ratio, shuffle=True)

    # use tqdm to show progress bar and loss
    for epoch in range(args.num_epochs):
        running_loss = 0.0
        for i, data in enumerate(tqdm(dataloader)):
            inputs, labels, _ = data
            inputs, labels = inputs.to(device), labels.to(device).float()
            # give more weight to the near positive points
            pos_weight = torch.ones_like(labels)
            pos_weight[labels == 1] = 100

            inputs = inputs.permute(0, 2, 1)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs[:, :, 0]
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--beg_ratio', type=float, default=0)
    parser.add_argument('--end_ratio', type=float, default=0.9)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--exp_path', type=str, default="/project_data/held/ziyuw2/Robogen-sim2real/test_PointNet2/exps")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)

