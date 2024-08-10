from test_PointNet2.model import PointNet2_small2
import torch
from tqdm import tqdm
from test_PointNet2.dataset import get_dataloader
import argparse
import einops
import matplotlib.pyplot as plt

def visualize_output(pointcloud, output, gt):
    """
    pointcloud: (1, N, 3)
    output: (1, 4, 3)
    gt: (1, 4, 3)
    """
    pointcloud = pointcloud[0].detach().cpu().numpy()
    output = output[0].detach().cpu().numpy()
    gt = gt[0].detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], c='b', marker='o', s=1)
    ax.scatter(output[:, 0], output[:, 1], output[:, 2], c='r', marker='o', s=10)
    ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], c='g', marker='o', s=10)
    plt.show()

def draw_l2_distribution(l2_distribution, path=None):
    # l2: B
    if path is None:
        path = 'displacement_weighted/l2_distribution.png'
    plt.figure()
    plt.hist(l2_distribution, bins=20, density=True, stacked=True)
    plt.savefig(path)
    # plt.show()

def visualize_weight_pointcloud(pointcloud, weights):
    """
    pointcloud: (1, N, 3)
    weights: (1, N)
    """
    pointcloud = pointcloud[0].detach().cpu().numpy()
    weights = weights[0].detach().cpu().numpy()

    # normalize the weights
    weights = (weights - weights.min()) / (weights.max() - weights.min())

    colors = plt.cm.jet(weights)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], c=colors, marker='o', s=1)
    plt.colorbar(plt.cm.ScalarMappable(cmap='jet'))
    plt.show()


def eval(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PointNet2_small2(num_classes=13).to(device)
    if device == torch.device('cpu'):
        model.load_state_dict(torch.load(args.load_model_path, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(args.load_model_path))
    model.eval()

    dataset = get_dataloader(zarr_path=args.zarr_path, batch_size=args.batch_size, beg_ratio=args.beg_ratio, end_ratio=args.end_ratio, shuffle=True)

    displacement_l2 = []
    l2_distribution = []

    for i, data in enumerate(tqdm(dataset)):
        inputs, _, gripper_points = data
        inputs, gripper_points = inputs.to(device), gripper_points.to(device)
        inputs_ = inputs.permute(0, 2, 1)
        outputs = model(inputs_) # B, N, 13
        weights = outputs[:, :, -1] # B, N
        outputs = outputs[:, :, :-1] # B, N, 12


        B, N, _ = outputs.shape
        outputs = outputs.view(B, N, 4, 3)
        outputs = outputs + inputs.unsqueeze(2) # B, N, 4, 3

        # softmax the weights
        weights = torch.nn.functional.softmax(weights, dim=1)

        # sum the displacement of the predicted gripper point cloud according to the weights
        outputs = outputs * weights.unsqueeze(-1).unsqueeze(-1)
        outputs = outputs.sum(dim=1)

        if i < args.num_visualize and args.visualize:
            # visualize_output(inputs, outputs, gripper_points)
            visualize_weight_pointcloud(inputs, weights)

        l2 = torch.norm(outputs - gripper_points, dim=-1).detach()
        # import pdb; pdb.set_trace()
        l2_distribution.append(l2.mean(dim=1))

        displacement_l2.append(l2.mean().item())

    print(f"Average l2 distance: {sum(displacement_l2) / len(displacement_l2)}")
    l2_distribution = torch.cat(l2_distribution, dim=0).detach().cpu().numpy()
    draw_l2_distribution(l2_distribution)

    print('Finished Evaluation')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zarr_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--beg_ratio', type=float, default=0)
    parser.add_argument('--end_ratio', type=float, default=1.0)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--exp_path', type=str, default="/project_data/held/ziyuw2/Robogen-sim2real/test_PointNet2/exps")
    parser.add_argument('--load_model_path', type=str, default="/project_data/held/ziyuw2/Robogen-sim2real/test_PointNet2/exps/model_100.pth")
    parser.add_argument('--num_visualize', type=int, default=20)
    parser.add_argument('--visualize', action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    eval(args)
