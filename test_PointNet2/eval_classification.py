from test_PointNet2.dataset import get_dataloader, visualize_pointcloud
import torch
from test_PointNet2.model import PointNet2_small2
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

def visualize_output(pointcloud, output):
    # pointcloud: (1, 3, N)
    # output: (1, N, 1)
    label = output[0]
    label = label > 0
    label = label[0]
    pointcloud = pointcloud.permute(0, 2, 1)[0]

    true_points = pointcloud[label == 1].detach().cpu().numpy().T
    false_points = pointcloud[label == 0].detach().cpu().numpy().T
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(true_points[0], true_points[1], true_points[2], c='r', marker='o',s=10)
    ax.scatter(false_points[0], false_points[1], false_points[2], c='b', marker='o',s=1)
    plt.show()

def draw_l2_distribution(l2_distribution, path=None):
    # l2: B
    if path is None:
        path = 'classification/l2_distribution.png'
    plt.figure()
    plt.hist(l2_distribution, bins=20, density=True, stacked=True)
    plt.savefig(path)
    # plt.show()


def eval(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PointNet2_small2(num_classes=1).to(device)
    # load the model 
    if device == torch.device('cpu'):
        model.load_state_dict(torch.load(args.load_model_path, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(args.load_model_path))
    model.eval()
    
    dataloader = get_dataloader(zarr_path=args.zarr_path, batch_size=args.batch_size, beg_ratio=args.beg_ratio, end_ratio=args.end_ratio, shuffle=True)

    # calculate the accuracy, precision, recall, f1 score
    all_accuracy = []
    all_precision = []
    all_recall = []
    all_f1 = []

    # calculate the average l2 distance between the predicted gripper point cloud and the ground truth gripper point cloud
    all_l2 = []
    all_min_l2 = []

    for i, data in enumerate(tqdm(dataloader)):
        if i > args.num_visualize and args.visualize:
            break
        inputs, labels, _ = data
        inputs, labels = inputs.to(device), labels.to(device).long()
        inputs = inputs.permute(0, 2, 1)
        outputs = model(inputs)
        # import pdb; pdb.set_trace()
        if args.visualize:
            visualize_pointcloud(inputs[0].permute(1, 0).detach().cpu().numpy(), binary_mask=labels[0].detach().cpu().numpy())
            outputs = outputs.permute(0, 2, 1)
            visualize_output(inputs, outputs)

        # outputs B, N, 1
        # labels B, N
        outputs = outputs.squeeze(-1)
        outputs = outputs > 0
        labels = labels > 0
        if torch.sum(outputs) > 0 and torch.sum(labels) > 0:
            TP = torch.sum(outputs & labels)
            FP = torch.sum(outputs & ~labels)
            FN = torch.sum(~outputs & labels)
            TN = torch.sum(~outputs & ~labels)
            accuracy = (TP + TN) / (TP + FP + FN + TN)
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 = 2 * precision * recall / (precision + recall)
            all_accuracy.append(accuracy)
            all_precision.append(precision)
            all_recall.append(recall)
            if precision + recall != 0:
                all_f1.append(f1)

            # inpus B, 3, N
            inputs = inputs.permute(0, 2, 1)
            true_label_points = inputs[labels == 1] # N, 3
            pred_label_points = inputs[outputs == 1] # M, 3
            # calculate the l2 distance between the predicted gripper point cloud and the closest ground truth gripper point cloud
            l2 = torch.norm(true_label_points.unsqueeze(0) - pred_label_points.unsqueeze(1), dim=-1) # M, N
            if l2.size(0) > 0 and l2.size(1) > 0:

                min_l2 = torch.min(l2, dim=-1)[0]
                all_min_l2.append(torch.mean(min_l2))
                # calculate the average l2 distance between the predicted gripper point cloud and the ground truth gripper point cloud
                all_l2.append(torch.mean(l2))


    print('Accuracy: ', sum(all_accuracy) / len(all_accuracy))
    print('Precision: ', sum(all_precision) / len(all_precision))
    print('Recall: ', sum(all_recall) / len(all_recall))
    print('F1: ', sum(all_f1) / len(all_f1))
    print("----- L2 Distance [Only meanful when eval batch size 1] -----")
    print('Average L2 distance: ', sum(all_l2) / len(all_l2))
    print('Average Min L2 distance: ', sum(all_min_l2) / len(all_min_l2))

    import pdb; pdb.set_trace()
    all_l2 = torch.tensor(all_l2).detach().cpu().numpy()
    draw_l2_distribution(all_l2)
    all_min_l2 = torch.tensor(all_min_l2).detach().cpu().numpy()
    draw_l2_distribution(all_min_l2, path='classification/min_l2_distribution.png')

    print('Finished Testing')

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
    parser.add_argument('--num_visualize', type=int, default=10)
    parser.add_argument('--visualize', action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    eval(args)

