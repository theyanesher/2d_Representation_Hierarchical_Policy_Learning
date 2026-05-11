# from test_PointNet2.dataset_from_disk import get_dataloader, get_dataloader_from_pickle
import torch
# from test_PointNet2.model_attn import AttnModel
from tqdm import tqdm
import argparse
# import einops
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime
import os
from torch.utils.data import DataLoader
from third_party.robogen.test_PointNet2.dataset_from_disk import get_dataset_from_pickle
import wandb
from termcolor import cprint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all_zarr_path', type=str, default=None)
    parser.add_argument('--num_train_objects', default='square_D2_abs')
    parser.add_argument('--dataset_prefix', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--beg_ratio', type=float, default=0)
    parser.add_argument('--end_ratio', type=float, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--only_first_stage', action='store_true')
    parser.add_argument('--exp_path', type=str, default="/project_data/held/mnakuraf/RoboGen-sim2real/test_PointNet2/exps")
    parser.add_argument('--model_type', type=str, default='pointnet2')
    parser.add_argument('--load_model_path', type=str, default=None)
    parser.add_argument('--output_obj_pcd_only', action='store_true')
    parser.add_argument('--weight_loss_weight', type=float, default=10)
    parser.add_argument('--use_all_data', action='store_true')
    parser.add_argument('--use_combined_action', action='store_true')
    parser.add_argument('--model_invariant', action='store_true')
    parser.add_argument('--predict_two_goals', action='store_true')
    parser.add_argument('--keep_gripper_in_fps', type=int, default=0)
    parser.add_argument('--add_one_hot_encoding', type=int, default=0)
    parser.add_argument('--using_weight', type=int, default=1)
    parser.add_argument('--exp_name', type=str, default="")
    parser.add_argument('--n_obs_steps', type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset = get_dataset_from_pickle(all_obj_paths=args.all_zarr_path, beg_ratio=args.beg_ratio,
                                      end_ratio=args.end_ratio, only_first_stage=args.only_first_stage,
                                      use_all_data=args.use_all_data, use_combined_action=args.use_combined_action, 
                                      dataset_prefix=args.dataset_prefix, num_train_objects=args.num_train_objects,
                                      predict_two_goals=args.predict_two_goals, n_obs_steps=args.n_obs_steps)
    
    for i in range(10000):
        pointcloud, gripper_pcd, goal_gripper_pcd, gripper_pcd_history = dataset[i]
        cprint(f'pointcloud shape: {pointcloud.shape}', 'cyan')
        cprint(f'gripper_pcd shape: {gripper_pcd.shape}', 'cyan')
        cprint(f'goal_gripper_pcd shape: {goal_gripper_pcd.shape}', 'cyan')
        cprint(f'gripper_pcd_history shape: {gripper_pcd_history.shape}', 'cyan')

        # --- plot --------------------------------------------------
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(pointcloud[:,0], 
                   pointcloud[:,1], 
                   pointcloud[:,2],
                   s=1, c='b', label='pointcloud')

        ax.scatter(goal_gripper_pcd[:,0],
                   goal_gripper_pcd[:,1],
                   goal_gripper_pcd[:,2],
                   s=30, c='g', marker='^', label='goal gripper')

        ax.scatter(gripper_pcd[:,0],
                   gripper_pcd[:,1],
                   gripper_pcd[:,2],
                   s=30, c='r', marker='s', label='gripper')

        hist_pts = gripper_pcd_history.reshape(-1, 3)
        ax.scatter(hist_pts[:,0],
                   hist_pts[:,1],
                   hist_pts[:,2],
                   s=30, c='gray', label='history')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.title(f"Sample #{i}")
        plt.tight_layout()
        plt.show()
        # -----------------------------------------------------------
