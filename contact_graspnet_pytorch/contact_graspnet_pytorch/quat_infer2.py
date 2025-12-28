from genericpath import exists
import os
import sys
import argparse
from datetime import datetime
import numpy as np
import time
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch

os.environ['PYOPENGL_PLATFORM'] = 'egl'  # To get pyrender to work headless

# Import pointnet library
CONTACT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))

sys.path.append(os.path.join(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, 'Pointnet_Pointnet2_pytorch'))

import config_utils
from acronym_dataloader import AcryonymDataset
from contact_graspnet_pytorch.contact_graspnet import ContactGraspnet, ContactGraspnetLoss
from contact_graspnet_pytorch import utils
from contact_graspnet_pytorch.checkpoints import CheckpointIO 
from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
import torch.nn.functional as F

def build_6d_grasp_from_four_points(four_point_head, gripper_depth = 0.1034):
    B, N, _, _ = four_point_head.shape
    grasp_t = four_point_head[:, :, 0].unsqueeze(3)  # B x N x 3 x 1
    
    
    approach_direction = four_point_head[:, :, -1] - four_point_head[:, :, 0]  # B x N x 3
    baseline_direction = four_point_head[:, :, 2] - four_point_head[:, :, 1]  # B x N x 3
    
    # baseline_direction_normed = F.normalize(baseline_direction, p=2, dim=2)  # B x N x 3
    # dot_product = torch.sum(approach_direction * baseline_direction_normed, dim=2, keepdim=True)  # B x N x 1
    # projection = dot_product * baseline_direction_normed  # B x N x 3
    # approach_direction_orthog = F.normalize(approach_direction - projection, p=2, dim=2)  # B x N x 3
    # grasp_R = torch.stack([baseline_direction_normed, torch.cross(approach_direction_orthog, baseline_direction_normed),approach_direction_orthog], dim=3)  # B x N x 3 x 3
    
    approach_direction_normed = F.normalize(approach_direction, p=2, dim=2)  # B x N x 3
    dot_product = torch.sum(baseline_direction * approach_direction_normed, dim=2, keepdim=True)  # B x N x 1
    projection = dot_product * approach_direction_normed  # B x N x 3
    baseline_direction_orthog = F.normalize(baseline_direction - projection, p=2, dim=2)  # B x N x 3
    grasp_R = torch.stack([baseline_direction_orthog, torch.cross(approach_direction_normed, baseline_direction_orthog),approach_direction_normed], dim=3)  # B x N x 3 x 3
    
    ones = torch.ones((B, N, 1, 1), dtype=torch.float32).to(four_point_head.device)  # B x N x 1 x 1
    zeros = torch.zeros((B, N, 1, 3), dtype=torch.float32).to(four_point_head.device)  # B x N x 1 x 3
    homog_vec = torch.cat([zeros, ones], dim=3)  # B x N x 1 x 4
    grasps = torch.cat([torch.cat([grasp_R, grasp_t], dim=3), homog_vec], dim=2)  # B x N x 4 x 4
    
    offset = torch.norm(four_point_head[:, :, 2] - four_point_head[:, :, 1], dim=-1, keepdim=True)  # B x N x 1
    
    return grasps, offset

def batched_rotation_distance(R1, R2):
    # R1, R2: shape (B, N, 3, 3)
    R_diff = np.matmul(np.transpose(R1, (0, 1, 3, 2)), R2)  # R1ᵀ * R2, shape (B, N, 3, 3)
    trace = np.trace(R_diff, axis1=-2, axis2=-1)            # trace along the last 2 dims → (B, N)
    cos_theta = (trace - 1) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)               # numerical stability
    theta = np.arccos(cos_theta)                            # in radians
    return np.rad2deg(theta)  # shape (B, N)

def get_bin_vals(global_config):
    """
    Creates bin values for grasping widths according to bounds defined in config

    Arguments:
        global_config {dict} -- config

    Returns:
        tf.constant -- bin value tensor
    """
    bins_bounds = np.array(global_config['DATA']['labels']['offset_bins'])
    if global_config['TEST']['bin_vals'] == 'max':
        bin_vals = (bins_bounds[1:] + bins_bounds[:-1])/2
        bin_vals[-1] = bins_bounds[-1]
    elif global_config['TEST']['bin_vals'] == 'mean':
        bin_vals = bins_bounds[1:]
    else:
        raise NotImplementedError

    if not global_config['TEST']['allow_zero_margin']:
        bin_vals = np.minimum(bin_vals, global_config['DATA']['gripper_width']-global_config['TEST']['extra_opening'])
        
    return bin_vals

def compute_labels( processed_pc_cams: torch.Tensor, 
                    camera_poses: torch.Tensor, 
                    pos_contact_points: torch.Tensor,
                    pos_contact_dirs: torch.Tensor,
                    pos_finger_diffs: torch.Tensor, 
                    pos_approach_dirs: torch.Tensor):
    """
    Project grasp labels defined on meshes onto rendered point cloud 
    from a camera pose via nearest neighbor contacts within a maximum radius. 
    All points without nearby successful grasp contacts are considered 
    negative contact points.

    Here N is the number of points returned by the PointNet Encoder (2048) while
    M is the number of points in the ground truth data.  B is the batch size.
    We are trying to assign a label to each of the PointNet points by 
    sampling the nearest ground truth points.

    Arguments:
        pc_cam_pl (torch.Tensor): (B, N, 3) point cloud in camera frame
        camera_pose_pl (torch.Tensor): (B, 4, 4) homogenous camera pose
        pos_contact_points (torch.Tensor): (B, M, 3) contact points in world frame (3 DoF points)
        pos_contact_dirs (torch.Tensor): (B, M, 3) contact directions (origin centered vectors?)
        pos_finger_diffs (torch.Tensor): (B, M, ) finger diffs in world frame  (scalar distances)
        pos_approach_dirs (torch.Tensor): (B, M, 3) approach directions in world frame (origin centered vectors?)
    """
    label_config = global_config['DATA']['labels']

    nsample = label_config['k']  # Currently set to 1
    radius = label_config['max_radius']
    filter_z = label_config['filter_z']
    z_val = label_config['z_val']

    _, N, _ = processed_pc_cams.shape
    B, M, _ = pos_contact_points.shape

    # -- Make sure pcd is B x N x 3 -- #
    if processed_pc_cams.shape[2] != 3:
        xyz_cam = processed_pc_cams[:,:,:3]  # N x 3
    else:
        xyz_cam = processed_pc_cams

    # -- Transform Ground Truth to Camera Frame -- #
    # Transform contact points to camera frame  (This is a homogenous transform)
    # We use matmul to accommodate batch
    # pos_contact_points_cam = pos_contact_points @ (camera_poses[:3,:3].T) + camera_poses[:3,3][None,:]
    pos_contact_points_cam = torch.matmul(pos_contact_points, camera_poses[:, :3, :3].transpose(1, 2)) \
        + camera_poses[:,:3,3][:, None,:]

    # Transform contact directions to camera frame (Don't translate because its a direction vector)
    # pos_contact_dirs_cam = pos_contact_dirs @ camera_poses[:3,:3].T
    pos_contact_dirs_cam = torch.matmul(pos_contact_dirs, camera_poses[:, :3,:3].transpose(1, 2))
    
    # Make finger diffs B x M x 1
    pos_finger_diffs = pos_finger_diffs[:, :, None]

    # Transform approach directions to camera frame (Don't translate because its a direction vector)
    # pos_approach_dirs_cam = pos_approach_dirs @ camera_poses[:3,:3].T
    pos_approach_dirs_cam = torch.matmul(pos_approach_dirs, camera_poses[:, :3,:3].transpose(1, 2))

    # -- Filter Direction -- #
    # TODO: Figure out what is going on here
    if filter_z:
        # Filter out directions that are too far
        dir_filter_passed = (pos_contact_dirs_cam[:, :, 2:3] > z_val).repeat(1, 1, 3)
        pos_contact_points_cam = torch.where(dir_filter_passed, 
                                                pos_contact_points_cam, 
                                                torch.ones_like(pos_contact_points_cam) * 10000)
    
    # -- Compute Distances -- #
    # We want to compute the distance between each point in the point cloud and each contact point
    # We can do this by expanding the dimensions of the tensors and then summing the squared differences
    xyz_cam_expanded = torch.unsqueeze(xyz_cam, 2)  # B x N x 1 x 3
    pos_contact_points_cam_expanded = torch.unsqueeze(pos_contact_points_cam, 1)  # B x 1 x M x 3
    squared_dists_all = torch.sum((xyz_cam_expanded - pos_contact_points_cam_expanded)**2, dim=3)  # B x N x M

    # B x N x k, B x N x k
    squared_dists_k, close_contact_pt_idcs = torch.topk(squared_dists_all, 
        k=nsample, dim=2, largest=False, sorted=False)

    # -- Group labels -- #
    grouped_contact_dirs_cam = utils.index_points(pos_contact_dirs_cam, close_contact_pt_idcs)  # B x N x k x 3
    grouped_finger_diffs = utils.index_points(pos_finger_diffs, close_contact_pt_idcs)  # B x N x k x 1
    grouped_approach_dirs_cam = utils.index_points(pos_approach_dirs_cam, close_contact_pt_idcs)  # B x N x k x 3

    # grouped_contact_dirs_cam = pos_contact_dirs_cam[close_contact_pt_idcs, :]  # B x N x k x 3
    # grouped_finger_diffs = pos_finger_diffs[close_contact_pt_idcs]  # B x N x k x 1
    # grouped_approach_dirs_cam = pos_approach_dirs_cam[close_contact_pt_idcs, :]  # B x N x k x 3

    # -- Compute Labels -- #
    # Take mean over k nearest neighbors and normalize
    dir_label = grouped_contact_dirs_cam.mean(dim=2)  # B x N x 3
    dir_label = F.normalize(dir_label, p=2, dim=2)  # B x N x 3

    diff_label = grouped_finger_diffs.mean(dim=2)# B x N x 1

    approach_label = grouped_approach_dirs_cam.mean(dim=2)  # B x N x 3
    approach_label = F.normalize(approach_label, p=2, dim=2)  # B x N x 3

    grasp_success_label = torch.mean(squared_dists_k, dim=2, keepdim=True) < radius**2  # B x N x 1 
    grasp_success_label = grasp_success_label.type(torch.float32)  

    # debug = dict(
    #     xyz_cam = xyz_cam,
    #     pos_contact_points_cam = pos_contact_points_cam,
    # )
    debug = {}


    return dir_label, diff_label, grasp_success_label, approach_label, debug


def get_gt_4_points(target, pred_points):
    pos_contact_points = target['pos_contact_points']    # B x M x 3
    pos_contact_dirs = target['pos_contact_dirs']        # B x M x 3
    pos_finger_diffs = target['pos_finger_diffs']        # B x M
    pos_approach_dirs = target['pos_approach_dirs']      # B x M x 3
    camera_pose = target['camera_pose']                  # B x 4 x 4

    dir_labels_pc_cam, \
    grasp_offset_labels_pc, \
    grasp_success_labels_pc, \
    approach_labels_pc_cam, \
    debug = compute_labels(pred_points, 
                                camera_pose,
                                pos_contact_points,
                                pos_contact_dirs,
                                pos_finger_diffs,
                                pos_approach_dirs)
    
    thickness_gt = grasp_offset_labels_pc[:, :, 0]
    gt_grasps_proj = utils.build_6d_grasp(approach_labels_pc_cam, dir_labels_pc_cam, pred_points, thickness_gt, use_torch=True, device=pos_contact_points.device) # b x N x 4 x 4
    # Select positive grasps I think?
    success_mask = grasp_success_labels_pc.bool()[:, :, :, None] # B x N x 1 x 1
    success_mask = torch.broadcast_to(success_mask, gt_grasps_proj.shape) # B x N x 4 x 4
    pos_gt_grasps_proj = torch.where(success_mask, gt_grasps_proj, torch.ones_like(gt_grasps_proj) * 100000) # B x N x 4 x 4

    pose, gripper_width = pos_gt_grasps_proj, grasp_offset_labels_pc
    
    return get_4_points_from_pose(pose.cpu().numpy(), gripper_width.cpu().numpy()), pose
    
def get_4_points_from_pose(pose, gripper_width):
    first_point = pose[..., :3, 3]
    z_dir = pose[..., :3, 2]  # Approach direction
    last_point = first_point + 0.1034 * z_dir  # 0.1034 is the gripper depth
    mid_point = first_point + 0.08 * z_dir  # TODO: get and fix this 0.08 thing
    finger_open_close_dir = pose[..., :3, 0]  # Base direction
    left_point = mid_point + finger_open_close_dir * (gripper_width / 2)
    right_point = mid_point - finger_open_close_dir * (gripper_width / 2)
    # if not flip:
    return np.stack([first_point, left_point, right_point, last_point], axis=-2)
    # else:
    #     return torch.stack([first_point, right_point, left_point, last_point], dim=-2)  # B x N x 4 x 3

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# ckpt_dir = "checkpoints/contact_graspnet"
ckpt_dir = "checkpoints/articubot_gmm_no_sym_grad_schmit"
# ckpt_dir = "checkpoints/gmm-no-sigmoid"
# ckpt_dir = "checkpoints/test_4_point_training"
data_path = "/project_data/held/yufeiw2/contact_graspnet_pytorch/acronym"
batch_size = 6
global_config = config_utils.load_config(ckpt_dir, None, batch_size=batch_size, data_path=data_path, save=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### set up the test dataset
num_workers = 0  # Increase after debug
global_config['DATA']['num_test_scenes'] = 100
test_dataset = AcryonymDataset(global_config, train=False, device=device, use_saved_renders=True, eval_with_fixed_cam=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers)

### load the pretrained model
grasp_estimator = GraspEstimator(global_config)
model_checkpoint_dir = os.path.join(ckpt_dir, 'checkpoints')
checkpoint_io = CheckpointIO(checkpoint_dir=model_checkpoint_dir, model=grasp_estimator.model)
# load_dict = checkpoint_io.load('model_best.pt')
load_dict = checkpoint_io.load('model.pt')
grasp_network = grasp_estimator.model

grasp_network.eval()

top_k = 100
with torch.no_grad():
    four_point_dist = []
    rotational_dist = []
    for val_it, data in enumerate(tqdm(test_dataloader)):
        # print("Validation iteration: ", val_it)
        
        utils.send_dict_to_device(data, device)
        # Target contains input and target values
        pc_cam = data['pc_cam']
        pred = grasp_network(pc_cam)
        
        ### get the gt 4 points
        pred_points = pred['pred_points']  
        gt_4_points, gt_pose = get_gt_4_points(data, pred_points)  # B x N x 4 x 3
        gt_rotation = gt_pose[:, :, :3, :3].cpu().numpy()
        B, N, _, _ = gt_4_points.shape
        gt_grasp_center = gt_4_points[:, :, -1, :] # B, N,  3
        gt_grasp_center_expanded = gt_grasp_center[:, None, :, :] # B, 1, N, 3
        
        ### get the pred 4 points
        pred_grasps_cam_2 = None
        if global_config["MODEL"]['loss_mode'] == 'articubot_gmm':
            pred_scores = pred['pred_scores']                   # B x N x 1, the weights for each points
            pred_points = pred['pred_points']                   # B x N x 3
            pred_offsets = pred['pred_offsets']       # B x N x 4 x 3, the predicted displacement to the goal points
            B, N, _, _ = pred_offsets.shape

            pred_scores = pred_scores.squeeze().cpu().numpy()
            pred_points = pred_points.unsqueeze(2).cpu().numpy() # B x N x 1 x 3
            pred_offsets = pred_offsets.cpu().numpy() # B X N x 4 x 3
            
            pred_4_points = pred_points + pred_offsets      
            pred_grasps_cam = pred['pred_grasps_cam'].cpu().numpy()           # B x N x 4 x 4
            flipped_pred_4_points = pred_4_points.copy()
            flipped_pred_4_points[:, :, 1] = pred_4_points[:, :, 2]
            flipped_pred_4_points[:, :, 2] = pred_4_points[:, :, 1]
            pred_grasps_cam_2, _ = build_6d_grasp_from_four_points(torch.from_numpy(flipped_pred_4_points))
            
        elif global_config["MODEL"]['loss_mode'] == 'contact_graspnet':
            # import pdb; pdb.set_trace()
            pred_grasps_cam = pred['pred_grasps_cam'].cpu().numpy()           # B x N x 4 x 4
            pred_scores = pred['pred_scores'].squeeze().cpu().numpy()                   # B x N x 1
            pred_points = pred['pred_points'].cpu().numpy()                   # B x N x 3
            grasp_offset_head = pred['grasp_offset_head'].permute(0, 2, 1).cpu().numpy()       # B x N x 10
            bin_vals = get_bin_vals(global_config)
            thickness_pred = bin_vals[np.argmax(grasp_offset_head, axis=2)]
            pred_4_points = get_4_points_from_pose(pred_grasps_cam, np.expand_dims(thickness_pred, -1))  # B x N x 4 x 3
        elif global_config['MODEL']['loss_mode'] == 'contact_graspnet_4_points':
            print("Using this branch")
            pred_grasps_cam = pred['pred_grasps_cam'].cpu().numpy()           # B x N x 4 x 4
            pred_scores = pred['pred_scores'].squeeze().cpu().numpy()                   # B x N x 1
            pred_points = pred['pred_points'].cpu().numpy()  
            pred_4_points = pred['pred_4_points'].cpu().numpy()  
            

        top_k_score_idx = np.argsort(-pred_scores, axis=-1)
        top_k_4_point_pred = pred_4_points[np.arange(B)[:, None], top_k_score_idx][:, :top_k]
            
        top_k_grasp_center = top_k_4_point_pred[:, :, -1, :] # B, k, 3
        top_k_grasp_center_expanded = top_k_grasp_center[:, :, None, :] # B, k, 1, 3
        dist = top_k_grasp_center_expanded - gt_grasp_center_expanded # B, k, N, 3
        dist = np.sum(dist ** 2, -1) # B, k, N
        min_dist_idx = np.argmin(dist, axis=-1) # B, k
        
        ### compute the 4 point error
        min_dist_gt_4_points = gt_4_points[np.arange(B)[:, None], min_dist_idx] # B, k, 4, 3        
        dist = np.linalg.norm(top_k_4_point_pred - min_dist_gt_4_points, axis=-1)
        dist = np.mean(dist)
        four_point_dist.append(dist)
        
        ### compute the rotation error
        min_dist_gt_rotation = gt_rotation[np.arange(B)[:, None], min_dist_idx]
        pred_top_k_rotation = pred_grasps_cam[np.arange(B)[:, None], top_k_score_idx][:, :top_k][:, :, :3, :3]
        rotation_dist = batched_rotation_distance(pred_top_k_rotation, min_dist_gt_rotation)
        if pred_grasps_cam_2 is not None:
            pred_top_k_rotation2 = pred_grasps_cam_2[np.arange(B)[:, None], top_k_score_idx][:, :top_k][:, :, :3, :3]
            rotation_dist2 = batched_rotation_distance(pred_top_k_rotation2, min_dist_gt_rotation)
            rotation_dist = np.minimum(rotation_dist, rotation_dist2)
        rotation_dist = np.mean(rotation_dist)
        rotational_dist.append(rotation_dist)
        
            
    four_point_dist = np.mean(four_point_dist)
    rotational_dist = np.mean(rotational_dist)

print(f"{ckpt_dir}: four point loss: {four_point_dist:.4f} rotational diff: {rotational_dist:.4f}")    


