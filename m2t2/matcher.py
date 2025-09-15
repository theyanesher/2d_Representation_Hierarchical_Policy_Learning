# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# Author: Wentao Yuan
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
from scipy.optimize import linear_sum_assignment
from torch.cuda.amp import autocast
import numpy as np
import torch
import torch.nn.functional as F


def dice_loss_matrix(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs:  A float tensor with shape [N, ...].
                 Predicted logits for each query.
        targets: A float tensor with shape [M, ...].
                 Ground truth binary mask for each object.
    Returns:
        loss matrix of shape [N, M], averaged across all pixels/points
    """
    inputs = inputs.sigmoid()
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def bce_loss_matrix(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs:  A float tensor with shape [N, ...].
                 Predicted logits for each query.
        targets: A float tensor with shape [M, ...].
                 Ground truth binary mask for each object.
    Returns:
        loss matrix of shape [N, M], averaged across all pixels/points
    """
    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    num_points = inputs.shape[1]
    with autocast(enabled=False):
        loss = torch.einsum("nc,mc->nm", pos.float(), targets) \
             + torch.einsum("nc,mc->nm", neg.float(), (1 - targets))

    return loss / num_points

def gmm_loss(pred, target, fixed_variance=0.001, return_mean=False):
    # import pdb; pdb.set_trace()
    
    pred_scores = pred['pred_scores']                   # B x N x 1, the weights for each points
    pred_points = pred['pred_points']                   # B x N x 3
    pred_offsets = pred['pred_offset']       # B x N x 4 x 3, the predicted displacement to the goal points
    B, N, _, _ = pred_offsets.shape
    
    ### for debug
    # pred_scores = torch.ones_like(pred_scores) 
    
    if target['goal_gripper_mask'] is not None: ### cgn case
        gt_4_points = target['goal_gripper_pcd']  # B x N x 4 x 3
        gt_4_points_expanded = gt_4_points.unsqueeze(2) # B x N x 1 x 4 x 3
        pred_points_expanded = pred_points.unsqueeze(1).unsqueeze(3) # B x 1 x N x 1 x 3

        # import pdb; pdb.set_trace()
        gt_label_diff = gt_4_points_expanded - pred_points_expanded  # B x N x N x 4 x 3 ### first N is all grasps, second N is all points
        labels = gt_label_diff
        outputs = pred_offsets.unsqueeze(1)  # B x 1 x N x 4 x 3, the predicted displacement to the goal points
        diff = outputs - labels  # Shape: B x N x N x 4 x 3
        diff = diff.view(B, N, N, -1)  # Reshape to B x N x N x 12 (4 points * 3 dimensions)
        exponent = -0.5 * torch.sum((diff ** 2) / fixed_variance, dim=-1)  # Shape: (B, N, N), sum over the guassian dimension
        
        log_gaussians = exponent 
        # import pdb; pdb.set_trace()

        # Compute log mixing coefficients
        weights = pred_scores.unsqueeze(1).squeeze(-1) # B x 1 x N. expand to have a all grasp dimension
        log_mixing_coeffs = torch.log_softmax(weights, dim=2) # softmax the weight along the per-point dimension, shape B x 1 x N
        log_mixing_coeffs = torch.clamp(log_mixing_coeffs, min=-10)  # Prevent extreme values

        max_log = torch.max(log_gaussians + log_mixing_coeffs, dim=2, keepdim=True).values # get the per-batch and per-grasp max log along all the points, B, N, 1
        log_probs = max_log.squeeze(2) + torch.logsumexp(log_gaussians + log_mixing_coeffs - max_log, dim=2) # B, N over batch and all grasp dimension

        # import pdb; pdb.set_trace()
        
        binary_grasp_success_labels = target['goal_gripper_mask']  # B x N, binary labels indicating if the grasp is successful
        pos_grasps_in_view = torch.sum(binary_grasp_success_labels, dim=1, keepdim=True)  # B x 1, number of grasps in view

        log_probs = log_probs * binary_grasp_success_labels.squeeze()  # B x N
        
        log_probs = torch.sum(log_probs, dim=1)  
        log_probs = log_probs.squeeze() / pos_grasps_in_view.squeeze()  # B x 1    
        
        if return_mean:
            loss = -torch.mean(log_probs) # mean of the negative log likelihood      
        else:
            loss = -log_probs
        
        return loss                  
        
    else: ### articubot case, there is only one goal gripper pcd
        # import pdb; pdb.set_trace()
        outputs = pred['pred_offset']
        pred_points = pred['pred_points']
        weights = pred['pred_scores'].squeeze(-1)
            
        gripper_points = target['goal_gripper_pcd'].squeeze(1)
        # class_weight = target['class_weight'] ### TODO: add class weight
        labels = gripper_points.unsqueeze(1) - pred_points.unsqueeze(2)
        B, N, _, _ = labels.shape
        labels = labels.view(B, N, -1) # B, N, 12
        
        outputs = outputs.view(B, N, -1)
        # fixed_variance = args.fixed_variance

        diff = outputs - labels  # Shape: (B, N, 12)
        # fixed_variance = random.choice(args.fixed_variance)
        ### looping through these two possible variance values
        # loss = 0
        
        # for fixed_variance, variance_loss_scale in zip(args.fixed_variance, args.variance_loss_scale):
        exponent = -0.5 * torch.sum((diff ** 2) / fixed_variance, dim=2)  # Shape: (B, N), sum over the guassian dimension
        log_gaussians = exponent 

        # Compute log mixing coefficients
        log_mixing_coeffs = torch.log_softmax(weights, dim=1) # softmax the weight along the per-point dimension, shape B, N
        log_mixing_coeffs = torch.clamp(log_mixing_coeffs, min=-20)  # Prevent extreme values

        max_log = torch.max(log_gaussians + log_mixing_coeffs, dim=1, keepdim=True).values # get the per-batch max log along all the points, B, 1
        log_probs = max_log.squeeze(1) + torch.logsumexp(log_gaussians + log_mixing_coeffs - max_log, dim=1) # B,
        
        # this_loss = -torch.mean(log_probs * class_weight)  # B,
        
        if return_mean:
            return -torch.mean(log_probs)
        else:
            return -log_probs

        # loss += this_loss
        # loss += this_loss * variance_loss_scale
        
   

class HungarianMatcher(torch.nn.Module):
    """This class computes a 1-to-1 assignment between the targets and the
    network's predictions. The targets only include objects, so in general,
    there are more predictions than targets. The un-matched predictions are
    treated as non-objects).
    """
    def __init__(self, object_weight, bce_weight, dice_weight, wdp_weight=None, gmm_weight=None):
        super(HungarianMatcher, self).__init__()
        self.object_weight = object_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.wdp_weight = wdp_weight if wdp_weight is not None else 0.0
        self.gmm_weight = gmm_weight if gmm_weight is not None else 0.0

    @classmethod
    def from_config(cls, cfg):
        args = {}
        args['object_weight'] = cfg.object_weight
        args['bce_weight'] = cfg.bce_weight
        args['dice_weight'] = cfg.dice_weight
        args['wdp_weight'] = cfg.get("wdp_weight", 0.0)
        args['gmm_weight'] = cfg.get("gmm_weight", 0.0)
        return cls(**args)

    @torch.no_grad()
    def forward(self, outputs, data):
        """Performs the matching
        Params:
            outputs: a dict that contains these entries:
                "objectness":     dim [batch_size, num_queries]
                                  logits for the objectness score
                "instance_masks": dim [batch_size, num_queries, ...]
                                  predicted object instance masks
                "contact_masks":  dim [batch_size, num_queries, ...]
                                  predicted grasp contact masks
            targets: a dict that contains these entries:
                "instance_masks": a list of batch_size tensors
                                  ground truth object instance masks
                "contact_masks":  a list of batch_size tensors
                                  ground truth grasp contact masks
        Returns:
            indices: a list of length batch_size, containing indices of the
                     predictions that match the best with each target
        """
        indices, cost_matrices, target_indices = [], [], []
        # import pdb; pdb.set_trace()
        for i in range(len(outputs['objectness'])): ## NOTE: this loops over the batch 
            # NOTE: I think we need to have this objectness as well for choosig one query at test time.
            
            # We approximate objectness NLL loss with 1 - prob.
            # The 1 is a constant that can be ommitted.
            # print("in matching, i is : ", i)
            if not self.gmm_weight > 0:
                if "pred_offset" not in outputs: ## default m2t2 case
                    cost = self.object_weight * (
                        -outputs['objectness'][i:i+1].T.sigmoid()
                    ) + self.bce_weight * bce_loss_matrix(
                        outputs['grasping_masks'][i], data['grasping_masks'][i]
                    ) + self.dice_weight * dice_loss_matrix(
                        outputs['grasping_masks'][i], data['grasping_masks'][i]
                    )
                    ## cost shape: num_queries x num_objects_masks, e.g., 100 x 7
                    ## data['grasping_masks shape']: 7 x 16384
                    ## outputs['objectness'][i:i+1].shape: 1 x 100
                    ## TODO: so in our case, the goal eef points should be of size B x N x 12, where N is the all possible goal gripper points
                    ## for articubot it will just be B x 1 x 12
                    ## and we change the loss to be multiplying the weights to the offsets 
                    ## TODO: pass in the offsets
                else: ## articubot and cgn case
                    # import pdb; pdb.set_trace()
                    N = outputs['pred_offset'].shape[1] # number of points in scene, e.g., 4500 in articubot and 16384 in m2t2
                    pred_offset = outputs['pred_offset'][i].view(N, 4, 3) # N, 4, 3
                    all_weights = outputs['grasping_masks'][i] # num_querys, N. grasping masks is actually weights in our case. 
                    all_weights = torch.softmax(all_weights, dim=1) # num_queries, N, dim1 sum to 1
                    # if 'goal_gripper_mask' not in data:
                    #     input_positions = data['inputs'][i, :, :3] # N, 3
                    # else:
                    ### This is a bug before, should just use pred_points
                    # import pdb; pdb.set_trace()
                    input_positions = data['pred_points'][i, :, :3] # N, 3
                    pred_goal_points = pred_offset + input_positions.unsqueeze(1) # N, 4, 3
                    pred_goal_points = pred_goal_points.view(N, -1) # N, 12
                    
                    all_query_pred_goal_points = torch.einsum("qn,nd->qd", all_weights, pred_goal_points) # num_queries, 12
                    
                    # import pdb; pdb.set_trace()
                    all_gt_goal_points = data['goal_gripper_pcd'][i].view(-1, 12) # M, 12, where M is the # of possible goals with this input
                    if 'goal_gripper_mask' in data: ### cgn case
                        this_mask = data['goal_gripper_mask'][i] # N
                        all_gt_goal_points = all_gt_goal_points[this_mask]
                    
                    diff = all_query_pred_goal_points.unsqueeze(1) - all_gt_goal_points.unsqueeze(0) # num_queries, M, 12
                    diff_squared = diff ** 2 # num_queries, M, 12
                    mse_cost = diff_squared.mean(dim=-1) # num_queries, M
                    
                    cost = self.object_weight * (-outputs['objectness'][i:i+1].T.sigmoid()) + self.wdp_weight * mse_cost
                
            else:
                N = outputs['pred_offset'].shape[1] # number of points in scene, e.g., 4500 in articubot and 16384 in m2t2
                pred_offset = outputs['pred_offset'][i].view(N, 4, 3) # N, 4, 3
                all_weights = outputs['grasping_masks'][i] # num_querys, N. grasping masks is actually weights in our case. 
                    
                # import pdb; pdb.set_trace()
                ### TODO: maybe add the class weight for articubot here
                
                num_queries = all_weights.shape[0] # num_queries, N, dim1 sum to 1
                this_pred = {
                    "pred_offset": pred_offset.unsqueeze(0).repeat(num_queries, 1, 1, 1), # num_queries, N, 4, 3
                    "pred_scores": all_weights.unsqueeze(-1), # num_queries, N
                    "pred_points": data['pred_points'][i].unsqueeze(0).repeat(num_queries, 1, 1) if 'pred_points' in data else data['inputs'][i, :, :3].unsqueeze(0).repeat(num_queries, 1, 1) # num_queries, N, 3
                }
                this_target = {
                    "goal_gripper_pcd": data['goal_gripper_pcd'][i].unsqueeze(0).repeat(num_queries, 1, 1, 1), # num_queries, M, 4, 3
                    'goal_gripper_mask': data['goal_gripper_mask'][i].unsqueeze(0).repeat(num_queries, 1) if 'goal_gripper_mask' in data else None, # num_queries, M
                }
                
                # import pdb; pdb.set_trace()
                cost = gmm_loss(this_pred, this_target, fixed_variance=data.get('gmm_variance', 0.05), return_mean=False) # num_queries, M
                # print("in matching, gmm loss is: ", cost.item())
                # import pdb; pdb.set_trace()

                cost = cost.view(num_queries, 1)
                
                # import pdb; pdb.set_trace()
            
            # TODO: consider the case where gt grasping pose is more than num queries
            output_idx, target_idx = linear_sum_assignment(cost.cpu().numpy())
            output_idx = output_idx[np.argsort(target_idx)]
            target_idx = np.sort(target_idx)
            indices.append(torch.from_numpy(output_idx).long().to(cost.device))
            target_indices.append(torch.from_numpy(target_idx).long().to(cost.device))
            cost_matrices.append(cost)
        return indices, cost_matrices, target_indices
