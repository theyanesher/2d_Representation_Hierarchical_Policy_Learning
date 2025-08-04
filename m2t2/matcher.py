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


class HungarianMatcher(torch.nn.Module):
    """This class computes a 1-to-1 assignment between the targets and the
    network's predictions. The targets only include objects, so in general,
    there are more predictions than targets. The un-matched predictions are
    treated as non-objects).
    """
    def __init__(self, object_weight, bce_weight, dice_weight, wdp_weight=None):
        super(HungarianMatcher, self).__init__()
        self.object_weight = object_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.wdp_weight = wdp_weight if wdp_weight is not None else 0.0

    @classmethod
    def from_config(cls, cfg):
        args = {}
        args['object_weight'] = cfg.object_weight
        args['bce_weight'] = cfg.bce_weight
        args['dice_weight'] = cfg.dice_weight
        args['wdp_weight'] = cfg.get("wdp_weight", 0.0)
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
        indices, cost_matrices = [], []
        for i in range(len(outputs['objectness'])): ## NOTE: this loops over the batch 
            # NOTE: I think we need to have this objectness as well for choosig one query at test time.
            
            # We approximate objectness NLL loss with 1 - prob.
            # The 1 is a constant that can be ommitted.
            
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
                input_positions = data['inputs'][i, :, :3] # N, 3
                pred_goal_points = pred_offset + input_positions.unsqueeze(1) # N, 4, 3
                pred_goal_points = pred_goal_points.view(N, -1) # N, 12
                
                all_query_pred_goal_points = torch.einsum("qn,nd->qd", all_weights, pred_goal_points) # num_queries, 12
                
                all_gt_goal_points = data['goal_gripper_pcd'][i].view(-1, 12) # M, 12, where M is the # of possible goals with this input
                
                diff = all_query_pred_goal_points.unsqueeze(1) - all_gt_goal_points.unsqueeze(0) # num_queries, M, 12
                diff_squared = diff ** 2 # num_queries, M, 12
                mse_cost = diff_squared.mean(dim=-1) # num_queries, M
                
                cost = self.object_weight * (-outputs['objectness'][i:i+1].T.sigmoid()) + self.wdp_weight * mse_cost 
            
            # TODO: consider the case where gt grasping pose is more than num queries
            output_idx, target_idx = linear_sum_assignment(cost.cpu().numpy())
            output_idx = output_idx[np.argsort(target_idx)]
            indices.append(torch.from_numpy(output_idx).long().to(cost.device))
            cost_matrices.append(cost)
        return indices, cost_matrices
