import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from test_PointNet2.cgn import utils
import random

class ContactGraspnetLoss(nn.Module):
    def __init__(self, global_config, device):
        super(ContactGraspnetLoss, self).__init__()
        self.global_config = global_config

        # -- Process config -- #
        config_losses = [
            'pred_contact_base',
            'pred_contact_success', # True
            'pred_contact_offset',  # True
            'pred_contact_approach',
            'pred_grasps_adds',  # True
            'pred_grasps_adds_gt2pred',
        ]

        config_weights = [
            'dir_cosine_loss_weight',
            'score_ce_loss_weight',  # True
            'offset_loss_weight',  # True
            'approach_cosine_loss_weight',
            'adds_loss_weight',  # True
            'adds_gt2pred_loss_weight',
        ]

        self.device = device

        bin_weights = global_config['DATA']['labels']['bin_weights']
        self.bin_weights = torch.tensor(bin_weights).to(self.device)
        self.bin_vals = self._get_bin_vals().to(self.device)

        for config_loss, config_weight in zip(config_losses, config_weights):
            if config_loss in global_config['MODEL'] and global_config['MODEL'][config_loss]:
                setattr(self, config_weight, global_config['OPTIMIZER'][config_weight])
            else:
                setattr(self, config_weight, 0.0)
                
        self.four_points_loss_weight = global_config['OPTIMIZER']["four_points_loss_weight"]

        if  self.global_config['MODEL']['loss_mode'] == 'contact_graspnet':
            self.gripper = mesh_utils.create_gripper('panda')

            n_copies = 1  # We will repeat this according to the batch size
            gripper_control_points = self.gripper.get_control_point_tensor(n_copies) # b x 5 x 3
            sym_gripper_control_points = self.gripper.get_control_point_tensor(n_copies, symmetric=True)

            self.gripper_control_points_homog = torch.cat([gripper_control_points,
                torch.ones((n_copies, gripper_control_points.shape[1], 1))], dim=2)  # b x 5 x 4
            self.sym_gripper_control_points_homog = torch.cat([sym_gripper_control_points,
                torch.ones((n_copies, gripper_control_points.shape[1], 1))], dim=2)  # b x 5 x 4

            self.gripper_control_points_homog = self.gripper_control_points_homog.to(self.device)
            self.sym_gripper_control_points_homog = self.sym_gripper_control_points_homog.to(self.device)
        
        


    def forward(self, pred, target, compute_topk_4_points_loss=False):
        """
        Computes loss terms from pointclouds, network predictions and labels

        Arguments:
            pointclouds_pl {tf.placeholder} -- bxNx3 input point clouds
            end_points {dict[str:tf.variable]} -- endpoints of the network containing predictions
            dir_labels_pc_cam {tf.variable} -- base direction labels in camera coordinates (bxNx3)
            offset_labels_pc {tf.variable} -- grasp width labels (bxNx1)
            grasp_success_labels_pc {tf.variable} -- contact success labels (bxNx1)
            approach_labels_pc_cam {tf.variable} -- approach direction labels in camera coordinates (bxNx3)
            global_config {dict} -- config dict

        Returns:
            [dir_cosine_loss, bin_ce_loss, offset_loss, approach_cosine_loss, adds_loss,
            adds_loss_gt2pred, gt_control_points, pred_control_points, pos_grasps_in_view] -- All losses (not all are used for training)
        """
            
        if  self.global_config['MODEL']['loss_mode'] == 'contact_graspnet':
            pred_grasps_cam = pred['pred_grasps_cam']           # B x N x 4 x 4
            pred_scores = pred['pred_scores']                   # B x N x 1
            pred_points = pred['pred_points']                   # B x N x 3
            # offset_pred = pred['offset_pred']                   # B x N  # We use the grasp_offset_head instead of this
            grasp_offset_head = pred['grasp_offset_head'].permute(0, 2, 1)       # B x N x 10


            # # Generated in acronym_dataloader.py
            # grasp_success_labels_pc = target['grasp_success_label']  # B x N
            # grasp_offset_labels_pc = target['grasp_diff_label']    # B x N x 3

            # approach_labels_pc_cam = target['grasp_approach_label']    # B x N x 3
            # dir_labels_pc_cam = target['grasp_dir_label']              # B x N x 3
            # pointclouds_pl = target['pc_cam']                    # B x N x 3

            # -- Interpolate Labels -- #
            pos_contact_points = target['pos_contact_points']    # B x M x 3
            pos_contact_dirs = target['pos_contact_dirs']        # B x M x 3
            pos_finger_diffs = target['pos_finger_diffs']        # B x M
            pos_approach_dirs = target['pos_approach_dirs']      # B x M x 3
            camera_pose = target['camera_pose']                  # B x 4 x 4

            dir_labels_pc_cam, \
            grasp_offset_labels_pc, \
            grasp_success_labels_pc, \
            approach_labels_pc_cam, \
            debug = self._compute_labels(pred_points, 
                                        camera_pose,
                                        pos_contact_points,
                                        pos_contact_dirs,
                                        pos_finger_diffs,
                                        pos_approach_dirs)
                
            # I think this is the number of positive grasps that are in view
            min_geom_loss_divisor = float(self.global_config['LOSS']['min_geom_loss_divisor'])  # This is 1.0
            pos_grasps_in_view = torch.clamp(grasp_success_labels_pc.sum(dim=1), min=min_geom_loss_divisor)  # B
            # pos_grasps_in_view = torch.maximum(grasp_success_labels_pc.sum(dim=1), min_geom_loss_divisor)  # B

            total_loss = 0.0

            if self.dir_cosine_loss_weight > 0:
                raise NotImplementedError

            # -- Grasp Confidence Loss -- #
            if self.score_ce_loss_weight > 0:  # TODO (bin_ce_loss)
                bin_ce_loss = F.binary_cross_entropy(pred_scores, grasp_success_labels_pc, reduction='none')  # B x N x 1
                if 'topk_confidence' in self.global_config['LOSS'] \
                    and self.global_config['LOSS']['topk_confidence']:
                    bin_ce_loss, _ = torch.topk(bin_ce_loss.squeeze(), k=self.global_config['LOSS']['topk_confidence'])
                bin_ce_loss = torch.mean(bin_ce_loss)

                total_loss += self.score_ce_loss_weight * bin_ce_loss
                
            # -- Grasp Offset / Thickness Loss -- #
            if self.offset_loss_weight > 0:  # TODO  (offset_loss)
                if self.global_config['MODEL']['bin_offsets']:
                    # Convert labels to multihot
                    bin_vals = self.global_config['DATA']['labels']['offset_bins']
                    grasp_offset_labels_multihot = self._bin_label_to_multihot(grasp_offset_labels_pc, 
                                                                            bin_vals)

                    if self.global_config['LOSS']['offset_loss_type'] == 'softmax_cross_entropy':
                        raise NotImplementedError

                    else:
                        offset_loss = F.binary_cross_entropy_with_logits(grasp_offset_head,
                                                                        grasp_offset_labels_multihot, reduction='none')  # B x N x 1
                        if 'too_small_offset_pred_bin_factor' in self.global_config['LOSS'] \
                            and self.global_config['LOSS']['too_small_offset_pred_bin_factor']:
                            raise NotImplementedError

                        # Weight loss for each bin
                        shaped_bin_weights = self.bin_weights[None, None, :]
                        offset_loss = (shaped_bin_weights * offset_loss).mean(axis=2)
                else:
                    raise NotImplementedError
                masked_offset_loss = offset_loss * grasp_success_labels_pc.squeeze()
                # Divide each batch by the number of successful grasps in the batch
                offset_loss = torch.mean(torch.sum(masked_offset_loss, axis=1, keepdim=True) / pos_grasps_in_view)

                total_loss += self.offset_loss_weight * offset_loss

            if self.approach_cosine_loss_weight > 0:
                raise NotImplementedError

            # -- 6 Dof Pose Loss -- #
            if self.adds_loss_weight > 0:  # TODO  (adds_loss)
                # Build groudn truth grasps and compare distances to predicted grasps
                # import pdb; pdb.set_trace()

                ### ADS Gripper PC Loss
                # Get 6 DoF pose of predicted grasp 
                # ### NOTE yufei: this is thickness gt. the returned grsap_offset_labels_pc is already the real width value 
                # instead of a predicted bin index (which is the output of the network). That's why I commented the first branch here
                
                # import pdb; pdb.set_trace()
                # if self.global_config['MODEL']['bin_offsets']:
                #     thickness_gt = self.bin_vals[torch.argmax(grasp_offset_labels_pc, dim=2)]
                # else:
                thickness_gt = grasp_offset_labels_pc[:, :, 0]

                # TODO: Move this to dataloader? 
                pred_grasps = pred_grasps_cam  # B x N x 4 x 4
                gt_grasps_proj = utils.build_6d_grasp(approach_labels_pc_cam, dir_labels_pc_cam, pred_points, thickness_gt, use_torch=True, device=self.device) # b x N x 4 x 4
                
                # Select positive grasps I think?
                success_mask = grasp_success_labels_pc.bool()[:, :, :, None] # B x N x 1 x 1
                success_mask = torch.broadcast_to(success_mask, gt_grasps_proj.shape) # B x N x 4 x 4
                pos_gt_grasps_proj = torch.where(success_mask, gt_grasps_proj, torch.ones_like(gt_grasps_proj) * 100000) # B x N x 4 x 4

                # Expand gripper control points to match number of points
                # only use per point pred grasps but not per point gt grasps
                control_points = self.gripper_control_points_homog.unsqueeze(1)  # 1 x 1 x 5 x 4
                control_points = control_points.repeat(pred_points.shape[0], pred_points.shape[1], 1, 1)  # b x N x 5 x 4

                sym_control_points = self.sym_gripper_control_points_homog.unsqueeze(1)  # 1 x 1 x 5 x 4
                sym_control_points = sym_control_points.repeat(pred_points.shape[0], pred_points.shape[1], 1, 1)  # b x N x 5 x 4

                pred_control_points = torch.matmul(control_points, pred_grasps.permute(0, 1, 3, 2))[:, :, :, :3]  # b x N x 5 x 3

                # Transform control points to ground truth locations
                gt_control_points = torch.matmul(control_points, pos_gt_grasps_proj.permute(0, 1, 3, 2))[:, :, :, :3]  # b x N x 5 x 3
                sym_gt_control_points = torch.matmul(sym_control_points, pos_gt_grasps_proj.permute(0, 1, 3, 2))[:, :, :, :3]  # b x N x 5 x 3

                # Compute distances between predicted and ground truth control points
                expanded_pred_control_points = pred_control_points.unsqueeze(2)         # B x N x 1 x 5 x 3
                expanded_gt_control_points = gt_control_points.unsqueeze(1)             # B x 1 x N' x 5 x 3  I think N' == N
                expanded_sym_gt_control_points = sym_gt_control_points.unsqueeze(1)     # B x 1 x N' x 5 x 3  I think N' == N

                # Sum of squared distances between all points
                squared_add = torch.sum((expanded_pred_control_points - expanded_gt_control_points)**2, dim=(3, 4))  # B x N x N'
                sym_squared_add = torch.sum((expanded_pred_control_points - expanded_sym_gt_control_points)**2, dim=(3, 4))  # B x N x N'

                # Combine distances between gt and symmetric gt grasps
                squared_adds = torch.concat([squared_add, sym_squared_add], dim=2)  # B x N x 2N'

                # Take min distance to gt grasp for each predicted grasp
                squared_adds_k = torch.topk(squared_adds, k=1, dim=2, largest=False)[0]  # B x N

                # Mask negative grasps
                # TODO: If there are bugs, its prob here.  The original code sums on axis=1
                # Which just determines if there is a successful grasp in the batch.  
                # I think we just want to select the positive grasps so the sum is redundant.
                sum_grasp_success_labels = torch.sum(grasp_success_labels_pc, dim=2, keepdim=True)
                binary_grasp_success_labels = torch.clamp(sum_grasp_success_labels, 0, 1) ### NOTE yufei: these two lines seem to be doing nothing
                min_adds = binary_grasp_success_labels * torch.sqrt(squared_adds_k)  # B x N x 1
                adds_loss = torch.sum(pred_scores * min_adds, dim=(1), keepdim=True)  # B x 1
                adds_loss = adds_loss.squeeze() / pos_grasps_in_view.squeeze()  # B x 1
                adds_loss = torch.mean(adds_loss)
                total_loss += self.adds_loss_weight * adds_loss
            
            if self.four_points_loss_weight > 0:
                success_mask = grasp_success_labels_pc.bool()[:, :, :, None] # B x N x 1 x 1
                success_mask = torch.broadcast_to(success_mask, gt_grasps_proj.shape) # B x N x 4 x 4
                pos_gt_grasps_proj = torch.where(success_mask, gt_grasps_proj, torch.ones_like(gt_grasps_proj) * 100000) # B x N x 4 x 4
                
                # pred_grasps = pred_grasps_cam
                # pred_mask = pred_scores[:, :, 0] > self.global_config['TEST']['first_thres']
                # pred_mask = pred_mask.unsqueeze(2).unsqueeze(3)
                # pred_mask = torch.broadcast_to(pred_mask, pred_grasps.shape)  # B x N x 4 x 4
                # pred_grasp_poses = torch.where(pred_mask, pred_grasps, torch.ones_like(pred_grasps) * 100000)  # B x N x 4 x 4
                pred_grasp_poses = pred_grasps
                
                gt_4_points = self._get_4_points_from_pose(pos_gt_grasps_proj, grasp_offset_labels_pc)  # B x N x 4 x 3
                sym_gt_4_points = self._get_4_points_from_pose(pos_gt_grasps_proj, grasp_offset_labels_pc, flip=True)  # B x N x 4 x 3
                
                thickness_pred = self.bin_vals[torch.argmax(grasp_offset_head, dim=2)]
                pred_4_points = self._get_4_points_from_pose(pred_grasp_poses, thickness_pred.unsqueeze(-1))  # B x N x 4 x 3
                
                expanded_pred_4_points = pred_4_points.unsqueeze(2)         # B x N x 1 x 5 x 3
                expanded_gt_4_points = gt_4_points.unsqueeze(1)             # B x 1 x N' x 5 x 3  I think N' == N
                expanded_sym_gt_4_points = sym_gt_4_points.unsqueeze(1)     # B x 1 x N' x 5 x 3  I think N' == N

                # Sum of squared distances between all points
                squared_add = torch.sum((expanded_pred_4_points - expanded_gt_4_points)**2, dim=(3, 4))  # B x N x N'
                sym_squared_add = torch.sum((expanded_pred_4_points - expanded_sym_gt_4_points)**2, dim=(3, 4))  # B x N x N'
                squared_adds = torch.concat([squared_add, sym_squared_add], dim=2)  # B x N x 2N'

                # compute loss over ground-truth contact points
                sum_grasp_success_labels = torch.sum(grasp_success_labels_pc, dim=2, keepdim=True)
                binary_grasp_success_labels = torch.clamp(sum_grasp_success_labels, 0, 1) ### NOTE yufei: these two lines seem to be doing nothing

                # Take min distance to gt grasp for each predicted grasp
                squared_adds_k = torch.topk(squared_adds, k=1, dim=2, largest=False)[0]  # B x N
                four_point_loss = torch.sqrt(squared_adds_k)
                four_point_loss_ori = four_point_loss.clone()
                # pred_mask = pred_scores[:, :, 0] > self.global_config['TEST']['first_thres']
                # four_point_loss = four_point_loss * pred_mask.unsqueeze(-1)  # B x N
                four_point_loss = four_point_loss * binary_grasp_success_labels  # B x N
                four_point_loss = four_point_loss * pred_scores # B x N
                # import pdb; pdb.set_trace()
                four_point_loss = torch.sum(four_point_loss, dim=1)  
                four_point_loss = four_point_loss.squeeze() / pos_grasps_in_view.squeeze()  # B x 1
                four_point_loss = torch.mean(four_point_loss)
                
                if compute_topk_4_points_loss:
                    # import pdb; pdb.set_trace()
                    _, topk_indices = torch.topk(pred_scores.squeeze(2), k=10, dim=1)
                    topk_losses  = torch.gather(four_point_loss_ori.squeeze(2), dim=1, index=topk_indices)
                    # import pdb; pdb.set_trace()
                    topk_losses = torch.mean(topk_losses)
                else:
                    topk_losses = 0

            if self.adds_gt2pred_loss_weight > 0:
                raise NotImplementedError
            
            loss_info = {
                'bin_ce_loss': bin_ce_loss,  # Grasp success loss
                'offset_loss': offset_loss,  # Grasp width loss
                'adds_loss': adds_loss,  # Pose loss
                'four_point_loss': four_point_loss if self.four_points_loss_weight > 0 else 0,
                "topk_4_point_loss": topk_losses if compute_topk_4_points_loss else 0,
            }

            return total_loss, loss_info
        
        elif self.global_config['MODEL']['loss_mode'] == 'contact_graspnet_4_points':
            pred_scores = pred['pred_scores']                   # B x N x 1
            pred_points = pred['pred_points']                   # B x N x 3
            pred_4_points = pred['pred_4_points']       # B x N x 4 x 3
            B, N, _, _ = pred_4_points.shape

            # -- Interpolate Labels -- #
            pos_contact_points = target['pos_contact_points']    # B x M x 3
            pos_contact_dirs = target['pos_contact_dirs']        # B x M x 3
            pos_finger_diffs = target['pos_finger_diffs']        # B x M
            pos_approach_dirs = target['pos_approach_dirs']      # B x M x 3
            camera_pose = target['camera_pose']                  # B x 4 x 4

            dir_labels_pc_cam, \
            grasp_offset_labels_pc, \
            grasp_success_labels_pc, \
            approach_labels_pc_cam, \
            debug = self._compute_labels(pred_points, 
                                        camera_pose,
                                        pos_contact_points,
                                        pos_contact_dirs,
                                        pos_finger_diffs,
                                        pos_approach_dirs)
                
            # I think this is the number of positive grasps that are in view
            min_geom_loss_divisor = float(self.global_config['LOSS']['min_geom_loss_divisor'])  # This is 1.0
            pos_grasps_in_view = torch.clamp(grasp_success_labels_pc.sum(dim=1), min=min_geom_loss_divisor)  # B

            total_loss = 0.0
            
            # -- Grasp Confidence Loss -- #
            if self.score_ce_loss_weight > 0:  # TODO (bin_ce_loss)
                bin_ce_loss = F.binary_cross_entropy(pred_scores, grasp_success_labels_pc, reduction='none')  # B x N x 1
                if 'topk_confidence' in self.global_config['LOSS'] \
                    and self.global_config['LOSS']['topk_confidence']:
                    bin_ce_loss, _ = torch.topk(bin_ce_loss.squeeze(), k=self.global_config['LOSS']['topk_confidence'])
                bin_ce_loss = torch.mean(bin_ce_loss)

                total_loss += self.score_ce_loss_weight * bin_ce_loss
                
            if self.four_points_loss_weight > 0:
                thickness_gt = grasp_offset_labels_pc[:, :, 0]
                gt_grasps_proj = utils.build_6d_grasp(approach_labels_pc_cam, dir_labels_pc_cam, pred_points, thickness_gt, use_torch=True, device=self.device) # b x N x 4 x 4

                success_mask = grasp_success_labels_pc.bool()[:, :, :, None] # B x N x 1 x 1
                success_mask = torch.broadcast_to(success_mask, gt_grasps_proj.shape) # B x N x 4 x 4
                pos_gt_grasps_proj = torch.where(success_mask, gt_grasps_proj, torch.ones_like(gt_grasps_proj) * 100000) # B x N x 4 x 4 
                gt_4_points = self._get_4_points_from_pose(pos_gt_grasps_proj, grasp_offset_labels_pc)  # B x N x 4 x 3
                sym_gt_4_points = self._get_4_points_from_pose(pos_gt_grasps_proj, grasp_offset_labels_pc, flip=True)  # B x N x 4 x 3
                
                expanded_pred_4_points = pred_4_points.unsqueeze(2)         # B x N x 1 x 5 x 3
                expanded_gt_4_points = gt_4_points.unsqueeze(1)             # B x 1 x N' x 5 x 3  I think N' == N
                expanded_sym_gt_4_points = sym_gt_4_points.unsqueeze(1)     # B x 1 x N' x 5 x 3  I think N' == N

                # Sum of squared distances between all points
                squared_add = torch.sum((expanded_pred_4_points - expanded_gt_4_points)**2, dim=(3, 4))  # B x N x N'
                sym_squared_add = torch.sum((expanded_pred_4_points - expanded_sym_gt_4_points)**2, dim=(3, 4))  # B x N x N'
                squared_adds = torch.concat([squared_add, sym_squared_add], dim=2)  # B x N x 2N'

                sum_grasp_success_labels = torch.sum(grasp_success_labels_pc, dim=2, keepdim=True)
                binary_grasp_success_labels = torch.clamp(sum_grasp_success_labels, 0, 1) ### NOTE yufei: these two lines seem to be doing nothing

                # Take min distance to gt grasp for each predicted grasp
                squared_adds_k = torch.topk(squared_adds, k=1, dim=2, largest=False)[0]  # B x N
                four_point_loss = torch.sqrt(squared_adds_k)
                four_point_loss_ori = four_point_loss.clone()  # B x N
                four_point_loss = four_point_loss * binary_grasp_success_labels
                four_point_loss = four_point_loss * pred_scores # B x N
                four_point_loss = torch.sum(four_point_loss, dim=1)  
                four_point_loss = four_point_loss.squeeze() / pos_grasps_in_view.squeeze()  # B x 1
                four_point_loss = torch.mean(four_point_loss)

                total_loss += self.four_points_loss_weight * four_point_loss
                
                if compute_topk_4_points_loss:
                    # import pdb; pdb.set_trace()
                    _, topk_indices = torch.topk(pred_scores, k=100, dim=1)
                    topk_losses  = torch.gather(four_point_loss_ori, dim=1, index=topk_indices)
                    topk_losses = torch.mean(topk_losses)
                else:
                    topk_losses = 0

            loss_info = {
                'bin_ce_loss': bin_ce_loss,  # Grasp success loss
                'four_point_loss': four_point_loss,
                'topk_4_point_loss': topk_losses if compute_topk_4_points_loss else 0,  # Top-k 4 points loss
            }

            return total_loss, loss_info
        
        elif self.global_config["MODEL"]['loss_mode'] == 'articubot_gmm':
            pred_scores = pred['pred_scores']                   # B x N x 1, the weights for each points
            pred_points = pred['pred_points']                   # B x N x 3
            pred_offsets = pred['pred_offsets']       # B x N x 4 x 3, the predicted displacement to the goal points
            B, N, _, _ = pred_offsets.shape
            
            
            # -- Interpolate Labels -- #
            pos_contact_points = target['pos_contact_points']    # B x M x 3
            pos_contact_dirs = target['pos_contact_dirs']        # B x M x 3
            pos_finger_diffs = target['pos_finger_diffs']        # B x M
            pos_approach_dirs = target['pos_approach_dirs']      # B x M x 3
            camera_pose = target['camera_pose']                  # B x 4 x 4

            dir_labels_pc_cam, \
            grasp_offset_labels_pc, \
            grasp_success_labels_pc, \
            approach_labels_pc_cam, \
            dir_labels_pc_world, approach_labels_pc_world = self._compute_labels(pred_points, 
                                        camera_pose,
                                        pos_contact_points,
                                        pos_contact_dirs,
                                        pos_finger_diffs,
                                        pos_approach_dirs)
                
            # I think this is the number of positive grasps that are in view
            min_geom_loss_divisor = float(self.global_config['LOSS']['min_geom_loss_divisor'])  # This is 1.0
            pos_grasps_in_view = torch.clamp(grasp_success_labels_pc.sum(dim=1), min=min_geom_loss_divisor)  # B

            total_loss = 0.0
            
            ### get the ground-truth 4 points
            thickness_gt = grasp_offset_labels_pc[:, :, 0]
            gt_grasps_proj = utils.build_6d_grasp(approach_labels_pc_cam, dir_labels_pc_cam, pred_points, thickness_gt, use_torch=True, device=self.device) # b x N x 4 x 4
            

            success_mask = grasp_success_labels_pc.bool()[:, :, :, None] # B x N x 1 x 1
            success_mask = torch.broadcast_to(success_mask, gt_grasps_proj.shape) # B x N x 4 x 4
            pos_gt_grasps_proj = torch.where(success_mask, gt_grasps_proj, torch.ones_like(gt_grasps_proj) * 100000) # B x N x 4 x 4 
            if self.global_config["MODEL"]['use_gt_gripper_q']:
                gt_4_points = self._get_4_points_from_pose(pos_gt_grasps_proj, grasp_offset_labels_pc)  # B x N x 4 x 3
                sym_gt_4_points = self._get_4_points_from_pose(pos_gt_grasps_proj, grasp_offset_labels_pc, flip=True)  # B x N x 4 x 3
            else:
                gripper_width = torch.ones_like(grasp_offset_labels_pc).to(self.device) * 0.08
                gt_4_points = self._get_4_points_from_pose(pos_gt_grasps_proj, gripper_width)  # B x N x 4 x 3
                sym_gt_4_points = self._get_4_points_from_pose(pos_gt_grasps_proj, gripper_width, flip=True)  # B x N x 4 x 3
                
            gt_4_points_expanded = gt_4_points.unsqueeze(2) # B x N x 1 x 4 x 3
            pred_points_expanded = pred_points.unsqueeze(1).unsqueeze(3) # B x 1 x N x 1 x 3

            fixed_variance = random.choice(self.global_config["LOSS"]['gmm_fixed_variance'])
            if not self.global_config['MODEL'].get('gmm_take_min_over_sym', False):
                if not self.global_config['MODEL'].get('grad_schimit_4_points', False):
                    gt_label_diff = gt_4_points_expanded - pred_points_expanded  # B x N x N x 4 x 3 ### first N is all grasps, second N is all points
                    labels = gt_label_diff
                    outputs = pred_offsets.unsqueeze(1)  # B x 1 x N x 4 x 3, the predicted displacement to the goal points
                    diff = outputs - labels  # Shape: B x N x N x 4 x 3
                    diff = diff.view(B, N, N, -1)  # Reshape to B x N x N x 12 (4 points * 3 dimensions)
                    exponent = -0.5 * torch.sum((diff ** 2) / fixed_variance, dim=-1)  # Shape: (B, N, N), sum over the guassian dimension
                    
                    log_gaussians = exponent 
                else:
                    # import pdb; pdb.set_trace()
                    pred_4_points = pred_points_expanded + pred_offsets.unsqueeze(1)  # B x 1 x N x 4 x 3, the predicted 4 points
                    
                    approach_direction = pred_4_points[:, :, :, -1] - pred_4_points[:, :, :, 0]  # B x 1 x N x 3
                    baseline_direction = pred_4_points[:, :, :, 2] - pred_4_points[:, :, :, 1]  # B x 1 x N x 3
                    baseline_dir_dist = torch.norm(baseline_direction, p=2, dim=-1, keepdim=True)  # B x 1 x N x 1
                    
                    
                    approach_direction_normed = F.normalize(approach_direction, p=2, dim=-1)  # B x 1 x N x 3
                    dot_product = torch.sum(baseline_direction * approach_direction_normed, dim=-1, keepdim=True)  # B x 1 x N x 1
                    projection = dot_product * approach_direction_normed  # B x 1 x N x 3
                    baseline_direction_orthog = F.normalize(baseline_direction - projection, p=2, dim=-1)  # B x 1 x N x 3
                    new_point_1 = pred_4_points[:, :, :, 2] - baseline_direction_orthog * baseline_dir_dist  # B x 1 x N x 3
                    orthog_pred_4_points = torch.stack([pred_4_points[:, :, :, 0], new_point_1, pred_4_points[:, :, :, 2], pred_4_points[:, :, :, 3]], dim=3)  # B x 1 x N x 4 x 3
                    
                    diff = orthog_pred_4_points - gt_4_points_expanded  # Shape: B x N x N x 4 x 3
                    diff = diff.view(B, N, N, -1)  # Reshape to B x N x N x 12 (4 points * 3 dimensions)
                    exponent = -0.5 * torch.sum((diff ** 2) / fixed_variance, dim=-1)
                    log_gaussians = exponent  # Shape: (B, N, N), sum over the guassian dimension
            else:
                sym_gt_4_points_expanded = sym_gt_4_points.unsqueeze(2) # B x N x 1 x 4 x 3
                sym_gt_label_diff = sym_gt_4_points_expanded - pred_points_expanded  # B x N x N x 4 x 3
                sym_labels = sym_gt_label_diff
                
                outputs = pred_offsets.unsqueeze(1)  # B x 1 x N x 4 x 3, the predicted displacement to the goal points
                diff = outputs - labels  # Shape: B x N x N x 4 x 3
                diff = diff.view(B, N, N, -1)  # Reshape to B x N x N x 12 (4 points * 3 dimensions)
                diff_squared = diff ** 2
                
                sym_diff = outputs - sym_labels  # Shape: B x N x N x 4 x 3
                sym_diff = sym_diff.view(B, N, N, -1)  # Reshape to B x N x N x 12 (4 points * 3 dimensions)
                sym_diff_squared = sym_diff ** 2
                min_diff_squared = torch.min(diff_squared, sym_diff_squared)
                log_gaussians = -0.5 * torch.sum((min_diff_squared) / fixed_variance, dim=-1)

            # Compute log mixing coefficients
            weights = pred_scores.unsqueeze(1).squeeze(-1) # B x 1 x N. expand to have a all grasp dimension
            log_mixing_coeffs = torch.log_softmax(weights, dim=2) # softmax the weight along the per-point dimension, shape B x 1 x N
            log_mixing_coeffs = torch.clamp(log_mixing_coeffs, min=-10)  # Prevent extreme values

            max_log = torch.max(log_gaussians + log_mixing_coeffs, dim=2, keepdim=True).values # get the per-batch and per-grasp max log along all the points, B, N, 1
            log_probs = max_log.squeeze(2) + torch.logsumexp(log_gaussians + log_mixing_coeffs - max_log, dim=2) # B, N over batch and all grasp dimension
            
            sum_grasp_success_labels = torch.sum(grasp_success_labels_pc, dim=2, keepdim=True)
            binary_grasp_success_labels = torch.clamp(sum_grasp_success_labels, 0, 1) ### NOTE yufei: these two lines seem to be doing nothing

            log_probs = log_probs * binary_grasp_success_labels.squeeze()  # B x N
            
            log_probs = torch.sum(log_probs, dim=1)  
            log_probs = log_probs.squeeze() / pos_grasps_in_view.squeeze()  # B x 1                    
            loss = -torch.mean(log_probs) # mean of the negative log likelihood      
            
            if compute_topk_4_points_loss:
                # import pdb; pdb.set_trace()
                squared_adds = torch.sum((diff)**2, dim=(3))  # B x N x N'

                # Take min distance to gt grasp for each predicted grasp
                # import pdb; pdb.set_trace()
                squared_adds_k = torch.topk(squared_adds, k=1, dim=1, largest=False)[0]  # B x N
                four_point_loss = torch.sqrt(squared_adds_k)
                
                _, topk_indices = torch.topk(pred_scores, k=100, dim=1)
                topk_losses  = torch.gather(four_point_loss, dim=1, index=topk_indices)
                topk_losses = torch.mean(topk_losses)
            else:
                topk_losses = 0
            
            loss_info = {
                'gmm_loss_{}'.format(fixed_variance): loss.item(),  # Grasp success loss
            }

            return loss, loss_info      
        
    def _get_4_points_from_pose(self, pose, gripper_width, flip=False):
        ### NOTE: the pose is defined at the gripper base, not the gripper tip
        first_point = pose[..., :3, 3]
        z_dir = pose[..., :3, 2]  # Approach direction
        last_point = first_point + 0.1034 * z_dir  # 0.1034 is the gripper depth
        mid_point = first_point + 0.08 * z_dir  # TODO: get and fix this 0.08 thing
        finger_open_close_dir = pose[..., :3, 0]  # Base direction
        left_point = mid_point + finger_open_close_dir * (gripper_width / 2)
        right_point = mid_point - finger_open_close_dir * (gripper_width / 2)
        if not flip:
            return torch.stack([first_point, left_point, right_point, last_point], dim=-2)
        else:
            return torch.stack([first_point, right_point, left_point, last_point], dim=-2)  # B x N x 4 x 3
        

    def _get_bin_vals(self):
        """
        Creates bin values for grasping widths according to bounds defined in config

        Arguments:
            global_config {dict} -- config

        Returns:
            tf.constant -- bin value tensor
        """
        bins_bounds = np.array(self.global_config['DATA']['labels']['offset_bins'])
        if self.global_config['TEST']['bin_vals'] == 'max':
            bin_vals = (bins_bounds[1:] + bins_bounds[:-1])/2
            bin_vals[-1] = bins_bounds[-1]
        elif self.global_config['TEST']['bin_vals'] == 'mean':
            bin_vals = bins_bounds[1:]
        else:
            raise NotImplementedError

        if not self.global_config['TEST']['allow_zero_margin']:
            bin_vals = np.minimum(bin_vals, self.global_config['DATA']['gripper_width']-self.global_config['TEST']['extra_opening'])

        bin_vals = torch.tensor(bin_vals, dtype=torch.float32)
        return bin_vals
    
    def _bin_label_to_multihot(self, cont_labels, bin_boundaries):
        """
        Computes binned grasp width labels from continuous labels and bin boundaries

        Arguments:
            cont_labels {torch.Tensor} -- continuous labels
            bin_boundaries {list} -- bin boundary values

        Returns:
            torch.Tensor -- one/multi hot bin labels
        """
        bins = []
        for b in range(len(bin_boundaries)-1):
            bins.append(torch.logical_and(torch.greater_equal(cont_labels, bin_boundaries[b]), torch.less(cont_labels, bin_boundaries[b+1])))
        multi_hot_labels = torch.cat(bins, dim=2)
        multi_hot_labels = multi_hot_labels.to(torch.float32)

        return multi_hot_labels

    
    def _compute_labels(self, 
                        processed_pc_cams: torch.Tensor, 
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
        label_config = self.global_config['DATA']['labels']

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

        ### group labels in the world frame
        grouped_contact_dirs = utils.index_points(pos_contact_dirs, close_contact_pt_idcs)
        grouped_approach_dirs = utils.index_points(pos_approach_dirs, close_contact_pt_idcs)  # B x N x k x 3

        # -- Compute Labels -- #
        # Take mean over k nearest neighbors and normalize
        dir_label = grouped_contact_dirs_cam.mean(dim=2)  # B x N x 3
        dir_label = F.normalize(dir_label, p=2, dim=2)  # B x N x 3

        diff_label = grouped_finger_diffs.mean(dim=2)# B x N x 1

        approach_label = grouped_approach_dirs_cam.mean(dim=2)  # B x N x 3
        approach_label = F.normalize(approach_label, p=2, dim=2)  # B x N x 3
        
        ### do the same for the world points
        dir_label_world = grouped_contact_dirs.mean(dim=2)  # B x N x 3
        dir_label_world = F.normalize(dir_label_world, p=2, dim=2)  # B x N x 3
        approach_label_world = grouped_approach_dirs.mean(dim=2)  # B x N x 3
        approach_label_world = F.normalize(approach_label_world, p=2, dim=2)  # B x N x 3

        grasp_success_label = torch.mean(squared_dists_k, dim=2, keepdim=True) < radius**2  # B x N x 1 
        grasp_success_label = grasp_success_label.type(torch.float32)  

        # debug = dict(
        #     xyz_cam = xyz_cam,
        #     pos_contact_points_cam = pos_contact_points_cam,
        # )
        debug = {}


        return dir_label, diff_label, grasp_success_label, approach_label, dir_label_world, approach_label_world






