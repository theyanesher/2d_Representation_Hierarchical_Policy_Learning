# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# Author: Wentao Yuan
'''
Top-level M2T2 network.
'''
import torch
import torch.nn as nn

from m2t2.action_decoder import ActionDecoder, infer_placements
from m2t2.contact_decoder import ContactDecoder
from m2t2.criterion import SetCriterion, GraspCriterion, PlaceCriterion
from m2t2.matcher import HungarianMatcher
from m2t2.pointnet2 import PointNet2MSG, PointNet2MSGCls
import m2t2.cgn_utils as utils
import torch.nn.functional as F

class M2T2(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        transformer: nn.Module,
        object_encoder: nn.Module = None,
        grasp_mlp: nn.Module = None,
        set_criterion: nn.Module = None,
        grasp_criterion: nn.Module = None,
        place_criterion: nn.Module = None,
        cgn_cfg=None,
    ):
        super(M2T2, self).__init__()
        self.backbone = backbone
        self.object_encoder = object_encoder
        self.transformer = transformer
        self.grasp_mlp = grasp_mlp
        self.set_criterion = set_criterion
        self.grasp_criterion = grasp_criterion
        self.place_criterion = place_criterion
        self.cgn_cfg = cgn_cfg

    @classmethod
    def from_config(cls, cfg, cgn_cfg=None):
        args = {}
        args['backbone'] = PointNet2MSG.from_config(cfg.scene_encoder)
        channels = args['backbone'].out_channels
        obj_channels = None
        if cfg.contact_decoder.num_place_queries > 0:
            args['object_encoder'] = PointNet2MSGCls.from_config(
                cfg.object_encoder
            )
            obj_channels = args['object_encoder'].out_channels
            args['place_criterion'] = PlaceCriterion.from_config(
                cfg.place_loss
            )
            
        args['transformer'] = ContactDecoder.from_config(
            cfg.contact_decoder, channels, obj_channels
        )
        
        if cfg.contact_decoder.num_grasp_queries > 0:
            args['grasp_mlp'] = ActionDecoder.from_config(
                cfg.action_decoder, args['transformer']
            )
            matcher = HungarianMatcher.from_config(cfg.matcher)
            args['set_criterion'] = SetCriterion.from_config(
                cfg.grasp_loss, matcher
            )
            args['grasp_criterion'] = GraspCriterion.from_config(
                cfg.grasp_loss
            )
            
        args['cgn_cfg'] = cgn_cfg
            
        return cls(**args)
    
    def compute_cgn_gt(self, target, pred_points):
        B, N, _ = pred_points.shape
                
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
        # min_geom_loss_divisor = float(self.global_config['LOSS']['min_geom_loss_divisor'])  # This is 1.0
        # pos_grasps_in_view = torch.clamp(grasp_success_labels_pc.sum(dim=1), min=min_geom_loss_divisor)  # B
        
        ### get the ground-truth 4 points
        thickness_gt = grasp_offset_labels_pc[:, :, 0]
        gt_grasps_proj = utils.build_6d_grasp(approach_labels_pc_cam, dir_labels_pc_cam, pred_points, thickness_gt, use_torch=True, device=pred_points.device) # b x N x 4 x 4

        success_mask = grasp_success_labels_pc.bool()

        gripper_width = torch.ones_like(grasp_offset_labels_pc).to(pred_points.device) * 0.08
        gt_4_points = self._get_4_points_from_pose(gt_grasps_proj, gripper_width)  # B x N x 4 x 3
        
        # TODO: only take the successful ones
        mask = success_mask.squeeze(-1)  # shape: [B, N]

        return gt_4_points, mask

    def forward(self, data, cfg):
        # import pdb; pdb.set_trace()
        
        # NOTE: this needs to include the displacement output
        scene_feat = self.backbone(data['inputs'])
        # import pdb; pdb.set_trace()
        
        # object_inputs = data['object_inputs']
        # object_feat = {}
        # if self.object_encoder is not None:
        #     object_feat = self.object_encoder(object_inputs)
        # if 'task_is_place' in data:
        #     for key, val in object_feat['features'].items():
        #         object_feat['features'][key] = (
        #             val * data['task_is_place'].view(
        #                 data['task_is_place'].shape[0], 1, 1
        #             )
        #         )
        
        lang_tokens = data.get('lang_tokens')
        embedding, outputs = self.transformer(
            scene_feat, None, lang_tokens
        )
        # import pdb; pdb.set_trace()
        
        ## TODO: update outputs to have key pred_offsets
        for output in outputs:
            output["pred_offset"] = scene_feat["pred_offset"]

        # import pdb; pdb.set_trace()
        losses = {}
        # import pdb; pdb.set_trace()
        # if self.place_criterion is not None:
        #     losses, stats = self.place_criterion(outputs, data)
        #     outputs[-1].update(stats)
        
        ### TODO: should compute the cgn ground-truth here
        if 'cgn' in data:
            # import pdb; pdb.set_trace()
            # NOTE: cgn does not fp all the back to the original # of points
            gt_4_points, success_mask = self.compute_cgn_gt(data, scene_feat['context_pos']['res1']) # B,
            data['goal_gripper_pcd'] = gt_4_points
            data['goal_gripper_mask'] = success_mask
            data['pred_points'] = scene_feat['context_pos']['res1']

        # import pdb; pdb.set_trace()
        assert self.set_criterion is not None
        if self.set_criterion is not None:
            # TODO: for articubot, change this to be computing the mse loss
            # TODO: figure out the data format here
            set_losses, outputs = self.set_criterion(outputs, data)
            losses.update(set_losses)
        else:
            outputs = outputs[-1]

        ### NOTE: can no longer have the per-point loss because what is the target?? 
        # or maybe integrate the per-point loss into the matcher, which also does not make too much sense
        # ### TODO: for articubot, compute the per-point loss here
        # gripper_points = data['goal_gripper_pcd'] # B, 4, 3
        # input_positions = data['inputs'] # B, N, 3
        # import pdb; pdb.set_trace()
        # labels = gripper_points.unsqueeze(1) - input_positions.unsqueeze(2) # B, N, 4, 3
        # B, N, _, _ = labels.shape
        # labels = labels.view(B, N, -1) # B, N, 12
        # pred_offsets = outputs['pred_offset'].view(B, N, -1) # B, N, 12
        # perpoint_loss = torch.nn.functional.mse_loss(
        #     pred_offsets, labels, reduction='mean'
        # )
        # losses.update({'perpoint_loss': perpoint_loss})
        
        
        # import pdb; pdb.set_trace()
        # if self.grasp_mlp is not None:
        #     mask_features = scene_feat['features'][
        #         self.transformer.mask_feature
        #     ]
        #     obj_embedding = [emb[idx] for emb, idx in zip(
        #         embedding['grasp'], outputs['matched_idx']
        #     )]
        #     confidence = [
        #         mask.sigmoid() for mask in outputs['matched_grasping_masks']
        #     ]
        #     grasp_outputs = self.grasp_mlp(
        #         data['points'], mask_features, confidence,
        #         cfg.mask_thresh, obj_embedding, data['grasping_masks']
        #     )
        #     import pdb; pdb.set_trace()
        #     outputs.update(grasp_outputs)
        #     contact_losses = self.grasp_criterion(outputs, data)
        #     losses.update(contact_losses)

        return outputs, losses

    def infer(self, data, cfg):
        B, N, _ = data['inputs'].shape
        scene_feat = self.backbone(data['inputs'])

        # object_feat = self.object_encoder(data['object_inputs'])
        # if 'task_is_place' in data:
        #     for key in object_feat['features']:
        #         object_feat['features'][key] = (
        #             object_feat['features'][key] * data['task_is_place'].view(
        #                 data['task_is_place'].shape[0], 1, 1
        #             )
        #         )
        object_feat = None
        
        lang_tokens = data.get('lang_tokens')
        embedding, outputs = self.transformer(scene_feat, object_feat, lang_tokens)
        outputs = outputs[-1]
        import pdb; pdb.set_trace()


        # if 'place' in embedding and embedding['place'].shape[1] > 0:
        #     import pdb; pdb.set_trace()
        #     cam_pose = None if cfg.world_coord else data['cam_pose']
        #     placement_outputs = infer_placements(
        #         data['points'], outputs['placement_masks'],
        #         data['bottom_center'], data['ee_pose'],
        #         cam_pose, cfg.mask_thresh, cfg.placement_height
        #     )
        #     outputs.update(placement_outputs)
        #     outputs['placement_masks'] = (
        #         outputs['placement_masks'].sigmoid() > cfg.mask_thresh
        #     )
        
        ### assuming it's batch 1 for now
        objectness = outputs['objectness'].sigmoid()
        best_weight_idx = objectness.argmax(dim=1)
        # best_weight = torch.gather(outputs['grasping_masks'], dim=1, index=best_weight_idx.unsqueeze(1))
        best_weight = outputs['grasping_masks'].squeeze(0)[best_weight_idx] # N
        
        pred_offsets = scene_feat["pred_offset"].squeeze(0).view(B, 4, 3) # N, 4, 3 
        input = data['inputs'].squeeze(0) # N, 3
        all_preds = input.unsqueeze(1) + pred_offsets # N, 4, 3
        best_weight = torch.softmax(best_weight, dim=1)
        weighted_pred = (all_preds * best_weight.unsqueeze(1).unsqueeze(2)).sum(dim=0) # 4, 3
        
        # if 'grasp' in embedding and embedding['grasp'].shape[1] > 0:
        #     masks = outputs['grasping_masks'].sigmoid() > cfg.mask_thresh
        #     mask_features = scene_feat['features'][self.transformer.mask_feature]
        #     if 'objectness' in outputs:
        #         objectness = outputs['objectness'].sigmoid()
        #         object_ids = [torch.where((score > cfg.object_thresh) & mask.sum(dim=1) > 0)[0]for score, mask in zip(objectness, masks)]
        #         outputs['objectness'] = [
        #             score[idx] for score, idx in zip(objectness, object_ids)
        #         ]
        #         confidence = [
        #             logits.sigmoid()[idx]
        #             for logits, idx in zip(outputs['grasping_masks'], object_ids)
        #         ]
        #         outputs['grasping_masks'] = [
        #             mask[idx] for mask, idx in zip(masks, object_ids)
        #         ]
        #         obj_embedding = [emb[idx] for emb, idx in zip(
        #             embedding['grasp'], object_ids
        #         )]
        #     else:
        #         obj_embedding = embedding['grasp']
        #         confidence = [
        #             logits.sigmoid() for logits in outputs['grasping_masks']
        #         ]
        #     grasp_outputs = self.grasp_mlp(
        #         data['points'], mask_features, confidence,
        #         cfg.mask_thresh, obj_embedding
        #     )
        #     outputs.update(grasp_outputs)

        return weighted_pred

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
        label_config = self.cgn_cfg['DATA']['labels']

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