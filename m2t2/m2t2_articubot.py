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


class M2T2(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        transformer: nn.Module,
        object_encoder: nn.Module = None,
        grasp_mlp: nn.Module = None,
        set_criterion: nn.Module = None,
        grasp_criterion: nn.Module = None,
        place_criterion: nn.Module = None
    ):
        super(M2T2, self).__init__()
        self.backbone = backbone
        self.object_encoder = object_encoder
        self.transformer = transformer
        self.grasp_mlp = grasp_mlp
        self.set_criterion = set_criterion
        self.grasp_criterion = grasp_criterion
        self.place_criterion = place_criterion

    @classmethod
    def from_config(cls, cfg):
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
        return cls(**args)

    def forward(self, data, cfg):
        # import pdb; pdb.set_trace()
        
        # NOTE: this needs to include the displacement output
        scene_feat = self.backbone(data['inputs'])
        
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
        
        ## TODO: update outputs to have key pred_offsets
        for output in outputs:
            output["pred_offset"] = scene_feat["pred_offset"]

        # import pdb; pdb.set_trace()
        losses = {}
        # import pdb; pdb.set_trace()
        # if self.place_criterion is not None:
        #     losses, stats = self.place_criterion(outputs, data)
        #     outputs[-1].update(stats)

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
        # import pdb; pdb.set_trace()


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
        
        pred_offsets = scene_feat["pred_offset"].squeeze(0).view(N, 4, 3) # N, 4, 3 
        input = data['inputs'].squeeze(0) # N, 3
        all_preds = input.unsqueeze(1) + pred_offsets # N, 4, 3
        best_weight = torch.softmax(best_weight, dim=1).squeeze(0)
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

        return weighted_pred.unsqueeze(0), best_weight
