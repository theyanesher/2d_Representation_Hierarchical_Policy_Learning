import os
import hydra
import torch
from omegaconf import OmegaConf
from train_ddp import TrainDP3Workspace
from diffusion_policy_3d.common.pytorch_util import dict_apply
from manipulation.utils import build_up_env, save_numpy_as_gif, save_env
import numpy as np
from copy import deepcopy
from termcolor import cprint
from manipulation.robogen_wrapper import RobogenPointCloudWrapper
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
import json
import yaml
import argparse
from typing import Optional
from collections import deque
from manipulation.utils import load_env
from tqdm import tqdm
from torch.utils.data import DataLoader
from diffuser_actor_3d.robogen_utils import get_gripper_pos_orient_from_4_points
from scipy.spatial.transform import Rotation as R

def get_4_points_from_gripper_pos_orient(gripper_pos, gripper_orn, cur_joint_angle):
    original_gripper_pcd = np.array([[ 0.5648266,   0.05482348,  0.34434554],
        [ 0.5642125,   0.02702148,  0.2877661 ],
        [ 0.53906703,  0.01263776,  0.38347825],
        [ 0.54250515, -0.00441092,  0.32957944]]
    )
    original_gripper_orn = np.array([0.21120763,  0.75430543, -0.61925177, -0.05423936])
    
    gripper_pcd_right_finger_closed = np.array([ 0.55415434,  0.02126799,  0.32605097])
    gripper_pcd_left_finger_closed = np.array([ 0.54912525,  0.01839125,  0.3451934 ])
    gripper_pcd_closed_finger_angle = 2.6652539383870777e-05
 
    original_gripper_pcd[1] = gripper_pcd_right_finger_closed + (original_gripper_pcd[1] - gripper_pcd_right_finger_closed) / (0.04 - gripper_pcd_closed_finger_angle) * (cur_joint_angle - gripper_pcd_closed_finger_angle)
    original_gripper_pcd[2] = gripper_pcd_left_finger_closed + (original_gripper_pcd[2] - gripper_pcd_left_finger_closed) / (0.04 - gripper_pcd_closed_finger_angle) * (cur_joint_angle - gripper_pcd_closed_finger_angle)
 
    # goal_R = R.from_quat(gripper_orn)
    # import pdb; pdb.set_trace()
    goal_R = R.from_quat(gripper_orn)
    original_R = R.from_quat(original_gripper_orn)
    rotation_transfer = goal_R * original_R.inv()
    original_pcd = original_gripper_pcd - original_gripper_pcd[3]
    rotated_pcd = rotation_transfer.apply(original_pcd)
    gripper_pcd = rotated_pcd + gripper_pos
    return gripper_pcd

def infer_pointnetplus_model(inputs, goal_prediction_model, cat_embedding=None, high_level_args=None, args=None):
    inputs = inputs.to('cuda')
    pred_dict = goal_prediction_model(inputs, cat_embedding) 
    outputs = pred_dict['pred_offsets']
    pred_points = pred_dict['pred_points'] 
    weights = pred_dict['pred_scores'].squeeze(-1)
    inputs = pred_points
    B, N, _, _ = outputs.shape
    outputs = outputs.view(B, N, -1)
    
    outputs = outputs.view(B, N, 4, 3)
    
    if 'gmm' in high_level_args.articubot and high_level_args.articubot.gmm:
        # print("Using GMM sampling")
        # import pdb; pdb.set_trace()
        ### sample an displacement according to the weight
        probabilities = weights  # Must sum to 1
        probabilities = torch.nn.functional.softmax(weights, dim=1)

        # Sample one index based on the probabilities
        if not args.argmax:
            sampled_index = torch.multinomial(probabilities, num_samples=1)
            sampled_index = sampled_index.item()
        else:
            sampled_index = torch.argmax(probabilities.squeeze(0), dim=1)
                    
        displacement_mean = outputs[torch.arange(B), sampled_index, :, :] # B, 4, 3
        input_point_pos = inputs[torch.arange(B), sampled_index, :] # B, 3
        prediction = input_point_pos.unsqueeze(1) + displacement_mean # B, 4, 3
    else:
        outputs = outputs.view(B, N, 4, 3)
        outputs = outputs + inputs[:, :, :3].unsqueeze(2)
        weights = torch.nn.functional.softmax(weights, dim=1)
        outputs = outputs * weights.unsqueeze(-1).unsqueeze(-1)
        outputs = outputs.sum(dim=1)
        prediction = outputs
        
    return prediction

def infer_3dfa_model(batch, high_level_model):
    # import pdb; pdb.set_trace()
    B, N, _ = batch['pcd'].shape
    prediction_len = 1
    instruction = ["open the door of the storage furniture"]
    instruction = tokenizer(instruction).cuda().repeat(B, 1)
    with torch.no_grad():
        output = high_level_model(None, torch.full([B, prediction_len, 1], False).cuda(non_blocking=True), batch['rgb'].cuda(), None, batch['pcd'].cuda(), instruction, batch['proprioception'].cuda(), run_inference=True).view(B, prediction_len, 8)

    output = output.view(B, 8).detach().cpu().numpy()
    predicted_pos = output[:, :3]
    predicted_quat = output[:, 3:7]
    open_finger = output[:, 7].round()

    return predicted_pos, predicted_quat


def load_high_level_model(path):
    from omegaconf import OmegaConf
    import json
    ckpt_path = os.path.dirname(path)
    config_path = os.path.join(ckpt_path, "config.json")
    cfg = json.load(open(config_path, "r"))
    cfg = OmegaConf.create(cfg)
    args = cfg
    
    device = torch.device("cuda")
    general_args = args.general
    input_channel = 5 if general_args.add_one_hot_encoding else 3
    output_dim = 13 
    if general_args.policy_class == 'pointnet2':
        from test_PointNet2.model_invariant import PointNet2_super_multitask
        policy_class = PointNet2_super_multitask
    elif general_args.policy_class == 'pointnext':
        from test_PointNet2.model_invariant import PointNet2_super_next_multitask
        policy_class = PointNet2_super_next_multitask
    elif general_args.policy_class == 'pointnext_fp':
        from test_PointNet2.model_invariant import PointNet2_super_next_fp_multitask
        policy_class = PointNet2_super_next_fp_multitask
    
    
    if "category_embedding_type" not in general_args:
        general_args.category_embedding_type = None
    if general_args.category_embedding_type == "one_hot":
        embedding_dim = args.num_categories
    elif general_args.category_embedding_type == "siglip":
        embedding_dim = 768
    else:
        embedding_dim = None
    
    model = policy_class(num_classes=output_dim, keep_gripper_in_fps=general_args.keep_gripper_in_fps, input_channel=input_channel,
                                      first_sa_point=general_args.get("first_sa_point", 2048),
                                      fp_to_full=general_args.get("fp_to_full", False),
                                      replace_bn_w_gn=general_args.get("replace_bn_with_gn", False),
                                      replace_bn_w_in=general_args.get("replace_bn_with_in", False),
                                      embedding_dim=embedding_dim,
                                      film_in_sa_and_fp=general_args.get("film_in_sa_and_fp", False),
                                      embedding_as_input=general_args.get("embedding_as_input", False),
                                      replace_bn_w_ln=general_args.get("replace_bn_with_ln", False),
                                      ).to(device)
    
    model.load_state_dict(torch.load(path, map_location=device)['model'])
    print("Successfully load model from: ", path)
    model.eval()
    # model.train()
        
    return model, args

def load_3dfa_models(args, checkpoint_path):
    print("Loading model from", checkpoint_path, flush=True)

    ### TODO: change to be the actual 3dfa package path
    from articubot_3dfa.modeling.policy.denoise_actor_pcd import DenoiseActor as DenoiseActorpcd
    from articubot_3dfa.modeling.encoder.text import fetch_tokenizers

    model = DenoiseActorpcd(
        backbone=args.backbone,
        num_vis_instr_attn_layers=args.num_vis_instr_attn_layers,
        fps_subsampling_factor=args.fps_subsampling_factor,
        embedding_dim=args.embedding_dim,
        num_attn_heads=args.num_attn_heads,
        nhist=args.num_history,
        nhand=2 if args.bimanual else 1,
        num_shared_attn_layers=args.num_shared_attn_layers,
        relative=args.relative_action,
        rotation_format=args.rotation_format,
        denoise_timesteps=args.denoise_timesteps,
        denoise_model=args.denoise_model
    )

    # Load model weights
    model_dict = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    model_dict_weight = {}
    
    for key in model_dict["weight"]:
        # import pdb; pdb.set_trace()
        _key = key[7:]
        model_dict_weight[_key] = model_dict["weight"][key]
    
    # import pdb; pdb.set_trace()
    
    model.load_state_dict(model_dict_weight, strict=False)
    model.eval()

    tokenizer = fetch_tokenizers("clip")

    return model.cuda(), tokenizer

def get_dataloader(args, dataset_prefix=None, num_train_objects=None, end_ratio=0.01):
    if args.model_type == 'pointnet++' or args.model_type == 'ptv3':
        from test_PointNet2.dataset_from_disk import get_dataset_from_pickle
        dataset = get_dataset_from_pickle(all_obj_paths=None, beg_ratio=0, 
                                          end_ratio=end_ratio, 
                                        use_all_data=False, 
                                        dataset_prefix=dataset_prefix, # TODO: change this 
                                        num_train_objects=num_train_objects,
                                    )
    
    
        dataloader = DataLoader(dataset, 
                    shuffle=False,
                    batch_size=15,
                    num_workers=3, 
                    pin_memory=False,
        )
    elif args.model_type == '3dfa':
        from articubot_3dfa.datasets.articubot_dataset import get_dataset_from_pickle
        def base_collate_fn(batch):
            """Custom collate_fn, measured to be faster than default."""
            _dict = {}

            # Values for these come as lists
            list_keys = ["task", "instr"]
            for key in list_keys:
                if key not in batch[0].keys():
                    continue
                _dict[key] = []
                for item in batch:
                    _dict[key].extend(item[key])

            # Treat rest as tensors
            _dict.update({
                k_: (
                    torch.cat([item[k_] for item in batch])
                    if batch[0][k_] is not None else None
                )
                for k_ in batch[0].keys() if k_ not in list_keys
            })

            return _dict
        
        dataset = get_dataset_from_pickle(all_obj_paths=None, beg_ratio=0, end_ratio=end_ratio, 
                                        use_all_data=False, 
                                        dataset_prefix=dataset_prefix, # TODO: change this 
                                        num_train_objects=num_train_objects,
                                    )

        dataloader = DataLoader(dataset, 
                    shuffle=False,
                    batch_size=15,
                    num_workers=3, 
                    pin_memory=False,
                    collate_fn=base_collate_fn,
        )
    
    return dataloader

def compute_quat_error(quat1, quat2):
    temp = 2 * np.dot(quat1, quat2)**2 - 1
    temp = np.clip(temp, -1, 1)
    length = np.arccos(temp)
    return length

def flip_orientation(pos, q):
    four_points = get_4_points_from_gripper_pos_orient(pos, q, 0.04)
    flipped_4_points = four_points[[0, 2, 1, 3]]
    _, flipped_quat = get_gripper_pos_orient_from_4_points(flipped_4_points)
    return flipped_quat

import argparse 
parser = argparse.ArgumentParser()
parser.add_argument("--high_level_ckpt_name", type=str, default=None)
parser.add_argument("--model_type", type=str, default="pointnet++")
parser.add_argument('--argmax', type=int, default=1)
parser.add_argument('--gmm', type=int, default=1)
parser.add_argument('--val', type=int, default=0)
args = parser.parse_args()

### load the policy
if args.model_type == 'pointnet++':
    load_model_path = args.high_level_ckpt_name
    high_level_model, model_args = load_high_level_model(load_model_path)
elif args.model_type == '3dfa':
    # "/project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/articubot_3dfa/train_logs/2025-0817-test_articubot_50/best.pth"
    load_model_path = args.high_level_ckpt_name
    load_model_dir = os.path.dirname(load_model_path)
    load_config = os.path.join(load_model_dir, "config.yaml")
    model_cfg = OmegaConf.load(load_config)
    high_level_model, tokenizer = load_3dfa_models(model_cfg, load_model_path)
    model_args = model_cfg
elif args.model_type == 'ptv3':
    from ptv3.highlevel_ptv3 import HighlevelPTv3
    import hydra
    
    load_model_path = args.high_level_ckpt_name
    load_model_dir = os.path.dirname(load_model_path)
    load_config = os.path.join(load_model_dir, ".hydra/config.yaml") # TODO: implement the overrides
    model_cfg = OmegaConf.load(load_config)
    pointnet2_model: HighlevelPTv3 = hydra.utils.instantiate(model_cfg.model)
    pointnet2_model = pointnet2_model.to('cuda')
    
    state_dict = torch.load(load_model_path)['model']
    pointnet2_model.load_state_dict(state_dict)
    high_level_model = pointnet2_model
    high_level_model.eval()
    model_args = model_cfg

### get the training dataset and data loader
dataset_prefix = "/project_data/held/chenyuah/RoboGen-sim2real/data/dp3_demo/165-obj"
if not args.val:
    num_train_objects = '50'
    end_ratio = 0.01
else:
    num_train_objects = 'test_50'
    end_ratio = 0.1
dataloader = get_dataloader(args, dataset_prefix=dataset_prefix, num_train_objects=num_train_objects, end_ratio=end_ratio)

### load language embedding 
siglip_text_features = torch.load("/project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/siglip_text_features_close.pt")
# categories = ['bucket', 'faucet', 'foldingchair', 'laptop', 'stapler', 'toilet']
# cat_idx = 0
# for i, cat in enumerate(categories):
#     if cat in args.exp_dir:
#         cat_idx = i + 1
#         break
    

### for each training env, run the policy and get the error
position_error = []
orientation_error = []
orientation_error_best = []
for batch in tqdm(dataloader):
    
    with torch.no_grad():
        if args.model_type == 'pointnet++' or args.model_type == 'ptv3':
            pointcloud, gripper_pcd, goal_gripper_pcd, cat_idx, class_weight = batch
    
            high_level_args = model_args
            if high_level_args and high_level_args.general.category_embedding_type == "one_hot":
                cat_embedding = torch.nn.functional.one_hot(cat_idx_cuda, num_classes=embedding_dim).float().to(pointcloud.device)
            elif high_level_args and high_level_args.general.category_embedding_type == "siglip":
                cat_embedding = siglip_text_features[cat_idx].float().to(pointcloud.device).unsqueeze(0)
            else:
                cat_embedding = None

            goal_pcd = goal_gripper_pcd
            
            inputs = torch.cat([pointcloud, gripper_pcd], dim=1).cuda()
            cat_embedding = siglip_text_features[cat_idx].float().to(inputs.device)
            # import pdb; pdb.set_trace()
            if args.model_type == 'pointnet++':
                inputs = inputs.permute(0, 2, 1)
            pred_goal = infer_pointnetplus_model(inputs, high_level_model, 
                                                cat_embedding=cat_embedding,
                                                high_level_args=model_args, args=args)
            
            pred_positions = []
            pred_orientations = []
            pred_orientations_flipped = []
            gt_positions = []
            gt_orientations = []
            for pred_4_points, goal_4_points in zip(pred_goal, goal_pcd):
                pos, orient = get_gripper_pos_orient_from_4_points(pred_4_points.cpu().numpy())
                pos2, orient2 = get_gripper_pos_orient_from_4_points(pred_4_points.cpu().numpy()[[0, 2, 1, 3]])
                pred_positions.append(pos)
                pred_orientations.append(orient)
                pred_orientations_flipped.append(orient2)
                gt_pos, gt_orient = get_gripper_pos_orient_from_4_points(goal_4_points.cpu().numpy().reshape(4, 3))
                gt_positions.append(gt_pos)
                gt_orientations.append(gt_orient)
                
            pos_error = np.linalg.norm(np.array(pred_positions) - np.array(gt_positions), axis=1)
            quat_error = [compute_quat_error(q1, q2) for q1, q2 in zip(pred_orientations, gt_orientations)]
            quat_error2 = [compute_quat_error(q1, q2) for q1, q2 in zip(pred_orientations_flipped, gt_orientations)]
            position_error.extend(pos_error.tolist())
            orientation_error.extend(quat_error)
            orientation_error_best.extend([min(e1, e2) for e1, e2 in zip(quat_error, quat_error2)])
            
            batch_pos_mean = np.mean(pos_error)
            batch_quat_mean = np.mean(quat_error)
            batch_quat_mean2 = np.mean(quat_error2)
            min_quat_mean = min(batch_quat_mean, batch_quat_mean2)
            print("Batch position error: ", batch_pos_mean)
            print("Batch orientation error (rad): ", batch_quat_mean)
            print("Batch orientation error min (rad): ", min_quat_mean)
            
        elif args.model_type == '3dfa':
            pred_positions, pred_orientations = infer_3dfa_model(batch, high_level_model)
            
            flipped_pred_orientations = [flip_orientation(p, q) for p, q in zip(pred_positions, pred_orientations)]
            
            gt_positions = batch['action'][:, 0, 0, :3].cpu().numpy()
            gt_orientations = batch['action'][:, 0, 0, 3:7].cpu().numpy()
            pos_error = np.linalg.norm(np.array(pred_positions) - np.array(gt_positions), axis=1)
            quat_error = [compute_quat_error(q1, q2) for q1, q2 in zip(pred_orientations, gt_orientations)]
            quat_error2 = [compute_quat_error(q1, q2) for q1, q2 in zip(flipped_pred_orientations, gt_orientations)]
            
            position_error.extend(pos_error.tolist())
            orientation_error.extend(quat_error)
            orientation_error_best.extend([min(e1, e2) for e1, e2 in zip(quat_error, quat_error2)])
            
            batch_pos_mean = np.mean(pos_error)
            batch_quat_mean = np.mean(quat_error)
            batch_quat_mean2 = np.mean(quat_error2)
            min_quat_mean = min(batch_quat_mean, batch_quat_mean2)
            print("Batch position error: ", batch_pos_mean)
            print("Batch orientation error (rad): ", batch_quat_mean)
            print("Batch orientation error min (rad): ", min_quat_mean)
            
mean_pos_error = np.mean(position_error)
mean_quat_error = np.mean(orientation_error)
mean_best_quat_error = np.mean(orientation_error_best)
print("Mean position error: ", mean_pos_error)
print("Mean orientation error (rad): ", mean_quat_error)
print("Mean orientation error (degree): ", np.rad2deg(mean_quat_error))
print("Mean orientation error best (rad): ", mean_best_quat_error)
print("Mean orientation error best (degree): ", np.rad2deg(mean_best_quat_error))
