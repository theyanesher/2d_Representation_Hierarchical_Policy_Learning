import pybullet as p
import os
import numpy as np
import fpsample
import time
import pickle
from scipy.spatial.transform import Rotation as R
from termcolor import cprint
import os
import sys
import argparse
import time
import torch
from matplotlib import pyplot as pslt
import h5py
import open3d as o3d
import trimesh
from collections import defaultdict
import json

import hydra
import numpy as np
import torch

from m2t2.dataset import load_rgb_xyz, collate
from m2t2.dataset_utils import denormalize_rgb, sample_points
from m2t2.meshcat_utils import (
    create_visualizer, make_frame, visualize_grasp, visualize_pointcloud
)
from m2t2.m2t2 import M2T2
from m2t2.plot_utils import get_set_colors
from m2t2.train_utils import to_cpu, to_gpu
import copy

from moviepy.editor import ImageSequenceClip
from manipulation.envs.eval_grasp_env import ContactGraspNetEnv, save_numpy_as_gif

def load_m2t2(cfg):
    model = M2T2.from_config(cfg.m2t2)
    ckpt = torch.load("./m2t2.pth")
    model.load_state_dict(ckpt['model'])
    model = model.cuda().eval()
    return model


def predict_m2t2(model, pcd, cfg, topk=1):
    zero_centered_pcd = pcd - np.mean(pcd, axis=0, keepdims=True)
    data = {
        "inputs": torch.from_numpy(zero_centered_pcd).float(),
        "points": torch.from_numpy(pcd).float(),
        "object_inputs": torch.from_numpy(zero_centered_pcd).float(),
    }
    data['task'] = 'pick'

    outputs = {
        'grasps': [],
        'grasp_confidence': [],
        'grasp_contacts': [],
    }
    
    for _ in range(1):        
        data_batch = collate([data])
        to_gpu(data_batch)
        
        with torch.no_grad():
            model_ouputs = model.infer(data_batch, cfg.eval)
                
        to_cpu(model_ouputs)
        for key in outputs:
            outputs[key].extend(model_ouputs[key][0])

    # import pdb; pdb.set_trace()
    pred_grasps = outputs['grasps']
    if len(pred_grasps) == 0:
        null_matrix = np.zeros((1, 4, 4))
        null_matrix[0, :3, :3] = np.eye(3)
        return null_matrix
    
    pred_grasps = np.concatenate(pred_grasps, axis=0)
    pred_confidence = outputs['grasp_confidence']
    pred_confidence = np.concatenate([x.cpu().numpy() for x in pred_confidence], axis=0)
    
    topk_idx = np.argsort(-pred_confidence)[:topk]
    return pred_grasps[topk_idx]

def parallel_eval(args):
    pred_grasp, scene_path, env_state, precontact = args
    new_env = ContactGraspNetEnv(scene_path=scene_path, gui=False, env_state=env_state, precontact=precontact, obs_mode='m2t2', act_mode='m2t2')
    success, res_string = new_env.step(pred_grasp)
    images = new_env.rendered_images
    new_env.close()
    return success, res_string, images

@hydra.main(config_path='.', config_name='config', version_base='1.3')
def main(cfg):
    from multiprocessing import Pool
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate M2T2')
    parser.add_argument("--ckpt_dir", type=str, default="./")
    parser.add_argument("--save_name", type=str, default="", help="additional name to save the results")
    parser.add_argument("--precontact", type=int, default=1, help="whether to first goto a precontact pose before grasping")
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--visual_grasp", type=int, default=0)
    args = parser.parse_args()
    
    all_scenes = os.listdir("/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/contact_graspnet/acronym/scene_contacts")
    all_scenes = sorted(all_scenes)
    # eval_scenes = all_scenes[-100:]  # for testing, use the last 10 scenes
    eval_scenes = [
        "009765", "009782", "009800", "009822", "009839", "009857", "009881", "009919", "009937", "009955", "009971", "009990", "010007",
        "009766", "009783", "009802", "009823", "009840", "009859", "009882", "009920", "009938", "009956", "009972", "009991", "010008",
        "009767", "009784", "009804", "009824", "009842", "009860", "009883", "009922", "009939", "009957", "009973", "009992", "010009",
        "009768", "009785", "009805", "009825", "009843", "009863", "009885", "009923", "009940", "009958", "009975", "009993", "010010",
        "009769", "009786", "009808", "009827", "009844", "009864", "009888", "009924", "009941", "009959", "009976", "009994", "010011",
        "009770", "009787", "009809", "009828", "009845", "009866", "009891", "009925", "009942", "009960", "009978", "009995", "010013",
        "009771", "009788", "009810", "009829", "009846", "009867", "009892", "009926", "009943", "009961", "009979", "009996", "010014",
        "009772", "009789", "009811", "009830", "009847", "009869", "009893", "009927", "009944", "009962", "009980", "009997", "010015",
        "009773", "009791", "009812", "009831", "009848", "009872", "009894", "009928", "009945", "009963", "009981", "009998",
        "009774", "009793", "009813", "009832", "009849", "009873", "009895", "009929", "009947", "009964", "009983", "010000",
        "009775", "009794", "009814", "009833", "009851", "009874", "009896", "009930", "009949", "009965", "009984", "010001",
        "009776", "009795", "009815", "009834", "009852", "009875", "009898", "009931", "009950", "009966", "009985", "010002",
        "009777", "009796", "009816", "009835", "009853", "009877", "009900", "009932", "009951", "009967", "009986", "010003",
        "009778", "009797", "009818", "009836", "009854", "009878", "009916", "009934", "009952", "009968", "009987", "010004",
        "009779", "009798", "009819", "009837", "009855", "009879", "009917", "009935", "009953", "009969", "009988", "010005",
        "009781", "009799", "009820", "009838", "009856", "009880", "009918", "009936", "009954", "009970", "009989", "010006",
    ]
    
    eval_scenes = [
        "009784",
        "009785",
        "009789",
        "009816",
    ]
    
    scene_path_list = ["scene_contacts/{}.npz".format(scene) for scene in eval_scenes]
    
    ckpt_name = args.ckpt_dir.split("/")[-1] + args.save_name
    # save_dir = "data/cgn_eval_results_200/{}".format(ckpt_name)
    save_dir = "data/cgn_eval_results_200_visual/{}".format(ckpt_name)
    if args.precontact:
        save_dir += "_precontact"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    model = load_m2t2(cfg)
    
    meta_results = defaultdict(int)
    for scene_path in scene_path_list:
        env = ContactGraspNetEnv(scene_path=scene_path, gui=False, obs_mode='m2t2', precontact=args.precontact, 
                                 num_points_in_pc=cfg.data.num_points, act_mode='m2t2')
        
        ### get an pcd observation from the scene
        rgb, depth, pc_in_world, pc_center = env.get_obs()
        pc_in_world_unnormlzed = pc_in_world + pc_center
        env_state = env.stablized_state
        
        ### use open3d to show the pcd
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pc_in_camera[:, :3])
        # pcd.paint_uniform_color([0.5, 0.5, 0.5])  # yellow color
        # o3d.visualization.draw_geometries([pcd])
        
        ### run it through the trained contact graspnet model
        # pred_grasps = predict_m2t2(model, pc_in_world_unnormlzed, cfg, topk=args.topk)
        pred_grasps = predict_m2t2(model, pc_in_world_unnormlzed, cfg, topk=32)
    
        ### execute the grasp, determine its success 
        this_env_results = defaultdict(int)
        
        ### serial version
        # results = defaultdict(int)
        # for idx, grasp in enumerate(pred_grasps):
        #     new_env = ContactGraspNetEnv(scene_path=scene_path, gui=False, env_state=env_state, precontact=args.precontact, obs_mode='m2t2', act_mode='m2t2')
        #     success, res_string = new_env.step(grasp)
        #     images = new_env.rendered_images
        #     this_env_results[res_string] += 1
        #     cprint("grasp try idx {} success {} reason {}".format(idx, success, res_string), "green")
        #     new_env.close()
        #     if len(images) > 0:
        #         save_numpy_as_gif(np.array(images), os.path.join(save_dir, "{}_{}_{}.gif".format(scene_path.split("/")[-1].replace(".npz", ""), idx, res_string)))
            
        visual_grasp = True
        if not visual_grasp:
            results = defaultdict(int)
            for idx, grasp in enumerate(pred_grasps):
                new_env = ContactGraspNetEnv(scene_path=scene_path, gui=False, env_state=env_state, precontact=args.precontact, obs_mode='m2t2', act_mode='m2t2')
                success, res_string = new_env.step(grasp)
                images = new_env.rendered_images
                this_env_results[res_string] += 1
                cprint("grasp try idx {} success {} reason {}".format(idx, success, res_string), "green")
                new_env.close()
                if len(images) > 0:
                    save_numpy_as_gif(np.array(images), os.path.join(save_dir, "{}_{}_{}.gif".format(scene_path.split("/")[-1].replace(".npz", ""), idx, res_string)))
            
        else:
            new_env = ContactGraspNetEnv(scene_path=scene_path, gui=True, env_state=env_state, precontact=args.precontact, obs_mode='m2t2', act_mode='m2t2')
            visual_save_path = os.path.join(save_dir, "{}".format(scene_path.split("/")[-1].replace(".npz", "")))
            success, res_string = new_env.step(pred_grasps[0], all_grasps=pred_grasps[1:], debug=False, visual=True, visual_save_path=visual_save_path)
            new_env.close()
            
        ### parallel version
        # all_args = [(pred_grasps[i], scene_path, env_state, args.precontact) for i in range(len(pred_grasps))]
        # with Pool(processes=20) as pool:
        #     results = pool.map(parallel_eval, all_args)  
            
        # for idx, res in enumerate(results):
        #     success, string, images = res
        #     if success:
        #         cprint(string, "green")
        #     else:
        #         cprint(string, "red")
        #     this_env_results[string] += 1
        #     meta_results[string] += 1
        #     if len(images) > 0:
        #         save_numpy_as_gif(np.array(images), os.path.join(save_dir, "{}_{}_{}.gif".format(scene_path.split("/")[-1].replace(".npz", ""), idx, string)))
            
        with open(os.path.join(save_dir, scene_path.split("/")[-1].replace(".npz", ".json")), 'w') as f:
            json.dump(this_env_results, f, indent=4)
        
        env.close()      

    with open(os.path.join(save_dir, "meta_results.json"), 'w') as f:
        json.dump(meta_results, f, indent=4)


if __name__ == "__main__":
    main()
    
        
            
        
        
        