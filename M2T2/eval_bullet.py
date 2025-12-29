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
    args = parser.parse_args()
    
    all_scenes = os.listdir("/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/contact_graspnet/acronym/scene_contacts")
    all_scenes = sorted(all_scenes)
    eval_scenes = all_scenes[-100:]  # for testing, use the last 10 scenes
    
    scene_path_list = ["scene_contacts/{}".format(scene) for scene in eval_scenes]
    
    ckpt_name = args.ckpt_dir.split("/")[-1] + args.save_name
    save_dir = "data/cgn_eval_results/{}".format(ckpt_name)
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
        pred_grasps = predict_m2t2(model, pc_in_world_unnormlzed, cfg)
    
        ### execute the grasp, determine its success 
        this_env_results = defaultdict(int)
        
        ### serial version
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
    
        
            
        
        
        