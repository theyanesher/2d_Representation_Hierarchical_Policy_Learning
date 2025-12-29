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
from matplotlib import pyplot as plt
import h5py
import open3d as o3d
import trimesh
from collections import defaultdict
import json
from moviepy.editor import ImageSequenceClip
from manipulation.envs.eval_grasp_env import ContactGraspNetEnv

def load_contact_graspnet(load_path, args):
    # from test_PointNet2.model_invariant import PointNet2_super_multitask
    # if siglip:
    #     embedding_dim = 
    # model = PointNet2_super_multitask(num_classes=13, keep_gripper_in_fps=False, input_channel=3).to(device)
    # total_params = sum(p.numel() for p in model.parameters())
    # cprint(f"model has parameters {total_params}", "red")
    
    device = torch.device("cuda")
    general_args = args.general
    input_channel = 5 if general_args.add_one_hot_encoding else 3
    output_dim = 13 
    from test_PointNet2.model_invariant import PointNet2_super_multitask
    
    if "category_embedding_type" not in general_args:
        general_args.category_embedding_type = None
    if general_args.category_embedding_type == "one_hot":
        embedding_dim = args.num_categories
    elif general_args.category_embedding_type == "siglip":
        embedding_dim = 768
    else:
        embedding_dim = None
    
    model = PointNet2_super_multitask(num_classes=output_dim, keep_gripper_in_fps=general_args.keep_gripper_in_fps, input_channel=input_channel,
                                      first_sa_point=general_args.get("first_sa_point", 2048),
                                      fp_to_full=general_args.get("fp_to_full", False),
                                      replace_bn_w_gn=general_args.get("replace_bn_with_gn", False),
                                      replace_bn_w_in=general_args.get("replace_bn_with_in", False),
                                      embedding_dim=embedding_dim,
                                      film_in_sa_and_fp=general_args.get("film_in_sa_and_fp", False),
                                      embedding_as_input=general_args.get("embedding_as_input", False),
                                      replace_bn_w_ln=general_args.get("replace_bn_with_ln", False),
                                      ).to(device)
    
    model.load_state_dict(torch.load(load_path, map_location=device)['model'])
    print("Successfully load model from: ", load_path)
    model.eval()
    # model.train()
    return model


def infer_contact_graspnet(model, pcd, topk=10, device=torch.device("cuda"), siglip_embedding=None):
    pcd = torch.from_numpy(pcd).to(device).float()
    pcd = pcd.unsqueeze(0)  # B x N x 3
    pcd = pcd.permute(0, 2, 1)  # B x 3 x N, to match the input shape of PointNet2
    B = 1
    
    with torch.no_grad():
        # print(siglip_embedding)
        # exit()
        if siglip_embedding is not None:
            embedding = siglip_embedding.unsqueeze(0).repeat(pcd.shape[0], 1)
        
        pred = model(pcd, build_grasp=True, embedding=embedding) 
        pred_scores = pred['pred_scores']                   # B x N x 1, the weights for each points
        pred_points = pred['pred_points']                   # B x N x 3
        pred_offsets = pred['pred_offsets']       # B x N x 4 x 3, the predicted displacement to the goal points
        pred_scores = pred_scores.squeeze().cpu().numpy()
        pred_points = pred_points.unsqueeze(2).cpu().numpy() # B x N x 1 x 3
        pred_offsets = pred_offsets.cpu().numpy() # B X N x 4 x 3
        pred_4_points = pred_points + pred_offsets      
        pred_grasps_cam = pred['pred_grasps_cam'].cpu().numpy()           # B x N x 4 x 4
        
    top_k_score_idx = np.argsort(-pred_scores, axis=-1)
    pred_top_k_grasp = pred_grasps_cam[np.arange(B)[:, None], top_k_score_idx][:, :topk]
    
    return pred_top_k_grasp[0]

def infer_m2t2(model, pcd, topk=10, device=torch.device("cuda"), siglip_embedding=None):
    pcd = torch.from_numpy(pcd).to(device).float()
    pcd = pcd.unsqueeze(0)  # B x N x 3
    
    with torch.no_grad():
        data_input = {
            "inputs": pcd,
        }
        # import pdb; pdb.set_trace()
        topk_grasps, weights = model.infer_cgn(data_input, None, topk=topk)
        
    return topk_grasps[0].cpu().numpy()

def parallel_eval(args):
    pred_grasp, scene_path, env_state, precontact, world_frame = args
    new_env = ContactGraspNetEnv(scene_path=scene_path, gui=False, env_state=env_state, precontact=precontact, world_frame=world_frame)
    success, res_string = new_env.step(pred_grasp)
    images = new_env.rendered_images
    new_env.close()
    return success, res_string, images

def load_scene(render_path):
    """
    Return point cloud and camera pose.  Used for loading saved renders.
    Arguments:
        scene_id {str} -- scene index
        cam_pose_id {str} -- camera pose index as length 3 string with
                            leading zeros if necessary.

    Returns:
        [pc, camera_pose] -- [point cloud, camera pose]
        or returns False if not found
    """
    # print('Loading: ', render_path)
    data = np.load(render_path, allow_pickle=True)
    pc_cam = data['pc_cam']
    camera_pose = data['camera_pose']
    return pc_cam[:, :3], camera_pose

if __name__ == "__main__":
    from multiprocessing import Pool
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Contact GraspNet')
    parser.add_argument("--ckpt_name", type=str, default="checkpoints/contact_graspnet", help="Path to the checkpoint directory")
    parser.add_argument("--save_name", type=str, default="", help="additional name to save the results")
    parser.add_argument("--precontact", type=int, default=1, help="whether to first goto a precontact pose before grasping")
    parser.add_argument("--num_point", type=int, default=20000)
    parser.add_argument("--model_type", type=str, default="pointnet++")
    parser.add_argument("--world_frame", type=int, default=0)
    args = parser.parse_args()
    
    # this_file_dir = os.path.dirname(os.path.abspath(__file__))
    
    all_scenes = os.listdir("/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/contact_graspnet/acronym/scene_contacts")
    all_scenes = sorted(all_scenes)
    eval_scenes = all_scenes[-100:]  # for testing, use the last 10 scenes
    scene_path_list = ["scene_contacts/{}".format(scene) for scene in eval_scenes]
    
    ckpt_name = "_".join(args.ckpt_name.split("/")[-2:]) + args.save_name
    save_dir = "data/cgn_eval_results/{}".format(ckpt_name)
    if args.precontact:
        save_dir += "_precontact"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    from omegaconf import OmegaConf
    import json
    
    siglip_text_features = None
    if args.model_type == "pointnet++":
        ckpt_path = os.path.dirname(args.ckpt_name)
        config_path = os.path.join(ckpt_path, "config.json")
        cfg = json.load(open(config_path, "r"))
        cfg = OmegaConf.create(cfg)
        model = load_contact_graspnet(args.ckpt_name, cfg)
        if cfg['general'].category_embedding_type == "siglip":
            project_dir = os.environ["PROJECT_DIR"]
            siglip_text_features = torch.load(os.path.join(project_dir, "siglip_text_features.pt")).float().to("cuda")
            siglip_text_features = siglip_text_features[-1]
    elif args.model_type == 'm2t2':
        from m2t2.m2t2_articubot import M2T2
        load_model_path = args.ckpt_name
        load_model_dir = os.path.dirname(load_model_path)
        load_config = os.path.join(load_model_dir, "config.yaml")
        m2t2_config = OmegaConf.load(load_config)
        high_level_model = M2T2.from_config(m2t2_config.m2t2, cgn_cfg=m2t2_config.cgn)
        ckpt = torch.load(load_model_path)
        high_level_model.load_state_dict(ckpt['model'])
        high_level_model = high_level_model.cuda().eval()
        model = high_level_model
        args.num_point = 12000
    elif args.model_type == "ptv3":
        from ptv3.highlevel_ptv3 import HighlevelPTv3
        import hydra
        
        load_model_path = args.ckpt_name
        load_model_dir = os.path.dirname(load_model_path)
        load_config = os.path.join(load_model_dir, ".hydra/config.yaml") # TODO: implement the overrides
        model_cfg = OmegaConf.load(load_config)
        pointnet2_model: HighlevelPTv3 = hydra.utils.instantiate(model_cfg.model)
        pointnet2_model = pointnet2_model.to('cuda')
        
        state_dict = torch.load(load_model_path)['model']
        pointnet2_model.load_state_dict(state_dict)
        high_level_model = pointnet2_model
        high_level_model.eval()
        model = high_level_model
        if model_cfg['general'].category_embedding_type == "siglip":
            project_dir = os.environ["PROJECT_DIR"]
            siglip_text_features = torch.load(os.path.join(project_dir, "siglip_text_features.pt")).float().to("cuda")
            siglip_text_features = siglip_text_features[-1]    
  
    
    meta_results = defaultdict(int)
    for scene_path in scene_path_list:
        env = ContactGraspNetEnv(scene_path=scene_path, gui=False, num_points_in_pc=args.num_point, world_frame=args.world_frame)
        
        ### get an pcd observation from the scene
        rgb, depth, pc_in_camera, pc_center = env.get_obs()
        env_state = env.stablized_state
        # plt.imshow(rgb)
        # plt.show()
        
        ### use open3d to show the pcd
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pc_in_camera[:, :3])
        # pcd.paint_uniform_color([0.5, 0.5, 0.5])  # yellow color
        # o3d.visualization.draw_geometries([pcd])
        # import pdb; pdb.set_trace()
        
        ### run it through the trained contact graspnet model
        # cprint("loading contact graspnet model", "green")
        # cprint("running grasping inference", "green")
        
        ### for debugging purposes
        # render_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/contact_graspnet/acronym/renders/000000/000.npz"
        # pc_in_camera, camera_pose = load_scene(render_path)
        # pc_in_camera = pc_in_camera[pc_in_camera[:, 2] > 0]
        
        # with open("data/debug/pcd.pkl", 'wb') as f:
        #     import pdb; pdb.set_trace()
        #     pickle.dump(pc_in_camera, f)
        # exit()
        
        if args.model_type == 'pointnet++':
            pred_grasps = infer_contact_graspnet(model, pc_in_camera, topk=10, siglip_embedding=siglip_text_features)
        elif args.model_type == 'm2t2':
            pred_grasps = infer_m2t2(model, pc_in_camera, topk=10, siglip_embedding=siglip_text_features)
        elif args.model_type == 'ptv3':
            pred_grasps = infer_contact_graspnet(model, pc_in_camera, topk=10, siglip_embedding=siglip_text_features)
        # cprint("visualizing predicted grasps", "green")
        # env.visualize_grasp(pc_in_camera, pred_grasps, topk=10)
        # exit()
        
        ### convert back to opengl camera frame and add center back
        if not args.world_frame:
            pred_grasps[:, :3, 3] += pc_center
            pred_grasps[:, [0, 2]] *= -1
        else:
            pred_grasps[:, :3, 3] += pc_center
        
        ### execute the grasp, determine its success 
        this_env_results = defaultdict(int)
        
        ### serial version
        # for idx, grasp in enumerate(pred_grasps):
        #     new_env = ContactGraspNetEnv(scene_path=scene_path, gui=False, env_state=env_state, precontact=args.precontact)
        #     success, res_string = new_env.step(grasp)
        #     images = new_env.rendered_images
        #     this_env_results[res_string] += 1
        #     cprint("grasp try idx {} success {} reason {}".format(idx, success, res_string), "green")
        #     new_env.close()
        #     if len(images) > 0:
        #         save_numpy_as_gif(np.array(images), os.path.join(save_dir, "{}_{}_{}.gif".format(scene_path.split("/")[-1].replace(".npz", ""), idx, res_string)))
            
                
        ### parallel version
        all_args = [(pred_grasps[i], scene_path, env_state, args.precontact, args.world_frame) for i in range(len(pred_grasps))]
        with Pool(processes=10) as pool:
            results = pool.map(parallel_eval, all_args)  
        for idx, res in enumerate(results):
            success, string, images = res
            if success:
                cprint(string, "green")
            else:
                cprint(string, "red")
            this_env_results[string] += 1
            meta_results[string] += 1
            if len(images) > 0:
                save_numpy_as_gif(np.array(images), os.path.join(save_dir, "{}_{}_{}.gif".format(scene_path.split("/")[-1].replace(".npz", ""), idx, string)))
            
        with open(os.path.join(save_dir, scene_path.split("/")[-1].replace(".npz", ".json")), 'w') as f:
            json.dump(this_env_results, f, indent=4)
        
        env.close()      

with open(os.path.join(save_dir, "meta_results.json"), 'w') as f:
    json.dump(meta_results, f, indent=4)


        
            
        
        
        