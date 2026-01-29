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

# Import pointnet library
CONTACT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))

sys.path.append(os.path.join(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, 'Pointnet_Pointnet2_pytorch'))
from contact_graspnet_pytorch import config_utils
from contact_graspnet_pytorch.checkpoints import CheckpointIO 
from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
from moviepy.editor import ImageSequenceClip
from manipulation.envs.eval_grasp_env import ContactGraspNetEnv, save_numpy_as_gif

def load_contact_graspnet(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # ckpt_dir = "checkpoints/contact_graspnet"
    # ckpt_dir = "checkpoints/articubot_gmm_no_sym_grad_schmit"
    # ckpt_dir = "checkpoints/gmm-no-sigmoid"
    # ckpt_dir = "checkpoints/test_4_point_training"
    ckpt_dir = args.ckpt_dir
    data_path = "/project_data/held/yufeiw2/contact_graspnet_pytorch/acronym"
    batch_size = 6
    global_config = config_utils.load_config(ckpt_dir, batch_size=batch_size, data_path=data_path, save=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### load the pretrained model
    grasp_estimator = GraspEstimator(global_config)
    model_checkpoint_dir = os.path.join(ckpt_dir, 'checkpoints')
    checkpoint_io = CheckpointIO(checkpoint_dir=model_checkpoint_dir, model=grasp_estimator.model)
    # load_dict = checkpoint_io.load('model_best.pt')
    load_dict = checkpoint_io.load('model.pt')
    grasp_network = grasp_estimator.model
    grasp_network.to(device)
    grasp_network.eval()
    
    return grasp_network, global_config

def infer_contact_graspnet(model, pcd, global_config, topk=10):
    pcd = torch.from_numpy(pcd).to(model.device).float()
    pcd = pcd.unsqueeze(0)  # B x N x 3
    B = 1
    
    with torch.no_grad():
        pred = model(pcd)
    if global_config["MODEL"]['loss_mode'] == 'articubot_gmm':
        pred_scores = pred['pred_scores']                   # B x N x 1, the weights for each points
        pred_points = pred['pred_points']                   # B x N x 3
        pred_offsets = pred['pred_offsets']       # B x N x 4 x 3, the predicted displacement to the goal points
        pred_scores = pred_scores.squeeze().cpu().numpy()
        pred_points = pred_points.unsqueeze(2).cpu().numpy() # B x N x 1 x 3
        pred_offsets = pred_offsets.cpu().numpy() # B X N x 4 x 3
        pred_4_points = pred_points + pred_offsets      
        pred_grasps_cam = pred['pred_grasps_cam'].cpu().numpy()           # B x N x 4 x 4
        
    elif global_config["MODEL"]['loss_mode'] == 'contact_graspnet':
        pred_grasps_cam = pred['pred_grasps_cam'].cpu().numpy()           # B x N x 4 x 4
        pred_scores = pred['pred_scores'].squeeze().cpu().numpy()                   # B x N x 1
        pred_points = pred['pred_points'].cpu().numpy()                   # B x N x 3
        grasp_offset_head = pred['grasp_offset_head'].permute(0, 2, 1).cpu().numpy()       # B x N x 10
    elif global_config['MODEL']['loss_mode'] == 'contact_graspnet_4_points':
        pred_grasps_cam = pred['pred_grasps_cam'].cpu().numpy()           # B x N x 4 x 4
        pred_scores = pred['pred_scores'].squeeze().cpu().numpy()                   # B x N x 1
        pred_points = pred['pred_points'].cpu().numpy()  

    top_k_score_idx = np.argsort(-pred_scores, axis=-1)
    pred_top_k_grasp = pred_grasps_cam[np.arange(B)[:, None], top_k_score_idx][:, :topk]
    
    return pred_top_k_grasp[0]

def parallel_eval(args):
    pred_grasp, scene_path, env_state, precontact = args
    new_env = ContactGraspNetEnv(scene_path=scene_path, gui=False, env_state=env_state, precontact=precontact)
    success, res_string = new_env.step(pred_grasp)
    images = new_env.rendered_images
    new_env.close()
    return success, res_string, images

if __name__ == "__main__":
    from multiprocessing import Pool
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Contact GraspNet')
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/contact_graspnet", help="Path to the checkpoint directory")
    parser.add_argument("--save_name", type=str, default="", help="additional name to save the results")
    parser.add_argument("--precontact", type=int, default=1, help="whether to first goto a precontact pose before grasping")
    args = parser.parse_args()
    
    contact_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/contact_graspnet/acronym/scene_contacts"
    all_scenes = os.listdir(contact_path)
    all_scenes = sorted(all_scenes)
    # all_scenes = [x for x in all_scenes if x > ""009959".npy"]
    # print(all_scenes)
    # exit()
    # eval_scenes = all_scenes[-100:]  # for testing, use the last 10 scenes
    # eval_scenes = all_scenes[-1:]  # for testing, use the last 10 scenes
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
    # eval_scenes = [os.path.join(contact_path, "{}.npy".format(x)) for x in eval_scenes]
    scene_path_list = ["scene_contacts/{}.npz".format(scene) for scene in eval_scenes]
    
    ckpt_name = args.ckpt_dir.split("/")[-1] + args.save_name
    save_dir = "data/cgn_eval_results/{}".format(ckpt_name)
    if args.precontact:
        save_dir += "_precontact"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    model, gloabl_config = load_contact_graspnet(args)
    
    meta_results = defaultdict(int)
    for scene_path in scene_path_list:
        env = ContactGraspNetEnv(scene_path=scene_path, gui=False)
        
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
        
        ### run it through the trained contact graspnet model
        # cprint("loading contact graspnet model", "green")
        # cprint("running grasping inference", "green")
        pred_grasps = infer_contact_graspnet(model, pc_in_camera, gloabl_config, topk=1)
        # cprint("visualizing predicted grasps", "green")
        # env.visualize_grasp(pc_in_camera, pred_grasps, topk=10)
        
        ### convert back to opengl camera frame and add center back
        pred_grasps[:, :3, 3] += pc_center
        pred_grasps[:, [0, 2]] *= -1
        
        ### execute the grasp, determine its success 
        this_env_results = defaultdict(int)
        
        ### serial version
        # results = defaultdict(int)
        # for idx, grasp in enumerate(pred_grasps):
        #     new_env = ContactGraspNetEnv(scene_path=scene_path, gui=True, env_state=env_state, precontact=args.precontact)
        #     success, res_string = new_env.step(grasp, debug=True)
        #     results[res_string] += 1
        #     cprint("grasp try idx {} success {} reason {}".format(idx, success, res_string), "green")
        #     new_env.close()
        
        ### parallel version
        all_args = [(pred_grasps[i], scene_path, env_state, args.precontact) for i in range(len(pred_grasps))]
        with Pool(processes=20) as pool:
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


        
            
        
        
        