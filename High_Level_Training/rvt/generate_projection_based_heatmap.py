from peract_colab.rlbench.utils import get_stored_demo
from rvt.libs.peract.helpers.demo_loading_utils import keypoint_discovery
import os 
import pickle
import rvt.utils.rvt_utils as rvt_utils
import torchvision.utils as vutils
import numpy as np
import torch
from peract_colab.arm.utils import stack_on_channel
import rvt.mvt.utils as mvt_utils
# from rvt.mvt.renderer import BoxRenderer
from point_renderer.rvt_renderer import RVTBoxRenderer as BoxRenderer
from torch.cuda.amp import autocast, GradScaler
renderer_device = "cuda:0" #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
img_size = 224
rend_three_views = True
add_depth = True
cumulative_idx_heatmap = 0
# import pdb; pdb.set_trace()
renderer = BoxRenderer(
            device=renderer_device,
            img_size=(img_size, img_size),
            three_views=rend_three_views,
            with_depth=add_depth,
        )


# (Pdb) len(pc)
# 8
# (Pdb) pc[0].shape
# torch.Size([42721, 3])
# (Pdb) img_feat.shape
# *** AttributeError: 'list' object has no attribute 'shape'
# (Pdb) len(img_feat)
# 8
# (Pdb) img_feat[0].shape
# torch.Size([42721, 3])
dyn_cam_info = None
add_corr = True
norm_corr = True

def render(pc, img_feat, dyn_cam_info):
        import pdb; pdb.set_trace();
        with torch.no_grad():
            with autocast(enabled=False):
                if dyn_cam_info is None:
                    dyn_cam_info_itr = (None,) * len(pc)
                else:
                    dyn_cam_info_itr = dyn_cam_info

                if add_corr: #mvt.add_corr:
                    if norm_corr: #mvt.norm_corr:
                        img = []
                        import pdb; pdb.set_trace();
                        for _pc, _img_feat, _dyn_cam_info in zip(
                            pc, img_feat, dyn_cam_info_itr
                        ):
                            import pdb; pdb.set_trace();
                            # fix when the pc is empty
                            max_pc = 1.0 if len(_pc) == 0 else torch.max(torch.abs(_pc))
                            # from rvt.mvt.utils import ForkedPdb;
                            # ForkedPdb().set_trace()
                            img.append(
                                renderer(
                                    _pc,
                                    torch.cat((_pc / max_pc, _img_feat), dim=-1),
                                    fix_cam=True,
                                    dyn_cam_info=(_dyn_cam_info,)
                                    if not (_dyn_cam_info is None)
                                    else None,
                                ).unsqueeze(0)
                            )
                    else:
                        import pdb; pdb.set_trace();
                        img = [
                            renderer(
                                _pc,
                                torch.cat((_pc, _img_feat), dim=-1),
                                fix_cam=True,
                                dyn_cam_info=(_dyn_cam_info,)
                                if not (_dyn_cam_info is None)
                                else None,
                            ).unsqueeze(0)
                            for (_pc, _img_feat, _dyn_cam_info) in zip(
                                pc, img_feat, dyn_cam_info_itr
                            )
                        ]
                else:
                    import pdb; pdb.set_trace();
                    img = [
                        renderer(
                            _pc,
                            _img_feat,
                            fix_cam=True,
                            dyn_cam_info=(_dyn_cam_info,)
                            if not (_dyn_cam_info is None)
                            else None,
                        ).unsqueeze(0)
                        for (_pc, _img_feat, _dyn_cam_info) in zip(
                            pc, img_feat, dyn_cam_info_itr
                        )
                    ]

        img = torch.cat(img, 0)
        img = img.permute(0, 1, 4, 2, 3)


        return img



def save_batch_trans_maps(trans_maps, episodes, indices, base_dir="outputs_5th/slide_block_to_color_target_TEMPORARY"):
        """
        Save normalized and unnormalized trans_maps under 2 folders:
        heatmap/ (normalized)
        unnormalized_heatmap/ (raw)
        Folder structure:
        rendered_output/<type>/episodeX/cameraY/<index>.png
        """
        # ForkedPdb().set_trace()
        global cumulative_idx_heatmap
        B, num_cams, num_feats, H, W = trans_maps.shape  # [batch_size, num_cams, H, W]

        folders = {
            "heatmap": os.path.join(base_dir, "heatmap_dist_from_rvt"),
            "heatmap_raw": os.path.join(base_dir, "unnormalized_heatmap_dist_from_rvt"),
            "heatmap_images": os.path.join(base_dir, "heatmap_images_dist_from_rvt"),
            "heatmap_raw_images": os.path.join(base_dir, "unnormalized_heatmap_images_dist_from_rvt")
        }
        for f in folders.values():
            os.makedirs(f, exist_ok=True)

        for b in range(B):
            episode_id = int(episodes[b].item()) if torch.is_tensor(episodes[b]) else int(episodes[b])
            frame_idx = int(indices[b].item()) if torch.is_tensor(indices[b]) else int(indices[b])
            cams = trans_maps[b]  # [num_cams, H, W]
            if not os.path.exists(os.path.join(folders["heatmap"], f"episode{episode_id}")):
                    cumulative_idx_heatmap = 0
            for cam_idx in range(num_cams):
                trans = cams[cam_idx][3:6]  # [H, W]

                # Normalize for visualization
                trans_norm = (trans - trans.min()) / (trans.max() - trans.min() + 1e-8)

                # Prepare directories
                dirs = {
                    "heatmap": os.path.join(folders["heatmap"], f"episode{episode_id}", f"camera{cam_idx+1}"),
                    "heatmap_raw": os.path.join(folders["heatmap_raw"], f"episode{episode_id}", f"camera{cam_idx+1}"),
                    "heatmap_images": os.path.join(folders["heatmap_images"], f"episode{episode_id}", f"camera{cam_idx+1}"), 
                    "heatmap_raw_images": os.path.join(folders["heatmap_raw_images"], f"episode{episode_id}", f"camera{cam_idx+1}")
                }
                for d in dirs.values():
                    os.makedirs(d, exist_ok=True)

                # File paths
                norm_path = os.path.join(dirs["heatmap"], f"{cumulative_idx_heatmap}.npy")
                raw_path = os.path.join(dirs["heatmap_raw"], f"{cumulative_idx_heatmap}.npy")
                norm_path_images = os.path.join(dirs["heatmap_images"], f"{cumulative_idx_heatmap}.png")
                raw_path_images = os.path.join(dirs["heatmap_raw_images"], f"{cumulative_idx_heatmap}.png")
                # ForkedPdb().set_trace()
                # Save images
                vutils.save_image(trans_norm.unsqueeze(0), norm_path_images)
                vutils.save_image(trans.unsqueeze(0), raw_path_images)
                np.save(norm_path, np.array(trans_norm.detach().cpu()))
                np.save(raw_path, np.array(trans.detach().cpu()))

                print(f"Saved episode {episode_id}, camera {cam_idx+1}, frame {cumulative_idx_heatmap} heatmaps.")
            cumulative_idx_heatmap += 1



def stack_on_channel_np(x):
    # expect (B, T, C, ...)
    # move T next to C and merge them
    B, T, C = x.shape[:3]
    remaining_dims = x.shape[3:]

    x = x.reshape(B, T * C, *remaining_dims)
    return x

def _norm_rgb(x):
    return (x.float() / 255.0) * 2.0 - 1.0

# def _norm_rgb_np(x):
#     return (x.astype(np.float32) / 255.0) * 2.0 - 1.0

def _preprocess_inputs(data, d_idx, cameras, rgb_members):
    obs, pcds = [], []
    for camera, rgb_member in zip(cameras, rgb_members):
        # import pdb; pdb.set_trace();
        rgb = stack_on_channel(torch.tensor(getattr(data[d_idx], rgb_member)).permute(2, 0, 1).unsqueeze(0).unsqueeze(0))
        pcd = stack_on_channel(torch.tensor(getattr(data[d_idx], camera)).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)) # ["%s_rgb" % n] data[d_idx].n
        # import pdb; pdb.set_trace();
        rgb = _norm_rgb(rgb)
        obs.append(
            [rgb, pcd]
        )
        pcds.append(pcd)  # only pointcloud
    return  pcds, obs


_place_with_mean = False
scene_bounds = [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]
move_pc_in_bound = True
_place_with_mean = False
data_path = "/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/RVT2_GMM_Codebase/Bimanual_Manipulation/rvt/data/train/close_jar/all_variations/episodes/"
gripper_3d_path = "/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/RVT2_GMM_Codebase/RVT2_Codebase/RVT_Backup_IDK/RVT/data_diffusion_policy/outputs_1st_close_jar/gripper_pose/"
cameras = ["front_point_cloud", "left_shoulder_point_cloud", "right_shoulder_point_cloud", "wrist_point_cloud"]
rgbs = ["front_rgb", "left_shoulder_rgb", "right_shoulder_rgb", "wrist_rgb"]
for d_idx in range(101):
    # import pdb; pdb.set_trace();
    demo = get_stored_demo(data_path=data_path, index=d_idx)
    episode_keypoints = keypoint_discovery(demo)
    print(episode_keypoints)
    # import pdb; pdb.set_trace();
    for keypoint in episode_keypoints:
        with open(os.path.join(gripper_3d_path, f"episode{d_idx}", f"{keypoint-1}.pkl"), "rb") as f:
            action_data = pickle.load(f)
        device = torch.device("cuda:0")
        action_trans_con = torch.tensor(action_data["gripper_pose"][:,:3]).to(device)
        # pcds_keypoint, obs_keypoint = _preprocess_inputs(demo, d_idx, cameras, rgbs)
        # # import pdb; pdb.set_trace();
        # with torch.no_grad():
        #     pc, img_feat = rvt_utils.get_pc_img_feat(
        #         obs_keypoint,
        #         pcds_keypoint,
        #     )
        # # import pdb; pdb.set_trace();


        # pc, img_feat = rvt_utils.move_pc_in_bound(
        #     pc, img_feat, scene_bounds, no_op= not move_pc_in_bound #not self.move_pc_in_bound self.scene_bounds
        # )
        # # import pdb; pdb.set_trace();
        # # ForkedPdb().set_trace()
        # wpt = [x[:3] for x in action_trans_con]
        # # ForkedPdb().set_trace()
        # # import pdb; pdb.set_trace();
        # wpt_local = []
        # rev_trans = []
        # for _pc, _wpt in zip(pc, wpt):
        #     a, b = mvt_utils.place_pc_in_cube(
        #         _pc,
        #         _wpt,
        #         with_mean_or_bounds= _place_with_mean, #self._place_with_mean,
        #         scene_bounds=None if _place_with_mean else scene_bounds,
        #     )
        #     wpt_local.append(a.unsqueeze(0))
        #     rev_trans.append(b)
        # import pdb; pdb.set_trace();
        # # wpt_local = torch.cat(wpt_local, axis=0)
        device = torch.device("cuda:0")
        img = render(pc = [action_trans_con], img_feat = [torch.tensor([1,1,1]).unsqueeze(0).to(device)], dyn_cam_info = None) # [batch_size, num_cams, num_feats, H, W] # wpt_local[0].to(device)
        import pdb; pdb.set_trace()
        save_batch_trans_maps(trans_maps = img, episodes = [d_idx], indices = [keypoint], base_dir="data_diffusion_policy/outputs_1st_insert_onto_square_peg")



