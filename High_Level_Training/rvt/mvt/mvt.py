# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import copy
import torch
import numpy as np
from torch import nn
from torch.cuda.amp import autocast
# import open3d as o3d
import rvt.mvt.utils as mvt_utils

from rvt.mvt.mvt_single import MVT as MVTSingle
from rvt.mvt.config import get_cfg_defaults
from rvt.mvt.renderer import BoxRenderer
import torchvision.utils as vutils
import os
from rvt.mvt.utils import ForkedPdb
import pickle


class MVT(nn.Module):
    def __init__(
        self,
        depth,
        img_size,
        add_proprio,
        proprio_dim,
        add_lang,
        lang_dim,
        lang_len,
        img_feat_dim,
        feat_dim,
        im_channels,
        attn_dim,
        attn_heads,
        attn_dim_head,
        activation,
        weight_tie_layers,
        attn_dropout,
        decoder_dropout,
        img_patch_size,
        final_dim,
        self_cross_ver,
        add_corr,
        norm_corr,
        add_pixel_loc,
        add_depth,
        rend_three_views,
        use_point_renderer,
        pe_fix,
        feat_ver,
        wpt_img_aug,
        inp_pre_pro,
        inp_pre_con,
        cvx_up,
        xops,
        rot_ver,
        num_rot,
        stage_two,
        st_sca,
        st_wpt_loc_aug,
        st_wpt_loc_inp_no_noise,
        img_aug_2,
        renderer_device="cuda:0",
    ):
        """MultiView Transfomer
        :param stage_two: whether or not there are two stages
        :param st_sca: scaling of the pc in the second stage
        :param st_wpt_loc_aug: how much noise is to be added to wpt_local when
            transforming the pc in the second stage while training. This is
            expressed as a percentage of total pc size which is 2.
        :param st_wpt_loc_inp_no_noise: whether or not to add any noise to the
            wpt_local location which is fed to stage_two. This wpt_local
            location is used to extract features for rotation prediction
            currently. Other use cases might also arise later on. Even if
            st_wpt_loc_aug is True, this will compensate for that if set to
            True.
        :param img_aug_2: similar to img_aug in rvt repo but applied only to
            point feat and not the whole point cloud
        """
        super().__init__()

        self.use_point_renderer = use_point_renderer
        if self.use_point_renderer:
            from point_renderer.rvt_renderer import RVTBoxRenderer as BoxRenderer
        else:
            from mvt.renderer import BoxRenderer
        global BoxRenderer

        # creating a dictonary of all the input parameters
        args = copy.deepcopy(locals())
        del args["self"]
        del args["__class__"]
        del args["stage_two"]
        del args["st_sca"]
        del args["st_wpt_loc_aug"]
        del args["st_wpt_loc_inp_no_noise"]
        del args["img_aug_2"]

        self.rot_ver = rot_ver
        self.num_rot = num_rot
        self.stage_two = stage_two
        self.st_sca = st_sca
        self.st_wpt_loc_aug = st_wpt_loc_aug
        self.st_wpt_loc_inp_no_noise = st_wpt_loc_inp_no_noise
        self.img_aug_2 = img_aug_2

        # for verifying the input
        self.feat_ver = feat_ver
        self.img_feat_dim = img_feat_dim
        self.add_proprio = add_proprio
        self.proprio_dim = proprio_dim
        self.add_lang = add_lang
        if add_lang:
            lang_emb_dim, lang_max_seq_len = lang_dim, lang_len
        else:
            lang_emb_dim, lang_max_seq_len = 0, 0
        self.lang_emb_dim = lang_emb_dim
        self.lang_max_seq_len = lang_max_seq_len
        ForkedPdb().set_trace()
        self.renderer = BoxRenderer(
            device=renderer_device,
            img_size=(img_size, img_size),
            three_views=rend_three_views,
            with_depth=add_depth,
        )
        self.num_img = self.renderer.num_img
        self.proprio_dim = proprio_dim
        self.img_size = img_size

        self.mvt1 = MVTSingle(
            **args,
            renderer=self.renderer,
            no_feat=self.stage_two,
        )
        if self.stage_two:
            self.mvt2 = MVTSingle(**args, renderer=self.renderer)


        self.cumulative_idx = 0
        self.cumulative_idx_heatmap = 0
        self.cumulative_idx_pcd = 0
    def get_pt_loc_on_img(self, pt, mvt1_or_mvt2, dyn_cam_info, out=None):
        """
        :param pt: point for which location on image is to be found. the point
            shoud be in the same reference frame as wpt_local (see forward()),
            even for mvt2
        :param out: output from mvt, when using mvt2, we also need to provide the
            origin location where where the point cloud needs to be shifted
            before estimating the location in the image
        """
        assert len(pt.shape) == 3
        bs, _np, x = pt.shape
        assert x == 3

        assert isinstance(mvt1_or_mvt2, bool)
        if mvt1_or_mvt2:
            assert out is None
            out = self.mvt1.get_pt_loc_on_img(pt, dyn_cam_info)
        else:
            assert self.stage_two
            assert out is not None
            assert out['wpt_local1'].shape == (bs, 3)
            pt, _ = mvt_utils.trans_pc(pt, loc=out["wpt_local1"], sca=self.st_sca)
            pt = pt.view(bs, _np, 3)
            out = self.mvt2.get_pt_loc_on_img(pt, dyn_cam_info)

        return out

    def get_wpt(self, out, mvt1_or_mvt2, dyn_cam_info, y_q=None):
        """
        Estimate the q-values given output from mvt
        :param out: output from mvt
        :param y_q: refer to the definition in mvt_single.get_wpt
        """
        assert isinstance(mvt1_or_mvt2, bool)
        if mvt1_or_mvt2:
            wpt = self.mvt1.get_wpt(
                out, dyn_cam_info, y_q,
            )
        else:
            assert self.stage_two
            wpt = self.mvt2.get_wpt(
                out["mvt2"], dyn_cam_info, y_q
            )
            wpt = out["rev_trans"](wpt)

        return wpt

    def render(self, pc, img_feat, img_aug, mvt1_or_mvt2, dyn_cam_info):
        assert isinstance(mvt1_or_mvt2, bool)
        if mvt1_or_mvt2:
            mvt = self.mvt1
        else:
            mvt = self.mvt2
        ForkedPdb().set_trace()
        with torch.no_grad():
            with autocast(enabled=False):
                if dyn_cam_info is None:
                    dyn_cam_info_itr = (None,) * len(pc)
                else:
                    dyn_cam_info_itr = dyn_cam_info

                if mvt.add_corr:
                    if mvt.norm_corr:
                        img = []
                        for _pc, _img_feat, _dyn_cam_info in zip(
                            pc, img_feat, dyn_cam_info_itr
                        ):
                            # fix when the pc is empty
                            max_pc = 1.0 if len(_pc) == 0 else torch.max(torch.abs(_pc))
                            # from rvt.mvt.utils import ForkedPdb;
                            # ForkedPdb().set_trace()
                            img.append(
                                self.renderer(
                                    _pc,
                                    torch.cat((_pc / max_pc, _img_feat), dim=-1),
                                    fix_cam=True,
                                    dyn_cam_info=(_dyn_cam_info,)
                                    if not (_dyn_cam_info is None)
                                    else None,
                                ).unsqueeze(0)
                            )
                    else:
                        img = [
                            self.renderer(
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
                    img = [
                        self.renderer(
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

        # for visualization purposes
        if mvt.add_corr:
            mvt.img = img[:, :, 3:].clone().detach()
        else:
            mvt.img = img.clone().detach()

        # image augmentation
        if img_aug != 0:
            stdv = img_aug * torch.rand(1, device=img.device)
            # values in [-stdv, stdv]
            noise = stdv * ((2 * torch.rand(*img.shape, device=img.device)) - 1)
            img = torch.clamp(img + noise, -1, 1)

        if mvt.add_pixel_loc:
            bs = img.shape[0]
            pixel_loc = mvt.pixel_loc.to(img.device)
            img = torch.cat(
                (img, pixel_loc.unsqueeze(0).repeat(bs, 1, 1, 1, 1)), dim=2
            )

        return img

    def verify_inp(
        self,
        pc,
        img_feat,
        proprio,
        lang_emb,
        img_aug,
        wpt_local,
        rot_x_y,
    ):
        bs = len(pc)
        assert bs == len(img_feat)

        if not self.training:
            # no img_aug when not training
            assert img_aug == 0
            assert rot_x_y is None, f"rot_x_y={rot_x_y}"

        if self.training:
            assert (
                (not self.feat_ver == 1)
                or (not wpt_local is None)
            )

            if self.rot_ver == 0:
                assert rot_x_y is None, f"rot_x_y={rot_x_y}"
            elif self.rot_ver == 1:
                assert rot_x_y.shape == (bs, 2), f"rot_x_y.shape={rot_x_y.shape}"
                assert (rot_x_y >= 0).all() and (
                    rot_x_y < self.num_rot
                ).all(), f"rot_x_y={rot_x_y}"
            else:
                assert False

        for _pc, _img_feat in zip(pc, img_feat):
            np, x1 = _pc.shape
            np2, x2 = _img_feat.shape

            assert np == np2
            assert x1 == 3
            assert x2 == self.img_feat_dim

        if self.add_proprio:
            bs3, x3 = proprio.shape
            assert bs == bs3
            assert (
                x3 == self.proprio_dim
            ), "Does not support proprio of shape {proprio.shape}"
        else:
            assert proprio is None, "Invalid input for proprio={proprio}"

        if self.add_lang:
            bs4, x4, x5 = lang_emb.shape
            assert bs == bs4
            assert (
                x4 == self.lang_max_seq_len
            ), "Does not support lang_emb of shape {lang_emb.shape}"
            assert (
                x5 == self.lang_emb_dim
            ), "Does not support lang_emb of shape {lang_emb.shape}"
        else:
            assert (lang_emb is None) or (
                torch.all(lang_emb == 0)
            ), f"Invalid input for lang={lang}"

        if not (wpt_local is None):
            bs5, x6 = wpt_local.shape
            assert bs == bs5
            assert x6 == 3, "Does not support wpt_local of shape {wpt_local.shape}"

        if self.training:
            assert (not self.stage_two) or (not wpt_local is None)

    def forward(
        self,
        pc,
        img_feat,
        proprio=None,
        lang_emb=None,
        img_aug=0,
        wpt_local=None,
        rot_x_y=None,
        episode_idx = None,
        indices = None,
        gripper_pose = None, 
        gripper_open_close = None, 
        gripper_action = None,
        gripper_delta_action = None,
        **kwargs,
    ):
        """
        :param pc: list of tensors, each tensor of shape (num_points, 3)
        :param img_feat: list tensors, each tensor of shape
            (bs, num_points, img_feat_dim)
        :param proprio: tensor of shape (bs, priprio_dim)
        :param lang_emb: tensor of shape (bs, lang_len, lang_dim)
        :param img_aug: (float) magnitude of augmentation in rgb image
        :param wpt_local: gt location of the wpt in 3D, tensor of shape
            (bs, 3)
        :param rot_x_y: (bs, 2) rotation in x and y direction
        """
        self.verify_inp(
            pc=pc,
            img_feat=img_feat,
            proprio=proprio,
            lang_emb=lang_emb,
            img_aug=img_aug,
            wpt_local=wpt_local,
            rot_x_y=rot_x_y,
        )
        with torch.no_grad():
            if self.training and (self.img_aug_2 != 0):
                for x in img_feat:
                    stdv = self.img_aug_2 * torch.rand(1, device=x.device)
                    # values in [-stdv, stdv]
                    noise = stdv * ((2 * torch.rand(*x.shape, device=x.device)) - 1)
                    x = x + noise
            ForkedPdb().set_trace()
            # img = self.render(
            #     pc= [gripper_pose[0][0][:3].unsqueeze(0).to(device)],
            #     img_feat= [torch.tensor([1,1,1]).unsqueeze(0).to(device)],
            #     img_aug=img_aug,
            #     mvt1_or_mvt2=True,
            #     dyn_cam_info=None,
            # )
            # img = self.render(
            #     pc= [pc[0][0].unsqueeze(0)],
            #     img_feat= [img_feat[0][0].unsqueeze(0)],
            #     img_aug=img_aug,
            #     mvt1_or_mvt2=True,
            #     dyn_cam_info=None,
            # )
            img = self.render(
                pc= pc,
                img_feat= img_feat,
                img_aug=img_aug,
                mvt1_or_mvt2=True,
                dyn_cam_info=None,
            )
        self.save_pointcloud(pc, episode_idx, save_dir = "data_diffusion_policy/outputs_1st_slide_block_to_color_target_TEMPORARY/pointcloud")
        if self.training:
            wpt_local_stage_one = wpt_local
            wpt_local_stage_one = wpt_local_stage_one.clone().detach()
        else:
            wpt_local_stage_one = wpt_local
        self.save_batch_images(img, episode_idx, indices, gripper_pose, gripper_open_close, gripper_action, gripper_delta_action = gripper_delta_action, base_dir="data_diffusion_policy/outputs_1st_slide_block_to_color_target_TEMPORARY")

        out = self.mvt1(
            img=img,
            proprio=proprio,
            lang_emb=lang_emb,
            wpt_local=wpt_local_stage_one,
            rot_x_y=rot_x_y,
            **kwargs,
        )


        trans_maps = out["trans"]
        self.save_batch_trans_maps(trans_maps, episode_idx, indices, base_dir="data_diffusion_policy/outputs_1st_slide_block_to_color_target_TEMPORARY")


        if self.stage_two:
            with torch.no_grad():
                # adding then noisy location for training
                if self.training:
                    # noise is added so that the wpt_local2 is not exactly at
                    # the center of the pc
                    wpt_local_stage_one_noisy = mvt_utils.add_uni_noi(
                        wpt_local_stage_one.clone().detach(), 2 * self.st_wpt_loc_aug
                    )
                    pc, rev_trans = mvt_utils.trans_pc(
                        pc, loc=wpt_local_stage_one_noisy, sca=self.st_sca
                    )

                    if self.st_wpt_loc_inp_no_noise:
                        wpt_local2, _ = mvt_utils.trans_pc(
                            wpt_local, loc=wpt_local_stage_one_noisy, sca=self.st_sca
                        )
                    else:
                        wpt_local2, _ = mvt_utils.trans_pc(
                            wpt_local, loc=wpt_local_stage_one, sca=self.st_sca
                        )

                else:
                    # bs, 3
                    wpt_local = self.get_wpt(
                        out, y_q=None, mvt1_or_mvt2=True,
                        dyn_cam_info=None,
                    )
                    pc, rev_trans = mvt_utils.trans_pc(
                        pc, loc=wpt_local, sca=self.st_sca
                    )
                    # bad name!
                    wpt_local_stage_one_noisy = wpt_local

                    # must pass None to mvt2 while in eval
                    wpt_local2 = None

                img = self.render(
                    pc=pc,
                    img_feat=img_feat,
                    img_aug=img_aug,
                    mvt1_or_mvt2=False,
                    dyn_cam_info=None,
                )

            out_mvt2 = self.mvt2(
                img=img,
                proprio=proprio,
                lang_emb=lang_emb,
                wpt_local=wpt_local2,
                rot_x_y=rot_x_y,
                **kwargs,
            )

            out["wpt_local1"] = wpt_local_stage_one_noisy
            out["rev_trans"] = rev_trans
            out["mvt2"] = out_mvt2

        return out

    def free_mem(self):
        """
        Could be used for freeing up the memory once a batch of testing is done
        """
        if not self.use_point_renderer:
            print("Freeing up some memory")
            self.renderer.free_mem()



    def save_pointcloud(self, pc, episodes, save_dir="outputs/pointclouds/"):
        """
        Save point cloud as a pickle file organized by episode and timestep.
        
        :param pc: list of tensors, each tensor of shape (num_points, 3)
                   or a single tensor of shape (num_points, 3)
        :param episode_idx: int, episode number (e.g., 0, 1, 2, ...)
        :param timestep: int, timestep number within the episode (e.g., 0, 1, 2, ..., 100)
        :param save_dir: str, base directory to save point clouds (default: "saved_pointclouds")
        :return: str, path to the saved file
        """
        # ForkedPdb().set_trace()
        B = len(pc)
        # B, _ = pc.shape
        for b in range(B):
            episode_id = int(episodes[b].item()) if torch.is_tensor(episodes[b]) else int(episodes[b])
            # Create episode directory
            if not os.path.exists(os.path.join(save_dir, f"episode{episode_id}")):
                self.cumulative_idx_pcd = 0
            episode_dir = os.path.join(save_dir, f"episode{episode_id}")
            os.makedirs(episode_dir, exist_ok=True)
            # ForkedPdb().set_trace()
            # If pc is a list of tensors, convert each to numpy
            pc_numpy = pc[b].cpu().numpy() 
            
            # Create file path
            file_path = os.path.join(episode_dir, f"{self.cumulative_idx_pcd}.pkl")
            
            # Save as pickle
            with open(file_path, 'wb') as f:
                pickle.dump(pc_numpy, f)
            self.cumulative_idx_pcd += 1
        return file_path
    

    ############################## ADDING TO SAVE DATAAAAAAA ###############################

    def save_batch_images(self, img, episodes, indices, gripper_pose, gripper_open_close, gripper_action, gripper_delta_action, base_dir="outputs_5th/slide_block_to_color_target_TEMPORARY"):
        """
        Save RGB and depth images (normalized + unnormalized) under 4 folders:
        images/, unnormalized_rgb/, depth/, unnormalized_depth/
        Folder structure is:
        rendered_output/<type>/episodeX/cameraY/<index>.png
        """
        ForkedPdb().set_trace()
        B, num_cams, num_feats, H, W = img.shape
        assert num_feats >= 7, "Expected at least 7 feature channels per camera."

        folders = {
            "rgb_norm": os.path.join(base_dir, "images"),
            "rgb_raw": os.path.join(base_dir, "unnormalized_rgb"),
            "depth_norm": os.path.join(base_dir, "depth"),
            "depth_raw": os.path.join(base_dir, "unnormalized_depth"),
            "gripper_pose": os.path.join(base_dir, "gripper_pose"), 
            # "pointcloud" : os.path.join(base_dir, "pointcloud")
        }
        for f in folders.values():
            os.makedirs(f, exist_ok=True)

        for b in range(B):
            episode_id = int(episodes[b].item()) if torch.is_tensor(episodes[b]) else int(episodes[b])
            frame_idx = int(indices[b].item()) if torch.is_tensor(indices[b]) else int(indices[b])
            cams = img[b]  # [num_cams, num_feats, H, W]
            gripper_idx_pose = gripper_pose[b]
            gripper_idx_open_close = gripper_open_close[b]
            gripper_idx_action = gripper_action[b]
            gripper_idx_delta_action = gripper_delta_action[b]
            if not os.path.exists(os.path.join(folders["rgb_norm"], f"episode{episode_id}")):
                    self.cumulative_idx = 0 #frame_idx
            gripper_pickle_folder_path = os.path.join(folders["gripper_pose"], f"episode{episode_id}") 
            os.makedirs(gripper_pickle_folder_path, exist_ok=True)
            gripper_dist = {"gripper_pose": gripper_idx_pose.detach().cpu(), "gripper_open_close": gripper_idx_open_close.detach().cpu(), "gripper_action": gripper_idx_action.detach().cpu(), "gripper_idx_delta_action": gripper_idx_delta_action.detach().cpu()}
            rgb_norm_path = os.path.join(gripper_pickle_folder_path, f"{self.cumulative_idx}.pkl") #frame_idx - self.cumulative_idx
            
            # ForkedPdb().set_trace()
            with open(rgb_norm_path, "wb") as f:  # 'wb' = write binary
                pickle.dump(gripper_dist, f)
            
            for cam_idx in range(num_cams):
                cam_feat = cams[cam_idx]  # [num_feats, H, W]

                # Extract RGB and depth
                ForkedPdb().set_trace()
                rgb = cam_feat[3:6, :, :]   # adjust slice if needed
                depth = cam_feat[6:7, :, :]
                pcdpoints = np.array(cam_feat[:3, :, :].detach().cpu())

                # from rvt.mvt.utils import ForkedPdb;
                # ForkedPdb().set_trace()
                # Normalize RGB
                rgb_norm = (rgb + 1.0) / 2.0
                rgb_norm = torch.clamp(rgb_norm, 0.0, 1.0)

                # Normalize depth
                depth_valid = depth.clone()
                mask = depth_valid > -0.99
                if mask.any():
                    dmin, dmax = depth_valid[mask].min(), depth_valid[mask].max()
                else:
                    dmin, dmax = depth.min(), depth.max()
                depth_norm = (depth_valid - dmin) / (dmax - dmin + 1e-8)
                depth_norm[~mask] = 0.0

                
                # Prepare directories
                dirs = {
                    "rgb_norm": os.path.join(folders["rgb_norm"], f"episode{episode_id}", f"camera{cam_idx+1}"),
                    "rgb_raw": os.path.join(folders["rgb_raw"], f"episode{episode_id}", f"camera{cam_idx+1}"),
                    "depth_norm": os.path.join(folders["depth_norm"], f"episode{episode_id}", f"camera{cam_idx+1}"),
                    "depth_raw": os.path.join(folders["depth_raw"], f"episode{episode_id}", f"camera{cam_idx+1}"),
                    #"pointcloud": os.path.join(folders["pointcloud"], f"episode{episode_id}", f"camera{cam_idx+1}")
                }
                for d in dirs.values():
                    os.makedirs(d, exist_ok=True)

                # File paths
                rgb_norm_path = os.path.join(dirs["rgb_norm"], f"{self.cumulative_idx}.png") # frame_idx - self.cumulative_idx
                rgb_raw_path = os.path.join(dirs["rgb_raw"], f"{self.cumulative_idx}.png") # frame_idx - self.cumulative_idx
                depth_norm_path = os.path.join(dirs["depth_norm"], f"{self.cumulative_idx}.png") # frame_idx - self.cumulative_idx
                depth_raw_path = os.path.join(dirs["depth_raw"], f"{self.cumulative_idx}.png") # frame_idx - self.cumulative_idx
                # ForkedPdb().set_trace()
                # with open(os.path.join(dirs["pointcloud"], f"{self.cumulative_idx}.pkl"), "wb") as f:
                #     pickle.dump(pcdpoints, f)

                # Save images
                vutils.save_image(rgb_norm, rgb_norm_path)
                vutils.save_image(rgb, rgb_raw_path)
                vutils.save_image(depth_norm, depth_norm_path)
                vutils.save_image(depth, depth_raw_path)

                print(f"Saved episode {episode_id}, camera {cam_idx+1}, frame {self.cumulative_idx}")
            self.cumulative_idx += 1





    def save_batch_trans_maps(self, trans_maps, episodes, indices, base_dir="outputs_5th/slide_block_to_color_target_TEMPORARY"):
        """
        Save normalized and unnormalized trans_maps under 2 folders:
        heatmap/ (normalized)
        unnormalized_heatmap/ (raw)
        Folder structure:
        rendered_output/<type>/episodeX/cameraY/<index>.png
        """
        # ForkedPdb().set_trace()
        B, num_cams, H, W = trans_maps.shape  # [batch_size, num_cams, H, W]

        folders = {
            "heatmap": os.path.join(base_dir, "heatmap"),
            "heatmap_raw": os.path.join(base_dir, "unnormalized_heatmap"),
            "heatmap_images": os.path.join(base_dir, "heatmap_images"),
            "heatmap_raw_images": os.path.join(base_dir, "unnormalized_heatmap_images")
        }
        for f in folders.values():
            os.makedirs(f, exist_ok=True)

        for b in range(B):
            episode_id = int(episodes[b].item()) if torch.is_tensor(episodes[b]) else int(episodes[b])
            frame_idx = int(indices[b].item()) if torch.is_tensor(indices[b]) else int(indices[b])
            cams = trans_maps[b]  # [num_cams, H, W]
            if not os.path.exists(os.path.join(folders["heatmap"], f"episode{episode_id}")):
                    self.cumulative_idx_heatmap = 0
            for cam_idx in range(num_cams):
                trans = cams[cam_idx]  # [H, W]

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
                norm_path = os.path.join(dirs["heatmap"], f"{self.cumulative_idx_heatmap}.npy")
                raw_path = os.path.join(dirs["heatmap_raw"], f"{self.cumulative_idx_heatmap}.npy")
                norm_path_images = os.path.join(dirs["heatmap_images"], f"{self.cumulative_idx_heatmap}.png")
                raw_path_images = os.path.join(dirs["heatmap_raw_images"], f"{self.cumulative_idx_heatmap}.png")
                # ForkedPdb().set_trace()
                # Save images
                vutils.save_image(trans_norm.unsqueeze(0), norm_path_images)
                vutils.save_image(trans.unsqueeze(0), raw_path_images)
                np.save(norm_path, np.array(trans_norm.detach().cpu()))
                np.save(raw_path, np.array(trans.detach().cpu()))

                print(f"Saved episode {episode_id}, camera {cam_idx+1}, frame {self.cumulative_idx_heatmap} heatmaps.")
            self.cumulative_idx_heatmap += 1
    
if __name__ == "__main__":
    cfg = get_cfg_defaults()
    mvt = MVT(**cfg)
    breakpoint()