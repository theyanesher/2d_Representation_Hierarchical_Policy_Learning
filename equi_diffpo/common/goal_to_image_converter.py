import torch 
import numpy as np
import third_party.robogen.robogen_utils as ru
import third_party.robogen.goal_cond_diffpo_utils as gdu

class Goal2ImageConverter:

    def __init__(self, img_size = (84,84), conditioning_type: str = '3d_flow_world_frame'):
        self.img_size = img_size
        self.original_size = (512,512)
        self.agentview_intrinsics = torch.Tensor(np.loadtxt(
            'sandbox/mino/camera_calibrations/agentview_camera_intrinsics.txt')).cuda()
        self.world_to_agentview_extrinsics = torch.Tensor(np.loadtxt(
            'sandbox/mino/camera_calibrations/agentview_world_to_cam_extrinsics.txt')).cuda()
        self.eye_in_hand_intrinsics = torch.Tensor(np.loadtxt(
            'sandbox/mino/camera_calibrations/eye_in_hand_camera_intrinsics.txt')).cuda()
        self.resize = self.img_size[0] / self.original_size[0]
        self.conditioning_type = conditioning_type
        if self.conditioning_type == '3d_flow_camera_frame':
            raise ValueError(f'conditioning type {self.conditioning_type} currently has a bug, do not use.')
        if self.conditioning_type not in ('3d_flow_world_frame', '2d_flow_world_frame',
                                          'pixel_goal_conditioning', 'pixel_goal_conditioning_clip',
                                          '3d_flow_camera_frame', 'heatmap_goal_conditioning'):
            raise ValueError(f"Unsupported conditioning type: {self.conditioning_type}")
    
    def generate_image_conditioning(self, subgoal_pred, obs_dict):
        """
        Args:
            subgoal_pred: torch.Tensor, (B, T, 4, 3)
        Returns:
            torch.Tensor, (B, T, C, H, W) - image conditioning
        """
        if self.conditioning_type == '3d_flow_world_frame':
            return self._generate_flow_world_frame(obs_dict, subgoal_pred, return_2d=False)
        elif self.conditioning_type == '2d_flow_world_frame':
            return self._generate_flow_world_frame(obs_dict, subgoal_pred, return_2d=True)
        elif self.conditioning_type == 'pixel_goal_conditioning':
            return self._generate_pixel_goal_conditioning(obs_dict, subgoal_pred, clip_coords=False)
        elif self.conditioning_type == 'pixel_goal_conditioning_clip':
            return self._generate_pixel_goal_conditioning(obs_dict, subgoal_pred, clip_coords=True)
        elif self.conditioning_type == '3d_flow_camera_frame':
            return self._generate_flow_camera_frame(obs_dict, subgoal_pred, return_2d=False)
        elif self.conditioning_type == 'heatmap_goal_conditioning':
            return self._generate_heatmap_goal_conditioning(obs_dict, subgoal_pred)
        else:
            raise ValueError(f"Unsupported conditioning type: {self.conditioning_type}")

    def _generate_flow_world_frame(self, obs_dict, subgoal_pred, return_2d=False):
        pcd_displacements = subgoal_pred - obs_dict['gripper_pcd']
        agentview_goal_coords = gdu.pytorch_project_points(subgoal_pred, self.agentview_intrinsics,
                                    self.world_to_agentview_extrinsics, normalization_factor=self.resize,
                                    return_2d=return_2d)
        wristview_goal_coords = gdu.pytorch_project_points(subgoal_pred, self.eye_in_hand_intrinsics,
                                    obs_dict['robot0_eye_in_hand_extrinsics'][:,-1], normalization_factor=self.resize,
                                    return_2d=return_2d)
        agentview_gripper_coords = gdu.pytorch_project_points(obs_dict['gripper_pcd'], self.agentview_intrinsics,
                                    self.world_to_agentview_extrinsics, normalization_factor=self.resize,
                                    return_2d=return_2d)
        wristview_gripper_coords = gdu.pytorch_project_points(obs_dict['gripper_pcd'], self.eye_in_hand_intrinsics,
                                    obs_dict['robot0_eye_in_hand_extrinsics'][:,-1], normalization_factor=(84./512.),
                                    return_2d=return_2d)
        obs_dict['agentview_cond_84'] = gdu.pytorch_coords_to_2d_image_displacements(agentview_goal_coords,
                                            obs_dict['agentview_image_84'], agentview_gripper_coords,
                                            pcd_displacements=pcd_displacements)
        obs_dict['robot0_eye_in_hand_cond_84'] = gdu.pytorch_coords_to_2d_image_displacements(wristview_goal_coords,
                                                    obs_dict['robot0_eye_in_hand_image_84'], wristview_gripper_coords,
                                                    pcd_displacements=pcd_displacements)
        return obs_dict

    def _generate_pixel_goal_conditioning(self, obs_dict, subgoal_pred, clip_coords=False):
        agentview_coords = gdu.pytorch_project_points(subgoal_pred,
                                            self.agentview_intrinsics,
                                            self.world_to_agentview_extrinsics,
                                            normalization_factor=(84./512.))
        wristview_coords = gdu.pytorch_project_points(subgoal_pred,
                                                    self.eye_in_hand_intrinsics,
                                                    obs_dict['robot0_eye_in_hand_extrinsics'][:,-1],
                                                    normalization_factor=(84./512.))
        obs_dict['agentview_image_84'] = gdu.pytorch_coords_to_2d_image(agentview_coords, obs_dict['agentview_image_84'], clip_coords=clip_coords)
        obs_dict['robot0_eye_in_hand_image_84'] = gdu.pytorch_coords_to_2d_image(wristview_coords, obs_dict['robot0_eye_in_hand_image_84'], clip_coords=clip_coords)
        return obs_dict

    def _generate_flow_camera_frame(self, obs_dict, subgoal_pred, return_2d=False):
        agentview_displacements = gdu.pytorch_camera_frame_coords(subgoal_pred, self.world_to_agentview_extrinsics) - \
                                    gdu.pytorch_camera_frame_coords(obs_dict['gripper_pcd'], self.world_to_agentview_extrinsics)
        agentview_goal_coords = gdu.pytorch_project_points(subgoal_pred, self.agentview_intrinsics,
                                    self.world_to_agentview_extrinsics, normalization_factor=self.resize,
                                    return_2d=return_2d)
        agentview_gripper_coords = gdu.pytorch_project_points(obs_dict['gripper_pcd'], self.agentview_intrinsics,
                                    self.world_to_agentview_extrinsics, normalization_factor=self.resize,
                                    return_2d=return_2d)
        obs_dict['agentview_cond_84'] = gdu.pytorch_coords_to_2d_image_displacements(agentview_goal_coords,
                                            obs_dict['agentview_image_84'], agentview_gripper_coords,
                                            pcd_displacements=agentview_displacements)

        wristview_displacements = gdu.pytorch_camera_frame_coords(subgoal_pred, obs_dict['robot0_eye_in_hand_extrinsics'][:,-1]) - \
                                    gdu.pytorch_camera_frame_coords(obs_dict['gripper_pcd'], obs_dict['robot0_eye_in_hand_extrinsics'][:,-1])
        wristview_goal_coords = gdu.pytorch_project_points(subgoal_pred, self.eye_in_hand_intrinsics,
                                    obs_dict['robot0_eye_in_hand_extrinsics'][:,-1], normalization_factor=self.resize,
                                    return_2d=return_2d)
        wristview_gripper_coords = gdu.pytorch_project_points(obs_dict['gripper_pcd'], self.eye_in_hand_intrinsics,
                                    obs_dict['robot0_eye_in_hand_extrinsics'][:,-1], normalization_factor=(84./512.),
                                    return_2d=return_2d)
        obs_dict['robot0_eye_in_hand_cond_84'] = gdu.pytorch_coords_to_2d_image_displacements(wristview_goal_coords,
                                                    obs_dict['robot0_eye_in_hand_image_84'], wristview_gripper_coords,
                                                    pcd_displacements=wristview_displacements)
        return obs_dict

    def _generate_heatmap_goal_conditioning(self, obs_dict, subgoal_pred, normalization_factor=84./512.):
        """
        Generate distance-based heatmaps for goal conditioning.
        
        Args:
            obs_dict: Observation dictionary containing images and point clouds
            subgoal_pred: Subgoal prediction from high-level torch.Tensor, (B, T, 4, 3)
            normalization_factor: Scale factor for coordinate projection
        
        Returns:
            tuple: (agentview_heatmap, wristview_heatmap) both of shape (n_images, H, W, n_points=4)
        """
        
        def _create_pytorch_distance_heatmap(coords_2d, image_shape):
            """Create distance heatmap from 2D coordinates."""
            device = coords_2d.device
            n_images, history, channels, height, width = image_shape
            n_points = coords_2d.shape[2]
            
            # Create pixel coordinate grid
            y_coords, x_coords = torch.meshgrid(
                torch.arange(height, device=device),
                torch.arange(width, device=device),
                indexing='ij'
            )
            pixel_coords = torch.stack([x_coords, y_coords], dim=-1)  # (H, W, 2)
            
            # Calculate max distance for normalization
            max_distance = (height**2 + width**2)**0.5
            
            # Initialize heatmap
            heatmap = torch.zeros((n_images, history, n_points, height, width), device=device)
            
            # Calculate distances for each point
            for i in range(n_points):
                target_points = coords_2d[:, :, i]  # (n_images, history, 2)
                # Vectorized distance calculation
                distances = torch.norm(
                    pixel_coords[None, None, :, :, :] - target_points[:, :, None, None, :], 
                    dim=-1
                )  # (n_images, history, H, W)
                heatmap[:, :, i] = distances
            
            # Apply square root transformation for steeper near-target gradients
            heatmap = torch.sqrt(heatmap / max_distance)
            return torch.clamp(heatmap, 0, 1).float()
        
        # Project goal gripper points to camera coordinates
        agentview_coords = gdu.pytorch_project_points(
            subgoal_pred,
            self.agentview_intrinsics,
            self.world_to_agentview_extrinsics,
            normalization_factor=normalization_factor,
        )

        wristview_coords = gdu.pytorch_project_points(
            subgoal_pred,
            self.eye_in_hand_intrinsics,
            obs_dict['robot0_eye_in_hand_extrinsics'][:, -1],
            normalization_factor=normalization_factor,
        )
        
        # Project to 2D images and get shapes
        agentview_image = gdu.pytorch_coords_to_2d_image(
            agentview_coords, 
            obs_dict['agentview_image_84'], 
            clip_coords=True
        )
        
        wristview_image = gdu.pytorch_coords_to_2d_image(
            wristview_coords, 
            obs_dict['robot0_eye_in_hand_image_84'], 
            clip_coords=True
        )
        
        # Generate heatmaps
        obs_dict["agentview_cond_84"] = _create_pytorch_distance_heatmap(agentview_coords, agentview_image.shape)
        obs_dict["robot0_eye_in_hand_cond_84"] = _create_pytorch_distance_heatmap(wristview_coords, wristview_image.shape)
        
        return obs_dict




class Goal2ImageConverterNumpy:

    def __init__(self, img_size = (84,84), conditioning_type: str = '3d_flow_world_frame'):
        self.img_size = img_size
        self.original_size = (512,512)
        self.agentview_intrinsics = np.loadtxt(
            'sandbox/mino/camera_calibrations/agentview_camera_intrinsics.txt')
        self.world_to_agentview_extrinsics = np.loadtxt(
            'sandbox/mino/camera_calibrations/agentview_world_to_cam_extrinsics.txt')
        self.eye_in_hand_intrinsics = np.loadtxt(
            'sandbox/mino/camera_calibrations/eye_in_hand_camera_intrinsics.txt')
        self.resize = self.img_size[0] / self.original_size[0]
        self.conditioning_type = conditioning_type
        if self.conditioning_type not in ('3d_flow_world_frame', '2d_flow_worlf_frame',
                                          'pixel_goal_conditioning', 'pixel_goal_conditioning_clip',
                                          '3d_flow_camera_frame', 'heatmap_goal_conditioning'):
            raise ValueError(f"Unsupported conditioning type: {self.conditioning_type}")

    def generate_image_conditioning(self, obs_dict):
        """
        Returns:
            torch.Tensor, (B, T, C, H, W) - image conditioning
        """
        if self.conditioning_type == '3d_flow_world_frame':
            return self._generate_flow_world_frame(obs_dict, return_2d=False)
        elif self.conditioning_type == '2d_flow_world_frame':
            return self._generate_flow_world_frame(obs_dict, return_2d=False)
        elif self.conditioning_type == 'pixel_goal_conditioning':
            return self._generate_pixel_goal_conditioning(obs_dict, clip_coords=False)
        elif self.conditioning_type == 'pixel_goal_conditioning_clip':
            return self._generate_pixel_goal_conditioning(obs_dict, clip_coords=True)
        elif self.conditioning_type == '3d_flow_camera_frame':
            return self._generate_flow_camera_frame(obs_dict, return_2d=False)
        elif self.conditioning_type == 'heatmap_goal_conditioning':
            return self._generate_heatmap_goal_conditioning(obs_dict)
        else:
            raise ValueError(f"Unsupported conditioning type: {self.conditioning_type}")

    def _generate_flow_world_frame(self, obs_dict, return_2d=False):
        pcd_displacements = obs_dict['goal_gripper_pcd'][()] - obs_dict['gripper_pcd'][()]
        agentview_goal_coords = gdu.project_points(obs_dict['goal_gripper_pcd'][()], self.agentview_intrinsics,
                                    self.world_to_agentview_extrinsics, normalization_factor=self.resize,
                                    return_2d=return_2d)
        wristview_goal_coords = gdu.project_points(obs_dict['goal_gripper_pcd'][()], self.eye_in_hand_intrinsics,
                                    obs_dict['robot0_eye_in_hand_extrinsics'][()], normalization_factor=self.resize,
                                    return_2d=return_2d)
        agentview_gripper_coords = gdu.project_points(obs_dict['gripper_pcd'][()], self.agentview_intrinsics,
                                    self.world_to_agentview_extrinsics, normalization_factor=self.resize,
                                    return_2d=return_2d)
        wristview_gripper_coords = gdu.project_points(obs_dict['gripper_pcd'][()], self.eye_in_hand_intrinsics,
                                    obs_dict['robot0_eye_in_hand_extrinsics'][()], normalization_factor=(84./512.),
                                    return_2d=return_2d)
        agentview_cond_84 = gdu.coords_to_2d_image_displacements(agentview_goal_coords,
                                            obs_dict['agentview_image_84'][()], agentview_gripper_coords,
                                            pcd_displacements=pcd_displacements)
        hand_cond_84 = gdu.coords_to_2d_image_displacements(wristview_goal_coords,
                                                    obs_dict['robot0_eye_in_hand_image_84'][()], wristview_gripper_coords,
                                                    pcd_displacements=pcd_displacements)
        return agentview_cond_84, hand_cond_84

    def _generate_pixel_goal_conditioning(self, obs_dict, clip_coords=False):
        agentview_coords = gdu.project_points(obs_dict['goal_gripper_pcd'][()],
                                            self.agentview_intrinsics,
                                            self.world_to_agentview_extrinsics,
                                            normalization_factor=(84./512.))
        wristview_coords = gdu.project_points(obs_dict['goal_gripper_pcd'][()],
                                            self.eye_in_hand_intrinsics,
                                            obs_dict['robot0_eye_in_hand_extrinsics'][()],
                                            normalization_factor=(84./512.))
        agentview_cond_84 = gdu.coords_to_2d_image(agentview_coords, obs_dict['agentview_image_84'][()], clip_coords=clip_coords)
        hand_cond_84 = gdu.coords_to_2d_image(wristview_coords, obs_dict['robot0_eye_in_hand_image_84'][()], clip_coords=clip_coords)
        return agentview_cond_84, hand_cond_84

    def _generate_flow_camera_frame(self, obs_dict, return_2d=False):
        agentview_displacements = gdu.camera_frame_coords(obs_dict['goal_gripper_pcd'][()], self.world_to_agentview_extrinsics) - \
                                    gdu.camera_frame_coords(obs_dict['gripper_pcd'][()], self.world_to_agentview_extrinsics)
        agentview_goal_coords = gdu.project_points(obs_dict['goal_gripper_pcd'][()], self.agentview_intrinsics,
                                    self.world_to_agentview_extrinsics, normalization_factor=self.resize,
                                    return_2d=return_2d)
        agentview_gripper_coords = gdu.project_points(obs_dict['gripper_pcd'][()], self.agentview_intrinsics,
                                    self.world_to_agentview_extrinsics, normalization_factor=self.resize,
                                    return_2d=return_2d)
        agentview_cond_84 = gdu.coords_to_2d_image_displacements(agentview_goal_coords,
                                            obs_dict['agentview_image_84'], agentview_gripper_coords,
                                            pcd_displacements=agentview_displacements)

        wristview_displacements = gdu.camera_frame_coords(obs_dict['goal_gripper_pcd'][()], obs_dict['robot0_eye_in_hand_extrinsics'][()]) - \
                                    gdu.camera_frame_coords(obs_dict['gripper_pcd'], obs_dict['robot0_eye_in_hand_extrinsics'][()])
        breakpoint()
        wristview_goal_coords = gdu.project_points(obs_dict['goal_gripper_pcd'][()], self.eye_in_hand_intrinsics,
                                    obs_dict['robot0_eye_in_hand_extrinsics'][()], normalization_factor=self.resize,
                                    return_2d=return_2d)
        wristview_gripper_coords = gdu.project_points(obs_dict['gripper_pcd'], self.eye_in_hand_intrinsics,
                                    obs_dict['robot0_eye_in_hand_extrinsics'][()], normalization_factor=self.resize,
                                    return_2d=return_2d)
        breakpoint()
        wristview_cond_84 = gdu.coords_to_2d_image_displacements(wristview_goal_coords,
                                                    obs_dict['robot0_eye_in_hand_image_84'], wristview_gripper_coords,
                                                    pcd_displacements=wristview_displacements)
        breakpoint()
        return agentview_cond_84, wristview_cond_84


    def _generate_heatmap_goal_conditioning(self, obs_dict, normalization_factor=84./512.):
        """
        Generate distance-based heatmaps for goal conditioning.
        
        Args:
            obs_dict: Observation dictionary containing images and point clouds
            normalization_factor: Scale factor for coordinate projection
        
        Returns:
            tuple: (agentview_heatmap, wristview_heatmap) both of shape (n_images, H, W, n_points=4)
        """
        
        def _create_distance_heatmap(coords_2d, image_shape):
            """Create distance heatmap from 2D coordinates."""
            n_images, height, width = image_shape
            n_points = coords_2d.shape[1]
            
            # Create pixel coordinate grid
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            pixel_coords = np.stack([x_coords, y_coords], axis=-1)  # (H, W, 2)
            
            # Calculate max distance for normalization
            max_distance = np.sqrt(height**2 + width**2)
            
            # Initialize heatmap
            heatmap = np.zeros((n_images, height, width, n_points))
            
            # Calculate distances for each point
            for i in range(n_points):
                target_points = coords_2d[:, i]  # (n_images, 2)
                # Vectorized distance calculation
                distances = np.linalg.norm(
                    pixel_coords[None, :, :, :] - target_points[:, None, None, :], 
                    axis=-1
                )  # (n_images, H, W)
                heatmap[:, :, :, i] = distances
            
            # Apply square root transformation for steeper near-target gradients
            heatmap = np.sqrt(heatmap / max_distance) * 255
            return np.clip(heatmap, 0, 255).astype(np.uint8)
        
        # Project goal gripper points to camera coordinates
        agentview_coords = gdu.project_points(
            obs_dict['goal_gripper_pcd'][()],
            self.agentview_intrinsics,
            self.world_to_agentview_extrinsics,
            normalization_factor=normalization_factor
        )
        
        wristview_coords = gdu.project_points(
            obs_dict['goal_gripper_pcd'][()],
            self.eye_in_hand_intrinsics,
            obs_dict['robot0_eye_in_hand_extrinsics'][()],
            normalization_factor=normalization_factor
        )
        
        # Project to 2D images and get shapes
        agentview_image = gdu.coords_to_2d_image(
            agentview_coords, 
            obs_dict['agentview_image_84'][()], 
            clip_coords=True
        )
        
        wristview_image = gdu.coords_to_2d_image(
            wristview_coords, 
            obs_dict['robot0_eye_in_hand_image_84'][()], 
            clip_coords=True
        )
        
        # Generate heatmaps
        agentview_cond_84 = _create_distance_heatmap(agentview_coords, agentview_image.shape)
        wristview_cond_84 = _create_distance_heatmap(wristview_coords, wristview_image.shape)
        
        return agentview_cond_84, wristview_cond_84


