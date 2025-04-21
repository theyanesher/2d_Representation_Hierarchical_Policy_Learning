import numpy as np
import torch
from manipulation.utils import get_pc, rotation_transfer_6D_to_matrix, rotation_transfer_matrix_to_6D, get_pixel_location, get_matrix_from_pos_rot
import pybullet as p
import numpy as np
from copy import deepcopy
import pytorch3d.ops as torch3d_ops
from gym import spaces
import open3d as o3d
import time
from scipy.spatial.transform import Rotation as R
import fpsample
import os
import json
import pickle
import cv2
import scipy
from scipy.interpolate import RectBivariateSpline
from scipy.spatial.distance import cdist

class RobogenPointCloudWrapper:
    def __init__(self, 
                 env, 
                 object_name, 
                 link_name,
                 rpy_mean_list=[[0, 0, -45], [0, 0, -135]], 
                 seed=None, 
                 in_gripper_frame=False,
                 num_points=4500,
                 handle_num_points=0,
                 horizon=400,
                 include_contact=False,
                 gripper_num_points=0, # 500
                 gripper_bbox=0.1, 
                 add_contact=False,
                 use_joint_angle=False,
                 use_color=False,
                 use_segmask=False,
                 only_handle_points=False,
                 observation_mode=None,
                 camera_height=480,
                 camera_width=640,
                 elevation=30,
                 only_object=True,
                 record_all_observation=False,
                 noise_real_world_pcd=False,
                 real_world_camera=False,
            ):
        np.random.seed(time.time_ns() % 2**32)
        if seed is not None:
            np.random.seed(seed)

        self.filter_with_plane = None # [a, b, c, d], filter the point cloud that ax + by + cz + d > 0

        self.noise_real_world_pcd = noise_real_world_pcd
        self.real_world_camera = real_world_camera

        self._env = env
        self._object_name = object_name
        self._link_name = link_name
        self.horizon = horizon
        self.record_all_observation = record_all_observation
        
        self.in_gripper_frame = in_gripper_frame
        self.num_points = num_points
        self.handle_num_points = handle_num_points
        self.include_contact = include_contact
        self.gripper_num_points = gripper_num_points
        self.gripper_bbox = gripper_bbox
        self.add_contact = add_contact
        self.use_joint_angle = use_joint_angle
        self.use_color = use_color
        self.use_segmask = use_segmask
        self.only_handle_points = only_handle_points
        self.observation_mode = observation_mode
        self.elevation = elevation

        self.camera_width = camera_width
        self.camera_height = camera_height

        self.action_low = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        self.action_high = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, dtype=np.float32)

        if 'dp3' in observation_mode:
            self.observation_space = spaces.Dict({
                'point_cloud': spaces.Box(low=-np.inf, high=np.inf, shape=(1, 1280, 3), dtype=np.float32),
                'agent_pos': spaces.Box(low=-np.inf, high=np.inf, shape=(1, 10), dtype=np.float32), # pos(3) + orient(6) + joint_angle(1): we use 6D representation for orientation
            })
        elif 'act3d_goal_mlp' in observation_mode:
            self.observation_space = spaces.Dict({
                'point_cloud': spaces.Box(low=-np.inf, high=np.inf, shape=(1, 1280, 3), dtype=np.float32),
                'agent_pos': spaces.Box(low=-np.inf, high=np.inf, shape=(1, 10), dtype=np.float32), # pos(3) + orient(6) + joint_angle(1): we use 6D representation for orientation
                'gripper_pcd': spaces.Box(low=-np.inf, high=np.inf, shape=(1, 4, 3), dtype=np.float32), # pos(3) + orient(6) + joint_angle(1): we use 6D representation for orientation
            })
        else:
            self.observation_space = spaces.Dict({
                'point_cloud': spaces.Box(low=-np.inf, high=np.inf, shape=(1, 1280, 3), dtype=np.float32),
                'agent_pos': spaces.Box(low=-np.inf, high=np.inf, shape=(1, 10), dtype=np.float32), # pos(3) + orient(6) + joint_angle(1): we use 6D representation for orientation
                'gripper_pcd': spaces.Box(low=-np.inf, high=np.inf, shape=(1, 4, 3), dtype=np.float32), # pos(3) + orient(6) + joint_angle(1): we use 6D representation for orientation
                'feature_map': spaces.Box(low=-np.inf, high=np.inf, shape=(1, 128, 128, 3), dtype=np.float32), # pos(3) + orient(6) + joint_angle(1): we use 6D representation for orientation
                'pcd_mask': spaces.Box(low=-np.inf, high=np.inf, shape=(1, 1280, 1), dtype=np.uint8), # pos(3) + orient(6) + joint_angle(1): we use 6D representation for orientation
            })

        if 'act3d' in observation_mode: 
            if 'goal' in observation_mode:
                self.observation_space['goal_gripper_pcd'] = spaces.Box(low=-np.inf, high=np.inf, shape=(1, 4, 3), dtype=np.float32) # pos(3) + orient(6) + joint_angle(1): we use 6D representation for orientation
            if 'displacement_gripper_to_object' in observation_mode:
                self.observation_space['displacement_gripper_to_object'] = spaces.Box(low=-np.inf, high=np.inf, shape=(1, 4, 3), dtype=np.float32) # pos(3) + orient(6) + joint_angle(1): we use 6D representation for orientation


        for name in self._env.urdf_ids: # randomly center at an object
            if name in ['robot', 'plane', 'init_table']: continue
            obj_id = self._env.urdf_ids[name]
            min_aabb, max_aabb = self._env.get_aabb(obj_id)
            center = (min_aabb + max_aabb) / 2
            self.mean_camera_target = center 
            self.mean_distance = np.linalg.norm(max_aabb - min_aabb) * 1.15
            break
        self.rpy_mean_list = rpy_mean_list
        if self.rpy_mean_list is None:
            self.rpy_mean_list = [[0, 0, -45], [0, 0, -135]]
        
        if 'act3d' in self.observation_mode:
            # TODO: handle multiple camera for act3d observation
            # TODO: figure out the right camera distance & position
            # self.rpy_mean_list = [[0, 0, -45]]
            self.rpy_mean_list = [[-10, 0, -45], [-10, 0, -135]]
            self.mean_distance = np.linalg.norm(max_aabb - min_aabb) * 0.9
            self.camera_height = 256
            self.camera_width = 256
            

        self.depth_near = 0.01
        self.depth_far = 100
        self.view_matrices = []
        self.project_matrices = []

        for rpy_mean in self.rpy_mean_list:
            # rpy = np.array(rpy_mean) + np.random.normal(0, 8, 3)
            # camera_center = self.mean_camera_target + np.random.normal(0, 0.05, 3)
            # distance = self.mean_distance + np.random.normal(0, 0.05, 1)
            rpy = np.array(rpy_mean)
            camera_center = self.mean_camera_target
            distance = self.mean_distance

            ### TODO: check the aspect ratio
            view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=camera_center, distance=distance, yaw=rpy[2], pitch=rpy[0], roll=rpy[1], upAxisIndex=2, physicsClientId=env.id)
            # project_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=640/480 ,nearVal=0.01, farVal=100, physicsClientId=env.id)
            project_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1 ,nearVal=self.depth_near, farVal=self.depth_far, physicsClientId=env.id)
            self.view_matrices.append(view_matrix)
            self.project_matrices.append(project_matrix)
            
        if self.real_world_camera:
            # TODO: make the camera at real-world pose
            self.camera_width = 640
            self.camera_height = 576
            
            camera_ids = [0, 3]
            view_matrices = []
            project_matrices = []
            camera_calibration_folder = os.path.join(os.environ["PROJECT_DIR"], 'data/real_world')
            self.camera_eyes = []
            for camera_id in camera_ids:
                camera_parameter_file = os.path.join(camera_calibration_folder, "cam{}_calibration.npz".format(camera_id))
                data = np.load(camera_parameter_file)
                camera_extrinsic = data['T'] # 4x4


                camera_eye = camera_extrinsic[:3, 3]
                camera_eye[2] = 1.0
                camera_target = [0.7, 0, 0.4]
                camera_eye = camera_eye + np.random.normal(0, 0.1, 3)
                camera_target = camera_target + np.random.normal(0, 0.1, 3)
                self.camera_eyes.append(camera_eye)

                view_matrix = p.computeViewMatrix(camera_eye, camera_target, [0, 0, 1])
                project_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=640/576 ,nearVal=self.depth_near, 
                                                              farVal=self.depth_far, physicsClientId=self._env.id)
                view_matrices.append(view_matrix)
                project_matrices.append(project_matrix)

            self.view_matrices = view_matrices
            self.project_matrices = project_matrices


        self.time_step = 0
        
        # [Chialiang]
        if "act3d_goal" in self.observation_mode or 'dp3_goal_gripper' in self.observation_mode:

            # [Chialiang]
            config_path = self._env.config_path
            task_name = self._env.task_name
            parent_path = os.path.dirname(config_path)
            if not self._env.mobile:
                state_path = os.path.join(parent_path, "states") 
            else:
                state_path = os.path.join(parent_path, "mobile_states")
                
            stage_lengths_json_file = os.path.join(parent_path, 'stage_lengths.json')
            with open(stage_lengths_json_file, 'r') as f:
                stage_lengths = json.load(f)
            open_begin_t_idx = stage_lengths['reach_handle'] + stage_lengths['reach_to_contact'] + stage_lengths['close_gripper']
            all_time_steps = stage_lengths['reach_handle'] + stage_lengths['reach_to_contact'] + stage_lengths['close_gripper'] + stage_lengths['open_door']
            goal_1_state = os.path.join(state_path, "state_{}.pkl".format(open_begin_t_idx))
            goal_2_state = os.path.join(state_path, "state_{}.pkl".format(all_time_steps - 1))
            
            # NOTE: load the goal state, reset the robot to there, record the eef pose as the goal.
            with open(goal_1_state, 'rb') as f:
                goal_1_state = pickle.load(f)
            with open(goal_2_state, 'rb') as f:
                goal_2_state = pickle.load(f)
            
            self._env.reset(reset_state=goal_1_state, object_name=self._object_name)
            grasping_eef_pc = self.get_gripper_pc()

            # Chialiang for dense goal pcd
            eef_pos, eef_rot = self._env.robot.get_pos_orient(self._env.robot.right_end_effector)
            self.grasping_goal_pose = get_matrix_from_pos_rot(eef_pos, eef_rot)
            
            self._env.reset(reset_state=goal_2_state, object_name=self._object_name)
            final_eef_pc = self.get_gripper_pc()
            
            # Chialiang for dense goal pcd
            eef_pos, eef_rot = self._env.robot.get_pos_orient(self._env.robot.right_end_effector)
            self.final_goal_pose = get_matrix_from_pos_rot(eef_pos, eef_rot)

            self.grasping_goal = grasping_eef_pc
            self.final_goal = final_eef_pc
            
            self.grasped_handle = False
            
            self.goal_gripper_pcd = None

        self.only_object = only_object

    def reset_random_cameras(self):
        # do a while loop to sample a new camera view
        try_times = 0

        # get handle point cloud
        # link_pc = get_link_pc(self._env, self._object_name, 'link_0')
        # link_pc = self._env.get_link_pc(self._object_name, self._link_name)
        all_handle_pos, _, _, _= self._env.get_handle_pos(self._object_name, return_median=False, custom_joint_name=self._env.handle_name)
        # handle_pc, handle_joint_id, handle_median, _ = get_link_handle(all_handle_pos, handle_joint_id, link_pc)
        handle_pc = np.concatenate(all_handle_pos, axis=0)

        while try_times < 1000:
            view_matrices = []
            project_matrices = []
            try_times += 1
            distance = np.random.uniform(0.8, 1.2) * self.mean_distance + np.random.normal(0, 0.05, 1)
            camera_center = self.mean_camera_target + np.random.normal(0, 0.05, 3)
            for _ in range(2):
                rpy = np.zeros(3)
                rpy[0] = np.random.uniform(-20, 20)
                rpy[1] = np.random.uniform(-40, 0)
                if np.random.uniform() > 0.5:
                    rpy[2] = np.random.uniform(-110, -160)
                else:
                    rpy[2] = np.random.uniform(-20, -70)

                view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=camera_center, distance=distance, yaw=rpy[2], pitch=rpy[0], roll=rpy[1], upAxisIndex=2, physicsClientId=self._env.id)
                project_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=640/480 ,nearVal=self.depth_near, farVal=self.depth_far, physicsClientId=self._env.id)
                view_matrices.append(view_matrix)
                project_matrices.append(project_matrix)

            self.view_matrices = view_matrices
            self.project_matrices = project_matrices
            
            if self.check_handle_observed_in_pc(handle_pc=handle_pc) > 3:
                break

            # import pdb; pdb.set_trace()

        if try_times >= 1000:
            raise ValueError("Cannot find a camera view that has handle points in the point cloud")

    def reset_real_world_camera(self, camera_calibration_folder="/data/ziyuw2/Projects/RoboGen-sim2real/manipulation/real_world_camera_calibration", add_randomness=False):
        # get the camera calibration parameters
        camera_ids = [0, 2, 3]
        view_matrices = []
        project_matrices = []
        for camera_id in camera_ids:
            camera_parameter_file = os.path.join(camera_calibration_folder, "cam{}_calibration.npz".format(camera_id))
            data = np.load(camera_parameter_file)
            camera_extrinsic = data['T'] # 4x4
            # view_matrix = np.linalg.inv(camera_extrinsic)
            # # view_matrix = camera_extrinsic
            # view_matrix = view_matrix.reshape(-1, order='F')
            # print("view_matrix: ", view_matrix)

            camera_eye = camera_extrinsic[:3, 3]
            camera_target = [0.7, -0.3, 0.15]
            if add_randomness:
                camera_target = [0.7 + np.random.uniform(-0.05, 0.05), -0.3 + np.random.uniform(-0.05, 0.05), 0.15 + np.random.uniform(-0.05, 0.05)]
                camera_eye = camera_eye + np.random.uniform(-0.05, 0.05, 3)
            view_matrix = p.computeViewMatrix(camera_eye, camera_target, [0, 0, 1])
            project_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=640/480 ,nearVal=self.depth_near, farVal=self.depth_far, physicsClientId=self._env.id)
            view_matrices.append(view_matrix)
            project_matrices.append(project_matrix)

        self.view_matrices = view_matrices
        self.project_matrices = project_matrices

    def check_handle_observed_in_pc(self, handle_pc=None):
        # given the current camera view, check if the handle is observed in the point cloud
        # return the number of points that are close to the handle
        if handle_pc is None:
            # get handle point cloud
            # link_pc = get_link_pc(self._env, self._object_name, 'link_0')
            all_handle_pos, _, _, _ = self._env.get_handle_pos(self._object_name, return_median=False, custom_joint_name=self._env.handle_name)
            # handle_pc, handle_joint_id, handle_median, _ = get_link_handle(all_handle_pos, handle_joint_id, link_pc)
            handle_pc = np.concatenate(all_handle_pos, axis=0)
        pcs = []
        rgbs, depths, segmasks, view_camera_matrices, project_camera_matrices = self.take_images_around_object(self._env, self._object_name.lower(), elevation=self.elevation,
                                            return_camera_matrices=True, camera_height=self.camera_height, camera_width=self.camera_width,
                                            only_object=True)
        for rgb, depth, segmask, view_matrix, project_matrix in zip(rgbs, depths, segmasks, view_camera_matrices, project_camera_matrices):
            pc = get_pc(proj_matrix=project_matrix, view_matrix=view_matrix, depth=depth, width=self.camera_width, height=self.camera_height, mask_infinite=False)
            
            segmask_obj_id = segmask & ((1 << 24) - 1)
            object_mask = np.zeros_like(depth).astype(np.float32)
            object_mask[segmask_obj_id == self._env.urdf_ids[self._object_name]] = 1
            object_mask_ = np.flatnonzero(object_mask.flatten())
            pcs.append(pc[object_mask_])
            
        pcs = np.concatenate(pcs, axis=0)

        ### visualize the point cloud
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pcs)
        # o3d.visualization.draw_geometries([pcd])

        # check there are some handle points in the point cloud
        # device = torch.device('cpu')
        
        # handle_pc_torch = torch.from_numpy(handle_pc).to(device)
        # pcs_torch = torch.from_numpy(pcs).to(device)
        # pcd_distance = torch.norm(pcs_torch.unsqueeze(1) - handle_pc_torch.unsqueeze(0), dim=-1)
        # min_distance = torch.min(pcd_distance, dim=-1)[0]
        # min_distance = min_distance[min_distance < 0.02]
        
        ### use scipy to compute the distance
        # handle_pc: (N, D), pcs: (M, D)

        # import pdb; pdb.set_trace()
        # Calculate pairwise distances
        pcd_distance = cdist(pcs, handle_pc)  # Shape: (M, N)

        # Find the minimum distance for each point in pcs
        min_distance = np.min(pcd_distance, axis=1)  # Shape: (M,)

        # Filter the distances that are less than 0.02
        min_distance = min_distance[min_distance < 0.02]
        
        
        return min_distance.shape[0]

    def reset(self, **kwargs):
        if "act3d_goal" in self.observation_mode:
            self.grasped_handle = False
        self._env.reset(**kwargs)
        self._env._get_info(object_name=self._object_name, handle_name=self._env.handle_name, link_name=self._link_name)
        self.time_step = 0
        return self._get_observation(only_object=self.only_object)
    
    def step(self, action, render=True):
        # beg = time.time()
        if not self.use_joint_angle:
            # beg = time.time()
            pos, orient = self._env.robot.get_pos_orient(self._env.robot.right_end_effector)
            current_rotate_matrix = np.array(p.getMatrixFromQuaternion(orient)).reshape(3, 3)
            
            # transfer the action to the gripper frame
            if self.in_gripper_frame:
                action[:3] = current_rotate_matrix @ np.array(action[:3])
                

            delta_orient = action[3:9]

            delta_rotate_matrix = rotation_transfer_6D_to_matrix(delta_orient)

            after_rotate_matrix = current_rotate_matrix @ delta_rotate_matrix
            
            orient = R.from_matrix(after_rotate_matrix).as_quat()
            euler = p.getEulerFromQuaternion(orient)

            cur_joint_angle = p.getJointState(self._env.robot.body, self._env.robot.right_gripper_indices[0], physicsClientId=self._env.id)

            pos = pos + np.array(action[:3])
            target_joint_angle = action[9] + cur_joint_angle[0]
            
            action = pos.tolist() + list(euler) + [target_joint_angle]
            # end = time.time()
            # cprint("preprocessing time {}".format(end - beg), "green")

            # beg = time.time()
            self._env.take_direct_action(action)
            # beg = time.time()
            # end = time.time()
            # cprint("take direct action time {}".format(end - beg), "blue")
        else:
            self._env.take_joint_action(action)
        
        beg = time.time()
        try:
            reward, success = self._env._compute_reward()
        except:
            reward, success = self._env.compute_reward()
        # end = time.time()

        # beg = time.time()
        info = self._env._get_info(object_name=self._object_name, handle_name=self._env.handle_name, link_name=self._link_name)
        done = self._env.time_step >= self.horizon
        # end = time.time()
        # cprint("compute reward & get info time {}".format(end - beg), "green")
        
        # beg = time.time()
        obs = self._get_observation(render=render, only_object=self.only_object)
        end = time.time()
        # cprint("get observation time {}".format(end - beg), "green")
        # cprint("step in robogen wrapper time {}".format(end - beg), "green")
        
        self.time_step += 1
        return obs, reward, done, info
    
    def get_gripper_pc(self):
        # get the point cloud of the gripper
        right_finger_pos, _ = self._env.robot.get_pos_orient(self._env.robot.right_gripper_indices[0])
        left_finger_pos, _ = self._env.robot.get_pos_orient(self._env.robot.right_gripper_indices[1])
        right_hand_pos, _ = self._env.robot.get_pos_orient(self._env.robot.right_hand)
        eef_pos, _ = self._env.robot.get_pos_orient(self._env.robot.right_end_effector)
        gripper_pc = np.array([right_hand_pos, right_finger_pos, left_finger_pos, eef_pos]).reshape(-1, 3)
        return gripper_pc.astype(np.float32)
    
    def _get_act3d_observation(self, rgbs, depths, segmasks, view_camera_matrices, project_camera_matrices, using_torch=False, only_object=True):
        pos, orient = self._env.robot.get_pos_orient(self._env.robot.right_end_effector)

        # get the 6D representation of orientation
        rotate_matrix = p.getMatrixFromQuaternion(orient)
        orient = rotation_transfer_matrix_to_6D(rotate_matrix)

        cur_joint_angle = p.getJointState(self._env.robot.body, self._env.robot.right_gripper_indices[0], physicsClientId=self._env.id)

        pos_ori = pos.tolist() + orient.tolist() + [cur_joint_angle[0]]
        
        # I need point cloud of target object
        # full segmentation mask + depth image, stacked together
        # gripper point cloud, which can be the left finger point, right finger point, and the eef point, and the grasping target point
        # gripper information
        pcs = []
        feature_maps = []
        gripper_pcd = []
        pcd_mask_indices = []
        for rgb, depth, segmask, view_matrix, project_matrix in zip(rgbs, depths, segmasks, view_camera_matrices, project_camera_matrices):
            
            beg = time.time()
            pc = get_pc(proj_matrix=project_matrix, view_matrix=view_matrix, depth=depth, width=self.camera_width, height=self.camera_height, mask_infinite=False)
            # print("get_pc time: ", time.time() - beg)
    
            gripper_pc = self.get_gripper_pc()
            gripper_pcd.append(gripper_pc)
            
            segmask_obj_id = segmask & ((1 << 24) - 1)
            robot_mask = np.zeros_like(depth).astype(np.float32)
            robot_mask[segmask_obj_id == self._env.urdf_ids['robot']] = 1
            object_mask = np.zeros_like(depth).astype(np.float32)
            object_mask[segmask_obj_id == self._env.urdf_ids[self._object_name]] = 1
            
            object_mask_ = np.flatnonzero(object_mask.flatten())
            pcs.append(pc[object_mask_])
            
            if not only_object:
                ret_object_mask = np.zeros_like(depth).astype(np.float32)
                # get the bounding box of the object mask
                min_bound = np.min(np.argwhere(object_mask), axis=0)
                max_bound = np.max(np.argwhere(object_mask), axis=0)
                x_min, y_min = min_bound
                x_max, y_max = max_bound
                x_range = x_max - x_min
                y_range = y_max - y_min
                x_min_new = int(max(0, x_min - x_range * np.random.uniform(0.01, 0.05)))
                x_max_new = int(min(self.camera_height, x_max + x_range * np.random.uniform(0.01, 0.05)))
                y_min_new = int(max(0, y_min - y_range * np.random.uniform(0.01, 0.05)))
                y_max_new = int(min(self.camera_width, y_max + y_range * np.random.uniform(0.01, 0.05)))
                ret_object_mask[x_min_new:x_max_new, y_min_new:y_max_new] = 1
                ret_object_mask_indices = np.flatnonzero(ret_object_mask.flatten())
                pcd_ = pc[ret_object_mask_indices]
                object_mask_ = np.flatnonzero(object_mask.flatten())
                object_pcd_ = pc[object_mask_]
                mean_object_pcd = np.mean(object_pcd_, axis=0)
                # crop the pcd_ to be near mean_object_pcd
                distance = np.linalg.norm(pcd_ - mean_object_pcd, axis=1)
                indices = np.flatnonzero(distance < 1.0)
                object_mask_indices = ret_object_mask_indices[indices]
                object_mask = np.zeros_like(depth).astype(np.float32).flatten()
                object_mask[object_mask_indices] = 1
                object_mask = object_mask.reshape(self.camera_height, self.camera_width)
                

            if "displacement_to_handle" in self.observation_mode:
                info = self._env._get_info(object_name=self._object_name, handle_name=self._env.handle_name, link_name=self._link_name)
                handle_pos = np.array(info['handle_pos'])
                delta_to_handle = handle_pos.reshape(1, 3) - pc
                feature_map = np.dstack([robot_mask, object_mask, pc.reshape(self.camera_height, self.camera_width, 3), delta_to_handle.reshape(self.camera_height, self.camera_width, 3)])
                assert feature_map.shape == (self.camera_height, self.camera_width, 8), f"Expected ({self.camera_height}, {self.camera_width}, 8), got {feature_map.shape}"
                feature_maps.append(feature_map)
                
            else:
                feature_map = np.dstack([robot_mask, object_mask, pc.reshape(self.camera_height, self.camera_width, 3)])
                assert feature_map.shape == (self.camera_height, self.camera_width, 5), f"Expected ({self.camera_height}, {self.camera_width}, 5), got {feature_map.shape}"
                feature_maps.append(feature_map)
            
            object_mask_indices = np.flatnonzero(object_mask.flatten())
            pcd_mask_indices.append(object_mask_indices)
        
            if 'goal' in self.observation_mode:
                # add goal as part of the observation. 
                # needs to judge when to switch the goal -- check if the handle has been grasped.  
                
                if self._env.grasped_handle:
                    # print("goal is to open the door")
                    goal_gripper_pcd = self.final_goal
                else:
                    # print("goal is to grasp the handle")
                    goal_gripper_pcd = self.grasping_goal
                self.goal_gripper_pcd = goal_gripper_pcd
        
        
        point_cloud = np.concatenate(pcs, axis=0)
        
        
        
        ### perform whatever we do in the real world 
        if self.noise_real_world_pcd:
            ### use open3d to visualize the point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)
            # o3d.visualization.draw_geometries([pcd])    
            
            ### NOTE: check these values
            real_world_radius = 0.02
            real_world_nb_neighbors = 20
            real_world_std_ratio = np.random.uniform(0.4, 0.6)
            real_world_voxel_size = 0.002
            ratio = np.random.uniform(0.3, 0.95)
            real_world_nb_points = int(((real_world_radius / real_world_voxel_size) ** 2) * ratio)

            beg = time.time()
            pcd = pcd.voxel_down_sample(real_world_voxel_size)
            # print("voxel down sample time: ", time.time() - beg)
            
            beg = time.time()
            pcd2, indices = pcd.remove_radius_outlier(nb_points=real_world_nb_points, radius=real_world_radius)
            # self.display_inlier_outlier(pcd, indices)
            # print("remove radius outlier time: ", time.time() - beg)
            
            # pcd2_py, indices_py = remove_radius_outliers_vectorized(np.array(pcd.points), real_world_nb_points, real_world_radius)
            # self.display_inlier_outlier(pcd, indices_py)
            
            beg = time.time()
            pcd3, indices = pcd2.remove_statistical_outlier(nb_neighbors=real_world_nb_neighbors, std_ratio=real_world_std_ratio)
            # self.display_inlier_outlier(pcd2, indices)
            # print("remove statistical outlier time: ", time.time() - beg)
            # o3d.visualization.draw_geometries([pcd3])
                    
            point_cloud = np.array(pcd3.points)
            distance_to_camera_eye_1 = np.linalg.norm(self.camera_eyes[0] - point_cloud, axis=1)
            distance_to_camera_eye_2 = np.linalg.norm(self.camera_eyes[1] - point_cloud, axis=1)
            condition = np.logical_and(distance_to_camera_eye_1 > 0.1, distance_to_camera_eye_2 > 0.1)
            point_cloud = point_cloud[condition]
            
            # pcd_o3d = o3d.geometry.PointCloud()
            # pcd_o3d.points = o3d.utility.Vector3dVector(point_cloud)
            # o3d.visualization.draw_geometries([pcd_o3d])
        
        # filter the point cloud with the segmentation plane
        if self.filter_with_plane is not None:
            # filter the point cloud that satisfies 
            # self.filter_with_plane[0] * x + self.filter_with_plane[1] * y + self.filter_with_plane[2] * z + self.filter_with_plane[3] > 0
            plane_normal = np.array(self.filter_with_plane[:3])
            plane_d = self.filter_with_plane[3]
            distance = np.dot(point_cloud, plane_normal) + plane_d
            point_cloud = point_cloud[distance > 0]
            
        beg = time.time()
        num_points = self.num_points
        if using_torch:
            point_cloud = torch.from_numpy(point_cloud).unsqueeze(0).cuda()
            num_points = torch.tensor([num_points]).cuda()
            _, sampled_indices = torch3d_ops.sample_farthest_points(points=point_cloud[...,:3], K=num_points)
            sampled_indices = sampled_indices.squeeze(0).cpu().numpy()
            sampled_indices = np.array(sorted(sampled_indices))
            point_cloud = point_cloud.squeeze(0).cpu().numpy()
            point_cloud = point_cloud[sampled_indices]
        else:
            if point_cloud.shape[0] < num_points:
                to_add_points_num = num_points - point_cloud.shape[0]
                random_sampled_points = np.random.choice(point_cloud.shape[0], to_add_points_num, replace=True)
                point_cloud = np.concatenate([point_cloud, point_cloud[random_sampled_points]], axis=0)
            
            h = min(9, np.log2(num_points))
            kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(point_cloud[:, :3], num_points, h=h)
            kdline_fps_samples_idx = np.array(sorted(kdline_fps_samples_idx))
            point_cloud = point_cloud[kdline_fps_samples_idx]

        # print("fps time: ", time.time() - beg)            
        # pcd5 = o3d.geometry.PointCloud()
        # pcd5.points = o3d.utility.Vector3dVector(point_cloud)
        # o3d.visualization.draw_geometries([pcd5])
        
        point_cloud = point_cloud.tolist()
        
        obs_dict_input = {}
        obs_dict_input['point_cloud'] = np.array(point_cloud).astype(np.float32)
        obs_dict_input['agent_pos'] = np.array(pos_ori).astype(np.float32)
        
        obs_dict_input['gripper_pcd'] = gripper_pcd[0].astype(np.float32)
        if 'goal' in self.observation_mode:
            obs_dict_input['goal_gripper_pcd'] = goal_gripper_pcd
  
        if 'displacement_gripper_to_object' in self.observation_mode:
            gripper_pcd = obs_dict_input['gripper_pcd']
            object_pcd = obs_dict_input['point_cloud']
            distance = scipy.spatial.distance.cdist(gripper_pcd, object_pcd)
            min_distance_obj_idx = np.argmin(distance, axis=1)
            closest_point = object_pcd[min_distance_obj_idx]
            displacement = closest_point - gripper_pcd
            obs_dict_input['displacement_gripper_to_object'] = displacement.astype(np.float32)
                
        return obs_dict_input
    
    def _get_dp3_observation(self, rgbs, depths, segmasks, view_camera_matrices, project_camera_matrices, using_torch=False, only_object=True):
        pos, orient = self._env.robot.get_pos_orient(self._env.robot.right_end_effector)

        # get the 6D representation of orientation
        rotate_matrix = p.getMatrixFromQuaternion(orient)
        orient = rotation_transfer_matrix_to_6D(rotate_matrix)

        cur_joint_angle = p.getJointState(self._env.robot.body, self._env.robot.right_gripper_indices[0], physicsClientId=self._env.id)

        pos_ori = pos.tolist() + orient.tolist() + [cur_joint_angle[0]]
        
        pcs = []
        feature_maps = []
        gripper_pcd = []
        pcd_mask_indices = []
        for rgb, depth, segmask, view_matrix, project_matrix in zip(rgbs, depths, segmasks, view_camera_matrices, project_camera_matrices):
            
            pc = get_pc(proj_matrix=project_matrix, view_matrix=view_matrix, depth=depth, width=self.camera_width, height=self.camera_height, mask_infinite=False)
            
            # [Chialiang]
            if 'dp3_goal_gripper_whole' == self.observation_mode:
                # add goal as part of the observation. 
                # needs to judge when to switch the goal -- check if the handle has been grasped.  
                
                if self._env.grasped_handle:
                    # print("goal is to open the door")
                    goal_gripper_pcd = self.final_goal
                else:
                    # print("goal is to grasp the handle")
                    goal_gripper_pcd = self.grasping_goal
                self.goal_gripper_pcd = goal_gripper_pcd

                new_pcd = [pc]
                for goal_i in self.goal_gripper_pcd:
                    new_pcd_i = goal_i - pc
                    new_pcd.append(new_pcd_i)
                
                new_pcd = np.concatenate(new_pcd, axis=-1)
                pcs.append(new_pcd)

            # [Chialiang]
            elif 'dp3_goal_gripper_part' == self.observation_mode:

                # extract segmentation ids of the gripper part
                segmask_joint_id = (segmask - 1) >> 24
                robot_id = None
                for obj_name, obj_id in self._env.urdf_ids.items():
                    if obj_name == 'robot':
                        robot_id = obj_id
                assert robot_id is not None

                num_joints = p.getNumJoints(robot_id)
                joint_indexes = []
                for joint_index in range(num_joints):
                    info = p.getJointInfo(robot_id, joint_index)
                    if info[12].decode('utf-8') == 'panda_hand':
                        joint_indexes.append(joint_index)
                    if info[12].decode('utf-8') == 'panda_leftfinger':
                        joint_indexes.append(joint_index)
                    if info[12].decode('utf-8') == 'panda_rightfinger':
                        joint_indexes.append(joint_index)
                    if info[12].decode('utf-8') == 'panda_grasptarget':
                        joint_indexes.append(joint_index)
                
                segmask_joint_id = segmask_joint_id.reshape(-1)
                ee_mask = np.zeros(segmask_joint_id.shape).astype(bool)
                for joint_index in joint_indexes:
                    ee_mask[np.where(segmask_joint_id == joint_index)] = True

                if self._env.grasped_handle:
                    # print("goal is to open the door")
                    goal_gripper_pcd = self.final_goal
                else:
                    # print("goal is to grasp the handle")
                    goal_gripper_pcd = self.grasping_goal

                self.goal_gripper_pcd = goal_gripper_pcd

                new_pcd = [pc]
                for goal_i in self.goal_gripper_pcd:
                    new_pcd_i = np.zeros(pc.shape)
                    new_pcd_i[ee_mask] = goal_i - pc[ee_mask]
                    new_pcd.append(new_pcd_i)
                new_pcd = np.concatenate(new_pcd, axis=-1)

                # segmask_obj_id = segmask & ((1 << 24) - 1)
                # robot_mask = np.zeros_like(depth).astype(np.float32)
                # robot_mask[segmask_obj_id == self._env.urdf_ids['robot']] = 1
                # object_mask = np.zeros_like(depth).astype(np.float32)
                # object_mask[segmask_obj_id == self._env.urdf_ids[self._object_name]] = 1
                # robot_mask = robot_mask.reshape((-1, 1))
                # object_mask = object_mask.reshape((-1, 1))
                # pcd_cond = np.where((robot_mask > 0.5) | (object_mask > 0.5))[0]

                # o3d_pc = o3d.geometry.PointCloud()
                # o3d_pc.points = o3d.utility.Vector3dVector(new_pcd[pcd_cond][:,:3])
                # o3d.visualization.draw_geometries([o3d_pc])
                # o3d_pc.points = o3d.utility.Vector3dVector(new_pcd[pcd_cond][:,:3] + new_pcd[pcd_cond][:,3:6])
                # o3d.visualization.draw_geometries([o3d_pc])

                pcs.append(new_pcd)

            # [Chialiang]
            elif 'dp3_goal_gripper_dense' == self.observation_mode:

                # extract segmentation ids of the gripper part
                segmask_joint_id = (segmask - 1) >> 24
                robot_id = None
                for obj_name, obj_id in self._env.urdf_ids.items():
                    if obj_name == 'robot':
                        robot_id = obj_id
                assert robot_id is not None

                num_joints = p.getNumJoints(robot_id)
                joint_indexes = []
                for joint_index in range(num_joints):
                    info = p.getJointInfo(robot_id, joint_index)
                    if info[12].decode('utf-8') == 'panda_hand':
                        joint_indexes.append(joint_index)
                    if info[12].decode('utf-8') == 'panda_leftfinger':
                        joint_indexes.append(joint_index)
                    if info[12].decode('utf-8') == 'panda_rightfinger':
                        joint_indexes.append(joint_index)
                    if info[12].decode('utf-8') == 'panda_grasptarget':
                        joint_indexes.append(joint_index)
                
                segmask_joint_id = segmask_joint_id.reshape(-1)
                ee_mask = np.zeros(segmask_joint_id.shape).astype(bool)
                for joint_index in joint_indexes:
                    ee_mask[np.where(segmask_joint_id == joint_index)] = True

                if self._env.grasped_handle:
                    # print("goal is to open the door")
                    goal_gripper_pcd = self.final_goal
                    goal_gripper_mat = self.final_goal_pose
                else:
                    # print("goal is to grasp the handle")
                    goal_gripper_pcd = self.grasping_goal
                    goal_gripper_mat = self.grasping_goal_pose

                self.goal_gripper_pcd = goal_gripper_pcd

                eef_pos, eef_rot = self._env.robot.get_pos_orient(self._env.robot.right_end_effector)
                self.current_gripper_pose = get_matrix_from_pos_rot(eef_pos, eef_rot)

                relative_pose = goal_gripper_mat @ np.linalg.inv(self.current_gripper_pose) 

                pc_homo = np.ones((len(pc), 4))
                pc_homo[:,:3] = pc
                pc_transform = (pc_homo @ relative_pose.T)[:,:3]

                new_pcd = np.zeros((len(pc), 6))
                new_pcd[:,:3] = pc 
                new_pcd[ee_mask,3:] = pc_transform[ee_mask] - pc[ee_mask]

                pcs.append(new_pcd)
        
        point_cloud = np.concatenate(pcs, axis=0)
        min_bound = np.array([-5, -5, 0.1])
        max_bound = np.array([5, 5, 5])
        input_pc_mask = np.all(point_cloud[:, :3] > min_bound, axis=1) & np.all(point_cloud[:, :3] < max_bound, axis=1)
        masked_indices = np.flatnonzero(input_pc_mask)
        point_cloud = point_cloud[input_pc_mask]
        
        if self.handle_num_points > 0 or self.gripper_num_points > 0:
            full_point_cloud = deepcopy(point_cloud)

        # filter the point cloud with the segmentation plane
        if self.filter_with_plane is not None:
            # filter the point cloud that satisfies 
            # self.filter_with_plane[0] * x + self.filter_with_plane[1] * y + self.filter_with_plane[2] * z + self.filter_with_plane[3] > 0
            plane_normal = np.array(self.filter_with_plane[:3])
            plane_d = self.filter_with_plane[3]
            distance = np.dot(point_cloud, plane_normal) + plane_d
            point_cloud = point_cloud[distance > 0]
                
        # do downsampling of the pcd
        num_points = self.num_points
        if using_torch:
            point_cloud = torch.from_numpy(point_cloud).unsqueeze(0).cuda()
            num_points = torch.tensor([num_points]).cuda()
            _, sampled_indices = torch3d_ops.sample_farthest_points(points=point_cloud[...,:3], K=num_points)
            sampled_indices = sampled_indices.squeeze(0).cpu().numpy()
            sampled_indices = np.array(sorted(sampled_indices))
            point_cloud = point_cloud.squeeze(0).cpu().numpy()
            point_cloud = point_cloud[sampled_indices]
        else:
            if point_cloud.shape[0] < num_points:
                to_add_points_num = num_points - point_cloud.shape[0]
                random_sampled_points = np.random.choice(point_cloud.shape[0], to_add_points_num, replace=True)
                point_cloud = np.concatenate([point_cloud, point_cloud[random_sampled_points]], axis=0)
            
            h = min(9, np.log2(num_points))
            kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(point_cloud[:, :3], num_points, h=h)
            kdline_fps_samples_idx = np.array(sorted(kdline_fps_samples_idx))
            point_cloud = point_cloud[kdline_fps_samples_idx]

        point_cloud = point_cloud.tolist()

        obs_dict_input = {}
        obs_dict_input['point_cloud'] = np.array(point_cloud).astype(np.float32)
        obs_dict_input['agent_pos'] = np.array(pos_ori).astype(np.float32)
            
        # [Chialiang]
        if self.observation_mode == 'dp3_goal_gripper_on_agent':

            if self._env.grasped_handle:
                # print("goal is to open the door")
                goal_gripper_pcd = self.final_goal
            else:
                # print("goal is to grasp the handle")
                goal_gripper_pcd = self.grasping_goal

            gripper_pc = self.get_gripper_pc()
            diff = goal_gripper_pcd - gripper_pc
            diff_flat = diff.reshape(-1)

            pos_ori = np.array(pos_ori).astype(np.float32)

            obs_dict_input['agent_pos'] = np.concatenate((pos_ori, diff_flat), axis=0)
        
        # [Chialiang]
        if self.observation_mode == 'dp3_goal_gripper_on_agent_abs':

            if self._env.grasped_handle:
                # print("goal is to open the door")
                goal_gripper_pcd = self.final_goal
            else:
                # print("goal is to grasp the handle")
                goal_gripper_pcd = self.grasping_goal

            pos_ori = np.array(pos_ori).astype(np.float32)
            obs_dict_input['agent_pos'] = np.concatenate((pos_ori, goal_gripper_pcd.reshape(-1)), axis=0)
            
        return obs_dict_input
    
    def add_edge_artifacts(self, depth_map):
        """
        Apply edge artifacts to a depth map using correlated depth noise via bilinear interpolation.
        
        Args:
            depth_map (numpy.ndarray): The input depth map of size (H, W).
        
        Returns:
            numpy.ndarray: The depth map with edge artifacts applied.
        """
        H, W = depth_map.shape
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))

        # Generate random shifts for each grid point
        shifts_x = np.random.normal(0, 0.5, size=(H, W))
        shifts_y = np.random.normal(0, 0.5, size=(H, W))
        
        # Apply shifts with a probability of 0.8
        mask = np.random.rand(H, W) < 0.8
        shifted_x = grid_x + shifts_x * mask
        shifted_y = grid_y + shifts_y * mask
        
        # Ensure shifted coordinates stay within valid bounds
        shifted_x = np.clip(shifted_x, 0, W - 1)
        shifted_y = np.clip(shifted_y, 0, H - 1)
        
        # Perform bilinear interpolation between original depth values and shifted grid
        interpolator = RectBivariateSpline(np.arange(H), np.arange(W), depth_map)
        adjusted_depth_map = interpolator(shifted_y, shifted_x, grid=False)

        return adjusted_depth_map

    def add_random_holes(self, depth_map):
        """
        Add random holes to a depth map to simulate irregularities in real-world depth maps.
        
        Args:
            depth_map (numpy.ndarray): The input depth map of size (H, W).
        
        Returns:
            numpy.ndarray: The depth map with random holes applied.
        """
        if np.random.rand() > 0.5:
        # if np.random.rand() > 1:
            # Skip random hole generation with probability 0.5
            return depth_map

        H, W = depth_map.shape
        
        # Create a random mask from U(0,1)
        random_mask = np.random.uniform(0, 1, size=(H, W))
        
        # Apply Gaussian blur to smooth the mask
        smoothed_mask = cv2.GaussianBlur(random_mask, (5, 5), sigmaX=1, sigmaY=1)
        
        # Normalize the mask to the range [0, 1]
        smoothed_mask = (smoothed_mask - smoothed_mask.min()) / (smoothed_mask.max() - smoothed_mask.min())
        
        # Randomly sample a threshold from U(0.6, 0.9)
        threshold = np.random.uniform(0.6, 0.9)
        
        # Zero out pixels where mask values exceed the threshold
        depth_map_with_holes = np.copy(depth_map)
        depth_map_with_holes[smoothed_mask > threshold] = self.depth_near
        
        return depth_map_with_holes

            
    def augment_depth_image(self, depth_images):
        final_images = []
        for image in depth_images:
            
            real_depth = self.get_real_depth(image)
            
            max_depth = np.max(real_depth[real_depth < self.depth_far * 0.9])
            real_depth[real_depth > self.depth_far * 0.9] = max_depth
            beg = time.time()
            edge_augmented_image = self.add_edge_artifacts(real_depth)
            # print("add edge artifacts time: ", time.time() - beg)
            beg = time.time()
            hole_augmented_image = self.add_random_holes(edge_augmented_image)
            # print("add random holes time: ", time.time() - beg)
            bullet_depth = self.get_bullet_depth(hole_augmented_image)
            
            
            # ax = plt.subplot(1, 4, 1)
            # ax.imshow(image)
            # ax = plt.subplot(1, 3, 1)
            # ax.imshow(real_depth)
            # ax = plt.subplot(1, 3, 2)
            # ax.imshow(edge_augmented_image)
            # ax = plt.subplot(1, 3, 3)
            # ax.imshow(hole_augmented_image)
            # plt.show()
            
            final_images.append(bullet_depth)
        
        return final_images
    
    def _get_observation(self, render=True, using_torch=False, only_object=True):
        if render:
            
            beg = time.time()
            rgbs, depths, segmasks, view_camera_matrices, project_camera_matrices = \
                self.take_images_around_object(self._env, self._object_name.lower(), elevation=self.elevation,
                                                return_camera_matrices=True, camera_height=self.camera_height, camera_width=self.camera_width, 
                                                only_object=False)
            # print("take images time: ", time.time() - beg)

            # ax = plt.subplot(1, 2, 1)
#             ax.imshow(rgbs[0])
#             ax = plt.subplot(1, 2, 2)
#             ax.imshow(rgbs[1])
#             plt.show()
                
            ### TODO: augment depth
            beg = time.time()
            if self.noise_real_world_pcd:
                depths = self.augment_depth_image(depths)
            # print("augment depth time: ", time.time() - beg)
            
            if only_object:
                segmented_depths = []
                for depth, segmask in zip(depths, segmasks):
                    segmask_obj_id = segmask & ((1 << 24) - 1)
                    object_mask = np.zeros_like(depth).astype(np.float32)
                    object_mask[segmask_obj_id == self._env.urdf_ids[self._object_name.lower()]] = 1
                    object_mask = object_mask.reshape(self.camera_height, self.camera_width)
                    # let the object point cloud be the only point cloud
                    depth = depth * object_mask
                    depth[depth == 0] = self.depth_far
                    
                    segmented_depths.append(depth)
                depths = segmented_depths
                        
            if not self.record_all_observation:
                if 'act3d' in self.observation_mode:
                    act3d_obs_dict = self._get_act3d_observation(rgbs, depths, segmasks, view_camera_matrices, project_camera_matrices, using_torch=using_torch, only_object=only_object)
                    obs_dict_input = act3d_obs_dict
                elif 'dp3' in self.observation_mode:
                    dp3_obs_dict = self._get_dp3_observation(rgbs, depths, segmasks, view_camera_matrices, project_camera_matrices, using_torch=using_torch, only_object=only_object)
                    obs_dict_input = dp3_obs_dict
            else:
                self.observation_mode = 'act3d_goal_displacement_gripper_to_object' # this contains the most information
                act3d_obs_dict = self._get_act3d_observation(rgbs, depths, segmasks, view_camera_matrices, project_camera_matrices, using_torch=using_torch, only_object=only_object)
                self.observation_mode = 'dp3_goal_gripper_dense' # this contains the most information
                dp3_obs_dict = self._get_dp3_observation(rgbs, depths, segmasks, view_camera_matrices, project_camera_matrices, using_torch=using_torch, only_object=only_object)
                ### All other observations should be able to be reconstructed from the combination of the two above
                # e.g., for dp3_goal_gripper_on_agent, the point cloud observation should be obs_dict_input['dp3_point_cloud'][:, :3]
                # the agent_pos can be combined from using obs_dict_input['dp3_agent_pos'] and obs_dict_input['goal_gripper_pcd']
                # another NOTE: point_cloud in act3d only contains the object point cloud. while dp3_point_cloud contains the point cloud of the whole scene. 
                
                obs_dict_input = {}
                for key in act3d_obs_dict.keys():
                    obs_dict_input[key] = act3d_obs_dict[key]
                for key in dp3_obs_dict.keys():
                    obs_dict_input["dp3_" + key] = dp3_obs_dict[key]
        else:
            obs_dict_input = {}
            obs_dict_input['point_cloud'] = np.zeros((1, 1280, 6))
            obs_dict_input['agent_pos'] = np.array(pos_ori)
        
        return obs_dict_input
    
    def _get_diffuser_actor_observation(self):
        ret_dict = {}
        pos, orient = self._env.robot.get_pos_orient(self._env.robot.right_end_effector)
        rotate_matrix = p.getMatrixFromQuaternion(orient)
        orient = rotation_transfer_matrix_to_6D(rotate_matrix)
        cur_joint_angle = p.getJointState(self._env.robot.body, self._env.robot.right_gripper_indices[0], physicsClientId=self._env.id)
        pos_ori = pos.tolist() + orient.tolist() + [cur_joint_angle[0]]
        ret_dict['agent_pos'] = np.array(pos_ori)

        rgbs, depths, segmasks, view_camera_matrices, project_camera_matrices = \
                self.take_images_around_object(self._env, self._object_name.lower(), elevation=self.elevation,
                                                return_camera_matrices=True, camera_height=self.camera_height, camera_width=self.camera_width, 
                                                only_object=True)
        pcs = []
        for rgb, depth, segmask, view_matrix, project_matrix in zip(rgbs, depths, segmasks, view_camera_matrices, project_camera_matrices):
            pc = get_pc(proj_matrix=project_matrix, view_matrix=view_matrix, depth=depth, width=self.camera_width, height=self.camera_height, mask_infinite=False)
            pcs.append(pc)
        point_cloud = np.concatenate(pcs, axis=0)
        min_bound = np.array([-5, -5, -5])
        max_bound = np.array([5, 5, 5])
        input_pc_mask = np.all(point_cloud[:, :3] > min_bound, axis=1) & np.all(point_cloud[:, :3] < max_bound, axis=1)
        point_cloud = point_cloud[input_pc_mask]
        ret_dict['point_cloud'] = np.array(point_cloud)

        rgbs, depths, segmasks, view_camera_matrices, project_camera_matrices = \
                self.take_images_around_object(self._env, self._object_name.lower(), elevation=self.elevation,
                                                return_camera_matrices=True, camera_height=self.camera_height, camera_width=self.camera_width,
                                                only_object=False)
        
        # [TODO] Chialiang: It may need to be changed
        feature_maps = []
        for rgb, depth, segmask, view_matrix, project_matrix in zip(rgbs, depths, segmasks, view_camera_matrices, project_camera_matrices):
            near = self.depth_near
            far = self.depth_far
            depth = far * near / (far - (far - near) * depth)
            depth = depth.reshape(self.camera_height, self.camera_width, 1)
            feature = np.concatenate([rgb, depth], axis=2)
            feature_maps.append(feature)

        ret_dict['feature_map'] = np.array(feature_maps)
        ret_dict['gripper_pcd'] = np.zeros((1, 1, 1)).astype(np.float32)
        ret_dict['pcd_mask'] = np.zeros((1, 1, 1)).astype(np.uint8)
        return ret_dict
            
        
    def get_real_depth(self, depth):
        near = self.depth_near
        far = self.depth_far
        depth = far * near / (far - (far - near) * depth)
        return depth
    
    def get_bullet_depth(self, real_depth):
        near = self.depth_near
        far = self.depth_far
        depth = (far - far * near / real_depth) / (far - near)
        return depth
    
    
    def _transfer_point_cloud_to_gripper_frame(self, point_cloud):
        pos, orient = self._env.robot.get_pos_orient(self._env.robot.right_end_effector)
        # TODO: check this with my way of performing the transformation 
        # rotate_matrix = p.getMatrixFromQuaternion(orient)
        # rotate_matrix = np.array(rotate_matrix).reshape(3, 3)
        # rotate_matrix = np.linalg.inv(rotate_matrix)
        # new_pc = point_cloud.copy()
        # new_pc[:, :3] -= pos
        # new_pc[:, :3] = new_pc[:, :3] @ rotate_matrix
        
        
        T_body_to_world = np.eye(4) # transformation from the parent body frame to the world frame
        T_body_to_world[:3, :3] = np.array(p.getMatrixFromQuaternion(orient)).reshape(3, 3)
        T_body_to_world[:3, 3] = pos
        T_world_to_body = np.linalg.inv(T_body_to_world)
        point_cloud = np.array(point_cloud).reshape(-1, 3)
        point_cloud_homogeneous = np.concatenate([point_cloud, np.ones((point_cloud.shape[0], 1))], axis=1)
        transformed_pc_homogeneous = (T_world_to_body @ point_cloud_homogeneous.T).T
        transformed_pc = transformed_pc_homogeneous[:, :3]
        return transformed_pc

    
    def render(self):
        if 'goal' not in self.observation_mode or 'dp3' in self.observation_mode:
            return self._env.render()
        else:
            image = self._env.render()
            image = np.array(image)
            # import pdb; pdb.set_trace()
            for point in self.goal_gripper_pcd:
                pixel_x, pixel_y, _ = get_pixel_location(self._env.projection_matrix, self._env.view_matrix, point, self._env.camera_width, self._env.camera_height)
                color = (0, 0, 255)  # Red color in BGR
                thickness = 2
                radius = 5
                image = cv2.circle(image, (pixel_x, pixel_y), radius, color, thickness)
            return image

    
    def take_images_around_object(self, env, object_name, elevation=30, return_camera_matrices=False, camera_height=480, camera_width=640, only_object=True):
        # if only_object:
        #     ### make all other objects invisiable
        #     prev_rgbas = []
        #     object_id = env.urdf_ids[object_name]
        #     for obj_name, obj_id in env.urdf_ids.items():
        #         if obj_name != object_name and obj_name != 'robot':
        #             num_links = p.getNumJoints(obj_id, physicsClientId=env.id)
        #             for link_idx in range(-1, num_links):
        #                 prev_rgba = p.getVisualShapeData(obj_id, link_idx, physicsClientId=env.id)[0][14:18]
        #                 prev_rgbas.append(prev_rgba)
        #                 p.changeVisualShape(obj_id, link_idx, rgbaColor=[0, 0, 0, 0], physicsClientId=env.id)

        rgbs = []
        depths = []
        segmasks = []
        view_camera_matrices = []
        project_camera_matrices = []
        for view_matrix, project_matrix in zip(self.view_matrices, self.project_matrices):
            w, h, img, depth, segmask = p.getCameraImage(camera_width, camera_height, view_matrix, project_matrix, 
                                                         flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX, renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=env.id)
            img = np.reshape(img, (h, w, 4))[:, :, :3]
            depth = np.reshape(depth, (h, w))
            if only_object:
                segmask_obj_id = segmask & ((1 << 24) - 1)
                object_mask = np.zeros_like(depth).astype(np.float32)
                object_mask[segmask_obj_id == env.urdf_ids[object_name]] = 1
                object_mask = object_mask.reshape(self.camera_height, self.camera_width)
                # let the object point cloud be the only point cloud
                # other depth values are set to be 1000
                depth = depth * object_mask
                depth[depth == 0] = self.depth_far

            rgbs.append(img)
            depths.append(depth)
            segmasks.append(segmask)
            view_camera_matrices.append(view_matrix)
            project_camera_matrices.append(project_matrix)

        # if only_object:
        #     cnt = 0
        #     object_id = env.urdf_ids[object_name]
        #     for obj_name, obj_id in env.urdf_ids.items():
        #         if obj_name != object_name and obj_name != 'robot':
        #             num_links = p.getNumJoints(obj_id, physicsClientId=env.id)
        #             for link_idx in range(-1, num_links):
        #                 p.changeVisualShape(obj_id, link_idx, rgbaColor=prev_rgbas[cnt], physicsClientId=env.id)
        #                 cnt += 1

        return rgbs, depths, segmasks, view_camera_matrices, project_camera_matrices
    
    def display_inlier_outlier(self, cloud, ind):
        # Compute normals to help visualize
        # cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        inlier_cloud = cloud.select_by_index(ind)
        inlier_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        outlier_cloud = cloud.select_by_index(ind, invert=True)

        print("Showing outliers (red) and inliers (gray): ")
        outlier_cloud.paint_uniform_color([1, 0, 0])
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    def set_point_cloud_filter(self, filter_with_plane):
        # filter_with_plane: [a, b, c, d] where a*x + b*y + c*z + d > 0
        self.filter_with_plane = filter_with_plane


def get_link_handle(all_handle_pos, handle_joint_id, link_pc):
    handle_median_points = np.array([np.median(handle_pos, axis=0) for handle_pos in all_handle_pos]).reshape(-1, 3)
    distance_handle_median_to_link_pc = scipy.spatial.distance.cdist(handle_median_points, link_pc)
    min_distance = np.min(distance_handle_median_to_link_pc, axis=1)
    min_distance_handle_idx = np.argmin(min_distance)
    handle_joint_id = handle_joint_id[min_distance_handle_idx]
    handle_pc = all_handle_pos[min_distance_handle_idx]
    handle_median = handle_median_points[min_distance_handle_idx]
    
    threshold = 0.02
    pc_to_handle_distance = scipy.spatial.distance.cdist(link_pc, handle_pc).min(axis=1)
    handle_pc = link_pc[pc_to_handle_distance < threshold]
    
    return handle_pc, handle_joint_id, handle_median, min_distance_handle_idx

from scipy.spatial import cKDTree
def remove_radius_outliers_vectorized(pcd, nb_points, search_radius):
    """
    Removes radius outliers from a PointCloud using a fully vectorized approach.

    Args:
        pcd (open3d.geometry.PointCloud): The input point cloud.
        nb_points (int): Minimum number of neighbors within the radius.
        search_radius (float): Radius for searching neighbors.

    Returns:
        open3d.geometry.PointCloud: Point cloud with outliers removed.
        list[int]: Indices of the remaining points.
    """
    if nb_points < 1 or search_radius <= 0:
        raise ValueError("The number of points and radius must be positive.")

    # Convert the point cloud to a NumPy array
    points = pcd

    # Build a KDTree using scipy for efficient radius searches
    kdtree = cKDTree(points)

    # Perform a batch radius search
    neighbors = kdtree.query_ball_point(points, search_radius)

    # Determine which points have enough neighbors
    mask = np.array([len(neigh) > nb_points for neigh in neighbors])

    # Select the remaining points
    indices = np.where(mask)[0]
    inlier_pcd = points[indices]

    return inlier_pcd, indices.tolist()