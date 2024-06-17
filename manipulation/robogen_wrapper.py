import wandb
import numpy as np
import torch
import tqdm 
from manipulation.utils import get_pc, get_pc_in_camera_frame, rotation_transfer_6D_to_matrix, rotation_transfer_matrix_to_6D
from manipulation.gpt_reward_api import get_handle_pos
from manipulation.gpt_primitive_api import get_pc_num_within_gripper
import pybullet as p
import numpy as np
from copy import deepcopy
import pytorch3d.ops as torch3d_ops
import gym
from gym import spaces
import open3d as o3d
import matplotlib.pyplot as plt
import time
from termcolor import cprint
from scipy.spatial.transform import Rotation as R
import fpsample
import os
import json
import pickle

class RobogenPointCloudWrapper:
    def __init__(self, 
                 env, 
                 object_name, 
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
            ):
        np.random.seed(time.time_ns() % 2**32)
        if seed is not None:
            np.random.seed(seed)


        self._env = env
        self._object_name = object_name
        self.horizon = horizon
        
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
        self.observation_space = spaces.Dict({
            'point_cloud': spaces.Box(low=-np.inf, high=np.inf, shape=(1, 1280, 3), dtype=np.float32),
            'agent_pos': spaces.Box(low=-np.inf, high=np.inf, shape=(1, 10), dtype=np.float32), # pos(3) + orient(6) + joint_angle(1): we use 6D representation for orientation
            'gripper_pcd': spaces.Box(low=-np.inf, high=np.inf, shape=(1, 4, 3), dtype=np.float32), # pos(3) + orient(6) + joint_angle(1): we use 6D representation for orientation
            'feature_map': spaces.Box(low=-np.inf, high=np.inf, shape=(1, 128, 128, 3), dtype=np.float32), # pos(3) + orient(6) + joint_angle(1): we use 6D representation for orientation
            'pcd_mask': spaces.Box(low=-np.inf, high=np.inf, shape=(1, 1280, 1), dtype=np.uint8), # pos(3) + orient(6) + joint_angle(1): we use 6D representation for orientation
            # "goal_gripper_pcd": spaces.Box(low=-np.inf, high=np.inf, shape=(1, 4, 3), dtype=np.float32), # pos(3) + orient(6) + joint_angle(1): we use 6D representation for orientation
        })
        if 'goal' in observation_mode:
            self.observation_space['goal_gripper_pcd'] = spaces.Box(low=-np.inf, high=np.inf, shape=(1, 4, 3), dtype=np.float32) # pos(3) + orient(6) + joint_angle(1): we use 6D representation for orientation

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

        self.view_matrices= []
        self.project_matrices = []

        for rpy_mean in self.rpy_mean_list:
            # rpy = np.array(rpy_mean) + np.random.normal(0, 8, 3)
            # camera_center = self.mean_camera_target + np.random.normal(0, 0.05, 3)
            # distance = self.mean_distance + np.random.normal(0, 0.05, 1)
            rpy = np.array(rpy_mean)
            camera_center = self.mean_camera_target
            distance = self.mean_distance

            view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=camera_center, distance=distance, yaw=rpy[2], pitch=rpy[0], roll=rpy[1], upAxisIndex=2, physicsClientId=env.id)
            project_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=640/480 ,nearVal=0.01, farVal=100, physicsClientId=env.id)
            self.view_matrices.append(view_matrix)
            self.project_matrices.append(project_matrix)
            # cprint(f"view_matrix: {view_matrix}, project_matrix: {project_matrix}", 'green')

        self.time_step = 0
        
        if "act3d_goal" in self.observation_mode:
            config_path = self._env.config_path
            task_name = self._env.task_name
            parent_path = os.path.dirname(config_path)
            state_path = os.path.join(parent_path, "grasp_the_door_handle_primitive", "states") # TODO: use the real substep name. 
            stage_lengths_json_file = os.path.join(parent_path, "grasp_the_door_handle_primitive", 'stage_lengths.json')
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
            
            self._env.reset(reset_state=goal_1_state)
            grasping_eef_pc = self.get_gripper_pc()
            
            self._env.reset(reset_state=goal_2_state)
            final_eef_pc = self.get_gripper_pc()
            
            self.grasping_goal = grasping_eef_pc
            self.final_goal = final_eef_pc
            
            self.grasped_handle = False

        self.only_object = only_object

    def reset(self, **kwargs):
        self._env.reset(**kwargs)
        self._env._get_info()
        self.time_step = 0
        if "act3d_goal" in self.observation_mode:
            self.grasped_handle = False
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
        info = self._env._get_info()            
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
    
    def _get_observation(self, render=True, using_torch=False, only_object=True):
        pos, orient = self._env.robot.get_pos_orient(self._env.robot.right_end_effector)

        # get the 6D representation of orientation
        rotate_matrix = p.getMatrixFromQuaternion(orient)
        orient = rotation_transfer_matrix_to_6D(rotate_matrix)

        cur_joint_angle = p.getJointState(self._env.robot.body, self._env.robot.right_gripper_indices[0], physicsClientId=self._env.id)

        if not self.use_joint_angle:
            pos_ori = pos.tolist() + orient.tolist() + [cur_joint_angle[0]]
        else:
            right_arm_indices = self._env.robot.right_arm_joint_indices
            ik_joints = right_arm_indices + list(self._env.robot.right_gripper_indices)
            all_joints = self._env.robot.get_joint_angles(ik_joints) # 9 dim
            ik_lower_limit = self._env.robot.ik_lower_limits[right_arm_indices] 
            ik_upper_limit = self._env.robot.ik_upper_limits[right_arm_indices]
            all_joints[right_arm_indices] = (all_joints[right_arm_indices] - ik_lower_limit) / (ik_upper_limit - ik_lower_limit)
            all_joints[-2:] = (all_joints[-2:]) / 0.04
            pos_ori = all_joints.tolist()
        
        if render:
            beg = time.time()
            rgbs, depths, segmasks, view_camera_matrices, project_camera_matrices = \
                self.take_images_around_object(self._env, self._object_name.lower(), elevation=self.elevation,
                                                return_camera_matrices=True, camera_height=self.camera_height, camera_width=self.camera_width, 
                                                only_object=only_object)
            end = time.time()

            # from matplotlib import pyplot as plt
            # plt.imshow(rgbs[0])
            # plt.show()
            # plt.imshow(rgbs[1])
            # plt.show()
            # import pdb; pdb.set_trace()
            # cprint("point cloud rendering time {}".format(end - beg), "green")
            pcs = []
            feature_maps = []
            gripper_pcd = []
            pcd_mask_indices = []
            for rgb, depth, segmask, view_matrix, project_matrix in zip(rgbs, depths, segmasks, view_camera_matrices, project_camera_matrices):
                
                pc = get_pc(proj_matrix=project_matrix, view_matrix=view_matrix, depth=depth, width=self.camera_width, height=self.camera_height, mask_infinite=False)
                # pc_in_camera = get_pc_in_camera_frame(proj_matrix=project_matrix, view_matrix=view_matrix, depth=depth, width=self.camera_width, height=self.camera_height, mask_infinite=False)
                if self.use_color:
                    rgb = rgb.reshape(-1, 3)
                    colorful_pc = np.concatenate([pc, rgb], axis=1)
                    pcs.append(colorful_pc)
                elif self.use_segmask or self.observation_mode == 'segmask':
                    segmask = segmask.reshape(-1, 1)
                    segmask_obj_id = segmask & ((1 << 24) - 1)
                    robot_mask = np.zeros_like(segmask).astype(np.float32)
                    robot_mask[segmask_obj_id == self._env.urdf_ids['robot']] = 1
                    object_mask = np.zeros_like(segmask).astype(np.float32)
                    # TODO: use the real object name
                    object_mask[segmask_obj_id == self._env.urdf_ids[self._object_name]] = 1
                    pc_with_mask = np.concatenate([pc, robot_mask, object_mask], axis=1)
                    pcs.append(pc_with_mask)
                elif self.only_handle_points or self.observation_mode == 'zoomed-in':
                    segmask = segmask.reshape(-1, 1)
                    segmask_obj_id = segmask & ((1 << 24) - 1)
                    segmask_link_id = (segmask >> 24) - 1
                    robot_mask = (segmask_obj_id == self._env.urdf_ids['robot']).flatten()
                    eef_mask = (segmask_link_id >= 7).flatten()
                    robot_eef_mask = robot_mask & eef_mask
                    robot_pc = pc[robot_eef_mask]
                    object_pc = pc[(segmask_obj_id == self._env.urdf_ids[self._object_name]).flatten()]
                    info = self._env._get_info()
                    handle_pos = np.array(info['handle_pos'])
                    distance = np.linalg.norm(handle_pos.reshape(1, 3) - np.array(object_pc).reshape(-1, 3), axis=1)
                    near_handle_object_pc = object_pc[distance < 0.1]
                    pc = np.concatenate([robot_pc, near_handle_object_pc], axis=0)
                    one_hot_encodings = np.zeros((pc.shape[0], 2))
                    one_hot_encodings[:robot_pc.shape[0], 0] = 1
                    one_hot_encodings[robot_pc.shape[0]:, 1] = 1
                    pc = np.concatenate([pc, one_hot_encodings], axis=1)
                    pcs.append(pc)
                elif 'act3d' in self.observation_mode:
                    # I need point cloud of target object
                    # full segmentation mask + depth image, stacked together
                    # gripper point cloud, which can be the left finger point, right finger point, and the eef point, and the grasping target point
                    # gripper information
                    
                    pcs.append(pc)
                    gripper_pc = self.get_gripper_pc()
                    # p.addUserDebugPoints(list(gripper_pc), [[0, 1, 0] for _ in range(len(gripper_pc))], 50, 0)
                    # import pdb; pdb.set_trace()
                    gripper_pcd.append(gripper_pc)
                    
                    segmask_obj_id = segmask & ((1 << 24) - 1)
                    robot_mask = np.zeros_like(depth).astype(np.float32)
                    robot_mask[segmask_obj_id == self._env.urdf_ids['robot']] = 1
                    object_mask = np.zeros_like(depth).astype(np.float32)
                    object_mask[segmask_obj_id == self._env.urdf_ids[self._object_name]] = 1

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
                        

                    feature_map = np.dstack([robot_mask, object_mask, pc.reshape(self.camera_height, self.camera_width, 3)])
                    assert feature_map.shape == (self.camera_height, self.camera_width, 5), f"Expected ({self.camera_height}, {self.camera_width}, 5), got {feature_map.shape}"
                    feature_maps.append(feature_map)
                    
                    object_mask_indices = np.flatnonzero(object_mask.flatten())
                    pcd_mask_indices.append(object_mask_indices)
                    
                    if 'goal' in self.observation_mode:
                        # print("add goal as part of the observation")
                        # add goal as part of the observation. 
                        # needs to judge when to switch the goal -- check if the handle has been grasped.  
                        
                                
                        if self._env.grasped_handle:
                            print("goal is to open the door")
                            goal_gripper_pcd = self.final_goal
                        else:
                            print("goal is to grasp the handle")
                            goal_gripper_pcd = self.grasping_goal
                            # for point in goal_gripper_pcd:
                                # p.addUserDebugPoints([point], [[1, 0, 0]], 10, 0)
                else:
                    pcs.append(pc)
                
            
            if 'act3d' not in self.observation_mode:
                point_cloud = np.concatenate(pcs, axis=0)
                min_bound = np.array([-5, -5, 0.1])
                max_bound = np.array([5, 5, 5])
                input_pc_mask = np.all(point_cloud[:, :3] > min_bound, axis=1) & np.all(point_cloud[:, :3] < max_bound, axis=1)
                masked_indices = np.flatnonzero(input_pc_mask)
                point_cloud = point_cloud[input_pc_mask]
                
                if self.handle_num_points > 0 or self.gripper_num_points > 0:
                    full_point_cloud = deepcopy(point_cloud)
            else:
                masked_pc = []
                all_masked_indices = []
                base_idx = 0
                for pc, pcd_mask_indices in zip(pcs, pcd_mask_indices):
                    mask_indices = base_idx + pcd_mask_indices
                    all_masked_indices.append(mask_indices)
                    masked_pc.append(pc[pcd_mask_indices])
                    base_idx += pc.shape[0]
                point_cloud = np.concatenate(masked_pc, axis=0)
                all_masked_indices = np.concatenate(all_masked_indices)
                
                # check
                # before_fps_pc = point_cloud.copy()
                # all_pc = np.concatenate(pcs, axis=0)
                # assert np.all(all_pc[all_masked_indices] == point_cloud), "Masked point cloud is not the same as the original point cloud"
           
            # do downsampling of the pcd
            num_points = self.num_points
            if self.gripper_num_points > 0:
                gripper_pc = self._transfer_point_cloud_to_gripper_frame(full_point_cloud)
                bounding_box = np.array([[-self.gripper_bbox, -self.gripper_bbox, -self.gripper_bbox], [self.gripper_bbox, self.gripper_bbox, self.gripper_bbox]])
                mask = np.all(gripper_pc[:, :3] > bounding_box[0], axis=1) & np.all(gripper_pc[:, :3] < bounding_box[1], axis=1)
                pc_within_gripper = full_point_cloud[mask]
                gripper_fps_num_point = min(self.gripper_num_points, pc_within_gripper.shape[0])
                # import pdb; pdb.set_trace()
                if len(pc_within_gripper) > 2:
                    if using_torch:
                        pc_within_gripper = torch.from_numpy(pc_within_gripper).unsqueeze(0).cuda()
                        fps_num_points = torch.tensor([gripper_fps_num_point]).cuda()
                        _, sampled_indices = torch3d_ops.sample_farthest_points(points=pc_within_gripper[...,:3], K=fps_num_points)
                        pc_within_gripper = pc_within_gripper.squeeze(0).cpu().numpy()
                        pc_within_gripper = pc_within_gripper[sampled_indices.squeeze(0).cpu().numpy()]
                    else:
                        kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(pc_within_gripper[:, :3], gripper_fps_num_point, h=1)
                        # kdline_fps_samples_idx = fpsample.bucket_fps_kdtree_sampling(pc, gripper_fps_num_point)
                        pc_within_gripper = pc_within_gripper[kdline_fps_samples_idx]
                    num_points -= gripper_fps_num_point
                
                else:
                    pc_within_gripper = np.array([])
            
            beg = time.time()
            if using_torch:
                point_cloud = torch.from_numpy(point_cloud).unsqueeze(0).cuda()
                num_points = torch.tensor([num_points]).cuda()
                _, sampled_indices = torch3d_ops.sample_farthest_points(points=point_cloud[...,:3], K=num_points)
                point_cloud = point_cloud.squeeze(0).cpu().numpy()
                point_cloud = point_cloud[sampled_indices.squeeze(0).cpu().numpy()]
            else:
                if point_cloud.shape[0] < num_points:
                    to_add_points_num = num_points - point_cloud.shape[0]
                    random_sampled_points = np.random.choice(point_cloud.shape[0], to_add_points_num, replace=True)
                    point_cloud = np.concatenate([point_cloud, point_cloud[random_sampled_points]], axis=0)
                
                h = min(9, np.log2(num_points))
                kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(point_cloud[:, :3], num_points, h=h)
                point_cloud = point_cloud[kdline_fps_samples_idx]

                if 'act3d' not in self.observation_mode:
                    new_input_mask = np.zeros_like(input_pc_mask)
                    new_input_mask[masked_indices[kdline_fps_samples_idx]] = 1
                else:
                    new_input_mask = np.zeros((sum([pc.shape[0] for pc in pcs]),), dtype=np.uint8)
                    new_input_mask[all_masked_indices[kdline_fps_samples_idx]] = 1
                    
                    # check
                    # set_a = set([(x[0], x[1], x[2]) for x in point_cloud])
                    # set_b = set([(x[0], x[1], x[2]) for x in all_pc[new_input_mask == 1]])
                    # assert set_a == set_b, "Masked point cloud is not the same as the original point cloud"
                    
            end = time.time()
                

            point_cloud = point_cloud.tolist()
            
            if self.gripper_num_points > 0:
                point_cloud = point_cloud + pc_within_gripper.tolist()
                
            if self.handle_num_points > 0:
                all_handle_pos, handle_joint_id = get_handle_pos(self._env, self._object_name.lower(), return_median=False)
                masks = []
                for handle_pos in all_handle_pos:
                    # compute the bounding box of this handle
                    min_bound = np.min(handle_pos, axis=0) - 0.003
                    max_bound = np.max(handle_pos, axis=0) + 0.003
                    mask = np.all(full_point_cloud[:, :3] > min_bound, axis=1) & np.all(full_point_cloud[:, :3] < max_bound, axis=1)
                    masks.append(mask)
                mask = np.any(masks, axis=0)
                handle_point_cloud = full_point_cloud[mask]
                handle_point_cloud = torch.from_numpy(handle_point_cloud).unsqueeze(0).cuda()
                num_points = torch.tensor([self.handle_num_points]).cuda()
                _, sampled_indices = torch3d_ops.sample_farthest_points(points=handle_point_cloud[...,:3], K=num_points)
                handle_point_cloud = handle_point_cloud.squeeze(0).cpu().numpy()
                handle_point_cloud = handle_point_cloud[sampled_indices.squeeze(0).cpu().numpy()]
                handle_point_cloud = handle_point_cloud.tolist()
                point_cloud = point_cloud + handle_point_cloud

            # # visualize point cloud
            # if self.time_step >= 50:
            #     pc = np.array(point_cloud)
            #     print("visualizing point cloud with shape: ", pc.shape)
            #     pcd = o3d.geometry.PointCloud()
            #     pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
            #     o3d.visualization.draw_geometries([pcd])
            
            obs_dict_input = {}
            obs_dict_input['point_cloud'] = np.array(point_cloud).astype(np.float32)
            obs_dict_input['agent_pos'] = np.array(pos_ori).astype(np.float32)
            if self.in_gripper_frame:
                obs_dict_input['point_cloud'] = self._transfer_point_cloud_to_gripper_frame(obs_dict_input['point_cloud'])
                
            if self.add_contact:
                points_left_finger = p.getContactPoints(bodyA=self._env.robot.body, linkIndexA=self._env.robot.right_gripper_indices[0], physicsClientId=self._env.id)
                points_right_finger = p.getContactPoints(bodyA=self._env.robot.body, linkIndexA=self._env.robot.right_gripper_indices[1], physicsClientId=self._env.id)
                contact_left = int(len(points_left_finger) > 0)
                contact_right = int(len(points_right_finger) > 0)
                obs_dict_input['agent_pos'] = np.concatenate([obs_dict_input['agent_pos'], [contact_left, contact_right]])
            
            if 'act3d' in self.observation_mode:
                # TODO: handle multiple camera for act3d observation
                obs_dict_input['feature_map'] = np.stack(feature_maps, axis=0).astype(np.float32)
                # import pdb; pdb.set_trace()
                obs_dict_input['gripper_pcd'] = gripper_pcd[0].astype(np.float32)
                obs_dict_input['pcd_mask'] = new_input_mask.astype(np.float32)
                if 'goal' in self.observation_mode:
                    # print("store goal as part of the observation")
                    obs_dict_input['goal_gripper_pcd'] = goal_gripper_pcd
            else:
                obs_dict_input['feature_map'] = np.zeros((1, 1, 1)).astype(np.float32)
                obs_dict_input['gripper_pcd'] = np.zeros((1, 1, 1)).astype(np.float32)
                obs_dict_input['pcd_mask'] = np.zeros((1, 1, 1)).astype(np.uint8)
            
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
        feature_maps = []
        for rgb, depth, segmask, view_matrix, project_matrix in zip(rgbs, depths, segmasks, view_camera_matrices, project_camera_matrices):
            near = 0.01
            far = 100
            depth = far * near / (far - (far - near) * depth)
            depth = depth.reshape(self.camera_height, self.camera_width, 1)
            feature = np.concatenate([rgb, depth], axis=2)
            feature_maps.append(feature)

        ret_dict['feature_map'] = np.array(feature_maps)
        ret_dict['gripper_pcd'] = np.zeros((1, 1, 1)).astype(np.float32)
        ret_dict['pcd_mask'] = np.zeros((1, 1, 1)).astype(np.uint8)
        return ret_dict
            
        
    
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
        return self._env.render()
    
    def take_images_around_object(self, env, object_name, elevation=30, return_camera_matrices=False, camera_height=480, camera_width=640, only_object=True):
        if only_object:
            ### make all other objects invisiable
            prev_rgbas = []
            object_id = env.urdf_ids[object_name]
            for obj_name, obj_id in env.urdf_ids.items():
                if obj_name != object_name and obj_name != 'robot':
                    num_links = p.getNumJoints(obj_id, physicsClientId=env.id)
                    for link_idx in range(-1, num_links):
                        prev_rgba = p.getVisualShapeData(obj_id, link_idx, physicsClientId=env.id)[0][14:18]
                        prev_rgbas.append(prev_rgba)
                        p.changeVisualShape(obj_id, link_idx, rgbaColor=[0, 0, 0, 0], physicsClientId=env.id)

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

            rgbs.append(img)
            depths.append(depth)
            segmasks.append(segmask)
            view_camera_matrices.append(view_matrix)
            project_camera_matrices.append(project_matrix)

        if only_object:
            cnt = 0
            object_id = env.urdf_ids[object_name]
            for obj_name, obj_id in env.urdf_ids.items():
                if obj_name != object_name and obj_name != 'robot':
                    num_links = p.getNumJoints(obj_id, physicsClientId=env.id)
                    for link_idx in range(-1, num_links):
                        p.changeVisualShape(obj_id, link_idx, rgbaColor=prev_rgbas[cnt], physicsClientId=env.id)
                        cnt += 1


        return rgbs, depths, segmasks, view_camera_matrices, project_camera_matrices