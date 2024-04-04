import wandb
import numpy as np
import torch
import tqdm 
from manipulation.utils import build_up_env, save_numpy_as_gif, get_pc, take_round_images_around_object, rotation_transfer_6D_to_matrix, rotation_transfer_matrix_to_6D
from manipulation.gpt_reward_api import get_handle_pos
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

class RobogenPointCloudWrapper:
    def __init__(self, env, object_name, rpy_mean_list=None, seed=None):
        np.random.seed(time.time_ns() % 2**32)
        if seed is not None:
            np.random.seed(seed)

        self._env = env
        self._object_name = object_name

        self.action_low = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        self.action_high = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, dtype=np.float32)
        self.observation_space = spaces.Dict({
            'point_cloud': spaces.Box(low=-np.inf, high=np.inf, shape=(1, 1280, 6), dtype=np.float32),
            'agent_pos': spaces.Box(low=-np.inf, high=np.inf, shape=(1, 10), dtype=np.float32) # pos(3) + orient(6) + joint_angle(1): we use 6D representation for orientation
        })

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


    def reset(self):
        self._env.reset()
        return self._get_observation()
    
    def step(self, action):
        pos, orient = self._env.robot.get_pos_orient(self._env.robot.right_end_effector)
        # orient = p.getEulerFromQuaternion(orient)
        current_rotate_matrix = np.array(p.getMatrixFromQuaternion(orient)).reshape(3, 3)

        delta_orient = action[3:9]

        delta_rotate_matrix = rotation_transfer_6D_to_matrix(delta_orient)

        after_rotate_matrix = current_rotate_matrix @ delta_rotate_matrix
        
        orient = R.from_matrix(after_rotate_matrix).as_quat()
        euler = p.getEulerFromQuaternion(orient)

        cur_joint_angle = p.getJointState(self._env.robot.body, self._env.robot.right_gripper_indices[0], physicsClientId=self._env.id)
        # pos_ori = pos.tolist() + list(orient) + [cur_joint_angle[0]]
        # action = action + np.array(pos_ori)

        pos = pos + np.array(action[:3])
        target_joint_angle = action[9] + cur_joint_angle[0]
        action = pos.tolist() + list(euler) + [target_joint_angle]

        p.addUserDebugPoints([pos], [[0, 1, 0]], 25)

        self._env.take_direct_action(action)
        reward, success = self._env._compute_reward()
        info = {'success': success}
        done = self._env.time_step >= 1000
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self):
        pos, orient = self._env.robot.get_pos_orient(self._env.robot.right_end_effector)
        # orient = p.getEulerFromQuaternion(orient)

        # get the 6D representation of orientation
        rotate_matrix = p.getMatrixFromQuaternion(orient)
        
        orient = rotation_transfer_matrix_to_6D(rotate_matrix)

        cur_joint_angle = p.getJointState(self._env.robot.body, self._env.robot.right_gripper_indices[0], physicsClientId=self._env.id)
        pos_ori = pos.tolist() + orient.tolist() + [cur_joint_angle[0]]
        # rgbs, depths, view_camera_matrices, project_camera_matrices = \
        #     take_round_images_around_object(self._env, self._object_name.lower(), elevation=30,
        #                                     return_camera_matrices=True, camera_height=480, camera_width=640, 
        #                                     only_object=True)
        rgbs, depths, view_camera_matrices, project_camera_matrices = \
            self.take_images_around_object(self._env, self._object_name.lower(), elevation=30,
                                            return_camera_matrices=True, camera_height=480, camera_width=640, 
                                            only_object=True)
        # time_4 = time.time()
        # print("take round images time: ", time_4 - time_3)
        pcs = []
        for rgb, depth, view_matrix, project_matrix in zip(rgbs, depths, view_camera_matrices, project_camera_matrices):
            pc = get_pc(proj_matrix=project_matrix, view_matrix=view_matrix, depth=depth, width=640, height=480, mask_infinite=False)
            rgb = rgb.reshape(-1, 3)
            colorful_pc = np.concatenate([pc, rgb], axis=1)
            pcs.append(colorful_pc)
        point_cloud = np.concatenate(pcs, axis=0)
        min_bound = np.array([-5, -5, 0.1])
        max_bound = np.array([5, 5, 5])
        if min_bound is not None:
            mask = np.all(point_cloud[:, :3] > min_bound, axis=1)
            point_cloud = point_cloud[mask]
        if max_bound is not None:
            mask = np.all(point_cloud[:, :3] < max_bound, axis=1)
            point_cloud = point_cloud[mask]


        # # visualize full point cloud
        # pc = np.array(point_cloud)
        # print("visualizing full point cloud with shape: ", pc.shape)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
        # pcd.colors = o3d.utility.Vector3dVector(pc[:, 3:6]/255.0)
        # o3d.visualization.draw_geometries([pcd])
            
        full_point_cloud = deepcopy(point_cloud)
        sampled_indices = np.random.choice(point_cloud.shape[0], 1024*20, replace=False)
        point_cloud = point_cloud[sampled_indices]
        point_cloud = torch.from_numpy(point_cloud).unsqueeze(0).cuda()
        num_points = torch.tensor([1024]).cuda()
        # remember to only use coord to sample
        _, sampled_indices = torch3d_ops.sample_farthest_points(points=point_cloud[...,:3], K=num_points)
        point_cloud = point_cloud.squeeze(0).cpu().numpy()
        point_cloud = point_cloud[sampled_indices.squeeze(0).cpu().numpy()]
        point_cloud = point_cloud.tolist()
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
        num_points = torch.tensor([256]).cuda()
        _, sampled_indices = torch3d_ops.sample_farthest_points(points=handle_point_cloud[...,:3], K=num_points)
        handle_point_cloud = handle_point_cloud.squeeze(0).cpu().numpy()
        handle_point_cloud = handle_point_cloud[sampled_indices.squeeze(0).cpu().numpy()]
        handle_point_cloud = handle_point_cloud.tolist()
        point_cloud = point_cloud + handle_point_cloud

        # # visualize point cloud
        # pc = np.array(point_cloud)
        # print("visualizing point cloud with shape: ", pc.shape)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
        # pcd.colors = o3d.utility.Vector3dVector(pc[:, 3:6]/255.0)
        # o3d.visualization.draw_geometries([pcd])
        

        obs_dict_input = {}
        obs_dict_input['point_cloud'] = np.array(point_cloud)
        obs_dict_input['agent_pos'] = np.array(pos_ori)

        return obs_dict_input
    
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
        view_camera_matrices = []
        project_camera_matrices = []
        
        for view_matrix, project_matrix in zip(self.view_matrices, self.project_matrices):
            w, h, img, depth, segmask = p.getCameraImage(640, 480, view_matrix, project_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=env.id)
            img = np.reshape(img, (h, w, 4))[:, :, :3]
            depth = np.reshape(depth, (h, w))

            rgbs.append(img)
            depths.append(depth)
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


        return rgbs, depths, view_camera_matrices, project_camera_matrices