"""
This file contains the robosuite environment wrapper that is used
to provide a standardized environment API for training policies and interacting
with metadata present in datasets.
"""
import json
import numpy as np
from copy import deepcopy
import open3d as o3d

import robosuite
from robosuite.utils.camera_utils import get_real_depth_map, get_camera_extrinsic_matrix, get_camera_intrinsic_matrix, get_camera_segmentation
try:
    # this is needed for ensuring robosuite can find the additional mimicgen environments (see https://mimicgen.github.io)
    import mimicgen_envs
except ImportError:
    pass
import itertools
import robomimic.utils.obs_utils as ObsUtils
import robomimic.envs.env_base as EB
# from mimicgen.utils.rotation_transformer import RotationTransformer
from mimicgen.utils.articubot_util import get_4_points_from_gripper_pos_orient, rotation_transfer_matrix_to_6D
from robosuite.utils import transform_utils as T
from scipy.spatial.transform import Rotation as R
# protect against missing mujoco-py module, since robosuite might be using mujoco-py or DM backend
try:
    import mujoco_py
    MUJOCO_EXCEPTIONS = [mujoco_py.builder.MujocoException]
except Exception:
    # not just ImportError: mujoco_py raises a bare Exception when the legacy
    # MuJoCo 2.1 binaries are absent (normal with robosuite >= 1.4).
    # MUJOCO_EXCEPTIONS only feeds rollout_exceptions, so this is inert.
    MUJOCO_EXCEPTIONS = []

def depth2fgpcd(depth, mask, cam_params):
    # depth: (h, w)
    # fgpcd: (n, 3)
    # mask: (h, w)
    h, w = depth.shape
    mask = np.logical_and(mask, depth > 0)
    # mask = (depth <= 0.599/0.8)
    fgpcd = np.zeros((mask.sum(), 3))
    fx, fy, cx, cy = cam_params
    pos_x, pos_y = np.meshgrid(np.arange(w), np.arange(h))
    pos_x = pos_x[mask]
    pos_y = pos_y[mask]
    fgpcd[:, 0] = (pos_x - cx) * depth[mask] / fx
    fgpcd[:, 1] = (pos_y - cy) * depth[mask] / fy
    fgpcd[:, 2] = depth[mask]
    return fgpcd

def np2o3d(pcd, color=None):
    # pcd: (n, 3)
    # color: (n, 3)
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    if color is not None and color.shape[0] > 0:
        assert pcd.shape[0] == color.shape[0]
        assert color.max() <= 1
        assert color.min() >= 0
        pcd_o3d.colors = o3d.utility.Vector3dVector(color)
    return pcd_o3d

class EnvRobosuite(EB.EnvBase):
    """Wrapper class for robosuite environments (https://github.com/ARISE-Initiative/robosuite)"""
    def __init__(
        self, 
        env_name, 
        render=False, 
        render_offscreen=False, 
        use_image_obs=False, 
        postprocess_visual_obs=True,
        data_gen=False,
        save_image_keys=[],
        **kwargs,
    ):
        """
        Args:
            env_name (str): name of environment. Only needs to be provided if making a different
                environment from the one in @env_meta.

            render (bool): if True, environment supports on-screen rendering

            render_offscreen (bool): if True, environment supports off-screen rendering. This
                is forced to be True if @env_meta["use_images"] is True.

            use_image_obs (bool): if True, environment is expected to render rgb image observations
                on every env.step call. Set this to False for efficiency reasons, if image
                observations are not required.

            postprocess_visual_obs (bool): if True, postprocess image observations
                to prepare for learning. This should only be False when extracting observations
                for saving to a dataset (to save space on RGB images for example).
        """
        self.postprocess_visual_obs = postprocess_visual_obs

        # robosuite version check
        self._is_v1 = (robosuite.__version__.split(".")[0] == "1")
        if self._is_v1:
            assert (int(robosuite.__version__.split(".")[1]) >= 2), "only support robosuite v0.3 and v1.2+"

        kwargs = deepcopy(kwargs)

        # update kwargs based on passed arguments
        update_kwargs = dict(
            has_renderer=render,
            has_offscreen_renderer=(render_offscreen or use_image_obs),
            ignore_done=True,
            use_object_obs=True,
            use_camera_obs=use_image_obs,
            camera_depths=True,
        )
        kwargs.update(update_kwargs)

        if self._is_v1:
            if kwargs["has_offscreen_renderer"]:
                # ensure that we select the correct GPU device for rendering by testing for EGL rendering
                # NOTE: this package should be installed from this link (https://github.com/StanfordVL/egl_probe)
                import egl_probe
                valid_gpu_devices = egl_probe.get_available_devices()
                if len(valid_gpu_devices) > 0:
                    kwargs["render_gpu_device_id"] = valid_gpu_devices[0]
        else:
            # make sure gripper visualization is turned off (we almost always want this for learning)
            kwargs["gripper_visualization"] = False
            del kwargs["camera_depths"]
            kwargs["camera_depth"] = False # rename kwarg

        self._env_name = env_name
        self._init_kwargs = deepcopy(kwargs)
        self.env = robosuite.make(self._env_name, **kwargs)

        if self._is_v1:
            # Make sure joint position observations and eef vel observations are active
            for ob_name in self.env.observation_names:
                if ("joint_pos" in ob_name) or ("eef_vel" in ob_name):
                    self.env.modify_observable(observable_name=ob_name, attribute="active", modifier=True)

        voxel_center = np.array([0, 0, 0.7])
        pc_center = np.array([0, 0, 0.7])
        if hasattr(self.env, 'table_offset'):
            voxel_center[:2] = self.env.table_offset[:2]
            pc_center = np.array(self.env.table_offset)
            pc_center[2] = pc_center[2] + 0.02
        self.ws_size = 0.6

        if env_name.startswith('Kitchen_'):
            self.ws_size = 0.7
            pc_center = self.env.table_offset
        elif env_name.startswith('PickPlace_'):
            pc_center = np.array([0, 0, 0.83])
            self.ws_size = 1.1
        elif env_name.startswith('MugCleanup_'):
            self.ws_size = 0.8

        self.voxel_workspace = np.array([
            [voxel_center[0] - self.ws_size/2, voxel_center[0] + self.ws_size/2],
            [voxel_center[1] - self.ws_size/2, voxel_center[1] + self.ws_size/2],
            [voxel_center[2], voxel_center[2] + self.ws_size]
        ])
        self.pc_workspace = np.array([
            [pc_center[0] - self.ws_size/2, pc_center[0] + self.ws_size/2],
            [pc_center[1] - self.ws_size/2, pc_center[1] + self.ws_size/2],
            [pc_center[2], pc_center[2] + self.ws_size]
        ])
        self.data_gen = data_gen
        self.save_image_keys = save_image_keys
        # Hierarchical rollout workers can disable point-cloud reconstruction
        # for intermediate actions and re-enable it at the next policy boundary.
        # Default True preserves all existing callers and dataset generation.
        self._point_cloud_enabled = True
        print(f"save_image_keys in init: {save_image_keys}")
        # import pdb; pdb.set_trace()

        self.body_name_to_id = {
            self.env.sim.model.body_id2name(id): id for id in range(self.env.sim.model.nbody)
        }
        self.body_name_to_geom_ids_map = self._get_body_name_to_geom_ids_map()
        self.scene_pcd_ids = []

        robot_body_names = ['world', 'table',  'mount0_controller_box',
                            'mount0_pedestal_feet', 'mount0_torso', 'mount0_pedestal']
        for body_name, geom_ids in self.body_name_to_geom_ids_map.items():
            if body_name not in robot_body_names and 'robot' not in body_name and 'gripper' not in body_name:
                self.scene_pcd_ids += geom_ids
        self.body_name_to_id = list(itertools.chain.from_iterable(self.body_name_to_id))
        # self.quat_to_rot6d_tf = RotationTransformer(from_rep='quaternion', to_rep='rotation_6d')

        robot = self.env.robots[0]               # SingleArm
        root_body = robot.robot_model.root_body  # name of robot root body
        base_pos_world = self.env.sim.data.get_body_xpos(root_body).copy()           # (3,)
        base_mat_world = self.env.sim.data.get_body_xmat(root_body).reshape(3, 3)    # (3,3)
        T_world_base = T.make_pose(base_pos_world, base_mat_world)  # R|t in world
        T_base_world = T.pose_inv(T_world_base)                     # invert pose
        self.base_world_T_base_robot = T_base_world
        # print(f"T_world_base: {T_world_base}")
        # import pdb; pdb.set_trace()


    def _get_body_name_to_geom_ids_map(self,):
        sim = self.env.sim
        model = sim.model
        body_to_geoms = {}
        for geom_id in range(model.ngeom):
            body_id = model.geom_bodyid[geom_id]
            body_name = model.body_id2name(body_id)
            if body_name not in body_to_geoms:
                body_to_geoms[body_name] = []
            body_to_geoms[body_name].append(geom_id)
        return body_to_geoms

    def step(self, action):
        """
        Step in the environment with an action.

        Args:
            action (np.array): action to take

        Returns:
            observation (dict): new observation dictionary
            reward (float): reward for this step
            done (bool): whether the task is done
            info (dict): extra information
        """
        obs, r, done, info = self.env.step(action)
        obs = self.get_observation(obs)
        return obs, r, self.is_done(), info

    def reset(self):
        """
        Reset environment.

        Returns:
            observation (dict): initial observation dictionary.
        """
        di = self.env.reset()
        return self.get_observation(di)

    def reset_to(self, state):
        """
        Reset to a specific simulator state.

        Args:
            state (dict): current simulator state that contains one or more of:
                - states (np.ndarray): initial state of the mujoco environment
                - model (str): mujoco scene xml
        
        Returns:
            observation (dict): observation dictionary after setting the simulator state (only
                if "states" is in @state)
        """
        should_ret = False
        if "model" in state:
            self.reset()
            robosuite_version_id = int(robosuite.__version__.split(".")[1])
            if robosuite_version_id <= 3:
                from robosuite.utils.mjcf_utils import postprocess_model_xml
                xml = postprocess_model_xml(state["model"])
            else:
                # v1.4 and above use the class-based edit_model_xml function
                xml = self.env.edit_model_xml(state["model"])
            self.env.reset_from_xml_string(xml)
            self.env.sim.reset()
            if not self._is_v1:
                # hide teleop visualization after restoring from model
                self.env.sim.model.site_rgba[self.env.eef_site_id] = np.array([0., 0., 0., 0.])
                self.env.sim.model.site_rgba[self.env.eef_cylinder_id] = np.array([0., 0., 0., 0.])
        if "states" in state:
            self.env.sim.set_state_from_flattened(state["states"])
            self.env.sim.forward()
            should_ret = True

        if "goal" in state:
            self.set_goal(**state["goal"])
        if should_ret:
            # only return obs if we've done a forward call - otherwise the observations will be garbage
            return self.get_observation()
        return None

    def render(self, mode="human", height=None, width=None, camera_name="agentview"):
        """
        Render from simulation to either an on-screen window or off-screen to RGB array.

        Args:
            mode (str): pass "human" for on-screen rendering or "rgb_array" for off-screen rendering
            height (int): height of image to render - only used if mode is "rgb_array"
            width (int): width of image to render - only used if mode is "rgb_array"
            camera_name (str): camera name to use for rendering
        """
        if mode == "human":
            cam_id = self.env.sim.model.camera_name2id(camera_name)
            self.env.viewer.set_camera(cam_id)
            return self.env.render()
        elif mode == "rgb_array":
            return self.env.sim.render(height=height, width=width, camera_name=camera_name)[::-1]
        else:
            raise NotImplementedError("mode={} is not implemented".format(mode))

    def set_point_cloud_enabled(self, enabled=True):
        """Control expensive point-cloud construction for subsequent observations.

        RGB, depth, state and camera geometry remain available while disabled.
        The optimized hierarchical evaluator turns this back on for the final
        action in each action chunk, immediately before its next HL forward.
        """
        self._point_cloud_enabled = bool(enabled)

    def get_observation(self, di=None):
        """
        Get current environment observation dictionary.

        Args:
            di (dict): current raw observation dictionary from robosuite to wrap and provide 
                as a dictionary. If not provided, will be queried from robosuite.
        """
        if di is None:
            di = self.env._get_observations(force_update=True) if self._is_v1 else self.env._get_observation()
        
        ret = {}
        for k in di:
            if (k in ObsUtils.OBS_KEYS_TO_MODALITIES) and ObsUtils.key_is_obs_modality(key=k, obs_modality="rgb"):
                ret[k] = di[k][::-1]
                # import pdb; pdb.set_trace()
                if self.postprocess_visual_obs:
                    ret[k] = ObsUtils.process_obs(obs=ret[k], obs_key=k)
                    # import pdb; pdb.set_trace()

                # from matplotlib import pyplot as plt
                # print(f"k: {k}")
                # plt.imshow(ret[k])
                # plt.show()


            if (k in ObsUtils.OBS_KEYS_TO_MODALITIES) and ObsUtils.key_is_obs_modality(key=k, obs_modality="depth"):
                depth_map = di[k][::-1]
                depth_map = np.clip(depth_map, 0, 1)
                ret[k] = get_real_depth_map(self.env.sim, depth_map)
                if self.postprocess_visual_obs:
                    ret[k] = ObsUtils.process_obs(obs=ret[k], obs_key=k)

        # "object" key contains object information
        ret["object"] = np.array(di["object-state"])

        # Point-cloud construction is one of the most expensive parts of image
        # evaluation (segmentation renders, Open3D fusion, then 4,500-point
        # farthest-point sampling). Image-only policies do not consume it, so
        # only build it when the active shape metadata registered that key.
        point_cloud_requested = (
            ObsUtils.OBS_KEYS_TO_MODALITIES is None
            or "point_cloud" in ObsUtils.OBS_KEYS_TO_MODALITIES
        )
        if self.env.use_camera_obs and point_cloud_requested and self._point_cloud_enabled:
            workspace = self.voxel_workspace

            # voxel_bound = np.array([
            #     [center[0] - ws_size/2, center[1] - ws_size/2, center[2] - 0.05],
            #     [center[0] + ws_size/2, center[1] + ws_size/2, center[2] - 0.05 + ws_size],
            # ])
            # voxel_bound = workspace.T
            # voxel_size = 64

            all_pcds = o3d.geometry.PointCloud()
            from termcolor import cprint
            # cprint(f"camera_names: {self.env.camera_names}", 'magenta')
            for cam_idx, camera_name in enumerate(self.env.camera_names):
                cam_height = self.env.camera_heights[cam_idx]
                cam_width = self.env.camera_widths[cam_idx]
                ext_mat = get_camera_extrinsic_matrix(self.env.sim, camera_name)
                int_mat = get_camera_intrinsic_matrix(self.env.sim, camera_name, cam_height, cam_width)
                depth = di[f'{camera_name}_depth'][::-1]
                depth = np.clip(depth, 0, 1)
                depth = get_real_depth_map(self.env.sim, depth)
                depth = depth[:, :, 0]
                color = di[f'{camera_name}_image'][::-1]
                
                # cprint(f"camera_name: {camera_name}", 'magenta')
                # from matplotlib import pyplot as plt
                # plt.imshow(color)
                # plt.show()
                # depth = ret[f'{camera_name}_depth'][:, :, 0]
                # color = ret[f'{camera_name}_image']
                # if camera_name != 'agentview':
                #     del ret[f'{camera_name}_depth']
                #     del ret[f'{camera_name}_image']
                cam_param = [int_mat[0, 0], int_mat[1, 1], int_mat[0, 2], int_mat[1, 2]]

                segmentations = get_camera_segmentation(self.env.sim, camera_name, cam_height, cam_width)[:, :, -1]
                seg_mask = np.isin(segmentations, self.scene_pcd_ids)
                # mask = np.ones_like(depth, dtype=bool)

                pcd = depth2fgpcd(depth, seg_mask, cam_param)

                # pose = np.linalg.inv(ext_mat)
                pose = ext_mat
                
                trans_pcd = pose @ np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0)
                trans_pcd = trans_pcd[:3, :].T

                mask = (trans_pcd[:, 0] > workspace[0, 0]) * (trans_pcd[:, 0] < workspace[0, 1]) * (trans_pcd[:, 1] > workspace[1, 0]) * (trans_pcd[:, 1] < workspace[1, 1]) * (trans_pcd[:, 2] > workspace[2, 0]) * (trans_pcd[:, 2] < workspace[2, 1])
                # mask = np.ones_like(trans_pcd[:, 0], dtype=bool)        

                pcd_o3d = np2o3d(trans_pcd[mask], color[seg_mask].reshape(-1, 3)[mask].astype(np.float64) / 255)

                all_pcds += pcd_o3d

            # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(all_pcds, voxel_size=self.ws_size/voxel_size+1e-4, min_bound=voxel_bound[0], max_bound=voxel_bound[1])
            # voxels = voxel_grid.get_voxels()  # returns list of voxels
            # if len(voxels) == 0:
            #     np_voxels = np.zeros([4, voxel_size, voxel_size, voxel_size], dtype=np.uint8)
            # else:
            #     indices = np.stack(list(vx.grid_index for vx in voxels))
            #     colors = np.stack(list(vx.color for vx in voxels))

            #     mask = (indices > 0) * (indices < voxel_size)
            #     indices = indices[mask.all(axis=1)]
            #     colors = colors[mask.all(axis=1)]

            #     np_voxels = np.zeros([4, voxel_size, voxel_size, voxel_size], dtype=np.uint8)
            #     np_voxels[0, indices[:, 0], indices[:, 1], indices[:, 2]] = 1
            #     np_voxels[1:, indices[:, 0], indices[:, 1], indices[:, 2]] = colors.T * 255

            # np_voxels = np.moveaxis(np_voxels, [0, 1, 2, 3], [0, 3, 2, 1])
            # np_voxels = np.flip(np_voxels, (1, 2))

            # import matplotlib.pyplot as plt
            # from mpl_toolkits.mplot3d import Axes3D

            # # Create a 3D plot
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            
            # # indices = np.argwhere(np_voxels[0] != 0)
            # # colors = np_voxels[1:, indices[:, 0], indices[:, 1], indices[:, 2]].T

            # ax.scatter(indices[:, 0], indices[:, 1], indices[:, 2], color=colors, marker='s')

            # # Set labels and show the plot
            # ax.set_xlabel('X Axis')
            # ax.set_ylabel('Y Axis')
            # ax.set_zlabel('Z Axis')
            # ax.set_xlim(0, 64)
            # ax.set_ylim(0, 64)
            # ax.set_zlim(0, 64)
            # plt.savefig('test2.png')
            # plt.close()

            # ret['voxels'] = np_voxels

            ### use open3d to visualize the all_pcd
            # o3d.visualization.draw_geometries([all_pcds])

            # bounding_box = o3d.geometry.AxisAlignedBoundingBox(self.pc_workspace.T[0], self.pc_workspace.T[1])
            # cropped_pcd = all_pcds.crop(bounding_box)
            cropped_pcd = all_pcds
            if len(cropped_pcd.points) > 50000:
                ratio = 50000 / len(cropped_pcd.points)
                cropped_pcd = cropped_pcd.uniform_down_sample(int(1 / ratio))
            if len(cropped_pcd.points) == 0:
                # create fake points
                cropped_pcd.points = o3d.utility.Vector3dVector(np.array([[0., 0., 0.]]))
                cropped_pcd.colors = o3d.utility.Vector3dVector(np.array([[0., 0., 0.]]))
            if len(cropped_pcd.points) < 4500:
                # random upsample to 4500
                num_pad = 4500 - len(cropped_pcd.points)
                indices = np.random.choice(len(cropped_pcd.points), num_pad)
                padded_xyz = np.asarray(cropped_pcd.points)[indices]
                padded_color = np.asarray(cropped_pcd.colors)[indices]
                xyz = np.concatenate([np.asarray(cropped_pcd.points), padded_xyz], 0)
                color = np.concatenate([np.asarray(cropped_pcd.colors), padded_color], 0)
                cropped_pcd = o3d.geometry.PointCloud()
                cropped_pcd.points = o3d.utility.Vector3dVector(xyz)
                cropped_pcd.colors = o3d.utility.Vector3dVector(color)
            sampled_pcds = cropped_pcd.farthest_point_down_sample(4500)
            sampled_pcds = sampled_pcds.transform(self.base_world_T_base_robot)
            xyz = np.asarray(sampled_pcds.points)
            color = np.asarray(sampled_pcds.colors)


            

            # # Save the plot
            # plt.savefig('1.png')
            # plt.close()

            # ret['point_cloud'] = np.concatenate([xyz, color], 1)
            ret['point_cloud'] = xyz.astype(np.float32)
            ret['rgb_values'] = color.astype(np.float32)
        if self._is_v1:
            for robot in self.env.robots:
                # add all robot-arm-specific observations. Note the (k not in ret) check
                # ensures that we don't accidentally add robot wrist images a second time
                pf = robot.robot_model.naming_prefix
                for k in di:
                    if k.startswith(pf) and (k not in ret) and \
                            (not k.endswith("proprio-state")):
                        ret[k] = np.array(di[k])
        else:
            # minimal proprioception for older versions of robosuite
            ret["proprio"] = np.array(di["robot-state"])
            ret["eef_pos"] = np.array(di["eef_pos"])
            ret["eef_quat"] = np.array(di["eef_quat"])
            ret["gripper_qpos"] = np.array(di["gripper_qpos"])
        
        gripper_pcd = get_4_points_from_gripper_pos_orient(ret['robot0_eef_pos'],
                                                            ret['robot0_eef_quat'],
                                                            ret['robot0_gripper_qpos'][0])
        gripper_pcd = self.base_world_T_base_robot @ np.concatenate([gripper_pcd.T, np.ones((1, gripper_pcd.shape[0]))], axis=0)
        ret['gripper_pcd'] = (gripper_pcd[:3, :].T).astype(np.float32)
        eef_pos_h = np.concatenate([ret['robot0_eef_pos'], [1.0]])
        ret['robot0_eef_pos'] = (self.base_world_T_base_robot @ eef_pos_h)[:3]
        eef_rot_robot = self.base_world_T_base_robot[:3, :3] @ T.quat2mat(ret['robot0_eef_quat'])
        ret['robot0_eef_quat'] = np.array(R.from_matrix(eef_rot_robot).as_quat()) # xyzw for scipy. pytorch3d accepts wxyz
        # ret['robot0_eef_rot6d'] = self.quat_to_rot6d_tf.forward(np.array(ret['robot0_eef_quat']))
        ret['robot0_eef_rot6d'] = rotation_transfer_matrix_to_6D(eef_rot_robot)

        # test1 = self.quat_to_rot6d_tf.forward(np.array(ret['robot0_eef_quat']))
        # test2 = rotation_transfer_matrix_to_6D(eef_rot_robot)
        # import pdb; pdb.set_trace()
        # assert np.allclose(test1, test2), "test1 and test2 are not close"


        ret['state'] = np.array([*ret['robot0_eef_pos'], *ret['robot0_eef_rot6d'], ret['robot0_gripper_qpos'][0]])

        # per-timestep camera extrinsics (cam-to-robot-base, 4x4) and intrinsics (3x3)
        # pre-multiplied by base_world_T_base_robot so unprojected point clouds are
        # in robot base frame, consistent with point_cloud/gripper_pcd/goal_gripper_pcd
        cam_h = self.env.camera_heights[0]
        cam_w = self.env.camera_widths[0]
        T_bw = self.base_world_T_base_robot
        ret['agentview_extrinsics'] = (T_bw @ get_camera_extrinsic_matrix(self.env.sim, 'agentview')).astype(np.float32)
        ret['agentview_intrinsics'] = get_camera_intrinsic_matrix(self.env.sim, 'agentview', cam_h, cam_w).astype(np.float32)
        ret['robot0_eye_in_hand_extrinsics'] = (T_bw @ get_camera_extrinsic_matrix(self.env.sim, 'robot0_eye_in_hand')).astype(np.float32)
        ret['robot0_eye_in_hand_intrinsics'] = get_camera_intrinsic_matrix(self.env.sim, 'robot0_eye_in_hand', cam_h, cam_w).astype(np.float32)

        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # # Scatter plot
        # ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=color, s=20)
        # ax.scatter(ret['gripper_pcd'][:, 0], ret['gripper_pcd'][:, 1], ret['gripper_pcd'][:, 2], c='blue', s=50)
        # ax.scatter([0],[0],[0], c='red', s=100)
        # ### plot the origin axis at x=0, y=0, z=0
        # # Plot coordinate frame at the origin
        # axis_length = 0.1
        # ax.quiver(0, 0, 0, axis_length, 0, 0, color='r', linewidth=2)
        # ax.quiver(0, 0, 0, 0, axis_length, 0, color='g', linewidth=2, arrow_length_ratio=0.1)
        # ax.quiver(0, 0, 0, 0, 0, axis_length, color='b', linewidth=2, arrow_length_ratio=0.1)
        
        # ax.axis('equal')
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # plt.show()

        if self.data_gen:
            keys_to_remove = [key for key in ret if key.endswith('image') or key.endswith('depth')]
            for key in keys_to_remove:
                # print(f"key: {key}")
                # print(f"key.split('_image')[0]: {key.split('_image')[0]}")
                # print(f"self.save_image_keys: {self.save_image_keys}")
                cam_name = key.split('_image')[0] if '_image' in key else key.split('_depth')[0]
                if cam_name in self.save_image_keys:
                    continue
                del ret[key]
        return ret

    def get_state(self):
        """
        Get current environment simulator state as a dictionary. Should be compatible with @reset_to.
        """
        xml = self.env.sim.model.get_xml() # model xml file
        state = np.array(self.env.sim.get_state().flatten()) # simulator state
        return dict(model=xml, states=state)

    def get_reward(self):
        """
        Get current reward.
        """
        return self.env.reward()

    def get_goal(self):
        """
        Get goal observation. Not all environments support this.
        """
        return self.get_observation(self.env._get_goal())

    def set_goal(self, **kwargs):
        """
        Set goal observation with external specification. Not all environments support this.
        """
        return self.env.set_goal(**kwargs)

    def is_done(self):
        """
        Check if the task is done (not necessarily successful).
        """

        # Robosuite envs always rollout to fixed horizon.
        return False

    def is_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """
        succ = self.env._check_success()
        if isinstance(succ, dict):
            assert "task" in succ
            return succ
        return { "task" : succ }

    @property
    def action_dimension(self):
        """
        Returns dimension of actions (int).
        """
        return self.env.action_spec[0].shape[0]

    @property
    def name(self):
        """
        Returns name of environment name (str).
        """
        return self._env_name

    @property
    def type(self):
        """
        Returns environment type (int) for this kind of environment.
        This helps identify this env class.
        """
        return EB.EnvType.ROBOSUITE_TYPE

    @property
    def version(self):
        """
        Returns version of robosuite used for this environment, eg. 1.2.0
        """
        return robosuite.__version__

    def serialize(self):
        """
        Save all information needed to re-instantiate this environment in a dictionary.
        This is the same as @env_meta - environment metadata stored in hdf5 datasets,
        and used in utils/env_utils.py.
        """
        return dict(
            env_name=self.name,
            env_version=self.version,
            type=self.type,
            env_kwargs=deepcopy(self._init_kwargs)
        )

    @classmethod
    def create_for_data_processing(
        cls, 
        env_name, 
        camera_names, 
        camera_height, 
        camera_width, 
        reward_shaping, 
        save_image_keys=[],
        **kwargs,
    ):
        """
        Create environment for processing datasets, which includes extracting
        observations, labeling dense / sparse rewards, and annotating dones in
        transitions. 

        Args:
            env_name (str): name of environment
            camera_names (list of str): list of camera names that correspond to image observations
            camera_height (int): camera height for all cameras
            camera_width (int): camera width for all cameras
            reward_shaping (bool): if True, use shaped environment rewards, else use sparse task completion rewards
        """
        is_v1 = (robosuite.__version__.split(".")[0] == "1")
        has_camera = (len(camera_names) > 0)

        new_kwargs = {
            "reward_shaping": reward_shaping,
        }

        if has_camera:
            if is_v1:
                new_kwargs["camera_names"] = list(camera_names)
                new_kwargs["camera_heights"] = camera_height
                new_kwargs["camera_widths"] = camera_width
            else:
                assert len(camera_names) == 1
                if has_camera:
                    new_kwargs["camera_name"] = camera_names[0]
                    new_kwargs["camera_height"] = camera_height
                    new_kwargs["camera_width"] = camera_width

        kwargs.update(new_kwargs)

        # also initialize obs utils so it knows which modalities are image modalities
        image_modalities = list(camera_names)
        if is_v1:
            image_modalities = ["{}_image".format(cn) for cn in camera_names]
            depth_modalities = ["{}_depth".format(cn) for cn in camera_names]
        elif has_camera:
            # v0.3 only had support for one image, and it was named "rgb"
            assert len(image_modalities) == 1
            image_modalities = ["rgb"]
        obs_modality_specs = {
            "obs": {
                "low_dim": [], # technically unused, so we don't have to specify all of them
                "rgb": image_modalities,
                "depth": depth_modalities,
            }
        }
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

        # print(f"save_image_keys before creating env: {save_image_keys}")
        # note that @postprocess_visual_obs is False since this env's images will be written to a dataset
        return cls(
            env_name=env_name,
            render=False, 
            render_offscreen=has_camera, 
            use_image_obs=has_camera, 
            postprocess_visual_obs=False,
            save_image_keys=save_image_keys,
            **kwargs,
        )

    @property
    def rollout_exceptions(self):
        """
        Return tuple of exceptions to except when doing rollouts. This is useful to ensure
        that the entire training run doesn't crash because of a bad policy that causes unstable
        simulation computations.
        """
        return tuple(MUJOCO_EXCEPTIONS)

    def __repr__(self):
        """
        Pretty-print env description.
        """
        return self.name + "\n" + json.dumps(self._init_kwargs, sort_keys=True, indent=4)
