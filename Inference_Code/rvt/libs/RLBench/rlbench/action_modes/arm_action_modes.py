from abc import abstractmethod

import numpy as np
from pyquaternion import Quaternion
from pyrep.const import ConfigurationPathAlgorithms as Algos
from pyrep.errors import ConfigurationPathError, IKError
from pyrep.const import ObjectType
import torch
import pickle
from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.robot import Robot
from rlbench.backend.scene import Scene
from rlbench.const import SUPPORTED_ROBOTS
from diffusion_policy.diffusion_policy.workspace.infer_diffusion_transformer_hybrid_workspace_CLASS import DiffusionHybridInference
import torch.nn.functional as F
import os
import clip

def _clip_encode_text(clip_model, text):
    # from rvt.mvt.utils import ForkedPdb;
    # ForkedPdb().set_trace()
    x = clip_model.token_embedding(text).type(
        clip_model.dtype
    )  # [batch_size, n_ctx, d_model]

    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)

    emb = x.clone()
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection

    return x, emb

def assert_action_shape(action: np.ndarray, expected_shape: tuple):
    if np.shape(action) != expected_shape:
        raise InvalidActionError(
            'Expected the action shape to be: %s, but was shape: %s' % (
                str(expected_shape), str(np.shape(action))))


def assert_unit_quaternion(quat):
    if not np.isclose(np.linalg.norm(quat), 1.0):
        raise InvalidActionError('Action contained non unit quaternion!')


def calculate_delta_pose(robot: Robot, action: np.ndarray):
    a_x, a_y, a_z, a_qx, a_qy, a_qz, a_qw = action
    x, y, z, qx, qy, qz, qw = robot.arm.get_tip().get_pose()
    new_rot = Quaternion(
        a_qw, a_qx, a_qy, a_qz) * Quaternion(qw, qx, qy, qz)
    qw, qx, qy, qz = list(new_rot)
    pose = [a_x + x, a_y + y, a_z + z] + [qx, qy, qz, qw]
    return pose


class ArmActionMode(object):

    @abstractmethod
    def action(self, scene: Scene, action: np.ndarray):
        pass

    @abstractmethod
    def action_shape(self, scene: Scene):
        pass

    def set_control_mode(self, robot: Robot):
        robot.arm.set_control_loop_enabled(True)


class JointVelocity(ArmActionMode):
    """Control the joint velocities of the arm.

    Similar to the action space in many continious control OpenAI Gym envs.
    """
    
    def action(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, self.action_shape(scene))
        scene.robot.arm.set_joint_target_velocities(action)
        scene.step()
        scene.robot.arm.set_joint_target_velocities(np.zeros_like(action))

    def action_shape(self, scene: Scene) -> tuple:
        return SUPPORTED_ROBOTS[scene.robot_setup][2],

    def set_control_mode(self, robot: Robot):
        robot.arm.set_control_loop_enabled(False)
        robot.arm.set_motor_locked_at_zero_velocity(True)


class JointPosition(ArmActionMode):
    """Control the target joint positions (absolute or delta) of the arm.

    The action mode opoerates in absolute mode or delta mode, where delta
    mode takes the current joint positions and adds the new joint positions
    to get a set of target joint positions. The robot uses a simple control
    loop to execute until the desired poses have been reached.
    It os the users responsibility to ensure that the action lies within
    a usuable range.
    """

    def __init__(self, absolute_mode: bool = True):
        """
        Args:
            absolute_mode: If we should opperate in 'absolute', or 'delta' mode.
        """
        self._absolute_mode = absolute_mode

    def action(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, self.action_shape(scene))
        a = action if self._absolute_mode else np.array(
            scene.robot.arm.get_joint_positions()) + action
        scene.robot.arm.set_joint_target_positions(a)
        scene.step()
        scene.robot.arm.set_joint_target_positions(
            scene.robot.arm.get_joint_positions())

    def action_shape(self, scene: Scene) -> tuple:
        return SUPPORTED_ROBOTS[scene.robot_setup][2],


class JointTorque(ArmActionMode):
    """Control the joint torques of the arm.
    """

    TORQUE_MAX_VEL = 9999

    def _torque_action(self, robot, action):
        tml = JointTorque.TORQUE_MAX_VEL
        robot.arm.set_joint_target_velocities(
            [(tml if t < 0 else -tml) for t in action])
        robot.arm.set_joint_forces(np.abs(action))

    def action(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, self.action_shape(scene))
        self._torque_action(scene.robot, action)
        scene.step()
        self._torque_action(scene.robot, scene.robot.arm.get_joint_forces())
        scene.robot.arm.set_joint_target_velocities(np.zeros_like(action))

    def action_shape(self, scene: Scene) -> tuple:
        return SUPPORTED_ROBOTS[scene.robot_setup][2],

    def set_control_mode(self, robot: Robot):
        robot.arm.set_control_loop_enabled(False)

class EndEffectorPoseViaLowLevelPolicy(ArmActionMode):
    def __init__(self, config_path, ):
        super().__init__()
        self.low_level_policy = DiffusionHybridInference(config_path)
        self.early_fusion_train = self.low_level_policy.cfg.early_fusion_train
        self.original_train = self.low_level_policy.cfg.original_train
        self.low_level_lang_cond = self.low_level_policy.cfg.low_level_lang_cond
        if self.low_level_lang_cond:
            self.clip_model, _ = clip.load("RN50", device="cuda:0")
            self.clip_model.eval()
        import pdb; pdb.set_trace();
        self.i = 0
        self._callable_each_step = None
        # import pdb; pdb.set_trace();

    def set_callable_each_step(self, callable_each_step):
        self._callable_each_step = callable_each_step
    
    def action(self, scene: Scene, action: np.ndarray, ignore_collisions: bool = True, obs_dict = None, gripper_action_mode = None):
        # import pdb; pdb.set_trace();
        # rgb_stack = np.stack(obs_dict["rgb_low_level"], axis=0)
        # # Reshape into (12, 3, 224, 224)
        # rgb_low_level = rgb_stack.reshape(-1, 3, 224, 224)
        rgb_list = obs_dict["rgb_low_level"]      # 4 × (3,3,224,224)
        stacked = []
        for t in range(len(rgb_list)):
            # shape: (3 cameras, 3 channels, H, W)
            rgb_t = rgb_list[t]
            # concat cameras along channel axis → (9, 224, 224)
            rgb_concat = np.concatenate(rgb_t, axis=0)
            stacked.append(rgb_concat)
        # final shape: (4, 9, 224, 224)
        rgb_low_level = np.stack(stacked, axis=0)
        depth_low_level  = np.stack(obs_dict["depth_low_level"], axis=0)
        heatmap_low_level = np.stack(obs_dict["heatmap_low_level"], axis=0)
        
        wrist_img = np.array(obs_dict["wrist_rgb"])
        wrist_img_float = torch.from_numpy(wrist_img).float() / 255.0 
        wrist_img_float_224 = F.interpolate(wrist_img_float, size=(224, 224), mode="bilinear", align_corners=False)
        wrist_img_float_224 = wrist_img_float_224.cpu().numpy()
        wrist_depth = np.array(obs_dict["wrist_depth"])
        wrist_depth_float = torch.from_numpy(wrist_depth).float() / 255.0 
        wrist_depth_float_224 = F.interpolate(wrist_depth_float, size=(224, 224), mode="bilinear", align_corners=False)
        wrist_depth_float_224 = wrist_depth_float_224.cpu().numpy()
        if self.early_fusion_train:
            import pdb; pdb.set_trace();
            # Match training format: per-camera [RGB(3) | heatmap(1) | depth(1)]
            # rgb_low_level: (T, 9, H, W) — cameras concatenated along channel axis
            # heatmap_low_level: (T, 3, H, W) — one channel per camera
            # depth_low_level:  (T, 3, H, W) — one channel per camera
            rgbd_heatmap_low_level_cam1 = np.concatenate(
                [rgb_low_level[:, 0:3, :, :], heatmap_low_level[:, 0:1, :, :], depth_low_level.squeeze()[:, 0:1, :, :]],
                axis=1  # (T, 5, H, W)
            )
            rgbd_heatmap_low_level_cam2 = np.concatenate(
                [rgb_low_level[:, 3:6, :, :], heatmap_low_level[:, 1:2, :, :], depth_low_level.squeeze()[:, 1:2, :, :]],
                axis=1  # (T, 5, H, W)
            )
            rgbd_heatmap_low_level_cam3 = np.concatenate(
                [rgb_low_level[:, 6:9, :, :], heatmap_low_level[:, 2:3, :, :], depth_low_level.squeeze()[:, 2:3, :, :]],
                axis=1  # (T, 5, H, W)
            )
            rgbd_heatmap_low_level_cam4 = np.concatenate(
                [wrist_img_float_224, wrist_depth_float_224],
                axis=1 
            ) # (T, 4, H, W)
        else:
            rgbd_heatmap_low_level = np.concatenate(
                [rgb_low_level, wrist_img_float_224, depth_low_level.squeeze(), wrist_depth_float_224 , heatmap_low_level],
                axis=1  
            )
        # rgbd_heatmap_low_level = np.concatenate(
        #     [rgb_low_level, depth_low_level.squeeze(), heatmap_low_level],
        #     axis=1  
        # )
        agent_pos_low_level = np.stack(obs_dict["gripper_pose_low_level"], axis=0)
        agent_open_low_level = np.stack(obs_dict["gripper_open_low_level"], axis=0)
        agent_pos = np.concatenate([agent_pos_low_level, agent_open_low_level], axis=1)

        # CLIP language encoding — matches training: obs_lang_emb = _clip_encode_text(clip_model, token_tensor)[1][0]
        if self.low_level_lang_cond:
            import pdb; pdb.set_trace();
            token_tensor = obs_dict["lang_goal_tokens"][0].long().to("cuda:0")  # same as rvt_agent.py:778
            with torch.no_grad():
                lang_feats, _ = _clip_encode_text(self.clip_model, token_tensor)
            lang_emb_obs = lang_feats[0].float().detach().cpu().numpy()  # [512]
            # T_obs = agent_pos.shape[0]
            # lang_emb_obs = np.stack([lang_emb] * T_obs, axis=0)    # [T_obs, 512]
            import pdb; pdb.set_trace();
        import pdb; pdb.set_trace();
        policy_dict = {}
        if self.early_fusion_train:
            policy_dict["obs"] = {}
            policy_dict["obs"]["image_cam1"] = torch.tensor(np.expand_dims(rgbd_heatmap_low_level_cam1, axis=0)).float()
            policy_dict["obs"]["image_cam2"] = torch.tensor(np.expand_dims(rgbd_heatmap_low_level_cam2, axis=0)).float()
            policy_dict["obs"]["image_cam3"] = torch.tensor(np.expand_dims(rgbd_heatmap_low_level_cam3, axis=0)).float()
            policy_dict["obs"]["image_cam4"] = torch.tensor(np.expand_dims(rgbd_heatmap_low_level_cam4, axis=0)).float()
            policy_dict["obs"]["agent_pos"] = torch.tensor(np.expand_dims(agent_pos, axis=0)).float()
            if self.low_level_lang_cond:
                import pdb; pdb.set_trace();
                policy_dict["obs_lang_emb"] = torch.tensor(np.expand_dims(lang_emb_obs, axis=0)).float()  # [1, T_obs, 77, 512] ["obs"]
        elif self.original_train:
            policy_dict["obs"] = {}
            policy_dict["obs"]["image"] = torch.tensor(np.expand_dims(rgbd_heatmap_low_level, axis=0)).float()
            policy_dict["obs"]["agent_pos"] = torch.tensor(np.expand_dims(agent_pos, axis=0)).float()
        # import pdb; pdb.set_trace();



        # ######## TEMPORARY #######################
        # temp_dict = policy_dict
        # x = policy_dict["obs"]["image"]  # shape: [1, 4, 19, 224, 224]

        # # Reshape to merge batch, modalities, and time so interpolate can process
        # B, M, T, H, W = x.shape
        # x_reshaped = x.view(B * M * T, 1, H, W)  # add channel dim for interpolate

        # # Resize
        # x_resized = F.interpolate(x_reshaped, size=(128, 128), mode='bilinear', align_corners=False)

        # # Restore original dimensions
        # x_resized = x_resized.view(B, M, T, 128, 128)
        # temp_dict["obs"]["image"] = x_resized

        import pdb; pdb.set_trace();
        ########################## ACTUAL LOW LEVEL POLICY CALLED ##########################################
        predicted_poses = self.low_level_policy.predict(policy_dict) # policy_dict
        movement_poses = predicted_poses["action"][0]
        # if self.i == 15:
        #     movement_poses[:, 2] -= 0.08
        # import pdb; pdb.set_trace();
        data_dict = {}
        data_dict["present_gripper"] = obs_dict["reverse_trans_low_level"][0](torch.tensor(np.array(obs_dict["gripper_pose_low_level"])[:,:3]).to("cuda:0")).detach().cpu().numpy()[-2:, :]
        data_dict["gripper"] = obs_dict["reverse_trans_low_level"][0](torch.tensor(movement_poses[:,:3]).to("cuda:0")).detach().cpu().numpy()
        data_dict["pointcloud"] = obs_dict["reverse_trans_low_level"][0](torch.tensor(obs_dict["pcd_low_level"][-1]).to("cuda:0")).detach().cpu().numpy()
        os.makedirs("PRETRAINED_RVT_ALL_DATA_DIFFUSION_MODEL", exist_ok=True)
        pickle.dump(data_dict, open(f'PRETRAINED_RVT_ALL_DATA_DIFFUSION_MODEL/pointcloud_and_pose_SIMULATOR_AFTER_REVERSE_TRANSFORM{self.i}.pkl', 'wb'))
        self.i += 1
        # import pdb; pdb.set_trace();
        for next_pose in movement_poses: #predicted_poses:
            # import pdb; pdb.set_trace();
            pos = obs_dict["reverse_trans_low_level"][0](next_pose[:3]).detach().cpu().numpy()
            quat = next_pose[3:7].detach().cpu().numpy()
            joint_positions = scene.robot.arm.solve_ik(position = pos, quaternion = quat)
            scene.robot.arm.set_joint_target_positions(joint_positions)
            done =  False
            prev_values = None
            while not done:
                scene.step()
                cur_positions = scene.robot.arm.get_joint_positions()
                # print("HEREEEEE", np.allclose(cur_positions, joint_positions, atol=0.01))
                reached = np.allclose(cur_positions, joint_positions, atol=0.01)
                not_moving = False
                if prev_values is not None:
                    not_moving = np.allclose(
                        cur_positions, prev_values, atol=0.001)
                    # print("NOTTTTT MOVING")
                prev_values = cur_positions
                done = reached or not_moving
            # import pdb; pdb.set_trace();
            # if self.i >= 16:
            #     gripper_action_mode.action(scene, torch.tensor([0]).to("cuda:0")) # next_pose[7:]
            # else:
            #     gripper_action_mode.action(scene, torch.tensor([1]).to("cuda:0"))
                # import pdb; pdb.set_trace()
            # if self.i >= 19:
            #     import pdb; pdb.set_trace();
            # if next_pose[7:] >= 0.20:
            #     gripper_action_mode.action(scene, torch.tensor([1]).to("cuda:0"))
            # else:
            #     gripper_action_mode.action(scene, torch.tensor([0]).to("cuda:0"))
            gripper_action_mode.action(scene, next_pose[7:])
            
            # scene.robot.arm.set_tip_target(position=pos, quaternion=quat)
            # scene.step()
            # import pdb; pdb.set_trace();
            if self._callable_each_step is not None:
                self._callable_each_step(scene.get_observation())
            success, terminate = scene.task.success()
            # If the task succeeds while traversing path, then break early
            if success and self._callable_each_step is None:
                break
        # import pdb; pdb.set_trace();
    def action_shape(self, scene: Scene) -> tuple:
        return 7,

    def record_end(self, scene, *args, **kwargs):
        # Nothing needed
        pass

    


class EndEffectorPoseViaPlanning(ArmActionMode):
    """High-level action where target pose is given and reached via planning.

    Given a target pose, a linear path is first planned (via IK). If that fails,
    sample-based planning will be used. The decision to apply collision
    checking is a crucial trade off! With collision checking enabled, you
    are guaranteed collision free paths, but this may not be applicable for task
    that do require some collision. E.g. using this mode on pushing object will
    mean that the generated path will actively avoid not pushing the object.

    Note that path planning can be slow, often taking a few seconds in the worst
    case.

    This was the action mode used in:
    James, Stephen, and Andrew J. Davison. "Q-attention: Enabling Efficient
    Learning for Vision-based Robotic Manipulation."
    arXiv preprint arXiv:2105.14829 (2021).
    """

    def __init__(self,
                 absolute_mode: bool = True,
                 frame: str = 'world',
                 collision_checking: bool = False):
        """
        If collision check is enbled, and an object is grasped, then we

        Args:
            absolute_mode: If we should opperate in 'absolute', or 'delta' mode.
            frame: Either 'world' or 'end effector'.
            collision_checking: IF collision checking is enabled.
        """
        self._absolute_mode = absolute_mode
        self._frame = frame
        self._collision_checking = collision_checking
        self._callable_each_step = None
        self._robot_shapes = None

        if frame not in ['world', 'end effector']:
            raise ValueError("Expected frame to one of: 'world, 'end effector'")

    def _quick_boundary_check(self, scene: Scene, action: np.ndarray):
        pos_to_check = action[:3]
        relative_to = None if self._frame == 'world' else scene.robot.arm.get_tip()
        if relative_to is not None:
            scene.target_workspace_check.set_position(pos_to_check, relative_to)
            pos_to_check = scene.target_workspace_check.get_position()
        if not scene.check_target_in_workspace(pos_to_check):
            raise InvalidActionError('A path could not be found because the '
                                     'target is outside of workspace.')

    def _pose_in_end_effector_frame(self, robot: Robot, action: np.ndarray):
        a_x, a_y, a_z, a_qx, a_qy, a_qz, a_qw = action
        x, y, z, qx, qy, qz, qw = robot.arm.get_tip().get_pose()
        new_rot = Quaternion(
            a_qw, a_qx, a_qy, a_qz) * Quaternion(qw, qx, qy, qz)
        qw, qx, qy, qz = list(new_rot)
        pose = [a_x + x, a_y + y, a_z + z] + [qx, qy, qz, qw]
        return pose

    def set_callable_each_step(self, callable_each_step):
        self._callable_each_step = callable_each_step

    def action(self, scene: Scene, action: np.ndarray, ignore_collisions: bool = True):
        import pdb; pdb.set_trace();
        assert_action_shape(action, (7,))
        assert_unit_quaternion(action[3:])
        if not self._absolute_mode and self._frame != 'end effector':
            action = calculate_delta_pose(scene.robot, action)
        relative_to = None if self._frame == 'world' else scene.robot.arm.get_tip()
        self._quick_boundary_check(scene, action)

        colliding_shapes = []
        if not ignore_collisions:
            if self._robot_shapes is None:
                self._robot_shapes = scene.robot.arm.get_objects_in_tree(
                    object_type=ObjectType.SHAPE)
            # First check if we are colliding with anything
            colliding = scene.robot.arm.check_arm_collision()
            if colliding:
                # Disable collisions with the objects that we are colliding with
                grasped_objects = scene.robot.gripper.get_grasped_objects()
                colliding_shapes = [
                    s for s in scene.pyrep.get_objects_in_tree(
                        object_type = ObjectType.SHAPE) if (
                            s.is_collidable() and
                            s not in self._robot_shapes and
                            s not in grasped_objects and
                            scene.robot.arm.check_arm_collision(
                                s))]
                [s.set_collidable(False) for s in colliding_shapes]

        try:
            # try once with collision checking (if ignore_collisions is true)
            try:
                path = scene.robot.arm.get_path(
                    action[:3],
                    quaternion=action[3:],
                    ignore_collisions=ignore_collisions,
                    relative_to=relative_to,
                    trials=100,
                    max_configs=10,
                    max_time_ms=10,
                    trials_per_goal=5,
                    algorithm=Algos.RRTConnect
                )
            except ConfigurationPathError as e:
                if ignore_collisions:
                    raise InvalidActionError(
                        'A path could not be found. Most likely due to the target '
                        'being inaccessible or a collison was detected.') from e
                else:
                    # try once more with collision checking disabled
                    path = scene.robot.arm.get_path(
                        action[:3],
                        quaternion=action[3:],
                        ignore_collisions=True,
                        relative_to=relative_to,
                        trials=100,
                        max_configs=10,
                        max_time_ms=10,
                        trials_per_goal=5,
                        algorithm=Algos.RRTConnect
                    )
        except ConfigurationPathError as e:
            raise InvalidActionError(
                'A path could not be found. Most likely due to the target '
                'being inaccessible or a collison was detected.') from e
        done = False
        while not done:
            import pdb; pdb.set_trace();
            done = path.step()
            scene.step()
            if self._callable_each_step is not None:
                # Record observations
                self._callable_each_step(scene.get_observation())
            success, terminate = scene.task.success()
            # If the task succeeds while traversing path, then break early
            if success and self._callable_each_step is None:
                break

    def action_shape(self, scene: Scene) -> tuple:
        return 7,

    def record_end(self, scene, steps=60, step_scene=True):
        if self._callable_each_step is not None:
            for _ in range(steps):
                if step_scene:
                    scene.step()
                self._callable_each_step(scene.get_observation())

class EndEffectorPoseViaIK(ArmActionMode):
    """High-level action where target pose is given and reached via IK.

    Given a target pose, IK via inverse Jacobian is performed. This requires
    the target pose to be close to the current pose, otherwise the action
    will fail. It is up to the user to constrain the action to
    meaningful values.

    The decision to apply collision checking is a crucial trade off!
    With collision checking enabled, you are guaranteed collision free paths,
    but this may not be applicable for task that do require some collision.
    E.g. using this mode on pushing object will mean that the generated
    path will actively avoid not pushing the object.
    """

    def __init__(self,
                 absolute_mode: bool = True,
                 frame: str = 'world',
                 collision_checking: bool = False):
        """
        Args:
            absolute_mode: If we should opperate in 'absolute', or 'delta' mode.
            frame: Either 'world' or 'end effector'.
            collision_checking: IF collision checking is enabled.
        """
        import pdb; pdb.set_trace();
        self._absolute_mode = absolute_mode
        self._frame = frame
        self._collision_checking = collision_checking
        if frame not in ['world', 'end effector']:
            raise ValueError(
                "Expected frame to one of: 'world, 'end effector'")

    def action(self, scene: Scene, action: np.ndarray):
        import pdb; pdb.set_trace();
        assert_action_shape(action, (7,))
        assert_unit_quaternion(action[3:])
        if not self._absolute_mode and self._frame != 'end effector':
            action = calculate_delta_pose(scene.robot, action)
        relative_to = None if self._frame == 'world' else scene.robot.arm.get_tip()

        try:
            joint_positions = scene.robot.arm.solve_ik_via_jacobian(
                action[:3], quaternion=action[3:], relative_to=relative_to)
            scene.robot.arm.set_joint_target_positions(joint_positions)
        except IKError as e:
            raise InvalidActionError(
                'Could not perform IK via Jacobian; most likely due to current '
                'end-effector pose being too far from the given target pose. '
                'Try limiting/bounding your action space.') from e
        done = False
        prev_values = None
        # Move until reached target joint positions or until we stop moving
        # (e.g. when we collide wth something)
        while not done:
            scene.step()
            cur_positions = scene.robot.arm.get_joint_positions()
            reached = np.allclose(cur_positions, joint_positions, atol=0.01)
            not_moving = False
            if prev_values is not None:
                not_moving = np.allclose(
                    cur_positions, prev_values, atol=0.001)
            prev_values = cur_positions
            done = reached or not_moving

    def action_shape(self, scene: Scene) -> tuple:
        return 7,
