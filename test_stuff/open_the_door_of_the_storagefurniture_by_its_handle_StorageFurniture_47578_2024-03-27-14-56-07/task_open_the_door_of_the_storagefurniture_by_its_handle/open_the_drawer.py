
from manipulation.sim import SimpleEnv
import numpy as np
from manipulation.gpt_reward_api import *
from manipulation.gpt_primitive_api import *
import gym
from gym import spaces
import pybullet as p

class open_the_drawer(SimpleEnv):

    def __init__(self, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.detected_position = dict()
        num_obs = 11
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_obs, ), dtype=np.float32)

        ## reward function
    def _compute_reward(self):
        # this reward encourages the end-effector to stay near the handle of the drawer to grasp it.
        left_finger_pos = np.array(p.getLinkState(self.robot.body, self.robot.right_gripper_indices[0], physicsClientId=self.id)[0])
        right_finger_pos = np.array(p.getLinkState(self.robot.body, self.robot.right_gripper_indices[1], physicsClientId=self.id)[0])
        handle_pos_list, _ = get_handle_pos(self, 'StorageFurniture', return_median=True)
        # get the handle that is closest to target link: link_0
        drawer_pos = get_link_state(self, "StorageFurniture", "link_0")
        handle_index = np.argmin(np.linalg.norm(handle_pos_list - drawer_pos, axis=1))
        handle_pos = handle_pos_list[handle_index]
        reward_near = - np.linalg.norm(left_finger_pos - handle_pos) - np.linalg.norm(right_finger_pos - handle_pos)
        
        # Get the joint state of the drawer. We know from the semantics and the articulation tree that joint_0 connects link_0 and is the joint that controls the sliding of the drawer.
        joint_angle = get_joint_state(self, "StorageFurniture", "joint_0") 
        joint_limit_low, joint_limit_high = get_joint_limit(self, "StorageFurniture", "joint_0")
        # The reward measures how much the drawer is opened. The more the drawer is opened, the higher the reward.
        reward_joint =  joint_angle
        reward = reward_near + 5 * reward_joint
        
        # The success condition is whether the drawer is opened enough.
        diff = np.abs(joint_angle - joint_limit_high)
        success = diff < 0.35 * (joint_limit_high - joint_limit_low) # for opening, we think 65 percent is enough
        
        return reward, success
        
        ## observation function
    def _get_obs(self):
        # initialize the observation
        obs = np.zeros(self.observation_space.shape[0])
        cnt = 0 # the final result of cnt should be the same as observation dimension
        # get the position of the handle
        handle_pos_list, _ = get_handle_pos(self, 'StorageFurniture', return_median=True)
        # get the handle that is closest to the drawer
        drawer_pos = get_link_state(self, "StorageFurniture", "link_0")
        handle_index = np.argmin(np.linalg.norm(handle_pos_list - drawer_pos, axis=1))
        handle_pos = handle_pos_list[handle_index]
        obs[cnt:cnt+3] = handle_pos
        cnt += 3
        # get the joint angle of the drawer
        joint_angle = get_joint_state(self, "StorageFurniture", "joint_0")
        obs[cnt] = joint_angle
        cnt += 1
        # get the position and orientation of the robot end-effector
        robot_eef_pos, robot_eef_orient = self.robot.get_pos_orient(self.robot.right_end_effector)
        robot_eef_euler_angle = p.getEulerFromQuaternion(robot_eef_orient)
        obs[cnt:cnt+3] = robot_eef_pos
        obs[cnt+3:cnt+6] = robot_eef_euler_angle
        cnt += 6
        # get joint angle of the gripper
        gripper_joint_angle = p.getJointState(self.robot.body, self.robot.right_gripper_indices[0], physicsClientId=self.id)[0]
        obs[cnt] = gripper_joint_angle
        cnt += 1
        return obs

gym.register(
    id='gym-open_the_drawer-v0',
    entry_point=open_the_drawer,
)
