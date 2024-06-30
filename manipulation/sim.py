import numpy as np
import pybullet as p
import gym
from gym.utils import seeding
from gym import spaces
import pickle
import yaml
import os.path as osp
from collections import defaultdict
from scipy.spatial.transform import Rotation as R
from manipulation.panda import Panda
from manipulation.ur5 import UR5
from manipulation.sawyer import Sawyer
from manipulation.utils import parse_config, load_env, download_and_parse_objavarse_obj_from_yaml_config, save_env
from manipulation.gpt_reward_api import get_joint_id_from_name, get_link_id_from_name, get_handle_pos, get_link_pc
from manipulation.gpt_primitive_api import get_link_handle
import matplotlib.pyplot as plt
import open3d 
from termcolor import cprint
import time
import scipy
import os
from manipulation.gpt_primitive_api import get_pc_num_within_gripper

class SimpleEnv(gym.Env):
    def __init__(self, 
                    dt=1/240, 
                    config_path=None, 
                    gui=False, 
                    control_step=2, 
                    # control_step=3, 
                    horizon=250, 
                    restore_state_file=None, 
                    rotation_mode='delta-axis-angle-local',
                    translation_mode='delta-translation', 
                    max_rotation=np.deg2rad(5), 
                    max_translation=0.0075,
                    use_suction=False,  # whether to use a suction gripper
                    object_candidate_num=6, # how many candidate objects to sample from objaverse
                    vhacd=True, # if to perform vhacd on the object for better collision detection for pybullet
                    randomize=0, # if to randomize the scene
                    obj_id=0, # which object to choose to use from the candidates
                    mobile=False,
                    task_name=None,
                    open_gripper_at_reset=True,
                ):
        
        super().__init__()
        
        # Task
        self.config_path = config_path
        self.task_name = task_name
        self.restore_state_file = restore_state_file
        self.control_step = control_step
        self.horizon = horizon
        self._max_episode_steps = horizon
        self.gui = gui
        self.object_candidate_num = object_candidate_num
        self.solution_path = None        
        self.success = False # not really used, keeped for now
        self.primitive_save_path = None # to be used for saving the primitives execution results
        self.randomize = randomize
        self.obj_id = obj_id # which object to choose to use from the candidates
        self.open_gripper_at_reset = open_gripper_at_reset
        
        # robot
        self.mobile = mobile

        # physics
        self.gravity = -9.81
        self.contact_constraint = None
        self.vhacd = vhacd
        
        # action space
        self.use_suction = use_suction
        self.rotation_mode = rotation_mode
        self.translation_mode = translation_mode
        self.max_rotation_angle = max_rotation
        self.max_translation = max_translation
        self.suction_to_obj_pose = 0
        self.suction_contact_link = None
        self.suction_obj_id = None
        self.activated = 0
        
        if self.gui:
            try:
                self.id = p.connect(p.GUI)
            except:
                self.id = p.connect(p.DIRECT)
        else:
            self.id = p.connect(p.DIRECT)

        self.asset_dir = osp.join(osp.dirname(osp.realpath(__file__)), "assets/")
        p.setTimeStep(dt, physicsClientId=self.id)

        self.init_state = None
        self.handle_joint = None
        self.grasped_handle = False
        self.seed()
        self.set_scene()
        self.setup_camera_rpy()
        self.scene_lower, self.scene_upper = self.get_scene_bounds()
        self.scene_center = (self.scene_lower + self.scene_upper) / 2
        self.scene_range = (self.scene_upper - self.scene_lower) / 2

        self.action_low = np.array([-1, -1, -1, -1, -1, -1, -1])
        self.action_high = np.array([1, 1, 1, 1, 1, 1, 1])

        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, dtype=np.float32) 
        self.base_action_space = spaces.Box(low=self.action_low, high=self.action_high, dtype=np.float32) 
        self.num_objects = len(self.urdf_ids) - 2 # exclude plane, robot
        distractor_object_num = np.sum(list(self.is_distractor.values()))
        self.num_objects -= distractor_object_num

        ### For RL policy learning, observation space includes:
        # 1. object positions and orientations (6 * num_objects)
        # 2. object min and max bounding box (6 * num_objects)
        # 3. articulated object joint angles (num_objects * num_joints) 
        # 4. articulated object link position and orientation (num_objects * num_joints * 6) 
        # 5. robot base position (xy)
        # 6. robot end-effector position and orientation (6)
        # 7. gripper suction activated/deactivate or gripper joint angle (if not using suction gripper) (1)
        num_obs = self.num_objects * 12 # obs 1 and 2
        for name in self.urdf_types:
            if self.urdf_types[name] == 'urdf' and not self.is_distractor[name]: # obs 3 and 4
                num_joints = p.getNumJoints(self.urdf_ids[name], physicsClientId=self.id) 
                num_obs += num_joints
                num_obs += 6 * num_joints
        num_obs += 2 + 6 + 1 # obs 5 6 7
        self.base_num_obs = num_obs

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_obs, ), dtype=np.float32) 
        self.base_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.base_num_obs, ), dtype=np.float32)

        self.detected_position = {} # not used for now, keep it
        self.time_step = 0
        self.success = False
        self.control_rgbs = []
        self.init_joint_angle = None
        self.ik_failure = False
        
    def normalize_position(self, pos):
        if self.translation_mode == 'normalized-direct-translation':
            return (pos - self.scene_center) / self.scene_range 
        else:
            return pos

    def seed(self, seed=None):
        self.np_random, _ = seeding.np_random()

    def get_aabb(self, id):
        num_joints = p.getNumJoints(id, physicsClientId=self.id)
        min_aabbs, max_aabbs = [], []
        for link_idx in range(-1, num_joints):
            min_aabb, max_aabb = p.getAABB(id, link_idx, physicsClientId=self.id)
            min_aabbs.append(list(min_aabb))
            max_aabbs.append(list(max_aabb))
        min_aabb = np.min(np.concatenate(min_aabbs, axis=0).reshape(-1, 3), axis=0)
        max_aabb = np.max(np.concatenate(max_aabbs, axis=0).reshape(-1, 3), axis=0)
        return min_aabb, max_aabb
    
    def get_aabb_link(self, id, link_id):
        min_aabb, max_aabb = p.getAABB(id, link_id, physicsClientId=self.id)
        return np.array(min_aabb), np.array(max_aabb)

    def get_scene_bounds(self):
        min_aabbs = []
        max_aabbs = []
        for name, id in self.urdf_ids.items():
            if name == 'plane': continue
            min_aabb, max_aabb = self.get_aabb(id)
            min_aabbs.append(min_aabb)
            max_aabbs.append(max_aabb)
        
        min_aabb = np.min(np.stack(min_aabbs, axis=0).reshape(-1, 3), axis=0)
        max_aabb = np.max(np.stack(max_aabbs, axis=0).reshape(-1, 3), axis=0)
        range = max_aabb - min_aabb
        return min_aabb - 0.5 * range, max_aabb + 0.5 * range

    def clip_within_workspace(self, robot_pos, ori_pos, on_table):
        pos = ori_pos.copy()
        # If objects are too close to the robot, push them away
        x_near_low, x_near_high = robot_pos[0] - 0.4, robot_pos[0] + 0.4
        y_near_low, y_near_high = robot_pos[1] - 0.4, robot_pos[1] + 0.4

        if pos[0] > x_near_low and pos[0] < x_near_high:
            pos[0] = x_near_low if pos[0] < robot_pos[0] else x_near_high

        if pos[1] > y_near_low and pos[1] < y_near_high:
            pos[1] = y_near_low if pos[1] < robot_pos[1] else y_near_high
        if not on_table:
            return pos    
        else:
            # Object is on table, should be within table's bounding box
            new_pos = pos.copy()
            new_pos[:2] = np.clip(new_pos[:2], self.table_bbox_min[:2], self.table_bbox_max[:2])
            return new_pos
        
    def clip_x_bbox_within_workspace(self, robot_pos, ori_pos, on_table , min_bbox, max_bbox):
        x_near_low = robot_pos[0] - 0.7
        x_near_high = robot_pos[0] + 0.7
        offset = 0
        if min_bbox[0] > robot_pos[0]:
            if min_bbox[0] < x_near_high:
                offset = x_near_high - min_bbox[0]
        elif max_bbox[0] < robot_pos[0]:
            if max_bbox[0] > x_near_low:
                offset = x_near_low - max_bbox[0]

        pos = ori_pos.copy()
        pos[0] += offset
        
        if not on_table:
            return pos
        else:
            # Object is on table, should be within table's bounding box
            new_pos = pos.copy()
            new_pos[:2] = np.clip(new_pos[:2], self.table_bbox_min[:2], self.table_bbox_max[:2])
            return new_pos

    def get_robot_base_pos(self):
        robot_base_pos = [0, 0, 0]
        return robot_base_pos
    
    def get_robot_init_joint_angles(self, robot_init_joint_angles=None):
        if robot_init_joint_angles is None:
            init_joint_angles = [0 for _ in range(len(self.robot.right_arm_joint_indices))]

            init_joint_angles[3] = -0.4
            init_joint_angles[5] = 0.4
            return init_joint_angles  
        return robot_init_joint_angles

    def set_scene(
        self,
        reset_state=None,
    ):
        ### simulation preparation
        p.resetSimulation(physicsClientId=self.id)
        if self.gui:
            p.resetDebugVisualizerCamera(cameraDistance=1.75, cameraYaw=-25, cameraPitch=-45, cameraTargetPosition=[-0.2, 0, 0.4], physicsClientId=self.id)
            p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=self.id)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.id)
        p.setRealTimeSimulation(0, physicsClientId=self.id)
        p.setGravity(0, 0, self.gravity, physicsClientId=self.id)

        ### load restore state
        restore_state = None
        if self.restore_state_file is not None:
            with open(self.restore_state_file, 'rb') as f:
                restore_state = pickle.load(f)

        ### load and parse task config (including semantically meaningful distractor objects)
        self.urdf_paths = {}
        self.urdf_types = {}
        self.init_positions = {}
        self.init_orientations = {}
        self.on_tables = {}
        self.simulator_sizes = {}
        self.is_distractor = {
            "robot": 0,
            "plane": 0,
        }
        urdf_paths, urdf_sizes, urdf_positions, urdf_orientations, urdf_names, urdf_types, urdf_on_table, urdf_movables, \
            use_table, articulated_init_joint_angles, spatial_relationships, robot_initial_joint_angles = self.load_and_parse_config(restore_state)

        ### load plane 
        planeId = p.loadURDF(osp.join(self.asset_dir, "plane", "plane.urdf"), physicsClientId=self.id)

        ### create and load a robot
        self.robot_base_pos = self.load_robot(restore_state, robot_initial_joint_angles=robot_initial_joint_angles)

        self.urdf_ids = {
            "robot": self.robot.body,
            "plane": planeId,
        }
        

        ### handle the case if there is a table
        self.load_table(use_table, restore_state)

        ### load each object from the task config
        self.load_object(urdf_paths, urdf_sizes, urdf_positions, urdf_orientations, urdf_names, urdf_types, urdf_on_table, urdf_movables)

        ### if a state is passed in, restore the state
        if reset_state is not None:
            load_env(self, state=reset_state)
            return

        ### after first set scene, the init state will be stored, and can be restored here, skipping the following steps to save time
        if self.init_state is not None:
            load_env(self, state=self.init_state)
            return
        
        ### adjusting object positions
        ### place the lowest point on the object to be the height where GPT specifies
        object_height = self.adjust_object_positions(self.robot_base_pos)

        ### resolve collisions between objects
        self.resolve_collision(self.robot_base_pos, object_height, spatial_relationships)

        ### handle any special relationships outputted by GPT
        # self.handle_gpt_special_relationships(spatial_relationships)
        ### set all object's joint angles to the lower joint limit
        self.set_to_default_joint_angles()

        ### overwrite joint angles specified by GPT
        self.handle_gpt_joint_angle(articulated_init_joint_angles)
          
        # open the gripper at reset 
        if self.open_gripper_at_reset:
            for _ in range(20):
                self.robot.set_gripper_open_position(self.robot.right_gripper_indices, [0.04, 0.04], set_instantly=False)

        ### stabilize the scene
        for _ in range(500):
            p.stepSimulation(physicsClientId=self.id)

        ### restore to a state if provided
        if self.restore_state_file is not None:
            load_env(self, self.restore_state_file)
            # print("Restored state from: ", self.restore_state_file)

        ### record initial joint angles and positions
        self.record_initial_joint_and_pose()

        ### Enable debug rendering
        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)
 
        self.init_state = save_env(self)

        
    def load_robot(self, restore_state, robot_initial_joint_angles=None):
        robot_classes = {
            "panda": Panda,
            # "sawyer": Sawyer,
            # "ur5": UR5,
        }
        robot_names = list(robot_classes.keys())
        self.robot_name = robot_names[np.random.randint(len(robot_names))]
        if restore_state is not None and "robot_name" in restore_state:
            self.robot_name = restore_state['robot_name']
        self.robot_class = robot_classes[self.robot_name]
      
        # Create robot
        self.robot = self.robot_class(slider=self.mobile)
        self.robot.init(self.asset_dir, self.id, self.np_random, fixed_base=True, use_suction=self.use_suction)
        self.agents = [self.robot]
        self.suction_id = self.robot.right_gripper_indices[0]

        # Set robot base position & orientation, and joint angles
        robot_base_pos = self.get_robot_base_pos()
        robot_base_orient = [0, 0, 0, 1]
        self.robot_base_orient = robot_base_orient
        self.robot.set_base_pos_orient(robot_base_pos, robot_base_orient)
        init_joint_angles = self.get_robot_init_joint_angles(robot_initial_joint_angles)
        self.robot.set_joint_angles(self.robot.right_arm_joint_indices, init_joint_angles)    
        self.robot.set_gravity(0, 0, 0)
        
        return robot_base_pos        
    
    def load_and_parse_config(self, restore_state):
        ### select and download objects from objaverse
        res = download_and_parse_objavarse_obj_from_yaml_config(self.config_path, candidate_num=self.object_candidate_num, vhacd=self.vhacd)
        if not res:
            print("=" * 20)
            print("some objects cannot be found in objaverse, task_build failed, now exit ...")
            print("=" * 20)
            exit()
        
        self.config = None
        while self.config is None:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        for obj in self.config:
            if "solution_path" in obj:
                self.solution_path = obj["solution_path"]
                break
        
        ### parse config
        urdf_paths, urdf_sizes, urdf_positions, urdf_orientations, urdf_names, urdf_types, urdf_on_table, \
            use_table, articulated_init_joint_angles, spatial_relationships, distractor_config_path, urdf_movables, \
                robot_initial_joint_angles = parse_config(self.config, 
                        use_bard=True, obj_id=self.obj_id,
                        use_vhacd=True)
        if not use_table:
            urdf_on_table = [False for _ in urdf_on_table]
        urdf_names = [x.lower() for x in urdf_names]
        for name in urdf_names:
            self.is_distractor[name] = 0
        
        ### parse distractor object config (semantically meaningful objects that are related but not used for the task)
        if distractor_config_path is not None:
            self.distractor_config_path = distractor_config_path
            res = download_and_parse_objavarse_obj_from_yaml_config(distractor_config_path, candidate_num=self.object_candidate_num, vhacd=self.vhacd)
            with open(distractor_config_path, 'r') as f:
                self.distractor_config = yaml.safe_load(f)
            distractor_urdf_paths, distractor_urdf_sizes, distractor_urdf_positions, _, distractor_urdf_names, distractor_urdf_types, \
                distractor_urdf_on_table, _, _, _, _, _, _ = \
                    parse_config(self.distractor_config, use_bard=True, obj_id=self.obj_id, use_vhacd=False)
            distractor_urdf_names = [x.lower() for x in distractor_urdf_names]
            if not use_table:
                distractor_urdf_on_table = [False for _ in distractor_urdf_on_table]
            
            for name in distractor_urdf_names:
                self.is_distractor[name] = 1
                
            distractor_movables = [True for _ in distractor_urdf_names]
            distractor_orientations = [[0,0,0,1] for _ in distractor_urdf_names]
            urdf_paths += distractor_urdf_paths
            urdf_sizes += distractor_urdf_sizes
            urdf_positions += distractor_urdf_positions
            urdf_orientations += distractor_orientations
            urdf_names += distractor_urdf_names
            urdf_types += distractor_urdf_types
            urdf_on_table += distractor_urdf_on_table
            urdf_movables += distractor_movables

        if restore_state is not None:
            if "urdf_paths" in restore_state:
                self.urdf_paths = {}
                for urdf_name in restore_state['urdf_paths']:
                    urdf_path = restore_state['urdf_paths'][urdf_name]
                    start_idx = urdf_path.find("data/dataset")
                    urdf_path = urdf_path[start_idx:]
                    urdf_path = os.path.join(os.environ["PROJECT_DIR"], urdf_path)
                    self.urdf_paths[urdf_name] = urdf_path
                    
                urdf_paths = [self.urdf_paths[name] for name in urdf_names]
            if "object_sizes" in restore_state:
                self.simulator_sizes = restore_state['object_sizes']
                urdf_sizes = [self.simulator_sizes[name] for name in urdf_names]
                
        return urdf_paths, urdf_sizes, urdf_positions, urdf_orientations, urdf_names, urdf_types, urdf_on_table, urdf_movables, \
            use_table, articulated_init_joint_angles, spatial_relationships, robot_initial_joint_angles
        
    def load_table(self, use_table, restore_state):
        self.use_table = use_table
        if use_table:
            from manipulation.table_utils import table_paths, table_scales, table_poses, table_bbox_scale_down_factors
            # self.table_path = table_paths[np.random.randint(len(table_paths))]
            self.table_path = "table_1d43b7612af94e1183f00e604d9edf4a"
            if restore_state is not None:
                self.table_path = restore_state['table_path']

            table_scale = table_scales[self.table_path] 
            table_pos = table_poses[self.table_path]
            table_orientation = [np.pi/2, 0, 0]

            self.table = p.loadURDF(osp.join(self.asset_dir, self.table_path, "material.urdf"), physicsClientId=self.id, useFixedBase=True, 
                                    globalScaling=table_scale)
            
            if not self.randomize:
                random_orientation = p.getQuaternionFromEuler(table_orientation, physicsClientId=self.id)
            else:
                random_orientations = [0, np.pi / 2, np.pi, np.pi * 3 / 2]
                random_orientation = p.getQuaternionFromEuler([np.pi/2, 0, random_orientations[np.random.randint(4)]], physicsClientId=self.id)

            p.resetBasePositionAndOrientation(self.table, table_pos, random_orientation, physicsClientId=self.id)
            self.table_bbox_min, self.table_bbox_max = self.get_aabb(self.table)
            
            table_range = self.table_bbox_max - self.table_bbox_min
            self.table_bbox_min[:2] += table_range[:2] * table_bbox_scale_down_factors[self.table_path]
            self.table_bbox_max[:2] -= table_range[:2] * table_bbox_scale_down_factors[self.table_path]
            self.table_height = self.table_bbox_max[2]
            p.addUserDebugLine([*self.table_bbox_min[:2], self.table_height], self.table_bbox_max, [1, 0, 0], lineWidth=10, lifeTime=0, physicsClientId=self.id)
            self.simulator_sizes["init_table"] = table_scale
            self.urdf_ids["init_table"] = self.table
            self.is_distractor['init_table'] = 0
            
            if not self.mobile:
                # set robot to be on the table
                robot_base_pos_table = np.array([0.10, 0.5, 0])
                robot_base_pos_world = np.array([0, 0, 0.])
                x_range = self.table_bbox_max[0] - self.table_bbox_min[0]
                y_range = self.table_bbox_max[1] - self.table_bbox_min[1]
                robot_base_pos_world[0] = self.table_bbox_min[0] + robot_base_pos_table[0] * x_range
                robot_base_pos_world[1] = self.table_bbox_min[1] + robot_base_pos_table[1] * y_range
                robot_base_pos_world[2] = self.table_height + 0.05
                robot_base_orient = [0, 0, 0, 1]
                self.robot_base_pos = robot_base_pos_world
                self.robot.set_base_pos_orient(robot_base_pos_world, robot_base_orient)
                
    def load_object(self, urdf_paths, urdf_sizes, urdf_positions, urdf_orientations, urdf_names, urdf_types, urdf_on_table, urdf_movables):
        for path, size, pos, urdf_ori, name, type, on_table, moveable in zip(urdf_paths, urdf_sizes, urdf_positions, urdf_orientations, urdf_names, urdf_types, urdf_on_table, urdf_movables):
            # print("Loading object: {} path {}".format(name, path))
            
            name = name.lower()
            # by default, all objects movable, except the urdf files
            use_fixed_base = (type == 'urdf' and not self.is_distractor[name])
            if type == 'urdf' and moveable: # if gpt specified the object is movable, then it is movable
                use_fixed_base = False
            
            if type == 'urdf':
                size = min(size, 1.2)
                size = max(size, 0.075) # if the object is too small, current gripper cannot really manipulate it.
            
            x_orient = np.pi/2 if type == 'mesh' else 0 # handle different coordinate axis by objaverse and partnet-mobility
            if self.randomize or self.is_distractor[name]:
                orientation = p.getQuaternionFromEuler([x_orient, 0, self.np_random.uniform(-np.pi/3, np.pi/3)], physicsClientId=self.id)
            else:
                orientation = p.getQuaternionFromEuler([x_orient, 0, 0], physicsClientId=self.id)

            # combine the orientation from the config file
            urdf_mat = R.from_quat(urdf_ori)
            ori_mat = R.from_quat(orientation)
            orientation = urdf_mat * ori_mat
            orientation = orientation.as_quat()

            if not on_table:
                load_pos = pos
            else: # change to be table coordinate
                table_xy_range = self.table_bbox_max[:2] - self.table_bbox_min[:2]
                obj_x = self.table_bbox_min[0] + pos[0] * table_xy_range[0]
                obj_y = self.table_bbox_min[1] + pos[1] * table_xy_range[1]
                obj_z = self.table_height + pos[2]
                load_pos = [obj_x, obj_y, obj_z]
            id = p.loadURDF(path, basePosition=load_pos, baseOrientation=orientation, physicsClientId=self.id, useFixedBase=use_fixed_base, globalScaling=size)

            # scale size 
            if name in self.simulator_sizes:
                p.removeBody(id, physicsClientId=self.id)
                saved_size = self.simulator_sizes[name]
                id = p.loadURDF(path, basePosition=load_pos, baseOrientation=orientation, physicsClientId=self.id, useFixedBase=use_fixed_base, globalScaling=saved_size)
            elif size == -1:
                id = p.loadURDF(path, basePosition=load_pos, baseOrientation=orientation, physicsClientId=self.id, useFixedBase=use_fixed_base)
                min_aabb, max_aabb = self.get_aabb(id)
                actual_size = np.linalg.norm(max_aabb - min_aabb)
                self.simulator_sizes[name] = np.sqrt(actual_size)
            else:
                min_aabb, max_aabb = self.get_aabb(id)
                actual_size = np.linalg.norm(max_aabb - min_aabb)
                if np.abs(actual_size - size) > 0.05:
                    p.removeBody(id, physicsClientId=self.id)
                    id = p.loadURDF(path, basePosition=load_pos, baseOrientation=orientation, physicsClientId=self.id, useFixedBase=use_fixed_base, globalScaling=size ** 2 / actual_size)
                    self.simulator_sizes[name] = size ** 2 / actual_size
                else:
                    self.simulator_sizes[name] = size

            self.urdf_ids[name] = id
            self.urdf_paths[name] = path
            self.urdf_types[name] = type
            self.init_positions[name] = np.array(load_pos)
            self.init_orientations[name] = orientation
            self.on_tables[name] = on_table

            # print("Finished loading object: ", name)
    
    def adjust_object_positions(self, robot_base_pos):
        object_height = {}
        for name, id in self.urdf_ids.items():
            if name == 'robot' or name == 'plane' or name == 'init_table': continue
            min_aabb, max_aabb = self.get_aabb(id)
            min_z = min_aabb[2]
            object_height[id] = 2 * self.init_positions[name][2] - min_z
            pos, orient = p.getBasePositionAndOrientation(id, physicsClientId=self.id)
            new_pos = np.array(pos) 
            # new_pos = self.clip_within_workspace(robot_base_pos, new_pos, self.on_tables[name])
            new_pos = self.clip_x_bbox_within_workspace(robot_base_pos, new_pos, self.on_tables[name], min_aabb, max_aabb)
            new_pos[2] = object_height[id]
            p.resetBasePositionAndOrientation(id, new_pos, orient, physicsClientId=self.id)
            self.init_positions[name] = new_pos
        
        return object_height
        
    def resolve_collision(self, robot_base_pos, object_height, spatial_relationships):
        collision = True
        collision_cnt = 1
        while collision:
            if collision_cnt % 50 == 0: # if collision is not resolved every 50 iterations, we randomly reset object's position
                for name, id in self.urdf_ids.items():
                    if name == 'robot' or name == 'plane' or name == "init_table": continue
                    pos = self.init_positions[name]
                    _, orient = p.getBasePositionAndOrientation(id, physicsClientId=self.id)
                    new_pos = np.array(pos) + np.random.uniform(-0.2, 0.2, size=3)
                    # new_pos = self.clip_within_workspace(robot_base_pos, new_pos, self.on_tables[name])
                    min_aabb, max_aabb = self.get_aabb(id)
                    new_pos = self.clip_x_bbox_within_workspace(robot_base_pos, new_pos, self.on_tables[name], min_aabb, max_aabb)
                    new_pos[2] = object_height[id]
                    p.resetBasePositionAndOrientation(id, new_pos, orient, physicsClientId=self.id)
                    p.stepSimulation(physicsClientId=self.id)

            push_directions = defaultdict(list) # store the push direction for each object

            # detect collisions between objects 
            detected_collision = False
            for name, id in self.urdf_ids.items():
                if name == 'robot' or name == 'plane' or name == 'init_table': continue
                for name2, id2 in self.urdf_ids.items():
                    if name == name2 or name2 == 'robot' or name2 == 'plane' or name2 == 'init_table': continue

                    # if gpt specifies obj a and obj b should have some special relationship, then skip collision resolution
                    skip = False
                    for spatial_relationship in spatial_relationships:
                        words = spatial_relationship.lower().split(",")
                        words = [word.strip().lstrip() for word in words]
                        if name in words and name2 in words:
                            skip = True
                            break

                    if skip: continue
                    
                    contact_points = p.getClosestPoints(id, id2, 0.01, physicsClientId=self.id)
                    if len(contact_points) > 0:
                        contact_point = contact_points[0]
                        push_direction = contact_point[7]
                        push_direction = np.array([push_direction[0], push_direction[1], push_direction[2]])

                        # both are distractors or both are not, push both objects away
                        if (self.is_distractor[name] and self.is_distractor[name2]) or \
                            (not self.is_distractor[name] and not self.is_distractor[name2]):
                            push_directions[id].append(-push_direction)
                            push_directions[id2].append(push_direction)
                        # only 1 is distractor, only pushes the distractor
                        if self.is_distractor[name] and not self.is_distractor[name2]:
                            push_directions[id].append(push_direction)
                        if not self.is_distractor[name] and self.is_distractor[name2]:
                            push_directions[id2].append(-push_direction)
                        
                        detected_collision = True

            # collisions between robot and objects, only push object away
            for name, id in self.urdf_ids.items():
                if name == 'robot' or name == 'plane' or name == 'init_table': 
                    continue

                contact_points = p.getClosestPoints(self.robot.body, id, 0.05, physicsClientId=self.id)
                if len(contact_points) > 0:
                    contact_point = contact_points[0]
                    push_direction = contact_point[7]
                    push_direction = np.array([push_direction[0], push_direction[1], push_direction[2]])
                    push_directions[id].append(-push_direction)
                    detected_collision = True

            # between table and objects that should not be placed on table
            if self.use_table:
                for name, id in self.urdf_ids.items():
                    if name == 'robot' or name == 'plane' or name == 'init_table': 
                        continue
                    if self.on_tables[name]:
                        continue

                    contact_points = p.getClosestPoints(self.robot.body, id, 0.05, physicsClientId=self.id)
                    if len(contact_points) > 0:
                        contact_point = contact_points[0]
                        push_direction = contact_point[7]
                        push_direction = np.array([push_direction[0], push_direction[1], push_direction[2]])
                        push_directions[id].append(-push_direction)
                        detected_collision = True
            
            # move objects
            push_distance = 0.1
            for id in push_directions:
                for direction in push_directions[id]:
                    pos, orient = p.getBasePositionAndOrientation(id, physicsClientId=self.id)
                    new_pos = np.array(pos) + push_distance * direction    
                    # new_pos = self.clip_within_workspace(robot_base_pos, new_pos, self.on_tables[name])
                    min_aabb, max_aabb = self.get_aabb(id)
                    new_pos = self.clip_x_bbox_within_workspace(robot_base_pos, new_pos, self.on_tables[name], min_aabb, max_aabb)
                    new_pos[2] = object_height[id]

                    p.resetBasePositionAndOrientation(id, new_pos, orient, physicsClientId=self.id)
                    p.stepSimulation(physicsClientId=self.id)

            collision = detected_collision
            collision_cnt += 1

            if collision_cnt > 1000:
                break
    
    def record_initial_joint_and_pose(self):
        self.initial_joint_angle = {}
        for name in self.urdf_ids:        
            obj_id = self.urdf_ids[name.lower()]
            if name == 'robot' or name == 'plane' or name == "init_table": continue
            if self.urdf_types[name.lower()] == 'urdf':
                self.initial_joint_angle[name] = {}
                num_joints = p.getNumJoints(obj_id, physicsClientId=self.id)
                for joint_idx in range(num_joints):
                    joint_name = p.getJointInfo(obj_id, joint_idx, physicsClientId=self.id)[1].decode("utf-8")
                    joint_angle = p.getJointState(obj_id, joint_idx, physicsClientId=self.id)[0]
                    self.initial_joint_angle[name][joint_name] = joint_angle
        
        self.initial_pos = {}
        self.initial_orient = {}
        for name in self.urdf_ids:
            obj_id = self.urdf_ids[name.lower()]
            if name == 'robot' or name == 'plane' or name == "init_table": continue
            pos, orient = p.getBasePositionAndOrientation(obj_id, physicsClientId=self.id)
            self.initial_pos[name] = pos
            self.initial_orient[name] = orient
            
        self.initial_eef_pos = self.robot.get_pos_orient(self.robot.right_end_effector)[0]
        
    def set_to_default_joint_angles(self):
        for obj_name in self.urdf_ids:
            if obj_name == 'robot' or obj_name == 'plane' or obj_name == "init_table": continue
            obj_id = self.urdf_ids[obj_name]
            num_joints = p.getNumJoints(obj_id, physicsClientId=self.id)
            for joint_idx in range(num_joints):
                joint_limit_low, joint_limit_high = p.getJointInfo(obj_id, joint_idx, physicsClientId=self.id)[8:10]
                if joint_limit_low > joint_limit_high:
                    joint_limit_low, joint_limit_high = joint_limit_high, joint_limit_low
                # joint_val = joint_limit_low + 0.06 * (joint_limit_high - joint_limit_low)
                joint_val = joint_limit_low
                p.resetJointState(obj_id, joint_idx, joint_val, physicsClientId=self.id)

    def handle_gpt_special_relationships(self, spatial_relationships):
        # we support "on" and "in" for now, but this can be extended to more relationships
        for spatial_relationship in spatial_relationships:
            words = spatial_relationship.lower().split(",")
            words = [word.strip().lstrip() for word in words]
            if words[0] == "on":
                obj_a = words[1]
                obj_b = words[2]
                if len(words) == 4:
                    obj_b_link = words[3]
                    obj_b_link_id = get_link_id_from_name(self, obj_b, obj_b_link)
                else:
                    obj_b_link_id = -1
                obj_a_id, obj_b_id = self.urdf_ids[obj_a], self.urdf_ids[obj_b]
                
                obj_a_bbox_min, obj_a_bbox_max = self.get_aabb(obj_a_id)
                obj_a_size = obj_a_bbox_max - obj_a_bbox_min
                target_aabb_min, target_aabb_max = self.get_aabb_link(obj_b_id, obj_b_link_id)
                id_line = p.addUserDebugLine(target_aabb_min, target_aabb_max, [1, 0, 0], lineWidth=10, lifeTime=0, physicsClientId=self.id)
                id_point = p.addUserDebugPoints([(target_aabb_min + target_aabb_max) / 2], [[0, 0, 1]], 10, 0, physicsClientId=self.id)

                new_pos = (target_aabb_min + target_aabb_max) / 2
                new_pos[2] = target_aabb_max[2] # put obj a on top of obj b.
                new_pos[2] += obj_a_size[2] # add the height of obj a
                if not self.randomize:
                    obj_a_orientation = p.getQuaternionFromEuler([np.pi/2, 0, 0], physicsClientId=self.id)
                else:
                    random_orientations = [0, np.pi / 2, np.pi, np.pi * 3 / 2]
                    obj_a_orientation = p.getQuaternionFromEuler([np.pi/2, 0, random_orientations[np.random.randint(4)]], physicsClientId=self.id)

                p.resetBasePositionAndOrientation(obj_a_id, new_pos, obj_a_orientation, physicsClientId=self.id)
                
                p.removeUserDebugItem(id_line, physicsClientId=self.id)
                p.removeUserDebugItem(id_point, physicsClientId=self.id)

            if words[0] == 'in':
                obj_a = words[1]
                obj_b = words[2]
                if len(words) == 4:
                    obj_b_link = words[3]
                    obj_b_link_id = get_link_id_from_name(self, obj_b, obj_b_link)
                else:
                    obj_b_link_id = -1
                obj_a_id, obj_b_id = self.urdf_ids[obj_a], self.urdf_ids[obj_b]
                
                # if after a lot of trying times, there is still collision, we should scale down the size of object A.
                cnt = 1
                collision_free = False
                obj_a_new_size = self.simulator_sizes[obj_a]
                obj_a_ori_pos, obj_a_orientation = p.getBasePositionAndOrientation(obj_a_id, physicsClientId=self.id)         
                target_aabb_min, target_aabb_max = self.get_aabb_link(obj_b_id, obj_b_link_id)

                while not collision_free:
                    if cnt % 100 == 0:
                        # print("scaling down! object size is {}".format(obj_a_new_size))
                        obj_a_new_size = obj_a_new_size * 0.9
                        p.removeBody(obj_a_id, physicsClientId=self.id)
                        obj_a_id = p.loadURDF(self.urdf_paths[obj_a],
                                            basePosition=obj_a_ori_pos,
                                            baseOrientation=obj_a_orientation,
                                            physicsClientId=self.id, useFixedBase=False, globalScaling=obj_a_new_size)
                        self.urdf_ids[obj_a] = obj_a_id
                        self.simulator_sizes[obj_a] = obj_a_new_size

                    obj_a_bbox_min, obj_a_bbox_max = self.get_aabb(obj_a_id)
                    obj_a_size = obj_a_bbox_max - obj_a_bbox_min
                    id_line = p.addUserDebugLine(target_aabb_min, target_aabb_max, [1, 0, 0], lineWidth=10, lifeTime=0, physicsClientId=self.id)
                    id_point = p.addUserDebugPoints([(target_aabb_min + target_aabb_max) / 2], [[0, 0, 1]], 10, 0, physicsClientId=self.id)

                    center_pos = (target_aabb_min + target_aabb_max) / 2
                    up_pos = center_pos.copy()
                    up_pos[2] += obj_a_size[2]
                    possible_locations = [center_pos, up_pos]
                    obj_a_orientation = p.getQuaternionFromEuler([np.pi/2, 0, 0], physicsClientId=self.id)
                    for pos in possible_locations: # we try two possible locations to put obj a in obj b
                        p.resetBasePositionAndOrientation(obj_a_id, pos, obj_a_orientation, physicsClientId=self.id)
                        contact_points = p.getClosestPoints(obj_a_id, obj_b_id, 0.002, physicsClientId=self.id)

                        if len(contact_points) == 0:
                            collision_free = True
                            break
                    
                    p.removeUserDebugItem(id_line, physicsClientId=self.id)
                    p.removeUserDebugItem(id_point, physicsClientId=self.id)

                    cnt += 1
                    if cnt > 1000: # if after scaling for 10 times it still does not work, let it be. 
                        break
                        

    def handle_gpt_joint_angle(self, articulated_init_joint_angles):
        # import pdb; pdb.set_trace()
        for name in articulated_init_joint_angles:
            obj_id = self.urdf_ids[name.lower()]

            if "set_joint_angle_joint_id" not in articulated_init_joint_angles[name].keys():
                for joint_name, joint_angle in articulated_init_joint_angles[name].items():
                    joint_idx = get_joint_id_from_name(self, name.lower(), joint_name)
                    joint_limit_low, joint_limit_high = p.getJointInfo(obj_id, joint_idx, physicsClientId=self.id)[8:10]
                    if joint_limit_low > joint_limit_high:
                        joint_limit_low, joint_limit_high = joint_limit_high, joint_limit_low
                    if 'random' not in joint_angle:
                        # joint_angle = float(joint_angle)
                        # joint_angle = min(joint_angle, 0.7)
                        # joint_angle = max(joint_angle, 0.06)
                        # joint_angle = joint_limit_low + joint_angle * (joint_limit_high - joint_limit_low)
                        joint_angle = joint_limit_low
                    else:
                        joint_angle = self.np_random.uniform(joint_limit_low, joint_limit_high)
            else:
                # TODO: account for cases when there are multiple joints to be set.
                p.resetJointState(obj_id, articulated_init_joint_angles[name]["set_joint_angle_joint_id"], 
                              articulated_init_joint_angles[name]['set_joint_angle_joint_angle'], physicsClientId=self.id)

    def reset(self, reset_state=None, object_name='StorageFurniture', open_gripper_at_reset=False):
        self.grasped_handle = False
        self.set_scene(reset_state)
            
        self.time_step = 0
        self.success = False
        object_name = object_name.lower()
        friction = 5
        num_links = p.getNumJoints(self.urdf_ids[object_name], physicsClientId=self.id)
        for l_id in range(num_links):
            p.changeDynamics(self.urdf_ids[object_name], l_id, lateralFriction=friction, physicsClientId=self.id)
            p.changeDynamics(self.urdf_ids[object_name], l_id, rollingFriction=friction, physicsClientId=self.id)
            p.changeDynamics(self.urdf_ids[object_name], l_id, spinningFriction=friction, physicsClientId=self.id)

        p.changeDynamics(self.robot.body, self.robot.right_gripper_indices[0], lateralFriction=friction, physicsClientId=self.id)
        p.changeDynamics(self.robot.body, self.robot.right_gripper_indices[1], lateralFriction=friction, physicsClientId=self.id)
        p.changeDynamics(self.robot.body, self.robot.right_gripper_indices[0], rollingFriction=friction, physicsClientId=self.id)
        p.changeDynamics(self.robot.body, self.robot.right_gripper_indices[1], rollingFriction=friction, physicsClientId=self.id)
        p.changeDynamics(self.robot.body, self.robot.right_gripper_indices[0], spinningFriction=friction, physicsClientId=self.id)
        p.changeDynamics(self.robot.body, self.robot.right_gripper_indices[1], spinningFriction=friction, physicsClientId=self.id)

        self.ik_failure = False

        return self._get_obs()

    def setup_camera(self, camera_eye=[0.5, -0.75, 1.5], camera_target=[-0.2, 0, 0.75], fov=60, camera_width=640, camera_height=480):
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.view_matrix = p.computeViewMatrix(camera_eye, camera_target, [0, 0, 1], physicsClientId=self.id)
        self.projection_matrix = p.computeProjectionMatrixFOV(fov, camera_width / camera_height, 0.01, 100, physicsClientId=self.id)
    
    def setup_camera_rpy(self, camera_target=None, distance=1.6, rpy=[0, -30, -30], fov=60, camera_width=640, camera_height=480):
        self.camera_width = camera_width
        self.camera_height = camera_height
        if camera_target is None:
            for name in self.urdf_ids: # randomly center at an object
                if name in ['robot', 'plane', 'init_table']: continue
                obj_id = self.urdf_ids[name]
                min_aabb, max_aabb = self.get_aabb(obj_id)
                center = (min_aabb + max_aabb) / 2
                camera_target = center 
                break

        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(camera_target, distance, rpy[2], rpy[1], rpy[0], 2, physicsClientId=self.id)
        self.projection_matrix = p.computeProjectionMatrixFOV(fov, camera_width / camera_height, 0.01, 100, physicsClientId=self.id)

    def render(self, return_depth=False, mode=None):
        assert self.view_matrix is not None, 'You must call env.setup_camera() or env.setup_camera_rpy() before getting a camera image'
        w, h, img, depth, segmask = p.getCameraImage(self.camera_width, self.camera_height, 
            self.view_matrix, self.projection_matrix, 
            renderer=p.ER_BULLET_HARDWARE_OPENGL, 
            physicsClientId=self.id)
        img = np.reshape(img, (h, w, 4))[:, :, :3]
        depth = np.reshape(depth, (h, w))

        if return_depth:
            return img, depth
        else:
            return img
        
    def render_with_segmask(self, name=None):
        name = name.lower()
        obj_id = self.urdf_ids[name]
        min_aabb, max_aabb = self.get_aabb(obj_id)
        center = (min_aabb + max_aabb) / 2
        camera_target = center
        camera_eye = center + np.array([0.2, 0.0, 0.3])
        near = 0.01
        far = 1000
        view_matrix = p.computeViewMatrix(camera_eye, camera_target, [0, 0, 1], physicsClientId=self.id)
        projection_matrix = p.computeProjectionMatrixFOV(60, self.camera_width / self.camera_height, near, far, physicsClientId=self.id)


        w, h, img, depth, segmask = p.getCameraImage(self.camera_width, self.camera_height, 
            view_matrix, projection_matrix, 
            renderer=p.ER_BULLET_HARDWARE_OPENGL, 
            physicsClientId=self.id)
        img = np.reshape(img, (h, w, 4))[:, :, :3]
        depth = np.reshape(depth, (h, w))

        # extract near and far from projection matrix
        depth = near * far / (far - (far - near) * depth)
        
        # extract the object's mask by matching the object's id
        mask = (segmask == obj_id).astype(np.uint8)

        # get camera matrix K
        camera_K = np.array(projection_matrix).reshape(4, 4).T
        camera_K = camera_K[:3, :3]
        camera_K = camera_K / camera_K[2, 2]

        return img, depth, camera_K, mask
    
    def take_direct_action(self, actions, gains=None, forces=None):
        if gains is None:
            gains = [a.motor_gains for a in self.agents]
        elif type(gains) not in (list, tuple): 
            gains = [gains]*len(self.agents)
        if forces is None:
            forces = [a.motor_forces for a in self.agents]
        elif type(forces) not in (list, tuple):
            forces = [forces]*len(self.agents)

        np.random.seed(time.time_ns() % 2**32)

        self.control_rgbs = []
        action_index = 0
        for i, agent in enumerate(self.agents):
            agent_action_len = self.base_action_space.shape[0] 
            action = np.copy(actions[action_index:action_index+agent_action_len])
            action_index += agent_action_len
            translation = action[:3]
            rotation = action[3:6]
            finger_joint_angle = action[6]
            original_joint_angles = agent.get_joint_angles(agent.all_joint_indices)

            ik_indices = [_ for _ in range(len(agent.right_arm_ik_indices))]
            
            pos = translation
            orient = p.getQuaternionFromEuler(rotation)

            # trying to use ikpy
            # agent.ik_ikpy_franka(pos, orient, ik_indices)
            # trying to use tracik
            # agent_joint_angles, ik_success = agent.ik_tracik_franka(pos, orient, ik_indices)
            tracIK_solutions = agent.ik_tracik_franka(pos, orient, ik_indices)
            # if not ik_success:
            #     cprint("tracIK failed, maintain current joint angle", "red")
            #     agent_joint_angles = original_joint_angles
            
            bullet_solutions = []
            old_state = save_env(self)
            ik_indices = [_ for _ in range(len(self.robot.right_arm_joint_indices))]
            for try_idx in range(25):
                if try_idx > 0: 
                    new_joint_angles = original_joint_angles[ik_indices] + np.random.uniform(-0.3, 0.3, size=len(ik_indices))
                    self.robot.set_joint_angles(ik_indices, new_joint_angles)

                ik_joint_angles = self.robot.ik(self.robot.right_end_effector, 
                                                pos, orient, 
                                                ik_indices=ik_indices, 
                                            max_iterations=10000, residualThreshold=1e-4)
                if np.all(ik_joint_angles >= self.robot.ik_lower_limits[ik_indices]) and np.all(ik_joint_angles <= self.robot.ik_upper_limits[ik_indices]):
                    bullet_solutions.append(ik_joint_angles)

            load_env(self, state = old_state)
            all_possible_solutions = tracIK_solutions + bullet_solutions
            if len(all_possible_solutions) > 0:
                all_possible_solutions = np.array(all_possible_solutions).reshape(-1, len(ik_indices))
                distance_to_cur_angle = np.linalg.norm(all_possible_solutions - original_joint_angles[ik_indices].reshape(1, -1), axis=1)
                min_idx = np.argmin(distance_to_cur_angle)
                min_joint_distance = distance_to_cur_angle[min_idx]
                best_joint_angles = all_possible_solutions[min_idx]
                agent_joint_angles = best_joint_angles
                ik_success = min_joint_distance < 0.3
            else:
                ik_success = False

            self.ik_failure = (not ik_success) or self.ik_failure
            
            # agent_joint_angles = agent_joint_angles[ik_indices]
            it = 0
            # old way of control till reach
            # control_total = 50 # previously it was 50
            # points_left_finger = p.getContactPoints(bodyA=self.robot.body, linkIndexA=self.robot.right_gripper_indices[0], physicsClientId=self.id)
            # points_right_finger = p.getContactPoints(bodyA=self.robot.body, linkIndexA=self.robot.right_gripper_indices[1], physicsClientId=self.id)
            # finger_contact = points_left_finger or points_right_finger
            
            
            beg = time.time()
            if ik_success:
                control_total = 50
                save_img_interval = 0
                # gripper
                for _ in range(2):
                    if not self.use_suction:
                        agent.set_gripper_open_position(agent.right_gripper_indices, [finger_joint_angle, finger_joint_angle], set_instantly=False)
                    p.stepSimulation(physicsClientId=self.id) 
                    
                while True:
                    # cur_finger_joint_angles = np.array(agent.get_joint_angles(agent.right_gripper_indices))
                    # if np.linalg.norm(cur_finger_joint_angles - np.array([finger_joint_angle, finger_joint_angle])) > 1e-3:
                    #     agent.set_gripper_open_position(agent.right_gripper_indices, [finger_joint_angle, finger_joint_angle], set_instantly=False)  

                    agent.control(agent.controllable_joint_indices, agent_joint_angles)
                    cur_joint_angles = agent.get_joint_angles(agent.controllable_joint_indices)
                    if np.linalg.norm(cur_joint_angles - agent_joint_angles) < 1e-4:
                        break

                    if save_img_interval > 0 and it % save_img_interval == 0:
                        rgb = self.render()
                        self.control_rgbs.append(rgb)

                    if it > control_total:
                        # cprint("++++control total steps reached++++", "red")
                        break
                    
                    it += 1
                    p.stepSimulation(physicsClientId=self.id) 
                    
                
                end = time.time()
            else:
                cprint("IK failed, not doing anything", "red")
            # cprint("control time: {}".format(end - beg), "red")
    
    def take_joint_action(self, action):
        # action is the normalized delta joint angle for right arm joints and the finger joint
        # import pdb; pdb.set_trace()

        right_arm_indices = self.robot.right_arm_joint_indices
        cur_joint_angle_right_arm = self.robot.get_joint_angles(right_arm_indices)
        ik_lower_limit = self.robot.ik_lower_limits[right_arm_indices] 
        ik_upper_limit = self.robot.ik_upper_limits[right_arm_indices]
        unormalized_arm_delta_joint = action[:7] * (ik_upper_limit - ik_lower_limit) 
        new_joint_angle_right_arm = cur_joint_angle_right_arm + unormalized_arm_delta_joint
        new_joint_angle_right_arm = np.clip(new_joint_angle_right_arm, ik_lower_limit, ik_upper_limit)
        
        
        cur_joint_angle_finger = self.robot.get_joint_angles(self.robot.right_gripper_indices)
        new_finger_joint_angle = cur_joint_angle_finger[0] + action[7] * 0.04
        # print("action[7]: ", action[7])
        new_finger_joint_angle = np.clip(new_finger_joint_angle, 0, 0.04)
        # print("old_finger_joint_angle: ", cur_joint_angle_finger)
        # print("new_finger_joint_angle: ", new_finger_joint_angle)
        
        agent = self.robot
        save_img_interval = 0
        # for _ in range(2):
        #     p.stepSimulation(physicsClientId=self.id)
        
        for it in range(50):
            agent.control(right_arm_indices, new_joint_angle_right_arm)
            agent.set_gripper_open_position(agent.right_gripper_indices, [new_finger_joint_angle, new_finger_joint_angle], set_instantly=False)
            cur_joint_angles = agent.get_joint_angles(right_arm_indices)
            if np.linalg.norm(cur_joint_angles - new_joint_angle_right_arm) < 1e-4:
                break

            if save_img_interval > 0 and it % save_img_interval == 0:
                rgb = self.render()
                self.control_rgbs.append(rgb)

            it += 1
            p.stepSimulation(physicsClientId=self.id) 
        
        
    
    def get_control_rgbs(self):
        return self.control_rgbs

    def take_step(self, actions, gains=None, forces=None):
        if gains is None:
            gains = [a.motor_gains for a in self.agents]
        elif type(gains) not in (list, tuple): 
            gains = [gains]*len(self.agents)
        if forces is None:
            forces = [a.motor_forces for a in self.agents]
        elif type(forces) not in (list, tuple):
            forces = [forces]*len(self.agents)

        action_index = 0
        for i, agent in enumerate(self.agents):
            agent_action_len = self.base_action_space.shape[0] 
            action = np.copy(actions[action_index:action_index+agent_action_len])
            action_index += agent_action_len
            action = np.clip(action, self.action_low, self.action_high)

            translation = action[:3]
            rotation = action[3:6]
            suction = action[6]

            joint = agent.right_end_effector if 'right' in agent.controllable_joints else agent.left_end_effector
            agent_joint_angles = agent.get_joint_angles(agent.controllable_joint_indices)
            ik_indices = [_ for _ in range(len(agent.right_arm_ik_indices))]
            pos, orient = agent.get_pos_orient(joint)

            # eef translation
            if self.translation_mode == 'delta-translation':
                pos += translation * self.max_translation
            elif self.translation_mode == 'normalized-direct-translation':
                pos = translation * self.scene_range + self.scene_center
            elif self.translation_mode == 'direct-translation':
                pos = translation 

            # eef rotation
            if self.rotation_mode == 'euler-angle':
                rotation = rotation * np.pi
                orient = p.getQuaternionFromEuler(rotation)
            elif 'delta-axis-angle' in self.rotation_mode or 'delta-euler-angle' in self.rotation_mode:
                orient = self.apply_delta_rotation(rotation, orient)

            agent_joint_angles = agent.ik(joint, pos, orient, ik_indices, max_iterations=5000)
            for _ in range(self.control_step):
                agent.control(agent.controllable_joint_indices, agent_joint_angles)

                # gripper
                if not self.use_suction:
                    # if suction >= 0:
                    #     agent.set_gripper_open_position(agent.right_gripper_indices, [0.04, 0.04], set_instantly=True)
                    # else:
                    #     agent.set_gripper_open_position(agent.right_gripper_indices, [0, 0], set_instantly=True)
                    cur_joint_angle = p.getJointState(self.robot.body, self.robot.right_gripper_indices[0], physicsClientId=self.id)[0]
                    new_joint_angle = cur_joint_angle + suction * 0.02
                    new_joint_angle = np.clip(new_joint_angle, 0, 0.04)
                    agent.set_gripper_open_position(agent.right_gripper_indices, [new_joint_angle, new_joint_angle], set_instantly=False)
                else:
                    if suction >= 0: self.activate_suction()
                    else: self.deactivate_suction()
                    
                p.stepSimulation(physicsClientId=self.id) 
                
        # self.enforce_joint_limits()
        
    def enforce_joint_limits(self):
        # for every articulated object, reset joint angle to be within joint limits
        for name in self.urdf_ids:
            if name == 'robot' or name == 'plane' or name == "init_table": continue
            # if it is articulated
            if self.urdf_types[name] == 'urdf' and not self.is_distractor[name]:
                num_joints = p.getNumJoints(self.urdf_ids[name], physicsClientId=self.id)
                for joint_idx in range(num_joints):
                    joint_state = p.getJointState(self.urdf_ids[name], joint_idx, physicsClientId=self.id)
                    joint_angle = joint_state[0]
                    joint_limit_low, joint_limit_high = p.getJointInfo(self.urdf_ids[name], joint_idx, physicsClientId=self.id)[8:10]
                    if joint_limit_low > joint_limit_high:
                        joint_limit_low, joint_limit_high = joint_limit_high, joint_limit_low
                    joint_angle = np.clip(joint_angle, joint_limit_low, joint_limit_high)
                    p.resetJointState(self.urdf_ids[name], joint_idx, joint_angle, physicsClientId=self.id)
                    for _ in range(5):
                        p.stepSimulation(physicsClientId=self.id)

    def apply_delta_rotation(self, delta_rotation, orient):
        if 'delta-axis-angle' in self.rotation_mode:
            dtheta = np.linalg.norm(delta_rotation)
            if dtheta > 0:
                delta_rotation = delta_rotation / dtheta
                dtheta = dtheta * self.max_rotation_angle / np.sqrt(3)
                delta_rotation_matrix = R.from_rotvec(delta_rotation * dtheta).as_matrix()
            else:
                delta_rotation_matrix = np.eye(3)
            current_matrix = np.array(p.getMatrixFromQuaternion(orient)).reshape(3, 3)

            if self.rotation_mode == 'delta-axis-angle-local':
                new_rotation = current_matrix @ delta_rotation_matrix
            elif self.rotation_mode == 'delta-axis-angle-global':
                new_rotation = delta_rotation_matrix @ current_matrix
            orient = R.from_matrix(new_rotation).as_quat()
        elif self.rotation_mode == 'delta-euler-angle':
            euler_angle = delta_rotation / np.sqrt(3) * self.max_rotation_angle
            delta_quaternion = p.getQuaternionFromEuler(euler_angle)
            orient = delta_quaternion * orient
            
        return orient
    

    def activate_suction(self):
        if not self.activated:
            # assume the suction is attached to the right end effector
            suction_id = self.suction_id
            points = p.getContactPoints(bodyA=self.robot.body, linkIndexA=suction_id, physicsClientId=self.id)
            if points:
                # Handle contact between suction with a rigid object.
                contact_object_id_link_cnts = defaultdict(int)
                for point in points:
                    obj_id, contact_link, contact_position_on_obj = point[2], point[4], point[6]
                    
                    if obj_id == self.urdf_ids['plane'] or obj_id == self.robot.body:
                        pass
                    else:
                        contact_object_id_link_cnts[(obj_id, contact_link)] += 1
                
                if len(contact_object_id_link_cnts) > 0:
                    # find the object that has the most contact points
                    obj_id, contact_link = max(contact_object_id_link_cnts.items(), key=lambda x: x[1])[0]
                    # print("contact with object: ", obj_id, contact_link)

                    suction_pose = p.getLinkState(self.robot.body, suction_id, physicsClientId=self.id)
                    if contact_link >= 0:
                        obj_link_pose = p.getLinkState(obj_id, contact_link, physicsClientId=self.id)
                    else:
                        obj_link_pose = p.getBasePositionAndOrientation(obj_id, physicsClientId=self.id)
                    world_to_suction = p.invertTransform(suction_pose[0], suction_pose[1])
                    obj_to_suction = p.multiplyTransforms(world_to_suction[0],
                                                        world_to_suction[1],
                                                        obj_link_pose[0], obj_link_pose[1])
                    
                    suction_to_obj = p.invertTransform(obj_to_suction[0], obj_to_suction[1])
                    
                    self.create_suction_constraint(obj_id, contact_link, suction_to_obj)
                    
                    self.activated = True
                    self.suction_obj_id = obj_id
                    self.suction_contact_link = contact_link
                    self.suction_to_obj_pose = suction_to_obj

    def create_suction_constraint(self, suction_obj_id, suction_contact_link, suction_to_obj_pose):
        suction_id = self.suction_id
        self.contact_constraint = p.createConstraint(
            parentBodyUniqueId=self.robot.body,
            parentLinkIndex=suction_id,
            childBodyUniqueId=suction_obj_id,
            childLinkIndex=suction_contact_link,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0), 
            parentFramePosition=(0, 0, 0),
            parentFrameOrientation=(0, 0, 0),
            childFramePosition=suction_to_obj_pose[0],
            childFrameOrientation=suction_to_obj_pose[1], 
            physicsClientId=self.id)
        
        p.changeConstraint(self.contact_constraint, maxForce=5000, physicsClientId=self.id)

    def deactivate_suction(self):
        self.activated = False
        if self.contact_constraint is not None:
            p.removeConstraint(self.contact_constraint, physicsClientId=self.id)
            self.contact_constraint = None


    def step(self, action):
        self.time_step += 1        
        self.take_step(action)
        obs = self._get_obs()                
        # to handle some stupid typing error in early prompts
        try:
            reward, success = self._compute_reward() 
        except:
            reward, success = self.compute_reward()
        self.success = success
        done = self.time_step == self.horizon
        info = self._get_info()
        return obs, reward, done, info

    def compute_reward(self):
        return 0, 0

    def _get_info(self):
        # TODO: this should be implemented by GPT
        object_name = 'storagefurniture'
        if self.handle_joint is None:
            # all_handle_pos, handle_joint_id = get_handle_pos(self, object_name, return_median=False)
            # handle_median_points = np.array([np.median(handle_pos, axis=0) for handle_pos in all_handle_pos]).reshape(-1, 3)
            # link_name = "link_0"
            # link_name = link_name.lower()
            # link_pc = get_link_pc(self, object_name, link_name)
            # distance_handle_median_to_link_pc = scipy.spatial.distance.cdist(handle_median_points, link_pc)
            # min_distance = np.min(distance_handle_median_to_link_pc, axis=1)
            # min_distance_handle_idx = np.argmin(min_distance)
            # handle_joint = handle_joint_id[min_distance_handle_idx]
            # self.handle_joint = handle_joint
            # handle_pos = handle_median_points[min_distance_handle_idx]

            all_handle_pos, all_handle_joint_id, handle_pts_obj_frame, mobility_info = get_handle_pos(self, object_name, return_median=False, return_info=True)
            self.handle_pts_obj_frame = handle_pts_obj_frame
            self.mobility_info = mobility_info
            link_name = "link_0"
            link_pc = get_link_pc(self, object_name, link_name)
            _, link_handle_joint_id, link_handle_median, min_link_idx = get_link_handle(all_handle_pos, all_handle_joint_id, link_pc)
            self.handle_joint = link_handle_joint_id
            self.handle_pos = link_handle_median
            self.min_link_idx = min_link_idx
            self.all_handle_points = all_handle_pos[min_link_idx]
        else:
            all_handle_pos, _ = get_handle_pos(self, object_name, return_median=False, handle_pts_obj_frame=self.handle_pts_obj_frame, mobility_info=self.mobility_info)
            handle_median_points = np.array([np.median(handle_pos, axis=0) for handle_pos in all_handle_pos]).reshape(-1, 3)
            self.handle_pos = handle_median_points[self.min_link_idx]
            self.all_handle_points = all_handle_pos[self.min_link_idx]
            
        opened_joint_angle = p.getJointState(self.urdf_ids[object_name], self.handle_joint, physicsClientId=self.id)[0]
        if self.init_joint_angle is None:
            self.init_joint_angle = opened_joint_angle
            
        cur_eef_pos, cur_eef_orient = self.robot.get_pos_orient(self.robot.right_end_effector)
        handle_points = self.all_handle_points
        num_handle_points_within_gripper = get_pc_num_within_gripper(cur_eef_pos, cur_eef_orient, handle_points)
        # cprint("num_handle_points_within_gripper: {}".format(num_handle_points_within_gripper), "red")
        distance_eef_to_handle = np.linalg.norm(self.handle_pos.flatten() - cur_eef_pos.flatten())
        if num_handle_points_within_gripper > 0:
            # left_finger_pos, _ = self.robot.get_pos_orient(self.robot.right_gripper_indices[0])
            # right_finger_pos, _ = self.robot.get_pos_orient(self.robot.right_gripper_indices[1])
            # distance_left = np.linalg.norm(handle_points - left_finger_pos.reshape(1, 3), axis=1)
            # distance_right = np.linalg.norm(handle_points - right_finger_pos.reshape(1, 3), axis=1)
            points_left_finger = p.getContactPoints(bodyA=self.robot.body, linkIndexA=self.robot.right_gripper_indices[0], physicsClientId=self.id)
            points_right_finger = p.getContactPoints(bodyA=self.robot.body, linkIndexA=self.robot.right_gripper_indices[1], physicsClientId=self.id)
            if len(points_left_finger) > 0 and len(points_right_finger) > 0:
                contact_points_left = np.array([point[6] for point in points_left_finger])
                contact_points_right = np.array([point[6] for point in points_right_finger])
                left_distance = scipy.spatial.distance.cdist(handle_points, contact_points_left)
                right_distance = scipy.spatial.distance.cdist(handle_points, contact_points_right)
                min_distance_left = np.min(left_distance)
                min_distance_right = np.min(right_distance)
                # if min_distance_left < 0.015 and min_distance_right < 0.015:
                if min_distance_left < 0.01 or min_distance_right < 0.01:
                    grasped_handle = True
                    self.grasped_handle = self.grasped_handle or grasped_handle
        
        right_finger_pos, _ = self.robot.get_pos_orient(self.robot.right_gripper_indices[0])
        left_finger_pos, _ = self.robot.get_pos_orient(self.robot.right_gripper_indices[1])
        finger_distance = np.linalg.norm(right_finger_pos - left_finger_pos)
        
        return {
            "opened_joint_angle": opened_joint_angle,
            "improved_joint_angle": opened_joint_angle - self.init_joint_angle,
            "handle_pos": self.handle_pos, 
            "initial_joint_angle": self.init_joint_angle,
            "ik_failure": self.ik_failure,
            "grasped_handle": self.grasped_handle,
            "finger_distance": finger_distance, 
        }

    def _get_obs(self):
        ### For RL policy learning, observation space includes:
        # 1. object positions and orientations (6 * num_objects)
        # 2. object min and max bounding box (6 * num_objects)
        # 3. articulated object joint angles (num_objects * num_joints) 
        # 4. articulated object link position and orientation (num_objects * num_joints * 6) 
        # 5. robot base position (xy)
        # 6. robot end-effector position and orientation (6)
        # 7. gripper suction activated/deactivate or gripper joint angle (if not using suction gripper) (1)
        obs = np.zeros(self.base_observation_space.shape[0])
            
        cnt = 0
        for name, id in self.urdf_ids.items():
            if name == 'plane' or name == 'robot':
                continue
            if self.is_distractor[name]:
                continue

            pos, orient = p.getBasePositionAndOrientation(id, physicsClientId=self.id)
            euler_angle = p.getEulerFromQuaternion(orient)
            obs[cnt:cnt+3] = self.normalize_position(pos)
            obs[cnt+3:cnt+6] = euler_angle
            cnt += 6

        for name, id in self.urdf_ids.items():
            if name == 'plane' or name == 'robot':
                continue
            if self.is_distractor[name]:
                continue
            min_aabb, max_aabb = self.get_aabb(id)
            obs[cnt:cnt+3] = self.normalize_position(min_aabb)
            obs[cnt+3:cnt+6] = self.normalize_position(max_aabb)
            cnt += 6

        for name in self.urdf_types:
            if self.urdf_types[name] == 'urdf' and not self.is_distractor[name]:
                num_joints = p.getNumJoints(self.urdf_ids[name], physicsClientId=self.id)
                for joint_idx in range(num_joints):
                    joint_angle = p.getJointState(self.urdf_ids[name], joint_idx, physicsClientId=self.id)[0]
                    obs[cnt] = joint_angle
                    cnt += 1
                    link_pos, link_orient = p.getLinkState(self.urdf_ids[name], joint_idx, physicsClientId=self.id)[:2]
                    link_pos = self.normalize_position(link_pos)
                    link_euler_angle = p.getEulerFromQuaternion(link_orient)
                    obs[cnt:cnt+3] = link_pos
                    obs[cnt+3:cnt+6] = link_euler_angle
                    cnt += 6

        robot_base_pos, robot_base_orient = self.robot.get_base_pos_orient()
        robot_base_pos = self.normalize_position(robot_base_pos)
        obs[cnt:cnt+2] = robot_base_pos[:2]
        cnt += 2

        robot_eef_pos, robot_eef_orient = self.robot.get_pos_orient(self.robot.right_end_effector)
        robot_eef_euler_angle = p.getEulerFromQuaternion(robot_eef_orient)
        obs[cnt:cnt+3] = self.normalize_position(robot_eef_pos)
        obs[cnt+3:cnt+6] = robot_eef_euler_angle
        cnt += 6

        if not self.use_suction:
            # get joint angle of the gripper
            left_finger_joint_angle = p.getJointState(self.robot.body, self.robot.right_gripper_indices[0], physicsClientId=self.id)[0]
            right_finger_joint_angle = p.getJointState(self.robot.body, self.robot.right_gripper_indices[1], physicsClientId=self.id)[0]
            obs[cnt] = left_finger_joint_angle
            cnt += 1
        else:
            obs[cnt] = int(self.activated)
            cnt += 1

        return obs

    def disconnect(self):
        p.disconnect(self.id)

    def close(self):
        p.disconnect(self.id)
    
    
if __name__ == "__main__":
    from manipulation.utils import build_up_env
    env, safe_config = build_up_env(
        # "example_tasks/Change_Lamp_Direction/Change_Lamp_Direction_The_robotic_arm_will_alter_the_lamps_light_direction_by_manipulating_the_lamps_head.yaml",
        # "example_tasks/Change_Lamp_Direction/task_Change_Lamp_Direction",
        # "grasp_the_lamps_head", 
        # "data/generated_task_from_description/put_a_bottle_in_microwave/put_a_bottle_in_microwave.yaml",
        # "data/generated_task_from_description/put_a_bottle_in_microwave/task_put_a_bottle_in_microwave",
        # "put_a_bottle_in_the_microwave",
        # "initial_states/bottle_lift_initialization_not_close.pkl",
        "data/generated_task_from_description/lift_a_box/lift_a_box.yaml",
        "data/generated_task_from_description/lift_a_box/task_lift_a_box",
        "lift_a_box",
        "initial_states/gold_bar_initialization_not_close.pkl",
        # "data/generated_task_from_description/lift_a_hamburger/lift_a_hamburger.yaml",
        # "data/generated_task_from_description/lift_a_hamburger/task_lift_a_hamburger",
        # "lift_a_hamburger",
        # "initial_states/hamburger_lift_initialization_not_close.pkl",
        # None,
        render=True, 
        randomize=False, 
        obj_id=0
    )
    # save_env(env, "initial_states/closer/gold_bar_lift_initialization_not_close.pkl")
    input("Press Enter to continue...")