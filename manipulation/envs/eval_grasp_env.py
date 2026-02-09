import pybullet as p
import os
import numpy as np
import fpsample
import time
import pickle
from scipy.spatial.transform import Rotation as R
from termcolor import cprint
import os
import sys
import argparse
import time
import torch
from matplotlib import pyplot as plt
import h5py
import open3d as o3d
import trimesh
from collections import defaultdict
import json
from moviepy.editor import ImageSequenceClip
import cv2

def replace_obj_path(urdf_name):
    file_path = urdf_name  # Replace with your actual file name

    old_str = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/contact_graspnet/acronym/"
    new_str = "/project_data/held/yufeiw2/contact_graspnet_pytorch/acronym/"

    # Read and replace
    with open(file_path, "r") as f:
        content = f.read()

    content = content.replace(old_str, new_str)

    # Overwrite the file with new content
    with open(file_path.replace(".urdf", "autobot.urdf"), "w") as f:
        f.write(content)

def save_numpy_as_gif(array, filename, fps=15, scale=1.0):
    """Creates a gif given a stack of images using moviepy
    Notes
    -----
    works with current Github version of moviepy (not the pip version)
    https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e
    Usage
    -----
    >>> X = randn(100, 64, 64)
    >>> gif('test.gif', X)
    Parameters
    ----------
    filename : string
        The filename of the gif to write to
    array : array_like
        A numpy array that contains a sequence of images
    fps : int
        frames per second (default: 10)
    scale : float
        how much to rescale each image by (default: 1.0)
    """

    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname + '.gif'

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    clip.write_gif(filename, fps=fps)
    return clip

def get_pc_in_camera_and_world_frame(proj_matrix, view_matrix, depth, width, height, mask_infinite=False):
    proj_matrix = np.asarray(proj_matrix).reshape([4, 4], order="F")
    view_matrix = np.asarray(view_matrix).reshape([4, 4], order="F")
    tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

    # create a grid with pixel coordinates and depth values
    y, x = np.mgrid[-1:1:2 / height, -1:1:2 / width]
    y *= -1.
    x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
    h = np.ones_like(z)

    pixels = np.stack([x, y, z, h], axis=1)
    # filter out "infinite" depths
    if mask_infinite:
        pixels = pixels[z < 0.99]
    pixels[:, 2] = 2 * pixels[:, 2] - 1
    # turn pixels to camera cooridnates
    points = np.matmul(np.linalg.inv(proj_matrix), pixels.T).T
    points /= points[:, 3: 4]
    points = points[:, :3]
    
    points_world = np.matmul(tran_pix_world, pixels.T).T
    points_world /= points_world[:, 3: 4]
    points_world = points_world[:, :3]
    return points, points_world

def run_vhacd(obj_file_path):
    p.connect(p.DIRECT)
    name_in = obj_file_path
    name_out = obj_file_path.replace(".obj", "_vhacd.obj")
    name_log = obj_file_path.replace(".obj", "_vhacd.log")
    p.vhacd(name_in, name_out, name_log)
    
def obj_to_urdf(obj_file_path, scale=1, mass=1):
    # NOTE: maybe I should compute the inertia and mass from the obj file, but for now I just use a fixed value
    
    #     <contact>
    #   <lateral_friction value="1.0"/>
    #   <rolling_friction value="0.0"/>
    #   <contact_cfm value="0.0"/>
    #   <contact_erp value="1.0"/>
    # </contact>
    
    header = """<?xml version="1.0" ?>
<robot name="cube.urdf">
  <link name="baseLink">
    <inertial>
      <origin rpy="0 0 0" xyz="0.0 0.0 0.0"/>
       <mass value="{}"/>
       <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
""".format(mass)

    material = """
        <material name="yellow">
            <color rgba="1 1 0.4 1"/>
        </material>
    """

    visual = """
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="{}" scale="{} {} {}"/>
      </geometry>
      {}
    </visual>
    """.format(obj_file_path, scale, scale, scale, material)

    collision = """
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
             <mesh filename="{}" scale="{} {} {}"/>
      </geometry>
    </collision>
  </link>
  </robot>
  """.format(obj_file_path, scale, scale, scale)

    urdf =  "".join([header, visual, collision])
    save_name = obj_file_path.replace(".obj", ".urdf")
    with open(save_name, 'w') as f:
        f.write(urdf)
        
def normalize_obj(obj_file_path):
    vertices = []
    with open(os.path.join(obj_file_path), 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:]])
    
    vertices = np.array(vertices).reshape(-1, 3)
    vertices = vertices - np.mean(vertices, axis=0) # center to zero

    with open(os.path.join(obj_file_path.replace(".obj", "_normalized.obj")), 'w') as f:
        vertex_idx = 0
        for line in lines:
            if line.startswith("v "):
                line = "v " + " ".join([str(x) for x in vertices[vertex_idx]]) + "\n"
                vertex_idx += 1
            f.write(line)
            
def fill_mesh(mesh_path, obj_scale):
    # import trimesh
    # import numpy as np
    # from skimage import measure

    # # Load your original thin mesh
    # mesh = trimesh.load(mesh_path)
    # if not isinstance(mesh, trimesh.Trimesh):
    #     mesh = mesh.dump().sum()  # If it's a scene, flatten to one mesh

    # # Define voxel size (resolution)
    # # voxel_pitch = 0.0025 / obj_scale  # Smaller = higher resolution
    # bounding_box = mesh.bounds
    # # padding = 0.01 / obj_scale
    # range = bounding_box[1] - bounding_box[0]
    # voxel_pitch = np.max(range) / 100.  # Ensure voxel pitch is not too small
    # padding = voxel_pitch * 2
    # min_bound = bounding_box[0] - padding
    # max_bound = bounding_box[1] + padding
    # grid_size = np.ceil((max_bound - min_bound) / voxel_pitch).astype(int)
    # # import pdb; pdb.set_trace()

    # # Create grid of 3D points
    # beg = time.time()
    # grid_x, grid_y, grid_z = np.mgrid[
    #     min_bound[0]:max_bound[0]:complex(grid_size[0]),
    #     min_bound[1]:max_bound[1]:complex(grid_size[1]),
    #     min_bound[2]:max_bound[2]:complex(grid_size[2])
    # ]
    # grid_points = np.vstack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel())).T

    # # Compute SDF: signed distances from mesh surface
    # sdf = mesh.nearest.signed_distance(grid_points)
    # sdf_grid = sdf.reshape(grid_x.shape)

    # # Run marching cubes to extract mesh where SDF=0 (surface)
    # verts, faces, _, _ = measure.marching_cubes(sdf_grid, level=0.0, spacing=(voxel_pitch,) * 3)
    # verts += min_bound  # translate back to world coords

    # # Build new mesh
    # filled_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    # filled_mesh.export(mesh_path.replace("_vhacd.obj", "_vhacd_fill_inside.obj"))
    # end = time.time()
    # cprint("Filled mesh in {:.2f} seconds".format(end - beg), "green")
    
    import trimesh
    import numpy as np
    from skimage import measure

    # Load the original mesh
    mesh = trimesh.load(mesh_path)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump().sum()

    # Voxelize the mesh with solid (filled) interior
    bounding_box = mesh.bounds
    range = bounding_box[1] - bounding_box[0]
    voxel_pitch = np.max(range) / 100.  # Ensure voxel pitch is not too small
    # voxel_pitch = 0.005  # You can adjust this for resolution
    vox = mesh.voxelized(pitch=voxel_pitch)
    vox_filled = vox.fill()  # Important: fills interior voxels

    # Extract dense 3D occupancy grid
    matrix = vox_filled.matrix  # Binary: True = filled voxel
    origin = vox_filled.origin  # World coordinate of voxel (0, 0, 0)

    # Apply marching cubes to binary occupancy grid
    verts, faces, _, _ = measure.marching_cubes(matrix.astype(np.float32), level=0.5)

    # Transform voxel coordinates back to world space
    verts = verts * voxel_pitch + origin

    # Create and export new mesh
    filled_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    filled_mesh.export(mesh_path.replace("_vhacd.obj", "_vhacd_fill_inside.obj"))

class ContactGraspNetEnv():
    def __init__(self, scene_path,
                 single_obj_mesh_path=None, 
                 gui=False,  dt=1/240, num_points_in_pc=20000, 
                 env_state=None,
                 camera_angle_nums=50,
                 precontact=0,
                 obs_mode='cgn',
                 act_mode='cgn',
                ):
        
        self.scene_root_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/contact_graspnet/acronym"
        self.scene_path = scene_path
        self.num_points_in_pc = num_points_in_pc
        self.gui = gui
        self.camera_angle_nums = camera_angle_nums
        self.precontact = precontact
        self.obs_mode = obs_mode
        self.act_mode = act_mode
        
        ### initialize pybullet
        if self.gui:
            try:
                self.id = p.connect(p.GUI)
            except:
                self.id = p.connect(p.DIRECT)
        else:
            self.id = p.connect(p.DIRECT)
            
        self.gravity = -9.81
        p.setTimeStep(dt, physicsClientId=self.id)
        
        p.resetSimulation(physicsClientId=self.id)
        if self.gui:
            p.resetDebugVisualizerCamera(cameraDistance=1.75, cameraYaw=-25, cameraPitch=-45, cameraTargetPosition=[-0.2, 0, 0.4], physicsClientId=self.id)
            p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=self.id)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.id)
        p.setRealTimeSimulation(0, physicsClientId=self.id)
        p.setGravity(0, 0, self.gravity, physicsClientId=self.id)
        res = p.getPhysicsEngineParameters(physicsClientId=self.id)
        # print(res)
        # p.setPhysicsEngineParameter(numSubSteps=10, contactERP=0.7, contactSlop=0.005, numSolverIterations=50, physicsClientId=self.id)
        res = p.getPhysicsEngineParameters(physicsClientId=self.id)
        # print(res)
        # import pdb; pdb.set_trace()
        
        ### load plane 
        this_file_path = os.path.dirname(os.path.abspath(__file__))
        parent_path = os.path.dirname(this_file_path)
        planeId = p.loadURDF(os.path.join(parent_path, "assets", "plane", "plane.urdf"), physicsClientId=self.id)

        ### create and load a robot
        panda_eef_id = self.load_robot()
        self.urdf_ids = {
            "robot": panda_eef_id,
            "plane": planeId,
        }
        
        ### load scene
        if single_obj_mesh_path is None:
            self.load_scene()
        else:
            self.mesh_path = single_obj_mesh_path
            self.load_single_obj()

        friction = 1.5
        # for object_name in self.urdf_ids:
        #     if object_name == "robot": continue
        #     if object_name == "plane": continue
        #     p.changeDynamics(self.urdf_ids[object_name], -1, lateralFriction=friction, physicsClientId=self.id)
        #     p.changeDynamics(self.urdf_ids[object_name], -1, rollingFriction=friction, physicsClientId=self.id)
        #     p.changeDynamics(self.urdf_ids[object_name], -1, spinningFriction=friction, physicsClientId=self.id)

        p.changeDynamics(self.urdf_ids['robot'], self.panda_joints['left_finger'], lateralFriction=friction, physicsClientId=self.id)
        p.changeDynamics(self.urdf_ids['robot'], self.panda_joints['right_finger'], lateralFriction=friction, physicsClientId=self.id)
        p.changeDynamics(self.urdf_ids['robot'], self.panda_joints['left_finger'], rollingFriction=friction, physicsClientId=self.id)
        p.changeDynamics(self.urdf_ids['robot'], self.panda_joints['right_finger'], rollingFriction=friction, physicsClientId=self.id)
        p.changeDynamics(self.urdf_ids['robot'], self.panda_joints['left_finger'], spinningFriction=friction, physicsClientId=self.id)
        p.changeDynamics(self.urdf_ids['robot'], self.panda_joints['right_finger'], spinningFriction=friction, physicsClientId=self.id)
        
        if env_state is None:
            # cprint("steping simulation to stablize", "green")
            for _ in range(4000):
            # for _ in range(10):
                p.stepSimulation(physicsClientId=self.id)
            # cprint("simulation stabilized", "green")
            
            self.stablized_state = {}
            for key in self.urdf_ids.keys():
                if key == "robot": continue
                self.stablized_state[key] = p.getBasePositionAndOrientation(self.urdf_ids[key], physicsClientId=self.id)
        else:
            self.reset_object(env_state)
            
        # p.setGravity(0, 0, 0, physicsClientId=self.id)        
        ### set camera
        self.set_camera()
        
    def get_robot_init_joint_angles(self, robot_init_joint_angles=None):
        if robot_init_joint_angles is None:
            init_joint_angles = [0 for _ in range(len(self.robot.right_arm_joint_indices))]

            init_joint_angles[3] = -0.4
            init_joint_angles[5] = 0.4
            return init_joint_angles  
        return robot_init_joint_angles
        
    def load_robot(self):
        ## NOTE: load a floating panda gripper  
        this_file_path = os.path.dirname(os.path.abspath(__file__))
        parent_path = os.path.dirname(this_file_path)
        panda_eef = p.loadURDF(os.path.join(parent_path, "assets", "panda_bullet", "panda_eef_floating.urdf"), basePosition=[-5, -5, 2], physicsClientId=self.id, useFixedBase=True)
        
        # num_joints = p.getNumJoints(panda_eef, physicsClientId=self.id)
        # for joint_index in range(num_joints):
        #     joint_info = p.getJointInfo(panda_eef, joint_index, physicsClientId=self.id)
        #     print(joint_info)
        
        self.panda_joints = {
            "z_joint": 2,
            "hand": 6,
            "left_finger": 7,
            "right_finger": 8,
        }
        p.resetJointState(panda_eef, self.panda_joints['left_finger'], 0.04, physicsClientId=self.id)
        p.resetJointState(panda_eef, self.panda_joints['right_finger'], 0.04, physicsClientId=self.id)
            
        return panda_eef
    
    def load_single_obj(self):
        filename = self.mesh_path
        data = h5py.File(filename, "r")
        mesh_fname = data["object/file"][()].decode('utf-8')
        mesh_scale = data["object/scale"][()]
        mesh_fname = "data/debug/examples/" + mesh_fname
        
        vhacd_fname = mesh_fname.replace(".obj", "_vhacd.obj")
        urdf_fname = vhacd_fname.replace(".obj", ".urdf")
        if not os.path.exists(mesh_fname):
            ### first run vhacd to the obj file
            run_vhacd(mesh_fname)
        if not os.path.exists(vhacd_fname):
            obj_to_urdf(vhacd_fname, scale=mesh_scale)
        
        obj_id = p.loadURDF(urdf_fname, basePosition=[0.0, 0, 0.3], baseOrientation=[0, 0, 0, 1], useFixedBase=True, physicsClientId=self.id)
        min_aabb, max_aabb = p.getAABB(obj_id, physicsClientId=self.id)
        
        self.urdf_ids['obj'] = obj_id
        self.scene_min_aabb = np.array(min_aabb)
        self.scene_max_aabb = np.array(max_aabb)
        
    def load_scene(self):
        inp = np.load(os.path.join(self.scene_root_path, self.scene_path))
        scene_filtered_grasps = inp['grasp_transforms']
        scene_contacts = inp['scene_contact_points']
        obj_transforms = inp['obj_transforms']
        obj_paths = inp['obj_paths']
        obj_scales = inp['obj_scales']
        
        contact_directions_01 = scene_contacts[:,0,:] - scene_contacts[:,1,:]
        all_finger_diffs = np.maximum(np.linalg.norm(contact_directions_01,axis=1), np.finfo(np.float32).eps)
        
        self.scene_filtered_grasps = scene_filtered_grasps
        self.all_finger_widths = all_finger_diffs
        
        min_aabbs = []
        max_aabbs = []
        for obj_path,obj_transform,obj_scale in zip(obj_paths,obj_transforms,obj_scales):
            real_obj_path = os.path.join(self.scene_root_path, obj_path)
            # obj_mesh = obj_mesh.apply_scale(obj_scale)
            # mesh_mean =  np.mean(obj_mesh.vertices, 0, keepdims=True)
            # obj_mesh.vertices -= mesh_mean
            obj_id = self.add_object(real_obj_path, obj_scale, obj_transform)
            if obj_id:
                self.urdf_ids[obj_path] = obj_id
                
                min_aabb, max_aabb = p.getAABB(obj_id, physicsClientId=self.id)
                min_aabbs.append(min_aabb)
                max_aabbs.append(max_aabb)

        self.scene_min_aabb = np.array(min_aabbs).min(axis=0)
        self.scene_max_aabb = np.array(max_aabbs).max(axis=0)
        
        # NOTE: reload the plane such that it is exactly at the bottom of all objects
        min_height = self.scene_min_aabb[2]
        # cprint("min height of the scene is: {}".format(min_height), "green")
        p.resetBasePositionAndOrientation(self.urdf_ids["plane"], [0, 0, min_height], [0, 0, 0, 1], physicsClientId=self.id)
        
        
    def add_object(self, obj_path, obj_scale, obj_transform):
        ### normalize obj mesh
        # import pdb; pdb.set_trace()
        if not os.path.exists(obj_path):
            return False
        
        normalize_obj(obj_path)
        
        ### turn obj to urdf. 
        obj_path = obj_path.replace(".obj", "_normalized.obj")
        vhacd_fname = obj_path.replace(".obj", "_vhacd.obj")
        if not os.path.exists(vhacd_fname):
            ### first run vhacd to the obj file
            run_vhacd(obj_path)

        ### get the correct obj mass assuming it has a uniform density (density of water)
        # Load the mesh from the .obj file
        mesh = trimesh.load(vhacd_fname, force='mesh')
        mesh.apply_scale(obj_scale)  # Apply the scale to the mesh
        # Check if the mesh is watertight; inertia is only reliable for watertight meshes
        # if not mesh.is_volume:
        #     print("Warning: Mesh is not watertight. Inertia may be inaccurate.")
        volume = mesh.volume
        density = 600 ## water density is 1000
        mass = volume * density 
        # print(f"{obj_path} volume {volume} mass {mass}")
        
        # fill_inside_fname = obj_path.replace(".obj", "_vhacd_fill_inside.obj")
        # print("Filling the mesh to make it solid...")
        # if not os.path.exists(fill_inside_fname):
        #     ### then fill the mesh to make it solid
        #     fill_mesh(vhacd_fname, obj_scale)
        
        # urdf_fname = fill_inside_fname.replace(".obj", ".urdf")
        urdf_fname = vhacd_fname.replace(".obj", ".urdf")
        if not os.path.exists(urdf_fname):
            ### then convert it to urdf  with the right scale
            obj_to_urdf(vhacd_fname, scale=obj_scale, mass=mass)
            

        
        ### load obj and apply transform
        pos = obj_transform[:3, 3]
        orn = R.from_matrix(obj_transform[:3, :3]).as_quat()
        
        # replace_obj_path(urdf_fname)
        # urdf_fname = urdf_fname.replace(".urdf", "autobot.urdf")
        # print(urdf_fname)
        # # import pdb; pdb.set_trace()
        obj_id = p.loadURDF(urdf_fname, basePosition=pos, baseOrientation=orn, useFixedBase=False, physicsClientId=self.id)

        return obj_id
        
    def setup_camera(self, camera_eye=[0.5, -0.75, 1.5], camera_target=[-0.2, 0, 0.75], fov=60, camera_width=640, camera_height=480):
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.view_matrix = p.computeViewMatrix(camera_eye, camera_target, [0, 0, 1], physicsClientId=self.id)
        self.projection_matrix = p.computeProjectionMatrixFOV(fov, camera_width / camera_height, 0.01, 1000, physicsClientId=self.id)
        
    def set_camera(self, camera_width=640, camera_height=480, distance_ratio=1.5):
        ### randomly sample a camera pose and render a depth image.
        ### return the depth and the camera pose
        
        scene_range = (self.scene_max_aabb - self.scene_min_aabb) / 2
        scene_center = (self.scene_max_aabb + self.scene_min_aabb) / 2
        
        if self.obs_mode == 'cgn':
            camera_target = np.random.uniform(scene_center - scene_range * 0.6, scene_center + scene_range * 0.6)
            distance = np.random.uniform(1, 2) * np.linalg.norm(scene_range)
            elevation = np.random.uniform(20, 60)  # elevation angle in degrees
            azimuth = np.random.uniform(0, 360)  # azimuth angle in degrees
            camera_target = scene_center
            distance = distance_ratio * np.linalg.norm(scene_range)
            elevation = 50
            azimuth = 0
            
            delta_z = distance * np.sin(np.deg2rad(elevation))
            xy_distance = distance * np.cos(np.deg2rad(elevation))
            delta_x = xy_distance * np.cos(np.deg2rad(azimuth))
            delta_y = xy_distance * np.sin(np.deg2rad(azimuth))
            
            camera_position = [camera_target[0] + delta_x, camera_target[1] + delta_y, camera_target[2] + delta_z]
            self.setup_camera(camera_position, camera_target, camera_width=camera_width, camera_height=camera_height)
        
        elif self.obs_mode == 'm2t2':
            self.rpy_mean_list = [[-10, 0, -45], [-10, 0, -135]]
            self.mean_distance = np.linalg.norm(self.scene_max_aabb - self.scene_min_aabb) * 0.9
            self.camera_height = 256
            self.camera_width = 256

            self.depth_near = 0.01
            self.depth_far = 100
            self.view_matrices = []
            self.project_matrices = []

            self.mean_camera_target = scene_center 
            self.mean_distance = np.linalg.norm(scene_range * 2) * 1.15

            for rpy_mean in self.rpy_mean_list:
                rpy = np.array(rpy_mean)
                camera_center = self.mean_camera_target
                distance = self.mean_distance

                view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=camera_center, distance=distance, yaw=rpy[2], pitch=rpy[0], roll=rpy[1], upAxisIndex=2, physicsClientId=self.id)
                project_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1 ,nearVal=self.depth_near, farVal=self.depth_far, physicsClientId=self.id)
                self.view_matrices.append(view_matrix)
                self.project_matrices.append(project_matrix)
                
            self.view_matrix = self.view_matrices[0]
            self.projection_matrix = self.project_matrices[0]
        
    def get_obs(self):
        if self.obs_mode == 'cgn':
            rgb, depth = self.render(return_depth=True, mode='depth')
            depth = self.augment_depth(depth)
            pc_in_camera, pc_in_world = get_pc_in_camera_and_world_frame(self.projection_matrix, self.view_matrix, depth, self.camera_width, self.camera_height)
            lower_bound = self.scene_min_aabb - 0.3
            upper_bound = self.scene_max_aabb + 0.3
            good_idx = np.logical_and(np.all(pc_in_world >= lower_bound, axis=1), np.all(pc_in_world <= upper_bound, axis=1))
            
            pc_in_camera = pc_in_camera[good_idx]
                        
            # Convert point cloud coordinates from OpenGL to internal coordinates (x left, y up, z front)
            # openGL: x right, y up, z back
            pc_in_camera[:, 0] = -pc_in_camera[:, 0]
            pc_in_camera[:, 2] = -pc_in_camera[:, 2]
            
            # import pdb; pdb.set_trace()
            
            ### perform the fps on the pcd
            kdline_fps_samples_idx = fpsample.fps_npdu_kdtree_sampling(pc_in_camera[:, :3], self.num_points_in_pc)
            pc_in_camera = pc_in_camera[kdline_fps_samples_idx]
            
            ### TODO: preprocess the pcd to be the same as the training coordinate of contactgraspnet
            mean = np.mean(pc_in_camera, axis=0, keepdims=True)
            pc_in_camera -= mean
            return rgb, depth, pc_in_camera, mean
        elif self.obs_mode == 'old_articubot_cgn_world':
            rgb, depth = self.render(return_depth=True, mode='depth')
            depth = self.augment_depth(depth)
            pc_in_camera, pc_in_world = get_pc_in_camera_and_world_frame(self.projection_matrix, self.view_matrix, depth, self.camera_width, self.camera_height)
            lower_bound = self.scene_min_aabb - 0.3
            upper_bound = self.scene_max_aabb + 0.3
            good_idx = np.logical_and(np.all(pc_in_world >= lower_bound, axis=1), np.all(pc_in_world <= upper_bound, axis=1))
        
            cprint("Using world frame!", "yellow")
            pc_in_world = pc_in_world[good_idx]        
            ### perform the fps on the pcd
            kdline_fps_samples_idx = fpsample.fps_npdu_kdtree_sampling(pc_in_world[:, :3], self.num_points_in_pc)
            pc_in_world = pc_in_world[kdline_fps_samples_idx]
            
            ### TODO: randomly sample 4 gripper points
            from test_PointNet2.cgn.acronym_dataloader import random_quaternion
            from test_PointNet2.dataset_from_disk import get_4_points_from_gripper_pos_orient
            scene_center = np.mean(pc_in_world, axis=0)
            low = np.copy(scene_center)
            low[0] = scene_center[0] - 0.7
            low[1] = scene_center[1] - 0.7
            low[2] = 0.4
            high = np.copy(scene_center)
            high[0] = scene_center[0] + 0.7
            high[1] = scene_center[1] + 0.7
            high[2] = 0.7 + 0.4
            gripper_pos = np.random.uniform(low, high)
            gripper_quat = random_quaternion()
            gripper_q = np.random.uniform(0, 0.04)
            gripper_pcd = get_4_points_from_gripper_pos_orient(gripper_pos, gripper_quat, gripper_q)
            
            pc_in_world = np.concatenate([pc_in_world, gripper_pcd], axis=0)
            
            ### TODO: preprocess the pcd to be the same as the training coordinate of contactgraspnet
            mean = np.mean(pc_in_world, axis=0, keepdims=True)
            pc_in_world -= mean
            return rgb, depth, pc_in_world, mean
        
        elif self.obs_mode == 'm2t2':
            pc_in_worlds = []
            for project_matrix, view_matrix in zip(self.project_matrices, self.view_matrices):
                rgb, depth = self.render(return_depth=True, mode='depth', view_matrix=view_matrix, projection_matrix=project_matrix, camera_width=self.camera_width, camera_height=self.camera_height)
                depth = self.augment_depth(depth)
                _, pc_in_world = get_pc_in_camera_and_world_frame(project_matrix, view_matrix, depth, self.camera_width, self.camera_height)
                pc_in_worlds.append(pc_in_world)
            pc_in_world = np.concatenate(pc_in_worlds, axis=0)
            
            lower_bound = self.scene_min_aabb - 0.1
            upper_bound = self.scene_max_aabb + 0.1
            good_idx = np.logical_and(np.all(pc_in_world >= lower_bound, axis=1), np.all(pc_in_world <= upper_bound, axis=1))
        
            pc_in_world = pc_in_world[good_idx]
            
            ### perform the fps on the pcd
            kdline_fps_samples_idx = fpsample.fps_npdu_kdtree_sampling(pc_in_world[:, :3], self.num_points_in_pc)
            pc_in_world = pc_in_world[kdline_fps_samples_idx]
            
            ### visualize the pcd using open3d
            # pcd_o3d = o3d.geometry.PointCloud()
            # pcd_o3d.points = o3d.utility.Vector3dVector(pc_in_world)
            # o3d.visualization.draw_geometries([pcd_o3d])
            
            ### TODO: preprocess the pcd to be the same as the training coordinate of contactgraspnet
            mean = np.mean(pc_in_world, axis=0, keepdims=True)
            pc_in_world -= mean
            return rgb, depth, pc_in_world, mean
        
    
    ### TODO: augment the depth image
    def augment_depth(self, depth):
        pass
        return depth

            
    def render(self, return_depth=False, mode=None, view_matrix=None, projection_matrix=None, camera_width=None, camera_height=None):
        # assert self.view_matrix is not None, 'You must call self.setup_camera() or self.setup_camera_rpy() before getting a camera image'
        if view_matrix is None:
            view_matrix = self.view_matrix
        if projection_matrix is None:
            projection_matrix = self.projection_matrix
        if camera_width is None:
            camera_width = self.camera_width
            camera_height = self.camera_height
        w, h, img, depth, segmask = p.getCameraImage(camera_width, camera_height, 
            view_matrix, projection_matrix, 
            renderer=p.ER_BULLET_HARDWARE_OPENGL, 
            physicsClientId=self.id)
        img = np.reshape(img, (h, w, 4))[:, :, :3]
        depth = np.reshape(depth, (h, w))

        if return_depth:
            return img, depth
        else:
            return img
        
    def set_debug_camera_to_scene(self):
        scene_center = (self.scene_min_aabb + self.scene_max_aabb) / 2.0
        scene_range = self.scene_max_aabb - self.scene_min_aabb
        distance = max(1e-3, np.linalg.norm(scene_range)) * 1.5
        p.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=-25,
            cameraPitch=-45,
            cameraTargetPosition=scene_center.tolist(),
            physicsClientId=self.id,
        )
        
    def project_world_to_pixel(self, point_world):
        view = np.asarray(self.view_matrix).reshape([4, 4], order="F")
        proj = np.asarray(self.projection_matrix).reshape([4, 4], order="F")
        point_h = np.array([point_world[0], point_world[1], point_world[2], 1.0], dtype=np.float32)
        clip = proj @ view @ point_h
        if clip[3] <= 1e-6:
            return None
        ndc = clip[:3] / clip[3]
        u = int((ndc[0] + 1.0) * 0.5 * self.camera_width)
        v = int((1.0 - ndc[1]) * 0.5 * self.camera_height)
        return u, v

        
    def draw_grasp_lines_on_image(self, img_rgb, grasps_cam, color=(0, 255, 0), thickness=2, baseline_axis=1):
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        for grasp in grasps_cam:
            if np.allclose(grasp, np.eye(4)):
                continue
            grasp_world = grasp
            for p1, p2 in self.grasp_lines_from_pose(grasp_world, baseline_axis=baseline_axis):
                pt1 = self.project_world_to_pixel(p1)
                pt2 = self.project_world_to_pixel(p2)
                if pt1 is None or pt2 is None:
                    continue
                cv2.line(img_bgr, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)
        return img_bgr
        
    def save_grasp_debug_image(self, save_path, grasps_cam=None, executed_grasp_cam=None, baseline_axis=1):
        hidden_rgba = {}
        if hasattr(self, "urdf_ids") and "robot" in self.urdf_ids and hasattr(self, "panda_joints"):
            robot_id = self.urdf_ids["robot"]
            link_indices = [
                self.panda_joints["hand"],
                self.panda_joints["left_finger"],
                self.panda_joints["right_finger"],
            ]
            hidden_rgba = self.set_links_invisible(robot_id, link_indices)

        img_rgb = self.render(return_depth=False)
        if grasps_cam is not None:
            img_bgr = self.draw_grasp_lines_on_image(img_rgb, grasps_cam, thickness=1, baseline_axis=baseline_axis)
        else:
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        if executed_grasp_cam is not None:
            if np.asarray(executed_grasp_cam).ndim == 2:
                executed_grasp_cam = [executed_grasp_cam]
            img_rgb_for_red = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_bgr = self.draw_grasp_lines_on_image(
                img_rgb_for_red,
                executed_grasp_cam,
                color=(0, 0, 255),
                baseline_axis=baseline_axis,
            )
        print("Saving grasp debug image to: {}".format(save_path))
        cv2.imwrite(save_path, img_bgr)

        if hidden_rgba:
            self.restore_link_visibility(robot_id, hidden_rgba)
            
    def restore_link_visibility(self, body_id, saved_rgba):
        for link_idx, rgba in saved_rgba.items():
            p.changeVisualShape(
                body_id,
                link_idx,
                rgbaColor=rgba,
                physicsClientId=self.id,
            )

        
    def set_links_invisible(self, body_id, link_indices):
        visual_data = p.getVisualShapeData(body_id, physicsClientId=self.id)
        if not visual_data:
            return {}
        saved_rgba = {}
        link_set = set(link_indices)
        for data in visual_data:
            link_idx = data[1]
            if link_idx not in link_set:
                continue
            rgba = data[7]
            if link_idx not in saved_rgba:
                saved_rgba[link_idx] = rgba
            p.changeVisualShape(
                body_id,
                link_idx,
                rgbaColor=[rgba[0], rgba[1], rgba[2], 0.0],
                physicsClientId=self.id,
            )
        return saved_rgba
        
    def grasp_lines_from_pose(self, grasp_pose, baseline_axis=1):
        grasp_pos = grasp_pose[:3, 3]
        grasp_orn = grasp_pose[:3, :3]

        approaching_dir = grasp_orn[:, 2]
        baseline_dir = grasp_orn[:, baseline_axis]

        root = grasp_pos
        grasping_center = grasp_pos + 0.105 * approaching_dir
        mid_point = grasp_pos + 0.05 * approaching_dir
        left_finger = mid_point - 0.04 * baseline_dir
        right_finger = mid_point + 0.04 * baseline_dir
        left_top = left_finger + (grasping_center - mid_point)
        right_top = right_finger + (grasping_center - mid_point)

        return [
            (left_finger, right_finger),
            (root, mid_point),
            (left_finger, left_top),
            (right_finger, right_top),
        ]
            
            
        
    
        
    def visualize_grasp(self, pcd_cam, grasps_cam, topk=10):
        # num_pts = 5000
        # random_idx = np.random.choice(pcd_cam.shape[0], num_pts, replace=False)
        # pcd_cam = pcd_cam[random_idx]
        
        # ax = plt.figure().add_subplot(projection='3d')
        # ax.scatter(pcd_cam[:, 0], pcd_cam[:, 1], pcd_cam[:, 2], s=1, c='gray')
        
        # color = 'green'
        # for pose in grasps_cam[:topk]:

        #     grasp_pos = pose[:3, 3]
        #     grasp_orn_matrix = pose[:3, :3]
            
        #     approaching_dir = grasp_orn_matrix[:, 2]  # The z-axis of the grasp orientation
        #     baseline_dir = grasp_orn_matrix[:, 0]  # The x-axis of the grasp orientation in contact graspnet
            
        #     root = grasp_pos
        #     grasping_center = grasp_pos + 0.105 * approaching_dir
        #     mid_point = grasp_pos + 0.075 * approaching_dir
        #     left_finger = mid_point - 0.04 * baseline_dir
        #     right_finger = mid_point + 0.04 * baseline_dir
            
        #     # left to right
        #     ax.plot([left_finger[0], right_finger[0]], [left_finger[1], right_finger[1]], [left_finger[2], right_finger[2]], color=color, linewidth=2)
            
        #     midd_point = mid_point
        #     # root to middle point
        #     ax.plot([root[0], midd_point[0]], [root[1], midd_point[1]], [root[2], midd_point[2]], color=color, linewidth=2)
            
        #     left_top = left_finger + (grasping_center - midd_point)
        #     right_top = right_finger + (grasping_center - midd_point)
        #     # left finger to left top
        #     ax.plot([left_finger[0], left_top[0]], [left_finger[1], left_top[1]], [left_finger[2], left_top[2]], color=color, linewidth=2)
        #     # right finger to right top
        #     ax.plot([right_finger[0], right_top[0]], [right_finger[1], right_top[1]], [right_finger[2], right_top[2]], color=color, linewidth=2)
    
        # plt.axis("off")
        # plt.axis("equal")
        # plt.show()
        
        # Convert point cloud to Open3D format
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_cam)
        pcd.paint_uniform_color([0.5, 0.5, 0.5])  # gray color

        geometries = [pcd]

        for pose in grasps_cam[:topk]:
            grasp_pos = pose[:3, 3]
            grasp_orn_matrix = pose[:3, :3]
            
            approaching_dir = grasp_orn_matrix[:, 2]  # z-axis
            baseline_dir = grasp_orn_matrix[:, 0]     # x-axis

            root = grasp_pos
            grasping_center = grasp_pos + 0.105 * approaching_dir
            mid_point = grasp_pos + 0.05 * approaching_dir
            left_finger = mid_point - 0.04 * baseline_dir
            right_finger = mid_point + 0.04 * baseline_dir

            left_top = left_finger + (grasping_center - mid_point)
            right_top = right_finger + (grasping_center - mid_point)

            # Define grasp line segments
            lines = [
                [0, 1],  # left finger to right finger
                [2, 3],  # root to mid_point
                [4, 5],  # left finger to left top
                [6, 7]   # right finger to right top
            ]

            points = np.vstack([
                left_finger,
                right_finger,
                root,
                mid_point,
                left_finger,
                left_top,
                right_finger,
                right_top
            ])

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0]] * len(lines))  # green lines

            geometries.append(line_set)

        o3d.visualization.draw_geometries(geometries)
            
    def set_eef_to_pose(self, target_pos, target_quat):
        """
        Set the end effector to a target position and orientation.
        """
        target_pos = np.array(target_pos)
        target_quat = np.array(target_quat)
        
        ### old way of using a floating base
        # p.resetBasePositionAndOrientation(self.urdf_ids["robot"], target_pos, target_quat, physicsClientId=self.id)

        ### I will need to run a ik solver now to solve for the joint angles for the floating joints such that the hand root is at the target pose. 
        target_link = self.panda_joints['hand']
        joint_angles = p.calculateInverseKinematics(self.urdf_ids['robot'], target_link, target_pos, target_quat, physicsClientId=self.id)
        for j in range(6):
            p.resetJointState(self.urdf_ids['robot'], j, joint_angles[j], 0, physicsClientId=self.id)

        ### open gripper fully
        p.resetJointState(self.urdf_ids['robot'], self.panda_joints['left_finger'], 0.04, 0, physicsClientId=self.id)
        p.resetJointState(self.urdf_ids['robot'], self.panda_joints['right_finger'], 0.04, 0, physicsClientId=self.id)
        
        ### check if the robot is in collision with any other objects
        p.stepSimulation(physicsClientId=self.id)
        res = p.getContactPoints(bodyA=self.urdf_ids['robot'], physicsClientId=self.id)
        if len(res) > 0:
            return False
        return True
        
    def step(self, grasp, all_grasps=None, debug=False, visual=False, visual_save_path=None):
        ### TODO: convert grasp from camera frame to world frame
        if self.act_mode == 'cgn':
            world_to_camera = np.asarray(self.view_matrix).reshape([4, 4], order="F")
            camera_to_world = np.linalg.inv(world_to_camera)
            grasp_in_world = camera_to_world @ grasp
        elif self.act_mode == 'old_articubot_cgn_world':
            grasp_in_world = grasp
        elif self.act_mode == 'm2t2':
            grasp_in_world = grasp
        
        target_pos, target_rotation = grasp_in_world[:3, 3], grasp_in_world[:3, :3]
        target_rotation = target_rotation @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])  # flip the xy axis
        target_quat = R.from_matrix(target_rotation).as_quat()
        
        grasp_in_world = np.eye(4)
        grasp_in_world[:3, :3] = target_rotation
        grasp_in_world[:3, 3] = target_pos
        
        if self.precontact:
            target_pos = target_pos - 0.05 * target_rotation[:, 2]  # move the end effector a bit back in the approaching direction
        
        collision_free = self.set_eef_to_pose(target_pos, target_quat)

        if debug: 
            import pdb; pdb.set_trace()
        
        self.rendered_images = []
        if not collision_free:
            if not self.precontact:
                return 0, 'grasp in collision'
            else:
                return 0, 'pregrasp in collision'
        
        ### save images
        ### zoom in to the object
        self.set_camera(camera_width=640, camera_height=480, distance_ratio=1.5)
        
        if visual:
            # self.set_debug_camera_to_scene()
            self.obs_mode = 'cgn'
            self.set_camera()
            if not os.path.exists(visual_save_path):
                os.makedirs(visual_save_path, exist_ok=True)
            other_grasps = []
            for other_grasp in all_grasps:
                if self.act_mode == 'cgn':
                    world_to_camera = np.asarray(self.view_matrix).reshape([4, 4], order="F")
                    camera_to_world = np.linalg.inv(world_to_camera)
                    other_grasp_in_world = camera_to_world @ other_grasp
                elif self.act_mode == 'm2t2':
                    other_grasp_in_world = other_grasp
                    
                o_target_pos, o_target_rotation = other_grasp_in_world[:3, 3], other_grasp_in_world[:3, :3]
                o_target_rotation = o_target_rotation @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])  # flip the xy axis
                
                other_grasp_in_world = np.eye(4)
                other_grasp_in_world[:3, :3] = o_target_rotation
                other_grasp_in_world[:3, 3] = o_target_pos
                other_grasps.append(other_grasp_in_world)
                
            self.save_grasp_debug_image(visual_save_path + "topk.png", grasps_cam=other_grasps, executed_grasp_cam=grasp_in_world)
            
        
        # move to the real grasp pose if self.precontact is True
        if self.precontact:
            target_pos = target_pos + 0.05 * target_rotation[:, 2]
            target_link = self.panda_joints['hand']
            joint_angles = p.calculateInverseKinematics(self.urdf_ids['robot'], target_link, target_pos, target_quat, physicsClientId=self.id)
            p.setJointMotorControlArray(self.urdf_ids['robot'], jointIndices=list(range(6)), 
                                        controlMode=p.POSITION_CONTROL, targetPositions=joint_angles[:6], physicsClientId=self.id)
            for _ in range(50):
                p.stepSimulation(physicsClientId=self.id)
                if _ % 2 == 0:
                    rgb, depth = self.render(return_depth=True, mode='depth')
                    self.rendered_images.append(rgb)
                if debug:
                    time.sleep(0.1)
        
        # close the gripper
        p.setJointMotorControlArray(self.urdf_ids['robot'], jointIndices=[self.panda_joints['left_finger'], self.panda_joints['right_finger']], 
                                    controlMode=p.POSITION_CONTROL, targetPositions=[0,0], physicsClientId=self.id)
        for _ in range(40):
            p.stepSimulation(physicsClientId=self.id)
            if _ % 2 == 0:
                rgb, depth = self.render(return_depth=True, mode='depth')
                self.rendered_images.append(rgb)
            if debug:
                time.sleep(0.1)
        
        ### get which object it is in contact with
        res = p.getContactPoints(bodyA=self.urdf_ids['robot'], physicsClientId=self.id)
        if len(res) > 0:
            contacts = set()
            for contact in res:
                bodyB = contact[2]  # bodyB is the object in contact with the robot
                contacts.add(bodyB)
        else:
            return 0, 'no contact with any object'
        
        ### contacting with multiple object is assumed to be a failure
        if len(contacts) > 1:
            return 0, 'contacting with multiple objects'
        
        contact_body = list(contacts)[0]  # get the first object in contact with the robot
        
        # lift up (velocity control?)
        # for _ in range(50):
        #     p.resetBaseVelocity(self.urdf_ids['robot'], [0, 0, 0.1], physicsClientId=self.id)
        #     p.stepSimulation(physicsClientId=self.id)
        #     time.sleep(0.2)

        timesteps = 50
        delta_movement = 0.015
        for i in range(timesteps):
            cur_z_joint_val = p.getJointState(self.urdf_ids['robot'], self.panda_joints['z_joint'], physicsClientId=self.id)[0]
            p.setJointMotorControlArray(self.urdf_ids['robot'], jointIndices=[self.panda_joints['z_joint']], 
                                        controlMode=p.POSITION_CONTROL, targetPositions=[cur_z_joint_val + delta_movement], physicsClientId=self.id)
            p.stepSimulation(physicsClientId=self.id)
            if i % 2 == 0:
                rgb, depth = self.render(return_depth=True, mode='depth')
                self.rendered_images.append(rgb)
            if debug:
                time.sleep(0.1)
            # if i % 10 == 0:
            #     cprint("lifting up, step: {}".format(i), "green")
        
        if debug:
            import pdb; pdb.set_trace()

        if visual:
            cv2.imwrite(visual_save_path + "lifted.png", self.render(return_depth=False)[:, :, ::-1])                
        
        ### TODO: check if the obj is still inside the gripper
        res = p.getContactPoints(bodyA=self.urdf_ids['robot'], bodyB=contact_body, physicsClientId=self.id)
        if len(res) > 0:
            return 1, 'success'
        else:
            return 0, 'object not in gripper after lifting up'
        
    def reset_object(self, env_state=None):
        states = self.stablized_state if env_state is None else env_state
        for key in self.urdf_ids.keys():
            if key == "robot": continue
            p.resetBasePositionAndOrientation(self.urdf_ids[key], states[key][0], states[key][1], physicsClientId=self.id)
            p.resetBaseVelocity(self.urdf_ids[key], [0, 0, 0], [0, 0, 0], physicsClientId=self.id)
        
    def close(self):
        p.disconnect(physicsClientId=self.id)
        cprint("pybullet disconnected", "blue")

        
        
        