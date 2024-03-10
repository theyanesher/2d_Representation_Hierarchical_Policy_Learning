import pybullet as p
import numpy as np
import open3d as o3d
from cem_policy.utils import save_numpy_as_gif
from manipulation.utils import add_sphere

def load_obj(fn):
    fin = open(fn, 'r')
    lines = [line.rstrip() for line in fin]
    fin.close()

    vertices = []; faces = [];
    for line in lines:
        if line.startswith('v '):
            vertices.append(np.float32(line.split()[1:4]))
        elif line.startswith('f '):
            faces.append(np.int32([item.split('/')[0] for item in line.split()[1:4]]))

    f = np.vstack(faces)
    v = np.vstack(vertices)

    return v, f

def find_nearest_point_on_line(line_pt1, line_pt2, target_pt):
    line_pt1 = np.array(line_pt1)
    line_pt2 = np.array(line_pt2)
    target_pt = np.array(target_pt)
    
    # Step 1: Compute the vector along the line
    line_vec = line_pt2 - line_pt1
    
    # Step 2: Compute the vector from line_pt1 to target_pt
    pt_vec = target_pt - line_pt1
    
    # Step 3: Project pt_vec onto line_vec to find the projection scalar
    # dot_product(pt_vec, line_vec) / dot_product(line_vec, line_vec) gives the scalar
    # by which to multiply line_vec to get the projection vector.
    projection_scalar = np.dot(pt_vec, line_vec) / np.dot(line_vec, line_vec)
    
    # Step 4: Find the nearest point on the line by scaling line_vec and adding it to line_pt1
    nearest_pt = line_pt1 + projection_scalar * line_vec
    
    return nearest_pt

def rotate_point_around_axis(pt, ax, theta_rad):
    """
    Rotate a point around a given axis by theta radiance.
    
    :param pt: The point to rotate (3D coordinates).
    :param ax: The rotation axis (3D unit vector).
    :param theta: The rotation angle in radians.
    :return: The rotated point's coordinates.
    """
    # Ensure ax is a unit vector
    ax = ax / np.linalg.norm(ax)
    
    # Rodrigues' rotation formula
    v_rot = (pt * np.cos(theta_rad) +
             np.cross(ax, pt) * np.sin(theta_rad) +
             ax * np.dot(ax, pt) * (1 - np.cos(theta_rad)))
    
    return v_rot

def setup_camera(camera_eye=[-3, 0, 1.2], camera_target=[0, 0, 0], fov=60, camera_width=640, camera_height=480, physics_id=0):
    view_matrix = p.computeViewMatrix(camera_eye, camera_target, [0, 0, 1], physicsClientId=physics_id)
    projection_matrix = p.computeProjectionMatrixFOV(fov, camera_width / camera_height, 0.01, 100, physicsClientId=physics_id)
    return view_matrix, projection_matrix

id = p.connect(p.GUI)
view_matrix, projection_matrix = setup_camera()

path = "data/dataset/7310/mobility.urdf"
load_pos = [0, 0.5, 0.7]
orientation = [0, 0, np.pi / 2, 1]
scaling = 0.5
obj_id = p.loadURDF(path, basePosition=load_pos, baseOrientation=orientation, physicsClientId=id, useFixedBase=True, globalScaling=scaling)

# get the parent frame of the joint. 
link_state = p.getLinkState(obj_id, 0)
link_urdf_world_pos, link_urdf_world_orn = link_state[0], link_state[1]
# this is the transformation from the parent frame to the world frame. 
T_body_to_world = np.eye(4) # transformation from the parent body frame to the world frame
T_body_to_world[:3, :3] = np.array(p.getMatrixFromQuaternion(link_urdf_world_orn)).reshape(3, 3)
T_body_to_world[:3, 3] = link_urdf_world_pos

# axis in parent frame, transform everything to world frame
axis_body = np.array([-0.6430403380134146, -0.42593899369239807, 0.5477944794777341]) * scaling
axis_world = T_body_to_world[:3, :3] @ axis_body + T_body_to_world[:3, 3]   
axis_dir_body = np.array([0, -1, 0])
axis_pt2_body = np.array(axis_body) + axis_dir_body
axis_end_world = T_body_to_world[:3, :3] @ axis_pt2_body + T_body_to_world[:3, 3]
axis_dir_world = axis_end_world - axis_world

# p.addUserDebugLine(axis_world, axis_end_world, [1, 0, 0], 25, 0)

# get the handle points in world frame
handle_obj_path = "data/dataset/7310/parts_render/handle.obj"
handle_pts, handle_faces = load_obj(handle_obj_path) # this is in object frame
handle_pts = handle_pts * scaling
# transform this to the world frame using the object *base*'s position and orientation
handle_points_world = T_body_to_world[:3, :3] @ handle_pts.T + T_body_to_world[:3, 3].reshape(3, 1) # 3 x N
handle_point_median = np.median(handle_points_world, axis=1)

# find the projection of the handle point to the rotation axis, in world frame. 
project_on_rotation_axis = find_nearest_point_on_line(axis_world, axis_end_world, handle_point_median)
# p.addUserDebugLine(project_on_rotation_axis, handle_point_median, [1, 0, 0], 25, 0)

imgs = []
for rotation_angle in np.linspace(0, np.pi / 2):
    p.resetJointState(obj_id, 1, rotation_angle)
    p.stepSimulation()
    # rotate the handle, in world frame. 
    rotated_handle_pt_local = rotate_point_around_axis(handle_point_median - project_on_rotation_axis, axis_dir_world, rotation_angle)
    rotated_handle_pt = project_on_rotation_axis + rotated_handle_pt_local
    # print("rotated handle point: ", rotated_handle_pt)
    # p.addUserDebugLine(handle_point_median, rotated_handle_pt,  [0, 0, 1], 25, 0)
    # p.addUserDebugPoints([rotated_handle_pt], [[0, 0, 1]], 25, 0)
    s_id = add_sphere(rotated_handle_pt)
    w, h, img, depth, segmask = p.getCameraImage(400, 400, 
        view_matrix, projection_matrix, 
        renderer=p.ER_BULLET_HARDWARE_OPENGL, 
        physicsClientId=id)
    imgs.append(img)
    p.removeBody(s_id)
    
save_numpy_as_gif(np.array(imgs), "manipulation/scripts/get_handle.mp4")
    
# microwave_mesh_data = p.getMeshData(obj_id)
# print(microwave_mesh_data)
# microwave_vertices = []
# # for i in range(0, p.getNumJoints(obj_id)):
# for i in range(0, 1):
#     joint_info = p.getJointInfo(obj_id,i)
#     joint_type = joint_info[2]
#     name = joint_info[1].decode('utf-8')
#     axis_local = joint_info[13]
#     parent_frame_pos = joint_info[14]
#     parent_frame_orn = joint_info[15]
#     parent_index = joint_info[16]
#     nb, vertices = p.getMeshData(obj_id, i)
#     if nb < 1:
#         continue
#     microwave_vertices.extend(vertices)
#     print(i, name,nb,len(vertices), axis_local, joint_type, parent_frame_pos, parent_index)
        
# microwave_vertices = np.array(microwave_vertices)

# # everything in open3d is object frame
# point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = o3d.utility.Vector3dVector(handle_pts)
# colors1 = np.zeros_like(handle_pts)  # Initialize color array.
# colors1[:, 0] = 1  # Red: set the first channel to max.
# point_cloud.colors = o3d.utility.Vector3dVector(colors1)

# point_cloud2 = o3d.geometry.PointCloud()
# point_cloud2.points = o3d.utility.Vector3dVector(microwave_vertices)
# colors2 = np.zeros_like(microwave_vertices)  # Initialize color array.
# colors2[:, 2] = 1  # Red: set the first channel to max.
# point_cloud2.colors = o3d.utility.Vector3dVector(colors2)

# axis_pts = np.linspace(axis_body, axis_pt2_body, 100)
# point_cloud3 = o3d.geometry.PointCloud()
# point_cloud3.points = o3d.utility.Vector3dVector(axis_pts)
# colors3 = np.zeros_like(axis_pts)  # Initialize color array.
# colors3[:, 1] = 1  # Red: set the first channel to max.
# point_cloud3.colors = o3d.utility.Vector3dVector(colors3)

# o3d.visualization.draw_geometries([point_cloud, point_cloud2, point_cloud3])

