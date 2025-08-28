import numpy as np
import open3d as o3d
import os
import torch
import torch.nn.functional as F

def rotz(theta):
    """Rotation matrix about z-axis by angle theta (radians)."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])
    return R

def load_scene(render_path):
    """
    Return point cloud and camera pose.  Used for loading saved renders.
    Arguments:
        scene_id {str} -- scene index
        cam_pose_id {str} -- camera pose index as length 3 string with
                            leading zeros if necessary.

    Returns:
        [pc, camera_pose] -- [point cloud, camera pose]
        or returns False if not found
    """
    # print('Loading: ', render_path)
    data = np.load(render_path, allow_pickle=True)
    pc_cam = data['pc_cam']
    camera_pose = data['camera_pose']
    return pc_cam[:, :3], camera_pose

def _process_contacts(scene_contact_points, grasp_transforms):
    """
    Processes contact information for a given scene

    num_contacts may not be the same for each scene.  We resample to make this
    the same

    Arguments:
        scene_contact_points {np.ndarray} -- (num_contacts, 2, 3) array of contact points
        grasp_transforms {np.ndarray} -- (num_contacts, 4, 4) array of grasp transforms

    Returns:
        Scene data
    """
    num_pos_contacts = 8000

    contact_directions_01 = scene_contact_points[:,0,:] - scene_contact_points[:,1,:]
    all_contact_points = scene_contact_points.reshape(-1, 3)
    all_finger_diffs = np.maximum(np.linalg.norm(contact_directions_01,axis=1), np.finfo(np.float32).eps)
    all_contact_directions = np.empty((contact_directions_01.shape[0]*2, contact_directions_01.shape[1],))
    all_contact_directions[0::2] = -contact_directions_01 / all_finger_diffs[:,np.newaxis]
    all_contact_directions[1::2] = contact_directions_01 / all_finger_diffs[:,np.newaxis]
    all_contact_suc = np.ones_like(all_contact_points[:,0])
    all_grasp_transform = grasp_transforms.reshape(-1,4,4)
    all_approach_directions = all_grasp_transform[:,:3,2]

    pos_idcs = np.where(all_contact_suc>0)[0]
    if len(pos_idcs) == 0:
        raise ValueError('No positive contacts found')
    # print('total positive contact points ', len(pos_idcs))

    all_pos_contact_points = all_contact_points[pos_idcs]
    all_pos_finger_diffs = all_finger_diffs[pos_idcs//2]
    all_pos_contact_dirs = all_contact_directions[pos_idcs]
    all_pos_approach_dirs = all_approach_directions[pos_idcs//2]

    # -- Sample Positive Contacts -- #
    # Use all positive contacts then mesh_utils with replacement
    if num_pos_contacts > len(all_pos_contact_points)/2:
        pos_sampled_contact_idcs = np.arange(len(all_pos_contact_points))
        pos_sampled_contact_idcs_replacement = np.random.choice(
            np.arange(len(all_pos_contact_points)),
            num_pos_contacts*2 - len(all_pos_contact_points),
            replace=True)
        pos_sampled_contact_idcs= np.hstack((pos_sampled_contact_idcs,
                                                pos_sampled_contact_idcs_replacement))
    else:
        pos_sampled_contact_idcs = np.random.choice(
            np.arange(len(all_pos_contact_points)),
            num_pos_contacts*2,
            replace=False)

    pos_contact_points = torch.from_numpy(all_pos_contact_points[pos_sampled_contact_idcs,:]).type(torch.float32)

    pos_contact_dirs = torch.from_numpy(all_pos_contact_dirs[pos_sampled_contact_idcs,:]).type(torch.float32)
    pos_contact_dirs = F.normalize(pos_contact_dirs, p=2, dim=1)

    pos_finger_diffs = torch.from_numpy(all_pos_finger_diffs[pos_sampled_contact_idcs]).type(torch.float32)

    pos_approach_dirs = torch.from_numpy(all_pos_approach_dirs[pos_sampled_contact_idcs]).type(torch.float32)
    pos_approach_dirs = F.normalize(pos_approach_dirs, p=2, dim=1)

    return pos_contact_points.numpy(), pos_contact_dirs.numpy(), pos_finger_diffs.numpy(), pos_approach_dirs.numpy()

def _load_contacts(scene_id):
    """
    Loads contact information for a given scene
    """
    return np.load(os.path.join(scene_contacts_dir, scene_id + '.npz'), allow_pickle=True)

scene_contacts_dir = '/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/contact_graspnet/acronym/scene_contacts'
scene_info = _load_contacts("000000")
scene_contact_points = scene_info['scene_contact_points']
obj_paths = scene_info['obj_paths']
obj_transforms = scene_info['obj_transforms']
obj_scales = scene_info['obj_scales']
grasp_transforms = scene_info['grasp_transforms']

pos_contact_points, pos_contact_dirs, pos_finger_diffs, \
            pos_approach_dirs = _process_contacts(scene_contact_points, grasp_transforms)

data_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/contact_graspnet/acronym/renders/000000/000.npz"
pc_cam, camera_pose = load_scene(data_path)


# import pdb; pdb.set_trace()
# pos_contact_points_cam = pos_contact_points @ camera_pose[:3, :3].T + camera_pose[:3,3][None,:]
            
T_world_to_cam = np.eye(4)
T_world_to_cam[:3, :3] = camera_pose[:3, :3]
T_world_to_cam[:3, 3] = camera_pose[:3, 3]
# pos_contact_points_homo = np.ones((pos_contact_points.shape[0], 4))
# pos_contact_points_homo[:, :3] = pos_contact_points
# pos_contact_points_cam_2 = T_world_to_cam @ pos_contact_points_homo.T
# pos_contact_points_cam_2 = (pos_contact_points_cam_2.T)[:, :3]

T_cam_to_world = np.linalg.inv(T_world_to_cam)
pc_cam_homo = np.ones((pc_cam.shape[0], 4))
pc_cam_homo[:, :3] = pc_cam
pc_world = (T_cam_to_world @ pc_cam_homo.T).T
pc_world = pc_world[:, :3]

### rotate around the z axis
rotation_z = rotz(np.pi / 4)
pc_world = (rotation_z @ pc_world.T).T
pos_contact_points = (rotation_z @ pos_contact_points.T).T


pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(pc_world)
pcd1.paint_uniform_color([1, 0, 0])  # red

pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(pos_contact_points)
pcd2.paint_uniform_color([0, 1, 0])  # green

coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.1, origin=[0, 0, 0]
)


# Visualize both together
o3d.visualization.draw_geometries([pcd1, pcd2, coord_frame])







