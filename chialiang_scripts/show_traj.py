import zarr, glob, time, os
import numpy as np
import open3d as o3d
import cv2
import imageio
from scipy.spatial.transform import Rotation as R

render_intr_mat = np.array([[616.35845947,            0, 539.5], 
                                [           0, 616.98779297, 404.5],
                                [           0,            0,      1]])
render_extr_mat = np.array([
    [1, 0, 0, 1],
    [0, 1, 0, 1],
    [0, 0, 1, 0.5],
    [0, 0, 0, 1],
])
render_img_shape = (1080, 810)

def render(pcds : list, extr : np.ndarray, intr : np.ndarray, img_size : tuple=(512,512), duration=0) -> np.ndarray:
    assert extr.shape == (4, 4)
    assert intr.shape == (3, 3)

    width, height = img_size
    fx, fy, cx, cy = intr[0, 0], intr[1, 1], width / 2.0 - 0.5, height / 2.0 - 0.5

    # Visualize Point Cloud
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=int(width), height=int(height), visible=True) # neet to be true

    for pcd in pcds:
        vis.add_geometry(pcd)

    vis.get_render_option().point_color_option = o3d.visualization.PointColorOption.Color
    vis.get_render_option().point_size = 3.0

    # # Read camera params
    # param = o3d.camera.PinholeCameraParameters()

    # param.extrinsic = extr
    
    # o3d_intr = o3d.camera.PinholeCameraIntrinsic()
    # o3d_intr.set_intrinsics(int(width), int(height), fx, fy, cx, cy)
    # param.intrinsic = o3d_intr

    # ctr = vis.get_view_control()
    # ctr.convert_from_pinhole_camera_parameters(param)

    # Updates
    for pcd in pcds:
        vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    # Capture image
    time.sleep(duration)
    # vis.capture_screen_image(path)
    image = vis.capture_screen_float_buffer()

    # Close
    vis.destroy_window()
    return (np.asarray(image) * 255).astype(np.uint8)

if __name__=="__main__":
    # input_root = 'one_traj/0712-obj-45305'
    # input_root = 'one_traj/0712-obj-45305-copy'
    input_root = 'one_traj/test_chained_diffuser'

    input_dirs = glob.glob(f'{input_root}/2024*')
    input_dirs = sorted(input_dirs)

    for input_dir in input_dirs[:20]:

        group_paths_raw = glob.glob(f'{input_dir}/*')
        group_paths = []
        for group_path in group_paths_raw:
            if os.path.isdir(group_path):
                group_paths.append(group_path)

        sorted_group_paths = sorted(group_paths, key=lambda x: int(x.split('/')[-1]))

        first = False
        for group_path in sorted_group_paths:
            print(group_path)

            group = zarr.open(group_path, mode='r')
            action_pts = []

            rendered_ress = []

            # one trajectory
            data_group = group['data']

            for k in data_group.keys():
                print(k, data_group[k].shape)

            point_cloud = np.asarray(data_group['point_cloud']).squeeze()
            # action = np.asarray(data_group['state']).squeeze()
            trajectory = np.asarray(data_group['trajectory']).squeeze()
            gripper_pcd = np.asarray(data_group['gripper_pcd']).squeeze()
            goal_gripper_pcd = np.asarray(data_group['goal_gripper_pcd']).squeeze()
            current_pose = np.asarray(data_group['init_pose']).squeeze()
            target_pose = np.asarray(data_group['target_pose']).squeeze()
            displacement_gripper_to_object = np.asarray(data_group['displacement_gripper_to_object']).squeeze()

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)

            current_trans = np.eye(4)
            current_trans[:3, :3] = R.from_quat(current_pose[3:7]).as_matrix()
            current_trans[:3, 3] = current_pose[:3]
            wpt = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            wpt.transform(current_trans)
            action_pts.append(wpt)

            target_trans = np.eye(4)
            target_trans[:3, :3] = R.from_quat(target_pose[3:7]).as_matrix()
            target_trans[:3, 3] = target_pose[:3]
            wpt = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            wpt.transform(target_trans)
            action_pts.append(wpt)

            for p in gripper_pcd:
                goal = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                goal.paint_uniform_color([0.706, 0, 1])
                goal.translate(p[:3].reshape(-1, 1))
                action_pts.append(goal)

            for p in goal_gripper_pcd:
                goal = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                goal.paint_uniform_color([1, 0.706, 0])
                goal.translate(p[:3].reshape(-1, 1))
                action_pts.append(goal)

            for i in range(trajectory.shape[0]):
                wpt = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                wpt.compute_vertex_normals()
                wpt.translate(trajectory[i][:3].reshape(-1, 1))
                action_pts.append(wpt)

            # o3d.visualization.draw_geometries([pcd, wpt])

                if first:
                    rendered_res = render(action_pts + [pcd], render_extr_mat, render_intr_mat, render_img_shape, 1)
                else :
                    rendered_res = render(action_pts + [pcd], render_extr_mat, render_intr_mat, render_img_shape, 0.1)

                rendered_ress.append(rendered_res) # BGR to RGB

            imageio.mimsave(f'{input_dir}.gif', rendered_ress, duration=0.1)
