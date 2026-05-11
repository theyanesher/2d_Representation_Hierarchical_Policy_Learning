#!/usr/bin/env python3
import h5py
import pickle
import typer
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import imageio

app = typer.Typer()
ROOT = Path('/data/minon/tax3d-conditioned-mimicgen/data/robomimic/datasets/')
# ROOT = Path('/project_data/held/mnakuraf/tax3d-conditioned-mimicgen/data/robomimic/datasets')

def render_open3d_frame(obs: dict, save_for_video: bool):
    """
    Create/render an Open3D visualization of obs. If save_for_video is True,
    return an image; else show interactively and return None.
    """
    pc = obs['point_cloud']
    geoms = []
    def make_pcd(points, color=None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if color is not None:
            arr = np.asarray(color)
            if arr.ndim == 1:
                arr = np.tile(arr, (points.shape[0], 1))
            pcd.colors = o3d.utility.Vector3dVector(arr)
        return pcd

    # point cloud
    color = obs.get('point_cloud_color')
    geoms.append(make_pcd(pc, color))
    # gripper
    if obs.get('gripper_pcd') is not None:
        geoms.append(make_pcd(obs['gripper_pcd'], [1.0, 0.0, 0.0]))
    # goals
    if obs.get('goal_gripper_pcd') is not None:
        geoms.append(make_pcd(obs['goal_gripper_pcd'], [0.0, 1.0, 0.0]))
    if obs.get('pred_goal_gripper_pcd') is not None:
        geoms.append(make_pcd(obs['pred_goal_gripper_pcd'], [1.0, 0.0, 1.0]))
    if obs.get('unique_goal_gripper_pcds') is not None:
        for i in range(3):
            geoms.append(make_pcd(obs['unique_goal_gripper_pcds'][i], [1.0, 0.5, 0.0]))

    if save_for_video:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        for g in geoms:
            vis.add_geometry(g)
        ctr = vis.get_view_control()
        ctr.set_lookat(np.mean(pc, axis=0))
        ctr.set_front([0.0, 1.0, 0.0]); ctr.set_up([0.0, 0.0, 1.0]); ctr.set_zoom(0.7)
        vis.poll_events(); vis.update_renderer()
        img = (np.asarray(vis.capture_screen_float_buffer(do_render=True)) * 255).astype(np.uint8)
        vis.destroy_window()
        return img
    else:
        o3d.visualization.draw_geometries(geoms)
        return None

def render_matplotlib_frame(obs: dict, save_for_video: bool):
    """
    Create/render a Matplotlib 3D scatter of obs. If save_for_video is True,
    return an image; else show interactively and return None.
    """
    pc = obs['point_cloud']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc[:,0], pc[:,1], pc[:,2], c='b')
    if obs.get('gripper_pcd') is not None:
        ax.scatter(*obs['gripper_pcd'].T, c='r')
    if obs.get('goal_gripper_pcd') is not None:
        ax.scatter(*obs['goal_gripper_pcd'].T, c='g')
    ax.view_init(elev=24, azim=-117)
    ax.set_title('3D Visualization'); ax.axis('equal')

    if save_for_video:
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img
    else:
        plt.show()
        return None


def visualize_frame(obs: dict,
                    use_open3d: bool = False,
                    save_for_video: bool = False):
    """Dispatch to the appropriate renderer based on use_open3d."""
    if use_open3d:
        return render_open3d_frame(obs, save_for_video)
    else:
        return render_matplotlib_frame(obs, save_for_video)

# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------

@app.command('vis-hdf5')
def vis_hdf5(
    task: str = typer.Option(..., help='Task name'),
    demo: int = typer.Option(0, help='Demo index'),
    use_open3d: bool = typer.Option(False, help='Use Open3D')
):
    """Visualize a single timestep from an HDF5 demo."""
    path = ROOT / task / f'{task}_pcd_abs.hdf5'
    with h5py.File(path, 'r') as f:
        grp = f['data'][f'demo_{demo}']['obs']
        for timestep in range(len(grp['point_cloud'])):
            obs = {
                'point_cloud': np.array(grp['point_cloud'][timestep][:,:3]),
                'gripper_pcd': np.array(grp['gripper_pcd'][timestep][:,:3]),
                'goal_gripper_pcd': np.array(grp['goal_gripper_pcd'][timestep])
                                        if 'goal_gripper_pcd' in grp else None,
                'pred_goal_gripper_pcd': np.array(grp['pred_goal_gripper_pcd'][timestep])
                                        if 'pred_goal_gripper_pcd' in grp else None,
                'point_cloud_color': np.array(grp['point_cloud'][timestep][:,3:])
                                        if 'point_cloud' in grp else None,
                # 'unique_goal_gripper_pcds': np.array(f['data'][f'demo_{demo}']['unique_goal_gripper_pcds'])
                #     if 'unique_goal_gripper_pcds' in f['data'][f'demo_{demo}'] else None
            }
            visualize_frame(obs, use_open3d)

@app.command('vis-pickle')
def vis_pickle(
    task: str = typer.Option(..., help='Task name'),
    episode: int = typer.Option(0, help='Episode index'),
    timestep: int = typer.Option(0, help='Timestep index'),
    use_open3d: bool = typer.Option(False, help='Use Open3D')
):
    """Visualize a single timestep from pickle files of an episode."""
    pkl = ROOT / task / f'{task}_abs'/ f'episode_{episode}' / f'{timestep}.pkl'
    with open(pkl, 'rb') as f:
        data = pickle.load(f)
    obs = {
        'point_cloud': np.array(data['point_cloud'][0]),
        'gripper_pcd': np.array(data['gripper_pcd'][0]),
        'goal_gripper_pcd': np.array(data.get('goal_gripper_pcd', [None])[0])
                                if data.get('goal_gripper_pcd') is not None else None,
        'pred_goal_gripper_pcd': None,
        'point_cloud_color': np.array(data['point_cloud_color'][0])
                                if 'point_cloud_color' in data else None
    }
    visualize_frame(obs, use_open3d)

@app.command('vis-hdf5-img')
def vis_hdf5_images(
    task: str = typer.Option(..., help='Task name'),
    demo: int = typer.Option(0, help='Demo index'),
    diffpo: bool = typer.Option(False, help='diffusion policy')
):
    """Visualize a single timestep from an HDF5 demo."""
    if diffpo:
        path = ROOT / task / f'{task}_pcd_abs_images_processed_mini_flow.hdf5'
    else:
        path = ROOT / task / f'{task}_abs.hdf5'
    with h5py.File(path, 'r') as f:
        grp = f['data'][f'demo_{demo}']['obs']
        for timestep in range(len(grp['point_cloud'])):
            if diffpo:
                fig, axs = plt.subplots(2, 2, figsize=(8, 8))
                axs[0, 0].imshow(grp['agentview_image_84'][timestep][...,:3])
                axs[0, 1].imshow(grp['robot0_eye_in_hand_image_84'][timestep][...,:3])
                axs[1, 0].imshow(grp['agentview_image_84'][timestep][...,3])
                axs[1, 1].imshow(grp['robot0_eye_in_hand_image_84'][timestep][...,3])
                for ax in axs.flat:
                    ax.axis('off')
                plt.show()
            else:
                plt.imshow(grp['agentview_image'][timestep])
                plt.show()
                            

@app.command('generate-hdf5-video')
def gen_hdf5_video(
    task: str = typer.Option(..., help='Task name'),
    demo: int = typer.Option(0, help='Demo index'),
    use_open3d: bool = typer.Option(False, help='Use Open3D'),
    fps: int = typer.Option(10, help='Frames per second')
):
    """Generate a video for one HDF5 demo."""
    h5path = ROOT / task / f'{task}_pcd_abs_512.hdf5'
    out_dir = ROOT / task / f'{task}_videos'
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5path, 'r') as f:
        grp = f['data'][f'demo_{demo}']
        num = len(grp['actions'])
        images = []
        obs_grp = grp['obs']
        for i in range(num):
            obs = {
                'point_cloud': np.array(obs_grp['point_cloud'][i]),
                'gripper_pcd': np.array(obs_grp['gripper_pcd'][i]),
                'goal_gripper_pcd': np.array(obs_grp['goal_gripper_pcd'][i])
                                        if 'goal_gripper_pcd' in obs_grp else None,
                'pred_goal_gripper_pcd': np.array(obs_grp['pred_goal_gripper_pcd'][i])
                                        if 'pred_goal_gripper_pcd' in obs_grp else None,
                'point_cloud_color': np.array(obs_grp['point_cloud_color'][i])
                                        if 'point_cloud_color' in obs_grp else None
            }
            images.append(visualize_frame(obs, use_open3d, save_for_video=True))
    vid = out_dir / f'demo_{demo}.mp4'
    writer = imageio.get_writer(str(vid), fps=fps, codec='libx264')
    for im in images:
        writer.append_data(im)
    writer.close()
    typer.echo(f"Saved HDF5 video: {vid}")

@app.command('gen-pickle-video')
def gen_pickle_video(
    task: str = typer.Option('square_d2', help='Task name'),
    use_open3d: bool = typer.Option(False, help='Use Open3D'),
    fps: int = typer.Option(10, help='Frames per second')
):
    """Generate a video for one pickle-based episode."""
    for episode in range(0,1000):
        ep_dir = ROOT / task / f'{task}_abs' / f'episode_{episode}'
        files = sorted(ep_dir.glob('*.pkl'), key=lambda p: int(p.stem))
        images = []
        for pkl in files:
            with open(pkl, 'rb') as f:
                data = pickle.load(f)
            obs = {
                'point_cloud': np.array(data['point_cloud'][0]),
                'gripper_pcd': np.array(data['gripper_pcd'][0]),
                'goal_gripper_pcd': np.array(data.get('goal_gripper_pcd', [None])[0])
                                        if data.get('goal_gripper_pcd') is not None else None,
                'pred_goal_gripper_pcd': None,
                'point_cloud_color': np.array(data['point_cloud_color'][0])
                                        if 'point_cloud_color' in data else None
            }
            images.append(visualize_frame(obs, use_open3d, save_for_video=True))
        out_dir = ROOT / task / f'{task}_videos'
        out_dir.mkdir(parents=True, exist_ok=True)
        vid = out_dir / f'episode_{episode}.mp4'
        writer = imageio.get_writer(str(vid), fps=fps, codec='libx264')
        for im in images:
            writer.append_data(im)
        writer.close()
        typer.echo(f"Saved pickle video: {vid}")

@app.command('hdf5-images-to-video')
def hdf5_images_to_video(
    task: str = typer.Option(..., help='Task name'),
    demo: int = typer.Option(0, help='Demo index'),
    diffpo: bool = typer.Option(False, help='Use diffusion-policy file'),
    fps: float = typer.Option(10.0, help='Frames per second')
):
    """
    Read one HDF5 demo and save it as a video (MP4) using imageio.
    """
    if diffpo:
        path = ROOT / task / f'{task}_pcd_abs_images_processed_mini_flow.hdf5'
    else:
        path = ROOT / task / f'{task}_abs.hdf5'
    out_dir = ROOT / task / f'{task}_videos'
    out_dir.mkdir(parents=True, exist_ok=True)
    vid = out_dir / f'episode_{demo}.mp4'
    writer = imageio.get_writer(str(vid), fps=fps, codec='libx264')
    with h5py.File(path, 'r') as f:
        grp = f['data'][f'demo_{demo}']['obs']
        length = len(grp['point_cloud'])

        for t in range(length):
            if diffpo:
                a = grp['agentview_image_84'][t][...,:3]
                b = grp['robot0_eye_in_hand_image_84'][t][...,:3]
                c = grp['agentview_image_84'][t][...,3] * 255
                d = grp['robot0_eye_in_hand_image_84'][t][...,3] * 255
                # stack grayscale into RGB
                c = np.stack([c]*3, axis=-1)
                d = np.stack([d]*3, axis=-1)
                # 2×2 mosaic
                top    = np.concatenate((a, b), axis=1)
                bottom = np.concatenate((c, d), axis=1)
                frame  = np.concatenate((top, bottom), axis=0)
            else:
                frame = grp['agentview_image'][t]
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            writer.append_data(frame)
        writer.close()

if __name__ == '__main__':
    app()
