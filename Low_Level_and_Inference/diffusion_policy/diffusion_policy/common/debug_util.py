import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
import os
from matplotlib.animation import FuncAnimation
import io
import imageio

def save_pointmap_visualization(batch, filename='pointmap_debug.png', sample_idx=0, timestep=0, downsample_rate=10):
    """
    Renders cam0, cam1, and cam2 pointmaps and saves the result as a PNG.
    """
    # Force non-interactive backend so it works on servers
    plt.switch_backend('Agg')
    
    cameras = ['cam0_pointmap', 'cam1_pointmap', 'cam2_pointmap']
    fig = plt.figure(figsize=(20, 7))
    
    for i, cam_key in enumerate(cameras):
        if cam_key not in batch['obs']:
            print(f"Skipping {cam_key}: Key not found.")
            continue
            
        # Extract: [B, T, C, H, W] -> [H, W, C]
        # Using .detach().cpu() to ensure it works regardless of device
        points = batch['obs'][cam_key][sample_idx, timestep].permute(1, 2, 0).detach().cpu().numpy()
        
        # Flatten and downsample
        points_flat = points.reshape(-1, 3)[::downsample_rate]
        
        # Plotting
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        
        # Scatters points; coloring by Z (depth) usually looks best for debugging
        scatter = ax.scatter(points_flat[:, 0], 
                             points_flat[:, 1], 
                             points_flat[:, 2], 
                             s=0.5, 
                             c=points_flat[:, 2], 
                             cmap='plasma',
                             alpha=0.6)
        
        ax.set_title(f"Camera: {cam_key}\nFrame: {timestep}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Optional: Adjust the view angle to see the scene better
        ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig) # Clean up memory
    print(f"Successfully saved visualization to: {os.path.abspath(filename)}")

# Usage in Pdb:
# save_pointmap_visualization(batch, filename='check_my_points.png')

def save_pointcloud_video(batch, filename='media/pointcloud_360.mp4', sample_idx=0, timestep=0, downsample_rate=20, clip_bounds=None):
    """
    Combines pointmaps and uses imageio to create a 360-degree MP4.
    clip_bounds: tuple like (-1.5, 1.5) to filter out points outside this range in X, Y, and Z.
    """
    plt.switch_backend('Agg')
    
    cameras = ['cam0_pointmap', 'cam1_pointmap', 'cam2_pointmap']
    all_points = []
    
    # 1. Gather all points
    for cam_key in cameras:
        if cam_key in batch['obs']:
            # Assuming (C, H, W) -> (H, W, C)
            pts = batch['obs'][cam_key][sample_idx, timestep].permute(1, 2, 0).detach().cpu().numpy()
            pts_flat = pts.reshape(-1, 3)[::downsample_rate]
            all_points.append(pts_flat)
    
    if not all_points:
        print("No pointmaps found!")
        return
        
    combined_points = np.vstack(all_points)

    # --- Clipping Logic ---
    if clip_bounds is not None:
        c_min, c_max = clip_bounds
        # Create a mask where all 3 coordinates (X, Y, Z) are within [c_min, c_max]
        mask = np.all((combined_points >= c_min) & (combined_points <= c_max), axis=1)
        combined_points = combined_points[mask]
        
        if combined_points.shape[0] == 0:
            print(f"Warning: All points clipped with bounds {clip_bounds}!")
            return
    # -----------------------
    
    # 2. Setup Plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Re-calculate bounds based on filtered points
    centroid = np.mean(combined_points, axis=0)
    
    # If using hard clipping, we might want to fix the range for a consistent zoom level
    if clip_bounds is not None:
        max_range = (clip_bounds[1] - clip_bounds[0]) / 2.0
    else:
        max_range = np.ptp(combined_points, axis=0).max() / 2.0
    
    frames = []
    print(f"Capturing frames for {len(combined_points)} points...")
    
    for angle in range(0, 360, 5):
        ax.clear()
        # Using a smaller s (size) since we might have many points
        ax.scatter(combined_points[:, 0], 
                   combined_points[:, 1], 
                   combined_points[:, 2], 
                   s=0.5, c=combined_points[:, 2], cmap='viridis', alpha=0.5)
        
        ax.set_title(f"360 View - Step {timestep}")
        
        # Consistent bounds for the cube
        ax.set_xlim(centroid[0] - max_range, centroid[0] + max_range)
        ax.set_ylim(centroid[1] - max_range, centroid[1] + max_range)
        ax.set_zlim(centroid[2] - max_range, centroid[2] + max_range)
        
        ax.view_init(elev=25, azim=angle)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        frames.append(imageio.imread(buf))
        
    plt.close(fig)

    print(f"Writing video to {filename}...")
    imageio.mimsave(filename, frames, fps=20, macro_block_size=None)
    print("Done!")

def save_rgb_video_with_heatmaps(batch, filename='media/eval_rgb_heatmap_video.mp4', sample_idx=0, episode_length=35):
    """
    Combines RGB frames with heatmaps and saves as MP4.
    """
    if not hasattr(save_rgb_video_with_heatmaps, "frames"):
        save_rgb_video_with_heatmaps.frames = []
    if not hasattr(save_rgb_video_with_heatmaps, "rollout_idx"):
        save_rgb_video_with_heatmaps.rollout_idx = 0

    cameras = ['cam0_image', 'cam1_image', 'cam2_image']
    heatmaps = ['cam0_heatmap', 'cam1_heatmap']  # Assuming only cam0 and cam1 have heatmaps

    # breakpoint()
    images = []
    for i, cam_key in enumerate(cameras):
        if cam_key in batch['obs']:
            rgb = batch['obs'][cam_key][sample_idx, -1].permute(1, 2, 0).detach().cpu().numpy()

            heatmap_key = f"cam{i}_heatmap"
            if heatmap_key in batch['obs']:
                heatmap = batch['obs'][heatmap_key][sample_idx, -1].squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
            blend = 0.7 * heatmap[:,:,:3] + 0.3 * rgb if heatmap_key in batch['obs'] else rgb
            images.append(blend)
    image = np.concatenate(images, axis=1)  # Concatenate along width

    save_rgb_video_with_heatmaps.frames.append(image)

    if len(save_rgb_video_with_heatmaps.frames) >= episode_length - 1:
        filename = f"{filename.split('.mp4')[0]}_rollout{save_rgb_video_with_heatmaps.rollout_idx}.mp4"
        imageio.mimsave(filename, save_rgb_video_with_heatmaps.frames, fps=10)
        save_rgb_video_with_heatmaps.rollout_idx += 1
        save_rgb_video_with_heatmaps.frames = []
