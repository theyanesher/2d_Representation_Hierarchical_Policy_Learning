import third_party.robogen.robogen_utils as ru
import matplotlib.pyplot as plt
import torch
import pickle 
from pathlib import Path
import imageio.v2 as imageio
import os
import random
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from datetime import datetime

def visualize_weights(prediction, weights, input_dict, azim=0, timestep=0, save_path=None):
    pointcloud = input_dict['point_cloud'][0]
    gripper_pcd = input_dict['gripper_pcd'][0]
    if 'goal_gripper_pcd' in input_dict:
        goal_gripper_pcd = input_dict['goal_gripper_pcd'][0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], c=weights, cmap='seismic')
    ax.scatter(gripper_pcd[:,0], gripper_pcd[:,1], gripper_pcd[:,2], c='r')
    if 'goal_gripper_pcd' in input_dict:
        ax.scatter(goal_gripper_pcd[:,0], goal_gripper_pcd[:,1], goal_gripper_pcd[:,2], c='g')
    ax.scatter(prediction[0, :, 0], prediction[0, :, 1], prediction[0, :, 2], c='cyan')
    ax.view_init(elev=24, azim=azim)
    ax.set_xlim([-0.3, 0.3])
    ax.set_ylim([-0.3, 0.3])
    ax.set_zlim([0.7, 1.1])
    ax.set_title(f'timestep {timestep}')
    ax.axis("equal")

    if save_path:
        plt.savefig(save_path)
    plt.close(fig)

def create_video_from_images(image_dir, output_path, fps=10, extension='mp4'):
    images = sorted([img for img in os.listdir(image_dir) if img.endswith('.png')])
    image_paths = [os.path.join(image_dir, img) for img in images]
    frames = [imageio.imread(img_path) for img_path in image_paths]
    if extension == 'mp4':
        imageio.mimsave(f'{output_path}.mp4', frames, fps=fps)
    elif extension == 'gif':
        imageio.mimsave(f'{output_path}.gif', frames, duration=1/fps)

def clear_image_folder(image_dir):
    for f in os.listdir(image_dir):
        if f.endswith('.png'):
            os.remove(os.path.join(image_dir, f))

def process_frame(j, pkl_path, model_path, temp_image_dir):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    for key, value in data.items():
        data[key] = torch.tensor(value).unsqueeze(0).float().cuda()
    high_level_policy = ru.load_high_level_weighted_displacement_policy(model_path)
    high_level_policy.cuda()
    high_level_policy.eval()
    with torch.no_grad():
        prediction, weights = ru.run_high_level_policy_inference(
            high_level_policy, data, return_weights=True
        )
        prediction = prediction.squeeze(0).cpu().numpy()
        weights = weights.squeeze(0).cpu().numpy()

    for key, value in data.items():
        data[key] = value.cpu().numpy().squeeze(0)
    save_path = os.path.join(temp_image_dir, f"frame_{j:04d}.png")
    visualize_weights(prediction, weights, data, timestep=j, save_path=save_path)

if __name__ == "__main__":
    dataset_path = Path('/data/minon/tax3d-conditioned-mimicgen/data/robomimic/datasets/three_piece_assembly_d2/three_piece_assembly_d2_abs')
    temp_image_dir = "/data/minon/tax3d-conditioned-mimicgen/visualizations/temp_images"
    os.makedirs(temp_image_dir, exist_ok=True)
    today = datetime.today().strftime('%m.%d.%H.%M')
    os.makedirs(f"/data/minon/tax3d-conditioned-mimicgen/visualizations/{today}", exist_ok=True)
    for i in range(random.randint(0,99),1000,100):
        episode_path = dataset_path / f'episode_{i}/'
        if not episode_path.exists():
            continue
        clear_image_folder(temp_image_dir)  # Reset for each episode
        timesteps = list(range(0, len(list(episode_path.glob('*.pkl'))),))
        mp.set_start_method('spawn', force=True)
        with ProcessPoolExecutor() as executor:
            futures = []
            for j in timesteps:
                pkl_path = episode_path / f'{j}.pkl'
                if not pkl_path.exists():
                    continue
                futures.append(executor.submit(process_frame, j, str(pkl_path), 'model_60', temp_image_dir))
                if j % 10 == 0:
                    # Wait for all to complete
                    for future in futures:
                        future.result()
        output_video_path = f"/data/minon/tax3d-conditioned-mimicgen/visualizations/{today}/episode_{i:04d}"
        create_video_from_images(temp_image_dir, output_video_path, fps=20, extension='gif')
        print(f"Saved video: {output_video_path}")
