from sandbox.mino.visualizations.visualize_high_level_outputs import visualize_weights, clear_image_folder, create_video_from_images
from pathlib import Path
import pickle
from datetime import datetime

def process_frame(j, pkl_path, temp_image_dir):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    for key, value in data['obs'].items():
        data['obs'][key] = value.detach().cpu().numpy()[0]
        if key == 'point_cloud':
            data['obs'][key] = data['obs'][key][:,:4500]
    for key, value in data['pred'].items():
        data['pred'][key] = value.detach().cpu().numpy()[0]
    visualize_weights(data['pred']['goal_gripper_pcd'], data['pred']['weights'], data['obs'],
                      azim=60, timestep=j, save_path=temp_image_dir / f"frame_{j:04d}.png")
                    # azim=(j*6)%360, timestep=j, save_path=temp_image_dir / f"frame_{j:04d}.png")


if __name__ == "__main__":
    today = datetime.today()
    today_str = today.strftime("%m.%d.%H.%M")
    output_path = Path('/project_data/held/mnakuraf/tax3d-conditioned-mimicgen/outputs/')
    temp_image_dir = output_path / 'temp_images'
    if not temp_image_dir.exists():
        temp_image_dir.mkdir(parents=True, exist_ok=True)
    output_dir_path = output_path / today_str
    if not output_dir_path.exists():
        output_dir_path.mkdir(parents=True, exist_ok=True)
    for i in range(50):
        episode_path = output_path / f'episode_{i}/'
        if not episode_path.exists():
            continue
        clear_image_folder(temp_image_dir)  # Reset for each episode
        timesteps = list(range(0, len(list(episode_path.glob('*.pkl'))),))
        for j in timesteps:
            pkl_path = episode_path / f'action_{j}.pkl'
            if not pkl_path.exists() or j >= 50:
                continue
            process_frame(j, str(pkl_path), temp_image_dir)
        output_video_path = output_dir_path / f"episode_{i:04d}"
        create_video_from_images(temp_image_dir, output_video_path, fps=5, extension='mp4')
        print(f"Saved video: {output_video_path}")