import h5py
import typer
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_overlay(dataset_input_dict,rollout_dict, timestep=0):
    # dataset dict
    pointcloud = dataset_input_dict['point_cloud'][0]
    gripper_pcd = dataset_input_dict['gripper_pcd'][0]
    if 'goal_gripper_pcd' in dataset_input_dict:
        goal_gripper_pcd = dataset_input_dict['goal_gripper_pcd'][0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], c='b')
    ax.scatter(gripper_pcd[:,0], gripper_pcd[:,1], gripper_pcd[:,2], c='r')
    if 'goal_gripper_pcd' in dataset_input_dict:
        ax.scatter(goal_gripper_pcd[:,0], goal_gripper_pcd[:,1], goal_gripper_pcd[:,2], c='g')

    # rollout dict
    pointcloud = rollout_dict['point_cloud'][0]
    gripper_pcd = rollout_dict['gripper_pcd'][0]
    if 'goal_gripper_pcd' in rollout_dict:
        goal_gripper_pcd = rollout_dict['goal_gripper_pcd'][0]

    ax.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], c='cyan')
    ax.scatter(gripper_pcd[:,0], gripper_pcd[:,1], gripper_pcd[:,2], c='magenta')
    if 'goal_gripper_pcd' in rollout_dict:
        ax.scatter(goal_gripper_pcd[:,0], goal_gripper_pcd[:,1], goal_gripper_pcd[:,2], c='purple')

    ax.view_init(elev=24, azim=-117)
    ax.set_title(f'timestep {timestep}')
    ax.axis("equal")
    plt.show()

if __name__ == "__main__":
    dataset_path = Path('/data/minon/tax3d-conditioned-mimicgen/data/robomimic/datasets/three_piece_assembly_d2/three_piece_assembly_d2_abs/')
    output_path = Path('/data/minon/tax3d-conditioned-mimicgen/outputs/')
    for i in range(50):
        episode_path = output_path / f'episode_{i}/'
        if not episode_path.exists():
            continue
        timesteps = list(range(0, len(list(episode_path.glob('*.pkl'))),))
        for j in timesteps:
            # rollout pickle
            pkl_path = episode_path / f'action_{j}.pkl'
            if not pkl_path.exists():
                continue
            with open(pkl_path, 'rb') as f:
                rollout_dict = pickle.load(f)
            for key, value in rollout_dict['obs'].items():
                rollout_dict['obs'][key] = value.detach().cpu().numpy()[0]
                if key == 'point_cloud':
                    rollout_dict['obs'][key] = rollout_dict['obs'][key][:,:4500]
            for key, value in rollout_dict['pred'].items():
                rollout_dict['pred'][key] = value.detach().cpu().numpy()[0]

            # dataset pickle
            episode_path = dataset_path / f'episode_{i}/'
            pkl_path = episode_path / f'{j}.pkl'
            with open(pkl_path, 'rb') as f:
                dataset_dict = pickle.load(f)
            visualize_overlay(dataset_dict, rollout_dict['obs'], timestep=j)
