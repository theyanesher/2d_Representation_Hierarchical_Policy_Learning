import pickle
import os
from tqdm import tqdm


def remove_feature_map(object_path):
    all_traj = os.listdir(object_path)
    all_traj = sorted(all_traj)
    for traj_path in tqdm(all_traj, desc="Removing feature map"):
        all_steps = os.listdir(os.path.join(object_path, traj_path))
        all_steps = sorted(all_steps, key=lambda x: int(x.split('.')[0]))
        
        for step in all_steps:
            path = os.path.join(object_path, traj_path, step)
            with open(path, 'rb') as f:
                data = pickle.load(f)
            data.pop('feature_map', None)
            data.pop("pcd_mask", None)
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default=None)
args = parser.parse_args()
remove_feature_map(args.output_dir)