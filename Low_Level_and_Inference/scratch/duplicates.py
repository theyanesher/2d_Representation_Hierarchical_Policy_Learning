import pickle
import os
import glob
import numpy as np

def check_trajectory_diversity(base_path):
    # Pattern to find all state_0.pkl files
    # Path: data/rgb_minimal_rand_eval/<obj_id>/<timestamp>/states/state_0.pkl
    search_pattern = os.path.join(base_path, "*/states/state_0.pkl")
    state_files = sorted(glob.glob(search_pattern))

    if not state_files:
        print(f"No state files found in {base_path}")
        return

    print(f"Found {len(state_files)} trajectories. Checking for duplicates...\n")

    duplicates_found = 0
    
    for i in range(len(state_files) - 1):
        file_a = state_files[i]
        file_b = state_files[i+1]

        try:
            with open(file_a, 'rb') as f:
                s0_a = pickle.load(f)['object_joint_angle_dicts']['robot']
            with open(file_b, 'rb') as f:
                s0_b = pickle.load(f)['object_joint_angle_dicts']['robot']
            
            # Use np.allclose to handle potential floating point precision noise
            if np.allclose(s0_a, s0_b):
                print(f"[DUPLICATE DETECTED]")
                print(f"  Path A: {file_a}")
                print(f"  Path B: {file_b}")
                print(f"  State:  {s0_a}\n")
                duplicates_found += 1
                
        except Exception as e:
            print(f"Error processing {file_a} or {file_b}: {e}")

    print("--- Summary ---")
    print(f"Total Trajectories: {len(state_files)}")
    print(f"Duplicates Found:   {duplicates_found}")

if __name__ == "__main__":
    # Update this to your local data directory
    DATA_DIR = "data/diverse_objects_all/41510/experiment/debug"
    check_trajectory_diversity(DATA_DIR)