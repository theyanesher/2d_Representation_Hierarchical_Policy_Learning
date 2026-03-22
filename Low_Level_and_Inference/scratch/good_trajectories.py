import os
import json

def analyze_trajectories(base_path):
    good_trajectories = []
    bad_trajectories = []

    # Iterate through each item in the directory
    for folder_name in sorted(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder_name)
        
        # Check if the path is a directory
        if os.path.isdir(folder_path):
            json_path = os.path.join(folder_path, 'label.json')
            
            # Ensure the label.json file exists
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    try:
                        data = json.load(f)
                        # Categorize based on 'good_traj' boolean
                        if data.get('good_traj') is True:
                            good_trajectories.append(folder_name)
                        else:
                            bad_trajectories.append(folder_name)
                    except json.JSONDecodeError:
                        print(f"Error: Could not decode JSON in {folder_name}")
            else:
                print(f"Warning: No label.json found in {folder_name}")

    # Output Results
    print("--- Results ---")
    print("\nGood Trajectories:")
    for traj in good_trajectories:
        print(f" [✓] {traj}")

    print("\nBad Trajectories:")
    for traj in bad_trajectories:
        print(f" [x] {traj}")

    print("\n--- Summary ---")
    print(f"Total Good: {len(good_trajectories)}")
    print(f"Total Bad:  {len(bad_trajectories)}")
    print(f"Total Processed: {len(good_trajectories) + len(bad_trajectories)}")

# Define the path
path = "data/diverse_objects_all/41510/experiment/debug_eval"
analyze_trajectories(path)