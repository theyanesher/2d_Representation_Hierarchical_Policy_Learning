import pickle5 as pickle

data_file = "data/dp3_demo/test_different_init_joint_angle_world/raw_data.pkl"
with open(data_file, "rb") as f:
    data = pickle.load(f)
    
pc_list, state_list, action_list, last_state_indices = data

