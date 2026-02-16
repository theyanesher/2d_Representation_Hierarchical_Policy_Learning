import matplotlib
matplotlib.use("Agg")
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Load ---
with open("just_simulator_result_check_AFTER_REVERSE_TRANSFORM_AFTER.pkl", "rb") as f: # just_simulator_result_check.pkl
    data = pickle.load(f)

gripper_pcd = data["gripper"].reshape(1, -1) # .detach().cpu().numpy()#data["gripper"].reshape(1, -1).detach().cpu().numpy() np.array([ 0.27851272, -0.0081652 ,  1.4719224 ]) 
pcd = data["pointcloud"] # .detach().cpu().numpy()
initial_pcd = np.array( [0.25400782, -0.00559501,  1.39805841]) # array([-0.23735051, -0.13573062,  0.68349177]) [ 0.27851272, -0.0081652 ,  1.4719224 ]
import pdb; pdb.set_trace();
# --- Plot ---
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# point cloud
ax.scatter(pcd[:,0], pcd[:,1], pcd[:,2], s=2)
# gripper marker in red
ax.scatter(gripper_pcd[:, 0], gripper_pcd[:, 1], gripper_pcd[:, 2],
           color='red', s=50, label="predicted_gripper")
ax.scatter(initial_pcd[0], initial_pcd[1], initial_pcd[2],
           color='black', s=50, label="initial_gripper")


ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()

# Save instead of show
plt.savefig("pcd_and_gripper_SIMULATO_CHECK_AFTER_REVERSE_TRANSFORM_AFTER.png", dpi=300, bbox_inches='tight')
plt.close(fig)



