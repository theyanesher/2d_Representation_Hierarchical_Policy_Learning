import matplotlib
matplotlib.use("Agg")
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Load ---
with open("check_low_level_inference_dict.pkl", "rb") as f:
    data = pickle.load(f)

gripper_pcd = data["gripper_idx_pose"][:3].reshape(1, -1)
pcd = data["pcd"]

# --- Plot ---
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# point cloud
ax.scatter(pcd[:,0], pcd[:,1], pcd[:,2], s=2)
predicted_pcd = np.array([ 0.2604, -0.0067,  1.4140])
# gripper marker in red
ax.scatter(gripper_pcd[:,0], gripper_pcd[:,1], gripper_pcd[:,2],
           color='red', s=50, label="gripper")

ax.scatter(predicted_pcd[0], predicted_pcd[1], predicted_pcd[2],
           color='black', s=50, label="predicted")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()

# Save instead of show
plt.savefig("pcd_and_gripper_INFERENCE_PIPELINE.png", dpi=300, bbox_inches='tight')
plt.close(fig)



