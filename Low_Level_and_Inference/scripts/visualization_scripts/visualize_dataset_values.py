import h5py
from matplotlib import pyplot as plt
import numpy as np


f = h5py.File('data/debug_rgb/41510/2025-10-30-21-05-53.h5')
actions = f['action'][:]

T = actions.shape[0]
dim = actions.shape[1]

# Create a time axis (frames or seconds)
time = np.arange(T)

T, dim = actions.shape
time = np.arange(T)

# Create a figure with 10 vertical subplots
fig, axes = plt.subplots(dim, 1, figsize=(10, 2 * dim), sharex=True)

for i in range(dim):
    axes[i].plot(time, actions[:, i], color='tab:blue')
    axes[i].set_ylabel(f'Dim {i}')
    axes[i].grid(True, alpha=0.3)
    
# Label the bottom-most plot
axes[-1].set_xlabel('Timestep')

plt.suptitle('Action Trajectories (10 Dimensions)', fontsize=14, y=1.0)
plt.tight_layout()
plt.show()