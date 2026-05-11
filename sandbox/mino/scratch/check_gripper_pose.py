# checks if train and test states are identical, should be 0 or else it means training and test sets have overlap

import h5py 
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from equi_diffpo.model.common.rotation_transformer import RotationTransformer
from third_party.robogen.test_PointNet2.model_invariant import PointNet2_super, PointNet2_small2


model = PointNet2_small2(60, 6)
# total parameters (all)
total_params = sum(p.numel() for p in model.parameters())
# trainable parameters only
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total params:       {total_params:,} ({total_params/1e6:.2f}M)")
print(f"Trainable params:   {trainable_params:,} ({trainable_params/1e6:.2f}M)")


ROOT = Path('/data/minon/tax3d-conditioned-mimicgen/data/robomimic/datasets/')

if __name__ == '__main__':
    task = 'square_d2'

    train_hdf5 = h5py.File(ROOT / task / f'{task}_pcd_abs.hdf5', 'r')
    demo = train_hdf5['data/demo_0']

    q2aa_transformer = RotationTransformer('quaternion', 'axis_angle')
    r6d2aa_transformer = RotationTransformer('rotation_6d', 'axis_angle')
    

    # load the full N×3 arrays
    xyz_actions   = demo['actions'][:,:3]              # shape (N,3)
    aa_actions = demo['actions'][:,3:6]
    agent_pos = demo['obs']['agent_pos'][:,:3]     # shape (N,3)
    # agent_aa = q2aa_transformer.forward(np.asarray(demo['obs']['robot0_eef_quat'][:]))
    agent_aa = r6d2aa_transformer.forward(np.asarray(demo['obs']['agent_pos'][:,3:9]))

    t = np.arange(xyz_actions.shape[0])
    plt.figure(figsize=(6,4))
    for d in range(3):
        plt.plot(t, agent_pos[:,d], '-' , label=f'pos {d}')
        plt.plot(t, xyz_actions[:,d]  , '--', label=f'act {d}')
    plt.xlabel('timestep')
    plt.ylabel('value')
    plt.title('Time series of positions and actions')
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.show()

    # new: axis‐angle components of agent vs. action
    plt.figure(figsize=(6,4))
    for d in range(3):
        plt.plot(t, aa_actions[:,d], '-', label=f'agent aa {d}')
        plt.plot(t, agent_aa[:,d], '--', label=f'action aa {d}')
    plt.xlabel('timestep')
    plt.ylabel('axis‐angle component')
    plt.title('Axis–angle of agent vs. action')
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.show()