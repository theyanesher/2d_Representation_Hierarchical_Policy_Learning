# checks if train and test states are identical, should be 0 or else it means training and test sets have overlap

import h5py 
from pathlib import Path
import numpy as np

ROOT = Path('/data/minon/tax3d-conditioned-mimicgen/data/robomimic/datasets/')

if __name__ == '__main__':
    task = 'square_d2'
    train_hdf5 = h5py.File(ROOT / task / f'{task}_pcd_abs.hdf5', 'r')
    test_hdf5 = h5py.File(ROOT / task / 'for_oracle' / f'{task}_150_pcd_abs.hdf5', 'r')
    train_states = [np.asarray(train_hdf5[f'data/demo_{i}/states/states'][0]) for i in range(len(train_hdf5['data']))]
    test_states = [np.asarray(test_hdf5[f'data/demo_{1000 + i}/states/states'][0]) for i in range(len(test_hdf5['data']))]
    train_states = np.asarray(train_states); test_states = np.asarray(test_states)
    train_states = np.expand_dims(train_states, axis=1); test_states = np.expand_dims(test_states, axis=0)
    diff = np.abs(train_states - test_states)
    diff = np.sum(diff, axis=-1)
    identical_mask = (diff == 0)
    print(identical_mask.sum())
