import h5py
from matplotlib import pyplot as plt
import numpy as np

f = h5py.File('data/rgb/41510/2025-10-30-21-05-53.h5')


actions = f['action'][:]
print("actions", actions.shape)

for key in f['obs'].keys():
    print(f"obs/{key}", f[f'obs/{key}'][:].shape, f[f'obs/{key}'].dtype) 
    print("min", f[f'obs/{key}'][:].min(), "max", f[f'obs/{key}'][:].max(), "mean", f[f'obs/{key}'][:].mean(), "std", f[f'obs/{key}'][:].std())
    print()