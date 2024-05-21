import os
import argparse
from plot.plot_utils import read_and_group_data, tolerant_mean
import numpy as np
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("data_paths", type=str, nargs='*')
parser.add_argument("--read_train", type=int, default=0)
args = parser.parse_args()

data_dirs = args.data_paths
print(data_dirs)

def label_func(variant):
    if int(variant['task.env_runner.use_segmask']) == 1:
        return "_".join(["segmask", str(variant['task.dataset.train_ratio'])])
    if int(variant['task.env_runner.only_handle_points']) == 1:
        return "_".join(["zoomed-in", str(variant['task.dataset.train_ratio'])])

all_res = read_and_group_data(data_dirs, read_train=args.read_train, label_function=label_func)
for l_idx, label in enumerate(all_res):
    print(label)
    values = all_res[label]
    values, error = tolerant_mean(values)
    # values = np.array(values[0])
    x = np.arange(len(values)) #* eval_freq[0]
    if 'segmask' in label:
        plt.plot(x, values, "-*", label=label)
    else:
        plt.plot(x, values, "-o", linestyle='dashed', label=label)
    
plt.legend()
plt.tight_layout()
plt.savefig("tmp.png")
plt.show()