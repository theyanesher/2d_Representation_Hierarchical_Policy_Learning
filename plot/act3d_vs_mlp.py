import os
import argparse
from plot.plot_utils import read_and_group_data, tolerant_mean, tolerant_max
import numpy as np
from matplotlib import pyplot as plt
from termcolor import cprint

parser = argparse.ArgumentParser()
parser.add_argument("data_paths", type=str, nargs='*')
parser.add_argument("--read_train", type=int, default=0)
args = parser.parse_args()

data_dirs = args.data_paths
print(data_dirs)

def label_func(variant):
    return variant.get("policy.encoder_type", "mlp")

all_res = read_and_group_data(data_dirs, read_train=args.read_train, label_function=label_func, mean=False)

for l_idx, label in enumerate(all_res):
    print(label)
    values = all_res[label]
    if label == 'mlp':
        first_eval_value = max([np.mean(x) for x in values[0]])
        best_idx = np.argmax([np.mean(x) for x in values[0]])
        print(f"{label} mean: {first_eval_value} best_idx: {best_idx}")
    elif label == 'act3d':
        first_eval_value = values[1][1]
    # plt.plot(first_eval_value, label=label)
    plt.axhline(y=np.mean(first_eval_value), label=f"{label} mean", linestyle='dashed')
    print(f"{label} mean: {np.mean(first_eval_value)}")

plt.legend()
plt.show()