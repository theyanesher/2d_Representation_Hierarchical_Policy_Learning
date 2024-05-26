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
    if int(variant['task.env_runner.use_segmask']) == 1:
        return "_".join(["segmask", str(variant['task.dataset.train_ratio'])])
    if int(variant['task.env_runner.only_handle_points']) == 1:
        return "_".join(["zoomed-in", str(variant['task.dataset.train_ratio'])])

all_res = read_and_group_data(data_dirs, read_train=args.read_train, label_function=label_func)

### line plot
# for l_idx, label in enumerate(all_res):
#     print(label)
#     values = all_res[label]
#     values, error = tolerant_mean(values)

#     x = np.arange(1, len(values) + 1) #* eval_freq[0]
#     if 'segmask' in label:
#         plt.plot(x, values, "-*", label=label.replace("segmask", "full_pc"))
#     else:
#         plt.plot(x, values, "-o", linestyle='dashed', label=label)
# plt.xlabel("epoch * 500")
# plt.ylabel("normalized performance")
    
    
### bar plot
for idx, ratio in enumerate([0.2, 0.4, 0.6, 0.8, 1]):
    cprint(f"ratio: {ratio}", 'green')
    label = f"segmask_{ratio}"
    values = all_res[label]
    print(values)
    if len(values) > 0:
        values, error = tolerant_max(values)
        if idx == 0:
            plt.bar(idx * 2, values.max(), label="full pc", width=0.5, color='tab:blue')
        else:
            plt.bar(idx * 2, values.max(), width=0.5, color='tab:blue')
    
    label = f"zoomed-in_{ratio}"
    values = all_res[label]
    print(values)
    if len(values) > 0:
        values, error = tolerant_max(values)
        if idx == 0:
            plt.bar(idx * 2 + 0.5, values.max(), label='zoomed-in', width=0.5, color='tab:orange')
        else:
            plt.bar(idx * 2 + 0.5, values.max(),  width=0.5, color='tab:orange')

plt.xticks(np.array([0, 2, 4, 6, 8]) + 0.25, ["0.2", "0.4", "0.6", "0.8", "1.0"])    
plt.xlabel("train ratio")
plt.ylabel("normalized performance")

plt.legend()
plt.tight_layout()
plt.savefig("tmp.png")
plt.show()