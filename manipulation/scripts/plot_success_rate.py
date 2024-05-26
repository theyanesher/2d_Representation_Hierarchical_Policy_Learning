from matplotlib import pyplot as plt
import os
import numpy as np
import json

eval_results = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/eval_results/0511-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first_2024.05.11_14.45.04_train_dp3_robogen_open_door_checkpoints_latest.ckpt/opened_joint_angles.json"
with open(eval_results, "r") as f:
    eval_results = json.load(f)

traj_info_data_dir = "data/dp3_demo/action-dist-0511-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first/action_dist"

fig, axes = plt.subplots(2, 4, figsize=(16, 4))
axes = axes.flatten()

good_traj_total_translations = []
good_traj_total_orientations = []
good_traj_average_translations = []
good_traj_average_orientations = []
good_traj_max_translations = []
good_traj_max_orientations = []
good_traj_reach_handle_times = []

bad_traj_total_translations = []
bad_traj_total_orientations = []
bad_traj_average_translations = []
bad_traj_average_orientations = []
bad_traj_max_translations = []
bad_traj_max_orientations = []
bad_traj_reach_handle_times = []


for experiment_name, res in eval_results.items():
    print("Processing", experiment_name)
    
    experiment = experiment_name.split("/")[-2]
    opened_angle = res[0]
    info = os.path.join(traj_info_data_dir, f"{experiment}.json")
    if not os.path.exists(info):
        print(f"Trajectory info {info} does not exist")
        continue
    
    with open(info, "r") as f:
        info = json.load(f)
    total_translation_for_reaching = info["total_translation_for_reaching"]
    total_orientation_for_reaching = info["total_orientation_for_reaching"] 
    reach_handle_time = info["reach_handle_time"]
    average_translation_for_reaching = total_translation_for_reaching / reach_handle_time
    average_orientation_for_reaching = total_orientation_for_reaching / reach_handle_time
    max_translation_for_reaching = info["max_translation_for_reaching"]
    max_orientation_for_reaching = info["max_orientation_for_reaching"]
    
    axes[0].scatter(total_translation_for_reaching, opened_angle)
    axes[1].scatter(total_orientation_for_reaching, opened_angle)
    axes[2].scatter(average_translation_for_reaching, opened_angle)
    axes[3].scatter(average_orientation_for_reaching, opened_angle)
    axes[4].scatter(max_translation_for_reaching, opened_angle)
    axes[5].scatter(max_orientation_for_reaching, opened_angle)
    axes[6].scatter(reach_handle_time, opened_angle)
    
    axes[0].set_title("Total translation for reaching")
    axes[1].set_title("Total orientation for reaching")
    axes[2].set_title("Average translation for reaching")
    axes[3].set_title("Average orientation for reaching")
    axes[4].set_title("Max translation for reaching")
    axes[5].set_title("Max orientation for reaching")
    axes[6].set_title("Reach handle time")
    
    if opened_angle > 0.1:
        good_traj_total_translations.append(total_translation_for_reaching)
        good_traj_total_orientations.append(total_orientation_for_reaching)
        good_traj_average_translations.append(average_translation_for_reaching)
        good_traj_average_orientations.append(average_orientation_for_reaching)
        good_traj_max_translations.append(max_translation_for_reaching)
        good_traj_max_orientations.append(max_orientation_for_reaching)
        good_traj_reach_handle_times.append(reach_handle_time)
    else:
        bad_traj_total_translations.append(total_translation_for_reaching)
        bad_traj_total_orientations.append(total_orientation_for_reaching)
        bad_traj_average_translations.append(average_translation_for_reaching)
        bad_traj_average_orientations.append(average_orientation_for_reaching)
        bad_traj_max_translations.append(max_translation_for_reaching)
        bad_traj_max_orientations.append(max_orientation_for_reaching)
        bad_traj_reach_handle_times.append(reach_handle_time)
    
    for i in range(7):
        axes[i].set_ylabel("Opened angle")

plt.tight_layout()
plt.show()
    
plt.close("all")
# box plot of the good and bad trajectory statistics
fig, axes = plt.subplots(2, 4, figsize=(16, 4))
axes = axes.flatten()

axes[0].boxplot(good_traj_total_translations, positions=[1], widths=0.6, patch_artist=True, showfliers=False)
axes[0].boxplot(bad_traj_total_translations, positions=[2], widths=0.6, patch_artist=True, showfliers=False)

axes[1].boxplot(good_traj_total_orientations, positions=[1], widths=0.6, patch_artist=True, showfliers=False)
axes[1].boxplot(bad_traj_total_orientations, positions=[2], widths=0.6, patch_artist=True, showfliers=False)

axes[2].boxplot(good_traj_max_translations, positions=[1], widths=0.6, patch_artist=True, showfliers=False)
axes[2].boxplot(bad_traj_max_translations, positions=[2], widths=0.6, patch_artist=True, showfliers=False)

axes[3].boxplot(good_traj_max_orientations, positions=[1], widths=0.6, patch_artist=True, showfliers=False)
axes[3].boxplot(bad_traj_max_orientations, positions=[2], widths=0.6, patch_artist=True, showfliers=False)

axes[4].boxplot(good_traj_average_translations, positions=[1], widths=0.6, patch_artist=True, showfliers=False)
axes[4].boxplot(bad_traj_average_translations, positions=[2], widths=0.6, patch_artist=True, showfliers=False)

axes[5].boxplot(good_traj_average_orientations, positions=[1], widths=0.6, patch_artist=True, showfliers=False)
axes[5].boxplot(bad_traj_average_orientations, positions=[2], widths=0.6, patch_artist=True, showfliers=False)

axes[6].boxplot(good_traj_reach_handle_times, positions=[1], widths=0.6, patch_artist=True, showfliers=False)
axes[6].boxplot(bad_traj_reach_handle_times, positions=[2], widths=0.6, patch_artist=True, showfliers=False)

axes[0].set_title("Total translation for reaching")
axes[1].set_title("Total orientation for reaching")
axes[2].set_title("Max translation for reaching")
axes[3].set_title("Max orientation for reaching")
axes[4].set_title("Average translation for reaching")
axes[5].set_title("Average orientation for reaching")
axes[6].set_title("Reach handle time")

for i in range(7):
    axes[i].set_xticks([1, 2])
    axes[i].set_xticklabels(["Good", "Bad"])
    axes[i].set_ylabel("opened angle")

plt.tight_layout()
plt.show()
    
    