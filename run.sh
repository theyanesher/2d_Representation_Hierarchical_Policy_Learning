# python manipulation/old_test_opening_primitve.py

demo_name=0511-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
# data_name=0512-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first-more-points-longer-horizon-depend
# save_data_name=0513-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first-joint-angle-action
# save_data_name=0513-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first-with-seg-mask
# save_data_name=0513-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first-only-handle-points
save_data_name=0514-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first-only-handle-points
exp_folder=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle
task_beg_idx=0
task_end_idx=1
use_joint_angle=0
use_segmask=0
only_handle_points=1
if [ $only_handle_points -eq 1 ]; then
    echo "only sample handle points"
    pointcloud_num=2000
else
    echo "not only handle points"
    pointcloud_num=4500
fi

# python 3d_diffusion_policy/extract_data_from_states_new.py --folder_name data/temp/ --object_name storagefurniture \
#     --save_path "data/dp3_demo/${save_data_name}" --exp_name "${demo_name}" \
#     --task_beg_idx "${task_beg_idx}" --task_end_idx "${task_end_idx}" \
#     --pointcloud_num "${pointcloud_num}" \
#     --use_extracted 1 --use_joint_angle "${use_joint_angle}" --use_segmask "${use_segmask}" \
#     --only_handle_points "${only_handle_points}" \
#     --num_experiment 2

cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy

horizon=4
n_obs_steps=2
train_ratio=1
# exp_name=0512-vary-obj-loc-ori-init-more-points-longer-horizon-depend
# exp_name=0513-vary-obj-loc-ori-init-joint-angle
# exp_name=0513-vary-obj-loc-ori-init-segmask
# exp_name=0513-vary-obj-loc-ori-only-handle-points
exp_name="0517-vary-obj-loc-ori-only-handle-points-correct-${train_ratio}"
if [ $use_joint_angle -eq 1 ]; then
    echo "use joint angle"
    action_dim=8
    agent_pos_dim=9
else
    echo "not use joint angle"
    action_dim=10
    agent_pos_dim=10
fi

if [ $use_segmask -eq 1 ]; then
    echo "use segmask"
    pc_channel=5
elif [ $only_handle_points -eq 1 ]; then
    echo "use only handle points"
    pc_channel=5
else
    echo "not use segmask"
    pc_channel=3
fi

python train.py --config-name=dp3.yaml task=robogen_open_door exp_name="${exp_name}" eval_first=0  \
    task.dataset.zarr_path="${PROJECT_DIR}/data/dp3_demo/${save_data_name}/" \
    task.env_runner.experiment_name="${demo_name}" \
    task.env_runner.experiment_folder="${exp_folder}" \
    task.env_runner.num_point_in_pc="${pointcloud_num}" \
    task.env_runner.use_joint_angle="${use_joint_angle}" \
    task.env_runner.use_segmask="${use_segmask}" \
    task.env_runner.only_handle_points="${only_handle_points}" \
    horizon="${horizon}" n_obs_steps="${n_obs_steps}" \
    task.shape_meta.obs.agent_pos.shape="[${agent_pos_dim}]" \
    task.shape_meta.action.shape="[${action_dim}]" \
    policy.pointcloud_encoder_cfg.in_channels="${pc_channel}" \
    task.dataset.train_ratio="${train_ratio}" \
    load_checkpoint_path=/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0517-vary-obj-loc-ori-only-handle-points-correct-1/2024.05.17/19.22.57_train_dp3_robogen_open_door/checkpoints/latest.ckpt \
    training.seed=43



# python eval_robogen_new.py --config-name=dp3.yaml task=robogen_open_door exp_name=eval
# python eval_robogen_new.py --config-name=dp3.yaml task=robogen_open_door exp_name=debug
# singularity shell --bind /project_data/held/yufeiw2/RoboGen_sim2real/:/mnt/RoboGen_sim2real/ --nv /project_data/held/yufeiw2/robogen-dp3.sif