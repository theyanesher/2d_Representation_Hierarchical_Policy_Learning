# python manipulation/old_test_opening_primitve.py

demo_name=0511-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
save_data_name=0527-act3d-always-close
exp_folder=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle
task_beg_idx=0
task_end_idx=1
observation_mode=act3d
pointcloud_num=4500

# python 3d_diffusion_policy/extract_data_from_states_2.py --folder_name data/temp/ --object_name storagefurniture \
#     --save_path "data/dp3_demo/${save_data_name}" --exp_name "${demo_name}" \
#     --task_beg_idx "${task_beg_idx}" --task_end_idx "${task_end_idx}" \
#     --pointcloud_num "${pointcloud_num}" \
#     --use_extracted 0 \
#     --num_experiment 1000 \
#     --observation_mode "${observation_mode}" \
#     --parallel 0

cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy

horizon=4
n_obs_steps=2
train_ratio=0.22
exp_name="0528-act3d-train-ratio-${train_ratio}"
# exp_name="debug"

action_dim=10
agent_pos_dim=10
pc_channel=3

python train.py --config-name=dp3.yaml task=robogen_open_door exp_name="${exp_name}" eval_first=0  \
    task.dataset.zarr_path="${PROJECT_DIR}/data/dp3_demo/${save_data_name}/" \
    task.env_runner.demo_experiment_path="${PROJECT_DIR}/data/dp3_demo/${save_data_name}/" \
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
    task.env_runner.observation_mode="${observation_mode}" \
    task.env_runner.observation_mode="${observation_mode}" \
    policy.encoder_type=act3d \
    policy.encoder_output_dim=60 \
    task.dataset.enumerate=True \
    training.rollout_every=200 \
    training.checkpoint_every=200 \
    load_checkpoint_path=/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0528-act3d-train-ratio-0.22/2024.05.28/16.11.34_train_dp3_robogen_open_door/checkpoints/latest.ckpt





# python eval_robogen_new.py --config-name=dp3.yaml task=robogen_open_door exp_name=eval
# python eval_robogen_new.py --config-name=dp3.yaml task=robogen_open_door exp_name=debug
# singularity shell --bind /project_data/held/yufeiw2/RoboGen_sim2real/:/mnt/RoboGen_sim2real/ --nv /project_data/held/yufeiw2/robogen-dp3.sif