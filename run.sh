# python manipulation/old_test_opening_primitve.py

demo_name=0511-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
data_name=0512-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first-more-points-longer-horizon-depend
exp_folder=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle
task_beg_idx=0
task_end_idx=1
pointcloud_num=8000

# python 3d_diffusion_policy/extract_data_from_states_new.py --folder_name data/temp/ --object_name storagefurniture \
#     --save_path "data/dp3_demo/${data_name}" --exp_name "${demo_name}" \
#     --task_beg_idx "${task_beg_idx}" --task_end_idx "${task_end_idx}" \
#     --pointcloud_num "${pointcloud_num}" \
#     --use_extracted 0

cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy

horizon=8
n_obs_steps=4
exp_name=0512-vary-obj-loc-ori-init-more-points-longer-horizon-depend

python train.py --config-name=dp3.yaml task=robogen_open_door exp_name="${exp_name}" eval_first=0  \
    task.dataset.zarr_path="${PROJECT_DIR}/data/dp3_demo/${data_name}/" \
    task.env_runner.experiment_name="${demo_name}" \
    task.env_runner.experiment_folder="${exp_folder}" \
    horizon="${horizon}" n_obs_steps="${n_obs_steps}" 
# python eval_robogen_new.py --config-name=dp3.yaml task=robogen_open_door exp_name=eval
# python eval_robogen_new.py --config-name=dp3.yaml task=robogen_open_door exp_name=debug
# singularity shell --bind /project_data/held/yufeiw2/RoboGen_sim2real/:/mnt/RoboGen_sim2real/ --nv /project_data/held/yufeiw2/robogen-dp3.sif