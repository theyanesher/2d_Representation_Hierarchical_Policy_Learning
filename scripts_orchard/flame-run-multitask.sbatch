#!/bin/bash

#SBATCH -N 1 # Number of nodes
#SBATCH --cpus-per-gpu=4          # tune for your dataloaders
#SBATCH --partition=flame
#SBATCH --qos=flame-16gpu_qos
#SBATCH --account=dheld
#SBATCH --gpus=8 #GPU specification. H100
#SBATCH -t 200:00:00 # Estimated time
#SBATCH --job-name articubot-cgn-multivariance
#SBATCH -o job_%j.out
#SBATCH -e job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=yufeiw2@andrew.cmu.edu


source "/project/flame/yufeiw2/miniconda3/etc/profile.d/conda.sh"
conda activate articubot
cd /project/flame/yufeiw2/RoboGen-sim2real/
source prepare.sh
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export OMP_NUM_THREADS=2    # often helps dataloaders
export MKL_NUM_THREADS=2

# bash scripts_orchard/download_articubot_data.sh
bash scripts_orchard/download_articubot_data2.sh
# bash scripts_orchard/download_cgn_data2.sh
# bash scripts_orchard/download_just_articulated_full.sh
# bash /project/flame/yufeiw2/RoboGen-sim2real/scripts_orchard/download_articubot_data_just_pap.sh
# bash /project/flame/yufeiw2/RoboGen-sim2real/scripts_orchard/download_articubot_data_just_open_and_reset.sh
# bash /project/flame/yufeiw2/RoboGen-sim2real/scripts_orchard/download_articubot_data_just_pap_grasp.sh
# bash /project/flame/yufeiw2/RoboGen-sim2real/scripts_orchard/download_articubot_data_just_pap_grasp_and_grasp_for_pnp.sh
# bash /project/flame/yufeiw2/RoboGen-sim2real/scripts_orchard/download_articubot_data_just_pap_grasp_1204_place.sh
# bash scripts_orchard/download_articubot_data_new_pap.sh
# bash /project/flame/yufeiw2/RoboGen-sim2real/scripts_orchard/download_articubot_data_just_grasping.sh

### print all folders in /tmp sorted by newest first
target=/tmp/
echo "Folders in $target (newest first):"
find "$target" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort

cd test_PointNet2

torchrun --standalone --nproc_per_node=8 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs    \
    general.exp_name=full-new-pick-place-0110 \
    articubot.batch_size=70   \
    articubot.num_train_objects=aritucbot_new_cat_camera_random_close_pick_and_place_0103_grasping_1017 \
    general.tasks="['articubot']" \
    general.use_lr_scheduler=1 \
    general.load_model_path=/project/flame/yufeiw2/RoboGen-sim2real/test_PointNet2/exps/2026-01-02full-new-pick-place-0101/model_1070001.pth 


# torchrun --standalone --nproc_per_node=8 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs    \
#     general.exp_name=full-new-pick-place-0101 \
#     articubot.batch_size=70   \
#     articubot.num_train_objects=aritucbot_new_cat_camera_random_close_pick_and_place_0101_grasping_1017 \
#     general.tasks="['articubot']" \
#     general.use_lr_scheduler=1 \
#     general.load_model_path=/project/flame/yufeiw2/RoboGen-sim2real/test_PointNet2/exps/2025-09-14multinode-cgn-world-articubot-all-w-pick-place/model_155001.pth 


# torchrun --standalone --nproc_per_node=8 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs    \
#     general.exp_name=test_attn_grasping\
#     articubot.batch_size=70   \
#     articubot.num_train_objects=grasping \
#     general.tasks="['articubot']" \
#     general.use_lr_scheduler=1 \
#     general.policy_class=pointnet2_attn 

# torchrun --standalone --nproc_per_node=8 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs    \
#     general.exp_name=test_attn_articubot_50\
#     articubot.batch_size=70   \
#     articubot.num_train_objects=50 \
#     general.tasks="['articubot']" \
#     general.use_lr_scheduler=1 \
#     general.policy_class=pointnet2_attn \
#     articubot.dataset_prefix=/tmp/dp3_demo_clean_distorted_goal

# torchrun --standalone --nproc_per_node=8 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs    \
#     general.exp_name=just-pick-and-place-new-grasp-1204-place \
#     articubot.batch_size=70   \
#     articubot.num_train_objects=pick_and_place_new_grasp_and_old_place \
#     general.tasks="['articubot']" \
#     general.use_lr_scheduler=1 

# torchrun --standalone --nproc_per_node=8 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs    \
#     general.exp_name=just-pick-and-place-grasp-plus-grasp-pap \
#     articubot.batch_size=70   \
#     articubot.num_train_objects=pick_and_place_grasp_plus_grasp_pap \
#     general.tasks="['articubot']" \
#     general.use_lr_scheduler=1 

# torchrun --standalone --nproc_per_node=8 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs    \
#     general.exp_name=just-pick-and-place-grasp \
#     articubot.batch_size=70   \
#     articubot.num_train_objects=pick_and_place_grasp \
#     general.tasks="['articubot']" \
#     general.use_lr_scheduler=1 

# torchrun --standalone --nproc_per_node=8 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs    \
#     general.exp_name=just-articubot-plus-reset \
#     articubot.batch_size=70   \
#     articubot.num_train_objects=articubot_and_reset \
#     general.tasks="['articubot']" \
#     general.use_lr_scheduler=1 

# torchrun --standalone --nproc_per_node=8 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs    \
#     general.exp_name=just-pap-new-1204_no_one_hot_goal_open \
#     articubot.batch_size=70   \
#     articubot.num_train_objects=pick_and_place_new_1024 \
#     general.tasks="['articubot']" \
#     general.use_lr_scheduler=1 \
#     articubot.goal_always_open=1 

# torchrun --standalone --nproc_per_node=8 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs    \
#     general.exp_name=just-pap-new-1204_one_hot_goal_open \
#     articubot.batch_size=70   \
#     articubot.num_train_objects=pick_and_place_new_1024 \
#     general.tasks="['articubot']" \
#     general.use_lr_scheduler=1 \
#     general.add_one_hot_encoding=1 \
#     articubot.goal_always_open=1 
    # general.load_model_path=/project/flame/yufeiw2/RoboGen-sim2real/test_PointNet2/exps/2025-12-13full-pick-place-and-grasping-lift_1017-pap-new-1204_one_hot/model_245001.pth 


# torchrun --standalone --nproc_per_node=8 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs    \
#     general.exp_name=just-pap-new-1204_one_hot_resumed \
#     articubot.batch_size=70   \
#     articubot.num_train_objects=pick_and_place_new_1024 \
#     general.tasks="['articubot']" \
#     general.use_lr_scheduler=1 \
#     general.add_one_hot_encoding=1 \
#     general.load_model_path=/project/flame/yufeiw2/RoboGen-sim2real/test_PointNet2/exps/2025-12-13full-pick-place-and-grasping-lift_1017-pap-new-1204_one_hot/model_245001.pth 

 

# torchrun --standalone --nproc_per_node=8 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs    \
#     general.exp_name=full-pick-place-and-grasping-lift_1017-pap-new-1204_one_hot_resumed \
#     articubot.batch_size=70   \
#     articubot.num_train_objects=aritucbot_new_cat_camera_random_close_pick_and_place_new_1204_grasping_1017 \
#     general.tasks="['articubot']" \
#     general.use_lr_scheduler=1 \
#     general.add_one_hot_encoding=1 \
#     general.load_model_path=/project/flame/yufeiw2/RoboGen-sim2real/test_PointNet2/exps/2025-12-13full-pick-place-and-grasping-lift_1017-pap-new-1204_one_hot/model_245001.pth 

 
# torchrun --standalone --nproc_per_node=8 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs    \
#     general.exp_name=full-pick-place-and-grasping-lift_1017-pap-new-1204_one_hot_initialize_from_no_one_hot \
#     articubot.batch_size=70   \
#     articubot.num_train_objects=aritucbot_new_cat_camera_random_close_pick_and_place_new_1204_grasping_1017 \
#     general.tasks="['articubot']" \
#     general.use_lr_scheduler=1 \
#     general.add_one_hot_encoding=1 \
#     general.load_model_path=/project/flame/yufeiw2/RoboGen-sim2real/test_PointNet2/exps/2025-12-05full-pick-place-and-grasping-lift_1017-pap-new-1204/model_245001.pth 


# torchrun --standalone --nproc_per_node=4 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs    \
#     general.exp_name=full-pick-place-and-grasping-lift_1017-pap-new-1204_resumed \
#     articubot.batch_size=70   \
#     articubot.num_train_objects=aritucbot_new_cat_camera_random_close_pick_and_place_new_1204_grasping_1017 \
#     general.tasks="['articubot']" \
#     general.use_lr_scheduler=1 \
#     general.load_model_path=/project/flame/yufeiw2/RoboGen-sim2real/test_PointNet2/exps/2025-12-05full-pick-place-and-grasping-lift_1017-pap-new-1204/model_245001.pth 

# torchrun --standalone --nproc_per_node=8 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs    \
#     general.exp_name=full-pick-place-and-grasping-lift_1017-pap-new-1204 \
#     articubot.batch_size=70   \
#     articubot.num_train_objects=aritucbot_new_cat_camera_random_close_pick_and_place_new_1204_grasping_1017 \
#     general.tasks="['articubot']" \
#     general.use_lr_scheduler=1 \
#     general.load_model_path=/project/flame/yufeiw2/RoboGen-sim2real/test_PointNet2/exps/2025-09-14multinode-cgn-world-articubot-all-w-pick-place/model_155001.pth 

# torchrun --standalone --nproc_per_node=8 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs    \
#     general.exp_name=baseline_ours_articulated_full \
#     articubot.batch_size=70   \
#     articubot.num_train_objects=articulated_full \
#     general.tasks="['articubot']" 

# torchrun --standalone --nproc_per_node=8 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs    \
#     general.exp_name=full-pick-place-and-grasping-lift_1017-no-cgn \
#     articubot.batch_size=70   \
#     articubot.num_train_objects=aritucbot_new_cat_camera_random_close_pick_and_place_more_1005_grasping_1017 \
#     general.tasks="['articubot']" \
#     general.load_model_path=/project/flame/yufeiw2/RoboGen-sim2real/test_PointNet2/exps/2025-09-14multinode-cgn-world-articubot-all-w-pick-place/model_155001.pth 


# torchrun --standalone --nproc_per_node=8 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs    \
#     general.exp_name=full-pick-place-and-grasping-lift_1017-and-cgn \
#     articubot.batch_size=70   \
#     articubot.num_train_objects=aritucbot_new_cat_camera_random_close_pick_and_place_more_1005_grasping_1017 \
#     cgn.batch_size=15  \
#     cgn.OPTIMIZER.batch_size=15 \
#     general.load_model_path=/project/flame/yufeiw2/RoboGen-sim2real/test_PointNet2/exps/2025-09-14multinode-cgn-world-articubot-all-w-pick-place/model_155001.pth 



# torchrun --standalone --nproc_per_node=8 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs    \
#     general.exp_name=full-pick-place-and-low-level-grasping_1009 \
#     articubot.batch_size=70   \
#     articubot.num_train_objects=aritucbot_new_cat_camera_random_close_pick_and_place_more_1005_grasping_1009 \
#     general.tasks="['articubot']" \
#     general.load_model_path=/project/flame/yufeiw2/RoboGen-sim2real/test_PointNet2/exps/2025-09-14multinode-cgn-world-articubot-all-w-pick-place/model_155001.pth 


# sleep 30000
# torchrun --standalone --nproc_per_node=8 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs    \
#     general.exp_name=debug \
#     articubot.batch_size=70   \
#     articubot.num_train_objects=debug \
#     general.tasks="['articubot']"
    # cgn.batch_size=15  \
    # cgn.OPTIMIZER.batch_size=15 \
    # general.load_model_path=/project/flame/yufeiw2/RoboGen-sim2real/test_PointNet2/exps/2025-08-14articubot_cgn_ln_gmm_multivariance_full_450_plus_dagger/model_300000.pth


# torchrun --standalone --nproc_per_node=8 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs    \
#     general.exp_name=articubot_cgn_ln_gmm_multivariance_full_450_plus_dagger_plus_random_cam_plus_close \
#     articubot.batch_size=70   \
#     articubot.num_train_objects=articulated_full_dagger \
#     cgn.batch_size=15  \
#     cgn.OPTIMIZER.batch_size=15 \
#     general.load_model_path=/project/flame/yufeiw2/RoboGen-sim2real/test_PointNet2/exps/2025-08-14articubot_cgn_ln_gmm_multivariance_full_450_plus_dagger/model_300000.pth

# torchrun --standalone --nproc_per_node=8 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs    \
#     general.exp_name=articubot_cgn_ln_gmm_multivariance_full_450_plus_dagger_load \
#     articubot.batch_size=60   \
#     articubot.num_train_objects=articulated_full_dagger \
#     cgn.batch_size=12  \
#     cgn.OPTIMIZER.batch_size=12 \
#     general.load_model_path=/project/flame/yufeiw2/RoboGen-sim2real/test_PointNet2/exps/2025-08-14articubot_cgn_ln_gmm_multivariance_full_450_plus_dagger/model_300000.pth