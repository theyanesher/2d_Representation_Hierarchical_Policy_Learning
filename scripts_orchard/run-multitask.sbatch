#!/bin/bash

#SBATCH -N 1 # Number of nodes
#SBATCH --cpus-per-gpu=4 
#SBATCH -p preempt
#SBATCH --gpus=4 #GPU specification. H100
#SBATCH -t 48:00:00 # Estimated time, 48hour max. DD-HH:MM.
#SBATCH --job-name articubot_alone_in_articulated
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
# bash scripts_orchard/download_articubot_data2.sh
# bash scripts_orchard/download_cgn_data2.sh
# bash scripts_orchard/download_just_articulated_full.sh
bash scripts_orchard/download_articubot_data_new_pap.sh

### print all folders in /tmp sorted by newest first
target=/tmp/
echo "Folders in $target (newest first):"
find "$target" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort

cd test_PointNet2


torchrun --standalone --nproc_per_node=4 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs    \
    general.exp_name=full-pick-place-and-grasping-lift_1017-pap-new-1204_resumed \
    articubot.batch_size=70   \
    articubot.num_train_objects=aritucbot_new_cat_camera_random_close_pick_and_place_new_1204_grasping_1017 \
    general.tasks="['articubot']" \
    general.use_lr_scheduler=1 \
    general.load_model_path=/project/flame/yufeiw2/RoboGen-sim2real/test_PointNet2/exps/2025-12-05full-pick-place-and-grasping-lift_1017-pap-new-1204/model_245001.pth 


# torchrun --standalone --nproc_per_node=8 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs    \
#     general.exp_name=full-pick-place-and-grasping-lift_1017-pap-new-1204 \
#     articubot.batch_size=70   \
#     articubot.num_train_objects=aritucbot_new_cat_camera_random_close_pick_and_place_new_1204_grasping_1017 \
#     general.tasks="['articubot']" \
#     general.use_lr_scheduler = 1 \
#     general.load_model_path=/project/flame/yufeiw2/RoboGen-sim2real/test_PointNet2/exps/2025-09-14multinode-cgn-world-articubot-all-w-pick-place/model_155001.pth 




# torchrun --standalone --nproc_per_node=1 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs    \
#     general.exp_name=test_gcloud \
#     articubot.batch_size=70   \
#     articubot.num_train_objects=articulated   \
#     cgn.batch_size=15  \
#     cgn.OPTIMIZER.batch_size=15 \

# torchrun --standalone --nproc_per_node=6 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs \
#     general.exp_name=articubot_cgn_both_ln_gmm_articulated_load_articulated_model \
#     articubot.batch_size=70 \
#     articubot.gmm=1 \
#     articubot.num_train_objects=articulated \
#     cgn.batch_size=15 \
#     cgn.OPTIMIZER.batch_size=15 \
#     cgn.loss_scale=0.02 \
#     general.load_model_path=/project/flame/yufeiw2/RoboGen-sim2real/test_PointNet2/exps/GMM__2025-07-27articubot_cgn_layernorm/model_65000.pth 


# torchrun --standalone --nproc_per_node=6 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs \
#     general.exp_name=articubot_cgn_both_ln_gmm_50_load_50_model \
#     articubot.batch_size=70 \
#     articubot.gmm=1 \
#     cgn.batch_size=15 \
#     cgn.OPTIMIZER.batch_size=15 \
#     cgn.loss_scale=0.02 \
#     general.load_model_path=/project/flame/yufeiw2/RoboGen-sim2real/test_PointNet2/exps/GMM__2025-08-02articubot_cgn_both_ln_gmm_50/model_30000.pth 

# torchrun --standalone --nproc_per_node=6 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs \
#     general.exp_name=articubot_cgn_both_ln_wdp_50_continue_no_load_opt \
#     articubot.batch_size=70 \
#     cgn.batch_size=15 \
#     cgn.OPTIMIZER.batch_size=15 \
#     cgn.loss_scale=0.0002 \
#     general.load_model_path=/project/flame/yufeiw2/RoboGen-sim2real/test_PointNet2/exps/GMM__2025-07-29articubot_cgn_both_ln_wdp_50/model_60000.pth \
    # articubot.camera_frame=1

# torchrun --standalone --nproc_per_node=6 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs \
#     general.exp_name=articubot_cgn_both_ln_wdp_50_lang_as_input_continue_no_load_opt_loss_balance_larger_lr \
#     articubot.batch_size=50 \
#     cgn.batch_size=15 \
#     cgn.OPTIMIZER.batch_size=15 \
#     general.embedding_as_input=1 \
#     cgn.loss_scale=0.001 \
#     general.lr=0.001 \
#     general.load_model_path=/project/flame/yufeiw2/RoboGen-sim2real/test_PointNet2/exps/GMM__2025-07-29articubot_cgn_both_ln_wdp_50_lang_as_input/model_55000.pth 



# torchrun --standalone --nproc_per_node=3  train_multitask_ddp_weighted_displacement_gmm.py --arg_configs \
#     general.tasks="['articubot']" \
#     general.exp_name=articubot_alone_layernorm_wdp_50 \
#     articubot.batch_size=70 \
#     geral.category_embedding_type=None 

# torchrun --standalone --nproc_per_node=5  train_multitask_ddp_weighted_displacement_gmm.py --arg_configs \
#     general.exp_name=articubot_cgn_layernorm \
#     articubot.gmm=1 \
#     articubot.batch_size=70 \
#     articubot.num_train_objects=articulated \
#     cgn.batch_size=15 \
#     cgn.OPTIMIZER.batch_size=15 \
#     cgn.loss_scale=0.02 

# torchrun --standalone --nproc_per_node=5 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs \
#     general.exp_name=articubot_cgn_both_bn_gmm_just_lang_embed \
#     articubot.batch_size=70 \
#     articubot.gmm=1 \
#     general.replace_bn_with_ln=0 \
#     articubot.num_train_objects=articulated \
#     cgn.batch_size=15 \
#     cgn.OPTIMIZER.batch_size=15 \
#     cgn.loss_scale=0.02 
    

# torchrun --standalone --nproc_per_node=4 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs \
#     general.exp_name=articubot_cgn_in_not_fp_to_full \
#     articubot.batch_size=70 \
#     cgn.batch_size=15 \
#     cgn.OPTIMIZER.batch_size=15 \
#     general.fp_to_full=0 \
#     general.first_sa_point=2048 \
#     cgn.DATA.further_fps_point=0 \


# torchrun --standalone --nproc_per_node=4 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs \
#     general.exp_name=articubot_alone_in_articulated \
#     general.tasks="['articubot']" \
#     articubot.batch_size=140 \

# torchrun --standalone --nproc_per_node=5 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs \
    # general.exp_name=articubot_alone_in_articulated_gmm \
    # general.tasks="['articubot']" \
    # articubot.batch_size=130 \
    # articubot.gmm=1 \