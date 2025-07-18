# bin/sh

singularity shell --bind /project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/:/mnt/RoboGen_sim2real/ --nv /project_data/held/yufeiw2/uni3d_articubot.sif

cd /mnt/RoboGen_sim2real
export PATH=/opt/conda/bin:$PATH
source /opt/conda/etc/profile.d/conda.sh
conda activate unisim
export PYTHONPATH=${PWD}:$PYTHONPATH
export PYTHONPATH=${PWD}/rl_games:$PYTHONPATH
export PYTHONPATH=${PWD}/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy:$PYTHONPATH
export PROJECT_DIR=${PWD}
source prepare.sh
export YUFEI_OPENAI_API_KEY="xxx" # TODO: embed this in singularity
cd test_PointNet2

singularity shell --bind ./:/mnt/ch/ --nv /project_data/held/yufeiw2/robogen-dp3-act3d.sif

torchrun --standalone --nproc_per_node=8 train_ddp_weighted_displacement_gmm.py --batch_size 50     --num_epochs 61 --model_type pointnet2_super --model_invariant     --exp_path /project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps/GMM/     --num_train_objects 50     --dataset_prefix /scratch/yufeiw2/dp3_demo     --exp_name _fixed_variance_0.001 --fixed_variance 0.001