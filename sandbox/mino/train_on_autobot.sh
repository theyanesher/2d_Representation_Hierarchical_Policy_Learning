source /mnt/RoboGen_sim2real/sandbox/mino/init_singularity.sh
source /mnt/RoboGen_sim2real/prepare.sh

cd /mnt/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy

# run training script
torchrun --standalone --nproc_per_node=4 train_ddp.py --config-name=train_low_level_200_objects.yaml
