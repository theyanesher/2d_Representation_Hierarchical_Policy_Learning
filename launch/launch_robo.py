import os

autobot_user_name = "ywang59"
autobot_project_folder = "RoboGen_sim2real"
os.system("rsync -avrz --exclude='pytorch_sac/exp' --exclude='data/dp3_demo' --exclude='data/local/' --exclude='data/'  --exclude='3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data' --exclude='objaverse_utils/data/default_tag_embeddings_*.pt' --exclude='__pycache__' --exclude='experiment' --exclude='.hydra' --exclude='.vscode' --exclude='runs' --exclude='video' --exclude='video_rl_game' --exclude='data/icml*' --exclude='data/partnet' --exclude='data/PartManip_DataRelease.zip' --exclude='data/dataset' --exclude='data/diverse*' ./ ywang59@bridges2.psc.edu:/ocean/projects/cis240052p/{}/{}/".format(
    autobot_user_name, autobot_project_folder
))
# --exclude='run-act3d.sh' --exclude='run-act3d-ddp.sh'
