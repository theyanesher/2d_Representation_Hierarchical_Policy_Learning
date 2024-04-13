import os

autobot_user_name = "yufeiw2"
autobot_project_folder = "RoboGen_sim2real"
os.system("rsync -avrz --exclude='pytorch_sac/exp' --exclude='objaverse_utils/data/default_tag_embeddings_*.pt' --exclude='__pycache__' --exclude='experiment' --exclude='.hydra' --exclude='.vscode' --exclude='runs' --exclude='video' --exclude='video_rl_game' --exclude='data/icml*' --exclude='data/partnet' --exclude='data/PartManip_DataRelease.zip' ./ autobot:/project_data/held/{}/{}/".format(
    autobot_user_name, autobot_project_folder
))
