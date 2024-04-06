import os

autobot_user_name = "yufeiw2"
os.system("rsync -avrz --exclude='__pycache__' --exclude='.hydra' --exclude='.vscode' --exclude='runs' --exclude='video' --exclude='video_rl_game' --exclude='data/icml*' --exclude='data/partnet' --exclude='data/PartManip_DataRelease.zip' ./ autobot:/project_data/held/{}/Robogen-sim2real/".format(
    autobot_user_name
))
