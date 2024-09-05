# bin/sh

singularity shell --bind /project_data/held/yufeiw2/RoboGen_sim2real/:/mnt/RoboGen_sim2real/ --nv /project_data/held/yufeiw2/robogen-dp3-act3d.sif

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

singularity shell --bind ./:/mnt/ch/ --nv /project_data/held/yufeiw2/robogen-dp3-act3d.sif
#
# Objects 48700
# Objects 45526
# Objects 45661
# Objects 45694
# Objects 45780
# Objects 45910
# Objects 45961
# Objects 46408
# Objects 46417
# Objects 46440
# Objects 46490
# Objects 46762
# Objects 46825
# Objects 46893
# Objects 47235
# Objects 47281
# Objects 47315
# Objects 47529
# Objects 47669
# Objects 47944
# Objects 48063
# Objects 48177
# Objects 48356
# Objects 48623
# Objects 48876
# Objects 49025
# Objects 49062
# Objects 49132
# Objects 49133
# Objects 40417
# Objects 41085
# Objects 41452
# Objects 45162
# Objects 45176
# Objects 45194
# Objects 45203
# Objects 45248
# Objects 45271
# Objects 45290
# Objects 45305