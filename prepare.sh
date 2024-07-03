# bin/sh

export PATH=/opt/conda/bin:$PATH
source /opt/conda/etc/profile.d/conda.sh
conda activate unisim
export YUFEI_OPENAI_API_KEY="xxx" # TODO: embed this in singularity
export PYTHONPATH=${PWD}:$PYTHONPATH
export PYTHONPATH=${PWD}/rl_games:$PYTHONPATH

export PYTHONPATH=${PWD}:$PYTHONPATH
export PYTHONPATH=${PWD}/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy:$PYTHONPATH
export PROJECT_DIR=${PWD}
export NUMEXPR_MAX_THREADS=90
export HYDRA_FULL_ERROR=1
export YUFEI_OPENAI_API_KEY=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