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


# TODO: change this to dp3 training parameters
echo "exp_name: $1"
echo "dataset_name: $2"
echo "in_gripper_frame: $3"
echo "cuda device: $4"

export CUDA_VISIBLE_DEVICES="$4"
cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy
python train.py --config-name=dp3.yaml task=robogen_open_door exp_name="$1" \
 task.dataset.zarr_path="${PROJECT_DIR}/data/dp3_demo/${$2}" \
 task.in_gripper_frame="$3"