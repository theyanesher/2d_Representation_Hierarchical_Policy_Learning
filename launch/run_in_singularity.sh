cd /mnt/RoboGen_sim2real
export PATH=/opt/conda/bin:$PATH
source /opt/conda/etc/profile.d/conda.sh
conda activate unisim
export YUFEI_OPENAI_API_KEY="xxx" # TODO: embed this in singularity
export PYTHONPATH=${PWD}:$PYTHONPATH
export PYTHONPATH=${PWD}/rl_games:$PYTHONPATH

# TODO: add the python command to generate the demos here. 
echo "index_min: $1"
echo "index_max: $2"
echo "run_times: $3"
echo "train_minutes: $4"
echo "cuda device: $5"

export CUDA_VISIBLE_DEVICES="$5"
python manipulation/scripts/generate_opening_demos.py --index_min "$1" --index_max "$2" --run_times "$3" --train_minutes "$4"