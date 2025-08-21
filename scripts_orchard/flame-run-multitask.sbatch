#!/bin/bash

#SBATCH -N 1 # Number of nodes
#SBATCH -n 20
#SBATCH --partition=flame
#SBATCH --qos=flame-t3_g3_qos
#SBATCH --gpus=8 #GPU specification. H100
#SBATCH -t 200:00:00 # Estimated time
#SBATCH --job-name articubot-cgn-multivariance
#SBATCH -o job_%j.out
#SBATCH -e job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=yufeiw2@andrew.cmu.edu

source "/project/flame/yufeiw2/miniconda3/etc/profile.d/conda.sh"
conda activate articubot
cd /project/flame/yufeiw2/RoboGen-sim2real/
source prepare.sh
cd test_PointNet2
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

### download data here
source=gs://cmu-gpucloud-yufeiw2/dp3_demo_165
target=/tmp/

### loop through all zip files in this source gcloud bucket, and download them to target in parallel
# Get list of zip files in the GCS bucket
zip_list=$(gcloud storage ls ${source} | grep '\.zip$')
# zip_list=$(gcloud storage ls ${source} | grep '\.zip$' | head -n 10)

# Download in parallel (max 10 jobs)
echo "Starting download..."
download_start=$(date +%s)

parallel -j 20 gcloud storage cp {} ${target} ::: $zip_list

download_end=$(date +%s)
echo "Total download time: $((download_end - download_start)) seconds"

### unzip all the zip files to be also in /tmp/
echo "Starting unzip..."
unzip_start=$(date +%s)

find "$target" -name '*.zip' | parallel -j 20 '
    zipfile={}
    unzip -q "$zipfile" -d "/tmp/"
'

unzip_end=$(date +%s)
echo "Total unzip time: $((unzip_end - unzip_start)) seconds"

### delete the zip files
echo "Cleaning up ZIP files..."
find "$target" -maxdepth 1 -type f -name '*.zip' -delete
echo "Cleanup complete."

### unzip 165 obj folders
# find "/project/flame/yufeiw2/RoboGen-sim2real/data/dp3_demo/" -name '*.zip' | parallel -j 20 '
#     zipfile={}
#     unzip -q "$zipfile" -d "/tmp/"
# '

### unzip dagger folders
mkdir -p /tmp/weighted_full_dagger
find "/project/flame/yufeiw2/RoboGen-sim2real/data/weighted_full_dagger/" -name '*.zip' | parallel -j 20 '
    zipfile={}
    unzip -q "$zipfile" -d "/tmp/weighted_full_dagger/"
'

### print all folders in /tmp sorted by newest first
echo "Folders in $target (newest first):"
find "$target" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort

torchrun --standalone --nproc_per_node=8 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs    \
    general.exp_name=articubot_cgn_ln_gmm_multivariance_full_450_plus_dagger_load \
    articubot.batch_size=60   \
    articubot.num_train_objects=articulated_full_dagger \
    cgn.batch_size=12  \
    cgn.OPTIMIZER.batch_size=12 \
    general.load_model_path=/project/flame/yufeiw2/RoboGen-sim2real/test_PointNet2/exps/2025-08-14articubot_cgn_ln_gmm_multivariance_full_450_plus_dagger/model_300000.pth