#!/bin/bash

#SBATCH -N 1 # Number of nodes
#SBATCH -n 15
#SBATCH -p preempt
#SBATCH --gpus=1 #GPU specification. H100
#SBATCH -t 48:00:00 # Estimated time, 48hour max. DD-HH:MM.
#SBATCH --job-name articubot_alone_in_articulated
#SBATCH -o job_%j.out
#SBATCH -e job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=yufeiw2@andrew.cmu.edu

### download data here
source=gs://cmu-gpucloud-yufeiw2/dp3_demo_165
target=/tmp/

### loop through all zip files in this source gcloud bucket, and download them to target in parallel
# Get list of zip files in the GCS bucket
zip_list=$(gcloud storage ls ${source} | grep '\.zip$')
# zip_list=$(gcloud storage ls ${source} | grep '\.zip$' | head -n 5)

# Download in parallel (max 10 jobs)
echo "Starting download..."
download_start=$(date +%s)

parallel -j 10 gcloud storage cp {} ${target} ::: $zip_list

download_end=$(date +%s)
echo "Total download time: $((download_end - download_start)) seconds"

### unzip all the zip files to be also in /tmp/
echo "Starting unzip..."
unzip_start=$(date +%s)

find "$target" -name '*.zip' | parallel -j 15 '
    zipfile={}
    base=$(basename "$zipfile" .zip)
    mkdir -p "$target$base"
    unzip -q "$zipfile" -d "$target$base"
'

unzip_end=$(date +%s)
echo "Total unzip time: $((unzip_end - unzip_start)) seconds"

sleep 7200