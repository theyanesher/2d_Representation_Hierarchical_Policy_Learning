#!/bin/bash

#SBATCH -N 1 # Number of nodes
#SBATCH -n 15
#SBATCH -p preempt
#SBATCH --gpus=0 #GPU specification. H100
#SBATCH -t 10:00:00 # Estimated time, 48hour max. DD-HH:MM.
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
# zip_list=$(gcloud storage ls ${source} | grep '\.zip$')

set -euo pipefail

# Get list of zip files in the GCS bucket
zip_list=$(gcloud storage ls "${source}" | grep '\.zip$')

# Function to download with retries
download_with_retry() {
    local file="$1"
    local target_dir="$2"
    local max_retries=5
    local attempt=1

    while (( attempt <= max_retries )); do
        echo "[Attempt $attempt/$max_retries] Downloading $file ..."
        if gcloud storage cp "$file" "$target_dir"; then
            echo "✅ Success: $file"
            return 0
        else
            echo "⚠️  Failed: $file (Attempt $attempt)"
            (( attempt++ ))
            sleep 5
        fi
    done

    echo "❌ Giving up on $file after $max_retries attempts"
    return 1
}

export -f download_with_retry

echo "Starting download..."
download_start=$(date +%s)

# Run in parallel, up to 20 downloads at once
parallel -j 20 download_with_retry {} "${target}" ::: $zip_list

download_end=$(date +%s)
echo "Total download time: $((download_end - download_start)) seconds"


### unzip all the zip files to be also in /tmp/
echo "Starting unzip..."
unzip_start=$(date +%s)

find "$target" -name '*.zip' | parallel -j 10 '
    zipfile={}
    base=$(basename "$zipfile" .zip)
    unzip -q "$zipfile" -d "/tmp/"
'

unzip_end=$(date +%s)
echo "Total unzip time: $((unzip_end - unzip_start)) seconds"

sleep 7200




