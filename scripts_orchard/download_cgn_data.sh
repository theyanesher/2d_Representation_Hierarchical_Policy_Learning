#!/bin/bash

### unzip cgn render 
# echo "unzip cgn renders folders"
# unzip_start=$(date +%s)
# mkdir -p /tmp/acronym/renders
# find "/project/flame/yufeiw2/RoboGen-sim2real/data/acronym/renders/" -name '*.zip' | parallel -j 20 '
#     zipfile={}
#     unzip -q "$zipfile" -d "/tmp/acronym/renders/"
# '
# unzip_end=$(date +%s)
# echo "Total unzip time: $((unzip_end - unzip_start)) seconds"

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


### download 165 obj data here
source=gs://cmu-gpucloud-yufeiw2/acronym_renders
mkdir -p /tmp/acronym/renders
target=/tmp/acronym/renders

### loop through all zip files in this source gcloud bucket, and download them to target in parallel
# Get list of zip files in the GCS bucket
# zip_list=$(gcloud storage ls ${source} | grep '\.zip$')
zip_list=$(gcloud storage ls ${source} | grep '\.zip$' | head -n 10)

# Download in parallel (max 10 jobs)
echo "Starting download..."
download_start=$(date +%s)

parallel -j 20 download_with_retry {} "${target}" ::: $zip_list

download_end=$(date +%s)
echo "Total download time: $((download_end - download_start)) seconds"

### unzip all the zip files to be also in /tmp/
echo "Starting unzip..."
unzip_start=$(date +%s)

find "$target" -name '*.zip' | parallel -j 20 '
    zipfile={}
    unzip -q "$zipfile" -d "/tmp/acronym/renders"
'

unzip_end=$(date +%s)
echo "Total unzip time: $((unzip_end - unzip_start)) seconds"

### delete the zip files
echo "Cleaning up ZIP files..."
find "$target" -maxdepth 1 -type f -name '*.zip' -delete
echo "Cleanup complete."


## copy cgn scene contacts
echo "rsync cgn renders folders"
unzip_start=$(date +%s)
mkdir -p /tmp/acronym/scene_contacts
rsync -a /project/flame/yufeiw2/RoboGen-sim2real/data/acronym/scene_contacts/* /tmp/acronym/scene_contacts/
unzip_end=$(date +%s)
echo "Total rsync time: $((unzip_end - unzip_start)) seconds"