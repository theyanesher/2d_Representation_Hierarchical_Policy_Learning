#!/bin/bash

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
source=gs://cmu-gpucloud-yufeiw2/dp3_demo_165
target=/tmp/

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
# echo "unzip articubot dagger folders"
# unzip_start=$(date +%s)
# mkdir -p /tmp/weighted_full_dagger
# find "/project/flame/yufeiw2/RoboGen-sim2real/data/weighted_full_dagger/" -name '*.zip' | parallel -j 20 '
#     zipfile={}
#     unzip -q "$zipfile" -d "/tmp/weighted_full_dagger/"
# '
# unzip_end=$(date +%s)
# echo "Total unzip time: $((unzip_end - unzip_start)) seconds"


### unzip camera randomized other category data here
# echo "unzip articubot other category camera randomization folders"
# unzip_start=$(date +%s)
# mkdir -p /tmp/new_7_category_random_cam
# find "/project/flame/yufeiw2/RoboGen-sim2real/data/new_7_category_random_cam/" -name '*.zip' | parallel -j 20 '
#     zipfile={}
#     unzip -q "$zipfile" -d "/tmp/new_7_category_random_cam/"
# '
# unzip_end=$(date +%s)
# echo "Total unzip time: $((unzip_end - unzip_start)) seconds"

### unzip real camera randomized other category data here
# echo "unzip articubot other category camera randomization folders"
# unzip_start=$(date +%s)
# mkdir -p /tmp/new_7_category_real_cam
# find "/project/flame/yufeiw2/RoboGen-sim2real/data/new_7_category_real_cam/" -name '*.zip' | parallel -j 20 '
#     zipfile={}
#     unzip -q "$zipfile" -d "/tmp/new_7_category_real_cam/"
# '
# unzip_end=$(date +%s)
# echo "Total unzip time: $((unzip_end - unzip_start)) seconds"


### download and unzip close data here
# source=gs://cmu-gpucloud-chenyuah/dp3_demo/invert_push
# target=/tmp/invert_push
# mkdir -p /tmp/invert_push

# ### loop through all zip files in this source gcloud bucket, and download them to target in parallel
# # Get list of zip files in the GCS bucket
# # zip_list=$(gcloud storage ls ${source} | grep '\.zip$')
# zip_list=$(gcloud storage ls ${source} | grep '\.zip$' | head -n 10)

# # Download in parallel (max 10 jobs)
# echo "Starting download..."
# download_start=$(date +%s)

# parallel -j 20 gcloud storage cp {} ${target} ::: $zip_list

# download_end=$(date +%s)
# echo "Total download time: $((download_end - download_start)) seconds"

# ### unzip all the zip files to be also in /tmp/
# echo "Starting unzip..."
# unzip_start=$(date +%s)

# find "$target" -name '*.zip' | parallel -j 20 '
#     zipfile={}
#     unzip -q "$zipfile" -d "/tmp/"
# '

# unzip_end=$(date +%s)
# echo "Total unzip time: $((unzip_end - unzip_start)) seconds"

# ### delete the zip files
# echo "Cleaning up ZIP files..."
# find "$target" -maxdepth 1 -type f -name '*.zip' -delete
# echo "Cleanup complete."

