#!/bin/bash

# Tunables
JOBS="${JOBS:-30}"
ZIPS_LIMIT="${ZIPS_LIMIT:-}"
MAX_RETRIES="${MAX_RETRIES:-5}"
RETRY_DELAY="${RETRY_DELAY:-5}"

# Make sure child shells (spawned by parallel) see these:
export MAX_RETRIES RETRY_DELAY
export JOBS ZIPS_LIMIT

download_with_retry() {
  local file="$1"
  local target_dir="$2"
  local attempt=1
  local max="${MAX_RETRIES:-5}"
  local delay="${RETRY_DELAY:-5}"

  while (( attempt <= max )); do
    echo "[Attempt $attempt/$max] Downloading $file ..."
    if gcloud storage cp "$file" "$target_dir"; then
      echo "✅ Success: $file"
      return 0
    else
      echo "⚠️  Failed: $file (Attempt $attempt)"
      (( attempt++ ))
      sleep "$delay"
    fi
  done

  echo "❌ Giving up on $file after $max attempts"
  return 1
}
export -f download_with_retry

# Usage: sync_gcs_zip_folder gs://bucket/prefix /local/dir
sync_gcs_zip_folder() {
  local source="$1"
  local target="$2"

  mkdir -p "$target"

  # Build the zip list safely into an array
#   mapfile -t zips < <(
#     gcloud storage ls "$source" | grep '\.zip$' ${ZIPS_LIMIT:+| head -n "$ZIPS_LIMIT"}
#   )

    if [[ -n "${ZIPS_LIMIT:-}" ]]; then
    mapfile -t zips < <(
        gcloud storage ls "$source" | grep '\.zip$' | head -n "$ZIPS_LIMIT"
    )
    else
    mapfile -t zips < <(
        gcloud storage ls "$source" | grep '\.zip$'
    )
    fi


  if ((${#zips[@]}==0)); then
    echo "No zip files found under: $source"
    return 0
  fi

  echo "Starting download to $target ..."
  download_start=$SECONDS
  # Download in parallel with retries
  parallel -j "$JOBS" download_with_retry {} "$target" ::: "${zips[@]}"
    # parallel -j "$JOBS"download_with_retry "$0" "$1"' {} "$target" ::: "${zips[@]}"
  echo "Total download time: $((SECONDS - download_start)) seconds"
}
# (no need to export sync_gcs_zip_folder; only download_with_retry is used by parallel)


###
sync_gcs_zip_folder gs://cmu-gpucloud-yufeiw2/dp3_demo_new_7_category_real_cam /project/flame/yufeiw2/RoboGen-sim2real/data/new_7_category_real_cam/

sync_gcs_zip_folder gs://cmu-gpucloud-yufeiw2/invert_push /project/flame/yufeiw2/RoboGen-sim2real/data/invert_push/

sync_gcs_zip_folder gs://cmu-gpucloud-yufeiw2/dp3_demo/gen_grasp_1017 /project/flame/yufeiw2/RoboGen-sim2real/data/gen_grasp_1017 ## this is with lifting

sync_gcs_zip_folder gs://cmu-gpucloud-chenyuah/dp3_demo/top_cgn /project/flame/yufeiw2/RoboGen-sim2real/data/top_cgn_1204

sync_gcs_zip_folder gs://cmu-gpucloud-chenyuah/dp3_demo/inside_link_cgn /project/flame/yufeiw2/RoboGen-sim2real/data/inside_link_cgn_1204

sync_gcs_zip_folder gs://cmu-gpucloud-chenyuah/dp3_demo/inside_whole_cgn /project/flame/yufeiw2/RoboGen-sim2real/data/inside_whole_cgn_1204
