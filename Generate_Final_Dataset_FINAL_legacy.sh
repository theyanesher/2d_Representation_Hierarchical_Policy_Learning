#!/bin/bash
# Don't use set -e with parallel background jobs

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/usr/lib/nvidia:/home/pbhowal/.mujoco/mujoco210/bin
export PYTHONPATH=/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/Articubot_Data_For_RVT/SMITH_MimicGen/SMITH_on_mimicgen/external/robomimic:${PYTHONPATH:-}
export MUJOCO_GL=egl
export DISPLAY=:99

SCRIPT=external/mimicgen/mimicgen/scripts/convert_dataset.py
SRC=/scratch/pbhowal/Uncertainty_Dataset/ORIGINAL_DATASET
DST=/scratch/pbhowal/Uncertainty_Dataset/LOW_LEVEL_TRAIN_DATASET_SMITH_STYLE_DATASET

mkdir -p logs

run_job() {
    local gpu=$1 input=$2 outdir=$3 cfg=$4
    local name=$(basename "$outdir")
    echo "[GPU $gpu] starting $name"
    CUDA_VISIBLE_DEVICES=$gpu python "$SCRIPT" \
        --input "$SRC/$input" \
        --output_dir "$DST/$outdir" \
        --camera_height 256 --camera_width 256 \
        --num_workers 1 \
        --use_bayesian_decomp \
        --bocpd_config "$cfg" \
        > "logs/${name}.log" 2>&1 &
    echo "[GPU $gpu] $name pid=$!"
}

# run_job 0 coffee_d2.hdf5              Coffee_D2              third_party/robogen/bocpd_config_COFFEE_FINAL.yaml
run_job 1 mug_cleanup_d1.hdf5         Mug_Cleanup_D1         third_party/robogen/bocpd_config.yaml
run_job 2 kitchen_d1.hdf5             KITCHEN_D1             third_party/robogen/bocpd_config_kitchen.yaml
run_job 3 coffee_preperation_d1.hdf5  COFFEE_PREPERATION_D1  third_party/robogen/bocpd_config_COFFEE_PREPERATION.yaml
# run_job 4 hammer_cleanup_d1.hdf5      HAMMER_CLEANUP_D1      third_party/robogen/bocpd_config_HAMMER_CLEANUP.yaml

echo "All 5 jobs launched. Monitor with: tail -f logs/*.log"
wait
echo "All done."