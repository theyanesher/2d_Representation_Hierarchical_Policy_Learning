#!/bin/bash
# Don't use set -e with parallel background jobs

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/usr/lib/nvidia:/home/pbhowal/.mujoco/mujoco210/bin
export PYTHONPATH=/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/Articubot_Data_For_RVT/SMITH_MimicGen/SMITH_on_mimicgen/external/robomimic:${PYTHONPATH:-}
export MUJOCO_GL=egl
export DISPLAY=:99

SCRIPT=external/mimicgen/mimicgen/scripts/convert_dataset.py
SRC=/scratch/pbhowal/Uncertainty_Dataset/Original_Dataset
DST=/scratch/pbhowal/Uncertainty_Dataset/LOW_LEVEL_NO_GMM_DATASET_GROOT_STYLE_DATASET/D2    #Mug_Cleanup_D1/

# Throughput knob: parallel worker processes per job (i.e. per GPU).
# 2 is a safe default for both 3080 (10-12GB) and 3090 (24GB).
# Bump to 3 only on a 24GB GPU if you've verified VRAM headroom with nvidia-smi.
POOL_SIZE=${POOL_SIZE:-2}

mkdir -p logs

run_job() {
    local gpu=$1 input=$2 outdir=$3 cfg=$4
    local name=$(basename "$outdir")
    echo "[GPU $gpu] starting $name (pool_size=$POOL_SIZE)"
    CUDA_VISIBLE_DEVICES=$gpu python "$SCRIPT" \
        --input "$SRC/$input" \
        --output_dir "$DST/$outdir" \
        --camera_height 256 --camera_width 256 \
        --pool_size $POOL_SIZE \
        --use_bayesian_decomp \
        --bocpd_config "$cfg" \
        > "logs/${name}.log" 2>&1 &
    echo "[GPU $gpu] $name pid=$!"
}

# run_job 0 coffee_d2.hdf5              Coffee_D2              third_party/robogen/bocpd_config_COFFEE_FINAL.yaml
run_job 1 mug_cleanup_d1.hdf5         Mug_Cleanup_D1         third_party/robogen/bocpd_config.yaml
# run_job 2 kitchen_d1.hdf5             KITCHEN_D1             third_party/robogen/bocpd_config_kitchen.yaml
# run_job 3 coffee_preperation_d1.hdf5  COFFEE_PREPERATION_D1  third_party/robogen/bocpd_config_COFFEE_PREPERATION.yaml
# run_job 4 hammer_cleanup_d1.hdf5      HAMMER_CLEANUP_D1      third_party/robogen/bocpd_config_HAMMER_CLEANUP.yaml

echo "All 3 jobs launched. Monitor with: tail -f logs/*.log"
wait
echo "All done."