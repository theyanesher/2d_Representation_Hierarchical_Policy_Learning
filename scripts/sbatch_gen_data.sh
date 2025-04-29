#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=GPU
#SBATCH --cpus-per-task=24
#SBATCH --time=480:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=60G

# use the bash shell
set -x 
# echo each command to standard out before running it


echo "Starting job $SLURM_JOB_ID"
singularity exec --bind /data/chenyuah/RoboGen-sim2real:/mnt/RoboGen_sim2real/ --nv /data/ziyuw2/robogen-dp3-act3d.sif /mnt/RoboGen_sim2real/scripts/gen_data_parallel.sh ${1} ${2}