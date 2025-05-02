#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=GPU
#SBATCH --cpus-per-task=24
#SBATCH --time=480:00:00
#SBATCH --gres=gpu:8
#SBATCH --mem=100G
#SBATCH -o robogen_%j.out
#SBATCH -e robogen_%j.err


# use the bash shell
set -x 
# echo each command to standard out before running it


echo "Starting job $SLURM_JOB_ID"
singularity exec --bind /data/chenyuah/RoboGen-sim2real:/mnt/RoboGen-sim2real/ --nv /data/yufeiw2/robogen-dp3-act3d.sif /mnt/RoboGen-sim2real/scripts/merge_trajectories.sh