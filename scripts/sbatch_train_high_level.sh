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
singularity exec --bind /project_data/held/chenyuah/RoboGen-sim2real:/mnt/RoboGen_sim2real/ --nv /project_data/held/yufeiw2/robogen-dp3-act3d.sif /mnt/RoboGen_sim2real/scripts/weighted-displacement-high-level/train-weighted-displacement-100-obj.sh 