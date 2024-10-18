#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=GPU
#SBATCH --cpus-per-task=5
#SBATCH --time=480:00:00
#SBATCH --gres=gpu:0
#SBATCH --mem=10G

rsync -az /data/yufeiw2/RoboGen_sim2real/data/dp3_demo/"${1}" ywang59@bridges2.psc.edu:/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/