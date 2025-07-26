#!/bin/bash

#SBATCH -N 1 # Number of nodes
#SBATCH -n 1
#SBATCH -p preempt
#SBATCH --gpus=1 #GPU specification. H100
#SBATCH -t 1:05:00 # Estimated time, 48hour max. DD-HH:MM.
#SBATCH --job-name test
#SBATCH -o job_%j.out
#SBATCH -e job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=yufeiw2@andrew.cmu.edu

python scripts_orchard/hello.py