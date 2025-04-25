log_dir=/data/chenyuah/RoboGen-sim2real/high_level

sbatch -o ${log_dir}/stdout.log \
    -e ${log_dir}/stderr.log \
    -J high_level \
    scripts/sbatch_train_high_level.sh \
