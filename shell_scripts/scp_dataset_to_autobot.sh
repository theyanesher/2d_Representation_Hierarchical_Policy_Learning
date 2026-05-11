
# Define your list of tasks…
tasks=(
  # stack_d1
  # stack_three_d1
  # square_d2
  # threading_d2
  # coffee_d2
  three_piece_assembly_d2
  # hammer_cleanup_d1
  # mug_cleanup_d1
  # kitchen_d1
  nut_assembly_d0
  # pick_place_d0
  # coffee_preparation_d1
)

for i in "${!tasks[@]}"; do
  task="${tasks[i]}"

  echo "scp-ing ${task} dataset to autobot"

  scp -r \
    data/robomimic/datasets/${task}/for_oracle/${task}_150_pcd_abs.hdf5 \
    autobot-0-9:/project_data/held/mnakuraf/tax3d-conditioned-mimicgen/data/robomimic/datasets/${task}/for_oracle

  # scp -r \
  #   data/robomimic/datasets/${task}/${task}_pcd_abs_manual_goals.hdf5 \
  #   autobot-0-9:/project_data/held/mnakuraf/tax3d-conditioned-mimicgen/data/robomimic/datasets/${task}/

  #   data/robomimic/datasets/${task}/${task}_pcd_abs.hdf5 \
  #   autobot-0-9:/project_data/held/mnakuraf/tax3d-conditioned-mimicgen/data/robomimic/datasets/${task}/
done