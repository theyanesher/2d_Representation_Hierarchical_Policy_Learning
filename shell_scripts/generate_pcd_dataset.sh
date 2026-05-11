# Define your list of tasks…
tasks=(
  # stack_d1
  # stack_three_d1
  # square_d2
  # threading_d2
  # coffee_d2
  three_piece_assembly_d2
  nut_assembly_d0 #, uses sawyer
  # hammer_cleanup_d1
  # mug_cleanup_d1
  # kitchen_d1 # long horizon task
  # coffee_preparation_d1 # long horizon task
  # pick_place_d0 # long horizon task, uses sawyer
)

for i in "${!tasks[@]}"; do
  task="${tasks[i]}"

  echo "incorporating point clouds into ${task} dataset"

  python equi_diffpo/scripts/dataset_states_to_obs.py \
    --input data/robomimic/datasets/${task}/for_oracle/${task}_150_abs.hdf5 \
    --output data/robomimic/datasets/${task}/for_oracle/${task}_150_pcd_abs.hdf5 \
    --num_workers=30
done