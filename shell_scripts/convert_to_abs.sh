# Define your list of tasks…
tasks=(
  stack_d1
  stack_three_d1
  square_d2
  threading_d2
  coffee_d2
  three_piece_assembly_d2
  hammer_cleanup_d1
  mug_cleanup_d1
  kitchen_d1
  nut_assembly_d0
  pick_place_d0
  coffee_preparation_d1
)


for i in "${!tasks[@]}"; do
  task="${tasks[i]}"

  echo "converting ${task} to absolute space"

  python equi_diffpo/scripts/robomimic_dataset_conversion.py \
    -i data/robomimic/datasets/${task}/${task}_1150.hdf5 \
    -o data/robomimic/datasets/${task}/${task}_1150_abs.hdf5 \
    -n 30
done
