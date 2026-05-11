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
  echo "fixing gripper pcd in ${task}"
  python sandbox/mino/dataset_modifications/replace_gripper_pcd_hdf5.py ${task}
done
