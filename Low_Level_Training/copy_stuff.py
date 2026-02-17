import shutil
import os

direc_all = [
#     "close_jar",
#     "put_money_in_safe",
#     "light_bulb_in",
#     "insert_onto_square_peg",
#     "place_wine_at_rack_location",
#     "slide_block_to_color_target",
    "meat_off_grill",
    "place_shape_in_shape_sorter",
    "reach_and_drag",
    "stack_blocks",
    "open_drawer",
    "put_item_in_drawer",
    "sweep_to_dustpan_of_size",
    "place_cups",
    "push_buttons",
    "put_groceries_in_cupboard"
]
# direc_all = ["put_item_in_drawer"]

    # # "outputs_1st_open_drawer",
    # # "outputs_1st_put_item_in_drawer",
    # "outputs_1st_sweep_to_dustpan_of_size",
    # "outputs_1st_place_cups",
    # "outputs_1st_put_money_in_safe",

for direc in direc_all:
    source_dir = f"/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/RVT2_GMM_Codebase/RVT2_Codebase/RVT_Backup_IDK/RVT/data_diffusion_policy/outputs_1st_{direc}/trajectories_data_FINAL/"
    destination_dir = f"/scratch/pbhowal/data_Diffusion_Policy/{direc}"

    os.makedirs(destination_dir, exist_ok=True)

    for fname in os.listdir(source_dir):
        src_path = os.path.join(source_dir, fname)
        dst_path = os.path.join(destination_dir, fname)
        # import pdb; pdb.set_trace();
        if os.path.isdir(src_path):  # handles .zarr folders
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

