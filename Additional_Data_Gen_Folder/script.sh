pixi run python external/mimicgen/mimicgen/scripts/generate_extra_keypoints.py \
    --data_root /project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/Articubot_Data_For_RVT/SMITH_High_Level_FineTune/LOW_LEVEL_NO_GMM_DATASET_GROOT_STYLE_DATASET/D2 \
    --task COFFEE_PREPERATION_D1 --methods all --episodes 5 --dump_indices


pixi run python external/mimicgen/mimicgen/scripts/generate_extra_keypoints.py \
    --data_root /project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/Articubot_Data_For_RVT/SMITH_High_Level_FineTune/LOW_LEVEL_NO_GMM_DATASET_GROOT_STYLE_DATASET/D2 \
    --task Coffee_D2 --methods all --dump_indices

pixi run python external/mimicgen/mimicgen/scripts/generate_extra_keypoints.py \
    --data_root /project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/Articubot_Data_For_RVT/SMITH_High_Level_FineTune/LOW_LEVEL_NO_GMM_DATASET_GROOT_STYLE_DATASET/D2 \
    --task HAMMER_CLEANUP_D1 --methods all --dump_indices


pixi run python external/mimicgen/mimicgen/scripts/generate_extra_keypoints.py \
    --data_root /project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/Articubot_Data_For_RVT/SMITH_High_Level_FineTune/LOW_LEVEL_NO_GMM_DATASET_GROOT_STYLE_DATASET/D2 \
    --task Mug_Cleanup_D1 --methods all --dump_indices

pixi run python external/mimicgen/mimicgen/scripts/generate_extra_keypoints.py \
    --data_root /project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/Articubot_Data_For_RVT/SMITH_High_Level_FineTune/LOW_LEVEL_NO_GMM_DATASET_GROOT_STYLE_DATASET/D2 \
    --task KITCHEN_D1 --methods all --dump_indices