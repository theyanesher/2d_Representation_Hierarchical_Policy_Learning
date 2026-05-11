export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/pbhowal/.mujoco/mujoco210/bin


PYTHONPATH=/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/Articubot_Data_For_RVT/SMITH_MimicGen/SMITH_on_mimicgen/external/robomimic:$PYTHONPATH MUJOCO_GL=egl DISPLAY=:99 python external/mimicgen/mimicgen/scripts/convert_dataset.py --input /scratch/pbhowal/Uncertainty_Dataset/ORIGINAL_DATASET/coffee_d2.hdf5 --output_dir /scratch/pbhowal/Uncertainty_Dataset/LOW_LEVEL_TRAIN_DATASET_SMITH_STYLE_DATASET/Coffee_D2 --camera_height 256 --camera_width 256 --num_workers 1 --use_bayesian_decomp --bocpd_config third_party/robogen/bocpd_config_COFFEE_FINAL.yaml



PYTHONPATH=/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/Articubot_Data_For_RVT/SMITH_MimicGen/SMITH_on_mimicgen/external/robomimic:$PYTHONPATH MUJOCO_GL=egl DISPLAY=:99 python external/mimicgen/mimicgen/scripts/convert_dataset.py --input /scratch/pbhowal/Uncertainty_Dataset/ORIGINAL_DATASET/mug_cleanup_d1.hdf5 --output_dir /scratch/pbhowal/Uncertainty_Dataset/LOW_LEVEL_TRAIN_DATASET_SMITH_STYLE_DATASET/Mug_Cleanup_D1 --camera_height 256 --camera_width 256 --num_workers 1 --use_bayesian_decomp --bocpd_config third_party/robogen/bocpd_config.yaml



PYTHONPATH=/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/Articubot_Data_For_RVT/SMITH_MimicGen/SMITH_on_mimicgen/external/robomimic:$PYTHONPATH MUJOCO_GL=egl DISPLAY=:99 python external/mimicgen/mimicgen/scripts/convert_dataset.py --input /scratch/pbhowal/Uncertainty_Dataset/ORIGINAL_DATASET/kitchen_d1.hdf5 --output_dir /scratch/pbhowal/Uncertainty_Dataset/LOW_LEVEL_TRAIN_DATASET_SMITH_STYLE_DATASET/KITCHEN_D1 --camera_height 256 --camera_width 256 --num_workers 1 --use_bayesian_decomp --bocpd_config third_party/robogen/bocpd_config_kitchen.yaml


PYTHONPATH=/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/Articubot_Data_For_RVT/SMITH_MimicGen/SMITH_on_mimicgen/external/robomimic:$PYTHONPATH MUJOCO_GL=egl DISPLAY=:99 python external/mimicgen/mimicgen/scripts/convert_dataset.py --input /scratch/pbhowal/Uncertainty_Dataset/ORIGINAL_DATASET/coffee_preperation_d1.hdf5 --output_dir /scratch/pbhowal/Uncertainty_Dataset/LOW_LEVEL_TRAIN_DATASET_SMITH_STYLE_DATASET/COFFEE_PREPERATION_D1/ --camera_height 256 --camera_width 256 --num_workers 1 --use_bayesian_decomp --bocpd_config third_party/robogen/bocpd_config_COFFEE_PREPERATION.yaml

PYTHONPATH=/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/Articubot_Data_For_RVT/SMITH_MimicGen/SMITH_on_mimicgen/external/robomimic:$PYTHONPATH MUJOCO_GL=egl DISPLAY=:99 python external/mimicgen/mimicgen/scripts/convert_dataset.py --input /scratch/pbhowal/Uncertainty_Dataset/ORIGINAL_DATASET/hammer_cleanup_d1.hdf5 --output_dir /scratch/pbhowal/Uncertainty_Dataset/LOW_LEVEL_TRAIN_DATASET_SMITH_STYLE_DATASET/HAMMER_CLEANUP_D1 --camera_height 256 --camera_width 256 --num_workers 1 --use_bayesian_decomp --bocpd_config third_party/robogen/bocpd_config_HAMMER_CLEANUP.yaml





