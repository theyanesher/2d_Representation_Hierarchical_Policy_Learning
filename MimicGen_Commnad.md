PIXI_CACHE_DIR=/project_data/held/pratik/pixi_cache pixi run python scripts/train.py model=articubot dataset=coffeeTask model.use_rgb=False model.in_channels=4 training.batch_size=4

WANDB_CACHE_DIR=/project_data/held/pratik/wandb_cache WANDB_DATA_DIR=/project_data/held/pratik/wandb_data PYTHONNOUSERSITE=1 PIXI_CACHE_DIR=/project_data/held/pratik/pixi_cache pixi run python scripts/train.py model=articubot dataset=coffeeTask model.use_rgb=False model.in_channels=4 training.batch_size=16 wandb.entity=pbhowal-carnegie-mellon-university

WANDB_CACHE_DIR=/project_data/held/pratik/wandb_cache WANDB_DATA_DIR=/project_data/held/pratik/wandb_data PYTHONNOUSERSITE=1 PIXI_CACHE_DIR=/project_data/held/pratik/pixi_cache pixi run python scripts/train.py model=articubot dataset=coffeeTask model.use_rgb=False model.in_channels=4 training.batch_size=16 wandb.entity=pbhowal-carnegie-mellon-university "resources.gpus=[1,2,3,4,5,6]" wandb.name=coffeeTask_6gpu

WANDB_CACHE_DIR=/project_data/held/pratik/wandb_cache WANDB_DATA_DIR=/project_data/held/pratik/wandb_data PYTHONNOUSERSITE=1 PIXI_CACHE_DIR=/project_data/held/pratik/pixi_cache pixi run python scripts/train.py model=articubot dataset=coffeeTaskDebugGMM model.use_rgb=False model.in_channels=4 training.batch_size=16 wandb.entity=pbhowal-carnegie-mellon-university "resources.gpus=[7]" wandb.name=coffeeTaskDebugGMM_6gpu resources.num_workers=4

WANDB_CACHE_DIR=/project_data/held/pratik/wandb_cache WANDB_DATA_DIR=/project_data/held/pratik/wandb_data PYTHONNOUSERSITE=1 PIXI_CACHE_DIR=/project_data/held/pratik/pixi_cache pixi run python scripts/run_gmm_on_dataset.py --dataset_dir /project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/Articubot_Data_For_RVT/SMITH_High_Level_FineTune/HIGH_LEVEL_GMM_DEBUG_DATASET/Coffee_Task --ckpt_path logs/train_coffeeTask/2026-04-30/21-55-42/checkpoints/rmse=0.094.ckpt/<the .ckpt file inside> --max_files 1 --visualize --visualize_all_gmm_goals

WANDB_CACHE_DIR=/project_data/held/pratik/wandb_cache WANDB_DATA_DIR=/project_data/held/pratik/wandb_data PYTHONNOUSERSITE=1 PIXI_CACHE_DIR=/project_data/held/pratik/pixi_cache pixi run python scripts/run_gmm_on_dataset.py --dataset_dir /project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/Articubot_Data_For_RVT/SMITH_High_Level_FineTune/HIGH_LEVEL_FINETUNE_DATASET/Coffee_Task --ckpt_path logs/train_coffeeTask/2026-04-30/19-06-32/checkpoints/epoch=19-step=25000-val/rmse_and_std_combi=0.016.ckpt --max_files 1 --visualize --visualize_all_gmm_goals



pixi run python scripts/run_gmm_on_dataset.py --dataset_dir /home/pratik_final/Downloads/Bimanual/Articubot_Data_Experiments/Code/GMM_Based_Training_LOw_Level/Data_MimicGen/HIGH_LEVEL_FINETUNE_DATASET/Coffee_Preparation/ --ckpt_path /home/pratik_final/Downloads/Bimanual/Articubot_Data_Experiments/Code/GMM_Based_Training_LOw_Level/Data_MimicGen/IMPORTANT_HIGH_LEVEL_POLICY_MODELS/Coffee_Preperation_models_D0/epoch=19-step=25880-val/rmse_and_std_combi=0.024.ckpt --max_files 1 --visualize --visualize_all_gmm_goals


pixi run python scripts/run_gmm_on_dataset.py --dataset_dir /home/pratik_final/Downloads/Bimanual/Articubot_Data_Experiments/Code/GMM_Based_Training_LOw_Level/Data_MimicGen/HIGH_LEVEL_FINETUNE_DATASET/Kitchen/Kitchen/ --ckpt_path /home/pratik_final/Downloads/Bimanual/Articubot_Data_Experiments/Code/GMM_Based_Training_LOw_Level/Data_MimicGen/GHOST_CODEBASE/2d_Representation_Hierarchical_Policy_Learning/lfd3d/logs/train_Kitchen/2026-05-02/02-18-31/checkpoints/epoch\=9-step\=69610-val/rmse_and_std_combi\=0.018.ckpt --max_files 1 --visualize --visualize_all_gmm_goals

