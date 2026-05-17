MIMICGEN COMMANDS =>


Weighted cross-attention (GMM probabilities as log-bias on attention logits):


COFFEE D0 TASK

export PATH="$HOME/.pixi/bin:$PATH"

CUDA_VISIBLE_DEVICES=4 pixi run python diffusion_policy/train.py --config-name=train_flow_matching_dit_workspace.yaml task=MimicGen_Tasks/coffee_gmm_goal visual_encoder=dinov2 policy.use_goal_cross_attention=true policy.use_weighted_cross_attention=true policy.gmm_top_k=1024 logging.project=MimicGen_GMM_Low_Level_Policy logging.name=groot_GMM_Weighted_Cross_Attention_Coffee name=groot_GMM_Weighted_Cross_Attention_Coffee dataloader.batch_size=64 dataloader.num_workers=16

Resuming from Checkpoint => /project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/Articubot_Data_For_RVT/2D_Hierarchical_Policy_Learning_Github/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/outputs/2026.05.06/13.47.20_groot_GMM_Weighted_Cross_Attention_Coffee_coffee_gmm_goal/checkpoints/epoch_20.ckpt


pixi run python diffusion_policy/train.py --config-name=train_flow_matching_dit_workspace.yaml task=MimicGen_Tasks/coffee_gmm_goal visual_encoder=dinov2 policy.use_goal_cross_attention=true policy.use_weighted_cross_attention=true policy.gmm_top_k=1024 logging.project=MimicGen_GMM_Low_Level_Policy logging.name=groot_GMM_Weighted_Cross_Attention_Coffee name=groot_GMM_Weighted_Cross_Attention_Coffee dataloader.batch_size=64 dataloader.num_workers=16 training.resume=true
 
 


# policy.gmm_top_k=128

COFFEE D2 TASK

pixi run python diffusion_policy/train.py --config-name=train_flow_matching_dit_workspace.yaml task=MimicGen_Tasks/coffee_D2_gmm_goal visual_encoder=dinov2 policy.use_goal_cross_attention=true policy.use_weighted_cross_attention=true policy.gmm_top_k=1024 logging.project=MimicGen_GMM_Low_Level_Policy logging.name=groot_GMM_Weighted_Cross_Attention_Coffee name=groot_GMM_Weighted_Cross_Attention_Coffee dataloader.batch_size=64 dataloader.num_workers=16 training.resume=true












