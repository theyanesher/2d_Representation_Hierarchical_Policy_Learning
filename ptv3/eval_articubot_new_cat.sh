#!/bin/bash

# list=(40147 44817 44962 45132 45219 45243 45332 45378 45384 45463)
list=("bucket/100435" "bucket/100441" "faucet/149" "faucet/960" "faucet/991" "foldingchair/100520" "foldingchair/100521" "foldingchair/100526" "laptop/9748" "laptop/9912" "laptop/9960" "stapler/102990" "stapler/103095" "toilet/10320" "toilet/102620" "toilet/102621")
# list=(40147 44817)

gpu_count=8
# gpu_count=6
i=0

for id in "${list[@]}"; do
    gpu_id=$((i % gpu_count))
    # gpu_id=$(( (i % gpu_count) + 2 ))

    echo "Launching ID=$id on GPU=$gpu_id"
    safe_id=${id//\//_}

    CUDA_VISIBLE_DEVICES=$gpu_id python 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/eval_robogen_with_goal_PointNet_gmm_cleaned.py  \
        --low_level_exp_dir /project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/data/close_push_low   \
        --low_level_ckpt_name epoch-52.ckpt   \
        --high_level_ckpt_name /project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/articubot_3dfa/train_logs/articubot/2025-0919-articubot_articulated_large_pn/last.pth \
        --eval_exp_name eval_baseline_3dfa_articulated \
        --exp_dir data/${id} \
        --model_type 3dfa \
    > data/logs/eval_${safe_id}.log 2>&1 &

    # CUDA_VISIBLE_DEVICES=$gpu_id python 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/eval_robogen_with_goal_PointNet_gmm_cleaned.py  \
    #     --low_level_exp_dir /project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/data/close_push_low   \
    #     --low_level_ckpt_name epoch-52.ckpt   \
    #     --high_level_ckpt_name /project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/data/baseline_ptv3_articulated/2025.10.24/23.34.56/model_340000.pth \
    #     --eval_exp_name eval_baseline_ptv3_articulated \
    #     --exp_dir data/${id} \
    #     --model_type ptv3 \
    # > data/logs/eval_${safe_id}.log 2>&1 &
    
    # CUDA_VISIBLE_DEVICES=$gpu_id python 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/eval_robogen_with_goal_PointNet_gmm_cleaned.py  \
    #     --low_level_exp_dir /project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/data/close_push_low   \
    #     --low_level_ckpt_name epoch-52.ckpt   \
    #     --high_level_ckpt_name /project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/test_PointNet2/exps/2025-10-25baseline-ours-on-articulated/model_255001.pth \
    #     --eval_exp_name eval_baseline_ours_articulated \
    #     --exp_dir data/${id} \
    # > data/logs/eval_${safe_id}.log 2>&1 &

    # CUDA_VISIBLE_DEVICES=$gpu_id python 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/eval_robogen_with_goal_PointNet_gmm_cleaned.py  \
    #     --low_level_exp_dir /project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/data/close_push_low   \
    #     --low_level_ckpt_name epoch-52.ckpt   \
    #     --high_level_ckpt_name /project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/test_PointNet2/exps/2025-09-14multinode-cgn-world-articubot-all-w-pick-place/model_155001.pth \
    #     --eval_exp_name eval_articubot-all-w-pick-place \
    #     --exp_dir data/${id} \
    #     > data/logs/eval_${safe_id}.log 2>&1 &

    # CUDA_VISIBLE_DEVICES=$gpu_id python 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/eval_robogen_with_goal_PointNet_gmm_cleaned.py  \
    #     --low_level_exp_dir /project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/data/close_push_low   \
    #     --low_level_ckpt_name epoch-52.ckpt   \
    #     --high_level_ckpt_name /project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/test_PointNet2/exps/2025-09-25multinode-pointnext-cgn-world-articubot-format-aritucbot_new_cat_camera_random_close_w_pick_place/model_290001.pth \
    #     --eval_exp_name eval_pointnext_new_cat_camera_random_close_w_pick_place_new_low_level     \
    #     --exp_dir data/${id} \
    #     > data/logs/eval_${safe_id}.log 2>&1 &

    # CUDA_VISIBLE_DEVICES=$gpu_id python 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/eval_robogen_with_goal_PointNet_gmm_cleaned_old_data_format.py  \
    #     --low_level_exp_dir /project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/data/low-level-ckpt/   \
    #     --low_level_ckpt_name low-level.ckpt   \
    #     --high_level_ckpt_name /project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/test_PointNet2/exps/2025-09-21PointNext-articubot-50/model_232501.pth \
    #     --eval_exp_name eval_pointnext_new_cat_camera_random_close_w_pick_place     \
    #     --exp_dir data/diverse_objects/open_the_door_${id}/task_open_the_door_of_the_storagefurniture_by_its_handle  \
    #     > data/logs/eval_${id}.log 2>&1 &

    # CUDA_VISIBLE_DEVICES=$gpu_id python 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/eval_robogen_with_goal_PointNet_gmm_cleaned_old_data_format.py  \
    #     --low_level_exp_dir /project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/data/low-level-ckpt/   \
    #     --low_level_ckpt_name low-level.ckpt   \
    #     --high_level_ckpt_name /project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/test_PointNet2/exps/2025-09-21PointNext-articubot-50/model_232501.pth \
    #     --eval_exp_name eval_pointnext_50     \
    #     --exp_dir data/diverse_objects/open_the_door_${id}/task_open_the_door_of_the_storagefurniture_by_its_handle  \
    #     > data/logs/eval_${id}.log 2>&1 &

        # CUDA_VISIBLE_DEVICES=$gpu_id python 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/eval_robogen_with_goal_PointNet_gmm_cleaned_old_data_format.py  \
        # --low_level_exp_dir /project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/data/low-level-ckpt/   \
        # --low_level_ckpt_name low-level.ckpt   \
        # --high_level_ckpt_name /project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/articubot_3dfa/train_logs/articubot/2025-0919-articubot_articulated_large_pn/last.pth \
        # --eval_exp_name eval_3dfa_articulated_large     \
        # --exp_dir data/diverse_objects/open_the_door_${id}/task_open_the_door_of_the_storagefurniture_by_its_handle  \
        # --model_type 3dfa \
        # > data/logs/eval_${id}.log 2>&1 &

    #  CUDA_VISIBLE_DEVICES=$gpu_id python 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/eval_robogen_with_goal_PointNet_gmm_cleaned_old_data_format.py  \
    #     --low_level_exp_dir /project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/data/low-level-ckpt/   \
    #     --low_level_ckpt_name low-level.ckpt   \
    #     --high_level_ckpt_name /project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/articubot_3dfa/train_logs/articubot/2025-0914-articubot_articulated_larger_pn/last.pth \
    #     --eval_exp_name eval_3dfa_articulated_larger     \
    #     --exp_dir data/diverse_objects/open_the_door_${id}/task_open_the_door_of_the_storagefurniture_by_its_handle  \
    #     --model_type 3dfa \
    #     > data/logs/eval_${id}.log 2>&1 &

    # CUDA_VISIBLE_DEVICES=$gpu_id python 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/eval_robogen_with_goal_PointNet_gmm_cleaned_old_data_format.py  \
    #     --low_level_exp_dir /project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/data/low-level-ckpt/   \
    #     --low_level_ckpt_name low-level.ckpt   \
    #     --high_level_ckpt_name /project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/articubot_3dfa/train_logs/articubot/2025-0918-articubot_50_larger_pn_64_not_128/last.pth \
    #     --eval_exp_name eval_3dfa_even_larger_pn_64_not_128     \
    #     --exp_dir data/diverse_objects/open_the_door_${id}/task_open_the_door_of_the_storagefurniture_by_its_handle  \
    #     --model_type 3dfa \
    #     > data/logs/eval_${id}.log 2>&1 &

    # CUDA_VISIBLE_DEVICES=$gpu_id python 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/eval_robogen_with_goal_PointNet_gmm_cleaned_old_data_format.py  \
    #     --low_level_exp_dir /project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/data/low-level-ckpt/   \
    #     --low_level_ckpt_name low-level.ckpt   \
    #     --high_level_ckpt_name /project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/articubot_3dfa/train_logs/articubot/2025-0916-articubot_50_larger_pn/last.pth \
    #     --eval_exp_name eval_3dfa_even_larger_pn     \
    #     --exp_dir data/diverse_objects/open_the_door_${id}/task_open_the_door_of_the_storagefurniture_by_its_handle  \
    #     --model_type 3dfa \
    #     > data/logs/eval_${id}.log 2>&1 &


    # CUDA_VISIBLE_DEVICES=$gpu_id python 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/eval_robogen_with_goal_PointNet_gmm_cleaned_old_data_format.py  \
    #     --low_level_exp_dir /project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/data/low-level-ckpt/   \
    #     --low_level_ckpt_name low-level.ckpt   \
    #     --high_level_ckpt_name /project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/articubot_3dfa/train_logs/2025-0817-test_articubot_50/last.pth \
    #     --eval_exp_name eval_3dfa_small_pn_correct     \
    #     --exp_dir data/diverse_objects/open_the_door_${id}/task_open_the_door_of_the_storagefurniture_by_its_handle  \
    #     --model_type 3dfa \
    #     > data/logs/eval_${id}.log 2>&1 &

    # CUDA_VISIBLE_DEVICES=$gpu_id python 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/eval_robogen_with_goal_PointNet_gmm_cleaned_old_data_format.py  \
    #     --low_level_exp_dir /project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/data/low-level-ckpt/   \
    #     --low_level_ckpt_name low-level.ckpt   \
    #     --high_level_ckpt_name /project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/articubot_3dfa/train_logs/articubot/2025-0912-test_articubot_50/last.pth \
    #     --eval_exp_name eval_3dfa_new     \
    #     --exp_dir data/diverse_objects/open_the_door_${id}/task_open_the_door_of_the_storagefurniture_by_its_handle  \
    #     --model_type 3dfa \
    #     > data/logs/eval_${id}.log 2>&1 &


    # CUDA_VISIBLE_DEVICES=$gpu_id python 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/eval_robogen_with_goal_PointNet_gmm_cleaned_old_data_format.py  \
    #     --low_level_exp_dir /project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/data/low-level-ckpt/   \
    #     --low_level_ckpt_name low-level.ckpt   \
    #     --high_level_ckpt_name /project_data/held/yufeiw2/M2T2_articubot/logs/2025-09-08_articubot_cgn_GMM_LN_autobot/model_132500.pth \
    #     --eval_exp_name eval_m2t2_cgn_articubot_50_gmm     \
    #     --exp_dir data/diverse_objects/open_the_door_${id}/task_open_the_door_of_the_storagefurniture_by_its_handle  \
    #     --model_type m2t2 \
    #     > data/logs/eval_${id}.log 2>&1 &

    # CUDA_VISIBLE_DEVICES=$gpu_id python 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/eval_robogen_with_goal_PointNet_gmm_cleaned_old_data_format.py \
    #     --low_level_exp_dir /project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/data/low-level-ckpt/ \
    #     --low_level_ckpt_name low-level.ckpt \
    #     --high_level_ckpt_name /project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/data/articubot_ptv3_single-task_50/2025.09.02/01.50.27/model_217500.pth \
    #     --eval_exp_name eval_ptv3_articubot_50_single_2 \
    #     --exp_dir data/diverse_objects/open_the_door_${id}/task_open_the_door_of_the_storagefurniture_by_its_handle \
    #     --model_type ptv3 \
    #     > data/logs/eval_${id}.log 2>&1 &

    # CUDA_VISIBLE_DEVICES=$gpu_id python 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/eval_robogen_with_goal_PointNet_gmm_cleaned_old_data_format.py  \
    #     --low_level_exp_dir /project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/low-level-50-goal-always-open/2025.08.27/21.00.29_train_dp3_robogen_open_door   \
    #     --low_level_ckpt_name epoch-96.ckpt   \
    #     --high_level_ckpt_name /project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/articubot_3dfa/train_logs/2025-0817-test_articubot_50/last.pth \
    #     --eval_exp_name eval_3dfa_50_300k_low_level_open     \
    #     --exp_dir data/diverse_objects/open_the_door_${id}/task_open_the_door_of_the_storagefurniture_by_its_handle  \
    #     --model_type 3dfa \
    #     > data/logs/eval_${id}.log 2>&1 &

    # CUDA_VISIBLE_DEVICES=$gpu_id python 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/eval_robogen_with_goal_PointNet_gmm_cleaned_old_data_format.py  \
    #     --low_level_exp_dir /project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/data/low-level-ckpt/   \
    #     --low_level_ckpt_name low-level.ckpt   \
    #     --high_level_ckpt_name /project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/articubot_3dfa/train_logs/2025-0817-test_articubot_50/last.pth \
    #     --eval_exp_name eval_3dfa_50_300k     \
    #     --exp_dir data/diverse_objects/open_the_door_${id}/task_open_the_door_of_the_storagefurniture_by_its_handle  \
    #     --model_type 3dfa \
    #     > data/logs/eval_${id}.log 2>&1 &

    # CUDA_VISIBLE_DEVICES=$gpu_id python 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/eval_robogen_with_goal_PointNet_gmm_cleaned_old_data_format.py    \
    #     --low_level_exp_dir /project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/data/low-level-ckpt/    \
    #     --low_level_ckpt_name low-level.ckpt     \
    #     --high_level_ckpt_name /project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/test_PointNet2/exps/2025-08-18articubot_cgn_ln_gmm_multivariance_full_450_plus_dagger_load/model_80000.pth   \
    #     --eval_exp_name eval_articubot_both_450_dagger    \
    #     --exp_dir data/diverse_objects/open_the_door_${id}/task_open_the_door_of_the_storagefurniture_by_its_handle \
    #      > data/logs/eval_${id}.log 2>&1 &

    ((i++))
done

wait  # Wait for all background jobs to finish
echo "All jobs finished."
