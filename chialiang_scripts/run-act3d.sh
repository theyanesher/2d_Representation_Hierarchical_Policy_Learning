
# python manipulation/old_test_opening_primitve.py

func=${1}

if [ $# -lt 1 ]; then
    echo "Usage: $0 [func]"
    exit
fi


pointcloud_num=4500

# # python 3d_diffusion_policy/filter_simulation_error.py --folder_name data/temp/ --object_name storagefurniture     --save_path "data/dp3_demo/${save_data_name}" --exp_name "${demo_name}"     --task_beg_idx "${task_beg_idx}" --task_end_idx "${task_end_idx}"     --pointcloud_num "${pointcloud_num}"     --use_extracted 0     --num_experiment 1000     --observation_mode "${observation_mode}"     --parallel 0     --opened_threshold "${opened_threshold}" --demo_folder /project_data/held/yufeiw2/RoboGen_sim2real/data/dp3_demo/0531-act3d-obj-46462
# # python 3d_diffusion_policy/filter_simulation_error.py --folder_name data/temp/ --object_name storagefurniture     --save_path "data/dp3_demo/${save_data_name}" --exp_name "${demo_name}"     --task_beg_idx "${task_beg_idx}" --task_end_idx "${task_end_idx}"     --pointcloud_num "${pointcloud_num}"     --use_extracted 0     --num_experiment 1000     --observation_mode "${observation_mode}"     --parallel 0     --opened_threshold "${opened_threshold}" --demo_folder /project_data/held/yufeiw2/RoboGen_sim2real/data/dp3_demo/0531-act3d-obj-45448


cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy


export CUDA_VISIBLE_DEVICES=0

if [ $func = 'train' ]; then 

    # observation_mode="dp3_goal_gripper_whole"
    # observation_mode="dp3_goal_gripper_part"
    # observation_mode="act3d_goal"
    # observation_mode="act3d_goal_mlp"
    observation_mode='act3d_goal_mlp_displacement_gripper_to_object'
    # encoding_mode="keep_position_feature_in_attention_feature"
    encoding_mode="keep_position_feature_in_attention_feature_with_gripper_displacement_to_closest_object"

    # saved data paths
    save_data_name_0=0626-act3d-obj-41510-per-step-combine-2-action-gripper-goal-displacement-to-closest-obj-point-filtered-zero-closing-action
    save_data_name_1=0622-act3d-obj-45448-remove-reaching-collision-resize-2-full-per-step-gripper-goal-displacement-to-closest-obj-point
    save_data_name_2=0624-act3d-obj-46462-per-step-combine-2-action-gripper-goal-displacement-to-closest-obj-point-filtered-zero-closing-action
    save_data_name_3=0703-act3d-mlp-obj-46732-goal
    save_data_name_4=0703-act3d-mlp-obj-46801-goal
    save_data_name_5=0703-act3d-mlp-obj-46874-goal
    save_data_name_6=0703-act3d-mlp-obj-46922-goal
    save_data_name_7=0703-act3d-mlp-obj-46966-goal
    save_data_name_8=0703-act3d-mlp-obj-47570-goal
    save_data_name_9=0703-act3d-mlp-obj-47578-goal
    save_data_name_10=0703-act3d-mlp-obj-48700-goal

    demo_name_0=0511-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name_1=0511-vary-obj-2-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name_2=0511-vary-obj-4-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name_3=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name_4=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name_5=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name_6=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name_7=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name_8=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name_9=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name_10=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first

    exp_folder_0=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle
    exp_folder_1=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_45448_2024-03-27-22-40-39/task_open_the_door_of_the_storagefurniture_by_its_handle
    exp_folder_2=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46462_2024-03-27-23-35-10/task_open_the_door_of_the_storagefurniture_by_its_handle
    exp_folder_3=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46732_2024-03-27-18-46-00/task_open_the_door_of_the_storagefurniture_by_its_handle
    exp_folder_4=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46801_2024-03-27-20-37-05/task_open_the_door_of_the_storagefurniture_by_its_handle
    exp_folder_5=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46874_2024-03-27-13-57-49/task_open_the_door_of_the_storagefurniture_by_its_handle
    exp_folder_6=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46922_2024-03-27-19-42-45/task_open_the_door_of_the_storagefurniture_by_its_handle
    exp_folder_7=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46966_2024-03-27-16-55-33/task_open_the_door_of_the_storagefurniture_by_its_handle
    exp_folder_8=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_47570_2024-03-27-21-36-50/task_open_the_door_of_the_storagefurniture_by_its_handle
    exp_folder_9=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_47578_2024-03-27-14-56-07/task_open_the_door_of_the_storagefurniture_by_its_handle
    exp_folder_10=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_48700_2024-03-27-12-59-58/task_open_the_door_of_the_storagefurniture_by_its_handle

    save_data_name_11=0705-obj-45526
    demo_name_11=0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_11=data/diverse_objects/open_the_door_45526/task_open_the_door_of_the_storagefurniture_by_its_handle
    save_data_name_12=0705-obj-45661
    demo_name_12=0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_12=data/diverse_objects/open_the_door_45661/task_open_the_door_of_the_storagefurniture_by_its_handle
    save_data_name_13=0705-obj-45694
    demo_name_13=0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_13=data/diverse_objects/open_the_door_45694/task_open_the_door_of_the_storagefurniture_by_its_handle
    save_data_name_14=0705-obj-45780
    demo_name_14=0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_14=data/diverse_objects/open_the_door_45780/task_open_the_door_of_the_storagefurniture_by_its_handle
    save_data_name_15=0705-obj-45910
    demo_name_15=0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_15=data/diverse_objects/open_the_door_45910/task_open_the_door_of_the_storagefurniture_by_its_handle
    save_data_name_16=0705-obj-45961
    demo_name_16=0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_16=data/diverse_objects/open_the_door_45961/task_open_the_door_of_the_storagefurniture_by_its_handle
    save_data_name_17=0705-obj-46408
    demo_name_17=0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_17=data/diverse_objects/open_the_door_46408/task_open_the_door_of_the_storagefurniture_by_its_handle
    save_data_name_18=0705-obj-46417
    demo_name_18=0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_18=data/diverse_objects/open_the_door_46417/task_open_the_door_of_the_storagefurniture_by_its_handle
    save_data_name_19=0705-obj-46440
    demo_name_19=0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_19=data/diverse_objects/open_the_door_46440/task_open_the_door_of_the_storagefurniture_by_its_handle
    save_data_name_20=0705-obj-46490
    demo_name_20=0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_20=data/diverse_objects/open_the_door_46490/task_open_the_door_of_the_storagefurniture_by_its_handle
    save_data_name_21=0705-obj-46762
    demo_name_21=0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_21=data/diverse_objects/open_the_door_46762/task_open_the_door_of_the_storagefurniture_by_its_handle
    save_data_name_22=0705-obj-46825
    demo_name_22=0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_22=data/diverse_objects/open_the_door_46825/task_open_the_door_of_the_storagefurniture_by_its_handle
    save_data_name_23=0705-obj-46893
    demo_name_23=0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_23=data/diverse_objects/open_the_door_46893/task_open_the_door_of_the_storagefurniture_by_its_handle
    save_data_name_24=0705-obj-47235
    demo_name_24=0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_24=data/diverse_objects/open_the_door_47235/task_open_the_door_of_the_storagefurniture_by_its_handle
    save_data_name_25=0705-obj-47281
    demo_name_25=0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_25=data/diverse_objects/open_the_door_47281/task_open_the_door_of_the_storagefurniture_by_its_handle
    save_data_name_26=0705-obj-47315
    demo_name_26=0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_26=data/diverse_objects/open_the_door_47315/task_open_the_door_of_the_storagefurniture_by_its_handle
    save_data_name_27=0705-obj-47529
    demo_name_27=0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_27=data/diverse_objects/open_the_door_47529/task_open_the_door_of_the_storagefurniture_by_its_handle
    save_data_name_28=0705-obj-47669
    demo_name_28=0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_28=data/diverse_objects/open_the_door_47669/task_open_the_door_of_the_storagefurniture_by_its_handle
    save_data_name_29=0705-obj-47944
    demo_name_29=0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_29=data/diverse_objects/open_the_door_47944/task_open_the_door_of_the_storagefurniture_by_its_handle
    save_data_name_30=0705-obj-48063
    demo_name_30=0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_30=data/diverse_objects/open_the_door_48063/task_open_the_door_of_the_storagefurniture_by_its_handle
    save_data_name_31=0705-obj-48177
    demo_name_31=0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_31=data/diverse_objects/open_the_door_48177/task_open_the_door_of_the_storagefurniture_by_its_handle
    save_data_name_32=0705-obj-48356
    demo_name_32=0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_32=data/diverse_objects/open_the_door_48356/task_open_the_door_of_the_storagefurniture_by_its_handle
    save_data_name_33=0705-obj-48623
    demo_name_33=0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_33=data/diverse_objects/open_the_door_48623/task_open_the_door_of_the_storagefurniture_by_its_handle
    save_data_name_34=0705-obj-48876
    demo_name_34=0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_34=data/diverse_objects/open_the_door_48876/task_open_the_door_of_the_storagefurniture_by_its_handle
    save_data_name_35=0705-obj-49025
    demo_name_35=0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_35=data/diverse_objects/open_the_door_49025/task_open_the_door_of_the_storagefurniture_by_its_handle
    save_data_name_36=0705-obj-49062
    demo_name_36=0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_36=data/diverse_objects/open_the_door_49062/task_open_the_door_of_the_storagefurniture_by_its_handle
    save_data_name_37=0705-obj-49132
    demo_name_37=0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_37=data/diverse_objects/open_the_door_49132/task_open_the_door_of_the_storagefurniture_by_its_handle
    save_data_name_38=0705-obj-49133
    demo_name_38=0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_38=data/diverse_objects/open_the_door_49133/task_open_the_door_of_the_storagefurniture_by_its_handle
    save_data_name_39=0712-obj-40417
    demo_name_39=0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_39=data/diverse_objects_2/open_the_door_40417/task_open_the_door_of_the_storagefurniture_by_its_handle
    save_data_name_40=0712-obj-41085
    demo_name_40=0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_40=data/diverse_objects_2/open_the_door_41085/task_open_the_door_of_the_storagefurniture_by_its_handle
    save_data_name_41=0712-obj-41452
    demo_name_41=0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_41=data/diverse_objects_2/open_the_door_41452/task_open_the_door_of_the_storagefurniture_by_its_handle
    save_data_name_42=0712-obj-45162
    demo_name_42=0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_42=data/diverse_objects_2/open_the_door_45162/task_open_the_door_of_the_storagefurniture_by_its_handle
    save_data_name_43=0712-obj-45176
    demo_name_43=0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_43=data/diverse_objects_2/open_the_door_45176/task_open_the_door_of_the_storagefurniture_by_its_handle
    save_data_name_44=0712-obj-45194
    demo_name_44=0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_44=data/diverse_objects_2/open_the_door_45194/task_open_the_door_of_the_storagefurniture_by_its_handle
    save_data_name_45=0712-obj-45203
    demo_name_45=0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_45=data/diverse_objects_2/open_the_door_45203/task_open_the_door_of_the_storagefurniture_by_its_handle
    save_data_name_46=0712-obj-45248
    demo_name_46=0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_46=data/diverse_objects_2/open_the_door_45248/task_open_the_door_of_the_storagefurniture_by_its_handle
    save_data_name_47=0712-obj-45271
    demo_name_47=0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_47=data/diverse_objects_2/open_the_door_45271/task_open_the_door_of_the_storagefurniture_by_its_handle
    save_data_name_48=0712-obj-45290
    demo_name_48=0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_48=data/diverse_objects_2/open_the_door_45290/task_open_the_door_of_the_storagefurniture_by_its_handle
    save_data_name_49=0712-obj-45305
    demo_name_49=0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    exp_folder_49=data/diverse_objects_2/open_the_door_45305/task_open_the_door_of_the_storagefurniture_by_its_handle


    # horizon=4
    horizon=8
    n_obs_steps=2
    # num_load_episodes=10 # for debuging

    ##########
    train_ratio=0.9 # for generalization
    num_load_episodes=1000 # for generalization
    pc_channel=3 # we should modify this
    batch_size=480 #######
    encoder_type=act3d
    use_mlp=1
    in_channels=3 ####
    ##########

    time_stamp=$(date +%m%d%H%M)
    exp_name="${time_stamp}-${observation_mode}-horizon-${horizon}-num_load_episodes-${num_load_episodes}"

    action_dim=10
    agent_pos_dim=10
    
    python train.py --config-name=dp3.yaml task=robogen_open_door exp_name="${exp_name}" eval_first=0  \
        task.dataset.zarr_path="[\
            /scratch/yufei/dp3_demo/${save_data_name_0},\
            /scratch/yufei/dp3_demo/${save_data_name_1},\
            /scratch/yufei/dp3_demo/${save_data_name_2},\
            /scratch/chialiang/dp3_demo/${save_data_name_3},\
            /scratch/chialiang/dp3_demo/${save_data_name_4},\
            /scratch/chialiang/dp3_demo/${save_data_name_5},\
            /scratch/chialiang/dp3_demo/${save_data_name_6},\
            /scratch/chialiang/dp3_demo/${save_data_name_7},\
            /scratch/chialiang/dp3_demo/${save_data_name_8},\
            /scratch/chialiang/dp3_demo/${save_data_name_9},\
            /scratch/chialiang/dp3_demo/${save_data_name_10},\

            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_11},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_12},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_13},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_14},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_15},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_16},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_17},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_18},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_19},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_20},\

            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_21},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_22},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_23},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_24},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_25},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_26},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_27},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_28},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_29},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_30},\

            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_31},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_32},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_33},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_34},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_35},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_36},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_37},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_38},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_39},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_40},\

            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_41},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_42},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_43},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_44},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_45},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_46},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_47},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_48},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_49},\
        ]"\
        task.env_runner.demo_experiment_path="[\
            /scratch/yufei/dp3_demo/${save_data_name_0},\
            /scratch/yufei/dp3_demo/${save_data_name_1},\
            /scratch/yufei/dp3_demo/${save_data_name_2},\
            /scratch/chialiang/dp3_demo/${save_data_name_3},\
            /scratch/chialiang/dp3_demo/${save_data_name_4},\
            /scratch/chialiang/dp3_demo/${save_data_name_5},\
            /scratch/chialiang/dp3_demo/${save_data_name_6},\
            /scratch/chialiang/dp3_demo/${save_data_name_7},\
            /scratch/chialiang/dp3_demo/${save_data_name_8},\
            /scratch/chialiang/dp3_demo/${save_data_name_9},\
            /scratch/chialiang/dp3_demo/${save_data_name_10},\

            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_11},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_12},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_13},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_14},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_15},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_16},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_17},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_18},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_19},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_20},\

            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_21},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_22},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_23},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_24},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_25},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_26},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_27},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_28},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_29},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_30},\

            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_31},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_32},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_33},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_34},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_35},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_36},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_37},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_38},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_39},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_40},\

            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_41},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_42},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_43},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_44},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_45},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_46},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_47},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_48},\
            /project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/${save_data_name_49},\
        ]" \
        task.env_runner.experiment_name="[\
            ${demo_name_0},\
            ${demo_name_1},\
            ${demo_name_2},\
            ${demo_name_3},\
            ${demo_name_4},\
            ${demo_name_5},\
            ${demo_name_6},\
            ${demo_name_7},\
            ${demo_name_8},\
            ${demo_name_9},\
            ${demo_name_10},\

            ${demo_name_11},\
            ${demo_name_12},\
            ${demo_name_13},\
            ${demo_name_14},\
            ${demo_name_15},\
            ${demo_name_16},\
            ${demo_name_17},\
            ${demo_name_18},\
            ${demo_name_19},\
            ${demo_name_20},\

            ${demo_name_21},\
            ${demo_name_22},\
            ${demo_name_23},\
            ${demo_name_24},\
            ${demo_name_25},\
            ${demo_name_26},\
            ${demo_name_27},\
            ${demo_name_28},\
            ${demo_name_29},\
            ${demo_name_30},\

            ${demo_name_31},\
            ${demo_name_32},\
            ${demo_name_33},\
            ${demo_name_34},\
            ${demo_name_35},\
            ${demo_name_36},\
            ${demo_name_37},\
            ${demo_name_38},\
            ${demo_name_39},\
            ${demo_name_40},\

            ${demo_name_41},\
            ${demo_name_42},\
            ${demo_name_43},\
            ${demo_name_44},\
            ${demo_name_45},\
            ${demo_name_46},\
            ${demo_name_47},\
            ${demo_name_48},\
            ${demo_name_49},\
        ]" \
        task.env_runner.experiment_folder="[\
            ${exp_folder_0},\
            ${exp_folder_1},\
            ${exp_folder_2},\
            ${exp_folder_3},\
            ${exp_folder_4},\
            ${exp_folder_5},\
            ${exp_folder_6},\
            ${exp_folder_7},\
            ${exp_folder_8},\
            ${exp_folder_9},\
            ${exp_folder_10},\

            ${exp_folder_11},\
            ${exp_folder_12},\
            ${exp_folder_13},\
            ${exp_folder_14},\
            ${exp_folder_15},\
            ${exp_folder_16},\
            ${exp_folder_17},\
            ${exp_folder_18},\
            ${exp_folder_19},\
            ${exp_folder_20},\

            ${exp_folder_21},\
            ${exp_folder_22},\
            ${exp_folder_23},\
            ${exp_folder_24},\
            ${exp_folder_25},\
            ${exp_folder_26},\
            ${exp_folder_27},\
            ${exp_folder_28},\
            ${exp_folder_29},\
            ${exp_folder_30},\

            ${exp_folder_31},\
            ${exp_folder_32},\
            ${exp_folder_33},\
            ${exp_folder_34},\
            ${exp_folder_35},\
            ${exp_folder_36},\
            ${exp_folder_37},\
            ${exp_folder_38},\
            ${exp_folder_39},\
            ${exp_folder_40},\

            ${exp_folder_41},\
            ${exp_folder_42},\
            ${exp_folder_43},\
            ${exp_folder_44},\
            ${exp_folder_45},\
            ${exp_folder_46},\
            ${exp_folder_47},\
            ${exp_folder_48},\
            ${exp_folder_49},\
        ]" \
        task.env_runner.num_point_in_pc="${pointcloud_num}" \
        task.env_runner.use_joint_angle="${use_joint_angle}" \
        task.env_runner.use_segmask="${use_segmask}" \
        task.env_runner.only_handle_points="${only_handle_points}" \
        horizon="${horizon}" n_obs_steps="${n_obs_steps}" \
        task.shape_meta.obs.agent_pos.shape="[${agent_pos_dim}]" \
        task.shape_meta.action.shape="[${action_dim}]" \
        policy.pointcloud_encoder_cfg.in_channels="${pc_channel}" \
        task.dataset.observation_mode="${observation_mode}" \
        task.env_runner.observation_mode="${observation_mode}" \
        policy.encoder_type="${encoder_type}" \
        policy.encoder_output_dim=60 \
        policy.act3d_encoder_cfg.in_channels=${in_channels} \
        policy.act3d_encoder_cfg.goal_mode=cross_attention_to_goal \
        policy.act3d_encoder_cfg.mode="${encoding_mode}" \
        policy.act3d_encoder_cfg.use_mlp="${use_mlp}" \
        task.dataset.enumerate=True \
        load_checkpoint_path="/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/07081911-act3d_goal_mlp_displacement_gripper_to_object-horizon-8-num_load_episodes-1000/2024.07.08/19.11.50_train_dp3_robogen_open_door/checkpoints/latest.ckpt" \
        training.num_epochs=110 \
        training.rollout_every=50 \
        training.checkpoint_every=20 \
        task.env_runner.max_steps=35 \
        task.dataset.train_ratio="${train_ratio}" \
        task.dataset.num_load_episodes="${num_load_episodes}" \
        task.dataset.kept_in_disk=true \
        task.dataset.load_per_step=true \
        dataloader.batch_size="${batch_size}" \
        val_dataloader.batch_size="${batch_size}" \

fi 


if [ $func = 'eval' ]; then 
    python eval_robogen_parallel_new.py --config-name=dp3.yaml task=robogen_open_door exp_name=eval
    # python eval_robogen_new.py --config-name=dp3.yaml task=robogen_open_door exp_name=eval
    # python eval_robogen_new.py --config-name=dp3.yaml task=robogen_open_door exp_name=debug
    # singularity shell --bind /project_data/held/yufeiw2/RoboGen_sim2real/:/mnt/RoboGen_sim2real/ --nv /project_data/held/yufeiw2/robogen-dp3-act3d.sif
fi 



# # python manipulation/gen_demo/gen_demo.py

# demo_name=0511-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
# # save_data_name=0527-act3d-always-close
# save_data_name=0617-act3d-obj-41510-displacement-to-handle
# exp_folder=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle
# task_beg_idx=0
# task_end_idx=1
# opened_threshold=0.65

# # demo_name=0511-vary-obj-2-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
# # # save_data_name=0531-act3d-obj-45448
# # save_data_name=0611-act3d-obj-45448-distractors-wzy
# # exp_folder=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_45448_2024-03-27-22-40-39/task_open_the_door_of_the_storagefurniture_by_its_handle
# # task_beg_idx=2
# # task_end_idx=3
# # opened_threshold=0.4

# # # demo_name=0511-vary-obj-4-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
# # # save_data_name=0531-act3d-obj-46462
# # # exp_folder=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46462_2024-03-27-23-35-10/task_open_the_door_of_the_storagefurniture_by_its_handle
# # # task_beg_idx=4
# # # task_end_idx=5
# # # opened_threshold=2.6

# # observation_mode=act3d
# observation_mode=act3d_goal
# # observation_mode=act3d_displacement_to_handle
# pointcloud_num=4500

# # # python 3d_diffusion_policy/filter_simulation_error.py --folder_name data/temp/ --object_name storagefurniture     --save_path "data/dp3_demo/${save_data_name}" --exp_name "${demo_name}"     --task_beg_idx "${task_beg_idx}" --task_end_idx "${task_end_idx}"     --pointcloud_num "${pointcloud_num}"     --use_extracted 0     --num_experiment 1000     --observation_mode "${observation_mode}"     --parallel 0     --opened_threshold "${opened_threshold}" --demo_folder /project_data/held/yufeiw2/RoboGen_sim2real/data/dp3_demo/0531-act3d-obj-46462
# # # python 3d_diffusion_policy/filter_simulation_error.py --folder_name data/temp/ --object_name storagefurniture     --save_path "data/dp3_demo/${save_data_name}" --exp_name "${demo_name}"     --task_beg_idx "${task_beg_idx}" --task_end_idx "${task_end_idx}"     --pointcloud_num "${pointcloud_num}"     --use_extracted 0     --num_experiment 1000     --observation_mode "${observation_mode}"     --parallel 0     --opened_threshold "${opened_threshold}" --demo_folder /project_data/held/yufeiw2/RoboGen_sim2real/data/dp3_demo/0531-act3d-obj-45448

# # python 3d_diffusion_policy/extract_data_from_states_2.py --folder_name data/temp/ --object_name storagefurniture \
# #     --save_path "data/dp3_demo/${save_data_name}" --exp_name "${demo_name}" \
# #     --task_beg_idx "${task_beg_idx}" --task_end_idx "${task_end_idx}" \
# #     --pointcloud_num "${pointcloud_num}" \
# #     --use_extracted 0 \
# #     --num_experiment 40 \
# #     --observation_mode "${observation_mode}" \
# #     --parallel 0 \
# #     --opened_threshold "${opened_threshold}" \
# #     --add_distractors 0

# # exit

# cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy

# # save_data_name_0=0527-act3d-always-close
# # save_data_name_0=0527-act3d-always-close-with-goal
# # save_data_name_0=0607-act3d-obj-41510-remove-reaching-collision-resize-2
# save_data_name_0=0617-act3d-obj-41510-remove-reaching-collision-resize-2-goal
# # save_data_name_1=0531-act3d-obj-45448
# save_data_name_1=0607-act3d-obj-45448-remove-reaching-collision-resize-2-full
# # save_data_name_2=0531-act3d-obj-46462
# save_data_name_2=0607-act3d-obj-46462-remove-reaching-collision-resize-2

# demo_name_0=0511-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
# demo_name_1=0511-vary-obj-2-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
# demo_name_2=0511-vary-obj-4-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first

# exp_folder_0=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle
# exp_folder_1=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_45448_2024-03-27-22-40-39/task_open_the_door_of_the_storagefurniture_by_its_handle
# exp_folder_2=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46462_2024-03-27-23-35-10/task_open_the_door_of_the_storagefurniture_by_its_handle

# # horizon=4
# horizon=8
# n_obs_steps=2
# train_ratio=1
# num_load_episodes=10
# # exp_name="0528-act3d-train-ratio-${train_ratio}"
# # exp_name="0602-act3d-obj-45448-train-ratio-${train_ratio}"
# # exp_name="0604-act3d-obj-46462-train-ratio-${train_ratio}-filtered"
# # exp_name="0603-act3d-3-obj-train-ratio-${train_ratio}"
# # exp_name="0606-act3d-3-obj-train-ratio-${train_ratio}"
# # exp_name="0608-act3d-obj-41510-goal-train-ratio-${train_ratio}"
# # exp_name="0609-act3d-obj-45448-horizon-${horizon}-train-ratio-${train_ratio}"
# # exp_name="0612-act3d-3-obj-horizon-${horizon}-num_load_episodes-${num_load_episodes}"
# exp_name="0618-act3d-goal-horizon-${horizon}-num_load_episodes-${num_load_episodes}"
# exp_name="test-robo-cluster-singularity"

# action_dim=10
# agent_pos_dim=10
# pc_channel=3 
# batch_size=60

# # python train.py --config-name=dp3.yaml task=robogen_open_door exp_name="${exp_name}" eval_first=0  \
# #     task.dataset.zarr_path="[${PROJECT_DIR}/data/dp3_demo/${save_data_name_0}]" \
# #     task.env_runner.demo_experiment_path="[${PROJECT_DIR}/data/dp3_demo/${save_data_name_0}]" \
# #     task.env_runner.experiment_name="[${demo_name_0}]" \
# #     task.env_runner.experiment_folder="[${exp_folder_0}]" \
# #     task.env_runner.num_point_in_pc="${pointcloud_num}" \
# #     task.env_runner.use_joint_angle="${use_joint_angle}" \
# #     task.env_runner.use_segmask="${use_segmask}" \
# #     task.env_runner.only_handle_points="${only_handle_points}" \
# #     horizon="${horizon}" n_obs_steps="${n_obs_steps}" \
# #     task.shape_meta.obs.agent_pos.shape="[${agent_pos_dim}]" \
# #     task.shape_meta.action.shape="[${action_dim}]" \
# #     policy.pointcloud_encoder_cfg.in_channels="${pc_channel}" \
# #     task.dataset.observation_mode="${observation_mode}" \
# #     task.env_runner.observation_mode="${observation_mode}" \
# #     policy.encoder_type=act3d \
# #     policy.encoder_output_dim=60 \
# #     policy.act3d_encoder_cfg.goal_mode=cross_attention_to_goal \
# #     policy.act3d_encoder_cfg.mode=keep_position_feature_in_attention_feature \
# #     task.dataset.enumerate=True \
# #     training.rollout_every=200 \
# #     training.checkpoint_every=200 \
# #     task.env_runner.max_steps=35 \
# #     task.dataset.train_ratio="${train_ratio}" \
# #     task.dataset.num_load_episodes="${num_load_episodes}" \
# #     task.dataset.kept_in_disk=false \
# #     task.dataset.load_per_step=false \

# python eval_robogen_parallel_new.py --config-name=dp3.yaml task=robogen_open_door exp_name=eval
# # python eval_robogen_new.py --config-name=dp3.yaml task=robogen_open_door exp_name=debug
# # singularity shell --bind /project_data/held/yufeiw2/RoboGen_sim2real/:/mnt/RoboGen_sim2real/ --nv /project_data/held/yufeiw2/robogen-dp3-act3d.sif
