
# python manipulation/old_test_opening_primitve.py

func=${1}


if [ $# -lt 1 ]; then
    echo "Usage: $0 [func]"
    exit
fi

# demo_name=0511-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
# # save_data_name=0527-act3d-always-close
# save_data_name=0626-act3d-obj-41510-displacement-to-handle
# exp_folder=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle
# task_beg_idx=0
# task_end_idx=1 # for debugging
# opened_threshold=0.65

# # demo_name=0511-vary-obj-4-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
# # save_data_name=0531-act3d-obj-46462
# # exp_folder=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46462_2024-03-27-23-35-10/task_open_the_door_of_the_storagefurniture_by_its_handle
# # task_beg_idx=4
# # task_end_idx=5
# # opened_threshold=2.6

# observation_mode=act3d
# observation_mode=act3d_goal
# observation_mode=dp3_goal_gripper
# observation_mode=dp3_goal_gripper
# observation_mode=act3d_goal_gripper_4
pointcloud_num=4500


if [ $func = 'collect' ]; then 

    demo_name=0511-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name=0511-vary-obj-2-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name=0511-vary-obj-4-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first

    save_data_name=0705-dp3-obj-41510-goal_dense_gripper_on_pcd
    save_data_name=0705-dp3-obj-45448-goal_dense_gripper_on_pcd
    save_data_name=0705-dp3-obj-46462-goal_dense_gripper_on_pcd
    save_data_name=0705-dp3-obj-46732-goal_dense_gripper_on_pcd
    save_data_name=0705-dp3-obj-46801-goal_dense_gripper_on_pcd
    save_data_name=0705-dp3-obj-46874-goal_dense_gripper_on_pcd
    save_data_name=0705-dp3-obj-46922-goal_dense_gripper_on_pcd
    save_data_name=0705-dp3-obj-46966-goal_dense_gripper_on_pcd
    save_data_name=0705-dp3-obj-47570-goal_dense_gripper_on_pcd
    save_data_name=0705-dp3-obj-47578-goal_dense_gripper_on_pcd
    save_data_name=0705-dp3-obj-48700-goal_dense_gripper_on_pcd

    task_beg_idx=0
    task_end_idx=1

    task_beg_idx=2
    task_end_idx=3
    
    task_beg_idx=4
    task_end_idx=5
    
    task_beg_idx=5
    task_end_idx=6
    
    task_beg_idx=6
    task_end_idx=7
    
    task_beg_idx=7
    task_end_idx=8
    
    task_beg_idx=8
    task_end_idx=9
    
    task_beg_idx=9
    task_end_idx=10
    
    task_beg_idx=10
    task_end_idx=11
    
    task_beg_idx=11
    task_end_idx=12
    
    task_beg_idx=12
    task_end_idx=13

    post_fix='dp3_goal_gripper_dense'
    observation_mode="${post_fix}"

    python 3d_diffusion_policy/extract_data_from_states_2.py --folder_name data/temp/ --object_name storagefurniture \
        --save_path "dp3_demo/${save_data_name}" --exp_name "${demo_name}" \
        --task_beg_idx "${task_beg_idx}" --task_end_idx "${task_end_idx}" \
        --pointcloud_num "${pointcloud_num}" \
        --observation_mode "${observation_mode}" \
        --parallel 1 \
        --add_distractors 0 \
        --num_experiment 1000
fi

cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy


# export CUDA_VISIBLE_DEVICES=5
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=4,5,6,7

if [ $func = 'train' ]; then 

    source_dir="/local"

    observation_mode="act3d_goal_mlp"
    # observation_mode='act3d_goal_mlp_displacement_gripper_to_object'
    encoding_mode="keep_position_feature_in_attention_feature"
    # encoding_mode="keep_position_feature_in_attention_feature_with_gripper_displacement_to_closest_object"

    # horizon=4
    horizon=8
    n_obs_steps=2 # 2 or 4
    # num_load_episodes=10 # for debuging

    ##########
    num_epochs=51
    train_ratio=0.9 # for generalization
    num_load_episodes=1000    # for generalization
    pc_channel=3 # we should modify this
    # batch_size=256 #######
    batch_size=72 #######
    encoder_type=act3d
    use_mlp=0
    use_lightweight_unet=0
    in_channels=3 ####
    self_attention=false
    final_attention=false
    
    # normalize_action=true
    # augmentation_rot=false
    # augmentation_pcd=false
    normalize_action=true
    augmentation_rot=false
    augmentation_pcd=true
    augmentation_scale=false
    use_absolute_waypoint=false
    scale_scene_by_pcd=false
    use_chained_diffuser=false
    ##########
    use_attn_for_point_features=false
    pointcloud_backbone='pointnet2'
    ##########
    is_pickle=true
    ##########

    time_stamp=$(date +%m%d%H%M)
    exp_name="${time_stamp}-${observation_mode}-n_obs_steps-${n_obs_steps}-horizon-${horizon}-num_load_episodes-${num_load_episodes}-all_object-pn2-aug_pcd-epsilon"

    action_dim=10
    agent_pos_dim=10
    
    # saved data paths
    save_data_name_0="0622-act3d-obj-45448-remove-reaching-collision-resize-2-full-per-step-gripper-goal-displacement-to-closest-obj-point"
    save_data_name_1="0624-act3d-obj-46462-per-step-combine-2-action-gripper-goal-displacement-to-closest-obj-point-filtered-zero-closing-action"
    save_data_name_2="0626-act3d-obj-41510-per-step-combine-2-action-gripper-goal-displacement-to-closest-obj-point-filtered-zero-closing-action"
    save_data_name_3="0628-act3d-obj-46732-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1"
    save_data_name_4="0628-act3d-obj-46801-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1"
    save_data_name_5="0628-act3d-obj-46874-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1"
    save_data_name_6="0628-act3d-obj-46922-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1"
    save_data_name_7="0628-act3d-obj-46966-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1"
    save_data_name_8="0628-act3d-obj-47570-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1"
    save_data_name_9="0628-act3d-obj-47578-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1"
    save_data_name_10="0628-act3d-obj-48700-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1"    
    save_data_name_11="0705-obj-45526"
    save_data_name_12="0705-obj-45661"
    save_data_name_13="0705-obj-45694"
    save_data_name_14="0705-obj-45780"
    save_data_name_15="0705-obj-45910"
    save_data_name_16="0705-obj-45961"
    save_data_name_17="0705-obj-46408"
    save_data_name_18="0705-obj-46417"
    save_data_name_19="0705-obj-46440"
    save_data_name_20="0705-obj-46490"
    save_data_name_21="0705-obj-46762"
    save_data_name_22="0705-obj-46825"
    save_data_name_23="0705-obj-46893"
    save_data_name_24="0705-obj-47235"
    save_data_name_25="0705-obj-47281"
    save_data_name_26="0705-obj-47315"
    save_data_name_27="0705-obj-47529"
    save_data_name_28="0705-obj-47669"
    save_data_name_29="0705-obj-47944"
    save_data_name_30="0705-obj-48063"
    save_data_name_31="0705-obj-48177"
    save_data_name_32="0705-obj-48356"
    save_data_name_33="0705-obj-48623"
    save_data_name_34="0705-obj-48876"
    save_data_name_35="0705-obj-49025"
    save_data_name_36="0705-obj-49062"
    save_data_name_37="0705-obj-49132"
    save_data_name_38="0705-obj-49133"
    save_data_name_39="0712-obj-40417"
    save_data_name_40="0712-obj-41085"
    save_data_name_41="0712-obj-41452"
    save_data_name_42="0712-obj-45162"
    save_data_name_43="0712-obj-45176"
    save_data_name_44="0712-obj-45194"
    save_data_name_45="0712-obj-45203"
    save_data_name_46="0712-obj-45248"
    save_data_name_47="0712-obj-45271"
    save_data_name_48="0712-obj-45290"
    save_data_name_49="0712-obj-45305"
    save_data_name_50="0725-obj-45413"
    save_data_name_51="0725-obj-45420"
    save_data_name_52="0725-obj-45427"
    save_data_name_53="0725-obj-45594"
    save_data_name_54="0725-obj-45620"
    save_data_name_55="0725-obj-45623"
    save_data_name_56="0725-obj-45636"
    save_data_name_57="0725-obj-45670"
    save_data_name_58="0725-obj-45689"
    save_data_name_59="0725-obj-45696"
    save_data_name_60="0725-obj-45749"
    save_data_name_61="0725-obj-45759"
    save_data_name_62="0725-obj-45916"
    save_data_name_63="0725-obj-45936"
    save_data_name_64="0725-obj-45950"
    save_data_name_65="0725-obj-45984"
    save_data_name_66="0725-obj-46092"
    save_data_name_67="0725-obj-46130"
    save_data_name_68="0725-obj-46134"
    save_data_name_69="0725-obj-46197"
    save_data_name_70="0725-obj-46401"
    save_data_name_71="0725-obj-46456"
    save_data_name_72="0725-obj-46480"
    save_data_name_73="0725-obj-46481"
    save_data_name_74="0725-obj-46544"
    save_data_name_75="0725-obj-46641"
    save_data_name_76="0725-obj-47178"
    save_data_name_77="0725-obj-47182"
    save_data_name_78="0725-obj-47227"
    save_data_name_79="0725-obj-47577"
    save_data_name_80="0725-obj-47648"
    save_data_name_81="0725-obj-47747"
    save_data_name_82="0725-obj-47808"
    save_data_name_83="0725-obj-47976"
    save_data_name_84="0725-obj-48010"
    save_data_name_85="0725-obj-48258"
    save_data_name_86="0725-obj-48379"
    save_data_name_87="0725-obj-48797"
    save_data_name_88="0725-obj-48855"
    save_data_name_89="0725-obj-48859"
    save_data_name_90="0725-obj-49188"
    save_data_name_91="0730-obj-35059"
    save_data_name_92="0730-obj-41004"
    save_data_name_93="0730-obj-41083"
    save_data_name_94="0730-obj-41529"
    save_data_name_95="0730-obj-44781"
    save_data_name_96="0730-obj-44826"
    save_data_name_97="0730-obj-44853"
    save_data_name_98="0730-obj-45092"
    save_data_name_99="0730-obj-45130"
    save_data_name_100="0730-obj-45135"
    save_data_name_101="0730-obj-45146"
    save_data_name_102="0730-obj-45164"
    save_data_name_103="0730-obj-45168"
    save_data_name_104="0730-obj-45173"
    save_data_name_105="0730-obj-45212"
    save_data_name_106="0730-obj-45213"
    save_data_name_107="0730-obj-45372"
    save_data_name_108="0730-obj-45374"
    save_data_name_109="0730-obj-45387"
    save_data_name_110="0730-obj-45415"
    save_data_name_111="0730-obj-45419"
    save_data_name_112="0730-obj-45423"
    save_data_name_113="0730-obj-45503"
    save_data_name_114="0730-obj-45505"
    save_data_name_115="0730-obj-45524"
    save_data_name_116="0730-obj-45573"
    save_data_name_117="0730-obj-45575"
    save_data_name_118="0730-obj-45606"
    save_data_name_119="0730-obj-45612"
    save_data_name_120="0730-obj-45621"
    save_data_name_121="0730-obj-45622"
    save_data_name_122="0730-obj-45632"
    save_data_name_123="0730-obj-45638"
    save_data_name_124="0730-obj-45645"
    save_data_name_125="0730-obj-45662"
    save_data_name_126="0730-obj-45671"
    save_data_name_127="0730-obj-45676"
    save_data_name_128="0730-obj-45677"
    save_data_name_129="0730-obj-45687"
    save_data_name_130="0730-obj-45699"
    save_data_name_131="0730-obj-45710"
    save_data_name_132="0730-obj-45746"
    save_data_name_133="0730-obj-45756"
    save_data_name_134="0730-obj-45783"
    save_data_name_135="0730-obj-45784"
    save_data_name_136="0730-obj-45790"
    save_data_name_137="0730-obj-45801"
    save_data_name_138="0730-obj-45822"
    save_data_name_139="0730-obj-45853"
    save_data_name_140="0730-obj-45855"
    save_data_name_141="0730-obj-45915"
    save_data_name_142="0730-obj-45948"
    save_data_name_143="0730-obj-45949"
    save_data_name_144="0730-obj-45963"
    save_data_name_145="0730-obj-45964"
    save_data_name_146="0730-obj-46002"
    save_data_name_147="0730-obj-46019"
    save_data_name_148="0730-obj-46029"
    save_data_name_149="0730-obj-46033"
    save_data_name_150="0730-obj-46037"
    save_data_name_151="0730-obj-46044"
    save_data_name_152="0730-obj-46045"
    save_data_name_153="0730-obj-46060"
    save_data_name_154="0730-obj-46084"
    save_data_name_155="0730-obj-46108"
    save_data_name_156="0730-obj-46117"
    save_data_name_157="0730-obj-46120"
    save_data_name_158="0730-obj-46123"
    save_data_name_159="0730-obj-46145"
    save_data_name_160="0730-obj-46179"
    save_data_name_161="0730-obj-46180"
    save_data_name_162="0730-obj-46199"
    save_data_name_163="0730-obj-46230"
    save_data_name_164="0730-obj-46277"
    save_data_name_165="0730-obj-46380"
    save_data_name_166="0730-obj-46427"
    save_data_name_167="0730-obj-46430"
    save_data_name_168="0730-obj-46439"
    save_data_name_169="0730-obj-46466"
    save_data_name_170="0730-obj-46537"
    save_data_name_171="0730-obj-46549"
    save_data_name_172="0730-obj-46556"
    save_data_name_173="0730-obj-46598"
    save_data_name_174="0730-obj-46616"
    save_data_name_175="0730-obj-46699"
    save_data_name_176="0730-obj-46700"
    save_data_name_177="0730-obj-46741"
    save_data_name_178="0730-obj-46744"
    save_data_name_179="0730-obj-46847"
    save_data_name_180="0730-obj-46856"
    save_data_name_181="0730-obj-46859"
    save_data_name_182="0730-obj-46889"
    save_data_name_183="0730-obj-46906"
    save_data_name_184="0730-obj-46944"
    save_data_name_185="0730-obj-46955"
    save_data_name_186="0730-obj-46981"
    save_data_name_187="0730-obj-47024"
    save_data_name_188="0730-obj-47089"
    save_data_name_189="0730-obj-47183"
    save_data_name_190="0730-obj-47207"
    save_data_name_191="0730-obj-47233"
    save_data_name_192="0730-obj-47252"
    save_data_name_193="0730-obj-47278"
    save_data_name_194="0730-obj-47290"
    save_data_name_195="0730-obj-47296"
    save_data_name_196="0730-obj-47438"
    save_data_name_197="0730-obj-47514"
    save_data_name_198="0730-obj-47595"
    save_data_name_199="0730-obj-47601"
    save_data_name_200="0730-obj-47632"
    save_data_name_201="0730-obj-47701"
    save_data_name_202="0730-obj-47729"
    save_data_name_203="0730-obj-47853"
    save_data_name_204="0730-obj-47926"
    save_data_name_205="0730-obj-48051"
    save_data_name_206="0730-obj-48413"
    save_data_name_207="0730-obj-48452"
    save_data_name_208="0730-obj-48467"
    save_data_name_209="0730-obj-48490"
    save_data_name_210="0730-obj-48513"
    save_data_name_211="0730-obj-48517"
    save_data_name_212="0730-obj-48721"
    save_data_name_213="0730-obj-48746"
    save_data_name_214="0730-obj-48878"
    save_data_name_215="0730-obj-49140"
    
    torchrun --standalone --nproc_per_node=4 \
        train_ddp.py --config-name=dp3.yaml task=robogen_open_door exp_name="${exp_name}" eval_first=0  \
        task.dataset.zarr_path="[\
            ${source_dir}/${save_data_name_0},\
            ${source_dir}/${save_data_name_1},\
            ${source_dir}/${save_data_name_2},\
            ${source_dir}/${save_data_name_3},\
            ${source_dir}/${save_data_name_4},\
            ${source_dir}/${save_data_name_5},\
            ${source_dir}/${save_data_name_6},\
            ${source_dir}/${save_data_name_7},\
            ${source_dir}/${save_data_name_8},\
            ${source_dir}/${save_data_name_9},\
            ${source_dir}/${save_data_name_10},\
            ${source_dir}/${save_data_name_11},\
            ${source_dir}/${save_data_name_12},\
            ${source_dir}/${save_data_name_13},\
            ${source_dir}/${save_data_name_14},\
            ${source_dir}/${save_data_name_15},\
            ${source_dir}/${save_data_name_16},\
            ${source_dir}/${save_data_name_17},\
            ${source_dir}/${save_data_name_18},\
            ${source_dir}/${save_data_name_19},\
            ${source_dir}/${save_data_name_20},\
            ${source_dir}/${save_data_name_21},\
            ${source_dir}/${save_data_name_22},\
            ${source_dir}/${save_data_name_23},\
            ${source_dir}/${save_data_name_24},\
            ${source_dir}/${save_data_name_25},\
            ${source_dir}/${save_data_name_26},\
            ${source_dir}/${save_data_name_27},\
            ${source_dir}/${save_data_name_28},\
            ${source_dir}/${save_data_name_29},\
            ${source_dir}/${save_data_name_30},\
            ${source_dir}/${save_data_name_31},\
            ${source_dir}/${save_data_name_32},\
            ${source_dir}/${save_data_name_33},\
            ${source_dir}/${save_data_name_34},\
            ${source_dir}/${save_data_name_35},\
            ${source_dir}/${save_data_name_36},\
            ${source_dir}/${save_data_name_37},\
            ${source_dir}/${save_data_name_38},\
            ${source_dir}/${save_data_name_39},\
            ${source_dir}/${save_data_name_40},\
            ${source_dir}/${save_data_name_41},\
            ${source_dir}/${save_data_name_42},\
            ${source_dir}/${save_data_name_43},\
            ${source_dir}/${save_data_name_44},\
            ${source_dir}/${save_data_name_45},\
            ${source_dir}/${save_data_name_46},\
            ${source_dir}/${save_data_name_47},\
            ${source_dir}/${save_data_name_48},\
            ${source_dir}/${save_data_name_49},\
            ${source_dir}/${save_data_name_50},\
            ${source_dir}/${save_data_name_51},\
            ${source_dir}/${save_data_name_52},\
            ${source_dir}/${save_data_name_53},\
            ${source_dir}/${save_data_name_54},\
            ${source_dir}/${save_data_name_55},\
            ${source_dir}/${save_data_name_56},\
            ${source_dir}/${save_data_name_57},\
            ${source_dir}/${save_data_name_58},\
            ${source_dir}/${save_data_name_59},\
            ${source_dir}/${save_data_name_60},\
            ${source_dir}/${save_data_name_61},\
            ${source_dir}/${save_data_name_62},\
            ${source_dir}/${save_data_name_63},\
            ${source_dir}/${save_data_name_64},\
            ${source_dir}/${save_data_name_65},\
            ${source_dir}/${save_data_name_66},\
            ${source_dir}/${save_data_name_67},\
            ${source_dir}/${save_data_name_68},\
            ${source_dir}/${save_data_name_69},\
            ${source_dir}/${save_data_name_70},\
            ${source_dir}/${save_data_name_71},\
            ${source_dir}/${save_data_name_72},\
            ${source_dir}/${save_data_name_73},\
            ${source_dir}/${save_data_name_74},\
            ${source_dir}/${save_data_name_75},\
            ${source_dir}/${save_data_name_76},\
            ${source_dir}/${save_data_name_77},\
            ${source_dir}/${save_data_name_78},\
            ${source_dir}/${save_data_name_79},\
            ${source_dir}/${save_data_name_80},\
            ${source_dir}/${save_data_name_81},\
            ${source_dir}/${save_data_name_82},\
            ${source_dir}/${save_data_name_83},\
            ${source_dir}/${save_data_name_84},\
            ${source_dir}/${save_data_name_85},\
            ${source_dir}/${save_data_name_86},\
            ${source_dir}/${save_data_name_87},\
            ${source_dir}/${save_data_name_88},\
            ${source_dir}/${save_data_name_89},\
            ${source_dir}/${save_data_name_90},\
            ${source_dir}/${save_data_name_91},\
            ${source_dir}/${save_data_name_92},\
            ${source_dir}/${save_data_name_93},\
            ${source_dir}/${save_data_name_94},\
            ${source_dir}/${save_data_name_95},\
            ${source_dir}/${save_data_name_96},\
            ${source_dir}/${save_data_name_97},\
            ${source_dir}/${save_data_name_98},\
            ${source_dir}/${save_data_name_99},\
            ${source_dir}/${save_data_name_100},\
            ${source_dir}/${save_data_name_101},\
            ${source_dir}/${save_data_name_102},\
            ${source_dir}/${save_data_name_103},\
            ${source_dir}/${save_data_name_104},\
            ${source_dir}/${save_data_name_105},\
            ${source_dir}/${save_data_name_106},\
            ${source_dir}/${save_data_name_107},\
            ${source_dir}/${save_data_name_108},\
            ${source_dir}/${save_data_name_109},\
            ${source_dir}/${save_data_name_110},\
            ${source_dir}/${save_data_name_111},\
            ${source_dir}/${save_data_name_112},\
            ${source_dir}/${save_data_name_113},\
            ${source_dir}/${save_data_name_114},\
            ${source_dir}/${save_data_name_115},\
            ${source_dir}/${save_data_name_116},\
            ${source_dir}/${save_data_name_117},\
            ${source_dir}/${save_data_name_118},\
            ${source_dir}/${save_data_name_119},\
            ${source_dir}/${save_data_name_120},\
            ${source_dir}/${save_data_name_121},\
            ${source_dir}/${save_data_name_122},\
            ${source_dir}/${save_data_name_123},\
            ${source_dir}/${save_data_name_124},\
            ${source_dir}/${save_data_name_125},\
            ${source_dir}/${save_data_name_126},\
            ${source_dir}/${save_data_name_127},\
            ${source_dir}/${save_data_name_128},\
            ${source_dir}/${save_data_name_129},\
            ${source_dir}/${save_data_name_130},\
            ${source_dir}/${save_data_name_131},\
            ${source_dir}/${save_data_name_132},\
            ${source_dir}/${save_data_name_133},\
            ${source_dir}/${save_data_name_134},\
            ${source_dir}/${save_data_name_135},\
            ${source_dir}/${save_data_name_136},\
            ${source_dir}/${save_data_name_137},\
            ${source_dir}/${save_data_name_138},\
            ${source_dir}/${save_data_name_139},\
            ${source_dir}/${save_data_name_140},\
            ${source_dir}/${save_data_name_141},\
            ${source_dir}/${save_data_name_142},\
            ${source_dir}/${save_data_name_143},\
            ${source_dir}/${save_data_name_144},\
            ${source_dir}/${save_data_name_145},\
            ${source_dir}/${save_data_name_146},\
            ${source_dir}/${save_data_name_147},\
            ${source_dir}/${save_data_name_148},\
            ${source_dir}/${save_data_name_149},\
            ${source_dir}/${save_data_name_150},\
            ${source_dir}/${save_data_name_151},\
            ${source_dir}/${save_data_name_152},\
            ${source_dir}/${save_data_name_153},\
            ${source_dir}/${save_data_name_154},\
            ${source_dir}/${save_data_name_155},\
            ${source_dir}/${save_data_name_156},\
            ${source_dir}/${save_data_name_157},\
            ${source_dir}/${save_data_name_158},\
            ${source_dir}/${save_data_name_159},\
            ${source_dir}/${save_data_name_160},\
            ${source_dir}/${save_data_name_161},\
            ${source_dir}/${save_data_name_162},\
            ${source_dir}/${save_data_name_163},\
            ${source_dir}/${save_data_name_164},\
            ${source_dir}/${save_data_name_165},\
            ${source_dir}/${save_data_name_166},\
            ${source_dir}/${save_data_name_167},\
            ${source_dir}/${save_data_name_168},\
            ${source_dir}/${save_data_name_169},\
            ${source_dir}/${save_data_name_170},\
            ${source_dir}/${save_data_name_171},\
            ${source_dir}/${save_data_name_172},\
            ${source_dir}/${save_data_name_173},\
            ${source_dir}/${save_data_name_174},\
            ${source_dir}/${save_data_name_175},\
            ${source_dir}/${save_data_name_176},\
            ${source_dir}/${save_data_name_177},\
            ${source_dir}/${save_data_name_178},\
            ${source_dir}/${save_data_name_179},\
            ${source_dir}/${save_data_name_180},\
            ${source_dir}/${save_data_name_181},\
            ${source_dir}/${save_data_name_182},\
            ${source_dir}/${save_data_name_183},\
            ${source_dir}/${save_data_name_184},\
            ${source_dir}/${save_data_name_185},\
            ${source_dir}/${save_data_name_186},\
            ${source_dir}/${save_data_name_187},\
            ${source_dir}/${save_data_name_188},\
            ${source_dir}/${save_data_name_189},\
            ${source_dir}/${save_data_name_190},\
            ${source_dir}/${save_data_name_191},\
            ${source_dir}/${save_data_name_192},\
            ${source_dir}/${save_data_name_193},\
            ${source_dir}/${save_data_name_194},\
            ${source_dir}/${save_data_name_195},\
            ${source_dir}/${save_data_name_196},\
            ${source_dir}/${save_data_name_197},\
            ${source_dir}/${save_data_name_198},\
            ${source_dir}/${save_data_name_199},\
            ${source_dir}/${save_data_name_200},\
            ${source_dir}/${save_data_name_201},\
            ${source_dir}/${save_data_name_202},\
            ${source_dir}/${save_data_name_203},\
            ${source_dir}/${save_data_name_204},\
            ${source_dir}/${save_data_name_205},\
            ${source_dir}/${save_data_name_206},\
            ${source_dir}/${save_data_name_207},\
            ${source_dir}/${save_data_name_208},\
            ${source_dir}/${save_data_name_209},\
            ${source_dir}/${save_data_name_210},\
            ${source_dir}/${save_data_name_211},\
            ${source_dir}/${save_data_name_212},\
            ${source_dir}/${save_data_name_213},\
            ${source_dir}/${save_data_name_214},\
            ${source_dir}/${save_data_name_215}\
        ]"\
        task.env_runner.demo_experiment_path="[\
            ${source_dir}/${save_data_name_0},\
            ${source_dir}/${save_data_name_1},\
            ${source_dir}/${save_data_name_2},\
            ${source_dir}/${save_data_name_3},\
            ${source_dir}/${save_data_name_4},\
            ${source_dir}/${save_data_name_5},\
            ${source_dir}/${save_data_name_6},\
            ${source_dir}/${save_data_name_7},\
            ${source_dir}/${save_data_name_8},\
            ${source_dir}/${save_data_name_9},\
            ${source_dir}/${save_data_name_10},\
            ${source_dir}/${save_data_name_11},\
            ${source_dir}/${save_data_name_12},\
            ${source_dir}/${save_data_name_13},\
            ${source_dir}/${save_data_name_14},\
            ${source_dir}/${save_data_name_15},\
            ${source_dir}/${save_data_name_16},\
            ${source_dir}/${save_data_name_17},\
            ${source_dir}/${save_data_name_18},\
            ${source_dir}/${save_data_name_19},\
            ${source_dir}/${save_data_name_20},\
            ${source_dir}/${save_data_name_21},\
            ${source_dir}/${save_data_name_22},\
            ${source_dir}/${save_data_name_23},\
            ${source_dir}/${save_data_name_24},\
            ${source_dir}/${save_data_name_25},\
            ${source_dir}/${save_data_name_26},\
            ${source_dir}/${save_data_name_27},\
            ${source_dir}/${save_data_name_28},\
            ${source_dir}/${save_data_name_29},\
            ${source_dir}/${save_data_name_30},\
            ${source_dir}/${save_data_name_31},\
            ${source_dir}/${save_data_name_32},\
            ${source_dir}/${save_data_name_33},\
            ${source_dir}/${save_data_name_34},\
            ${source_dir}/${save_data_name_35},\
            ${source_dir}/${save_data_name_36},\
            ${source_dir}/${save_data_name_37},\
            ${source_dir}/${save_data_name_38},\
            ${source_dir}/${save_data_name_39},\
            ${source_dir}/${save_data_name_40},\
            ${source_dir}/${save_data_name_41},\
            ${source_dir}/${save_data_name_42},\
            ${source_dir}/${save_data_name_43},\
            ${source_dir}/${save_data_name_44},\
            ${source_dir}/${save_data_name_45},\
            ${source_dir}/${save_data_name_46},\
            ${source_dir}/${save_data_name_47},\
            ${source_dir}/${save_data_name_48},\
            ${source_dir}/${save_data_name_49},\
            ${source_dir}/${save_data_name_50},\
            ${source_dir}/${save_data_name_51},\
            ${source_dir}/${save_data_name_52},\
            ${source_dir}/${save_data_name_53},\
            ${source_dir}/${save_data_name_54},\
            ${source_dir}/${save_data_name_55},\
            ${source_dir}/${save_data_name_56},\
            ${source_dir}/${save_data_name_57},\
            ${source_dir}/${save_data_name_58},\
            ${source_dir}/${save_data_name_59},\
            ${source_dir}/${save_data_name_60},\
            ${source_dir}/${save_data_name_61},\
            ${source_dir}/${save_data_name_62},\
            ${source_dir}/${save_data_name_63},\
            ${source_dir}/${save_data_name_64},\
            ${source_dir}/${save_data_name_65},\
            ${source_dir}/${save_data_name_66},\
            ${source_dir}/${save_data_name_67},\
            ${source_dir}/${save_data_name_68},\
            ${source_dir}/${save_data_name_69},\
            ${source_dir}/${save_data_name_70},\
            ${source_dir}/${save_data_name_71},\
            ${source_dir}/${save_data_name_72},\
            ${source_dir}/${save_data_name_73},\
            ${source_dir}/${save_data_name_74},\
            ${source_dir}/${save_data_name_75},\
            ${source_dir}/${save_data_name_76},\
            ${source_dir}/${save_data_name_77},\
            ${source_dir}/${save_data_name_78},\
            ${source_dir}/${save_data_name_79},\
            ${source_dir}/${save_data_name_80},\
            ${source_dir}/${save_data_name_81},\
            ${source_dir}/${save_data_name_82},\
            ${source_dir}/${save_data_name_83},\
            ${source_dir}/${save_data_name_84},\
            ${source_dir}/${save_data_name_85},\
            ${source_dir}/${save_data_name_86},\
            ${source_dir}/${save_data_name_87},\
            ${source_dir}/${save_data_name_88},\
            ${source_dir}/${save_data_name_89},\
            ${source_dir}/${save_data_name_90},\
            ${source_dir}/${save_data_name_91},\
            ${source_dir}/${save_data_name_92},\
            ${source_dir}/${save_data_name_93},\
            ${source_dir}/${save_data_name_94},\
            ${source_dir}/${save_data_name_95},\
            ${source_dir}/${save_data_name_96},\
            ${source_dir}/${save_data_name_97},\
            ${source_dir}/${save_data_name_98},\
            ${source_dir}/${save_data_name_99},\
            ${source_dir}/${save_data_name_100},\
            ${source_dir}/${save_data_name_101},\
            ${source_dir}/${save_data_name_102},\
            ${source_dir}/${save_data_name_103},\
            ${source_dir}/${save_data_name_104},\
            ${source_dir}/${save_data_name_105},\
            ${source_dir}/${save_data_name_106},\
            ${source_dir}/${save_data_name_107},\
            ${source_dir}/${save_data_name_108},\
            ${source_dir}/${save_data_name_109},\
            ${source_dir}/${save_data_name_110},\
            ${source_dir}/${save_data_name_111},\
            ${source_dir}/${save_data_name_112},\
            ${source_dir}/${save_data_name_113},\
            ${source_dir}/${save_data_name_114},\
            ${source_dir}/${save_data_name_115},\
            ${source_dir}/${save_data_name_116},\
            ${source_dir}/${save_data_name_117},\
            ${source_dir}/${save_data_name_118},\
            ${source_dir}/${save_data_name_119},\
            ${source_dir}/${save_data_name_120},\
            ${source_dir}/${save_data_name_121},\
            ${source_dir}/${save_data_name_122},\
            ${source_dir}/${save_data_name_123},\
            ${source_dir}/${save_data_name_124},\
            ${source_dir}/${save_data_name_125},\
            ${source_dir}/${save_data_name_126},\
            ${source_dir}/${save_data_name_127},\
            ${source_dir}/${save_data_name_128},\
            ${source_dir}/${save_data_name_129},\
            ${source_dir}/${save_data_name_130},\
            ${source_dir}/${save_data_name_131},\
            ${source_dir}/${save_data_name_132},\
            ${source_dir}/${save_data_name_133},\
            ${source_dir}/${save_data_name_134},\
            ${source_dir}/${save_data_name_135},\
            ${source_dir}/${save_data_name_136},\
            ${source_dir}/${save_data_name_137},\
            ${source_dir}/${save_data_name_138},\
            ${source_dir}/${save_data_name_139},\
            ${source_dir}/${save_data_name_140},\
            ${source_dir}/${save_data_name_141},\
            ${source_dir}/${save_data_name_142},\
            ${source_dir}/${save_data_name_143},\
            ${source_dir}/${save_data_name_144},\
            ${source_dir}/${save_data_name_145},\
            ${source_dir}/${save_data_name_146},\
            ${source_dir}/${save_data_name_147},\
            ${source_dir}/${save_data_name_148},\
            ${source_dir}/${save_data_name_149},\
            ${source_dir}/${save_data_name_150},\
            ${source_dir}/${save_data_name_151},\
            ${source_dir}/${save_data_name_152},\
            ${source_dir}/${save_data_name_153},\
            ${source_dir}/${save_data_name_154},\
            ${source_dir}/${save_data_name_155},\
            ${source_dir}/${save_data_name_156},\
            ${source_dir}/${save_data_name_157},\
            ${source_dir}/${save_data_name_158},\
            ${source_dir}/${save_data_name_159},\
            ${source_dir}/${save_data_name_160},\
            ${source_dir}/${save_data_name_161},\
            ${source_dir}/${save_data_name_162},\
            ${source_dir}/${save_data_name_163},\
            ${source_dir}/${save_data_name_164},\
            ${source_dir}/${save_data_name_165},\
            ${source_dir}/${save_data_name_166},\
            ${source_dir}/${save_data_name_167},\
            ${source_dir}/${save_data_name_168},\
            ${source_dir}/${save_data_name_169},\
            ${source_dir}/${save_data_name_170},\
            ${source_dir}/${save_data_name_171},\
            ${source_dir}/${save_data_name_172},\
            ${source_dir}/${save_data_name_173},\
            ${source_dir}/${save_data_name_174},\
            ${source_dir}/${save_data_name_175},\
            ${source_dir}/${save_data_name_176},\
            ${source_dir}/${save_data_name_177},\
            ${source_dir}/${save_data_name_178},\
            ${source_dir}/${save_data_name_179},\
            ${source_dir}/${save_data_name_180},\
            ${source_dir}/${save_data_name_181},\
            ${source_dir}/${save_data_name_182},\
            ${source_dir}/${save_data_name_183},\
            ${source_dir}/${save_data_name_184},\
            ${source_dir}/${save_data_name_185},\
            ${source_dir}/${save_data_name_186},\
            ${source_dir}/${save_data_name_187},\
            ${source_dir}/${save_data_name_188},\
            ${source_dir}/${save_data_name_189},\
            ${source_dir}/${save_data_name_190},\
            ${source_dir}/${save_data_name_191},\
            ${source_dir}/${save_data_name_192},\
            ${source_dir}/${save_data_name_193},\
            ${source_dir}/${save_data_name_194},\
            ${source_dir}/${save_data_name_195},\
            ${source_dir}/${save_data_name_196},\
            ${source_dir}/${save_data_name_197},\
            ${source_dir}/${save_data_name_198},\
            ${source_dir}/${save_data_name_199},\
            ${source_dir}/${save_data_name_200},\
            ${source_dir}/${save_data_name_201},\
            ${source_dir}/${save_data_name_202},\
            ${source_dir}/${save_data_name_203},\
            ${source_dir}/${save_data_name_204},\
            ${source_dir}/${save_data_name_205},\
            ${source_dir}/${save_data_name_206},\
            ${source_dir}/${save_data_name_207},\
            ${source_dir}/${save_data_name_208},\
            ${source_dir}/${save_data_name_209},\
            ${source_dir}/${save_data_name_210},\
            ${source_dir}/${save_data_name_211},\
            ${source_dir}/${save_data_name_212},\
            ${source_dir}/${save_data_name_213},\
            ${source_dir}/${save_data_name_214},\
            ${source_dir}/${save_data_name_215}\
        ]" \
        task.env_runner.experiment_name="[]" \
        task.env_runner.experiment_folder="[]" \
        task.env_runner.num_point_in_pc="${pointcloud_num}" \
        task.env_runner.use_joint_angle="${use_joint_angle}" \
        task.env_runner.use_segmask="${use_segmask}" \
        task.env_runner.only_handle_points="${only_handle_points}" \
        task.env_runner.use_absolute_waypoint="${use_absolute_waypoint}" \
        horizon="${horizon}" n_obs_steps="${n_obs_steps}" \
        task.shape_meta.obs.agent_pos.shape="[${agent_pos_dim}]" \
        task.shape_meta.action.shape="[${action_dim}]" \
        policy.pointcloud_encoder_cfg.in_channels="${pc_channel}" \
        task.dataset.observation_mode="${observation_mode}" \
        task.env_runner.observation_mode="${observation_mode}" \
        policy.encoder_type="${encoder_type}" \
        policy.encoder_output_dim=60 \
        policy.normalize_action=${normalize_action} \
        policy.scale_scene_by_pcd=${scale_scene_by_pcd} \
        policy.act3d_encoder_cfg.in_channels=${in_channels} \
        policy.act3d_encoder_cfg.goal_mode=cross_attention_to_goal \
        policy.act3d_encoder_cfg.mode="${encoding_mode}" \
        policy.act3d_encoder_cfg.use_mlp="${use_mlp}" \
        policy.act3d_encoder_cfg.self_attention="${self_attention}" \
        policy.act3d_encoder_cfg.use_attn_for_point_features="${use_attn_for_point_features}" \
        policy.act3d_encoder_cfg.pointcloud_backbone="${pointcloud_backbone}" \
        policy.act3d_encoder_cfg.use_lightweight_unet="${use_lightweight_unet}" \
        policy.act3d_encoder_cfg.final_attention="${final_attention}" \
        task.dataset.enumerate=True \
        training.num_epochs="${num_epochs}" \
        training.rollout_every=50 \
        training.checkpoint_every=2 \
        task.env_runner.max_steps=35 \
        task.dataset.train_ratio="${train_ratio}" \
        task.dataset.num_load_episodes="${num_load_episodes}" \
        task.dataset.kept_in_disk=true \
        task.dataset.load_per_step=true \
        task.dataset.augmentation_rot="${augmentation_rot}" \
        task.dataset.augmentation_pcd="${augmentation_pcd}" \
        task.dataset.augmentation_scale="${augmentation_scale}" \
        task.dataset.scale_scene_by_pcd="${scale_scene_by_pcd}" \
        task.dataset.use_absolute_waypoint="${use_absolute_waypoint}" \
        task.dataset.is_pickle="${is_pickle}" \
        dataloader.batch_size="${batch_size}" \
        val_dataloader.batch_size="${batch_size}"
        

fi 


if [ $func = 'eval' ]; then 
    python eval_robogen_parallel_new.py --config-name=dp3.yaml task=robogen_open_door exp_name=eval
    # python eval_robogen_new.py --config-name=dp3.yaml task=robogen_open_door exp_name=eval
    # python eval_robogen_new.py --config-name=dp3.yaml task=robogen_open_door exp_name=debug
    # singularity shell --bind /project_data/held/yufeiw2/RoboGen_sim2real/:/mnt/RoboGen_sim2real/ --nv /project_data/held/yufeiw2/robogen-dp3-act3d.sif
fi 

