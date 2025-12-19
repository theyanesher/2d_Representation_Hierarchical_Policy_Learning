torchrun --standalone --nproc_per_node=1 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs    \
    general.exp_name=fine_tune_our_on_sriram_towel_one_hot \
    articubot.batch_size=20   \
    articubot.num_train_objects=sriram_towel \
    articubot.is_pickle=0 \
    general.tasks="['articubot']" \
    general.use_lr_scheduler=1 \
    general.add_one_hot_encoding=1 \
    general.load_model_path=/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/test_PointNet2/exps/2025-09-14multinode-cgn-world-articubot-all-w-pick-place/model_155001.pth 


# torchrun --standalone --nproc_per_node=1 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs    \
#     general.exp_name=debug \
#     articubot.batch_size=20   \
#     articubot.num_train_objects=sriram_plate_new_rot_rgb \
#     articubot.is_pickle=0 \
#     general.tasks="['articubot']" \
#     general.use_lr_scheduler=1 \
#     general.use_rgb=1 \
#     general.load_model_path=/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/test_PointNet2/exps/2025-09-14multinode-cgn-world-articubot-all-w-pick-place/model_155001.pth 


# torchrun --standalone --nproc_per_node=1 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs    \
#     general.exp_name=fine_tune_our_on_sriram_new_rot \
#     articubot.batch_size=20   \
#     articubot.num_train_objects=sriram_plate_new_rot \
#     articubot.is_pickle=0 \
#     general.tasks="['articubot']" \
#     general.use_lr_scheduler=1 \
#     general.load_model_path=/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/test_PointNet2/exps/2025-09-14multinode-cgn-world-articubot-all-w-pick-place/model_155001.pth 