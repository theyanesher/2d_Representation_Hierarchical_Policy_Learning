torchrun --standalone --nproc_per_node=8 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs general.exp_name=cgn_instance_norm_full_fp general.tasks="['cgn']" general.fp_to_full=1 general.first_sa_point=1024 cgn.batch_size=2 cgn.OPTIMIZER.batch_size=2

torchrun --standalone --nproc_per_node=8 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs general.exp_name=articubot_single_wdp_not_full_fp general.tasks="['articubot']"  articubot.batch_size=40 general.fp_to_full=0 general.first_sa_point=2048

torchrun --standalone --nproc_per_node=8 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs general.exp_name=articubot_alone_groupnorm_wdp_articubot_50 articubot.batch_size=15 general_args.num_iterations 300000 general.tasks="['articubot']" articubot.gmm=0

torchrun --standalone --nproc_per_node=8 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs general.exp_name=articubot_alone_groupnorm_wdp_articubot_50 articubot.batch_size=15 general_args.num_iterations 300000 general.tasks="['articubot']" articubot.gmm=0