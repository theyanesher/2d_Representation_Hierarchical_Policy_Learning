"""
this file is used to load a pretrained high and low-level policy and run through mimicgen eval
"""

from numpy import False_
from termcolor import cprint
import os

### load a pretrained high and low-level policy
from eval_smith_utils import load_multitask_high_level_model, load_low_level_policy

# high_level_path = "/data/robogen/smith_mimicgen/data/high-level-ckpt/2026-03-07fine_tune_mimicgen_square_d2/model_97501.pth"
# low_level_exp_dir = "/data/robogen/smith_mimicgen/data/low-level-ckpt/0307_finetune_ours_mimicgen_square_d2_not_keep_old_normalizer"
# low_level_checkpoint = "epoch-240.ckpt"

### build the env runner
### Default kwargs from ArticubotPCDRunner.__init__ and equi_diffpo/config/task/articubot_pc_abs_eval_512.yaml
from equi_diffpo.env_runner.articubot_pcd_runner import ArticubotPCDRunner

# shape_meta from articubot_pc_abs_eval_512.yaml env_runner_shape_meta (no goal_gripper_pcd for runner)
DEFAULT_ENV_RUNNER_SHAPE_META = {
    "obs": {
        "robot0_eye_in_hand_image": {"shape": [3, 512, 512], "type": "rgb"},
        "agentview_image": {"shape": [3, 512, 512], "type": "rgb"},
        "sideview_image": {"shape": [3, 512, 512], "type": "rgb"},
        "birdview_image": {"shape": [3, 512, 512], "type": "rgb"},
        "point_cloud": {"shape": [4500, 3], "type": "point_cloud"},
        "gripper_pcd": {"shape": [4, 3], "type": "point_cloud"},
        "robot0_eef_pos": {"shape": [3]},
        "robot0_eef_quat": {"shape": [4]},
        "robot0_gripper_qpos": {"shape": [2]},
        # "agent_pos": {"shape": [10]},
        "state": {"shape": [10]},
    },
    "action": {"shape": [10]},
}

max_steps = {
    'stack_d1': 400,
    'stack_three_d1': 400,
    'square_d0': 400,
    'square_d2': 400,
    'threading_d2': 400,
    'coffee_d2': 400,
    'three_piece_assembly_d2': 500,
    'hammer_cleanup_d1': 500,
    'mug_cleanup_d1': 500,
    'kitchen_d1': 800,
    'nut_assembly_d0': 500,
    'pick_place_d0': 1000,
    'coffee_preparation_d1': 800,
    'tool_hang': 700,
    'can': 400,
    'lift': 400,
    'square': 400,
}

def get_articubot_pcd_runner_default_kwargs(
    output_dir,
    dataset_path,
    shape_meta=None,
    **overrides,
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    """Default kwargs for ArticubotPCDRunner from __init__ defaults + articubot_pc_abs_eval_512.yaml."""
    kwargs = {
        "output_dir": output_dir,
        "dataset_path": dataset_path,
        "shape_meta": shape_meta if shape_meta is not None else DEFAULT_ENV_RUNNER_SHAPE_META,
        "n_train": 6,
        "n_test": 50,
        "n_train_vis": 6,
        "train_start_idx": 0,
        "test_start_idx": 950,
        "n_test_vis": 100,
        "test_start_seed": 10000,
        "max_steps": max_steps[dataset_path.split("/")[-1].split(".")[0]],
        "n_obs_steps": 2,
        "n_action_steps": 4,
        "render_obs_key": "agentview_image",
        "fps": 10,
        "crf": 22,
        "past_action": False,
        "abs_action": False,
        "tqdm_interval_sec": 1.0,
        "n_envs": 1,
        "cat_idx": 0,
    }
    kwargs.update(overrides)
    return kwargs


if __name__ == "__main__":
    ### square d2
    # high_level_path = "/data/robogen/smith_mimicgen/data/high-level-ckpt/2026-03-08fine_tune_mimicgen_square_d2_correct/model_90001.pth"
    # high_level_path = "/data/robogen/smith_mimicgen/data/high-level-ckpt/2026-03-09fine_tune_mimicgen_square_d2_correct_one_hot/model_47501.pth"
    # # high_level_path = "/data/robogen/smith_mimicgen/data/high-level-ckpt/2026-03-10scratch_mimicgen_square_d2_one_hot/model_47501.pth"
    # high_level_path = "/data/robogen/smith_mimicgen/data/high-level-ckpt/2026-03-10scratch_mimicgen_square_d2_one_hot/model_97501.pth"

    ### three piece assembly d2
    # high_level_path = "/data/robogen/smith_mimicgen/data/high-level-ckpt/2026-03-10fine_tune_mimicgen_three_piece_assembly_d2_one_hot/model_80001.pth"
    # high_level_path = "/data/robogen/smith_mimicgen/data/high-level-ckpt/2026-03-10fine_tune_mimicgen_three_piece_assembly_d2_one_hot/model_97501.pth"
    # high_level_path = "/data/robogen/smith_mimicgen/data/high-level-ckpt/2026-03-11scratch_mimicgen_three_piece_assembly_d2_one_hot/model_97501.pth"

    ### threading d2
    # high_level_path = "/data/robogen/smith_mimicgen/data/high-level-ckpt/2026-03-13fine_tune_mimicgen_threading_d2_one_hot/model_97501.pth"
    high_level_path = "/data/robogen/smith_mimicgen/data/high-level-ckpt/2026-03-13fine_tune_mimicgen_threading_d2_one_hot/model_97501.pth"

    ### square d2 
    # low_level_exp_dir = "/data/robogen/smith_mimicgen/data/low-level-ckpt/0307_finetune_ours_mimicgen_square_d2_correct_not_keep_old_normalizer"
    # low_level_exp_dir = "/data/robogen/smith_mimicgen/data/low-level-ckpt/0307_finetune_ours_mimicgen_square_d2_correct_keep_old_normalizer"
    # low_level_exp_dir = "/data/robogen/smith_mimicgen/data/low-level-ckpt/0310_scratch_mimicgen_square_d2_correct/"

    ### three piece assembly d2
    # low_level_exp_dir = "/data/robogen/smith_mimicgen/data/low-level-ckpt/0310_finetune_ours_mimicgen_three_piece_assembly_d2_keep_old_normalizer"
    # low_level_exp_dir = "/data/robogen/smith_mimicgen/data/low-level-ckpt/0311_scratch_mimicgen_three_piece_assembly_d2"

    ### threading d2
    low_level_exp_dir = "/data/robogen/smith_mimicgen/data/low-level-ckpt/0313_finetune_ours_mimicgen_threading_d2_keep_old_normalizer"

    low_level_checkpoint = "epoch-300.ckpt"

    high_level_policy, high_level_args = load_multitask_high_level_model(high_level_path)
    low_level_policy = load_low_level_policy(low_level_exp_dir, low_level_checkpoint)

    output_dir = "./eval_output/debug_correct_one_hot_new_dataset_normalizer"
    output_dir = "./eval_output/debug_correct_one_hot_scratch_low_level"
    output_dir = "./eval_output/debug_correct_one_hot_scratch_high_level_more_epoch"
    output_dir = "./eval_output/three-piece-assembly_one_hot_500_steps"
    output_dir = "./eval_output/three-piece-assembly_500_steps_scrath_high_level"
    output_dir = "./eval_output/three-piece-assembly_500_steps_scrath_low_level"
    output_dir = "./eval_output/threading_d2"
    kwargs = get_articubot_pcd_runner_default_kwargs(
        output_dir=output_dir,
        # dataset_path="../datasets/core/square_d2.hdf5",  # set to your dataset
        # dataset_path="../datasets/core/three_piece_assembly_d2.hdf5",  # set to your dataset
        dataset_path="../datasets/core/threading_d2.hdf5",  # set to your dataset
    )
    env_runner = ArticubotPCDRunner(**kwargs)

    kwargs['high_level_path'] = high_level_path
    kwargs['low_level_exp_dir'] = low_level_exp_dir
    kwargs['low_level_checkpoint'] = low_level_checkpoint

    import json
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(kwargs, f, indent=4)

    ### run the evaluation loop
    runner_log = env_runner.run(high_level_policy, low_level_policy)
                
    cprint(f"---------------- Eval Results --------------", 'magenta')
    for key, value in runner_log.items():
        if isinstance(value, float):
            cprint(f"{key}: {value:.4f}", 'magenta')

    with open(os.path.join(output_dir, 'eval_results.txt'), 'w') as f:
        for key, value in runner_log.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")


