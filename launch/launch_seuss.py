import os
import json
import time
import click
import socket
from chester.run_exp import run_experiment_lite, VariantGenerator

def vv_to_params(vv):
    
    params = "{} {} {} {} {} {} {} {}".format(vv['env'], vv['vlm_label'], vv['vlm'], vv['flip_vlm_label'], vv['teacher_eps_mistake'], 
                                           vv['seed'], vv['reward'], vv['exp_name'])

    params += " {}".format(vv['cuda_id'])
    params += " {}".format(vv['reward_batch'])
    params += " {}".format(vv['segment'])
    params += " {}".format(vv['image_reward'])
        
    print("running params: ", params)
    return params

@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):\
    

    # exp_prefix = "2024-0322-icml-rebuttal-gemini-score-more-seeds"
    # exp_prefix = "2024-0322-icml-rebuttal-clip-score-more-seeds"
    # exp_prefix = "2024-0322-icml-rebuttal-blip-score-more-seeds"
    # exp_prefix = "2024-0322-icml-rebuttal-more-seeds"
    # exp_prefix = "2024-0324-icml-rebuttal-gt-more-seeds"
    # exp_prefix = "2024-0325-icml-rebuttal-gt-task-reward"
    exp_prefix = "2024-0325-icml-rebuttal-gt-preference"
    exp_prefix = "2024-0326-icml-rebuttal-sparse-task-reward"
    exp_prefix = "2024-0327-icml-rebuttal-door-open-blip"
    vg = VariantGenerator()

    vg.add("seed", [1, 2, 3])
    # vg.add("seed", [4, 5])
    vg.add("env", [
        # 'metaworld_soccer-v2', 
        # 'metaworld_sweep-into-v2',
        # 'metaworld_drawer-open-v2',
        'metaworld_door-open-v2',
    ])
    vg.add("vlm_label", [1])
    vg.add("vlm", [
        # "blip2_image_text_matching",
        "gemini_score",
        # "gemini_free_form",
        # "blip_image_text_matching",
    ])
    vg.add("reward", [
        # "learn_from_preference",
        # "learn_from_score",
        # "clip_image_text_matching",
        "blip2_image_text_matching",
        # "gt_task_reward"
        # "sparse_task_reward"
    ])
    vg.add("exp_name", [
        # "2024-0322-icml-rebuttal-gemini-score-more-seeds"
        # "2024-0322-icml-rebuttal-clip-score-more-seeds"
        # "2024-0322-icml-rebuttal-blip2-score-more-seeds"
        # "2024-0322-icml-rebuttal-more-seeds"
        # "2024-0324-icml-rebuttal-gt-more-seeds"
        # "2024-0325-icml-rebuttal-gt-task-reward"
        # "2024-0325-icml-rebuttal-gt-preference"
        # "2024-0326-icml-rebuttal-sparse-task-reward"
        "2024-0327-icml-rebuttal-door-open-blip"
    ])
    
    vg.add("flip_vlm_label", [0])
    vg.add("teacher_eps_mistake", [0])
    vg.add("cuda_id", [0])
    vg.add("reward_batch", [40]) 
    vg.add("segment", [1]) # cartpole, sweep-into, and all image-based
    vg.add("image_reward", [1]) # cartpole, sweep-into, and all image-based
    
    print('Number of configurations: ', len(vg.variants()))
    print("exp_prefix: ", exp_prefix)

    variations = set(vg.variations())
    task_per_gpu = 1
    all_vvs = vg.variants()
    slurm_nums = len(all_vvs) // task_per_gpu
    if len(all_vvs) % task_per_gpu != 0:
        slurm_nums += 1

    sub_process_popens = []
    
    target_method = run_task if mode == 'seuss' else run_task_local
    for idx in range(slurm_nums):
        beg = idx * task_per_gpu
        end = min((idx+1) * task_per_gpu, len(all_vvs))
        vvs = all_vvs[beg:end]
    # for idx, vv in enumerate(vg.variants()):
        while len(sub_process_popens) >= 10:
            sub_process_popens = [x for x in sub_process_popens if x.poll() is None]
            time.sleep(10)
        compile_script = wait_compile = None
        env_var = None
        
        cur_popen = run_experiment_lite(
            stub_method_call=target_method,
            variants=vvs,
            # variant=vv,
            mode=mode,
            dry=dry,
            use_gpu=True,
            exp_prefix=exp_prefix,
            wait_subprocess=debug,
            compile_script=compile_script,
            wait_compile=wait_compile,
            env=env_var,
            variations=variations,
            task_per_gpu=task_per_gpu
        )
        if cur_popen is not None:
            sub_process_popens.append(cur_popen)
        if debug:
            break

        time.sleep(10)

def run_task(vv, log_dir=None, exp_name=None):
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'variant.json'), 'w') as f:
        json.dump(vv, f, indent=2, sort_keys=True)
    
    params = vv_to_params(vv)
    command = "singularity exec --bind /data/yufeiw2/RoboGen_sim2real:/mnt/RoboGen_sim2real/ --nv /data/yufeiw2/vlm-reward.sif /mnt/RoboGen_sim2real/launch/run_in_singularity.sh {}".format(params)
    os.system(command)
    
def run_task_local(vv, log_dir=None, exp_name=None):
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'variant.json'), 'w') as f:
        json.dump(vv, f, indent=2, sort_keys=True)
    
    # rel_path = os.path.relpath(log_dir, os.getcwd())
    params = vv_to_params(vv)
    
    command = "singularity exec --bind ./:/mnt/RoboGen_sim2real/ --nv /media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/singularity_images/vlm-reward/vlm-reward.sif /mnt/RoboGen_sim2real/launch/run_in_singularity.sh {}".format(params)
    # command = "singularity exec --bind ./:/mnt/RoboGen_sim2real/ --nv /data/yufeiw2/vlm-reward.sif /mnt/RoboGen_sim2real/launch/run_in_singularity.sh {}".format(params)
    print(command)
    os.system(command)

if __name__ == '__main__':
    # first rscyn code
    
    main()


