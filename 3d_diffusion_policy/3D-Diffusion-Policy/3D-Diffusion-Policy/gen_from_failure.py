import os
import argparse
import yaml
import random
from tqdm import tqdm
import time
import datetime
import multiprocessing as mp
import numpy as np
from manipulation.utils import build_up_env, load_env, save_numpy_as_gif
import pybullet as p

def check_failure(
    exp_path: str,
    env_name: str = 'articulated'
):  
    config_file = os.path.join(exp_path, 'task_config.yaml')
    config = yaml.safe_load(open(config_file, "r"))
    link_name = 'link_0'
    for config_dict in config:
        if 'name' in config_dict:
            object_name = config_dict['name'].lower()
        if 'link_name' in config_dict:
            link_name = config_dict['link_name']
    states_path = os.path.join(exp_path, 'states')
    all_states = [f for f in os.listdir(states_path) if f.startswith('state_') and f.endswith('.pkl')]
    if len(all_states) == 0:
        return -1
    all_grasped_handles = [False for _ in range(4)]
    current_grasped_handle = False
    last_joint_angle = None
    # with tqdm(total=len(all_states)) as pbar:
    for state_idx in range(len(all_states)):
        state_path = os.path.join(states_path, f'state_{state_idx}.pkl')
        if not os.path.exists(state_path):
            raise ValueError(f"State file {state_path} does not exist.")
        env, _ = build_up_env( 
            task_config=config_file,
            env_name=env_name,
            restore_state_file=state_path,
            # render=False, 
            render=False, 
            randomize=False,
        )
        env.reset()
        info = env._get_info(object_name=object_name, link_name=link_name, handle_name=env.handle_name)
        env.close()
        joint_angle = info['opened_joint_angle']
        if last_joint_angle is not None:
            grasped_handle = abs(joint_angle - last_joint_angle) > 1e-6
        else:
            grasped_handle = False
        last_joint_angle = joint_angle
        all_grasped_handles.append(grasped_handle)
        if current_grasped_handle:
            if not any(all_grasped_handles[-4:]):
                # If the last 4 states are not grasped, consider it a failure
                print(f"Failure detected at state {state_idx} in {exp_path}.")
                return state_idx
        else:
            if all(all_grasped_handles[-2:]):
                # If the last 2 states are grasped, consider the current state successful
                current_grasped_handle = True
    return -1  # If all states are successful, return -1

def get_failure_exps(
    all_exps_path: str,
):
    # all_exps = [d for d in os.listdir(all_exps_path) if os.path.isdir(os.path.join(all_exps_path, d))]
    all_exps = ['2025-05-29-15-31-26_rollout']
    failure_exps = []
    failure_idxs = []
    with tqdm(total=len(all_exps), desc="Checking experiments for failures") as pbar:
        for exp in all_exps:
            # print(f"Checking experiment {exp} for failures...")
            exp_path = os.path.join(all_exps_path, exp)
            failure_idx = check_failure(exp_path)
            if failure_idx != -1:
                failure_exps.append(exp_path)
                failure_idxs.append(failure_idx)
                # print(f"Experiment {exp} failed at state index {failure_idx}.")
            pbar.update(1)
                
    return failure_exps, failure_idxs

def select_init_states(
    all_exps,
    start_idxs,
    num_states=10,
):
    all_configs = []
    all_states = []
    for exp, start_idx in zip(all_exps, start_idxs):
        config_file = os.path.join(exp, 'task_config.yaml')
        states_path = os.path.join(exp, 'states')
        num_states = len([f for f in os.listdir(states_path) if f.startswith('state_') and f.endswith('.pkl')])
        if start_idx < 0 or start_idx >= num_states:
            raise ValueError(f"Start index {start_idx} is out of bounds for experiment {exp}.")
        # all_configs += [config_file for _ in range(num_states - start_idx)]
        # all_states += [os.path.join(states_path, f'state_{i}.pkl') for i in range(start_idx, num_states)]
        all_configs += [config_file] * num_states 
        all_states += [os.path.join(states_path, f'state_{i}.pkl') for i in range(num_states)]
    
    states_count = len(all_states)
    if states_count == 0:
        raise ValueError("No valid states found in the provided experiments.")
    if states_count < num_states:
        print(f"Only {states_count} states found, selecting all of them.")
        return all_configs, all_states
    if num_states <= 0:
        raise ValueError(f"Number of states to select must be positive, got {num_states}.")
    selected_indices = random.sample(range(states_count), num_states)
    selected_configs = [all_configs[i] for i in selected_indices]
    selected_states = [all_states[i] for i in selected_indices]
    return selected_configs, selected_states
    
def execute(
    exp_path: str,
    config_path: str,
    state_path: str,
    env_name: str = 'articulated',
):  
    config = yaml.safe_load(open(config_path, "r"))
    link_name = 'link_0'
    for config_dict in config:
        if 'name' in config_dict:
            object_name = config_dict['name'].lower()
        if 'link_name' in config_dict:
            link_name = config_dict['link_name']
    # get init joint angle from task_config.yaml
    env, _ = build_up_env(
            task_config=config_path,
            env_name='articulated',
        )
    env.reset()
    info = env._get_info(object_name=object_name, link_name=link_name, handle_name=env.handle_name)
    init_joint_angle = info['opened_joint_angle']
    load_env(env, state_path)
    info = env._get_info(object_name=object_name, link_name=link_name, handle_name=env.handle_name)
    open_joint_angle = info['opened_joint_angle']
    # print(f"Initial joint angle: {init_joint_angle}, Opened joint angle: {open_joint_angle}")
    robot_joint_angles = env.robot.get_joint_angles(indices=env.robot.right_arm_joint_indices)
    for config_dict in config:
        if 'center' in config_dict:
            if config_dict['init_angle'] is not None:
                config_dict['init_angle'] = float(config_dict['init_angle']) + open_joint_angle - init_joint_angle
            else:
                config_dict['init_angle'] = open_joint_angle - init_joint_angle
            config_dict['init_joint_angles'] = str(tuple(robot_joint_angles))
    # save the modified config back to the file
    config_save_path = os.path.join(exp_path, 'task_config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, indent=4)
    print(f"Modified config saved to {config_save_path}")
    print(f"Config: {config}")
    env.close()
    
    print("execute primitive")
    # env, _ = build_up_env(config_path, env_name, restore_state_file=state_path)
    env, _ = build_up_env(config_save_path, env_name, restore_state_file=state_path)
    env.primitive_save_path = exp_path
    np.random.seed(time.time_ns() % 2**32)

    env.reset()
   
    print("Executing environment...")

    rgbs, states = env.execute()
    env.close()
    if len(states) > 10:
        with open(os.path.join(exp_path, "last_state_files.txt"), 'w') as f:
            f.write("\n".join(str(states[-1])))
        save_numpy_as_gif(np.array(rgbs), "{}/{}.gif".format(exp_path, "all"))
        # q.put(True)
        return True
    else:
        # q.put(False)
        return False
    # return False
# def execute(
#     exp_path: str,
#     config_path: str,
#     state_path: str,
#     env_name: str = 'articulated',
# ):
#     q = mp.Queue()
#     p = mp.Process(target=_execute, args=(q, exp_path, config_path, state_path, env_name))
#     p.start()
#     p.join()
#     success = q.get()
#     return success

def gen_from_failure(
    all_exps_path: str,
    output_path: str,
    num_exps: int = 100,
):
    if not os.path.exists(all_exps_path):
        raise ValueError(f"Path {all_exps_path} does not exist.")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    if os.path.exists(os.path.join(output_path, "failure_exps.txt")):
        print("Failure experiments already exist in the output directory. Skipping failure detection.")
        with open(os.path.join(output_path, "failure_exps.txt"), 'r') as f:
            failure_exps = []
            failure_idxs = []
            for line in f:
                exp, idx = line.strip().split()
                failure_exps.append(exp)
                failure_idxs.append(int(idx))
    else:
        print("Detecting failure experiments...")
        failure_exps, failure_idxs = get_failure_exps(all_exps_path)
        if len(failure_exps) == 0:
            print("No failure experiments found.")
            return
        print(f"Found {len(failure_exps)} failure experiments.")
        # save the failure experiments and their indices
        with open(os.path.join(output_path, "failure_exps.txt"), 'w') as f:
            for exp, idx in zip(failure_exps, failure_idxs):
                f.write(f"{exp} {idx}\n")

    selected_configs, selected_states = select_init_states(failure_exps, failure_idxs, num_states=num_exps)
    for config, state in zip(selected_configs, selected_states):
        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
        exp_path = os.path.join(output_path, time_string)
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        # os.system("cp {} {}".format(config, os.path.join(exp_path, "task_config.yaml")))
        print(f"Executing experiment with config {config} and state {state}...")
        success = execute(exp_path, config, state)
        if not success:
            print(f"Execution failed for config {config} and state {state}.")
        else:
            print(f"Experiment saved to {exp_path}.")
        # break

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--category', type=str, default=None)
    parser.add_argument('--obj_id', type=str, default=None)
    args = parser.parse_args()

    if args.exp_dir is not None:
        exp_dir = args.exp_dir
        category = exp_dir.split('/')[-2]
        obj_id = exp_dir.split('/')[-1]
    else:
        category = args.category
        obj_id = args.obj_id
    
    all_exps_path = os.path.join(args.data_dir, category, obj_id)
    if not os.path.exists(all_exps_path):
        raise ValueError(f"Path {all_exps_path} does not exist.")
    output_path = os.path.join(args.output_dir, category, obj_id)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    gen_from_failure(
        all_exps_path=all_exps_path,
        output_path=output_path
    )
