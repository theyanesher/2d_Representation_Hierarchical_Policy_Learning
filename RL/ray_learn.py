import os, sys, ray, shutil, glob
import numpy as np
from ray.rllib.agents import ppo, sac
from ray import tune
from manipulation.utils import save_env, save_numpy_as_gif
import pickle
import datetime
from ray.tune.logger import UnifiedLogger
import time

def custom_log_creator(log_dir):
   
    def logger_creator(config):

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return UnifiedLogger(config, log_dir, loggers=None)

    return logger_creator

def setup_config(algo, seed=0, env_config={}, eval=False, num_workers=8):
    if algo == 'ppo':
        config = ppo.DEFAULT_CONFIG.copy()
        config["rollout_fragment_length"] = 100
        config["train_batch_size"] = 6400
    elif algo == 'sac':
        config = sac.DEFAULT_CONFIG.copy()
        config['timesteps_per_iteration'] = 100
        config['learning_starts'] = 2000
        config['Q_model']['fcnet_hiddens'] = [256, 256, 256]
        config['policy_model']['fcnet_hiddens'] = [256, 256, 256]
        config['train_batch_size'] = 1024
        config['num_gpus'] = 1

    config['framework'] = 'torch'
    if not eval:
        config['num_workers'] = num_workers
    else:
        config['num_workers'] = 1
    config['seed'] = seed
    config['log_level'] = 'ERROR'
    config["env_config"] = env_config
    return config

def load_policy(algo, env_name, policy_path=None, seed=0, env_config={}, eval=False, num_workers=8, ray_save_path=None):
    ts = time.time()
    time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
    logdir_prefix = "{}_{}".format(env_name, time_string)
    log_dir = os.path.join("data/local/ray_results", logdir_prefix)
    if ray_save_path is not None:
        log_dir = os.path.join(ray_save_path, "ray_logs")
        
    Trainer = ppo.PPOTrainer if algo == 'ppo' else sac.SACTrainer
    agent = Trainer(setup_config(algo, seed, env_config, eval=eval, num_workers=num_workers), 
                            env_name,
                            logger_creator=custom_log_creator(log_dir)
    )
    if policy_path is not None:
        if 'checkpoint' in policy_path:
            agent.restore(policy_path)
        else:
            # Find the most recent policy in the directory
            directory = os.path.join(policy_path, algo, env_name)
            files = [f.split('_')[-1] for f in glob.glob(os.path.join(directory, 'checkpoint_*'))]
            files_ints = [int(f) for f in files]
            if files:
                checkpoint_max = max(files_ints)
                checkpoint_num = files_ints.index(checkpoint_max)
                checkpoint_path = os.path.join(directory, 'checkpoint_%s' % files[checkpoint_num], 'checkpoint-%d' % checkpoint_max)
                agent.restore(checkpoint_path)
            return agent, None
    return agent, None

def train(env_name, algo, timesteps_total=2000000, save_dir='./trained_models/', load_policy_path='', seed=0, 
          env_config={}, eval_interval=200000, render=False, num_cpus=64, save_every_eval=False):

    if not ray.is_initialized():
        ray.init(num_cpus=num_cpus, ignore_reinit_error=True, log_to_driver=False)
    agent, checkpoint_path = load_policy(algo, env_name, load_policy_path, env_config=env_config, 
                                         seed=seed, num_workers=num_cpus, ray_save_path=save_dir)

    env = make_env(env_config, render=render)

    best_model_save_path = os.path.join(save_dir, 'best_model')
    best_state_save_path = os.path.join(save_dir, 'best_state')
    if not os.path.exists(best_state_save_path):
        os.makedirs(best_state_save_path)

    timesteps = 0
    eval_time = 1
    best_ret = -np.inf
    best_rgbs = None
    best_state_files = None
    while timesteps < timesteps_total:
        result = agent.train()
        timesteps = result['timesteps_total']
        print(f"Iteration: {result['training_iteration']}, total timesteps: {result['timesteps_total']}, total time: {result['time_total_s']:.1f}, FPS: {result['timesteps_total']/result['time_total_s']:.1f}, mean reward: {result['episode_reward_mean']:.1f}, min/max reward: {result['episode_reward_min']:.1f}/{result['episode_reward_max']:.1f}")
        sys.stdout.flush()

        # Delete the old saved policy
        if checkpoint_path is not None:
            shutil.rmtree(os.path.dirname(checkpoint_path), ignore_errors=True)
        # Save the recently trained policy
        checkpoint_path = agent.save(save_dir)

        if timesteps > eval_time * eval_interval:
            obs = env.reset()
            done = False
            ret = 0
            rgbs = []
            states = []
            t_idx = 0
            
            if save_every_eval:
                state_save_path = os.path.join(save_dir, "eval_{}".format(eval_time))
                if not os.path.exists(state_save_path):
                    os.makedirs(state_save_path)
            while not done:
                # Compute the next action using the trained policy
                action = agent.compute_action(obs, explore=False)
                # Step the simulation forward using the action from our trained policy
                obs, reward, done, info = env.step(action)
                ret += reward
                rgb = env.render()
                rgbs.append(rgb)
                
                if save_every_eval:
                    state_file_path = os.path.join(state_save_path, "state_{}.pkl".format(t_idx))
                    state = save_env(env, save_path=state_file_path)
                else:
                    state = save_env(env)
                    
                states.append(state)
                t_idx += 1
                
            if save_every_eval:
                save_numpy_as_gif(np.array(rgbs), "{}/{}.gif".format(state_save_path, "execute"))

            print("evaluating at {} return is {}".format(timesteps, ret))
            eval_time += 1
            if ret > best_ret:
                best_ret = ret
                best_model_path = agent.save(best_model_save_path)
                best_rgbs = rgbs
                best_state_files = []
                for idx, state in enumerate(states):
                    sp = os.path.join(best_state_save_path, "state_{}.pkl".format(idx))
                    with open(sp, 'wb') as f:
                        pickle.dump(state, f, pickle.HIGHEST_PROTOCOL)
                    best_state_files.append(sp)
                save_numpy_as_gif(np.array(best_rgbs), "{}/{}-{}.gif".format(best_state_save_path, "best", round(best_ret, 3)))
                
    env.disconnect()
    return best_model_path, best_rgbs, best_state_files

def render_policy(env, env_name, algo, policy_path, seed=0, n_episodes=1, env_config={}):
    ray.init(num_cpus=1, ignore_reinit_error=True, log_to_driver=False)
    if env is None:
        env = make_env(env_name)
    test_agent, _ = load_policy(algo, env_name, policy_path, seed, env_config, eval=True)

    env.render()
    frames = []
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            # Compute the next action using the trained policy
            action = test_agent.compute_action(obs)
            # Step the simulation forward using the action from our trained policy
            obs, reward, done, info = env.step(action)
            env.render()
    env.disconnect()

def make_env(config, render=False):

    import yaml
    from manipulation.utils import build_up_env

    print(config)
    task_config_path = config['task_config_path']

    task_name = config['task_name']
    last_restore_state_file = config['last_restore_state_file']

    solution_path = config['solution_path']
    action_space = config['action_space']

    env, safe_config = build_up_env(
            task_config_path, 
            solution_path, 
            task_name, 
            last_restore_state_file, 
            render=render, 
            action_space=action_space, 
            randomize=config['randomize'], 
            obj_id=config['obj_id'],
        )

    return env

def run_RL(task_config_path, solution_path, task_name, last_restore_state_file, save_path, 
           action_space='delta-translation', algo="sac", timesteps_total=1000000, load_policy_path=None, seed=0, 
           render=False, randomize=False, use_bard=True, obj_id=0, 
           use_gpt_size=True, use_gpt_joint_angle=True, use_gpt_spatial_relationship=True,
           use_distractor=False):
    env_name = task_name

    env_config = {
        "task_config_path": task_config_path,
        "solution_path": solution_path,
        "task_name": task_name,
        "last_restore_state_file": last_restore_state_file,
        "action_space": action_space,
        "randomize": randomize,
        "use_bard": use_bard,
        "obj_id": obj_id,
        "use_gpt_size": use_gpt_size,
        "use_gpt_joint_angle": use_gpt_joint_angle,
        "use_gpt_spatial_relationship": use_gpt_spatial_relationship,
        "use_distractor": use_distractor
    }
    
    timesteps_total = 1000000 
    eval_interval = 20000 
    
    tune.register_env(env_name, lambda config: make_env(config))
    best_policy_path, rgbs, best_traj_state_paths = train(env_name, algo, timesteps_total=timesteps_total, 
                            load_policy_path=load_policy_path, save_dir=save_path, seed=seed, env_config=env_config, render=render,
                            eval_interval=eval_interval)
    
    return best_policy_path, rgbs, best_traj_state_paths