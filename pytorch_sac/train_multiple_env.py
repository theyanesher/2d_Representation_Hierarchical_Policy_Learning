#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl

from video import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
import utils, datetime
from multiprocessing import Pool

# import dmc2gym
import hydra


def make_env(cfg):
    """Helper function to create dm_control environment"""
    # if cfg.env == 'ball_in_cup_catch':
    #     domain_name = 'ball_in_cup'
    #     task_name = 'catch'
    # else:
    #     domain_name = cfg.env.split('_')[0]
    #     task_name = '_'.join(cfg.env.split('_')[1:])

    # env = dmc2gym.make(domain_name=domain_name,
    #                    task_name=task_name,
    #                    seed=cfg.seed,
    #                    visualize_reward=True)
    # env.seed(cfg.seed)
    
    from manipulation.utils import build_up_env
    env, safe_config = build_up_env(
        "data/generated_tasks_release/Microwave_7310_2024-03-04-21-20-19/Open_Microwave_Door_The_robotic_arm_will_open_the_microwave_door_to_insert_or_remove_items.yaml",
        "data/generated_tasks_release/Microwave_7310_2024-03-04-21-20-19/task_Open_Microwave_Door",
        "open_the_microwave_door", 
        "data/generated_tasks_release/Microwave_7310_2024-03-04-21-20-19/task_Open_Microwave_Door/experiment/2024-03-04-21-44-32/grasp_the_microwave_door_primitive/states/state_140.pkl", 
        render=True, 
        randomize=False, 
        obj_id=0
    )
    
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


def step_single_env(args):
    env, action = args
    return env.step(action)
    
def reset_single_env(args):
    env = args
    obs = env.reset()
    return obs

class vectorized_env(object):
    def __init__(self, num_env=16):
        self.num_env = num_env
        self.envs = []
        for _ in range(num_env):
            from manipulation.utils import build_up_env
            env, safe_config = build_up_env(
                "data/generated_tasks_release/Microwave_7310_2024-03-04-21-20-19/Open_Microwave_Door_The_robotic_arm_will_open_the_microwave_door_to_insert_or_remove_items.yaml",
                "data/generated_tasks_release/Microwave_7310_2024-03-04-21-20-19/task_Open_Microwave_Door",
                "open_the_microwave_door", 
                "data/generated_tasks_release/Microwave_7310_2024-03-04-21-20-19/task_Open_Microwave_Door/experiment/2024-03-04-21-44-32/grasp_the_microwave_door_primitive/states/state_140.pkl", 
                render=False, 
                randomize=False, 
                obj_id=0
            )
            self.envs.append(env)        

        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.pool = Pool(processes=num_env)
        self._max_episode_steps = self.envs[0]._max_episode_steps
        
        
    def step(self, actions):
        # using pool to map the step_single_env function to each env
        ret = self.pool.map(step_single_env, zip(self.envs, actions))
        obs= np.array([r[0] for r in ret])
        rewards = np.array([r[1] for r in ret])
        dones = [r[2] for r in ret]
        infos = [r[3] for r in ret]
        return obs, rewards, dones, infos
    
    def reset(self):
        ret = self.pool.map(reset_single_env, self.envs)
        # print("ret: ", ret)
        return np.array(ret)
    
    def close(self):
        for env in self.envs:
            env.close()

class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        self.num_env = self.cfg.num_env

        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
        log_dir = os.path.join(self.work_dir, 'pytorch_sac/exp', time_string, cfg.experiment)
        self.logger = Logger(log_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = vectorized_env(self.num_env)

        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          int(cfg.replay_buffer_capacity),
                                          self.device)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0

    def evaluate(self):
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            # self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, _ = self.env.step(action)
                # self.video_recorder.record(self.env)
                episode_reward += np.mean(reward)

            average_episode_reward += episode_reward
            # self.video_recorder.save(f'{self.step}.mp4')
        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.dump(self.step)

    def run(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            print(f'step: {self.step}')
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)

                print("before reset")
                obs = self.env.reset()
                print("after reset")
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = [self.env.action_space.sample() for _ in range(self.num_env)]
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, reward, done, _ = self.env.step(action)

            # allow infinite bootstrap
            # done = float(done)
            done_no_max = np.array([0 if episode_step + 1 == self.env._max_episode_steps else float(d) for d in done])
            episode_reward += np.mean(reward)

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max, len=self.num_env)

            obs = next_obs
            episode_step += 1
            self.step += self.num_env


@hydra.main(config_path='config/train.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
