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
from manipulation.utils import save_env as robogen_save_env
from pathlib import Path

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
        cfg.task_config_path,
        cfg.solution_path,
        cfg.substep,
        cfg.final_state_path,
        render=False,
        randomize=False,
        obj_id=0, 
    )
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env

class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
        log_dir = os.path.join(self.work_dir, 'pytorch_sac/exp', cfg.experiment, time_string)
        self.logger = Logger(log_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = make_env(cfg)

        cfg.agent.params.obs_dim = int(self.env.observation_space.shape[0])
        cfg.agent.params.action_dim = int(self.env.action_space.shape[0])
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

        self.time_limit = cfg.time_limit
        if cfg.rl_save_path is not None:
            self.save_checkpoint_path = os.path.join(cfg.rl_save_path, 'checkpoints')
            self.save_states_path = os.path.join(cfg.rl_save_path, 'states')
        if not os.path.exists(self.save_checkpoint_path):
            os.makedirs(self.save_checkpoint_path)
        if not os.path.exists(self.save_states_path):
            os.makedirs(self.save_states_path)

        self.highest_reward = None

    def evaluate(self):
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, _ = self.env.step(action)
                self.video_recorder.record(self.env)
                episode_reward += reward

            average_episode_reward += episode_reward
            self.video_recorder.save(f'{self.step}.mp4')
        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.dump(self.step)

        if (self.highest_reward is None or average_episode_reward > self.highest_reward) and self.cfg.rl_save_path is not None:
            self.highest_reward = average_episode_reward
            self.save_snapshot()
            print(f'New best model saved with reward {average_episode_reward}')

    def save_snapshot(self):
        ckpt_path = Path(self.save_checkpoint_path) / "pytorch_sac.pt"
        keys_to_save = ['agent', ]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with ckpt_path.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        ckpt_path = Path(self.save_checkpoint_path) / "pytorch_sac.pt"
        with ckpt_path.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v

    def run(self):
        episode, episode_reward, done = 0, 0, True
        begin_time = time.time()
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > self.cfg.num_seed_steps and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()
                    if (self.time_limit is not None) and (time.time() - begin_time > self.time_limit):
                        self.save_states_video()
                        print("Time limit reached")
                        return
                        

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, reward, done, _ = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
        
    def save_states_video(self):
        self.load_snapshot()
        it = 0
        obs = self.env.reset()
        self.agent.reset()
        done = False
        self.video_recorder.init(enabled=True)
        episode_reward = 0
        while not done:
            robogen_save_env(self.env, os.path.join(self.save_states_path, f'state_{it}.pkl'))
            with utils.eval_mode(self.agent):
                action = self.agent.act(obs, sample=False)
            obs, reward, done, _ = self.env.step(action)
            episode_reward += reward
            self.video_recorder.record(self.env)
            it += 1
        self.video_recorder.save(f'{self.save_checkpoint_path}/best_sac.mp4', path_from_work_dir=False)
        print("save video to", f'{self.save_checkpoint_path}/best_sac.mp4')

        # save reward to a file
        with open(os.path.join(self.save_checkpoint_path, "best_sac_score.txt"), "w") as f:
            f.write(str(episode_reward))


@hydra.main(config_path='config/train.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
