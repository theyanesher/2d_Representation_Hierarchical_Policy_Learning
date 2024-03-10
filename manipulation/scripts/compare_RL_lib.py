import numpy as np
from matplotlib import pyplot as plt
import pandas

sac_csv = pandas.read_csv("pytorch_sac/exp/2024-03-05-17-16-13/test_exp/train.csv")
# get the value of "episode reward"
value = sac_csv["episode_reward"]
durations = sac_csv["duration"]
durations = np.cumsum(durations)
durations = durations / 60
plt.plot(durations, value, label="SAC")

rl_game_ppo_csv = pandas.read_csv("runs/robogen_05-17-16-37/summaries.csv")
value = rl_game_ppo_csv["Value"]
times = rl_game_ppo_csv["Step"] / 60
plt.plot(times, value, label="rl_game_ppo")

rl_game_ppo_csv_2 = pandas.read_csv("runs/robogen_05-19-29-03/summaries (1).csv")
value = rl_game_ppo_csv_2["Value"]
times = rl_game_ppo_csv_2["Step"] / 60
plt.plot(times, value, label="rl_game_ppo_full_obs")

ray_ppo = pandas.read_csv("data/generated_tasks_release/Microwave_7310_2024-03-04-21-20-19/task_Open_Microwave_Door/experiment/2024-03-05-17-58-21/open_the_microwave_door_RL_ppo/ray_logs/progress.csv")
time = ray_ppo["time_total_s"] / 60
print(time)
reward = ray_ppo["episode_reward_mean"]
plt.plot(time, reward, label="ray_ppo")

ray_sac = pandas.read_csv("data/generated_tasks_release/Microwave_7310_2024-03-04-21-20-19/task_Open_Microwave_Door/experiment/2024-03-05-18-12-31/open_the_microwave_door_RL_sac/ray_logs/progress.csv")
time = ray_sac["time_total_s"] / 60
reward = ray_sac["episode_reward_mean"]
plt.plot(time, reward, label="ray_sac")

plt.xlabel("Time (minutes)")
plt.ylabel("Episode Reward")
plt.legend()
plt.show()