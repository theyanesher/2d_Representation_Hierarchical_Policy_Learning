import hydra

from omegaconf import DictConfig, OmegaConf
import datetime


def omegaconf_to_dict(d: DictConfig):
    """Converts an omegaconf DictConfig to a python Dict, respecting variable interpolation."""
    ret = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            ret[k] = omegaconf_to_dict(v)
        else:
            ret[k] = v
    return ret

# Hydra decorator to pass in the config. Looks for a config file in the specified path. This file in turn has links to other configs 
@hydra.main(config_path="./configs/robogen.yaml")
def launch_rlg_hydra(cfg: DictConfig):

    import logging
    import os

    import gym
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner

    # Creating a new function to return a pushT environment. This will then be added to rl_games env_configurations so that an env can be created from its name in the config
    from rl_games.common.vecenv import RayVecEnv

    # def create_pusht_env(**kwargs):
    #     from manipulation.utils import build_up_env
    #     env, safe_config = build_up_env(
    #         "data/generated_tasks_release/Microwave_7310_2024-03-04-21-20-19/Open_Microwave_Door_The_robotic_arm_will_open_the_microwave_door_to_insert_or_remove_items.yaml",
    #         "data/generated_tasks_release/Microwave_7310_2024-03-04-21-20-19/task_Open_Microwave_Door",
    #         "open_the_microwave_door", 
    #         "data/generated_tasks_release/Microwave_7310_2024-03-04-21-20-19/task_Open_Microwave_Door/experiment/2024-03-04-21-44-32/grasp_the_microwave_door_primitive/states/state_140.pkl", 
    #         render=True, 
    #         randomize=False, 
    #         obj_id=0
    #     )
    #     return env

    # # env_configurations.register adds the env to the list of rl_games envs. 
    # env_configurations.register('robogen', {
    #     'vecenv_type': 'RAY',
    #     'env_creator': lambda **kwargs: create_pusht_env(**kwargs),
    # })

    # vecenv register calls the following lambda function which then returns an instance of CUSTOMRAY. 
    # vecenv.register('RAY', lambda config_name, num_actors, **kwargs: RayVecEnv(config_name, num_actors, **kwargs))

    # Convert to a big dictionary
    rlg_config_dict = omegaconf_to_dict(cfg)

    # Build an rl_games runner. You can add other algos and builders here
    def build_runner():
        runner = Runner()
        return runner

    # create runner and set the settings
    runner = build_runner()
    runner.load(rlg_config_dict)
    runner.reset()
    
    from rl_games.common.env_configurations import configurations
    print(configurations)
    # Run either training or playing via the rl_games runner
    runner.run({
        # 'train': True,
        # 'play': False,
        'train': False,
        'play': True,
        # "checkpoint": "runs/robogen_08-18-27-18/nn/robogen.pth"
        # "checkpoint": "runs/robogen_08-17-50-08/nn/robogen.pth"
        "checkpoint": "runs/robogen_09-19-23-32/nn/robogen.pth"
    })


if __name__ == "__main__":
    launch_rlg_hydra()