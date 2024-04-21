import wandb
import numpy as np
import torch
import tqdm
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util
from termcolor import cprint
from manipulation.utils import build_up_env, save_numpy_as_gif
from manipulation.robogen_wrapper import RobogenPointCloudWrapper
import os, json

class RoboGenRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=3,
                 max_steps=200,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 task_name=None,
                 use_point_crop=True,
                 in_gripper_frame=False,
                 gripper_num_points=0,
                 add_contact=0,
                 start_after_reaching=0,
                 ):
        super().__init__(output_dir)
        self.task_name = task_name
        self.save_video_dir = os.path.join(output_dir, 'videos')
        if not os.path.exists(self.save_video_dir):
            os.makedirs(self.save_video_dir)

        steps_per_render = max(10 // fps, 1)

        self.in_gripper_frame = in_gripper_frame
        self.gripper_num_points = gripper_num_points
        self.add_contact = add_contact
        self.start_after_reaching = start_after_reaching
        
        self.eval_episodes = eval_episodes
        # self.env = env_fn()

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)
        
        after_reaching_init_state_files = []
        config_files = []
        experiment_folder = "{}/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle".format(os.environ['PROJECT_DIR'])
        experiment_path = os.path.join(experiment_folder, "experiment", "vary_robot_init_joint_near_handle")
        all_experiments = os.listdir(experiment_path)
        all_experiments = sorted(all_experiments)

        for experiment in all_experiments:
            first_step_folder = "grasp_the_door_handle_primitive"
            first_stage_states_path = os.path.join(experiment_path, experiment, first_step_folder, "states")
            stage_lengths = os.path.join(experiment_path, experiment, first_step_folder, "stage_lengths.json")
            with open(stage_lengths, "r") as f:
                stage_lengths = json.load(f)
            reaching_phase = stage_lengths['reach_handle']
            init_state_file = os.path.join(first_stage_states_path, "state_{}.pkl".format(reaching_phase))
            after_reaching_init_state_files.append(init_state_file)
            meta_info = os.path.join(experiment_path, experiment, "meta_info.json")
            with open(meta_info, "r") as f:
                meta_info = json.load(f)
            config_files.append(meta_info['config_path'])
        
        self.after_reaching_init_state_files = after_reaching_init_state_files
        self.config_files = config_files

    def build_env(self, idx):
        # TODO: change to the test configs, which should probably be passed in here. 
        
        env, _ = build_up_env(
            # "{}/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/open_the_door_of_the_storagefurniture_by_its_handle_The_robot_arm_will_open_the_door_of_the_storage_furniture_by_manipulating_its_handle_{}.yaml".format(os.environ['PROJECT_DIR'], idx),
            "{}/{}".format(os.environ['PROJECT_DIR'], self.config_files[idx]),
            "data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle",
            "grasp_the_door_handle",
            None if not self.start_after_reaching else self.after_reaching_init_state_files[idx],
            render=False, 
            randomize=False,
            obj_id=0,
            horizon=400,
        )
        env.reset()
        object_name = "StorageFurniture"
        pointcloud_env = RobogenPointCloudWrapper(env, object_name, in_gripper_frame=self.in_gripper_frame, 
                                                  gripper_num_points=self.gripper_num_points, add_contact=self.add_contact)
        env = MultiStepWrapper(pointcloud_env, n_obs_steps=self.n_obs_steps, n_action_steps=self.n_action_steps, 
                        max_episode_steps=self.max_steps, reward_agg_method='sum')
        
        return env

    def run(self, policy: BasePolicy, epoch: int):
        device = policy.device
        dtype = policy.dtype

        # all_goal_achieved = []
        all_success_rates = []
        
        
        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in RoboGen {self.task_name} Pointcloud Env",
                                     leave=False, mininterval=self.tqdm_interval_sec):
            
            env = self.build_env(episode_idx)
            frames = []
                
            # start rollout
            obs = env.reset()
            policy.reset()

            done = False
            num_goal_achieved = 0
            actual_step_count = 0
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                # run policy
                with torch.no_grad():
                    obs_dict_input = {}  # flush unused keys
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                    action_dict = policy.predict_action(obs_dict_input)
                    

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action'].squeeze(0)
                # step env
                obs, reward, done, info = env.step(action)
                # all_goal_achieved.append(info['goal_achieved']
                # num_goal_achieved += np.sum(info['goal_achieved'])
                image = env.env._env.render()
                frames.append(image)
                
                done = np.all(done)
                actual_step_count += 1
                # print("actual step count: ", actual_step_count)

            all_success_rates.append(info['opened_joint_angle'][-1])
            # all_goal_achieved.append(num_goal_achieved)
            env.env._env.close()
            del env

            # import pdb; pdb.set_trace()
            save_numpy_as_gif(np.array(frames), os.path.join(self.save_video_dir, 
                    "{}_eval_episode_{}_{:.3f}.mp4".format(epoch, episode_idx, info['opened_joint_angle'][-1])))

        # log
        log_data = dict()
        

        # log_data['mean_n_goal_achieved'] = np.mean(all_goal_achieved)
        log_data['mean_success_rates'] = np.mean(all_success_rates)

        log_data['test_mean_score'] = np.mean(all_success_rates)

        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

        # videos = env.env.get_video()
        # if len(videos.shape) == 5:
        #     videos = videos[:, 0]  # select first frame
        # videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
        # log_data[f'sim_video_eval'] = videos_wandb
        
        with open(os.path.join(self.save_video_dir, f'eval_results.txt'), 'a') as f:
            f.write(str(np.mean(all_success_rates)) + '\n')

        return log_data
