import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import math
import dill
import pickle
from pathlib import Path
import wandb.sdk.data_types.video as wv
from equi_diffpo.gym_util.async_vector_env import AsyncVectorEnv
from equi_diffpo.gym_util.sync_vector_env import SyncVectorEnv
from equi_diffpo.gym_util.multistep_wrapper import MultiStepWrapper
from equi_diffpo.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
# from equi_diffpo.model.common.rotation_transformer import RotationTransformer

from equi_diffpo.policy.base_image_policy import BaseImagePolicy
from equi_diffpo.common.pytorch_util import dict_apply
from equi_diffpo.env_runner.base_image_runner import BaseImageRunner
from equi_diffpo.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from termcolor import cprint
# from non_rigid.utils.script_utils import create_model, create_datamodule
from eval_smith_utils import infer_multitask_high_level_model, low_level_policy_infer


def create_env(env_meta, shape_meta, enable_render=True, is_eval=False):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=True, 
        render_offscreen=False,
        use_image_obs=enable_render,
        # is_eval=is_eval
    )
    return env


class ArticubotPCDRunner(BaseImageRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(self, 
            output_dir,
            dataset_path,
            shape_meta:dict,
            n_train=10,
            n_test=22,
            n_train_vis=3,
            train_start_idx=0,
            test_start_idx=950,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            render_obs_key='agentview_image',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            cat_idx=0,
        ):
        super().__init__(output_dir)
        self.count = 0
        if n_envs is None:
            n_envs = n_train + n_test

        ### TODO: load the siglip embedding
        self.siglip_text_features = torch.load("/data/robogen/smith_mimicgen/siglip_text_features_w_pick_and_place_w_grasp_and_lift.pt")
        self.siglip_text_features = self.siglip_text_features['values']
        self.cat_embedding = self.siglip_text_features[cat_idx].float().to("cuda")

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(
            dataset_path)
        # disable object state observation
        env_meta['env_kwargs']['use_object_obs'] = False

        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
        # rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        def env_fn():
            robomimic_env = create_env(
                env_meta=env_meta, 
                shape_meta=shape_meta,
                is_eval=True
            )
            # Robosuite's hard reset causes excessive memory consumption.
            # Disabled to run more envs.
            # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
            robomimic_env.env.hard_reset = False
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )
        
        # For each process the OpenGL context can only be initialized once
        # Since AsyncVectorEnv uses fork to create worker process,
        # a separate env_fn that does not create OpenGL context (enable_render=False)
        # is needed to initialize spaces.
        def dummy_env_fn():
            robomimic_env = create_env(
                    env_meta=env_meta, 
                    shape_meta=shape_meta,
                    enable_render=False,
                    is_eval=True
                )
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )


        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # Get controller output_max from a temporary env (AsyncVectorEnv has no get_attr)
        _dummy = dummy_env_fn()
        inner = _dummy.env.env.env.env  # robosuite env: MultiStep->VideoRec->RobomimicImage->EnvRobosuite->robosuite
        controller = inner.robots[0].controller
        self._max_dpos = float(controller.output_max[0])
        self._max_drot = float(controller.output_max[3])
        _dummy.close()
        del _dummy

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, 
                enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # switch to seed reset
                assert isinstance(env.env.env, RobomimicImageWrapper)
                env.env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)
        # env = SyncVectorEnv(env_fns)
        

        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        # self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec
        self.max_rewards = {}
        self.cat_idx = cat_idx
        for prefix in self.env_prefixs:
            self.max_rewards[prefix] = 0

        self.episode_num = 0

    def run(self, high_level_policy, low_level_policy):
        device = low_level_policy.device
        dtype = low_level_policy.dtype
        env = self.env
        
        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()

            past_action = None
            # low_level_policy.reset()
            # high_level_policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Image {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            
            done = False
            step_num = 0
            output_list = []

            ### TODO: check what is obs
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                # if self.past_action and (past_action is not None):
                #     # TODO: not tested
                #     np_obs_dict['past_action'] = past_action[
                #         :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))
                

                # get tax3d-predicted goal gripper pcd
                B, N, _, _ = obs_dict['point_cloud'].shape
                with torch.no_grad():
                    pc_last = obs_dict['point_cloud'][:, -1, :, :]
                    gp_last = obs_dict['gripper_pcd'][:, -1, :, :]
                    inputs = torch.cat([pc_last, gp_last], dim=1)
                    # import pdb; pdb.set_trace()
                    subgoal_pred = infer_multitask_high_level_model(
                        inputs, high_level_policy,
                        cat_embedding=self.cat_embedding, ### TODO: add cat_embedding
                        high_level_args=None,
                        extra=None,
                    )
                    subgoal_pred = subgoal_pred.unsqueeze(1)

                subgoal_pred = subgoal_pred.repeat(1, self.n_obs_steps, 1, 1)
                obs_dict['goal_gripper_pcd'] = subgoal_pred

                
                # INSERT_YOUR_CODE
                if False:
                    import matplotlib.pyplot as plt
                    from mpl_toolkits.mplot3d import Axes3D

                    # Visualize input and predicted PCDs in one axis
                    fig = plt.figure(figsize=(7, 6))
                    ax = fig.add_subplot(111, projection='3d')

                    pcl_np = pc_last[0].detach().cpu().numpy()           # (N, 3)
                    gp_np = gp_last[0].detach().cpu().numpy()            # (4, 3)
                    subgoal_pcd_np = subgoal_pred[0,0].detach().cpu().numpy()  # (4, 3)

                    ax.scatter(pcl_np[:, 0], pcl_np[:, 1], pcl_np[:, 2], c='b', s=1, label='Scene Point Cloud')
                    ax.scatter(gp_np[:, 0], gp_np[:, 1], gp_np[:, 2], c='g', s=40, label='Gripper PCD')
                    ax.scatter(subgoal_pcd_np[:, 0], subgoal_pcd_np[:, 1], subgoal_pcd_np[:, 2], c='r', s=80, label='Predicted Subgoal PCD')


                    data_path = "/data/robogen/smith_mimicgen/datasets/articubot_format/square_d2_correct/demo_1/10.npz"
                    data = np.load(data_path, allow_pickle=True)
                    pointcloud = data['point_cloud'][:][0].astype(np.float32)
                    gripper_pcd = data['gripper_pcd'][:][0].astype(np.float32)
                    goal_gripper_pcd = data['goal_gripper_pcd'][:][0].astype(np.float32)

                    obj_pcd_np = data['point_cloud'].reshape(-1, 3)
                    gripper_pcd_np = data['gripper_pcd'].reshape(-1, 3)
                    goal_gripper_pcd_np = data['goal_gripper_pcd'].reshape(-1, 3)

                    ax.scatter(obj_pcd_np[:, 0], obj_pcd_np[:, 1], obj_pcd_np[:, 2], c='y', s=1, label='Original Scene Point Cloud')
                    ax.scatter(gripper_pcd_np[:, 0], gripper_pcd_np[:, 1], gripper_pcd_np[:, 2], c='grey', s=40, label='Original Gripper PCD')
                    ax.scatter(goal_gripper_pcd_np[:, 0], goal_gripper_pcd_np[:, 1], goal_gripper_pcd_np[:, 2], c='orange', s=80, label='Original Goal Gripper PCD')


                    ax.set_title('Input and Predicted PCDs')
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    ax.legend()
                    ax.view_init(elev=30, azim=45)
                    plt.tight_layout()
                    # plt.show()

                # run policy
                with torch.no_grad():
                    ### TODO: fix this
                    # import pdb; pdb.set_trace()
                    action = low_level_policy_infer(
                        obs_dict['point_cloud'],
                        obs_dict['state'],
                        obs_dict['goal_gripper_pcd'],
                        obs_dict['gripper_pcd'],
                        low_level_policy,
                        cat_idx=self.cat_idx,
                    )
                    action_dict = {'action': action}
                  
                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']

                # INSERT_YOUR_CODE
                from eval_smith_utils import policy_action_batch_to_env_action

                # Get current end-effector quaternion for each batch item
                # Assumes obs_dict['robot_state'] contains EEF quaternions at a known index range or as a field
                # Here, try typical conventions: obs_dict['state'] is (B, something) with EEF pose as last 7 (pos + quat)
                eef_quat = obs_dict['robot0_eef_quat'][:, -1, :].detach().cpu().numpy()  # shape (B, 4)
                
                # Unnormalized actions (`action`) is (B, T, 10) or (B, 10)
                # max_dpos, max_drot from controller (stored at init; AsyncVectorEnv has no get_attr)
                max_dpos = self._max_dpos
                max_drot = self._max_drot

                # cprint(f"max_dpos: {max_dpos}", 'magenta')
                # cprint(f"max_drot: {max_drot}", 'magenta')
                

                # import pdb; pdb.set_trace()
                env_action = policy_action_batch_to_env_action(
                    action,
                    eef_quat,
                    max_dpos,
                    max_drot,
                )

                env_action = env_action[:, 1:1+self.n_action_steps, :] 
                # env_action[:, :, 3:6] = 0 # disable rotation
                # print(f"env_action: {env_action.shape}", 'magenta')
                
                # import matplotlib.pyplot as plt
                # from mpl_toolkits.mplot3d import Axes3D

                # fig = plt.figure(figsize=(8, 6))
                # ax = fig.add_subplot(111, projection='3d')

                # # point cloud: (N, 3)
                # pcd = obs_dict['point_cloud'][0][-1].detach().cpu().numpy()
                # ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], s=1, c='gray', label='scene point cloud')

                # # current gripper pcd: (4, 3)
                # gripper_pcd = obs_dict['gripper_pcd'][0][-1].detach().cpu().numpy()
                # ax.scatter(gripper_pcd[:, 0], gripper_pcd[:, 1], gripper_pcd[:, 2], color='blue', s=50, label='current gripper pcd')

                # # goal gripper pcd: (4, 3)
                # goal_gripper_pcd = obs_dict['goal_gripper_pcd'][0][-1].detach().cpu().numpy()
                # ax.scatter(goal_gripper_pcd[:, 0], goal_gripper_pcd[:, 1], goal_gripper_pcd[:, 2], color='orange', s=50, marker='^', label='goal gripper pcd')

                # # Show trajectory of the gripper (future EEF pos)
                # # action shape: (batch, steps, 10) or (batch, 10)
                # # Use batch = 0 always (run is per-env)
                # action_0 = action[0]

                # # Number of steps predicted for this env roll
                # num_steps = action_0.shape[0]

                # # Current EEF: use the last point in gripper_pcd as start
                # orig = gripper_pcd[-1]
                # traj = [orig]

                # for t in range(num_steps):
                #     dpos = action_0[t, :3]
                #     next_pos = traj[-1] + dpos
                #     traj.append(next_pos)

                # traj = np.stack(traj)
                # ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color='red', linewidth=2, label='future gripper trajectory')
                # ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color='red', marker='*', s=80, label='eef after step')

                # ax.legend()
                # ax.set_title('Point Cloud, Gripper PCD, Goal Gripper PCD, and EEF Trajectory')
                # ax.set_xlabel('X')
                # ax.set_ylabel('Y')
                # ax.set_zlabel('Z')
                # plt.tight_layout()
                # plt.show()


                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")
                
                # step env
                N, s, _ = env_action.shape
                if self.abs_action:
                    env_action = self.undo_transform_action(action)

                output_list.append(subgoal_pred.cpu().numpy())
                obs, reward, done, info = env.step(env_action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]

            # for additional visualizations later
            for i, video_path in enumerate(all_video_paths[this_global_slice]):
                p = Path(video_path)
                video_dir = p.parent / 'goal_predictions' / p.stem
                video_dir.mkdir(exist_ok=True, parents=True)
                for j, subgoal_pred in enumerate(output_list):
                    with open(video_dir / f'{j}.pkl', 'wb') as f:
                        pickle.dump({'subgoal_pred': subgoal_pred[i]}, f)

        # clear out video buffer
        _ = env.reset()
        
        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video
        
        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value
            self.max_rewards[prefix] = max(self.max_rewards[prefix], value)
            log_data[prefix+'max_score'] = self.max_rewards[prefix]

        return log_data

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction
