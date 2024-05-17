from multiprocessing import Pool
from manipulation.robogen_wrapper import RobogenPointCloudWrapper
from manipulation.sim import SimpleEnv
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.common.pytorch_util import dict_apply
from manipulation.utils import build_up_env

def reset_func(args):
    env = args
    return env.reset()

def step_func(args):
    env, action = args
    return env.step(action)

def render_func(env):
    return env.render()

class RobogenParallelWrapper:
    def __init__(self, list_of_sim_env_kwargs, list_of_pc_env_kwargs, list_of_multistep_env_kwargs, num_process=2):
        self.pool = Pool(num_process)
        
        self.envs = []
        for sim_env_kwargs, pc_env_kwargs, ms_env_kwargs in zip(list_of_sim_env_kwargs, list_of_pc_env_kwargs, list_of_multistep_env_kwargs):
            env, _ = build_up_env(**sim_env_kwargs)
            env.reset()
            self.envs.append(env)
            # env.reset()
            # pc_env = RobogenPointCloudWrapper(env, **pc_env_kwargs)
            # multistep_wrapper = MultiStepWrapper(pc_env, **ms_env_kwargs)
            # self.envs.append(multistep_wrapper)
            
    def wrap_obs(self, list_of_obs):
        parallel_input_dict = {}
        parallel_input_dict['point_cloud'] = np.concatenate([x['point_cloud'][None, ...] for x in list_of_obs], axis=0)
        parallel_input_dict['agent_pos'] = np.concatenate([x['agent_pos'][None, ...] for x in list_of_obs], axis=0)
        parallel_input_dict = dict_apply(parallel_input_dict, lambda x: torch.from_numpy(x).to('cuda'))
        return parallel_input_dict

    def reset(self):
        results = self.pool.map(reset_func, self.envs)
        return wrap_obs(results)

    def step(self, actions):
        results = self.pool.map(step_func, [(env, action) for env, action in zip(self.envs, actions)])
        obses = [x[0] for x in results]
        rewards = [x[1] for x in results]
        dones = [x[2] for x in results]
        infos = [x[3] for x in results]

        return wrap_obs(obses), rewards, dones, infos

    def render(self):
        results = self.pool.map(render_func, [env.env for env in self.envs])
        return results

    def close(self):
        for env in self.envs:
            env.env._env.close()