import numpy as np
import ray
import torch

from alg_parameters import *
from mapf_gym import MAPFEnv
from model import Model
from util import one_step, update_perf,global_reset_env,local_reset_env ,set_global_seeds


@ray.remote(num_cpus=1, num_gpus=SetupParameters.NUM_GPU / (TrainingParameters.N_ENVS + 1))
class Runner(object):
    """sub-process used to collect experience"""

    def __init__(self, env_id):
        """initialize model0 and environment"""
        self.ID = env_id
        set_global_seeds(env_id*123)
        self.local_num_agent = EnvParameters.LOCAL_N_AGENTS
        self.one_episode_perf = {'num_step': 0, 'reward': 0, 'invalid': 0, 'block': 0, 'num_leave_goal': 0,
                                 'wrong_blocking': 0, 'num_dynamic_collide': 0,"num_agent_collide":0}
        self.global_perf={"success_time":0,"num_iteration":[],
                          "task_global_collision":[],"destroy_weight_target":[],
                          "destroy_weight_collision":[],"destroy_weight_random":[],"makespan":[]}  # makespan may wrong
        self.num_iteration=0

        self.env = MAPFEnv(env_id)

        self.local_device = torch.device('cuda') if SetupParameters.USE_GPU_LOCAL else torch.device('cpu')
        self.local_model = Model(env_id, self.local_device)
        self.hidden_state = (
            torch.zeros((self.local_num_agent, NetParameters.NET_SIZE)).to(self.local_device),
            torch.zeros((self.local_num_agent, NetParameters.NET_SIZE)).to(self.local_device))

        self.num_global_collision = global_reset_env(self.env)
        self.global_done,self.destroy_weights,_,self.done, self.valid_actions, self.obs, \
        self.vector, self.train_valid, makespan = local_reset_env(self.env,EnvParameters.LOCAL_N_AGENTS,local_succ=False,first_time=True,ALNS=True)
        # assert (self.num_global_collision==global_num_collison)

    def run(self, weights,clear_flag):
        """run multiple steps and collect data for reinforcement learning"""
        with torch.no_grad():
            mb_obs, mb_vector, mb_rewards, mb_values, mb_done, mb_ps, mb_actions = [], [], [], [], [], [], []
            mb_hidden_state = []
            mb_train_valid, mb_blocking = [], []
            performance_dict = {'per_r': [],  'per_valid_rate': [],
                                'per_episode_len': [], 'per_block': [],
                                'per_leave_goal': [], 'per_final_goals': [], 'per_half_goals': [], 'per_block_acc': [],
                                'per_max_goals': [], 'per_num_dynamic_collide': [],'per_num_agent_collide': [], "episode_global_collision": []}
            if clear_flag:
                self.global_perf = {"success_time": 0, "num_iteration": [],
                                    "task_global_collision": [], "destroy_weight_target": [],
                                    "destroy_weight_collision": [], "destroy_weight_random": [],"makespan":[]}

            self.local_model.set_weights(weights)
            for _ in range(TrainingParameters.N_STEPS):
                mb_obs.append(self.obs)
                mb_vector.append(self.vector)
                mb_train_valid.append(self.train_valid)
                mb_hidden_state.append(
                    [self.hidden_state[0].cpu().detach().numpy(), self.hidden_state[1].cpu().detach().numpy()])
                mb_done.append(self.done)

                actions, ps, values, pre_block, output_state, num_invalid = \
                    self.local_model.step(self.obs, self.vector, self.valid_actions, self.hidden_state)
                self.one_episode_perf['invalid'] += num_invalid
                mb_values.append(values)
                mb_ps.append(ps)

                rewards, self.valid_actions, self.obs, self.vector, self.train_valid, self.done, blockings, \
                    num_on_goals, self.one_episode_perf, max_on_goals \
                    = one_step(self.env, self.one_episode_perf, actions, pre_block)

                mb_actions.append(actions)
                mb_rewards.append(rewards)
                mb_blocking.append(blockings)

                self.one_episode_perf['reward'] += np.sum(rewards)
                if self.one_episode_perf['num_step'] == EnvParameters.EPISODE_LEN // 2:
                    performance_dict['per_half_goals'].append(num_on_goals)

                if self.done:
                    self.num_iteration+=1
                    local_success=self.one_episode_perf['num_step']<EnvParameters.EPISODE_LEN

                    self.global_done, self.destroy_weights, self.global_num_collison, self.done, self.valid_actions, self.obs, \
                    self.vector, self.train_valid, makespan = local_reset_env(self.env, EnvParameters.LOCAL_N_AGENTS, local_success,
                                                                    first_time=False, ALNS=True)

                    performance_dict["episode_global_collision"].append(self.global_num_collison)

                    if self.global_done or self.num_iteration>TrainingParameters.ITERATION_LIMIT:
                        if self.global_done:
                            self.global_perf["success_time"]+=1
                        self.global_perf["task_global_collision"].append(self.global_num_collison)
                        self.global_perf["destroy_weight_target"].append(self.destroy_weights[0])
                        self.global_perf["destroy_weight_collision"].append(self.destroy_weights[1])
                        self.global_perf["destroy_weight_random"].append(self.destroy_weights[2])
                        self.global_perf["makespan"].append(makespan)

                        self.num_global_collision = global_reset_env(self.env)
                        self.global_done, self.destroy_weights, _, self.done, self.valid_actions, self.obs, \
                        self.vector, self.train_valid,makespan = local_reset_env(self.env, EnvParameters.LOCAL_N_AGENTS,
                                                                                 local_succ=False,first_time=True,ALNS=True)
                        # assert (self.num_global_collision == global_num_collison)
                        self.global_perf["num_iteration"].append(self.num_iteration)
                        self.num_iteration=0

                    performance_dict = update_perf(self.one_episode_perf, performance_dict, num_on_goals,
                                                   max_on_goals)
                    self.one_episode_perf = {'num_step': 0, 'reward': 0, 'invalid': 0, 'block': 0, 'num_leave_goal': 0,
                                             'wrong_blocking': 0, 'num_dynamic_collide': 0,"num_agent_collide":0}

                    self.done = True

                    self.hidden_state = (
                        torch.zeros((self.local_num_agent, NetParameters.NET_SIZE)).to(self.local_device),
                        torch.zeros((self.local_num_agent, NetParameters.NET_SIZE)).to(self.local_device))

            mb_obs = np.concatenate(mb_obs, axis=0)
            mb_vector = np.concatenate(mb_vector, axis=0)
            mb_rewards = np.concatenate(mb_rewards, axis=0)
            mb_values = np.squeeze(np.concatenate(mb_values, axis=0), axis=-1)
            mb_actions = np.asarray(mb_actions, dtype=np.int64)
            mb_ps = np.stack(mb_ps)
            mb_done = np.asarray(mb_done, dtype=np.bool_)
            mb_hidden_state = np.stack(mb_hidden_state)
            mb_train_valid = np.stack(mb_train_valid)
            mb_blocking = np.concatenate(mb_blocking, axis=0)

            last_values = np.squeeze(
                self.local_model.value(self.obs, self.vector, self.hidden_state))

            # calculate advantages
            mb_advs = np.zeros_like(mb_rewards)
            last_gaelam= 0
            for t in reversed(range(TrainingParameters.N_STEPS)):
                if t == TrainingParameters.N_STEPS - 1:
                    next_nonterminal = 1.0 - self.done
                    next_values = last_values
                else:
                    next_nonterminal = 1.0 - mb_done[t + 1]
                    next_values = mb_values[t + 1]

                delta = np.subtract(np.add(mb_rewards[t], TrainingParameters.GAMMA * next_nonterminal *
                                               next_values), mb_values[t])

                mb_advs[t] = last_gaelam = np.add(delta,
                                                        TrainingParameters.GAMMA * TrainingParameters.LAM
                                                        * next_nonterminal * last_gaelam)

            mb_returns = np.add(mb_advs, mb_values)

        return mb_obs, mb_vector, mb_returns, mb_values, mb_actions, mb_ps, mb_hidden_state, mb_train_valid, mb_blocking, \
            len(performance_dict['per_r']), performance_dict,self.global_perf



