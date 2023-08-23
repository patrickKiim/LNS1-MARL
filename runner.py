import numpy as np
import ray
import torch

from alg_parameters import *
from mapf_gym import MAPFEnv
from CL_mapf_gym import CL_MAPFEnv
from model import Model
from util import set_global_seeds,init_global_perf,init_one_episode_perf


class Runner(object):
    """sub-process used to collect experience"""

    def __init__(self, env_id):
        """initialize model0 and environment"""
        self.ID = env_id
        set_global_seeds(env_id*123)
        self.num_iteration=0
        self.local_num_agent=EnvParameters.LOCAL_N_AGENTS_LIST[0]

        self.env = MAPFEnv(env_id)
        self.env_cl =CL_MAPFEnv(env_id)

        self.local_device = torch.device('cuda') if SetupParameters.USE_GPU_LOCAL else torch.device('cpu')
        self.local_model = Model(env_id, self.local_device)

        self.env_cl._global_reset(0)
        self.first_time = True

    def run(self, weights,clear_flag,cl_task_num,switch_task):
        """run multiple steps and collect data for reinforcement learning"""
        with torch.no_grad():
            if switch_task:
                self.local_num_agent= EnvParameters.LOCAL_N_AGENTS_LIST[cl_task_num]
                self.global_reset_env(cl_task_num)
                self.num_iteration = 0
                self.global_perf = init_global_perf()
                self.first_time = True

            self.num_iteration+=1
            self.one_episode_perf = init_one_episode_perf()
            if clear_flag:
                self.global_perf = init_global_perf()

            self.global_done, destroy_weights, global_num_collison, done, valid_actions, obs, \
                vector, train_valid, makespan, num_update_path, reduced_collison = self.local_reset_env(first_time=self.first_time, ALNS=True)

            self.first_time = False
            self.one_episode_perf["num_update_path"] = num_update_path
            self.one_episode_perf["reduced_collide"] = reduced_collison

            hidden_state = (
                torch.zeros((self.local_num_agent, NetParameters.NET_SIZE)).to(self.local_device),
                torch.zeros((self.local_num_agent, NetParameters.NET_SIZE)).to(self.local_device))

            if self.global_done or self.num_iteration > TrainingParameters.ITERATION_LIMIT_LIST[cl_task_num]:
                if self.global_done:
                    self.global_perf["success_time"] += 1
                self.global_perf["task_global_collision"].append(global_num_collison)
                self.global_perf["destroy_weight_target"].append(destroy_weights[0])
                self.global_perf["destroy_weight_collision"].append(destroy_weights[1])
                self.global_perf["destroy_weight_random"].append(destroy_weights[2])
                self.global_perf["makespan"].append(makespan)

                self.global_reset_env(cl_task_num)
                self.global_done, _, _, done, valid_actions, obs, \
                    vector, train_valid, _, _, _ = self.local_reset_env(first_time=True, ALNS=True)
                self.global_perf["num_iteration"].append(self.num_iteration)
                self.num_iteration = 0

            mb_obs, mb_vector, mb_rewards, mb_values, mb_done, mb_ps, mb_actions = [], [], [], [], [], [], []
            mb_hidden_state = []
            mb_train_valid, mb_blocking = [], []
            mb_add_rewards = []

            self.local_model.set_weights(weights)
            while not done:
                mb_obs.append(obs)
                mb_vector.append(vector)
                mb_train_valid.append(train_valid)
                mb_hidden_state.append(
                    [hidden_state[0].cpu().detach().numpy(), hidden_state[1].cpu().detach().numpy()])
                mb_done.append(done)

                actions, ps, values, pre_block, hidden_state, num_invalid = \
                    self.local_model.step(obs, vector, valid_actions, hidden_state,self.local_num_agent)
                self.one_episode_perf['invalid'] += num_invalid
                mb_values.append(values)
                mb_ps.append(ps)

                rewards, valid_actions, obs,vector, train_valid, done, blockings, \
                    num_on_goals, max_on_goals,add_reward = self.one_step(actions, pre_block)

                mb_actions.append(actions)
                mb_rewards.append(rewards)
                mb_add_rewards.append(add_reward)
                mb_blocking.append(blockings)

                self.one_episode_perf['reward'] += np.sum(rewards)
                if self.one_episode_perf['num_step'] == EnvParameters.EPISODE_LEN // 2:
                    self.one_episode_perf['half_goals']=num_on_goals

            # post processing
            self.one_episode_perf['invalid']=(self.one_episode_perf['num_step'] * self.local_num_agent - self.one_episode_perf['invalid']) / (
                        self.one_episode_perf['num_step'] * self.local_num_agent)
            self.one_episode_perf['final_goals']=num_on_goals
            self.one_episode_perf['wrong_blocking']=(self.one_episode_perf['num_step'] * self.local_num_agent - self.one_episode_perf['wrong_blocking']) / (
                        self.one_episode_perf['num_step'] * self.local_num_agent)
            self.one_episode_perf['max_goals']=max_on_goals

            mb_obs = np.concatenate(mb_obs, axis=0)
            mb_vector = np.concatenate(mb_vector, axis=0)
            mb_rewards = np.concatenate(mb_rewards, axis=0)
            mb_add_rewards = np.concatenate(mb_add_rewards, axis=0)
            mb_values = np.squeeze(np.concatenate(mb_values, axis=0), axis=-1)
            mb_actions = np.asarray(mb_actions, dtype=np.int64)
            mb_ps = np.stack(mb_ps)
            mb_done = np.asarray(mb_done, dtype=np.bool_)
            mb_hidden_state = np.stack(mb_hidden_state)
            mb_train_valid = np.stack(mb_train_valid)
            mb_blocking = np.concatenate(mb_blocking, axis=0)

            for i in range(self.local_num_agent):
                if self.one_episode_perf['num_step'] >= EnvParameters.EPISODE_LEN or self.env.total_coll[i]>self.env.sipps_coll[i]:
                    mb_rewards[:,i] += mb_add_rewards[:,i]
                    self.one_episode_perf['be_penaltied'] += 1

            last_values = np.squeeze(
                self.local_model.value(obs, vector, hidden_state))
            # calculate advantages
            mb_advs = np.zeros_like(mb_rewards)
            last_gaelam= 0
            for t in reversed(range(self.one_episode_perf['num_step'])):
                if t == self.one_episode_perf['num_step'] - 1:
                    next_nonterminal = 1.0 - done
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
            self.one_episode_perf['num_step'], self.one_episode_perf,self.global_perf

    def cl_run(self, weights,cl_task_num,switch_task):
        """run multiple steps and collect data for reinforcement learning"""
        with torch.no_grad():
            if switch_task:
                self.local_num_agent= EnvParameters.LOCAL_N_AGENTS_LIST[cl_task_num]
                self.env_cl._global_reset(cl_task_num)
                self.num_iteration = 0
                self.first_time = True

            self.num_iteration += 1
            if self.num_iteration >= TrainingParameters.ITERATION_LIMIT_LIST[cl_task_num]:
                self.num_iteration = 0
                self.env_cl._global_reset(cl_task_num)
                self.first_time = True

            self.one_episode_perf = init_one_episode_perf()
            done, valid_actions, obs, vector, train_valid = self.cl_local_reset_env(self.first_time)
            hidden_state = (
                torch.zeros((self.local_num_agent, NetParameters.NET_SIZE)).to(self.local_device),
                torch.zeros((self.local_num_agent, NetParameters.NET_SIZE)).to(self.local_device))
            self.first_time = False

            mb_obs, mb_vector, mb_rewards, mb_values, mb_done, mb_ps, mb_actions = [], [], [], [], [], [], []
            mb_hidden_state = []
            mb_train_valid, mb_blocking = [], []
            mb_add_rewards = []

            self.local_model.set_weights(weights)
            while not done:
                mb_obs.append(obs)
                mb_vector.append(vector)
                mb_train_valid.append(train_valid)
                mb_hidden_state.append(
                    [hidden_state[0].cpu().detach().numpy(), hidden_state[1].cpu().detach().numpy()])
                mb_done.append(done)

                actions, ps, values, pre_block, hidden_state, num_invalid = \
                    self.local_model.step(obs, vector, valid_actions, hidden_state,self.local_num_agent)
                self.one_episode_perf['invalid'] += num_invalid

                mb_values.append(values)
                mb_ps.append(ps)

                rewards, valid_actions, obs, vector, train_valid, done, blockings, \
                    num_on_goals, max_on_goals,add_reward  = self.cl_one_step(actions, pre_block)

                mb_actions.append(actions)
                mb_rewards.append(rewards)
                mb_add_rewards.append(add_reward)
                mb_blocking.append(blockings)

                self.one_episode_perf['reward'] += np.sum(rewards)
                if self.one_episode_perf['num_step'] == EnvParameters.EPISODE_LEN // 2:
                    self.one_episode_perf['half_goals']=num_on_goals

            self.one_episode_perf['invalid']=(self.one_episode_perf['num_step'] * self.local_num_agent - self.one_episode_perf['invalid']) / (
                        self.one_episode_perf['num_step'] * self.local_num_agent)
            self.one_episode_perf['final_goals']=num_on_goals
            self.one_episode_perf['wrong_blocking']=(self.one_episode_perf['num_step'] * self.local_num_agent - self.one_episode_perf['wrong_blocking']) / (
                        self.one_episode_perf['num_step'] * self.local_num_agent)
            self.one_episode_perf['max_goals']=max_on_goals

            mb_obs = np.concatenate(mb_obs, axis=0)
            mb_vector = np.concatenate(mb_vector, axis=0)
            mb_rewards = np.concatenate(mb_rewards, axis=0)
            mb_add_rewards = np.concatenate(mb_add_rewards, axis=0)
            mb_values = np.squeeze(np.concatenate(mb_values, axis=0), axis=-1)
            mb_actions = np.asarray(mb_actions, dtype=np.int64)
            mb_ps = np.stack(mb_ps)
            mb_done = np.asarray(mb_done, dtype=np.bool_)
            mb_hidden_state = np.stack(mb_hidden_state)
            mb_train_valid = np.stack(mb_train_valid)
            mb_blocking = np.concatenate(mb_blocking, axis=0)

            last_values = np.squeeze(
                self.local_model.value(obs, vector, hidden_state))
            for i in range(self.local_num_agent):
                if self.one_episode_perf['num_step'] >= EnvParameters.EPISODE_LEN or self.env_cl.total_coll[i]>self.env_cl.sipps_coll[i]:
                    mb_rewards[:,i] += mb_add_rewards[:,i]
                    self.one_episode_perf['be_penaltied'] += 1

            # calculate advantages
            mb_advs = np.zeros_like(mb_rewards)
            last_gaelam= 0
            for t in reversed(range(self.one_episode_perf['num_step'])):
                if t == self.one_episode_perf['num_step'] - 1:
                    next_nonterminal = 1.0 - done
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
            self.one_episode_perf['num_step'], self.one_episode_perf,{"num_iteration":[]}

    def local_reset_env(self,first_time, ALNS):
        global_done, destroy_weights, global_num_collison, makespan, num_update_path, reduced_collison = self.env._local_reset(
            self.local_num_agent, first_time, ALNS)
        if global_done:
            assert (global_num_collison == 0)
            return True, destroy_weights, global_num_collison, False, 0, 0, 0, 0, makespan, num_update_path, reduced_collison

        valid_actions = []
        obs = np.zeros((1, self.local_num_agent, NetParameters.NUM_CHANNEL, EnvParameters.FOV_SIZE,
                        EnvParameters.FOV_SIZE),
                       dtype=np.float32)
        vector = np.zeros((1, self.local_num_agent, NetParameters.VECTOR_LEN), dtype=np.float32)
        train_valid = np.zeros((self.local_num_agent, EnvParameters.N_ACTIONS), dtype=np.float32)

        for i in range(self.local_num_agent):
            valid_action = self.env.list_next_valid_actions(i)
            s = self.env.observe(i)
            obs[:, i, :, :, :] = s[0]
            vector[:, i, : 3] = s[1]
            vector[:, i, 3] = self.env.sipps_coll[i]/(self.env.sipps_coll[i]+1)
            valid_actions.append(valid_action)
            train_valid[i, valid_action] = 1
        vector[:, :, 4] = sum(self.env.sipps_coll) / (sum(self.env.sipps_coll) + 1)
        return global_done, destroy_weights, global_num_collison, False, valid_actions, obs, vector, train_valid, makespan, num_update_path, reduced_collison

    def cl_local_reset_env(self,first_time=False):
        self.env_cl._local_reset(self.local_num_agent,first_time)
        valid_actions = []
        obs = np.zeros((1, self.local_num_agent, NetParameters.NUM_CHANNEL, EnvParameters.FOV_SIZE,
                        EnvParameters.FOV_SIZE),
                       dtype=np.float32)
        vector = np.zeros((1, self.local_num_agent, NetParameters.VECTOR_LEN), dtype=np.float32)
        train_valid = np.zeros((self.local_num_agent, EnvParameters.N_ACTIONS), dtype=np.float32)

        for i in range(self.local_num_agent):
            valid_action = self.env_cl.list_next_valid_actions(i)
            s = self.env_cl.observe(i)
            obs[:, i, :, :, :] = s[0]
            vector[:, i, : 3] = s[1]
            vector[:, i, 3] = self.env_cl.sipps_coll[i]/(self.env_cl.sipps_coll[i]+1)
            valid_actions.append(valid_action)
            train_valid[i, valid_action] = 1
        vector[:, :, 4] = sum(self.env_cl.sipps_coll) / (sum(self.env_cl.sipps_coll) + 1)
        return False, valid_actions, obs, vector, train_valid

    def global_reset_env(self,cl_task_num):
        """reset environment"""
        can_not_use = True
        while can_not_use:
            can_not_use, global_num_collision = self.env._global_reset(cl_task_num)
        return global_num_collision

    def one_step( self,actions, pre_block):
        """run one step"""
        train_valid = np.zeros(( self.local_num_agent, EnvParameters.N_ACTIONS), dtype=np.float32)

        obs, vector, rewards, done, next_valid_actions, blockings, num_blockings, leave_goals, \
            num_on_goal, max_on_goal, num_dynamic_collide, num_agent_collide,add_reward \
            = self.env.joint_step(actions)

        self.one_episode_perf['block'] += num_blockings
        self.one_episode_perf['num_leave_goal'] += leave_goals
        self.one_episode_perf['num_dynamic_collide'] += num_dynamic_collide
        self.one_episode_perf['num_agent_collide'] += num_agent_collide

        for i in range( self.local_num_agent):
            train_valid[i, next_valid_actions[i]] = 1
            if (pre_block[i] < 0.5) == blockings[:, i]:
                self.one_episode_perf['wrong_blocking'] += 1
        self.one_episode_perf['num_step'] += 1
        return rewards, next_valid_actions, obs, vector, train_valid, done, blockings, num_on_goal,max_on_goal,add_reward

    def cl_one_step( self,actions, pre_block):
        """run one step"""
        train_valid = np.zeros(( self.local_num_agent, EnvParameters.N_ACTIONS), dtype=np.float32)

        obs, vector, rewards, done, next_valid_actions, blockings, num_blockings, leave_goals, \
            num_on_goal, max_on_goal, num_dynamic_collide, num_agent_collide,add_reward \
            = self.env_cl.joint_step(actions)

        self.one_episode_perf['block'] += num_blockings
        self.one_episode_perf['num_leave_goal'] += leave_goals
        self.one_episode_perf['num_dynamic_collide'] += num_dynamic_collide
        self.one_episode_perf['num_agent_collide'] += num_agent_collide

        for i in range( self.local_num_agent):
            train_valid[i, next_valid_actions[i]] = 1
            if (pre_block[i] < 0.5) == blockings[:, i]:
                self.one_episode_perf['wrong_blocking'] += 1
        self.one_episode_perf['num_step'] += 1
        return rewards, next_valid_actions, obs, vector, train_valid, done, blockings, num_on_goal,max_on_goal,add_reward


@ray.remote(num_cpus=1, num_gpus=SetupParameters.NUM_GPU / (TrainingParameters.N_ENVS + 1))
class RLRunner(Runner):
    def __init__(self, meta_agent_id):
        super().__init__(meta_agent_id)

if __name__ == "__main__":
    import os
    if not os.path.exists("./record_files"):
        os.makedirs("./record_files")

    global_device = torch.device('cpu')
    local_device = torch.device('cpu')

    # initialize neural networks
    # model_path = '/home/marmot/Yutong/MAPF/training/cl_step_block/models/MAPF/cl_timestep11-08-231830/final'
    # path_checkpoint = model_path + "/net_checkpoint.pkl"
    # global_model = Model(0, global_device, True)
    # global_model.network.load_state_dict(torch.load(path_checkpoint)['model'])

    global_model = Model(0, global_device, True)

    # launch meta agents
    env = Runner(0)

    if global_device != local_device:
        net_weights = global_model.network.to(local_device).state_dict()
        global_model.network.to(global_device)
    else:
        net_weights = global_model.network.state_dict()
    #
    # job_results = env.cl_run(net_weights,0, False)
    job_results = env.run(net_weights, False, 4, True)

    print("test")



