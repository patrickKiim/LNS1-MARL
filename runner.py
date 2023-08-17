import numpy as np
import ray
import torch

from alg_parameters import *
from mapf_gym import MAPFEnv
from mapf_gym_imitation import MAPFEnv_imitation
from model import Model
from util import update_perf,set_global_seeds,init_global_perf,init_one_episode_perf,init_performance_dict_runner


class Runner(object):
    """sub-process used to collect experience"""

    def __init__(self, env_id):
        """initialize model0 and environment"""
        self.ID = env_id
        set_global_seeds(env_id*123)
        self.local_num_agent = EnvParameters.LOCAL_N_AGENTS
        self.one_episode_perf = init_one_episode_perf()
        self.global_perf=init_global_perf()
        self.num_iteration=0

        self.env = MAPFEnv(env_id)

        self.local_device = torch.device('cuda') if SetupParameters.USE_GPU_LOCAL else torch.device('cpu')
        self.local_model = Model(env_id, self.local_device)
        self.hidden_state = (
            torch.zeros((self.local_num_agent, NetParameters.NET_SIZE)).to(self.local_device),
            torch.zeros((self.local_num_agent, NetParameters.NET_SIZE)).to(self.local_device))

        self.global_num_collison = self.global_reset_env()
        self.global_done,self.destroy_weights,_,self.done, self.valid_actions, self.obs, \
        self.vector, self.train_valid, makespan,_,_ = self.local_reset_env(EnvParameters.LOCAL_N_AGENTS,first_time=True,ALNS=True)
        # assert (self.num_global_collision==global_num_collison)

    def run(self, weights,clear_flag):
        """run multiple steps and collect data for reinforcement learning"""
        with torch.no_grad():
            mb_obs, mb_vector, mb_rewards, mb_values, mb_done, mb_ps, mb_actions = [], [], [], [], [], [], []
            mb_hidden_state = []
            mb_train_valid, mb_blocking = [], []
            performance_dict = init_performance_dict_runner()
            if clear_flag:
                self.global_perf = init_global_perf()

            self.local_model.set_weights(weights)
            for _ in range(TrainingParameters.N_STEPS):
                mb_obs.append(self.obs)
                mb_vector.append(self.vector)
                mb_train_valid.append(self.train_valid)
                mb_hidden_state.append(
                    [self.hidden_state[0].cpu().detach().numpy(), self.hidden_state[1].cpu().detach().numpy()])
                mb_done.append(self.done)

                actions, ps, values, pre_block, self.hidden_state, num_invalid = \
                    self.local_model.step(self.obs, self.vector, self.valid_actions, self.hidden_state)
                self.one_episode_perf['invalid'] += num_invalid
                mb_values.append(values)
                mb_ps.append(ps)

                rewards, self.valid_actions, self.obs, self.vector, self.train_valid, self.done, blockings, \
                    num_on_goals, max_on_goals = self.one_step(actions, pre_block)

                mb_actions.append(actions)
                mb_rewards.append(rewards)
                mb_blocking.append(blockings)

                self.one_episode_perf['reward'] += np.sum(rewards)
                if self.one_episode_perf['num_step'] == EnvParameters.EPISODE_LEN // 2:
                    performance_dict['per_half_goals'].append(num_on_goals)

                if self.done:
                    self.num_iteration+=1

                    self.global_done, self.destroy_weights, self.global_num_collison, self.done, self.valid_actions, self.obs, \
                    self.vector, self.train_valid, makespan,num_update_path,reduced_collison = self.local_reset_env(EnvParameters.LOCAL_N_AGENTS,
                                                                    first_time=False, ALNS=True)

                    performance_dict["episode_global_collision"].append(self.global_num_collison)
                    performance_dict["per_num_update_path"]+=num_update_path
                    performance_dict["per_reduced_collide"]+=reduced_collison

                    if self.global_done or self.num_iteration>TrainingParameters.ITERATION_LIMIT:
                        if self.global_done:
                            self.global_perf["success_time"]+=1
                        self.global_perf["task_global_collision"].append(self.global_num_collison)
                        self.global_perf["destroy_weight_target"].append(self.destroy_weights[0])
                        self.global_perf["destroy_weight_collision"].append(self.destroy_weights[1])
                        self.global_perf["destroy_weight_random"].append(self.destroy_weights[2])
                        self.global_perf["makespan"].append(makespan)

                        self.global_num_collison= self.global_reset_env()
                        self.global_done, self.destroy_weights, _, self.done, self.valid_actions, self.obs, \
                        self.vector, self.train_valid,makespan,_,_ = self.local_reset_env(EnvParameters.LOCAL_N_AGENTS,first_time=True,ALNS=True)
                        # assert (self.num_global_collision == global_num_collison)
                        self.global_perf["num_iteration"].append(self.num_iteration)
                        self.num_iteration=0

                    performance_dict = update_perf(self.one_episode_perf, performance_dict, num_on_goals,
                                                   max_on_goals)
                    self.one_episode_perf = init_one_episode_perf()
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

    
    def imitation(self, weights):

        """run multiple steps and collect corresponding data for imitation learning"""
        # self.imitation_env = MAPFEnv_imitation(self.ID+TrainingParameters.N_ENVS+1)
        #print("IL ! ")
        self.local_model.set_weights(weights)

        mb_obs, mb_vector, mb_hidden_state, mb_actions = [], [], [], []
        step = 0
        episode = 0
        self.imitation_num_agent = EnvParameters.LOCAL_N_AGENTS
        self.imitation_env = MAPFEnv_imitation(self.ID)

        with torch.no_grad():
            
            _= self.global_reset_env_imitation()
            self.global_done_i,_,global_num_collison,self.done_i,_,_,_,_,_,_,_,self.planner_path = self.local_reset_env_imitation(EnvParameters.LOCAL_N_AGENTS,
                                                                first_time=True,ALNS=True)
            
            # self.planner_path = self.imitation_env.get_path_intermediate()

            while step <= TrainingParameters.N_STEPS:

                self.planner_path = self.imitation_env.get_path_intermediate()
                obs, vector, actions, hidden_state = self.parse_path(self.planner_path)    

                if obs is not None:
                    mb_obs.append(obs)
                    mb_vector.append(vector)
                    mb_actions.append(actions)
                    mb_hidden_state.append(hidden_state)
                    
                    step += np.shape(vector)[0]
                    episode += 1

                self.global_done_i,_,global_num_collison,self.done_i,_,_,_,_,_,_,_,self.planner_path = self.local_reset_env_imitation(EnvParameters.LOCAL_N_AGENTS,
                                                                first_time=False,ALNS=True)
                
                if self.global_done_i:
                    _ = self.global_reset_env_imitation()
                    self.global_done_i,_,global_num_collison,self.done_i,_,_,_,_,_,_,_,self.planner_path = self.local_reset_env_imitation(EnvParameters.LOCAL_N_AGENTS,
                                                                first_time=True,ALNS=True)
                # print(step)
                # if step == 0 and self.global_done_i == False:
                #     obs, vector, actions, hidden_state = self.parse_path(self.planner_path)
                #     if obs is not None:
                #         mb_obs.append(obs)
                #         mb_vector.append(vector)
                #         mb_actions.append(actions)
                #         mb_hidden_state.append(hidden_state)
                        
                #         step += np.shape(vector)[0]
                #         episode += 1
                #     continue

                # if self.global_done_i == False:
                #     '''This below step needed for local update in case of self._global_done_i = False'''

                #     self.global_done_i,_,global_num_collison,self.done_i,_,_,_,_,_,_,_,self.planner_path = self.local_reset_env_imitation(EnvParameters.LOCAL_N_AGENTS,
                #                                                                 first_time=False,ALNS=True)

                #     self.planner_path = self.imitation_env.get_path_intermediate()

                #     if self.global_done_i == False:
                #         obs, vector, actions, hidden_state = self.parse_path(self.planner_path)
                #         if obs is not None:
                #             mb_obs.append(obs)
                #             mb_vector.append(vector)
                #             mb_actions.append(actions)
                #             mb_hidden_state.append(hidden_state)
                            
                #             step += np.shape(vector)[0]
                #             episode += 1
                #     continue
                
                # if self.global_done_i == True :
              
                #     _ = self.global_reset_env_imitation()
                #     self.global_done_i,_,global_num_collison,self.done_i,_,_,_,_,_,_,_,self.planner_path = self.local_reset_env_imitation(EnvParameters.LOCAL_N_AGENTS,
                #                                                         first_time=True,ALNS=True)
                #     self.planner_path = self.imitation_env.get_path_intermediate()
                    
                #     if self.global_done_i == False:
                #         obs, vector, actions, hidden_state = self.parse_path(self.planner_path)
                #         if obs is not None:
                #             mb_obs.append(obs)
                #             mb_vector.append(vector)
                #             mb_actions.append(actions)
                #             mb_hidden_state.append(hidden_state)
                                
                #             step += np.shape(vector)[0]
                #             episode += 1
                #         continue
                
            mb_obs = np.concatenate(mb_obs, axis=0)
            mb_vector = np.concatenate(mb_vector, axis=0)
            mb_actions = np.concatenate(mb_actions, axis=0)
            mb_hidden_state = np.concatenate(mb_hidden_state, axis=0)

        return mb_obs, mb_vector, mb_actions, mb_hidden_state, episode, step
    
    def parse_path(self, path):
        """take the path generated from M* and create the corresponding inputs and actions"""
        mb_obs, mb_vector, mb_actions, mb_hidden_state = [], [], [], []

        hidden_state = (
            torch.zeros((self.imitation_num_agent, NetParameters.NET_SIZE)).to(self.local_device),
            torch.zeros((self.imitation_num_agent, NetParameters.NET_SIZE)).to(self.local_device))
        obs = np.zeros((1, self.imitation_num_agent, NetParameters.NUM_CHANNEL, EnvParameters.FOV_SIZE, EnvParameters.FOV_SIZE),
                       dtype=np.float32)
        vector = np.zeros((1, self.imitation_num_agent, NetParameters.VECTOR_LEN), dtype=np.float32)

        for i in range(self.imitation_num_agent):
            s = self.imitation_env.observe(i) # it was i+1 before
            obs[:, i, :, :, :] = s[0]
            vector[:, i, : 3] = s[1]

        max_path_len = 0
        for i in range(self.imitation_num_agent):
            if max_path_len < len(path[i]):
                max_path_len = len(path[i])
        
        # print(len(path))
        # print(len(path[0]))
        # print(max_path_len)

        for t in range(max_path_len-1):  # may be this will be max_path_len-1
            # print(t)
            mb_obs.append(obs)
            mb_vector.append(vector)
            mb_hidden_state.append([hidden_state[0].cpu().detach().numpy(), hidden_state[1].cpu().detach().numpy()])
            hidden_state = self.local_model.generate_state(obs, vector, hidden_state)
            # print("generate state ! ")
            actions = np.zeros(self.imitation_num_agent)
            for i in range(self.imitation_num_agent):
                if (t > len(path[i])-2):  # should not be problem due to loop guard 
                    direction = (0,0)
                else:
                    pos = path[i][t]
                    new_pos = path[i][t+1]  # guaranteed to be in bounds by loop guard
                    direction = (new_pos[0] - pos[0], new_pos[1] - pos[1])
                try:
                    # print("get action ! ")
                    actions[i] = self.imitation_env.world.get_action(direction)
                except:
                    print(pos, new_pos, i, t)
            mb_actions.append(actions)
            # print("Joint step ! ")
            obs, vector, rewards, self.done_imitation, valid_actions, _, _, _, _, _, _,_ = \
                self.imitation_env.joint_step(actions)
            vector[:, :, -1] = actions

            if not all(valid_actions):  # SIPPS can not generate collisions
                print('invalid action')
                return None, None, None, None

        mb_obs = np.concatenate(mb_obs, axis=0)
        mb_vector = np.concatenate(mb_vector, axis=0)
        mb_actions = np.asarray(mb_actions, dtype=np.int64)
        mb_hidden_state = np.stack(mb_hidden_state)
        # print("parse : ", mb_hidden_state.shape)
        # print("Parse ! ")
        return mb_obs, mb_vector, mb_actions, mb_hidden_state
    

    def local_reset_env(self,local_num_agents, first_time, ALNS):
        global_done, destroy_weights, global_num_collison, makespan, num_update_path, reduced_collison, sipps_path = self.env._local_reset(
            local_num_agents, first_time, ALNS)
        if global_done:
            assert (global_num_collison == 0)
            return True, destroy_weights, global_num_collison, False, 0, 0, 0, 0, makespan, num_update_path, reduced_collison

        local_done = False
        prev_action = np.zeros(EnvParameters.LOCAL_N_AGENTS)
        valid_actions = []
        obs = np.zeros((1, EnvParameters.LOCAL_N_AGENTS, NetParameters.NUM_CHANNEL, EnvParameters.FOV_SIZE,
                        EnvParameters.FOV_SIZE),
                       dtype=np.float32)
        vector = np.zeros((1, EnvParameters.LOCAL_N_AGENTS, NetParameters.VECTOR_LEN), dtype=np.float32)
        train_valid = np.zeros((EnvParameters.LOCAL_N_AGENTS, EnvParameters.N_ACTIONS), dtype=np.float32)

        for i in range(EnvParameters.LOCAL_N_AGENTS):
            valid_action = self.env.list_next_valid_actions(i)
            s = self.env.observe(i)
            obs[:, i, :, :, :] = s[0]
            vector[:, i, : 3] = s[1]
            vector[:, i, -1] = prev_action[i]
            valid_actions.append(valid_action)
            train_valid[i, valid_action] = 1
        return global_done, destroy_weights, global_num_collison, local_done, valid_actions, obs, vector, train_valid, makespan, num_update_path, reduced_collison

    def local_reset_env_imitation(self,local_num_agents, first_time, ALNS):
        global_done, destroy_weights, global_num_collison, makespan, num_update_path, reduced_collison, sipps_path = self.imitation_env._local_reset(
            local_num_agents, first_time, ALNS)
        if global_done:
            assert (global_num_collison == 0)
            return True, destroy_weights, global_num_collison, False, 0, 0, 0, 0, makespan, num_update_path, reduced_collison, sipps_path

        local_done = False
        prev_action = np.zeros(EnvParameters.LOCAL_N_AGENTS)
        valid_actions = []
        obs = np.zeros((1, EnvParameters.LOCAL_N_AGENTS, NetParameters.NUM_CHANNEL, EnvParameters.FOV_SIZE,
                        EnvParameters.FOV_SIZE),
                       dtype=np.float32)
        vector = np.zeros((1, EnvParameters.LOCAL_N_AGENTS, NetParameters.VECTOR_LEN), dtype=np.float32)
        train_valid = np.zeros((EnvParameters.LOCAL_N_AGENTS, EnvParameters.N_ACTIONS), dtype=np.float32)

        for i in range(EnvParameters.LOCAL_N_AGENTS):
            valid_action = self.env.list_next_valid_actions(i)
            s = self.env.observe(i)
            obs[:, i, :, :, :] = s[0]
            vector[:, i, : 3] = s[1]
            vector[:, i, -1] = prev_action[i]
            valid_actions.append(valid_action)
            train_valid[i, valid_action] = 1
        return global_done, destroy_weights, global_num_collison, local_done, valid_actions, obs, vector, train_valid, makespan, num_update_path, reduced_collison, sipps_path

    def global_reset_env(self):
        """reset environment"""
        can_not_use = True
        while can_not_use:
            can_not_use, global_num_collision = self.env._global_reset()
        return global_num_collision
    
    def global_reset_env_imitation(self):
        """reset environment"""
        can_not_use = True
        while can_not_use:
            can_not_use, global_num_collision = self.imitation_env._global_reset()
        return global_num_collision

    def one_step(self,actions, pre_block):
        """run one step"""
        train_valid = np.zeros((EnvParameters.LOCAL_N_AGENTS, EnvParameters.N_ACTIONS), dtype=np.float32)

        obs, vector, rewards, done, next_valid_actions, blockings, num_blockings, leave_goals, \
            num_on_goal, max_on_goal, num_dynamic_collide, num_agent_collide \
            = self.env.joint_step(actions)

        self.one_episode_perf['block'] += num_blockings
        self.one_episode_perf['num_leave_goal'] += leave_goals
        self.one_episode_perf['num_dynamic_collide'] += num_dynamic_collide
        self.one_episode_perf['num_agent_collide'] += num_agent_collide

        for i in range(EnvParameters.LOCAL_N_AGENTS):
            train_valid[i, next_valid_actions[i]] = 1
            if (pre_block[i] < 0.5) == blockings[:, i]:
                self.one_episode_perf['wrong_blocking'] += 1
        self.one_episode_perf['num_step'] += 1
        return rewards, next_valid_actions, obs, vector, train_valid, done, blockings, num_on_goal,max_on_goal

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
    global_model = Model(0, global_device, True)

    # launch meta agents
    env = Runner(0)

    if global_device != local_device:
        net_weights = global_model.network.to(local_device).state_dict()
        global_model.network.to(global_device)
    else:
        net_weights = global_model.network.state_dict()

    job_results = env.run(net_weights, False)
    print("test")

