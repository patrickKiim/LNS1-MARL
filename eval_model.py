import numpy as np
import torch

from alg_parameters import *
from mapf_gym import MAPFEnv
from util import set_global_seeds,eval_init_global_perf

class Eval(object):

    def __init__(self):
        self.global_device = torch.device('cuda') if SetupParameters.USE_GPU_GLOBAL else torch.device('cpu')
        set_global_seeds(SetupParameters.SEED)
        self.global_env=MAPFEnv(TrainingParameters.N_ENVS+1,eval=True)
        self.scalar_env = MAPFEnv(TrainingParameters.N_ENVS + 2,eval=True)

    def eval_global(self, model):
        with torch.no_grad():
            one_episode_perf = {'num_step': [], 'reward': [], 'invalid': [], 'block': [],  'reduced_collision': [],
                        "num_dynamic_collide": [], "num_agent_collide": [], "final_goal": []}
            self.dict_set = eval_init_global_perf()
            recording={'num_step': 0, 'reward': 0, 'invalid': 0, 'block': 0, 'num_dynamic_collide': 0, "num_agent_collide": 0}
            num_iteration =total_update=0
            first_time=True

            _,global_num_collison = self.global_env._global_reset(-1)
            while num_iteration <= RecordingParameters.EVAL_MAX_ITERATION:
                num_iteration+=1
                hidden_state = (
                    torch.zeros((RecordingParameters.EVAL_NUM_AGENT_GLOBAL, NetParameters.NET_SIZE)).to(
                        self.global_device),
                    torch.zeros((RecordingParameters.EVAL_NUM_AGENT_GLOBAL, NetParameters.NET_SIZE)).to(
                        self.global_device))
                self.global_done, global_num_collison,  local_done, valid_actions, obs, \
                    vector, makespan, num_update, reduced_collison = self.local_reset_env(RecordingParameters.EVAL_NUM_AGENT_GLOBAL,
                                                                                         first_time=first_time, ALNS=True)
                first_time = False
                total_update+=num_update
                one_episode_perf['reduced_collision'].append(reduced_collison)
                self.record_global(num_iteration,global_num_collison,total_update, makespan)

                if self.global_done:
                    break

                while not local_done:
                    actions, hidden_state, num_invalid = model.eval(obs, vector,valid_actions,hidden_state,RecordingParameters.EVAL_NUM_AGENT_GLOBAL)
                    obs, vector, rewards, local_done, valid_actions, _, num_blockings, _, \
                        num_on_goal, _, num_dynamic_collide, num_agent_collide \
                        = self.global_env.joint_step(actions)

                    recording['block'] += num_blockings
                    recording['num_dynamic_collide'] += num_dynamic_collide
                    recording['num_agent_collide'] += num_agent_collide
                    recording['num_step'] += 1
                    recording['invalid'] += num_invalid
                    recording['reward'] += np.sum(rewards)

                for keys in recording.keys():
                    one_episode_perf[keys].append(recording[keys])
                    recording[keys]=0
                one_episode_perf["final_goal"].append(num_on_goal)

        return one_episode_perf, self.dict_set

    def eval_scalar(self, model):
        """run multiple steps and collect data for reinforcement learning"""
        with torch.no_grad():
            recording = {'num_step': 0, 'reward': 0, 'invalid': 0, 'block': 0, 'num_dynamic_collide': 0, "num_agent_collide": 0,
                         "reduced_collison":0,"final_goal":0}
            _, global_num_collison = self.scalar_env._global_reset(-1)
            self.scalar_env._local_reset(RecordingParameters.EVAL_NUM_AGENT, True,True)
            valid_actions = []
            obs = np.zeros((1, RecordingParameters.EVAL_NUM_AGENT, NetParameters.NUM_CHANNEL, EnvParameters.FOV_SIZE,
                            EnvParameters.FOV_SIZE),
                           dtype=np.float32)
            vector = np.zeros((1, RecordingParameters.EVAL_NUM_AGENT, NetParameters.VECTOR_LEN), dtype=np.float32)

            for i in range(RecordingParameters.EVAL_NUM_AGENT):
                valid_action = self.scalar_env.list_next_valid_actions(i)
                s = self.scalar_env.observe(i)
                obs[:, i, :, :, :] = s[0]
                vector[:, i, : 3] = s[1]
                valid_actions.append(valid_action)
            hidden_state = (
                torch.zeros((RecordingParameters.EVAL_NUM_AGENT, NetParameters.NET_SIZE)).to(
                    self.global_device),
                torch.zeros((RecordingParameters.EVAL_NUM_AGENT, NetParameters.NET_SIZE)).to(
                    self.global_device))
            local_done = False

            while not local_done:
                actions, hidden_state, num_invalid = model.eval(obs, vector, valid_actions, hidden_state,
                                                                               RecordingParameters.EVAL_NUM_AGENT)
                obs, vector, rewards, local_done, valid_actions, _, num_blockings, _, \
                    num_on_goal, _, num_dynamic_collide, num_agent_collide \
                    = self.scalar_env.joint_step(actions)

                recording['block'] += num_blockings
                recording['num_dynamic_collide'] += num_dynamic_collide
                recording['num_agent_collide'] += num_agent_collide
                recording['num_step'] += 1
                recording['invalid'] += num_invalid
                recording['reward'] += np.sum(rewards)

            _,_,_,_,_,reduced_collison=self.scalar_env._local_reset(RecordingParameters.EVAL_NUM_AGENT, False, True)

            recording["reduced_collison"]=reduced_collison
            recording["final_goal"]=num_on_goal

            return recording

    def record_global(self,num_iteration,global_num_collison,num_update, makespan):
        if self.global_done:
            avaliable_set = [-1]
            if num_iteration <= 300:
                avaliable_set.append(2)
            if num_iteration <= 200:
                avaliable_set.append(1)
            if num_iteration <= 100:
                avaliable_set.append(0)

            for dict_number in avaliable_set:
                self.dict_set[dict_number]["success_time"] += 1
                self.dict_set[dict_number]["global_collision"] = global_num_collison
                self.dict_set[dict_number]["num_update"] = num_update
                self.dict_set[dict_number]["makespan"] = makespan
        else:
            record_number = 100000
            if num_iteration == 400:
                record_number = -1
            if num_iteration == 300:
                record_number = -2
            if num_iteration == 200:
                record_number = -3
            if num_iteration == 100:
                record_number = -4
            if record_number != 100000:
                self.dict_set[record_number]["global_collision"] = global_num_collison
                self.dict_set[record_number]["num_update"] = num_update
                self.dict_set[record_number]["makespan"] = makespan

    def local_reset_env(self,local_num_agents,first_time, ALNS):
        global_done, _, global_num_collison, makespan, num_update_path, reduced_collison = self.global_env._local_reset(
        local_num_agents, first_time, ALNS)

        if global_done:
            assert (global_num_collison == 0)
            return True, global_num_collison, False, 0, 0, 0, makespan, num_update_path, reduced_collison

        valid_actions = []
        obs = np.zeros((1, local_num_agents, NetParameters.NUM_CHANNEL, EnvParameters.FOV_SIZE,
                        EnvParameters.FOV_SIZE),
                       dtype=np.float32)
        vector = np.zeros((1, local_num_agents, NetParameters.VECTOR_LEN), dtype=np.float32)

        for i in range(local_num_agents):
            valid_action = self.global_env.list_next_valid_actions(i)
            s = self.global_env.observe(i)
            obs[:, i, :, :, :] = s[0]
            vector[:, i, : 3] = s[1]
            valid_actions.append(valid_action)
        return global_done, global_num_collison, False, valid_actions, obs, vector, makespan, num_update_path, reduced_collison

if __name__ == '__main__':
    import os
    from model import Model
    from util import set_global_seeds, write_to_wandb_global, write_to_wandb_scalar
    import wandb

    if not os.path.exists("./record_files"):
        os.makedirs("./record_files")

    wandb_id = wandb.util.generate_id()
    wandb.init(project=RecordingParameters.EXPERIMENT_PROJECT,
                   name=RecordingParameters.EXPERIMENT_NAME,
                   entity=RecordingParameters.ENTITY,
                   notes=RecordingParameters.EXPERIMENT_NOTE,
                   config=all_args,
                   id=wandb_id,
                   resume='allow')


    eval_env=Eval()
    global_device = torch.device('cuda')

    # initialize neural networks
    global_model = Model(0, global_device, True)

    # global_one_episode_perf, global_eval_perf = eval_env.eval_global(global_model)
    scalar_eval_perf = eval_env.eval_scalar(global_model)
    # write_to_wandb_global(0, global_one_episode_perf, global_eval_perf)
    write_to_wandb_scalar(0, scalar_eval_perf)


