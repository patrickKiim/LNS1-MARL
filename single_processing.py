import os
import os.path as osp

import numpy as np
import setproctitle
# from torch.utils.tensorboard import SummaryWriter
import torch
import wandb

from alg_parameters import *
from mapf_gym import MAPFEnv
from model import Model
from util import set_global_seeds,  write_to_wandb, make_gif,global_write_to_wandb, one_step, update_perf,global_reset_env,local_reset_env

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("Welcome to LNS2 MAPF!\n")

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

                    if self.global_done or self.num_iteration>1:
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


def main():
    """main code"""
    # preparing for training
    if RecordingParameters.RETRAIN:
        restore_path = './local_model'
        net_path_checkpoint = restore_path + "/net_checkpoint.pkl"
        net_dict = torch.load(net_path_checkpoint)

    if RecordingParameters.WANDB:
        if RecordingParameters.RETRAIN:
            wandb_id = None
        else:
            wandb_id = wandb.util.generate_id()
        wandb.init(project=RecordingParameters.EXPERIMENT_PROJECT,
                   name=RecordingParameters.EXPERIMENT_NAME,
                   entity=RecordingParameters.ENTITY,
                   notes=RecordingParameters.EXPERIMENT_NOTE,
                   config=all_args,
                   id=wandb_id,
                   resume='allow')
        print('id is:{}'.format(wandb_id))
        print('Launching wandb...\n')

    setproctitle.setproctitle(
        RecordingParameters.EXPERIMENT_PROJECT + RecordingParameters.EXPERIMENT_NAME + "@" + RecordingParameters.ENTITY)
    set_global_seeds(SetupParameters.SEED)

    # create classes
    global_device = torch.device('cuda') if SetupParameters.USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if SetupParameters.USE_GPU_LOCAL else torch.device('cpu')
    global_model = Model(0, global_device, True)

    if RecordingParameters.WANDB:
        wandb.watch(global_model.network, log_freq=500, log_graph=True)

    if RecordingParameters.RETRAIN:
        global_model.network.load_state_dict(net_dict['model'])
        global_model.net_optimizer.load_state_dict(net_dict['optimizer'])

    envs = Runner(1)
    # eval_env = MAPFEnv()

    if RecordingParameters.RETRAIN:
        curr_steps = net_dict["step"]
        curr_episodes = net_dict["episode"]
        curr_tasks=net_dict["tasks"]
    else:
        curr_steps = curr_episodes = curr_tasks= 0

    update_done = True
    best_perf =0
    last_test_t = -RecordingParameters.EVAL_INTERVAL - 1
    last_model_t = -RecordingParameters.SAVE_INTERVAL - 1
    last_best_t = -RecordingParameters.BEST_INTERVAL - 1
    last_gif_t = -RecordingParameters.GIF_INTERVAL - 1
    clear_flag=False

    global_perf = {"success_time": 0, "num_iteration": [],
                   "task_global_collision": [], "destroy_weight_target": [],
                   "destroy_weight_collision": [], "destroy_weight_random": [],"makespan":[]}

    # start training
    try:
        while curr_steps < TrainingParameters.N_MAX_STEPS:
            if update_done:
                # start a data collection
                if global_device != local_device:
                    net_weights = global_model.network.to(local_device).state_dict()
                    global_model.network.to(global_device)
                else:
                    net_weights = global_model.network.state_dict()

                job_results=envs.run(net_weights,clear_flag)
                clear_flag = False

            # get data from multiple processes

            curr_steps +=  TrainingParameters.N_STEPS
            data_buffer = {"obs": [], "vector": [], "returns": [], "values": [], "action": [], "ps": [],
                           "hidden_state": [], "train_valid": [], "blocking": []}
            performance_dict = {'per_r': [],  'per_valid_rate': [],
                                'per_episode_len': [], 'per_block': [],
                                'per_leave_goal': [], 'per_final_goals': [], 'per_half_goals': [], 'per_block_acc': [],
                                'per_max_goals': [], 'per_num_dynamic_collide': [], "episode_global_collision": []}
            i = 0
            for key in data_buffer.keys():
                data_buffer[key].append(job_results[i])
                i += 1
            curr_episodes += job_results[-3]
            for key in performance_dict.keys():
                performance_dict[key].append(np.nanmean(job_results[-2][key]))
            curr_tasks+=len(job_results[-1]["num_iteration"])
            if curr_tasks>0:
                for key in global_perf.keys():
                    if key=="success_time":
                        global_perf[key]+=job_results[-1][key]
                    else:
                        global_perf[key].append(np.nanmean(job_results[-1][key]))

            if curr_tasks%10==0 and curr_tasks!=0:
                clear_flag = True
                if RecordingParameters.WANDB:
                    global_write_to_wandb(curr_steps, global_perf)
                # if RecordingParameters.TENSORBOARD:
                #     global_write_to_tensorboard(global_summary, curr_steps, global_perf)
                print('episodes: {}, steps: {}, tasks:{},success rate:{} \n'.format(
                    curr_episodes, curr_steps, curr_tasks,global_perf["success_time"]/10))
                global_perf = {"success_time": 0, "num_iteration": [],
                               "task_global_collision": [], "destroy_weight_target": [],
                               "destroy_weight_collision": [], "destroy_weight_random": [],"makespan":[]}

            # training of reinforcement learning
            mb_loss = []
            inds = np.arange( TrainingParameters.N_STEPS)
            for _ in range(TrainingParameters.N_EPOCHS):
                np.random.shuffle(inds)
                for start in range(0, TrainingParameters.N_STEPS, TrainingParameters.MINIBATCH_SIZE):
                    end = start + TrainingParameters.MINIBATCH_SIZE
                    mb_inds = inds[start:end]
                    slices = (arr[mb_inds] for arr in
                              (data_buffer["obs"], data_buffer["vector"],data_buffer["returns"],data_buffer["values"],
                                   data_buffer["action"], data_buffer["ps"],data_buffer["hidden_state"],
                                   data_buffer["train_valid"], data_buffer["blocking"]))
                    mb_loss.append(global_model.train(*slices))

            # record training result
            if RecordingParameters.WANDB:
                write_to_wandb(curr_steps, performance_dict, mb_loss, evaluate=False)
            # if RecordingParameters.TENSORBOARD:
            #     write_to_tensorboard(global_summary, curr_steps, performance_dict, mb_loss, evaluate=False)
            if RecordingParameters.EVAL:
                pass

            # save model
            if (curr_steps - last_model_t) / RecordingParameters.SAVE_INTERVAL >= 1.0:
                last_model_t = curr_steps
                print('Saving Model !\n')
                model_path = osp.join(RecordingParameters.MODEL_PATH, '%.5i' % curr_steps)
                os.makedirs(model_path)
                path_checkpoint = model_path + "/net_checkpoint.pkl"
                net_checkpoint = {"model": global_model.network.state_dict(),
                                  "optimizer": global_model.net_optimizer.state_dict(),
                                  "step": curr_steps,
                                  "episode": curr_episodes,
                                  "tasks":curr_tasks}
                torch.save(net_checkpoint, path_checkpoint)

    except KeyboardInterrupt:
        print("CTRL-C pressed. killing remote workers")
    finally:
        # save final model
        print('Saving Final Model !\n')
        model_path = RecordingParameters.MODEL_PATH + '/final'
        os.makedirs(model_path)
        path_checkpoint = model_path + "/net_checkpoint.pkl"
        net_checkpoint = {"model": global_model.network.state_dict(),
                          "optimizer": global_model.net_optimizer.state_dict(),
                          "step": curr_steps,
                          "episode": curr_episodes,
                          "tasks":curr_tasks}
        torch.save(net_checkpoint, path_checkpoint)
        # global_summary.close()
        # killing
        if RecordingParameters.WANDB:
            wandb.finish()

if __name__ == "__main__":
    main()
