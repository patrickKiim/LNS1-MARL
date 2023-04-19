import random

import imageio
import numpy as np
import torch
import wandb

from alg_parameters import *


def set_global_seeds(i):
    """set seed for fair comparison"""
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)
    torch.cuda.manual_seed_all(i)
    np.random.seed(i)
    random.seed(i)
    torch.backends.cudnn.deterministic = True

def global_write_to_wandb(curr_steps, global_perf,num_task):
    wandb.log({'Perf_global/Success_rate': global_perf["success_time"]/num_task}, step=curr_steps)
    wandb.log({'Perf_global/Iteration_times': np.nanmean(global_perf["num_iteration"])}, step=curr_steps)
    wandb.log({'Perf_global/Num_collision': np.nanmean(global_perf["task_global_collision"]) }, step=curr_steps)
    wandb.log({'Perf_global/Weight_target': np.nanmean(global_perf["destroy_weight_target"] )}, step=curr_steps)
    wandb.log({'Perf_global/Weight_collision': np.nanmean(global_perf["destroy_weight_collision"] )}, step=curr_steps)
    wandb.log({'Perf_global/Weight_random': np.nanmean(global_perf["destroy_weight_random"])}, step=curr_steps)
    wandb.log({'Perf_global/Makespan': np.nanmean(global_perf["makespan"])}, step=curr_steps)


def write_to_wandb(step, performance_dict=None, mb_loss=None, evaluate=True, greedy=True):
    """record performance using wandb"""
    if evaluate:
        pass
        # if greedy:
        #     wandb.log({'Perf_greedy_eval/Reward': performance_dict['per_r']}, step=step)
        #     wandb.log({'Perf_greedy_eval/Valid_rate': performance_dict['per_valid_rate']}, step=step)
        #     wandb.log({'Perf_greedy_eval/Episode_length': performance_dict['per_episode_len']}, step=step)
        #     wandb.log({'Perf_greedy_eval/Num_block': performance_dict['per_block']}, step=step)
        #     wandb.log({'Perf_greedy_eval/Num_leave_goal': performance_dict['per_leave_goal']}, step=step)
        #     wandb.log({'Perf_greedy_eval/Final_goals': performance_dict['per_final_goals']}, step=step)
        #     wandb.log({'Perf_greedy_eval/Half_goals': performance_dict['per_half_goals']}, step=step)
        #     wandb.log({'Perf_greedy_eval/Block_accuracy': performance_dict['per_block_acc']}, step=step)
        #     wandb.log({'Perf_greedy_eval/Max_goals': performance_dict['per_max_goals']}, step=step)
        #     wandb.log({'Perf_greedy_eval/Num_collide': performance_dict['per_num_collide']}, step=step)
        #
        # else:
        #     wandb.log({'Perf_random_eval/Reward': performance_dict['per_r']}, step=step)
        #     wandb.log({'Perf_random_eval/Valid_rate': performance_dict['per_valid_rate']}, step=step)
        #     wandb.log({'Perf_random_eval/Episode_length': performance_dict['per_episode_len']}, step=step)
        #     wandb.log({'Perf_random_eval/Num_block': performance_dict['per_block']}, step=step)
        #     wandb.log({'Perf_random_eval/Num_leave_goal': performance_dict['per_leave_goal']}, step=step)
        #     wandb.log({'Perf_random_eval/Final_goals': performance_dict['per_final_goals']}, step=step)
        #     wandb.log({'Perf_random_eval/Half_goals': performance_dict['per_half_goals']}, step=step)
        #     wandb.log({'Perf_random_eval/Block_accuracy': performance_dict['per_block_acc']}, step=step)
        #     wandb.log({'Perf_random_eval/Max_goals': performance_dict['per_max_goals']}, step=step)
        #     wandb.log({'Perf_random_eval/Num_collide': performance_dict['per_num_collide']}, step=step)

    else:
        loss_vals = np.nanmean(mb_loss, axis=0)
        wandb.log({'Perf/Reward': performance_dict['per_r']}, step=step)
        wandb.log({'Perf/Valid_rate': performance_dict['per_valid_rate']}, step=step)
        wandb.log({'Perf/Episode_length': performance_dict['per_episode_len']}, step=step)
        wandb.log({'Perf/Num_block': performance_dict['per_block']}, step=step)
        wandb.log({'Perf/Num_leave_goal': performance_dict['per_leave_goal']}, step=step)
        wandb.log({'Perf/Final_goals': performance_dict['per_final_goals']}, step=step)
        wandb.log({'Perf/Half_goals': performance_dict['per_half_goals']}, step=step)
        wandb.log({'Perf/Block_accuracy': performance_dict['per_block_acc']}, step=step)
        wandb.log({'Perf/Max_goals': performance_dict['per_max_goals']}, step=step)
        wandb.log({'Perf/Num_dynamic_collide': performance_dict['per_num_dynamic_collide']},
                  step=step)
        wandb.log({'Perf/Num_agent_collide': performance_dict['per_num_agent_collide']},
                  step=step)
        wandb.log({'Perf/Ep_global_collision': performance_dict["episode_global_collision"]},
                  step=step)

        for (val, name) in zip(loss_vals, RecordingParameters.LOSS_NAME):
            if name == 'grad_norm':
                wandb.log({'Grad/' + name: val}, step=step)
            else:
                wandb.log({'Loss/' + name: val}, step=step)


def make_gif(images, file_name):
    """record gif"""
    imageio.mimwrite(file_name, images, subrectangles=True)
    print("wrote gif")

def local_reset_env(env,local_num_agents,local_succ,first_time, ALNS):
    global_done,destroy_weights,global_num_collison, makespan=env._local_reset(local_num_agents, local_succ, first_time, ALNS)
    if global_done:
        assert (global_num_collison==0)
        return True,destroy_weights,global_num_collison,False,0,0,0,0, makespan

    local_done=False
    prev_action = np.zeros(EnvParameters.LOCAL_N_AGENTS)
    valid_actions = []
    obs = np.zeros((1, EnvParameters.LOCAL_N_AGENTS, NetParameters.NUM_CHANNEL, EnvParameters.FOV_SIZE, EnvParameters.FOV_SIZE),
                   dtype=np.float32)
    vector = np.zeros((1, EnvParameters.LOCAL_N_AGENTS, NetParameters.VECTOR_LEN), dtype=np.float32)
    train_valid = np.zeros((EnvParameters.LOCAL_N_AGENTS, EnvParameters.N_ACTIONS), dtype=np.float32)

    for i in range(EnvParameters.LOCAL_N_AGENTS):
        valid_action = env.list_next_valid_actions(i)
        s = env.observe(i)
        obs[:, i, :, :, :] = s[0]
        vector[:, i, : 3] = s[1]
        vector[:, i, -1] = prev_action[i]
        valid_actions.append(valid_action)
        train_valid[i, valid_action] = 1
    return global_done,destroy_weights,global_num_collison,local_done, valid_actions, obs, vector, train_valid, makespan

def global_reset_env(env):
    """reset environment"""
    can_not_use=True
    while can_not_use:
        can_not_use, global_num_collision = env._global_reset()
    return global_num_collision

def one_step(env, one_episode_perf, actions, pre_block):
    """run one step"""
    train_valid = np.zeros((EnvParameters.LOCAL_N_AGENTS, EnvParameters.N_ACTIONS), dtype=np.float32)

    obs, vector, rewards, done, next_valid_actions,  blockings, num_blockings, leave_goals, \
        num_on_goal, max_on_goal, num_dynamic_collide, num_agent_collide \
        = env.joint_step(actions)

    one_episode_perf['block'] += num_blockings
    one_episode_perf['num_leave_goal'] += leave_goals
    one_episode_perf['num_dynamic_collide'] += num_dynamic_collide
    one_episode_perf['num_agent_collide'] += num_agent_collide

    for i in range(EnvParameters.LOCAL_N_AGENTS):
        train_valid[i, next_valid_actions[i]] = 1
        if (pre_block[i] < 0.5) == blockings[:, i]:
            one_episode_perf['wrong_blocking'] += 1
    one_episode_perf['num_step'] += 1
    return rewards, next_valid_actions, obs, vector, train_valid, done, blockings, num_on_goal, one_episode_perf, \
        max_on_goal


def update_perf(one_episode_perf, performance_dict, num_on_goals, max_on_goals):
    """record batch performance"""
    performance_dict['per_r'].append(one_episode_perf['reward'])
    performance_dict['per_valid_rate'].append(
        ((one_episode_perf['num_step'] * EnvParameters.LOCAL_N_AGENTS) - one_episode_perf['invalid']) / (
                one_episode_perf['num_step'] * EnvParameters.LOCAL_N_AGENTS))
    performance_dict['per_episode_len'].append(one_episode_perf['num_step'])
    performance_dict['per_block'].append(one_episode_perf['block'])
    performance_dict['per_leave_goal'].append(one_episode_perf['num_leave_goal'])
    performance_dict['per_num_dynamic_collide'].append(one_episode_perf['num_dynamic_collide'])
    performance_dict['per_num_agent_collide'].append(one_episode_perf['num_agent_collide'])
    performance_dict['per_final_goals'].append(num_on_goals)
    performance_dict['per_block_acc'].append(
        ((one_episode_perf['num_step'] * EnvParameters.LOCAL_N_AGENTS) - one_episode_perf['wrong_blocking']) / (
                one_episode_perf['num_step'] * EnvParameters.LOCAL_N_AGENTS))
    performance_dict['per_max_goals'].append(max_on_goals)
    return performance_dict
