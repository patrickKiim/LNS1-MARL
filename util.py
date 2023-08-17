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


def write_to_wandb(step, performance_dict=None, mb_loss=None):
    """record performance using wandb"""
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
    wandb.log({'Perf/Num_update': performance_dict['per_num_update_path']},
              step=step)
    wandb.log({'Perf/Reduced_collision': performance_dict["per_reduced_collide"]},
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


def init_performance_dict_driver():
    performance_dict = {'per_r': [], 'per_valid_rate': [],
                        'per_episode_len': [], 'per_block': [],
                        'per_leave_goal': [], 'per_final_goals': [], 'per_half_goals': [], 'per_block_acc': [],
                        'per_max_goals': [], 'per_num_dynamic_collide': [], 'per_num_agent_collide': [],
                        "episode_global_collision": [],
                        'per_num_update_path': [], 'per_reduced_collide': []}
    return performance_dict


def init_performance_dict_runner():
    performance_dict = {'per_r': [], 'per_valid_rate': [],
                        'per_episode_len': [], 'per_block': [],
                        'per_leave_goal': [], 'per_final_goals': [], 'per_half_goals': [], 'per_block_acc': [],
                        'per_max_goals': [], 'per_num_dynamic_collide': [], 'per_num_agent_collide': [],
                        "episode_global_collision": [],
                        'per_num_update_path': 0, 'per_reduced_collide': 0}
    return performance_dict


def init_global_perf():
    global_perf = {"success_time": 0, "num_iteration": [],
                   "task_global_collision": [], "destroy_weight_target": [],
                   "destroy_weight_collision": [], "destroy_weight_random": [], "makespan": []}
    return global_perf


def init_one_episode_perf():
    one_episode_perf = {'num_step': 0, 'reward': 0, 'invalid': 0, 'block': 0, 'num_leave_goal': 0,
                        'wrong_blocking': 0, 'num_dynamic_collide': 0, "num_agent_collide": 0}
    return one_episode_perf

def eval_init_global_perf():
    global_perf_100, global_perf_200, global_perf_300, global_perf_400 = {}, {}, {}, {},
    dict_set = [global_perf_100, global_perf_200, global_perf_300, global_perf_400]
    item_name = ["success_time", "global_collision", "num_update", "makespan"]
    for dict_name in dict_set:
        for item in item_name:
            dict_name[item] = 0
    return dict_set

def write_to_wandb_global(step, performance_dict,global_performance_dict):

    wandb.log({'Perf_global_eval/Reward': np.nanmean(performance_dict['reward'])}, step=step)
    wandb.log({'Perf_global_eval/Invalid': np.nanmean(performance_dict['invalid'])}, step=step)
    wandb.log({'Perf_global_eval/Episode_length': np.nanmean(performance_dict['num_step'])}, step=step)
    wandb.log({'Perf_global_eval/Num_block': np.nanmean(performance_dict['block'])}, step=step)
    wandb.log({'Perf_global_eval/Final_goals': np.nanmean(performance_dict['final_goal'])}, step=step)
    wandb.log({'Perf_global_eval/Num_dynamic_collide': np.nanmean(performance_dict['num_dynamic_collide'])},
              step=step)
    wandb.log({'Perf_global_eval/Num_agent_collide': np.nanmean(performance_dict['num_agent_collide'])},
              step=step)
    wandb.log({'Perf_global_eval/Reduced_collision': np.nanmean(performance_dict["reduced_collision"])},step=step)

    wandb.log({'Perf_100/Success': global_performance_dict[0]["success_time"]}, step=step)
    wandb.log({'Perf_100/Global_collision': global_performance_dict[0]["global_collision"]}, step=step)
    wandb.log({'Perf_100/Num_update': global_performance_dict[0]["num_update"]}, step=step)
    wandb.log({'Perf_100/Makespan': global_performance_dict[0]["makespan"]}, step=step)

    wandb.log({'Perf_200/Success': global_performance_dict[1]["success_time"]}, step=step)
    wandb.log({'Perf_200/Global_collision': global_performance_dict[1]["global_collision"]}, step=step)
    wandb.log({'Perf_200/Num_update': global_performance_dict[1]["num_update"]}, step=step)
    wandb.log({'Perf_200/Makespan': global_performance_dict[1]["makespan"]}, step=step)

    wandb.log({'Perf_300/Success': global_performance_dict[2]["success_time"]}, step=step)
    wandb.log({'Perf_300/Global_collision': global_performance_dict[2]["global_collision"]}, step=step)
    wandb.log({'Perf_300/Num_update': global_performance_dict[2]["num_update"]}, step=step)
    wandb.log({'Perf_300/Makespan': global_performance_dict[2]["makespan"]}, step=step)

    wandb.log({'Perf_400/Success': global_performance_dict[3]["success_time"]}, step=step)
    wandb.log({'Perf_400/Global_collision': global_performance_dict[3]["global_collision"]}, step=step)
    wandb.log({'Perf_400/Num_update': global_performance_dict[3]["num_update"]}, step=step)
    wandb.log({'Perf_400/Makespan': global_performance_dict[3]["makespan"]}, step=step)

def write_to_wandb_scalar(step, performance_dict):
    wandb.log({'Perf_scalar/Reward': performance_dict['reward']}, step=step)
    wandb.log({'Perf_scalar/Invalid': performance_dict['invalid']}, step=step)
    wandb.log({'Perf_scalar/Episode_length': performance_dict['num_step']}, step=step)
    wandb.log({'Perf_scalar/Num_block': performance_dict['block']}, step=step)
    wandb.log({'Perf_scalar/Final_goals': performance_dict['final_goal']}, step=step)
    wandb.log({'Perf_scalar/Num_dynamic_collide': performance_dict['num_dynamic_collide']},
              step=step)
    wandb.log({'Perf_scalar/Num_agent_collide': performance_dict['num_agent_collide']},
              step=step)
    wandb.log({'Perf_scalar/Reduced_collision': performance_dict["reduced_collison"]},step=step)


