import os
import os.path as osp

import numpy as np
import ray
import setproctitle
# from torch.utils.tensorboard import SummaryWriter
import torch
import wandb

from alg_parameters import *
from mapf_gym import MAPFEnv
from model import Model
from runner import Runner
from util import set_global_seeds,  write_to_wandb, make_gif,global_write_to_wandb, one_step, update_perf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ray.init(num_gpus=SetupParameters.NUM_GPU)
print("Welcome to LNS2 MAPF!\n")


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

    if not os.path.exists("./record_files"):
        os.makedirs("./record_files")

    # create classes
    global_device = torch.device('cuda') if SetupParameters.USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if SetupParameters.USE_GPU_LOCAL else torch.device('cpu')
    global_model = Model(0, global_device, True)

    if RecordingParameters.WANDB:
        wandb.watch(global_model.network, log_freq=RecordingParameters.GRAD_LOGFREQ, log_graph=True)

    if RecordingParameters.RETRAIN:
        global_model.network.load_state_dict(net_dict['model'])
        global_model.net_optimizer.load_state_dict(net_dict['optimizer'])

    envs = [Runner.remote(i + 1) for i in range(TrainingParameters.N_ENVS)]
    # eval_env = MAPFEnv()

    if RecordingParameters.RETRAIN:
        curr_steps = net_dict["step"]
        curr_episodes = net_dict["episode"]
        curr_tasks=net_dict["tasks"]
    else:
        curr_steps = curr_episodes = curr_tasks= 0

    update_done = True
    best_perf =0
    job_list = []
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
                net_weights_id = ray.put(net_weights)
                clear_flag_id = ray.put(clear_flag)

                for i, env in enumerate(envs):
                    job_list.append(env.run.remote(net_weights_id,clear_flag_id))
                clear_flag = False

            # get data from multiple processes
            done_id, job_list = ray.wait(job_list, num_returns=TrainingParameters.N_ENVS)
            update_done = True if job_list == [] else False
            done_len = len(done_id)
            job_results = ray.get(done_id)

            curr_steps += done_len * TrainingParameters.N_STEPS
            data_buffer = {"obs": [], "vector": [], "returns": [], "values": [], "action": [], "ps": [],
                           "hidden_state": [], "train_valid": [], "blocking": []}
            performance_dict = {'per_r': [],  'per_valid_rate': [],
                                'per_episode_len': [], 'per_block': [],
                                'per_leave_goal': [], 'per_final_goals': [], 'per_half_goals': [], 'per_block_acc': [],
                                'per_max_goals': [], 'per_num_dynamic_collide': [],'per_num_agent_collide': [], "episode_global_collision": []}
            temp_tasks = 0
            for results in range(done_len):
                i = 0
                for key in data_buffer.keys():
                    data_buffer[key].append(job_results[results][i])
                    i += 1
                curr_episodes += job_results[results][-3]
                for key in performance_dict.keys():
                    performance_dict[key].append(np.nanmean(job_results[results][-2][key]))
                temp_tasks+=len(job_results[results][-1]["num_iteration"])

            for key in data_buffer.keys():
                data_buffer[key] = np.concatenate(data_buffer[key], axis=0)

            for key in performance_dict.keys():
                performance_dict[key] = np.nanmean(performance_dict[key])

            if temp_tasks>=RecordingParameters.GLOBAL_INTERVAL:
                curr_tasks+=temp_tasks
                clear_flag = True
                for results in range(done_len):
                    for key in global_perf.keys():
                        if key == "success_time":
                            global_perf[key]+=job_results[results][-1][key]
                        else:
                            global_perf[key].append(np.nanmean(job_results[results][-1][key]))

                if RecordingParameters.WANDB:
                    global_write_to_wandb(curr_steps, global_perf,temp_tasks)
                # if RecordingParameters.TENSORBOARD:
                #     global_write_to_tensorboard(global_summary, curr_steps, global_perf)
                print('episodes: {}, steps: {}, tasks:{},success rate:{} \n'.format(
                    curr_episodes, curr_steps, curr_tasks,global_perf["success_time"]/temp_tasks))
                global_perf = {"success_time": 0, "num_iteration": [],
                               "task_global_collision": [], "destroy_weight_target": [],
                               "destroy_weight_collision": [], "destroy_weight_random": [],"makespan":[]}

            # training of reinforcement learning
            mb_loss = []
            inds = np.arange(done_len * TrainingParameters.N_STEPS)
            for _ in range(TrainingParameters.N_EPOCHS):
                np.random.shuffle(inds)
                for start in range(0, done_len * TrainingParameters.N_STEPS, TrainingParameters.MINIBATCH_SIZE):
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
                # if (curr_steps - last_test_t) / RecordingParameters.EVAL_INTERVAL >= 1.0:
                #     # if save gif
                #     if (curr_steps - last_gif_t) / RecordingParameters.GIF_INTERVAL >= 1.0:
                #         save_gif = True
                #         last_gif_t = curr_steps
                #     else:
                #         save_gif = False
                #
                #     # evaluate training model
                #     last_test_t = curr_steps
                #     with torch.no_grad():
                #         # greedy_eval_performance_dict = evaluate(eval_env,eval_memory, global_model,
                #         # global_device, save_gif, curr_steps, True)
                #         eval_performance_dict = evaluate(eval_env, global_model, global_device, save_gif,
                #                                          curr_steps, False)
                #     # record evaluation result
                #     if RecordingParameters.WANDB:
                #         # write_to_wandb(curr_steps, greedy_eval_performance_dict, evaluate=True, greedy=True)
                #         write_to_wandb(curr_steps, eval_performance_dict, evaluate=True, greedy=False)
                #     if RecordingParameters.TENSORBOARD:
                #         # write_to_tensorboard(global_summary, curr_steps, greedy_eval_performance_dict, evaluate=True,
                #         #                      greedy=True)
                #         write_to_tensorboard(global_summary, curr_steps, eval_performance_dict, evaluate=True, greedy=False,
                #                              )
                #
                #     print('episodes: {}, step: {},episode reward: {}, final goals: {} \n'.format(
                #         curr_episodes, curr_steps, eval_performance_dict['per_r'],
                #         eval_performance_dict['per_final_goals']))
                #     # save model with the best performance
                #     if RecordingParameters.RECORD_BEST:
                #         if eval_performance_dict['per_r'] > best_perf and (
                #                 curr_steps - last_best_t) / RecordingParameters.BEST_INTERVAL >= 1.0:
                #             best_perf = eval_performance_dict['per_r']
                #             last_best_t = curr_steps
                #             print('Saving best model \n')
                #             model_path = osp.join(RecordingParameters.MODEL_PATH, 'best_model')
                #             if not os.path.exists(model_path):
                #                 os.makedirs(model_path)
                #             path_checkpoint = model_path + "/net_checkpoint.pkl"
                #             net_checkpoint = {"model": global_model.network.state_dict(),
                #                               "optimizer": global_model.net_optimizer.state_dict(),
                #                               "step": curr_steps,
                #                               "episode": curr_episodes,
                #                               "reward": best_perf}
                #             torch.save(net_checkpoint, path_checkpoint)

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
        # killing
        for e in envs:
            ray.kill(e)
        if RecordingParameters.WANDB:
            wandb.finish()


def evaluate(eval_env, model, device, save_gif, curr_steps, greedy):
    pass
    """Evaluate Model."""
    # eval_performance_dict = {'per_r': [], 'per_valid_rate': [], 'per_episode_len': [],
    #                          'per_block': [], 'per_leave_goal': [], 'per_final_goals': [], 'per_half_goals': [],
    #                          'per_block_acc': [], 'per_max_goals': [], 'per_num_collide': []}
    # episode_frames = []
    #
    # for i in range(RecordingParameters.EVAL_EPISODES):
    #     num_agent = EnvParameters.N_AGENTS
    #
    #     # reset environment and buffer
    #     hidden_state = (torch.zeros((num_agent, NetParameters.NET_SIZE)).to(device),
    #                     torch.zeros((num_agent, NetParameters.NET_SIZE)).to(device))
    #
    #     done, valid_actions, obs, vector, _ = reset_env(eval_env, num_agent)
    #
    #     one_episode_perf = {'num_step': 0, 'episode_reward': 0, 'invalid': 0, 'block': 0,
    #                         'num_leave_goal': 0, 'wrong_blocking': 0, 'num_collide': 0}
    #     if save_gif:
    #         episode_frames.append(eval_env._render(mode='rgb_array', screen_width=900, screen_height=900))
    #
    #     # stepping
    #     while not done:
    #         # predict
    #         actions, pre_block, hidden_state, num_invalid, v, ps = model.evaluate(obs, vector,
    #                                                                                            valid_actions,
    #                                                                                            hidden_state,
    #                                                                                            greedy,num_agent)
    #         one_episode_perf['invalid'] += num_invalid
    #
    #         # move
    #         rewards, valid_actions, obs, vector, _, done, _, num_on_goals, one_episode_perf, max_on_goals, \
    #             _, _, on_goal = one_step(eval_env, one_episode_perf, actions, pre_block, model, v, hidden_state,
    #                                      ps,  num_agent)
    #
    #
    #         if save_gif:
    #             episode_frames.append(eval_env._render(mode='rgb_array', screen_width=900, screen_height=900))
    #
    #         one_episode_perf['episode_reward'] += np.sum(rewards)
    #         if one_episode_perf['num_step'] == EnvParameters.EPISODE_LEN // 2:
    #             eval_performance_dict['per_half_goals'].append(num_on_goals)
    #
    #         if done:
    #             # save gif
    #             if save_gif:
    #                 if not os.path.exists(RecordingParameters.GIFS_PATH):
    #                     os.makedirs(RecordingParameters.GIFS_PATH)
    #                 images = np.array(episode_frames)
    #                 make_gif(images,
    #                          '{}/steps_{:d}_reward{:.1f}_final_goals{:.1f}_greedy{:d}.gif'.format(
    #                              RecordingParameters.GIFS_PATH,
    #                              curr_steps, one_episode_perf[
    #                                  'episode_reward'],
    #                              num_on_goals, greedy))
    #                 save_gif = False
    #
    #             eval_performance_dict = update_perf(one_episode_perf, eval_performance_dict, num_on_goals, max_on_goals,
    #                                                 num_agent)
    #
    # # average performance of multiple episodes
    # for i in eval_performance_dict.keys():
    #     eval_performance_dict[i] = np.nanmean(eval_performance_dict[i])
    #
    # return eval_performance_dict


if __name__ == "__main__":
    main()
