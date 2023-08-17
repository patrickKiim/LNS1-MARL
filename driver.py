import os
import os.path as osp

import numpy as np
import ray
import setproctitle
import torch
import wandb

from alg_parameters import *
from model import Model
from runner import RLRunner
from util import set_global_seeds,  write_to_wandb, global_write_to_wandb,init_global_perf,init_performance_dict_driver,write_to_wandb_global,write_to_wandb_scalar
from eval_model import Eval

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ray.init(num_gpus=SetupParameters.NUM_GPU)
print("Welcome to LNS2 MAPF!\n")


def main():
    """main code"""
    # preparing for training
    if RecordingParameters.RETRAIN:
        restore_path = '/home/marmot/Yutong/new_training_framework_mappo/models/MAPF/mappo_new_frame02-08-231644/final'
        net_path_checkpoint = restore_path + "/net_checkpoint.pkl"
        net_dict = torch.load(net_path_checkpoint)

    if RecordingParameters.WANDB:
        if RecordingParameters.RETRAIN:
            wandb_id = '3kbjwju1'
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

    if not os.path.exists("./record_files_imitation"):
        os.makedirs("./record_files_imitation")

    # create classes
    global_device = torch.device('cuda') if SetupParameters.USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if SetupParameters.USE_GPU_LOCAL else torch.device('cpu')
    global_model = Model(0, global_device, True)

    if RecordingParameters.WANDB:
        wandb.watch(global_model.network, log_freq=RecordingParameters.GRAD_LOGFREQ, log_graph=True)

    if RecordingParameters.RETRAIN:
        global_model.network.load_state_dict(net_dict['model'])
        global_model.net_optimizer.load_state_dict(net_dict['optimizer'])

    envs = [RLRunner.remote(i + 1) for i in range(TrainingParameters.N_ENVS)]
    eval_env = Eval()

    if RecordingParameters.RETRAIN:
        curr_steps = net_dict["step"]
        curr_episodes = net_dict["episode"]
        curr_tasks=net_dict["tasks"]
    else:
        curr_steps = curr_episodes = curr_tasks= 0

    update_done = True
    best_perf =0
    job_list = []
    last_test_t_scala = -RecordingParameters.EVAL_INTERVAL_SCALA - 1
    last_test_t_global = -RecordingParameters.EVAL_INTERVAL_GLOBAL - 1
    last_model_t = -RecordingParameters.SAVE_INTERVAL - 1
    last_best_t = -RecordingParameters.BEST_INTERVAL - 1
    last_gif_t = -RecordingParameters.GIF_INTERVAL - 1
    clear_flag=False

    global_perf = init_global_perf()

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

                # for i, env in enumerate(envs):
                #     job_list.append(env.run.remote(net_weights_id,clear_flag_id))
                # clear_flag = False

                demon_probs = np.random.rand()
                if demon_probs < TrainingParameters.DEMONSTRATION_PROB:
                    demon = True
                    for i, env in enumerate(envs):
                            job_list.append(env.imitation.remote(net_weights_id))
                else:
                    demon = False
                    for i, env in enumerate(envs):
                        job_list.append(env.run.remote(net_weights_id, clear_flag_id))
                    clear_flag = False

            # get data from multiple processes
            done_id, job_list = ray.wait(job_list, num_returns=TrainingParameters.N_ENVS)
            update_done = True if job_list == [] else False
            done_len = len(done_id)
            job_results = ray.get(done_id)

            if demon:
                # get imitation learning data
                i_data_buffer = {"obs": [], "vector": [],"action": [], "hidden_state": []}
                
                # temp_tasks = 0
                for results in range(done_len):
                    i = 0
                    for key in i_data_buffer.keys():
                        i_data_buffer[key].append(job_results[results][i])
                        i += 1
                    # if i == 4:
                    curr_episodes += job_results[results][-2]
                    #     i+=1
                    # elif i==5:
                    curr_steps += job_results[results][-1]

                for key in i_data_buffer.keys():
                    i_data_buffer[key] = np.concatenate(i_data_buffer[key], axis=0)

                # training of imitation learning
                imitation_mb_loss = []
                inds = np.arange(done_len * TrainingParameters.N_STEPS)
                for _ in range(TrainingParameters.N_EPOCHS):
                    np.random.shuffle(inds)
                    for start in range(0, done_len * TrainingParameters.N_STEPS, TrainingParameters.MINIBATCH_SIZE):
                        end = start + TrainingParameters.MINIBATCH_SIZE
                        mb_inds = inds[start:end]
                        slices = (arr[mb_inds] for arr in
                                (i_data_buffer["obs"], i_data_buffer["vector"],
                                    i_data_buffer["action"], i_data_buffer["hidden_state"]))
                        imitation_mb_loss.append(global_model.imitation_train(*slices))
                imitation_mb_loss = np.nanmean(imitation_mb_loss, axis=0)

            else:
                curr_steps += done_len * TrainingParameters.N_STEPS
                data_buffer = {"obs": [], "vector": [], "returns": [], "values": [], "action": [], "ps": [],
                            "hidden_state": [], "train_valid": [], "blocking": []}
                performance_dict =init_performance_dict_driver()
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
                    global_perf = init_global_perf()

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
                    write_to_wandb(curr_steps, performance_dict, mb_loss)

            if RecordingParameters.EVAL and curr_steps>=1e7:
                if (curr_steps - last_test_t_scala) / RecordingParameters.EVAL_INTERVAL_SCALA >= 1.0:
                    last_test_t_scala = curr_steps
                    scalar_eval_perf = eval_env.eval_scalar(global_model)
                    if RecordingParameters.WANDB:
                        write_to_wandb_scalar(curr_steps, scalar_eval_perf)

                if (curr_steps - last_test_t_global) / RecordingParameters.EVAL_INTERVAL_GLOBAL >= 1.0:
                    last_test_t_global = curr_steps
                    global_one_episode_perf, global_eval_perf = eval_env.eval_global(global_model)
                    if RecordingParameters.WANDB:
                        write_to_wandb_global(curr_steps, global_one_episode_perf, global_eval_perf)

                    if RecordingParameters.RECORD_BEST:
                        pass

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
        global_one_episode_perf, global_eval_perf = eval_env.eval_global(global_model)
        scalar_eval_perf = eval_env.eval_scalar(global_model)
        if RecordingParameters.WANDB:
            write_to_wandb_global(curr_steps, global_one_episode_perf, global_eval_perf)
            write_to_wandb_scalar(curr_steps, scalar_eval_perf)
        if RecordingParameters.WANDB:
            wandb.finish()


if __name__ == "__main__":
    main()
