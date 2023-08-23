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
# from eval_model import Eval

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ray.init(num_gpus=SetupParameters.NUM_GPU)
print("Welcome to LNS2 MAPF!\n")


def main():
    """main code"""
    # preparing for training
    if RecordingParameters.RETRAIN:
        restore_path = '/home/marmot/Yutong/MAPF/training/cl_step_block_individual_ifbetter/models/MAPF/sipps_block_individual_if_better19-08-231805/final'
        net_path_checkpoint = restore_path + "/net_checkpoint.pkl"
        net_dict = torch.load(net_path_checkpoint)

    if RecordingParameters.WANDB:
        if RecordingParameters.RETRAIN:
            wandb_id = '1rpw7p7d'
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

    # if RecordingParameters.WANDB:
    #     wandb.watch(global_model.network, log_freq=RecordingParameters.GRAD_LOGFREQ, log_graph=True)

    if RecordingParameters.RETRAIN:
        global_model.network.load_state_dict(net_dict['model'])
        global_model.net_optimizer.load_state_dict(net_dict['optimizer'])

    envs = [RLRunner.remote(i + 1) for i in range(TrainingParameters.N_ENVS)]
    # eval_env = Eval()

    if RecordingParameters.RETRAIN:
        curr_steps = net_dict["step"]
        curr_episodes = net_dict["episode"]
    else:
        curr_steps = curr_episodes = 0

    update_done = True
    best_perf =0
    job_list = []
    last_test_t_scala = -RecordingParameters.EVAL_INTERVAL_SCALA - 1
    last_test_t_global = -RecordingParameters.EVAL_INTERVAL_GLOBAL - 1
    last_model_t = -RecordingParameters.SAVE_INTERVAL - 1
    last_best_t = -RecordingParameters.BEST_INTERVAL - 1
    last_gif_t = -RecordingParameters.GIF_INTERVAL - 1
    clear_flag=False
    cl_task_num=0
    prev_cl_task_num=0

    global_perf = init_global_perf()

    # start training
    try:
        while curr_steps < TrainingParameters.N_MAX_STEPS:
            if update_done:
                # start a data collection
                switch_task = False
                if global_device != local_device:
                    net_weights = global_model.network.to(local_device).state_dict()
                    global_model.network.to(global_device)
                else:
                    net_weights = global_model.network.state_dict()
                net_weights_id = ray.put(net_weights)

                if curr_steps>=EnvParameters.SWEITCH_TIMESTEP[0] and curr_steps<EnvParameters.SWEITCH_TIMESTEP[1]:
                    cl_task_num = 1
                if curr_steps>=EnvParameters.SWEITCH_TIMESTEP[1] and curr_steps<EnvParameters.SWEITCH_TIMESTEP[2]:
                    cl_task_num = 2
                if curr_steps>=EnvParameters.SWEITCH_TIMESTEP[2] and curr_steps<EnvParameters.SWEITCH_TIMESTEP[3]:
                    cl_task_num = 3
                if curr_steps>=EnvParameters.SWEITCH_TIMESTEP[3] and curr_steps<EnvParameters.SWEITCH_TIMESTEP[4]:
                    cl_task_num = 4
                if curr_steps>=EnvParameters.SWEITCH_TIMESTEP[4]:
                    cl_task_num = 5
                if cl_task_num-prev_cl_task_num>0:
                    switch_task=True
                    print("switch to task {}".format(cl_task_num))
                prev_cl_task_num=cl_task_num

                cl_task_num_id= ray.put(cl_task_num)
                switch_task_id= ray.put(switch_task)

                if cl_task_num<3:
                    for i, env in enumerate(envs):
                        job_list.append(env.cl_run.remote(net_weights_id,cl_task_num_id,switch_task_id))
                else:
                    clear_flag_id = ray.put(clear_flag)
                    for i, env in enumerate(envs):
                        job_list.append(env.run.remote(net_weights_id,clear_flag_id,cl_task_num_id,switch_task_id))
                    clear_flag = False

            # get data from multiple processes
            done_id, job_list = ray.wait(job_list, num_returns=TrainingParameters.N_ENVS)
            update_done = True if job_list == [] else False
            done_len = len(done_id)
            job_results = ray.get(done_id)

            data_buffer = {"obs": [], "vector": [], "returns": [], "values": [], "action": [], "ps": [],
                           "hidden_state": [], "train_valid": [], "blocking": []}
            performance_dict =init_performance_dict_driver()
            curr_episodes += done_len
            temp_tasks = 0
            tem_steps = 0
            for results in range(done_len):
                i = 0
                for key in data_buffer.keys():
                    data_buffer[key].append(job_results[results][i])
                    i += 1
                curr_steps +=job_results[results][-3]
                tem_steps+=job_results[results][-3]
                for key in performance_dict.keys():
                    performance_dict[key].append(job_results[results][-2][key])
                temp_tasks+=len(job_results[results][-1]["num_iteration"])

            for key in data_buffer.keys():
                data_buffer[key] = np.concatenate(data_buffer[key], axis=0)

            for key in performance_dict.keys():
                performance_dict[key] = np.nanmean(performance_dict[key])

            if temp_tasks>=RecordingParameters.GLOBAL_INTERVAL:
                clear_flag = True
                for results in range(done_len):
                    for key in global_perf.keys():
                        if key == "success_time":
                            global_perf[key]+=job_results[results][-1][key]
                        else:
                            global_perf[key].append(np.nanmean(job_results[results][-1][key]))

                if RecordingParameters.WANDB:
                    global_write_to_wandb(curr_steps, global_perf,temp_tasks)

                print('episodes: {}, steps: {},success rate:{} \n'.format(
                    curr_episodes, curr_steps,global_perf["success_time"]/temp_tasks))
                global_perf = init_global_perf()

            # training of reinforcement learning
            mb_loss = []
            inds = np.arange(tem_steps)
            for _ in range(TrainingParameters.N_EPOCHS):
                np.random.shuffle(inds)
                for start in range(0, tem_steps, TrainingParameters.MINIBATCH_SIZE):
                    end = start + TrainingParameters.MINIBATCH_SIZE
                    if end > tem_steps:
                        end=tem_steps
                    mb_inds = inds[start:end]
                    slices = (arr[mb_inds] for arr in
                              (data_buffer["obs"], data_buffer["vector"],data_buffer["returns"],data_buffer["values"],
                                   data_buffer["action"], data_buffer["ps"],data_buffer["hidden_state"],
                                   data_buffer["train_valid"], data_buffer["blocking"]))
                    mb_loss.append(global_model.train(*slices))

            # record training result
            if RecordingParameters.WANDB:
                write_to_wandb(curr_steps, performance_dict, mb_loss)

            # if RecordingParameters.EVAL and curr_steps>=2e7:
            #     if (curr_steps - last_test_t_scala) / RecordingParameters.EVAL_INTERVAL_SCALA >= 1.0:
            #         last_test_t_scala = curr_steps
            #         scalar_eval_perf = eval_env.eval_scalar(global_model)
            #         if RecordingParameters.WANDB:
            #             write_to_wandb_scalar(curr_steps, scalar_eval_perf)
            #
            #     if (curr_steps - last_test_t_global) / RecordingParameters.EVAL_INTERVAL_GLOBAL >= 1.0:
            #         last_test_t_global = curr_steps
            #         global_one_episode_perf, global_eval_perf = eval_env.eval_global(global_model)
            #         if RecordingParameters.WANDB:
            #             write_to_wandb_global(curr_steps, global_one_episode_perf, global_eval_perf)
            #
            #         if RecordingParameters.RECORD_BEST:
            #             pass

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
                                  "episode": curr_episodes}
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
                          "episode": curr_episodes}
        torch.save(net_checkpoint, path_checkpoint)
        # killing
        for e in envs:
            ray.kill(e)
        # global_one_episode_perf, global_eval_perf = eval_env.eval_global(global_model)
        # scalar_eval_perf = eval_env.eval_scalar(global_model)
        # if RecordingParameters.WANDB:
        #     write_to_wandb_global(curr_steps, global_one_episode_perf, global_eval_perf)
        #     write_to_wandb_scalar(curr_steps, scalar_eval_perf)
        if RecordingParameters.WANDB:
            wandb.finish()


if __name__ == "__main__":
    main()
