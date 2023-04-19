import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from alg_parameters import *
from net import Net


class Model(object):
    """model0 of agents"""

    def __init__(self, env_id, device, global_model=False):
        """initialization"""
        self.ID = env_id
        self.device = device
        self.network = Net().to(device)  # neural network
        if global_model:
            self.net_optimizer = optim.Adam(self.network.parameters(), lr=TrainingParameters.lr)
            # self.multi_gpu_net = torch.nn.DataParallel(self.network) # training on multiple GPU
            self.net_scaler = GradScaler()  # automatic mixed precision

    def step(self, observation, vector, valid_action, input_state):
        """using neural network in training for prediction"""
        num_invalid = 0
        observation = torch.from_numpy(observation).to(self.device)
        vector = torch.from_numpy(vector).to(self.device)
        ps, v, block, _, output_state, _= self.network(observation, vector, input_state)

        actions = np.zeros(EnvParameters.LOCAL_N_AGENTS)
        ps = np.squeeze(ps.cpu().detach().numpy())
        v = v.cpu().detach().numpy()
        block = np.squeeze(block.cpu().detach().numpy())

        for i in range(EnvParameters.LOCAL_N_AGENTS):
            if np.argmax(ps[i], axis=-1) not in valid_action[i]:
                num_invalid += 1
            # choose action from complete action distribution
            valid_dist = np.array([ps[i, valid_action[i]]])
            valid_dist /= np.sum(valid_dist)
            actions[i] = valid_action[i][np.random.choice(range(valid_dist.shape[1]), p=valid_dist.ravel())]
        return actions, ps, v, block, output_state, num_invalid

    def evaluate(self, observation, vector, valid_action, input_state, greedy):
        """using neural network in evaluations of training code for prediction"""
        num_invalid = 0
        eval_action = np.zeros(EnvParameters.LOCAL_N_AGENTS)
        observation = torch.from_numpy(np.asarray(observation)).to(self.device)
        vector = torch.from_numpy(vector).to(self.device)
        ps, v, block, _, output_state, _ = self.network(observation, vector, input_state)
        ps = np.squeeze(ps.cpu().detach().numpy())
        block = np.squeeze(block.cpu().detach().numpy())
        v = v.cpu().detach().numpy()
        greedy_action = np.argmax(ps, axis=-1)

        for i in range(EnvParameters.LOCAL_N_AGENTS):
            if greedy_action[i] not in valid_action[i]:
                num_invalid += 1
            if not greedy:
                valid_dist = np.array([ps[i, valid_action[i]]])
                valid_dist /= np.sum(valid_dist)
                eval_action[i] = valid_action[i][np.random.choice(range(valid_dist.shape[1]), p=valid_dist.ravel())]
        if greedy:
            eval_action = greedy_action
        return eval_action, block, output_state, num_invalid, v, ps

    def final_evaluate(self, observation, vector, valid_action, input_state,local_num_agent):
        """using neural network in evaluations of training code for prediction"""
        eval_action = np.zeros(local_num_agent)
        observation = torch.from_numpy(np.asarray(observation)).to(self.device)
        vector = torch.from_numpy(vector).to(self.device)
        ps, v, _, _, output_state, _ = self.network(observation, vector, input_state)
        ps = np.squeeze(ps.cpu().detach().numpy())
        v = v.cpu().detach().numpy()
        for i in range(EnvParameters.LOCAL_N_AGENTS):
            valid_dist = np.array([ps[i, valid_action[i]]])
            valid_dist /= np.sum(valid_dist)
            eval_action[i] = valid_action[i][np.random.choice(range(valid_dist.shape[1]), p=valid_dist.ravel())]

        return eval_action, output_state, v, ps

    def value(self, obs, vector, input_state):
        """using neural network to predict state values"""
        obs = torch.from_numpy(obs).to(self.device)
        vector = torch.from_numpy(vector).to(self.device)
        _, v, _, _, _, _ = self.network(obs, vector, input_state)
        v = v.cpu().detach().numpy()
        return v

    def train(self, observation, vector, returns, old_v, action,
              old_ps, input_state, train_valid, target_blockings):
        """train model0 by reinforcement learning"""
        self.net_optimizer.zero_grad()
        # from numpy to torch
        observation = torch.from_numpy(observation).to(self.device)
        vector = torch.from_numpy(vector).to(self.device)
        returns = torch.from_numpy(returns).to(self.device)
        old_v = torch.from_numpy(old_v).to(self.device)
        action = torch.from_numpy(action).to(self.device)
        action = torch.unsqueeze(action, -1)
        old_ps = torch.from_numpy(old_ps).to(self.device)
        train_valid = torch.from_numpy(train_valid).to(self.device)
        target_blockings = torch.from_numpy(target_blockings).to(self.device)
        input_state_h = torch.from_numpy(
            np.reshape(input_state[:, 0], (-1, NetParameters.NET_SIZE))).to(self.device)
        input_state_c = torch.from_numpy(
            np.reshape(input_state[:, 1], (-1, NetParameters.NET_SIZE))).to(self.device)
        input_state = (input_state_h, input_state_c)

        advantage = returns - old_v
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)

        with autocast():
            new_ps, new_v, block, policy_sig, _, _ = self.network(observation, vector, input_state)
            new_p = new_ps.gather(-1, action)
            old_p = old_ps.gather(-1, action)
            ratio = torch.exp(torch.log(torch.clamp(new_p, 1e-6, 1.0)) - torch.log(torch.clamp(old_p, 1e-6, 1.0)))

            entropy = torch.mean(-torch.sum(new_ps * torch.log(torch.clamp(new_ps, 1e-6, 1.0)), dim=-1, keepdim=True))

            # intrinsic critic loss
            new_v = torch.squeeze(new_v)
            new_v_clipped = old_v + torch.clamp(new_v - old_v, - TrainingParameters.CLIP_RANGE,
                                                      TrainingParameters.CLIP_RANGE)
            value_losses1 = torch.square(new_v- returns)
            value_losses2 = torch.square(new_v_clipped - returns)
            critic_loss = torch.mean(torch.maximum(value_losses1, value_losses2))

            # actor loss
            ratio = torch.squeeze(ratio)
            policy_losses = advantage * ratio
            policy_losses2 = advantage * torch.clamp(ratio, 1.0 - TrainingParameters.CLIP_RANGE,
                                                     1.0 + TrainingParameters.CLIP_RANGE)
            policy_loss = torch.mean(torch.min(policy_losses, policy_losses2))

            # valid loss and blocking loss decreased by supervised learning
            valid_loss = - torch.mean(torch.log(torch.clamp(policy_sig, 1e-6, 1.0 - 1e-6)) *
                                      train_valid + torch.log(torch.clamp(1 - policy_sig, 1e-6, 1.0 - 1e-6)) * (
                                              1 - train_valid))
            block = torch.squeeze(block)
            blocking_loss = - torch.mean(target_blockings * torch.log(torch.clamp(block, 1e-6, 1.0 - 1e-6))
                                         + (1 - target_blockings) * torch.log(torch.clamp(1 - block, 1e-6, 1.0 - 1e-6)))

            # total loss
            all_loss = -policy_loss - entropy * TrainingParameters.ENTROPY_COEF + \
                TrainingParameters.VALUE_COEF * critic_loss + TrainingParameters.VALID_COEF * valid_loss \
                + TrainingParameters.BLOCK_COEF * blocking_loss

        clip_frac = torch.mean(torch.greater(torch.abs(ratio - 1.0), TrainingParameters.CLIP_RANGE).float())

        self.net_scaler.scale(all_loss).backward()
        self.net_scaler.unscale_(self.net_optimizer)

        # Clip gradient
        grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), TrainingParameters.MAX_GRAD_NORM)

        self.net_scaler.step(self.net_optimizer)
        self.net_scaler.update()

        stats_list = [all_loss.cpu().detach().numpy(), policy_loss.cpu().detach().numpy(),
                      entropy.cpu().detach().numpy(),
                      critic_loss.cpu().detach().numpy(),
                      valid_loss.cpu().detach().numpy(),
                      blocking_loss.cpu().detach().numpy(),
                      clip_frac.cpu().detach().numpy(), grad_norm.cpu().detach().numpy(),
                      torch.mean(advantage).cpu().detach().numpy()]  # for recording

        return stats_list

    def set_weights(self, weights):
        """load global weights to local models"""
        self.network.load_state_dict(weights)

