import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast

from alg_parameters import *


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class Net(nn.Module):
    """network with transformer-based communication mechanism"""

    def __init__(self):
        """initialization"""
        super(Net, self).__init__()
        gain = nn.init.calculate_gain('relu')

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)

        def init2_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), NetParameters.GAIN)

        def init3_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        # observation encoder
        self.conv1 = init_(nn.Conv2d(NetParameters.NUM_CHANNEL, NetParameters.NET_SIZE // 4, 3, 1, 1))
        self.conv1a = init_(nn.Conv2d(NetParameters.NET_SIZE // 4, NetParameters.NET_SIZE // 4, 3, 1, 1))
        self.conv1b = init_(nn.Conv2d(NetParameters.NET_SIZE // 4, NetParameters.NET_SIZE // 4, 3, 1, 1))
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = init_(nn.Conv2d(NetParameters.NET_SIZE // 4, NetParameters.NET_SIZE // 2, 2, 1, 1))
        self.conv2a = init_(nn.Conv2d(NetParameters.NET_SIZE // 2, NetParameters.NET_SIZE // 2, 2, 1, 1))
        self.conv2b = init_(nn.Conv2d(NetParameters.NET_SIZE // 2, NetParameters.NET_SIZE // 2, 2, 1, 1))
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = init_(nn.Conv2d(NetParameters.NET_SIZE // 2, NetParameters.NET_SIZE - NetParameters.GOAL_REPR_SIZE, 3,
                                     1, 0))
        self.fully_connected_1 = init_(nn.Linear(NetParameters.VECTOR_LEN, NetParameters.GOAL_REPR_SIZE))
        self.fully_connected_2 = init_(nn.Linear(NetParameters.NET_SIZE, NetParameters.NET_SIZE))
        self.fully_connected_3 = init_(nn.Linear(NetParameters.NET_SIZE, NetParameters.NET_SIZE))
        self.lstm_memory = nn.LSTMCell(input_size=NetParameters.NET_SIZE, hidden_size=NetParameters.NET_SIZE)
        for name, param in self.lstm_memory.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        # output heads
        self.policy_layer = init2_(nn.Linear(NetParameters.NET_SIZE, EnvParameters.N_ACTIONS))
        self.softmax_layer = nn.Softmax(dim=-1)
        self.value_layer = init3_(nn.Linear(NetParameters.NET_SIZE, 1))
        self.blocking_layer = init2_(nn.Linear(NetParameters.NET_SIZE, 1))

        self.feature_norm = nn.LayerNorm(NetParameters.VECTOR_LEN)
        self.layer_norm_1 = nn.LayerNorm(NetParameters.NET_SIZE - NetParameters.GOAL_REPR_SIZE)
        self.layer_norm_2 = nn.LayerNorm(NetParameters.GOAL_REPR_SIZE)
        self.layer_norm_3 = nn.LayerNorm(NetParameters.NET_SIZE)
        self.layer_norm_4 = nn.LayerNorm(NetParameters.NET_SIZE)
        self.layer_norm_5 = nn.LayerNorm(NetParameters.NET_SIZE)

    @autocast()
    def forward(self, obs, vector, input_state):
        """run neural network"""
        num_agent = obs.shape[1]
        obs = torch.reshape(obs, (-1,  NetParameters.NUM_CHANNEL, EnvParameters.FOV_SIZE, EnvParameters.FOV_SIZE))
        vector = torch.reshape(vector, (-1, NetParameters.VECTOR_LEN))
        # matrix input
        x_1 = F.relu(self.conv1(obs))
        x_1 = F.relu(self.conv1a(x_1))
        x_1 = F.relu(self.conv1b(x_1))
        x_1 = self.pool1(x_1)
        x_1 = F.relu(self.conv2(x_1))
        x_1 = F.relu(self.conv2a(x_1))
        x_1 = F.relu(self.conv2b(x_1))
        x_1 = self.pool2(x_1)
        x_1 = self.conv3(x_1)
        x_1 = F.relu(x_1.view(x_1.size(0), -1))
        x_1=self.layer_norm_1(x_1)

        # vector input
        x_2=self.feature_norm(vector)
        x_2 = F.relu(self.fully_connected_1(x_2))
        x_2=self.layer_norm_2(x_2)
        # Concatenation
        x_3 = torch.cat((x_1, x_2), -1)
        x_3 = self.layer_norm_3(x_3)
        h1 = F.relu(self.fully_connected_2(x_3))
        h1 = self.fully_connected_3(h1)
        h2 = F.relu(h1 + x_3)
        h2 = self.layer_norm_4(h2)

        # LSTM cell
        memories, memory_c = self.lstm_memory(h2, input_state)
        output_state = (memories, memory_c)
        memories = torch.reshape(memories, (-1, num_agent, NetParameters.NET_SIZE))
        memories =self.layer_norm_5(memories)

        policy_layer = self.policy_layer(memories)
        policy = self.softmax_layer(policy_layer)
        policy_sig = torch.sigmoid(policy_layer)
        value = self.value_layer(memories)
        blocking = torch.sigmoid(self.blocking_layer(memories))
        return policy, value, blocking, policy_sig, output_state, policy_layer

if __name__ == '__main__':
    net=Net()
    observation = torch.torch.rand(
        (2, EnvParameters.LOCAL_N_AGENTS, NetParameters.NUM_CHANNEL, EnvParameters.FOV_SIZE, EnvParameters.FOV_SIZE),
        dtype=torch.float32)  # [B,A,4,F,F]
    vectors = torch.torch.rand((2, EnvParameters.LOCAL_N_AGENTS, NetParameters.VECTOR_LEN))  # [B,A,3]
    hidden_state = (
        torch.torch.rand((EnvParameters.LOCAL_N_AGENTS * 2, NetParameters.NET_SIZE )),
        torch.torch.rand((EnvParameters.LOCAL_N_AGENTS* 2, NetParameters.NET_SIZE )))  # [B*A,3]

    policy, value, blocking, policy_sig, output_state, policy_layer = net(observation, vectors, hidden_state)
    print("test")


