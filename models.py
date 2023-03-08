import torch
import torch.nn as nn
import torch.nn.functional as F


# import torchvision.models as models


class ExperimentBinaryModule(nn.Module):
    def __init__(self, input_dim, hidden_neurons_num):
        super(ExperimentBinaryModule, self).__init__()
        self.linear_1 = nn.Linear(input_dim, hidden_neurons_num)
        self.linear_2 = nn.Linear(hidden_neurons_num, 1)

    def forward(self, x):
        temp = F.relu(self.linear_1(x))
        return self.linear_2(temp)


class DQNModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNModule, self).__init__()
        self.linear_1 = nn.Linear(input_dim, 128)
        self.linear_2 = nn.Linear(128, 128)
        self.linear_3 = nn.Linear(128, output_dim)

    def forward(self, x):
        temp = self.linear_1(x)
        temp = F.relu(temp)
        temp = self.linear_2(temp)
        temp = F.relu(self.linear_2(temp))
        temp = self.linear_3(temp)
        return temp
