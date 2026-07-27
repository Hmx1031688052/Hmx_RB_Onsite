import torch
import torch.nn as nn
from config import Config
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal

torch.manual_seed(248794110)
class QNet(nn.Module):

    def __init__(self, state_num, action_num=2):
        super().__init__()
        # self.seed = torch.manual_seed(914)
        self.state_num = state_num
        self.action_num = action_num
        self.fc1 = nn.Linear(self.state_num + self.action_num, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 2)

    def forward(self, observation, action):
        x = torch.cat([observation, action], dim=-1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc3(x)
        x = F.gelu(x)
        logits = self.fc4(x)
        value_mean, value_std = torch.chunk(logits, chunks=2, dim=-1)
        value_log_std = torch.nn.functional.softplus(value_std)  # avoid 0

        return torch.cat((value_mean, value_log_std), dim=-1)
    
    def load(self, checkpoint_path, optimizer=None):
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(torch.load(checkpoint_path))
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])


class PolicyNet(nn.Module):

    def __init__(self, state_num, action_num):
        super().__init__()
        # self.seed = torch.manual_seed(914)
        self.state_num = state_num
        self.action_num = action_num
        self.fc1 = nn.Linear(self.state_num, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, self.action_num * 2)
        self.min_log_std = -20
        self.max_log_std = 0.5

    def forward(self, observation):
        x = observation
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc3(x)
        x = F.gelu(x)
        logits = self.fc4(x)
        action_mean, action_log_std = torch.chunk(
            logits, chunks=2, dim=-1
        )  # output the mean
        action_std = torch.clamp(
            action_log_std, self.min_log_std, self.max_log_std
        ).exp()
        action_mean = action_mean
        return torch.cat((action_mean, action_std), dim=-1)

    def load(self, checkpoint_path, optimizer=None):
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(torch.load(checkpoint_path))
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])



class HNet(nn.Module):

    def __init__(self, set_state_num):
        super().__init__()
        self.set_state_num = set_state_num
        self.con1 = nn.Sequential(
            nn.Conv1d(1, 32, 7, 7),
            nn.ReLU())
        self.con2 = nn.Sequential(
            nn.Conv1d(32, 32, 1, 1),
            nn.ReLU())
        self.pool = nn.MaxPool1d(6)
        self.fc1 = nn.Linear(self.set_state_num, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 32)

    def forward(self, set_observation):
        x = set_observation.unsqueeze(1)
        x_other = self.con1(x)
        x_other = self.con2(x_other)
        logits = self.pool(x_other).squeeze(-1)
        return logits

    def load(self, checkpoint_path, optimizer=None):
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(torch.load(checkpoint_path))
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
