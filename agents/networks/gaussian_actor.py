import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(GaussianActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean_layer = nn.Linear(hidden_size, action_dim)
        self.log_std_layer = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x).clamp(-20, 2)
        return mean, log_std.exp()
