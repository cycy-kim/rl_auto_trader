import torch
import torch.nn as nn
import torch.nn.functional as F

class FCActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(FCActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))
        return action
