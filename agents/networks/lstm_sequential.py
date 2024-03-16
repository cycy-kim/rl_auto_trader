import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMSequential(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(LSTMSequential, self).__init__()
        self.lstm = nn.LSTM(state_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc3 = nn.Linear(hidden_dim // 4, action_dim)

    def forward(self, state, hidden=None):
        x, hidden = self.lstm(state, hidden)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        probabilities = F.softmax(logits, dim=-1)
        return probabilities, hidden
