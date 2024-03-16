import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMTwinQ(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(LSTMTwinQ, self).__init__()
        self.lstm1 = nn.LSTM(state_dim + action_dim, hidden_dim, batch_first=True)
        self.fc1_1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc1_2 = nn.Linear(hidden_dim // 2, 1)

        self.lstm2 = nn.LSTM(state_dim + action_dim, hidden_dim, batch_first=True)
        self.fc2_1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2_2 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, state, action, hidden=None):
        xu = torch.cat([state, action], dim=-1)
        q1, hidden1 = self.lstm1(xu, hidden)
        q1 = F.relu(self.fc1_1(q1))
        q1 = self.fc1_2(q1)

        q2, hidden2 = self.lstm2(xu, hidden)
        q2 = F.relu(self.fc2_1(q2))
        q2 = self.fc2_2(q2)

        return q1, q2, hidden1, hidden2
    
    def Q1(self, state, action, hidden=None):
        xu = torch.cat([state, action], dim=-1)
        q1, _ = self.lstm1(xu, hidden)
        q1 = F.relu(self.fc1_1(q1))
        q1 = self.fc1_2(q1)
        return q1
