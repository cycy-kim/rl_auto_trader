import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(SharedActorCritic, self).__init__()
        
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        shared_features = self.shared_layers(state)
        action_probs = F.softmax(self.actor(shared_features), dim=-1)
        state_value = self.critic(shared_features)
        return action_probs, state_value
