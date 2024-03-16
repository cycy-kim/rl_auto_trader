import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from agents.base_agent import BaseAgent
from agents.networks.fc_actor import FCActor
from agents.networks.fc_critic import FCCritic
from utils.utils import USE_CUDA
from utils.buffers import ReplayBuffer

class DDPGAgent(BaseAgent):
    # def __init__(self, state_dim, action_dim, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.005, buffer_size=100000):
    def __init__(self, state_dim, action_dim, **params):
        super(DDPGAgent, self).__init__(state_dim, action_dim)
        
        self.actor_lr = params.get('actor_lr')
        self.critic_lr = params.get('critic_lr')
        self.gamma = params.get('gamma')
        self.tau = params.get('tau')
        self.buffer_size = params.get('buffer_size')
        self.batch_size = params.get('batch_size')

        self.actor = FCActor(state_dim, action_dim)
        self.actor_target = FCActor(state_dim, action_dim)
        self.critic = FCCritic(state_dim, action_dim)
        self.critic_target = FCCritic(state_dim, action_dim)
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.device = torch.device("cuda" if USE_CUDA else "cpu")
        if USE_CUDA:
            self.cuda()
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.buffer = ReplayBuffer(self.buffer_size)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state)
        action_probs = torch.softmax(action, dim=-1)  # 확률 분포로 변환
        return action_probs.cpu().data.numpy().flatten()

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.append(state, action, reward, next_state, done)

    def train(self):
        if self.buffer.get_memlen() < self.batch_size:
            return
        
        state, action, reward, next_state, done = self.buffer.sample_and_split(self.batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)


        next_actions = self.actor_target(next_state)
        next_Q_values = self.critic_target(next_state, next_actions)
        expected_Q_values = reward + (1 - done) * self.gamma * next_Q_values
        
        current_Q_values = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_Q_values, expected_Q_values.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.critic, self.critic_target, self.tau)
        self.soft_update(self.actor, self.actor_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic.state_dict(), filename + "_critic.pth")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor.pth"))
        self.critic.load_state_dict(torch.load(filename + "_critic.pth"))
