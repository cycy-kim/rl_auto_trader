import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from agents.base_agent import BaseAgent
from agents.networks.gaussian_actor import GaussianActor
from agents.networks.fc_critic import FCCritic

from utils.utils import USE_CUDA
from utils.buffers import ReplayBuffer

class SACAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, **params):
        super(SACAgent, self).__init__(state_dim, action_dim)

        self.lr = params.get('lr')
        self.batch_size = params.get('batch_size')
        self.gamma = params.get('gamma')
        self.tau = params.get('tau')
        self.buffer_size = params.get('buffer_size')

        self.actor = GaussianActor(state_dim, action_dim)
        self.critic1 = FCCritic(state_dim, action_dim)
        self.critic2 = FCCritic(state_dim, action_dim)
        self.target_critic1 = FCCritic(state_dim, action_dim)
        self.target_critic2 = FCCritic(state_dim, action_dim)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.lr)

        self.device = torch.device("cuda" if USE_CUDA else "cpu")
        if USE_CUDA:
            self.cuda()

        self.buffer = ReplayBuffer(capacity=self.buffer_size)
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        mean, std = self.actor(state)
        dist = Normal(mean, std)
        action = dist.sample()
        print(action.detach().cpu().numpy())
        return action.detach().cpu().numpy()

    def train(self):
        state, action, reward, next_state, done = self.buffer.sample_and_split(self.batch_size)

        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        done = torch.FloatTensor(done).unsqueeze(1)

        with torch.no_grad():
            next_action, next_log_pi = self.actor.sample(next_state)
            q_target_next1 = self.target_critic1(next_state, next_action)
            q_target_next2 = self.target_critic2(next_state, next_action)
            q_target_next = torch.min(q_target_next1, q_target_next2) - next_log_pi
            q_target = reward + (1 - done) * self.gamma * q_target_next

        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)

        q1_loss = F.mse_loss(q1, q_target)
        q2_loss = F.mse_loss(q2, q_target)

        self.critic1_optimizer.zero_grad()
        q1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        q2_loss.backward()
        self.critic2_optimizer.step()

        new_action, log_pi = self.actor.sample(state)
        q1_new = self.critic1(state, new_action)
        q2_new = self.critic2(state, new_action)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (log_pi - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def cuda(self):
        self.actor.cuda()
        self.critic1.cuda()
        self.critic2.cuda()
        self.target_critic1.cuda()
        self.target_critic2.cuda()

    def save(self, filename):
        torch.save(self.actor.state_dict(), f"{filename}_actor.pth")
        torch.save(self.critic1.state_dict(), f"{filename}_critic1.pth")
        torch.save(self.critic2.state_dict(), f"{filename}_critic2.pth")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(f"{filename}_actor.pth"))
        self.critic1.load_state_dict(torch.load(f"{filename}_critic1.pth"))
        self.critic2.load_state_dict(torch.load(f"{filename}_critic2.pth"))