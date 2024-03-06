import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from agent.actor import Actor
from agent.critic import Critic
from utils.buffers import ReplayBuffer
from utils.utils import USE_CUDA
from utils.gaussian_noise import GaussianNoise

class TD3Agent:
    def __init__(self, state_dim, action_dim, **params):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.sequence_length = params.get('sequence_length')
        self.iterations = params.get('iterations')
        self.batch_size = params.get('batch_size')
        self.discount = params.get('discount')
        self.tau = params.get('tau')
        self.noise_clip = params.get('noise_clip')
        self.policy_freq = params.get('policy_freq')

        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.exploration_noise = GaussianNoise(action_dim, mean=0.0, std_deviation=1.5, decay_rate=0.99995) 
        self.target_policy_noise = GaussianNoise(action_dim, mean=0.0, std_deviation=0.2, decay_rate=1.0)  # Target Policy Smoothing Noise, 감쇠X
        self.device = torch.device("cuda" if USE_CUDA else "cpu")
        if USE_CUDA:
            self.cuda()
        

        # cuda 서순 조심
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)  # 0.001이 default
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = 0.001)

        self.replay_buffer = ReplayBuffer(capacity=100000)


    def select_action(self, state, add_noise=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_values, _ = self.actor(state)
        if add_noise:
            noise = self.exploration_noise.generate_noise().to(self.device)
            action_values = action_values + noise
        action_probs = torch.softmax(action_values, dim=-1)  # 확률 분포로 변환
        return action_probs.detach().cpu().numpy().flatten()

        

    def train(self):
        if self.replay_buffer.get_memlen() < self.batch_size:
            return
        for it in range(self.iterations):
            state, action, reward, next_state, done = self.replay_buffer.sample_and_split(self.batch_size)

            state = torch.FloatTensor(state).to(self.device)
            action = torch.FloatTensor(action).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            reward = torch.FloatTensor(reward).to(self.device)
            done = torch.FloatTensor(done).to(self.device)

            # Target Q-value 계산
            # noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip)
            # action_values, _ = self.actor_target(next_state)
            # next_action = (action_values + noise).clamp(-1, 1)
            noise = self.target_policy_noise.generate_noise().to(self.device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            action_values, _ = self.actor_target(next_state)
            next_action = (action_values + noise).clamp(-1, 1)

            
            target_Q1, target_Q2, _, _ = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * self.discount * target_Q).detach()

            # Critic 업데이트
            current_Q1, current_Q2, _, _ = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            if it % self.policy_freq == 0:
                actor_output, _ = self.actor(state)
                actor_loss = -self.critic.Q1(state, actor_output).mean()
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Soft update
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def _init_hidden_state(self, batch_size):
        h0 = torch.zeros(self.actor.lstm.num_layers, batch_size, self.actor.lstm.hidden_size).to(self.device)
        c0 = torch.zeros(self.actor.lstm.num_layers, batch_size, self.actor.lstm.hidden_size).to(self.device)
        return h0, c0

    def _soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.critic.state_dict(), filename + "_critic")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.critic.load_state_dict(torch.load(filename + "_critic"))
