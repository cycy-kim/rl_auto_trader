import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from agents.base_agent import BaseAgent
from agents.networks.shared_actorcritic import SharedActorCritic
from utils.buffers import PPOBuffer
from utils.utils import USE_CUDA

# params needed: clip_param, ppo_update_times, lr, sub_batch_size, batch_size, buffer_size, gamma, lmbda, entropy_eps
class PPOAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, **params):
        super(PPOAgent, self).__init__(state_dim, action_dim)

        self.clip_param = params.get('clip_param', 0.2)
        self.ppo_update_times = params.get('ppo_update_times', 5)
        self.batch_size = params.get('batch_size', 64)
        self.sub_batch_size = params.get('sub_batch_size', 64)
        self.gamma = params.get('gamma', 0.99)
        self.lmbda = params.get('lmbda', 0.95)
        self.entropy_eps = params.get('entropy_eps', 1e-4) 

        self.network = SharedActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=params.get('lr', 1e-3))

        self.buffer = PPOBuffer(params.get('buffer_size', 100000))

        self.device = torch.device("cuda" if USE_CUDA else "cpu")
        if USE_CUDA:
            self.cuda()

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_logits, state_value = self.network(state_tensor)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action_probs.detach().cpu().numpy().flatten(), log_prob.item(), state_value.item()


    def store_transition(self, state, action, reward, value_pred, log_prob, next_state, done):
        self.buffer.append(state, action, reward, value_pred, log_prob, next_state, done)

    def train(self):
        if self.buffer.get_memlen() < self.batch_size:
            return
        for _ in range(self.ppo_update_times):
            states, actions, rewards, value_preds, log_probs, next_states, dones = self.buffer.sample(self.batch_size)
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            value_preds = value_preds.to(self.device)
            log_probs = log_probs.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)

            new_log_probs, state_values = self.evaluate(states, actions)
            dist = Categorical(new_log_probs.exp())
            entropy = dist.entropy().mean()

            advantages = rewards + self.gamma * value_preds * (~dones) - state_values.detach()
            ratios = torch.exp(new_log_probs - log_probs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_eps * entropy  # 엔트로피 추가
            value_loss = F.mse_loss(state_values, rewards.float() + self.gamma * value_preds * (~dones).float())

            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def evaluate(self, state, action):
        action_probs, state_values = self.network(state)
        dist = Categorical(action_probs)

        action_log_probs = dist.log_prob(action)
        return action_log_probs, state_values

        
    def cuda(self):
        self.network.cuda()
        
    def save(self, filename):
        torch.save(self.network.state_dict(), filename + "_ppo_network.pth")

    def load(self, filename):
        self.network.load_state_dict(torch.load(filename + "_ppo_network.pth"))