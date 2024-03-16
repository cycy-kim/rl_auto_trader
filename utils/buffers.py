import random
import numpy as np
import torch
from collections import namedtuple, deque

ReplayBufferTransition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])
class ReplayBuffer:
    def __init__(self, capacity, sequence_length=8):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.memory = deque(maxlen=capacity)

    def append(self, state, action, reward, next_state, done):
        self.memory.append(ReplayBufferTransition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = []
        for _ in range(batch_size):
            while True:
                index = random.randint(0, len(self.memory) - self.sequence_length)
                
                sequence = [self.memory[i] for i in range(index, index + self.sequence_length)]
                
                if not any(transition.done for transition in sequence[:-1]):
                    samples.append(sequence)
                    break
        return samples

    def sample_and_split(self, batch_size):
        sequences = self.sample(batch_size)
        batch = ReplayBufferTransition(*zip(*[item for seq in sequences for item in seq]))
        
        state_batch = np.array(batch.state).reshape(batch_size, self.sequence_length, -1)
        action_batch = np.array(batch.action).reshape(batch_size, self.sequence_length, -1)
        reward_batch = np.array(batch.reward).reshape(batch_size, self.sequence_length, -1)
        next_state_batch = np.array(batch.next_state).reshape(batch_size, self.sequence_length, -1)
        done_batch = np.array(batch.done).reshape(batch_size, self.sequence_length, -1)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def get_memlen(self):
        return len(self.memory)

class SlidingWindowBuffer:
    def __init__(self, intervals):
        self.buffers = {interval: deque(maxlen=interval) for interval in intervals}

    def add(self, data):
        for interval, buffer in self.buffers.items():
            buffer.append(data)

    def get_data(self, intervals):
        return [list(self.buffers[interval]) for interval in intervals]

    def get_max(self, intervals):
        return [max(self.buffers[interval]) if self.buffers[interval] else None for interval in intervals]

    def get_min(self, intervals):
        return [min(self.buffers[interval]) if self.buffers[interval] else None for interval in intervals]

    def get_mean(self, intervals):
        return [(sum(self.buffers[interval]) / len(self.buffers[interval])) if self.buffers[interval] else None for interval in intervals]

    def clear(self):
        for buffer in self.buffers.values():
            buffer.clear()


PPOBufferTransition = namedtuple('PPOBufferTransition', ['state', 'action', 'reward', 'value_pred', 'log_prob', 'next_state', 'done'])

class PPOBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        
    def append(self, state, action, reward, value_pred, log_prob, next_state, done):
        self.memory.append(PPOBufferTransition(state, action, reward, value_pred, log_prob, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.memory, batch_size)

        states = torch.stack([torch.tensor(sample.state).float() for sample in samples])
        actions = torch.tensor([sample.action for sample in samples])
        rewards = torch.tensor([sample.reward for sample in samples])
        value_preds = torch.tensor([sample.value_pred for sample in samples])
        log_probs = torch.tensor([sample.log_prob for sample in samples])
        next_states = torch.stack([torch.tensor(sample.next_state).float() for sample in samples])
        dones = torch.tensor([sample.done for sample in samples])

        return states, actions, rewards, value_preds, log_probs, next_states, dones

    def clear(self):
        self.memory.clear()

    def get_memlen(self):
        return len(self.memory)