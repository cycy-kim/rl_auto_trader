import torch

class GaussianNoise:
    def __init__(self, action_dimension, mean=0.0, std_deviation=0.1, decay_rate=0.99):
        self.action_dimension = action_dimension
        self.mean = mean
        self.std_dev = std_deviation
        self.decay_rate = decay_rate

    def generate_noise(self):
        self.std_dev *= self.decay_rate
        return torch.randn(self.action_dimension) * self.std_dev + self.mean
