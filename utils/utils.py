import os
import torch
import numpy as np

USE_CUDA = torch.cuda.is_available()

def get_random_action(action_dim):
    alpha = np.ones(action_dim)

    values = np.random.dirichlet(alpha)

    return values
