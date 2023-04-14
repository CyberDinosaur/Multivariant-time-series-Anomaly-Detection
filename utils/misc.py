import torch
import numpy as np
import random


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    return seed

def get_device(device):
    return torch.device(device if torch.cuda.is_available() else 'cpu')

