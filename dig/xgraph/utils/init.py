import random

import numpy as np
import torch


def fix_random_seed(random_seed: int):
    r"""
    Fix multiple random seeds including python, numpy, torch, torch.cuda, and torch.backends.

    Args:
        random_seed (int): The random seed.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
