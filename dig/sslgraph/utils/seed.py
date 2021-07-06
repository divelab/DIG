import random
import torch
import numpy as np


def setup_seed(seed):
    r"""To setup seed for reproducible experiments.
    
    Args:
        seed (int, or float): The number used as seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
