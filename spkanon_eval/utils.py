import random

import numpy as np
import torch


def seed_everything(seed: int):
    """Set the seed for Python, Numpy and Torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
