"""
Diffusion schedulers
"""
from .ddpm import DDPMScheduler

import numpy as np


def get_equal_steps(max_steps, num_steps):
    steps = [0]
    steps.extend([i * max_steps // num_steps - 1 
                  for i in range(1, num_steps + 1)])
    return np.array(steps)