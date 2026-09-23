"""Reproducible seeding for training and evaluation runs.

Two independent generators have to be pinned:

* the process-global torch/numpy/random state, which drives network
  initialisation and action sampling;
* VMAS's private state, which it swaps in around every environment method
  (see ``vmas.simulator.environment.environment.local_seed``) and which drives
  agent spawn positions and obstacle layouts.

Seeding only the first leaves environment resets random, which is why runs that
called ``torch.manual_seed`` still diverged.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_global_seeds(seed: int) -> None:
    """Seed the process-global RNGs used by the policy and the training loop."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_everything(env, seed: int) -> None:
    """Seed process-global RNGs and the VMAS environment's private RNG."""
    set_global_seeds(seed)
    if env is not None:
        env.seed(seed)
