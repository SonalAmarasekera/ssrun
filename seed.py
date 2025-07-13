"""
utils/seed.py
Set a single random seed across Python, NumPy and PyTorch (CPU + CUDA).

Example
-------
>>> from utils.seed import seed_everything
>>> seed_everything(42)
"""
import os, random
import numpy as np
import torch

def seed_everything(seed: int = 42) -> None:
    """
    Fix all RNG seeds for full-run reproducibility.

    Parameters
    ----------
    seed : int
        The seed to set everywhere (default = 42).
    """
    # 1. Python built-in RNG
    random.seed(seed)

    # 2. Environment variable (affects hashing, some libraries)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 3. NumPy RNG
    np.random.seed(seed)

    # 4. PyTorch RNGs
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 5. CuDNN / determinism flags
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

