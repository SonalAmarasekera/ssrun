"""
utils/lr_sched.py
Cosine-decay learning-rate schedule with linear warm-up.

Example
-------
>>> optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
>>> scheduler = CosineWarmup(optimizer, warmup=1000, max_steps=40000)
>>> for step in range(40000):
...     train_step()
...     optimizer.step()
...     scheduler.step()
"""
import math
from typing import List
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class CosineWarmup(_LRScheduler):
    """
    Linearly warm-up to `base_lr`, then cosine-decay down to `min_lr`.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Wrapped optimiser.
    warmup : int
        Number of warm-up steps.
    max_steps : int
        Total number of scheduler steps (warm-up + decay).
    min_lr : float, default 1e-5
        Final learning-rate value after cosine decay.
    last_epoch : int, default -1
        PyTorch bookkeeping; leave at default.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup: int,
        max_steps: int,
        min_lr: float = 1e-5,
        last_epoch: int = -1,
    ):
        self.warmup = warmup
        self.max_steps = max_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        step = self.last_epoch + 1  # because _LRScheduler calls step()+1
        base_lrs = self.base_lrs

        # Phase 1 – linear warm-up
        if step <= self.warmup and self.warmup > 0:
            return [
                lr * step / self.warmup
                for lr in base_lrs
            ]

        # Phase 2 – cosine decay
        progress = (step - self.warmup) / max(1, self.max_steps - self.warmup)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return [
            self.min_lr + (lr - self.min_lr) * cosine_decay
            for lr in base_lrs
        ]

