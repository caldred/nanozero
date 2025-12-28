"""
nanozero/common.py - Shared utilities
"""
import os
import random
import torch
import numpy as np
from typing import Optional

def get_device() -> torch.device:
    """
    Detect best available device.
    Returns 'cuda' if available, else 'mps' if available, else 'cpu'.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print0(*args, **kwargs) -> None:
    """Print (for future distributed compatibility)."""
    print(*args, **kwargs)

class AverageMeter:
    """Computes and stores running average."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    path: str
) -> None:
    """Save model checkpoint."""
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration,
    }, path)

def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> int:
    """Load model checkpoint. Returns iteration number."""
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt.get('iteration', 0)
