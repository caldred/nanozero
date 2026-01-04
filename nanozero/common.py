"""
nanozero/common.py - Shared utilities
"""
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
    path: str,
    scaler: Optional[object] = None
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration,
    }
    if scaler is not None:
        checkpoint['scaler'] = scaler.state_dict()
    torch.save(checkpoint, path)

def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[object] = None
) -> int:
    """Load model checkpoint. Returns iteration number."""
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scaler is not None and 'scaler' in ckpt:
        scaler.load_state_dict(ckpt['scaler'])
    return ckpt.get('iteration', 0)


def sample_action(probs: np.ndarray, temperature: float = 1.0) -> int:
    """
    Sample an action from probability distribution.

    Args:
        probs: Probability distribution over actions, shape (action_size,)
        temperature: Temperature for sampling.
                    0 = greedy (argmax)
                    1 = sample from distribution
                    >1 = more uniform
                    <1 = more peaked

    Returns:
        Sampled action index
    """
    if temperature == 0:
        return int(np.argmax(probs))

    if temperature != 1.0:
        log_probs = np.log(probs + 1e-8)
        log_probs = log_probs / temperature
        probs = np.exp(log_probs - np.max(log_probs))
        probs = probs / probs.sum()

    return int(np.random.choice(len(probs), p=probs))
