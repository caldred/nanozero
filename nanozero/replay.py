"""
nanozero/replay.py - Experience replay buffer

Fixed-size circular buffer for storing training examples from self-play.
"""
import numpy as np
from typing import List, Tuple
import random


class ReplayBuffer:
    """
    Fixed-size circular buffer for experience replay.

    Stores (state, policy, value) tuples from self-play games.
    When buffer is full, oldest examples are overwritten.
    """

    def __init__(self, capacity: int):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of examples to store
        """
        self.capacity = capacity
        self.buffer: List[Tuple[np.ndarray, np.ndarray, float]] = []
        self.position = 0

    def push(self, state: np.ndarray, policy: np.ndarray, value: float) -> None:
        """
        Add a single example to the buffer.

        Args:
            state: Game state (canonical form), shape (H, W)
            policy: Target policy from MCTS, shape (action_size,)
            value: Target value (game outcome from this state's perspective)
        """
        example = (state.copy(), policy.copy(), value)

        if len(self.buffer) < self.capacity:
            self.buffer.append(example)
        else:
            self.buffer[self.position] = example

        self.position = (self.position + 1) % self.capacity

    def extend(self, examples: List[Tuple[np.ndarray, np.ndarray, float]]) -> None:
        """
        Add multiple examples to the buffer.

        Args:
            examples: List of (state, policy, value) tuples
        """
        for state, policy, value in examples:
            self.push(state, policy, value)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random batch of examples.

        Args:
            batch_size: Number of examples to sample

        Returns:
            Tuple of:
                - states: np.ndarray of shape (batch_size, H, W)
                - policies: np.ndarray of shape (batch_size, action_size)
                - values: np.ndarray of shape (batch_size,)
        """
        # Sample without replacement (capped at buffer size)
        indices = random.sample(range(len(self.buffer)), min(batch_size, len(self.buffer)))

        states = []
        policies = []
        values = []

        for idx in indices:
            state, policy, value = self.buffer[idx]
            states.append(state)
            policies.append(policy)
            values.append(value)

        return (
            np.array(states),
            np.array(policies),
            np.array(values, dtype=np.float32)
        )

    def __len__(self) -> int:
        """Return current number of examples in buffer."""
        return len(self.buffer)

    def clear(self) -> None:
        """Remove all examples from the buffer."""
        self.buffer.clear()
        self.position = 0
