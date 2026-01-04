"""
nanozero/config.py - Configuration dataclasses
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class GameConfig:
    """Configuration for a game."""
    name: str
    board_height: int
    board_width: int
    action_size: int
    num_players: int = 2

    @property
    def board_size(self) -> int:
        return self.board_height * self.board_width

@dataclass
class ModelConfig:
    """Configuration for the Transformer model."""
    board_size: int
    action_size: int
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0

    def __post_init__(self):
        # Scale embedding dim with depth (like nanochat)
        if self.n_embd == 128:
            self.n_embd = max(64, 64 * self.n_layer)
        assert self.n_embd % self.n_head == 0

@dataclass
class MCTSConfig:
    """Configuration for Monte Carlo Tree Search."""
    num_simulations: int = 100
    c_puct: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0

@dataclass
class BayesianMCTSConfig:
    """Configuration for Bayesian BAI-MCTS.

    Uses Gaussian beliefs with Thompson sampling instead of PUCT.
    Optimizes for best arm identification rather than cumulative regret.
    Note: No Dirichlet noise - Thompson sampling provides exploration naturally.
    Note: No temperature - policy is probability of optimality via Thompson sampling.
    """
    num_simulations: int = 1000

    # Bayesian prior hyperparameters
    sigma_0: float = 0.5          # Prior std for logit-shifted initialization
    obs_var: float = 0.25         # Observation variance (NN value uncertainty)

    # IDS hyperparameters
    ids_alpha: float = 0.0        # Pseudocount for IDS allocation

    # Variance aggregation
    prune_threshold: float = 0.01  # Soft-prune children with P(optimal) < threshold
    optimality_weight: float = 0.3  # Base blend: 0=visit-proportional, 1=pure optimality
    adaptive_weight: bool = True   # If True, increase optimality_weight as visits grow
    visit_scale: float = 50.0      # Visits at which adaptive weight reaches ~0.86 of max

    # Early stopping
    early_stopping: bool = True   # Whether to stop when confident about best action
    confidence_threshold: float = 0.95  # P(leader is optimal) threshold for stopping
    min_simulations: int = 10     # Minimum simulations before early stopping

    # Numerical stability
    min_variance: float = 1e-6    # Floor for variance to avoid division by zero

@dataclass
class TrainConfig:
    """Configuration for training loop."""
    num_iterations: int = 100
    games_per_iteration: int = 100
    training_steps: int = 100
    batch_size: int = 64
    buffer_size: int = 100000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    checkpoint_interval: int = 10
    eval_interval: int = 10
    mcts_simulations: int = 50
    temperature_threshold: int = 15

def get_game_config(name: str) -> GameConfig:
    """Get predefined game configuration."""
    configs = {
        'tictactoe': GameConfig(
            name='tictactoe',
            board_height=3,
            board_width=3,
            action_size=9,
        ),
        'connect4': GameConfig(
            name='connect4',
            board_height=6,
            board_width=7,
            action_size=7,
        ),
        'go9x9': GameConfig(
            name='go9x9',
            board_height=9,
            board_width=9,
            action_size=82,  # 81 board positions + pass
        ),
        'go19x19': GameConfig(
            name='go19x19',
            board_height=19,
            board_width=19,
            action_size=362,  # 361 board positions + pass
        ),
    }
    if name not in configs:
        raise ValueError(f"Unknown game: {name}")
    return configs[name]

def get_model_config(game_config: GameConfig, n_layer: int = 4) -> ModelConfig:
    """Get model configuration for a game."""
    return ModelConfig(
        board_size=game_config.board_size,
        action_size=game_config.action_size,
        n_layer=n_layer,
    )
