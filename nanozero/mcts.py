"""
nanozero/mcts.py - Monte Carlo Tree Search using Rust backend

This module provides MCTS implementations using the Rust backend.
The Rust backend (nanozero_mcts_rs) is required.
"""
import numpy as np
import torch
from typing import Optional, Tuple
from nanozero.config import MCTSConfig, BayesianMCTSConfig
from nanozero.game import Game
from nanozero.common import sample_action

# Import Rust MCTS backend (required)
from nanozero_mcts_rs import RustBatchedMCTS as _RustBatchedMCTS
from nanozero_mcts_rs import RustBayesianMCTS as _RustBayesianMCTS

# For backwards compatibility
HAS_RUST_MCTS = True


class BatchedMCTS:
    """
    Batched MCTS using Rust backend for tree operations.

    Runs the entire MCTS search in Rust, only calling back to Python
    for neural network inference. This minimizes Python-Rust boundary
    crossings for maximum performance.
    """

    def __init__(self, game: Game, config: MCTSConfig, virtual_loss: float = 1.0,
                 leaves_per_batch: int = 0, use_transposition_table: bool = True):
        """
        Initialize Rust-backed batched MCTS.

        Args:
            game: Game instance
            config: MCTS configuration
            virtual_loss: Virtual loss value for penalizing in-flight paths
            leaves_per_batch: Number of leaves to collect per NN call (0 = auto)
            use_transposition_table: Whether to use transposition table with symmetry support
        """
        self.game = game
        self.config = config
        self._model = None
        self._device = None
        self.backend = 'rust'
        self._use_transposition_table = use_transposition_table

        # Create Rust MCTS instance with virtual loss batching
        self._rust_mcts = _RustBatchedMCTS(
            c_puct=config.c_puct,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=config.dirichlet_epsilon,
            num_simulations=config.num_simulations,
            leaves_per_batch=leaves_per_batch,
            virtual_loss_value=virtual_loss,
            use_transposition_table=use_transposition_table,
        )

        # Determine which game-specific search method to use
        game_name = type(game).__name__.lower()
        if 'tictactoe' in game_name:
            self._search_fn = self._rust_mcts.search_tictactoe
            self._is_go = False
        elif 'connect4' in game_name:
            self._search_fn = self._rust_mcts.search_connect4
            self._is_go = False
        elif 'go' in game_name:
            self._go_board_size = getattr(game, 'height', 9)
            self._search_fn = self._search_go
            self._is_go = True
        else:
            raise ValueError(f"MCTS does not support game: {game_name}")

    def _search_go(self, states, nn_callback, num_simulations=None, add_noise=True):
        """Wrapper to call Rust search_go with board_size."""
        return self._rust_mcts.search_go(
            states, nn_callback,
            board_size=self._go_board_size,
            num_simulations=num_simulations,
            add_noise=add_noise
        )

    def clear_cache(self):
        """Clear the transposition table.

        Call this when the model is retrained to invalidate cached evaluations.
        """
        self._rust_mcts.clear_cache()

    def cache_stats(self) -> Tuple[int, int, int]:
        """Get transposition table statistics.

        Returns:
            Tuple of (hits, misses, num_entries)
        """
        return self._rust_mcts.cache_stats()

    def search(
        self,
        states: np.ndarray,
        model: torch.nn.Module,
        num_simulations: Optional[int] = None,
        add_noise: bool = True,
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Run batched MCTS on multiple states using Rust backend.

        The entire search runs in Rust, only calling back to Python for
        neural network inference. This minimizes boundary crossings.

        Args:
            states: Batch of game states (shape is game-defined)
            model: Neural network for policy/value prediction
            num_simulations: Number of MCTS simulations per state
            add_noise: Whether to add Dirichlet noise at roots
            batch_size: Not used (kept for API compatibility)

        Returns:
            Policy array of shape (B, action_size)
        """
        if num_simulations is None:
            num_simulations = self.config.num_simulations

        # Store model and device for callback
        self._model = model
        self._device = next(model.parameters()).device

        # Flatten states for Rust: (batch, h, w) -> (batch, h*w)
        batch_size = states.shape[0]
        flat_states = states.reshape(batch_size, -1)
        flat_states = np.ascontiguousarray(flat_states, dtype=np.int8)

        # Run search in Rust with callback to Python for NN inference
        policies = self._search_fn(
            flat_states,
            self._nn_callback,
            num_simulations=num_simulations,
            add_noise=add_noise
        )

        return np.array(policies)

    def _nn_callback(self, states: np.ndarray, legal_masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Callback for Rust MCTS to get neural network predictions.

        Args:
            states: Batch of state tensors (already canonical, mapped to 0,1,2)
            legal_masks: Batch of legal action masks

        Returns:
            Tuple of (policies, values) as numpy arrays
        """
        # Convert to torch tensors
        state_tensors = torch.from_numpy(states).long().to(self._device)
        action_masks = torch.from_numpy(legal_masks).float().to(self._device)

        # Get predictions
        with torch.inference_mode():
            pred_policies, pred_values = self._model.predict(state_tensors, action_masks)

        policies = pred_policies.cpu().numpy().astype(np.float32)
        values = pred_values.cpu().numpy().flatten().astype(np.float32)

        return policies, values


class BayesianMCTS:
    """
    Bayesian MCTS using Rust backend for tree operations.

    Uses Gaussian beliefs and Top-Two Thompson Sampling with IDS allocation.
    Runs the entire search in Rust, only calling back to Python for
    neural network inference.
    """

    def __init__(self, game: Game, config: BayesianMCTSConfig, leaves_per_batch: int = 0,
                 virtual_loss: float = 1.0, use_transposition_table: bool = True,
                 seed: Optional[int] = None):
        """
        Initialize Rust-backed Bayesian MCTS.

        Args:
            game: Game instance
            config: BayesianMCTSConfig configuration
            leaves_per_batch: Number of leaves to collect per NN call (0 = auto)
            virtual_loss: Virtual loss value for penalizing in-flight paths
            use_transposition_table: Whether to use transposition table with symmetry support
            seed: Random seed for reproducibility
        """
        self.game = game
        self.config = config
        self._model = None
        self._device = None
        self.backend = 'rust'
        self._use_transposition_table = use_transposition_table

        # Create Rust Bayesian MCTS instance
        self._rust_mcts = _RustBayesianMCTS(
            num_simulations=config.num_simulations,
            sigma_0=config.sigma_0,
            obs_var=config.obs_var,
            ids_alpha=config.ids_alpha,
            prune_threshold=config.prune_threshold,
            early_stopping=config.early_stopping,
            confidence_threshold=config.confidence_threshold,
            min_simulations=config.min_simulations,
            min_variance=config.min_variance,
            leaves_per_batch=leaves_per_batch,
            virtual_loss_value=virtual_loss,
            seed=seed,
            use_transposition_table=use_transposition_table,
        )

        # Determine which game-specific search method to use
        game_name = type(game).__name__.lower()
        if 'tictactoe' in game_name:
            self._search_fn = self._rust_mcts.search_tictactoe
            self._is_go = False
        elif 'connect4' in game_name:
            self._search_fn = self._rust_mcts.search_connect4
            self._is_go = False
        elif 'go' in game_name:
            self._go_board_size = getattr(game, 'height', 9)
            self._search_fn = self._search_go
            self._is_go = True
        else:
            raise ValueError(f"Bayesian MCTS does not support game: {game_name}")

    def _search_go(self, states, nn_callback, num_simulations=None):
        """Wrapper to call Rust search_go with board_size."""
        return self._rust_mcts.search_go(
            states, nn_callback,
            board_size=self._go_board_size,
            num_simulations=num_simulations
        )

    def clear_cache(self):
        """Clear the transposition table.

        Call this when the model is retrained to invalidate cached evaluations.
        """
        self._rust_mcts.clear_cache()

    def cache_stats(self) -> Tuple[int, int, int]:
        """Get transposition table statistics.

        Returns:
            Tuple of (hits, misses, num_entries)
        """
        return self._rust_mcts.cache_stats()

    def search(
        self,
        states: np.ndarray,
        model: torch.nn.Module,
        num_simulations: Optional[int] = None
    ) -> np.ndarray:
        """
        Run Bayesian MCTS on multiple states using Rust backend.

        The entire search runs in Rust, only calling back to Python for
        neural network inference. Uses Thompson sampling for exploration.

        Args:
            states: Batch of game states (shape is game-defined)
            model: Neural network for policy/value prediction
            num_simulations: Number of MCTS simulations per state

        Returns:
            Policy array of shape (B, action_size)
        """
        if num_simulations is None:
            num_simulations = self.config.num_simulations

        # Store model and device for callback
        self._model = model
        self._device = next(model.parameters()).device

        # Flatten states for Rust: (batch, h, w) -> (batch, h*w)
        batch_size = states.shape[0]
        flat_states = states.reshape(batch_size, -1)
        flat_states = np.ascontiguousarray(flat_states, dtype=np.int8)

        # Run search in Rust with callback to Python for NN inference
        policies = self._search_fn(
            flat_states,
            self._nn_callback,
            num_simulations=num_simulations
        )

        return np.array(policies)

    def _nn_callback(self, states: np.ndarray, legal_masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Callback for Rust MCTS to get neural network predictions.

        Args:
            states: Batch of state tensors (already canonical, mapped to 0,1,2)
            legal_masks: Batch of legal action masks

        Returns:
            Tuple of (policies, values) as numpy arrays
        """
        # Convert to torch tensors
        state_tensors = torch.from_numpy(states).long().to(self._device)
        action_masks = torch.from_numpy(legal_masks).float().to(self._device)

        # Get predictions
        with torch.inference_mode():
            pred_policies, pred_values = self._model.predict(state_tensors, action_masks)

        policies = pred_policies.cpu().numpy().astype(np.float32)
        values = pred_values.cpu().numpy().flatten().astype(np.float32)

        return policies, values


# Backwards compatibility aliases
RustBatchedMCTS = BatchedMCTS
RustBayesianMCTS = BayesianMCTS
MCTS = BatchedMCTS  # For scripts that import MCTS directly


# ============================================================================
# Factory Functions
# ============================================================================

def get_batched_mcts(game: Game, config: MCTSConfig, use_rust: bool = True, **kwargs) -> BatchedMCTS:
    """
    Get a BatchedMCTS instance.

    Args:
        game: Game instance
        config: MCTS configuration
        use_rust: Ignored (Rust is always used). Kept for API compatibility.
        **kwargs: Additional arguments passed to constructor

    Returns:
        BatchedMCTS instance
    """
    return BatchedMCTS(game, config, **kwargs)


def get_bayesian_mcts(game: Game, config: BayesianMCTSConfig, use_rust: bool = True, **kwargs) -> BayesianMCTS:
    """
    Get a BayesianMCTS instance.

    Args:
        game: Game instance
        config: BayesianMCTSConfig configuration
        use_rust: Ignored (Rust is always used). Kept for API compatibility.
        **kwargs: Additional arguments passed to constructor

    Returns:
        BayesianMCTS instance
    """
    return BayesianMCTS(game, config, **kwargs)


def is_rust_mcts_available() -> bool:
    """Check if the Rust MCTS backend is available. Always True since Rust is required."""
    return True


# Re-export MCTSConfig and BayesianMCTSConfig for convenience
MCTSConfig = MCTSConfig
BayesianMCTSConfig = BayesianMCTSConfig
