"""
nanozero/game.py - Game interface and Rust implementations

This module provides the Game interface and Rust-backed game implementations.
The Rust backend (nanozero_mcts_rs) is required.
"""
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import List, Tuple
from nanozero.config import GameConfig

# Import Rust backend (required)
from nanozero_mcts_rs import RustTicTacToe, RustConnect4, RustGo

# For backwards compatibility
RUST_AVAILABLE = True
HAS_RUST_GAMES = True


class Game(ABC):
    """Abstract base class for two-player zero-sum games."""

    def __init__(self, config: GameConfig):
        self.config = config

    @abstractmethod
    def initial_state(self) -> np.ndarray:
        """Return the initial game state."""
        pass

    @abstractmethod
    def current_player(self, state: np.ndarray) -> int:
        """Return 1 if player 1's turn, -1 if player 2's turn."""
        pass

    @abstractmethod
    def legal_actions(self, state: np.ndarray) -> List[int]:
        """Return list of legal action indices."""
        pass

    @abstractmethod
    def next_state(self, state: np.ndarray, action: int) -> np.ndarray:
        """Apply action and return new state (copy)."""
        pass

    @abstractmethod
    def is_terminal(self, state: np.ndarray) -> bool:
        """Check if game has ended."""
        pass

    @abstractmethod
    def terminal_reward(self, state: np.ndarray) -> float:
        """Get reward at terminal state from current player's perspective."""
        pass

    def canonical_state(self, state: np.ndarray) -> np.ndarray:
        """Convert state so current player's pieces are +1."""
        player = self.current_player(state)
        if player == 1:
            return state.copy()
        return -state.copy()

    def to_tensor(self, state: np.ndarray) -> torch.Tensor:
        """Convert canonical state to tensor. Maps {-1,0,1} to {0,1,2}."""
        flat = state.flatten()
        tokens = (flat + 1).astype(np.int64)
        return torch.from_numpy(tokens)

    @abstractmethod
    def symmetries(self, state: np.ndarray, policy: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return symmetric equivalents for data augmentation."""
        pass

    def legal_actions_mask(self, state: np.ndarray) -> np.ndarray:
        """Return binary mask of legal actions."""
        mask = np.zeros(self.config.action_size, dtype=np.float32)
        for action in self.legal_actions(state):
            mask[action] = 1.0
        return mask

    def display(self, state: np.ndarray) -> str:
        """Return string representation of board."""
        pass


# ============================================================================
# Rust-backed Game Implementations
# ============================================================================

class TicTacToe(Game):
    """TicTacToe using Rust backend with Python Game interface."""

    def __init__(self, config: GameConfig):
        super().__init__(config)
        self._rust = RustTicTacToe()
        self.backend = 'rust'

    def initial_state(self) -> np.ndarray:
        return self._rust.initial_state().reshape(3, 3)

    def current_player(self, state: np.ndarray) -> int:
        return int(self._rust.current_player(state.flatten()))

    def legal_actions(self, state: np.ndarray) -> List[int]:
        return [int(a) for a in self._rust.legal_actions(state.flatten())]

    def legal_actions_mask(self, state: np.ndarray) -> np.ndarray:
        mask = np.array(self._rust.legal_actions_mask(state.flatten()), dtype=np.float32)
        return mask

    def next_state(self, state: np.ndarray, action: int) -> np.ndarray:
        return self._rust.next_state(state.flatten(), action).reshape(3, 3)

    def is_terminal(self, state: np.ndarray) -> bool:
        return self._rust.is_terminal(state.flatten())

    def terminal_reward(self, state: np.ndarray) -> float:
        return float(self._rust.terminal_reward(state.flatten()))

    def canonical_state(self, state: np.ndarray) -> np.ndarray:
        return self._rust.canonical_state(state.flatten()).reshape(3, 3)

    def to_tensor(self, state: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(self._rust.to_tensor(state.flatten()))

    def symmetries(self, state: np.ndarray, policy: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        result = []
        pb = policy.reshape(3, 3)
        for i in range(4):
            rs = np.rot90(state, i)
            rp = np.rot90(pb, i)
            result.append((rs.copy(), rp.flatten().copy()))
            result.append((np.fliplr(rs).copy(), np.fliplr(rp).flatten().copy()))
        return result

    def display(self, state: np.ndarray) -> str:
        return self._rust.render(state.flatten())


class Connect4(Game):
    """Connect4 using Rust backend with Python Game interface."""

    def __init__(self, config: GameConfig):
        super().__init__(config)
        self._rust = RustConnect4()
        self.backend = 'rust'

    def initial_state(self) -> np.ndarray:
        return self._rust.initial_state().reshape(6, 7)

    def current_player(self, state: np.ndarray) -> int:
        return int(self._rust.current_player(state.flatten()))

    def legal_actions(self, state: np.ndarray) -> List[int]:
        return [int(a) for a in self._rust.legal_actions(state.flatten())]

    def legal_actions_mask(self, state: np.ndarray) -> np.ndarray:
        mask = np.array(self._rust.legal_actions_mask(state.flatten()), dtype=np.float32)
        return mask

    def next_state(self, state: np.ndarray, action: int) -> np.ndarray:
        return self._rust.next_state(state.flatten(), action).reshape(6, 7)

    def is_terminal(self, state: np.ndarray) -> bool:
        return self._rust.is_terminal(state.flatten())

    def terminal_reward(self, state: np.ndarray) -> float:
        return float(self._rust.terminal_reward(state.flatten()))

    def canonical_state(self, state: np.ndarray) -> np.ndarray:
        return self._rust.canonical_state(state.flatten()).reshape(6, 7)

    def to_tensor(self, state: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(self._rust.to_tensor(state.flatten()))

    def symmetries(self, state: np.ndarray, policy: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        return [
            (state.copy(), policy.copy()),
            (np.fliplr(state).copy(), np.flip(policy).copy()),
        ]

    def display(self, state: np.ndarray) -> str:
        return self._rust.render(state.flatten())


class Go(Game):
    """Go using Rust backend with Python Game interface."""

    def __init__(self, config: GameConfig):
        super().__init__(config)
        self.height = config.board_height
        self.width = config.board_width
        self._rust = RustGo(self.height)
        self.backend = 'rust'

    def initial_state(self) -> np.ndarray:
        return np.array(self._rust.initial_state(), dtype=np.int8)

    def current_player(self, state: np.ndarray) -> int:
        return int(self._rust.current_player(state))

    def legal_actions(self, state: np.ndarray) -> List[int]:
        return [int(a) for a in self._rust.legal_actions(state)]

    def legal_actions_mask(self, state: np.ndarray) -> np.ndarray:
        mask = np.array(self._rust.legal_actions_mask(state), dtype=np.float32)
        return mask

    def next_state(self, state: np.ndarray, action: int) -> np.ndarray:
        return np.array(self._rust.next_state(state, action), dtype=np.int8)

    def is_terminal(self, state: np.ndarray) -> bool:
        return self._rust.is_terminal(state)

    def terminal_reward(self, state: np.ndarray) -> float:
        return float(self._rust.terminal_reward(state))

    def canonical_state(self, state: np.ndarray) -> np.ndarray:
        return np.array(self._rust.canonical_state(state), dtype=np.int8)

    def to_tensor(self, state: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(self._rust.to_tensor(state))

    def symmetries(self, state: np.ndarray, policy: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        board = state[:self.height, :]
        metadata = state[self.height:, :]
        pass_action = self.height * self.width
        policy_board = policy[:pass_action].reshape(self.height, self.width)
        pass_prob = policy[pass_action]

        results = []
        for i in range(4):
            rot_board = np.rot90(board, i)
            rot_policy = np.rot90(policy_board, i)
            rot_state = np.vstack([rot_board, metadata])
            rot_policy_flat = np.append(rot_policy.flatten(), pass_prob)
            results.append((rot_state.copy(), rot_policy_flat.copy()))

            flip_board = np.fliplr(rot_board)
            flip_policy = np.fliplr(rot_policy)
            flip_state = np.vstack([flip_board, metadata])
            flip_policy_flat = np.append(flip_policy.flatten(), pass_prob)
            results.append((flip_state.copy(), flip_policy_flat.copy()))

        return results

    def display(self, state: np.ndarray) -> str:
        return self._rust.render(state)


# Backwards compatibility aliases
RustTicTacToeWrapper = TicTacToe
RustConnect4Wrapper = Connect4
RustGoWrapper = Go


def get_game(name: str, use_rust: bool = True) -> Game:
    """
    Factory function to get a game instance.

    Args:
        name: Game name ('tictactoe', 'connect4', 'go9x9', 'go19x19')
        use_rust: Ignored (Rust is always used). Kept for API compatibility.

    Returns:
        Game instance
    """
    from nanozero.config import get_game_config
    config = get_game_config(name)

    games = {
        'tictactoe': TicTacToe,
        'connect4': Connect4,
        'go9x9': Go,
        'go19x19': Go,
    }
    if name not in games:
        raise ValueError(f"Unknown game: {name}")

    game = games[name](config)
    return game


def is_rust_available() -> bool:
    """Check if the Rust backend is available. Always True since Rust is required."""
    return True
