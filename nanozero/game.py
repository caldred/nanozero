"""
nanozero/game.py - Game interface and implementations
"""
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import List, Tuple
from nanozero.config import GameConfig

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


class TicTacToe(Game):
    """3x3 Tic-Tac-Toe. Actions 0-8 map to positions row-major."""

    def initial_state(self) -> np.ndarray:
        return np.zeros((3, 3), dtype=np.int8)

    def current_player(self, state: np.ndarray) -> int:
        p1 = np.sum(state == 1)
        p2 = np.sum(state == -1)
        return 1 if p1 == p2 else -1

    def legal_actions(self, state: np.ndarray) -> List[int]:
        return [i for i in range(9) if state.flat[i] == 0]

    def next_state(self, state: np.ndarray, action: int) -> np.ndarray:
        if action not in self.legal_actions(state):
            raise ValueError(f"Illegal action {action}")
        new = state.copy()
        new.flat[action] = self.current_player(state)
        return new

    def _check_winner(self, state: np.ndarray) -> int:
        """Return 1, -1, or 0."""
        for i in range(3):
            if abs(state[i, :].sum()) == 3:
                return int(np.sign(state[i, 0]))
            if abs(state[:, i].sum()) == 3:
                return int(np.sign(state[0, i]))
        d1 = state[0,0] + state[1,1] + state[2,2]
        d2 = state[0,2] + state[1,1] + state[2,0]
        if abs(d1) == 3: return int(np.sign(d1))
        if abs(d2) == 3: return int(np.sign(d2))
        return 0

    def is_terminal(self, state: np.ndarray) -> bool:
        return self._check_winner(state) != 0 or len(self.legal_actions(state)) == 0

    def terminal_reward(self, state: np.ndarray) -> float:
        winner = self._check_winner(state)
        if winner == 0:
            return 0.0
        return float(winner * self.current_player(state))

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
        sym = {-1: 'O', 0: '.', 1: 'X'}
        lines = []
        for r in range(3):
            lines.append(' | '.join(sym[state[r, c]] for c in range(3)))
        return '\n---------\n'.join(lines)


class Connect4(Game):
    """Connect4 on 6x7 board. Actions 0-6 are column drops."""

    def initial_state(self) -> np.ndarray:
        return np.zeros((6, 7), dtype=np.int8)

    def current_player(self, state: np.ndarray) -> int:
        p1 = np.sum(state == 1)
        p2 = np.sum(state == -1)
        return 1 if p1 == p2 else -1

    def legal_actions(self, state: np.ndarray) -> List[int]:
        return [c for c in range(7) if state[0, c] == 0]

    def _drop_row(self, state: np.ndarray, col: int) -> int:
        for r in range(5, -1, -1):
            if state[r, col] == 0:
                return r
        return -1

    def next_state(self, state: np.ndarray, action: int) -> np.ndarray:
        if action not in self.legal_actions(state):
            raise ValueError(f"Illegal action {action}")
        new = state.copy()
        row = self._drop_row(state, action)
        new[row, action] = self.current_player(state)
        return new

    def _check_winner(self, state: np.ndarray) -> int:
        # Horizontal
        for r in range(6):
            for c in range(4):
                w = state[r, c:c+4]
                if abs(w.sum()) == 4:
                    return int(np.sign(w[0]))
        # Vertical
        for r in range(3):
            for c in range(7):
                w = state[r:r+4, c]
                if abs(w.sum()) == 4:
                    return int(np.sign(w[0]))
        # Diagonals
        for r in range(3):
            for c in range(4):
                w = [state[r+i, c+i] for i in range(4)]
                if abs(sum(w)) == 4:
                    return int(np.sign(w[0]))
        for r in range(3, 6):
            for c in range(4):
                w = [state[r-i, c+i] for i in range(4)]
                if abs(sum(w)) == 4:
                    return int(np.sign(w[0]))
        return 0

    def is_terminal(self, state: np.ndarray) -> bool:
        return self._check_winner(state) != 0 or len(self.legal_actions(state)) == 0

    def terminal_reward(self, state: np.ndarray) -> float:
        winner = self._check_winner(state)
        if winner == 0:
            return 0.0
        return float(winner * self.current_player(state))

    def symmetries(self, state: np.ndarray, policy: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        return [
            (state.copy(), policy.copy()),
            (np.fliplr(state).copy(), np.flip(policy).copy()),
        ]

    def display(self, state: np.ndarray) -> str:
        sym = {-1: 'O', 0: '.', 1: 'X'}
        lines = []
        for r in range(6):
            lines.append(' '.join(sym[state[r, c]] for c in range(7)))
        lines.append('0 1 2 3 4 5 6')
        return '\n'.join(lines)


def get_game(name: str) -> Game:
    """Factory function to get a game instance."""
    from nanozero.config import get_game_config
    config = get_game_config(name)
    games = {'tictactoe': TicTacToe, 'connect4': Connect4}
    if name not in games:
        raise ValueError(f"Unknown game: {name}")
    return games[name](config)
