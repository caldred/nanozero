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


class Go(Game):
    """
    Go game implementation with Chinese rules.

    State shape: (board_height + 1, board_width).
    Actions: 0 to height*width-1 are board positions, height*width is pass.
    """

    # Metadata indices
    META_PASSES = 0
    META_KO_ROW = 1
    META_KO_COL = 2
    META_TURN = 3  # 0 = black's turn, 1 = white's turn (avoids int8 overflow)

    def __init__(self, config: GameConfig):
        super().__init__(config)
        self.height = config.board_height
        self.width = config.board_width
        self.komi = 7.5  # White's compensation

    def initial_state(self) -> np.ndarray:
        """Return empty board with metadata row."""
        state = np.zeros((self.height + 1, self.width), dtype=np.int8)
        # Initialize ko position to invalid (-1)
        state[self.height, self.META_KO_ROW] = -1
        state[self.height, self.META_KO_COL] = -1
        return state

    def _get_board(self, state: np.ndarray) -> np.ndarray:
        """Get board portion of state (view, not copy)."""
        return state[:self.height, :]

    def _get_metadata(self, state: np.ndarray) -> np.ndarray:
        """Get metadata row (view, not copy)."""
        return state[self.height, :]

    def _get_ko_point(self, state: np.ndarray) -> tuple:
        """Get ko point or None if no ko."""
        meta = self._get_metadata(state)
        row, col = int(meta[self.META_KO_ROW]), int(meta[self.META_KO_COL])
        if row == -1:
            return None
        return (row, col)

    def current_player(self, state: np.ndarray) -> int:
        """Black (1) plays first, then alternates."""
        turn = int(state[self.height, self.META_TURN])
        return 1 if turn == 0 else -1

    def _neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get valid orthogonal neighbors."""
        result = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                result.append((nr, nc))
        return result

    def _find_group(self, board: np.ndarray, row: int, col: int) -> Tuple[set, set]:
        """
        Find all stones in the group containing (row, col).
        Returns: (group positions, liberty positions)
        """
        color = board[row, col]
        if color == 0:
            return set(), set()

        group = set()
        liberties = set()
        stack = [(row, col)]

        while stack:
            r, c = stack.pop()
            if (r, c) in group:
                continue
            group.add((r, c))

            for nr, nc in self._neighbors(r, c):
                if board[nr, nc] == 0:
                    liberties.add((nr, nc))
                elif board[nr, nc] == color and (nr, nc) not in group:
                    stack.append((nr, nc))

        return group, liberties

    def _get_captures(self, board: np.ndarray, row: int, col: int, player: int) -> List[Tuple[int, int]]:
        """
        Find opponent stones captured by placing a stone at (row, col).
        Must be called AFTER placing the stone on the board.
        """
        captures = []
        opponent = -player

        for nr, nc in self._neighbors(row, col):
            if board[nr, nc] == opponent:
                group, liberties = self._find_group(board, nr, nc)
                if len(liberties) == 0:
                    captures.extend(group)

        return captures

    def _is_suicide(self, board: np.ndarray, row: int, col: int, player: int) -> bool:
        """Check if placing stone would be suicide (no liberties, no captures)."""
        # Temporarily place stone
        board[row, col] = player

        # Check if we capture anything
        captures = self._get_captures(board, row, col, player)
        if captures:
            board[row, col] = 0  # Restore
            return False

        # Check if our group has liberties
        group, liberties = self._find_group(board, row, col)
        board[row, col] = 0  # Restore
        return len(liberties) == 0

    def _find_all_groups(self, board: np.ndarray, color: int) -> List[Tuple[set, set]]:
        """Find all groups of a given color and their liberties."""
        visited = set()
        groups = []

        for r in range(self.height):
            for c in range(self.width):
                if board[r, c] == color and (r, c) not in visited:
                    group, liberties = self._find_group(board, r, c)
                    visited.update(group)
                    groups.append((group, liberties))

        return groups

    def legal_actions(self, state: np.ndarray) -> List[int]:
        """
        Return list of legal actions.
        Uses optimization: precompute capture points to avoid suicide checks.
        """
        board = self._get_board(state).copy()  # Copy for temporary modifications
        ko_point = self._get_ko_point(state)
        player = self.current_player(state)
        opponent = -player

        # Find all opponent groups with exactly 1 liberty (capture points)
        opponent_groups = self._find_all_groups(board, opponent)
        capture_points = set()
        for group, liberties in opponent_groups:
            if len(liberties) == 1:
                capture_points.update(liberties)

        legal = []
        pass_action = self.height * self.width

        for action in range(pass_action):
            row, col = action // self.width, action % self.width

            # Must be empty
            if board[row, col] != 0:
                continue

            # Can't play on ko point
            if ko_point and (row, col) == ko_point:
                continue

            # If this captures something, it's definitely legal (not suicide)
            if (row, col) in capture_points:
                legal.append(action)
                continue

            # Check for suicide
            if not self._is_suicide(board, row, col, player):
                legal.append(action)

        # Pass is always legal
        legal.append(pass_action)
        return legal

    def next_state(self, state: np.ndarray, action: int) -> np.ndarray:
        """Apply action and return new state."""
        new_state = state.copy()
        board = self._get_board(new_state)
        meta = self._get_metadata(new_state)
        player = self.current_player(state)
        pass_action = self.height * self.width

        if action == pass_action:
            # Pass move
            meta[self.META_PASSES] += 1
            meta[self.META_KO_ROW] = -1  # Clear ko
            meta[self.META_KO_COL] = -1
        else:
            # Place stone
            row, col = action // self.width, action % self.width
            board[row, col] = player

            # Remove captures
            captures = self._get_captures(board, row, col, player)
            for r, c in captures:
                board[r, c] = 0

            # Check for ko
            if len(captures) == 1:
                cap_r, cap_c = captures[0]
                # Ko if capturing stone has exactly one liberty (the captured pos)
                _, liberties = self._find_group(board, row, col)
                if len(liberties) == 1 and (cap_r, cap_c) in liberties:
                    meta[self.META_KO_ROW] = cap_r
                    meta[self.META_KO_COL] = cap_c
                else:
                    meta[self.META_KO_ROW] = -1
                    meta[self.META_KO_COL] = -1
            else:
                meta[self.META_KO_ROW] = -1
                meta[self.META_KO_COL] = -1

            # Reset consecutive passes
            meta[self.META_PASSES] = 0

        # Toggle turn
        meta[self.META_TURN] = 1 - meta[self.META_TURN]
        return new_state

    def is_terminal(self, state: np.ndarray) -> bool:
        """Game ends after two consecutive passes."""
        return int(state[self.height, self.META_PASSES]) >= 2

    def _flood_fill_empty(self, board: np.ndarray, row: int, col: int,
                          visited: np.ndarray) -> Tuple[List[Tuple[int, int]], set]:
        """Flood fill empty region, returning region and border colors."""
        region = []
        borders = set()
        stack = [(row, col)]

        while stack:
            r, c = stack.pop()
            if visited[r, c]:
                continue
            if board[r, c] != 0:
                borders.add((r, c))
                continue

            visited[r, c] = True
            region.append((r, c))

            for nr, nc in self._neighbors(r, c):
                if not visited[nr, nc]:
                    stack.append((nr, nc))

        return region, borders

    def _score(self, board: np.ndarray) -> Tuple[float, float]:
        """
        Chinese scoring: stones + territory.
        Returns (black_score, white_score).
        """
        black_score = float(np.sum(board == 1))
        white_score = float(np.sum(board == -1))

        visited = np.zeros((self.height, self.width), dtype=bool)

        for r in range(self.height):
            for c in range(self.width):
                if board[r, c] == 0 and not visited[r, c]:
                    region, borders = self._flood_fill_empty(board, r, c, visited)

                    if not borders:
                        continue

                    border_colors = set(board[br, bc] for br, bc in borders)

                    if border_colors == {1}:  # Only black borders
                        black_score += len(region)
                    elif border_colors == {-1}:  # Only white borders
                        white_score += len(region)
                    # Mixed borders = neutral (dame)

        return black_score, white_score

    def terminal_reward(self, state: np.ndarray) -> float:
        """
        Return reward from current player's perspective.
        Positive if current player wins, negative if loses.
        """
        board = self._get_board(state)
        black, white = self._score(board)
        white += self.komi

        # +1 if black wins, -1 if white wins
        if black > white:
            result = 1.0
        elif white > black:
            result = -1.0
        else:
            result = 0.0

        player = self.current_player(state)
        return result * player

    def to_tensor(self, state: np.ndarray) -> torch.Tensor:
        """Convert canonical board state to tensor, excluding metadata row."""
        board = state[:self.height, :]  # Exclude metadata row
        flat = board.flatten()
        tokens = (flat + 1).astype(np.int64)
        return torch.from_numpy(tokens)

    def symmetries(self, state: np.ndarray, policy: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return 8 symmetric transformations (4 rotations x 2 flips)."""
        board = state[:self.height, :]
        metadata = state[self.height:, :]
        pass_action = self.height * self.width
        policy_board = policy[:pass_action].reshape(self.height, self.width)
        pass_prob = policy[pass_action]

        results = []
        for i in range(4):
            rot_board = np.rot90(board, i)
            rot_policy = np.rot90(policy_board, i)

            # Reconstruct full state with metadata
            rot_state = np.vstack([rot_board, metadata])
            rot_policy_flat = np.append(rot_policy.flatten(), pass_prob)
            results.append((rot_state.copy(), rot_policy_flat.copy()))

            # Flip
            flip_board = np.fliplr(rot_board)
            flip_policy = np.fliplr(rot_policy)
            flip_state = np.vstack([flip_board, metadata])
            flip_policy_flat = np.append(flip_policy.flatten(), pass_prob)
            results.append((flip_state.copy(), flip_policy_flat.copy()))

        return results

    def display(self, state: np.ndarray) -> str:
        """Return string representation of board."""
        board = self._get_board(state)
        sym = {-1: 'O', 0: '.', 1: 'X'}

        lines = []
        # Column labels
        col_labels = '   ' + ' '.join(chr(ord('A') + c) if c < 8 else chr(ord('A') + c + 1)
                                       for c in range(self.width))
        lines.append(col_labels)

        for r in range(self.height):
            row_num = self.height - r
            row_str = f"{row_num:2d} " + ' '.join(sym[board[r, c]] for c in range(self.width)) + f" {row_num:2d}"
            lines.append(row_str)

        lines.append(col_labels)

        # Add game info
        meta = self._get_metadata(state)
        player = self.current_player(state)
        player_str = "Black (X)" if player == 1 else "White (O)"
        ko = self._get_ko_point(state)
        ko_str = f"Ko: {chr(ord('A') + ko[1])}{self.height - ko[0]}" if ko else "No ko"
        lines.append(f"\nTo play: {player_str} | {ko_str}")

        return '\n'.join(lines)


def get_game(name: str, use_rust: bool = True) -> Game:
    """
    Factory function to get a game instance.

    Args:
        name: Game name ('tictactoe', 'connect4', 'go9x9', 'go19x19')
        use_rust: Whether to use Rust implementation if available (default True)

    Returns:
        Game instance
    """
    from nanozero.config import get_game_config
    config = get_game_config(name)

    # Try Rust implementations if requested
    if use_rust and RUST_AVAILABLE:
        rust_games = {
            'tictactoe': lambda cfg: RustTicTacToeWrapper(cfg),
            'connect4': lambda cfg: RustConnect4Wrapper(cfg),
            'go9x9': lambda cfg: RustGoWrapper(cfg),
            'go19x19': lambda cfg: RustGoWrapper(cfg),
        }
        if name in rust_games:
            return rust_games[name](config)

    # Fall back to Python implementations
    python_games = {
        'tictactoe': TicTacToe,
        'connect4': Connect4,
        'go9x9': Go,
        'go19x19': Go,
    }
    if name not in python_games:
        raise ValueError(f"Unknown game: {name}")
    return python_games[name](config)


# ============================================================================
# Rust Integration
# ============================================================================

# Try to import Rust module
try:
    from nanozero_mcts_rs import RustTicTacToe, RustConnect4, RustGo
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


class RustTicTacToeWrapper(Game):
    """TicTacToe using Rust backend with Python Game interface."""

    def __init__(self, config: GameConfig):
        super().__init__(config)
        if not RUST_AVAILABLE:
            raise ImportError("nanozero_mcts_rs not installed")
        self._rust = RustTicTacToe()

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
        # Use Python implementation for symmetries (complex to wrap)
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


class RustConnect4Wrapper(Game):
    """Connect4 using Rust backend with Python Game interface."""

    def __init__(self, config: GameConfig):
        super().__init__(config)
        if not RUST_AVAILABLE:
            raise ImportError("nanozero_mcts_rs not installed")
        self._rust = RustConnect4()

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


class RustGoWrapper(Game):
    """Go using Rust backend with Python Game interface."""

    def __init__(self, config: GameConfig):
        super().__init__(config)
        if not RUST_AVAILABLE:
            raise ImportError("nanozero_mcts_rs not installed")
        self.height = config.board_height
        self.width = config.board_width
        self._rust = RustGo(self.height)

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
        # Use Python implementation for symmetries
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
