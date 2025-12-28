# NanoZero: Complete Implementation Specification

> A minimal AlphaZero implementation in the style of Karpathy's nanoGPT/nanochat.

## Overview

NanoZero implements the AlphaZero algorithm for training game-playing AI through self-play. It uses a Transformer architecture for generality and batched MCTS for GPU efficiency.

**Target**: ~1200 lines of clean, readable Python.

**Philosophy**:
- Single slider of complexity (`--n_layer`)
- Minimal dependencies (PyTorch + NumPy only)
- Runs on single GPU (or CPU with patience)
- Educational but functional

---

## File Structure

```
nanozero/
├── nanozero/
│   ├── __init__.py           # Empty
│   ├── common.py             # Utilities
│   ├── config.py             # Configuration dataclasses
│   ├── game.py               # Game interface + TicTacToe, Connect4
│   ├── model.py              # Transformer policy-value network
│   ├── mcts.py               # Batched MCTS implementation
│   └── replay.py             # Replay buffer
├── scripts/
│   ├── train.py              # Main training loop
│   ├── eval.py               # Evaluate against baselines
│   ├── play.py               # Interactive play
│   └── arena.py              # Pit models against each other
├── tests/
│   ├── test_game.py
│   ├── test_model.py
│   ├── test_mcts.py
│   └── test_integration.py
├── speedrun.sh
├── pyproject.toml
└── README.md
```

---

# SPEC 1: `nanozero/common.py` (~50 lines)

## Purpose
Shared utilities used across all modules.

## Implementation

```python
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
```

---

# SPEC 2: `nanozero/config.py` (~80 lines)

## Purpose
Central configuration dataclasses for all components.

## Implementation

```python
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
```

---

# SPEC 3: `nanozero/game.py` (~200 lines)

## Purpose
Define the game interface and implement TicTacToe and Connect4.

## Key Conventions
- Board values: -1 (player 2), 0 (empty), 1 (player 1)
- Player 1 moves first
- `canonical_state()` ensures current player's pieces are always +1

## Implementation

```python
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
```

---

# SPEC 4: `nanozero/model.py` (~200 lines)

## Purpose
Transformer-based policy-value network.

## Architecture
```
Input (B, board_size) → Token+Pos Embed → Transformer × n_layer → 
    Policy head: mean pool → Linear → log_softmax
    Value head: mean pool → MLP → tanh
```

## Implementation

```python
"""
nanozero/model.py - Transformer policy-value network
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from nanozero.config import ModelConfig

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)

class SelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.training else 0, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(y))

class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.norm2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class AlphaZeroTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(3, config.n_embd)
        self.pos_emb = nn.Embedding(config.board_size, config.n_embd)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        self.policy_head = nn.Linear(config.n_embd, config.action_size, bias=False)
        self.value_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 4, bias=False),
            nn.GELU(),
            nn.Linear(config.n_embd // 4, 1, bias=False),
            nn.Tanh(),
        )
        self.apply(self._init_weights)
        self.n_params = sum(p.numel() for p in self.parameters())
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, std=0.02)
    
    def forward(self, x, action_mask=None):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        x = self.tok_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        pooled = x.mean(dim=1)
        logits = self.policy_head(pooled)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, float('-inf'))
        policy = F.log_softmax(logits, dim=-1)
        value = self.value_head(pooled)
        return policy, value
    
    def predict(self, x, action_mask=None):
        self.eval()
        with torch.no_grad():
            log_p, v = self.forward(x, action_mask)
            return log_p.exp(), v
    
    def count_parameters(self) -> int:
        return self.n_params
```

---

See **SPEC_PART2.md** for:
- SPEC 5: `nanozero/mcts.py`
- SPEC 6: `nanozero/replay.py`
- SPEC 7-10: Scripts (train.py, eval.py, play.py, arena.py)
- SPEC 11: speedrun.sh
- SPEC 12: Integration tests
- Implementation priority and dependencies

See **AGENTS.md** for quick reference assignment summary.
# NanoZero Specification - Part 2: Scripts and Integration

This continues from SPEC.md with the remaining components.

---

# SPEC 8: `scripts/eval.py`

## Purpose
Evaluate a trained model against various baselines.

## Implementation (~100 lines)

```python
"""
scripts/eval.py - Evaluate trained model

Usage:
    python -m scripts.eval --game=tictactoe --checkpoint=checkpoints/tictactoe_final.pt
"""
import argparse
import numpy as np
import torch

from nanozero.game import get_game
from nanozero.model import AlphaZeroTransformer
from nanozero.mcts import BatchedMCTS, sample_action
from nanozero.config import get_game_config, get_model_config, MCTSConfig
from nanozero.common import get_device, print0, load_checkpoint


def play_game(game, player1_fn, player2_fn):
    """
    Play a single game between two players.
    
    Args:
        game: Game instance
        player1_fn: Function(state) -> action for player 1
        player2_fn: Function(state) -> action for player 2
    
    Returns:
        1 if player1 wins, -1 if player2 wins, 0 for draw
    """
    state = game.initial_state()
    
    while not game.is_terminal(state):
        current = game.current_player(state)
        if current == 1:
            action = player1_fn(state)
        else:
            action = player2_fn(state)
        state = game.next_state(state, action)
    
    # Get result from player 1's perspective
    reward = game.terminal_reward(state)
    final_player = game.current_player(state)
    if final_player == 1:
        return -reward  # Player 2 just moved
    else:
        return reward  # Player 1 just moved


def make_random_player(game):
    """Create a random player function."""
    def play(state):
        legal = game.legal_actions(state)
        return np.random.choice(legal)
    return play


def make_mcts_player(game, model, mcts, num_simulations=50):
    """Create an MCTS player function."""
    def play(state):
        canonical = game.canonical_state(state)
        policy = mcts.search(
            canonical[np.newaxis, ...],
            model,
            num_simulations=num_simulations,
            add_noise=False
        )[0]
        return sample_action(policy, temperature=0)
    return play


def make_greedy_player(game, model):
    """Create a greedy player that uses raw network policy (no MCTS)."""
    def play(state):
        canonical = game.canonical_state(state)
        state_tensor = game.to_tensor(canonical).unsqueeze(0)
        action_mask = torch.from_numpy(
            game.legal_actions_mask(canonical)
        ).unsqueeze(0).float()
        
        device = next(model.parameters()).device
        state_tensor = state_tensor.to(device)
        action_mask = action_mask.to(device)
        
        policy, _ = model.predict(state_tensor, action_mask)
        return int(policy.argmax().item())
    return play


def evaluate(
    game,
    model,
    mcts,
    opponent_type: str = 'random',
    num_games: int = 100,
    mcts_simulations: int = 50
) -> dict:
    """
    Evaluate model against an opponent.
    
    Args:
        game: Game instance
        model: Neural network
        mcts: MCTS instance
        opponent_type: 'random', 'greedy', or 'mcts'
        num_games: Number of games to play
        mcts_simulations: Simulations for MCTS player
    
    Returns:
        Dict with win/draw/loss counts and rates
    """
    model.eval()
    
    # Create players
    model_player = make_mcts_player(game, model, mcts, mcts_simulations)
    
    if opponent_type == 'random':
        opponent = make_random_player(game)
    elif opponent_type == 'greedy':
        opponent = make_greedy_player(game, model)
    elif opponent_type == 'mcts':
        opponent = make_mcts_player(game, model, mcts, mcts_simulations)
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")
    
    wins, draws, losses = 0, 0, 0
    
    for i in range(num_games):
        # Alternate who plays first
        if i % 2 == 0:
            result = play_game(game, model_player, opponent)
        else:
            result = -play_game(game, opponent, model_player)
        
        if result > 0:
            wins += 1
        elif result < 0:
            losses += 1
        else:
            draws += 1
        
        if (i + 1) % 20 == 0:
            print0(f"  Progress: {i+1}/{num_games}")
    
    return {
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'win_rate': wins / num_games,
        'draw_rate': draws / num_games,
        'loss_rate': losses / num_games,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate NanoZero model')
    parser.add_argument('--game', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--num_games', type=int, default=100)
    parser.add_argument('--mcts_simulations', type=int, default=50)
    parser.add_argument('--device', type=str, default='auto')
    
    args = parser.parse_args()
    
    device = get_device() if args.device == 'auto' else torch.device(args.device)
    
    # Load game and model
    game = get_game(args.game)
    model_config = get_model_config(game.config, n_layer=args.n_layer)
    model = AlphaZeroTransformer(model_config).to(device)
    
    load_checkpoint(args.checkpoint, model)
    model.eval()
    
    mcts_config = MCTSConfig(num_simulations=args.mcts_simulations)
    mcts = BatchedMCTS(game, mcts_config)
    
    print0(f"Evaluating {args.game} model from {args.checkpoint}")
    print0(f"Playing {args.num_games} games against each opponent\n")
    
    # Evaluate against different opponents
    for opponent in ['random', 'greedy']:
        print0(f"vs {opponent}:")
        results = evaluate(
            game, model, mcts,
            opponent_type=opponent,
            num_games=args.num_games,
            mcts_simulations=args.mcts_simulations
        )
        print0(f"  Wins: {results['wins']}, Draws: {results['draws']}, Losses: {results['losses']}")
        print0(f"  Win rate: {results['win_rate']:.1%}")
        print0()


if __name__ == '__main__':
    main()
```

---

# SPEC 9: `scripts/play.py`

## Purpose
Interactive play against trained model.

## Implementation (~100 lines)

```python
"""
scripts/play.py - Play against trained model

Usage:
    python -m scripts.play --game=tictactoe --checkpoint=checkpoints/tictactoe_final.pt
"""
import argparse
import numpy as np
import torch

from nanozero.game import get_game
from nanozero.model import AlphaZeroTransformer
from nanozero.mcts import BatchedMCTS, sample_action
from nanozero.config import get_game_config, get_model_config, MCTSConfig
from nanozero.common import get_device, load_checkpoint


def get_human_action(game, state) -> int:
    """Get action from human player."""
    legal = game.legal_actions(state)
    
    print(f"\nLegal actions: {legal}")
    
    while True:
        try:
            action = int(input("Your move: "))
            if action in legal:
                return action
            else:
                print(f"Illegal move! Choose from {legal}")
        except ValueError:
            print("Please enter a number")


def play_interactive(game, model, mcts, human_first: bool = True):
    """Play an interactive game against the model."""
    
    state = game.initial_state()
    human_player = 1 if human_first else -1
    
    print("\n" + "="*40)
    print("New game!")
    print(f"You are: {'X (first)' if human_first else 'O (second)'}")
    print("="*40)
    
    while not game.is_terminal(state):
        print("\n" + game.display(state))
        
        current = game.current_player(state)
        
        if current == human_player:
            action = get_human_action(game, state)
        else:
            print("\nAI is thinking...")
            canonical = game.canonical_state(state)
            policy = mcts.search(
                canonical[np.newaxis, ...],
                model,
                num_simulations=100,
                add_noise=False
            )[0]
            action = sample_action(policy, temperature=0)
            print(f"AI plays: {action}")
        
        state = game.next_state(state, action)
    
    # Game over
    print("\n" + game.display(state))
    print("\n" + "="*40)
    
    reward = game.terminal_reward(state)
    final_player = game.current_player(state)
    
    # Determine winner from human's perspective
    if final_player == human_player:
        human_result = -reward
    else:
        human_result = reward
    
    if human_result > 0:
        print("You win! 🎉")
    elif human_result < 0:
        print("AI wins! 🤖")
    else:
        print("It's a draw! 🤝")
    
    print("="*40)


def main():
    parser = argparse.ArgumentParser(description='Play against NanoZero')
    parser.add_argument('--game', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--device', type=str, default='auto')
    
    args = parser.parse_args()
    
    device = get_device() if args.device == 'auto' else torch.device(args.device)
    
    # Load game and model
    game = get_game(args.game)
    model_config = get_model_config(game.config, n_layer=args.n_layer)
    model = AlphaZeroTransformer(model_config).to(device)
    
    load_checkpoint(args.checkpoint, model)
    model.eval()
    
    mcts_config = MCTSConfig(num_simulations=100, temperature=0)
    mcts = BatchedMCTS(game, mcts_config)
    
    print(f"Loaded {args.game} model from {args.checkpoint}")
    print(f"Model has {model.count_parameters():,} parameters")
    
    # Play games
    while True:
        first = input("\nDo you want to go first? (y/n): ").lower()
        human_first = first != 'n'
        
        play_interactive(game, model, mcts, human_first)
        
        again = input("\nPlay again? (y/n): ").lower()
        if again != 'y':
            break
    
    print("\nThanks for playing!")


if __name__ == '__main__':
    main()
```

---

# SPEC 10: `scripts/arena.py`

## Purpose
Pit two models against each other.

## Implementation (~80 lines)

```python
"""
scripts/arena.py - Pit two models against each other

Usage:
    python -m scripts.arena --game=tictactoe \
        --model1=checkpoints/iter50.pt \
        --model2=checkpoints/iter100.pt
"""
import argparse
import numpy as np
import torch

from nanozero.game import get_game
from nanozero.model import AlphaZeroTransformer
from nanozero.mcts import BatchedMCTS, sample_action
from nanozero.config import get_game_config, get_model_config, MCTSConfig
from nanozero.common import get_device, load_checkpoint, print0


def arena(
    game,
    model1, mcts1,
    model2, mcts2,
    num_games: int = 100,
    mcts_simulations: int = 50
) -> dict:
    """
    Play games between two models.
    
    Returns results from model1's perspective.
    """
    
    def make_player(model, mcts):
        def play(state):
            canonical = game.canonical_state(state)
            policy = mcts.search(
                canonical[np.newaxis, ...],
                model,
                num_simulations=mcts_simulations,
                add_noise=False
            )[0]
            return sample_action(policy, temperature=0)
        return play
    
    player1 = make_player(model1, mcts1)
    player2 = make_player(model2, mcts2)
    
    wins, draws, losses = 0, 0, 0
    
    for i in range(num_games):
        state = game.initial_state()
        
        # Alternate who plays first
        if i % 2 == 0:
            p1_turn = 1
        else:
            p1_turn = -1
        
        while not game.is_terminal(state):
            current = game.current_player(state)
            if current == p1_turn:
                action = player1(state)
            else:
                action = player2(state)
            state = game.next_state(state, action)
        
        # Get result
        reward = game.terminal_reward(state)
        final_player = game.current_player(state)
        
        if final_player == p1_turn:
            p1_result = -reward
        else:
            p1_result = reward
        
        if p1_result > 0:
            wins += 1
        elif p1_result < 0:
            losses += 1
        else:
            draws += 1
        
        if (i + 1) % 20 == 0:
            print0(f"  Progress: {i+1}/{num_games}")
    
    return {
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'win_rate': wins / num_games,
    }


def main():
    parser = argparse.ArgumentParser(description='Arena: pit models against each other')
    parser.add_argument('--game', type=str, required=True)
    parser.add_argument('--model1', type=str, required=True)
    parser.add_argument('--model2', type=str, required=True)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--num_games', type=int, default=100)
    parser.add_argument('--mcts_simulations', type=int, default=50)
    parser.add_argument('--device', type=str, default='auto')
    
    args = parser.parse_args()
    
    device = get_device() if args.device == 'auto' else torch.device(args.device)
    
    game = get_game(args.game)
    model_config = get_model_config(game.config, n_layer=args.n_layer)
    
    # Load both models
    model1 = AlphaZeroTransformer(model_config).to(device)
    model2 = AlphaZeroTransformer(model_config).to(device)
    
    load_checkpoint(args.model1, model1)
    load_checkpoint(args.model2, model2)
    
    model1.eval()
    model2.eval()
    
    mcts_config = MCTSConfig(num_simulations=args.mcts_simulations)
    mcts1 = BatchedMCTS(game, mcts_config)
    mcts2 = BatchedMCTS(game, mcts_config)
    
    print0(f"Arena: {args.game}")
    print0(f"Model 1: {args.model1}")
    print0(f"Model 2: {args.model2}")
    print0(f"Playing {args.num_games} games...\n")
    
    results = arena(
        game, model1, mcts1, model2, mcts2,
        num_games=args.num_games,
        mcts_simulations=args.mcts_simulations
    )
    
    print0(f"\nResults (Model 1 perspective):")
    print0(f"  Wins: {results['wins']}")
    print0(f"  Draws: {results['draws']}")
    print0(f"  Losses: {results['losses']}")
    print0(f"  Win rate: {results['win_rate']:.1%}")


if __name__ == '__main__':
    main()
```

---

# SPEC 11: `speedrun.sh`

## Purpose
Train TicTacToe to strong play in ~5 minutes.

```bash
#!/bin/bash
# NanoZero Speedrun: Train TicTacToe in ~5 minutes
set -e

echo "==================================="
echo "  NanoZero Speedrun - TicTacToe"
echo "==================================="

# Setup environment
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found!"
    exit 1
fi

# Create virtual environment if needed
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q torch numpy

# Train
echo ""
echo "Starting training..."
echo "Game: TicTacToe"
echo "Config: n_layer=2, 50 iterations, 100 games/iter"
echo ""

python -m scripts.train \
    --game=tictactoe \
    --n_layer=2 \
    --num_iterations=50 \
    --games_per_iteration=100 \
    --training_steps=50 \
    --mcts_simulations=25 \
    --batch_size=64 \
    --eval_interval=10 \
    --checkpoint_interval=25

# Evaluate
echo ""
echo "Final evaluation..."
python -m scripts.eval \
    --game=tictactoe \
    --checkpoint=checkpoints/tictactoe_final.pt \
    --n_layer=2 \
    --num_games=50

echo ""
echo "==================================="
echo "  Training complete!"
echo "  Play: python -m scripts.play --game=tictactoe --checkpoint=checkpoints/tictactoe_final.pt --n_layer=2"
echo "==================================="
```

---

# SPEC 12: Integration Tests

## Purpose
End-to-end tests that verify components work together.

```python
"""
tests/test_integration.py - Integration tests
"""
import numpy as np
import torch
import tempfile
import os

from nanozero.game import TicTacToe, get_game
from nanozero.model import AlphaZeroTransformer
from nanozero.mcts import BatchedMCTS, MCTS, sample_action
from nanozero.replay import ReplayBuffer
from nanozero.config import get_game_config, get_model_config, MCTSConfig
from nanozero.common import save_checkpoint, load_checkpoint


class TestEndToEnd:
    """Test that all components work together."""
    
    def setup_method(self):
        self.game = get_game('tictactoe')
        model_config = get_model_config(self.game.config, n_layer=2)
        self.model = AlphaZeroTransformer(model_config)
        self.mcts_config = MCTSConfig(num_simulations=10)
        self.mcts = BatchedMCTS(self.game, self.mcts_config)
    
    def test_mcts_with_model(self):
        """MCTS should work with untrained model."""
        state = self.game.initial_state()
        canonical = self.game.canonical_state(state)
        
        policy = self.mcts.search(
            canonical[np.newaxis, ...],
            self.model,
            num_simulations=10
        )[0]
        
        assert policy.shape == (9,)
        assert np.isclose(policy.sum(), 1.0)
        
        # Should only assign probability to legal actions
        for i in range(9):
            if i not in self.game.legal_actions(state):
                assert policy[i] == 0
    
    def test_self_play_game(self):
        """Can play a complete self-play game."""
        state = self.game.initial_state()
        examples = []
        
        while not self.game.is_terminal(state):
            canonical = self.game.canonical_state(state)
            policy = self.mcts.search(
                canonical[np.newaxis, ...],
                self.model,
                num_simulations=5
            )[0]
            
            examples.append((canonical.copy(), policy.copy()))
            
            action = sample_action(policy)
            state = self.game.next_state(state, action)
        
        # Game should end
        assert self.game.is_terminal(state)
        # Should have collected some examples
        assert len(examples) > 0
    
    def test_training_step(self):
        """Can perform a training step."""
        # Create some fake training data
        buffer = ReplayBuffer(1000)
        
        for _ in range(100):
            state = np.random.randint(-1, 2, (3, 3), dtype=np.int8)
            policy = np.random.rand(9)
            policy /= policy.sum()
            value = np.random.uniform(-1, 1)
            buffer.push(state, policy, value)
        
        # Sample batch
        states, policies, values = buffer.sample(32)
        
        # Convert to tensors
        state_tensors = torch.stack([
            self.game.to_tensor(s) for s in states
        ])
        
        policies_tensor = torch.from_numpy(policies).float()
        values_tensor = torch.from_numpy(values).float()
        
        # Forward pass
        pred_policies, pred_values = self.model(state_tensors)
        
        # Compute loss
        policy_loss = -torch.mean(policies_tensor * pred_policies)
        value_loss = torch.nn.functional.mse_loss(
            pred_values.squeeze(-1), 
            values_tensor
        )
        loss = policy_loss + value_loss
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        for param in self.model.parameters():
            assert param.grad is not None
    
    def test_checkpoint_roundtrip(self):
        """Can save and load checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.pt")
            
            optimizer = torch.optim.Adam(self.model.parameters())
            
            # Save
            save_checkpoint(self.model, optimizer, 42, path)
            
            # Create new model
            model_config = get_model_config(self.game.config, n_layer=2)
            new_model = AlphaZeroTransformer(model_config)
            
            # Load
            iteration = load_checkpoint(path, new_model)
            
            assert iteration == 42
            
            # Check weights match
            for p1, p2 in zip(self.model.parameters(), new_model.parameters()):
                assert torch.allclose(p1, p2)
    
    def test_model_improves_with_training(self):
        """
        Sanity check: model predictions should change after training.
        (Not necessarily improve, but should change)
        """
        state = self.game.initial_state()
        state_tensor = self.game.to_tensor(state).unsqueeze(0)
        
        # Get initial prediction
        self.model.eval()
        with torch.no_grad():
            initial_policy, initial_value = self.model(state_tensor)
        
        # Do some training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
        for _ in range(10):
            self.model.train()
            pred_policy, pred_value = self.model(state_tensor)
            
            # Random target
            target_policy = torch.rand(1, 9)
            target_policy = target_policy / target_policy.sum()
            target_value = torch.tensor([0.5])
            
            loss = -torch.sum(target_policy * pred_policy) + \
                   (pred_value.squeeze() - target_value) ** 2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Get new prediction
        self.model.eval()
        with torch.no_grad():
            new_policy, new_value = self.model(state_tensor)
        
        # Predictions should have changed
        assert not torch.allclose(initial_policy, new_policy, atol=1e-3)


class TestConnect4Integration:
    """Integration tests for Connect4."""
    
    def setup_method(self):
        self.game = get_game('connect4')
        model_config = get_model_config(self.game.config, n_layer=2)
        self.model = AlphaZeroTransformer(model_config)
        self.mcts_config = MCTSConfig(num_simulations=5)
        self.mcts = BatchedMCTS(self.game, self.mcts_config)
    
    def test_mcts_with_connect4(self):
        """MCTS should work with Connect4."""
        state = self.game.initial_state()
        canonical = self.game.canonical_state(state)
        
        policy = self.mcts.search(
            canonical[np.newaxis, ...],
            self.model,
            num_simulations=5
        )[0]
        
        assert policy.shape == (7,)  # 7 columns
        assert np.isclose(policy.sum(), 1.0)
    
    def test_full_game(self):
        """Can play a complete Connect4 game."""
        state = self.game.initial_state()
        moves = 0
        
        while not self.game.is_terminal(state):
            canonical = self.game.canonical_state(state)
            policy = self.mcts.search(
                canonical[np.newaxis, ...],
                self.model,
                num_simulations=3
            )[0]
            
            action = sample_action(policy)
            state = self.game.next_state(state, action)
            moves += 1
            
            # Safety check
            assert moves <= 42  # Max moves in Connect4
        
        assert self.game.is_terminal(state)
```

---

# Implementation Priority and Dependencies

## Dependency Graph

```
common.py (no deps)
    ↓
config.py (no deps)
    ↓
game.py (depends: config)
    ↓
model.py (depends: config)
    ↓
replay.py (no deps)
    ↓
mcts.py (depends: game, model, config)
    ↓
scripts/train.py (depends: all above)
    ↓
scripts/eval.py, play.py, arena.py (depends: all above)
```

## Implementation Order for Sub-Agents

**Agent 1: Core Infrastructure**
- `nanozero/common.py`
- `nanozero/config.py`
- `tests/test_common.py` (inline above)
- `tests/test_config.py` (inline above)

**Agent 2: Game Logic**
- `nanozero/game.py`
- `tests/test_game.py` (inline above)

**Agent 3: Neural Network**
- `nanozero/model.py`
- `tests/test_model.py` (inline above)

**Agent 4: MCTS**
- `nanozero/mcts.py`
- `nanozero/replay.py`
- `tests/test_mcts.py` (inline above)
- `tests/test_replay.py` (inline above)

**Agent 5: Training & Scripts**
- `scripts/train.py`
- `scripts/eval.py`
- `scripts/play.py`
- `scripts/arena.py`
- `speedrun.sh`
- `tests/test_integration.py`

---

# Success Criteria

1. **TicTacToe Speedrun**: Training completes in <5 min on single GPU, achieves >90% win rate vs random

2. **Connect4**: Training completes in <1 hour, achieves >80% win rate vs random

3. **Code Quality**: 
   - All tests pass
   - Total lines ~1200
   - No external dependencies beyond PyTorch + NumPy

4. **Usability**:
   - `bash speedrun.sh` works out of the box
   - Can play against trained model interactively
