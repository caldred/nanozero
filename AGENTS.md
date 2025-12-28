# NanoZero: Sub-Agent Assignment Summary

Quick reference for implementation. Full specs in `FULL_SPEC.md`

## Project Overview

**Goal**: Minimal AlphaZero in ~1200 lines, Karpathy style.

**Tech Stack**: Python 3.10+, PyTorch, NumPy only.

**Target**: Train TicTacToe in 5 min, Connect4 in 1 hour.

---

## File Structure

```
nanozero/
├── nanozero/
│   ├── __init__.py       # Empty
│   ├── common.py         # Agent 1
│   ├── config.py         # Agent 1
│   ├── game.py           # Agent 2
│   ├── model.py          # Agent 3
│   ├── mcts.py           # Agent 4
│   └── replay.py         # Agent 4
├── scripts/
│   ├── train.py          # Agent 5
│   ├── eval.py           # Agent 5
│   ├── play.py           # Agent 5
│   └── arena.py          # Agent 5
├── tests/                # Each agent writes tests for their modules
├── speedrun.sh           # Agent 5
├── pyproject.toml
└── README.md
```

---

## Agent Assignments

### Agent 1: Core Infrastructure (~110 lines)

**Files**: `common.py`, `config.py`

**Key Functions**:
```python
# common.py
def get_device() -> torch.device
def set_seed(seed: int) -> None
def print0(*args, **kwargs) -> None
class AverageMeter: reset(), update(val, n=1), avg property
def save_checkpoint(model, optimizer, iteration, path) -> None
def load_checkpoint(path, model, optimizer=None) -> int

# config.py
@dataclass GameConfig: name, board_height, board_width, action_size, num_players
@dataclass ModelConfig: board_size, action_size, n_layer, n_head, n_embd, dropout
@dataclass MCTSConfig: num_simulations, c_puct, dirichlet_alpha, dirichlet_epsilon, temperature
@dataclass TrainConfig: num_iterations, games_per_iteration, training_steps, batch_size, ...
def get_game_config(name: str) -> GameConfig
def get_model_config(game_config, n_layer=4) -> ModelConfig
```

**Dependencies**: None

---

### Agent 2: Game Logic (~200 lines)

**Files**: `game.py`

**Key Classes**:
```python
class Game(ABC):
    def initial_state(self) -> np.ndarray
    def current_player(self, state) -> int  # 1 or -1
    def legal_actions(self, state) -> List[int]
    def next_state(self, state, action) -> np.ndarray
    def is_terminal(self, state) -> bool
    def terminal_reward(self, state) -> float  # -1, 0, or 1
    def canonical_state(self, state) -> np.ndarray
    def to_tensor(self, state) -> torch.Tensor
    def symmetries(self, state, policy) -> List[Tuple]
    def legal_actions_mask(self, state) -> np.ndarray
    def display(self, state) -> str

class TicTacToe(Game): ...
class Connect4(Game): ...
def get_game(name: str) -> Game
```

**Key Conventions**:
- Board values: -1 (player 2), 0 (empty), 1 (player 1)
- Player 1 moves first
- `canonical_state()` flips signs so current player always sees their pieces as +1

**Dependencies**: `config.py`

---

### Agent 3: Neural Network (~200 lines)

**Files**: `model.py`

**Key Classes**:
```python
class RMSNorm(nn.Module): ...
class SelfAttention(nn.Module): ...  # No causal mask!
class MLP(nn.Module): ...
class TransformerBlock(nn.Module): ...

class AlphaZeroTransformer(nn.Module):
    def __init__(self, config: ModelConfig)
    def forward(self, x, action_mask=None) -> (log_policy, value)
    def predict(self, x, action_mask=None) -> (policy, value)  # For inference
    def count_parameters(self) -> int
```

**Architecture**:
```
Input: (B, board_size) token indices {0,1,2}
  → Token Embedding (3 → n_embd)
  → Position Embedding
  → n_layer × TransformerBlock
  → RMSNorm
  → Policy head: mean pool → Linear → log_softmax
  → Value head: mean pool → Linear → GELU → Linear → tanh
Output: (policy, value)
```

**Key Points**:
- No causal masking (can see whole board)
- Three tokens: 0=opponent piece, 1=empty, 2=own piece
- Action mask sets illegal actions to -inf before softmax

**Dependencies**: `config.py`

---

### Agent 4: MCTS & Replay (~350 lines)

**Files**: `mcts.py`, `replay.py`

**Key Classes**:
```python
# mcts.py
class Node:
    prior: float
    visit_count: int
    value_sum: float
    children: Dict[int, Node]
    def value(self) -> float
    def expanded(self) -> bool

def ucb_score(parent, child, c_puct) -> float

class MCTS:
    def run(self, state, model, num_simulations, add_noise=True) -> policy

class BatchedMCTS:
    def search(self, states, model, num_simulations=None, add_noise=True) -> policies

def sample_action(probs, temperature=1.0) -> int

# replay.py
class ReplayBuffer:
    def push(self, state, policy, value)
    def extend(self, examples: List[Tuple])
    def sample(self, batch_size) -> (states, policies, values)
    def __len__(self) -> int
    def clear()
```

**MCTS Algorithm**:
1. SELECT: Walk tree using UCB score
2. EXPAND: At leaf, use network to get priors, create children
3. BACKUP: Propagate value up (alternating sign)
4. Return visit count distribution as policy

**UCB Formula**: `Q + c_puct * P * sqrt(N_parent) / (1 + N_child)`

**Dependencies**: `game.py`, `model.py`, `config.py`

---

### Agent 5: Training & Scripts (~530 lines)

**Files**: `train.py`, `eval.py`, `play.py`, `arena.py`, `speedrun.sh`

**train.py Key Functions**:
```python
def self_play_game(game, model, mcts, temp_threshold) -> List[examples]
def self_play_games(game, model, mcts, num_games, temp_threshold) -> List[examples]
def train_step(model, optimizer, game, states, policies, values, device) -> (loss, p_loss, v_loss)
def evaluate_vs_random(game, model, mcts, num_games) -> win_rate
def main(): # argparse + training loop
```

**Training Loop**:
```
for iteration in 1..num_iterations:
    1. Self-play: generate games, collect (state, policy, value) examples
    2. Train: sample from buffer, compute loss = policy_loss + value_loss, backprop
    3. Eval: play vs random every N iterations
    4. Checkpoint: save every N iterations
```

**Loss Functions**:
- Policy: Cross-entropy `L_p = -sum(target_policy * log(pred_policy))`
- Value: MSE `L_v = (pred_value - target_value)^2`

**Dependencies**: All other modules

---

## Critical Interfaces (Copy-Paste Ready)

### Game → Model
```python
state_tensor = game.to_tensor(canonical_state)  # (board_size,) LongTensor
action_mask = game.legal_actions_mask(state)     # (action_size,) float
policy, value = model(state_tensor.unsqueeze(0), action_mask.unsqueeze(0))
```

### MCTS → Model
```python
policy = mcts.search(
    states,           # (B, H, W) canonical states
    model,            # AlphaZeroTransformer
    num_simulations,  # int
    add_noise         # bool, True during training
)  # Returns (B, action_size) policies
```

### Training Example Format
```python
(state, policy, value)
# state: np.ndarray (H, W), canonical form
# policy: np.ndarray (action_size,), sums to 1
# value: float in [-1, 1]
```

---

## Testing Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_game.py -v

# Run with coverage
python -m pytest tests/ --cov=nanozero
```

---

## Quick Validation

After all agents complete, this should work:

```bash
# Install
pip install torch numpy

# Quick smoke test
python -c "
from nanozero.game import get_game
from nanozero.model import AlphaZeroTransformer
from nanozero.mcts import BatchedMCTS
from nanozero.config import get_game_config, get_model_config, MCTSConfig
import numpy as np

game = get_game('tictactoe')
model = AlphaZeroTransformer(get_model_config(game.config, n_layer=2))
mcts = BatchedMCTS(game, MCTSConfig(num_simulations=10))

state = game.initial_state()
policy = mcts.search(state[np.newaxis], model)[0]
print(f'Policy: {policy}')
print('Success!')
"

# Full speedrun
bash speedrun.sh
```

---

## Style Guide (Karpathy-esque)

1. **No classes where functions suffice**
2. **Inline comments for non-obvious code**
3. **Single file per concept**
4. **No dependencies beyond PyTorch + NumPy**
5. **Type hints for function signatures**
6. **Docstrings with Args/Returns**
7. **Keep it under 1200 total lines**
