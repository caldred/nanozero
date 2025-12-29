# NanoZero: Development Notes

Practical notes for extending the codebase. Full specs live in `FULL_SPEC.md`.

## Project Overview

- Goal: Minimal AlphaZero in ~1200 lines, Karpathy style.
- Stack: Python 3.10+, PyTorch, NumPy only.
- Targets: TicTacToe in ~5 min, Connect4 in ~1 hour.

## File Structure

```
nanozero/
├── nanozero/
│   ├── common.py         # Utilities
│   ├── config.py         # Configuration dataclasses
│   ├── game.py           # Game interface + implementations
│   ├── model.py          # Transformer policy-value network
│   ├── mcts.py           # MCTS (single + batched)
│   └── replay.py         # Replay buffer
├── scripts/
│   ├── train.py          # Main training loop
│   ├── eval.py           # Evaluate against baselines
│   ├── play.py           # Interactive play
│   └── arena.py          # Pit models against each other
├── tests/
├── speedrun.sh
└── README.md
```

## Core Conventions

- Board values: -1 (player 2), 0 (empty), 1 (player 1).
- Player 1 moves first.
- `canonical_state()` flips signs so current player always sees their pieces as +1.
- The model treats the state as a flat token sequence (0=opponent, 1=empty, 2=own).
- State shape is game-defined; MCTS only relies on the game API.

## Critical Interfaces

### Game → Model
```python
state_tensor = game.to_tensor(canonical_state)  # (board_size,) LongTensor
action_mask = game.legal_actions_mask(state)     # (action_size,) float
policy, value = model(state_tensor.unsqueeze(0), action_mask.unsqueeze(0))
```

### MCTS → Model
```python
policy = mcts.search(
    states,           # (B, ...) canonical states
    model,            # AlphaZeroTransformer
    num_simulations,  # int
    add_noise         # bool, True during training
)  # Returns (B, action_size) policies
```

### Training Example Format
```python
(state, policy, value)
# state: np.ndarray (game-defined shape), canonical form
# policy: np.ndarray (action_size,), sums to 1
# value: float in [-1, 1]
```

## Testing Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_game.py -v

# Run with coverage
python -m pytest tests/ --cov=nanozero
```

## Style Guide (Karpathy-esque)

1. No classes where functions suffice
2. Inline comments for non-obvious code
3. Single file per concept
4. No dependencies beyond PyTorch + NumPy
5. Type hints for function signatures
6. Docstrings with Args/Returns
7. Keep it under 1200 total lines
