# NanoZero

> A minimal AlphaZero implementation in ~1200 lines of Python.

Inspired by Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and [nanochat](https://github.com/karpathy/nanochat), NanoZero is a clean, readable, hackable implementation of the [AlphaZero](https://www.nature.com/articles/nature24270) algorithm.

## Features

- 🧠 **Transformer-based** policy-value network (generalizes across games)
- 🌳 **Batched MCTS** for efficient GPU utilization
- 🎮 **Multiple games**: TicTacToe, Connect4 (extensible to more)
- 📦 **Minimal dependencies**: PyTorch + NumPy only
- 🚀 **Single slider of complexity**: Just change `--n_layer` to scale

## Quick Start

```bash
# Clone and setup
git clone https://github.com/yourname/nanozero
cd nanozero
pip install -e .

# Train TicTacToe in ~5 minutes
bash speedrun.sh

# Or manually:
python -m scripts.train --game=tictactoe --n_layer=2 --num_iterations=50

# Play against your trained model
python -m scripts.play --game=tictactoe --checkpoint=checkpoints/tictactoe_final.pt --n_layer=2
```

## Training Recipes

| Game | Command | Time | Win Rate |
|------|---------|------|----------|
| TicTacToe | `bash speedrun.sh` | ~5 min | >95% vs random |
| Connect4 | See below | ~1 hour | >80% vs random |

```bash
# Connect4 (bigger model, more compute)
python -m scripts.train \
    --game=connect4 \
    --n_layer=4 \
    --num_iterations=100 \
    --games_per_iteration=200 \
    --mcts_simulations=100
```

## How It Works

NanoZero implements the AlphaZero algorithm:

1. **Self-Play**: Use MCTS + neural network to play games against itself
2. **Collect Data**: Record (state, MCTS policy, game outcome) tuples
3. **Train**: Update network to predict MCTS policy and game outcome
4. **Repeat**: The network improves, making MCTS stronger, generating better data

### Architecture

```
Board State → Transformer → Policy (where to play) + Value (who's winning)
```

The Transformer sees the board as a sequence of tokens:
- Token 0: Opponent's piece
- Token 1: Empty square
- Token 2: Your piece

MCTS uses the network to guide search, then returns an improved policy.

## File Structure

```
nanozero/
├── nanozero/
│   ├── common.py         # Utilities
│   ├── config.py         # Configuration dataclasses
│   ├── game.py           # Game interface + TicTacToe, Connect4
│   ├── model.py          # Transformer policy-value network
│   ├── mcts.py           # Monte Carlo Tree Search
│   └── replay.py         # Replay buffer
├── scripts/
│   ├── train.py          # Main training loop
│   ├── eval.py           # Evaluate against baselines
│   ├── play.py           # Interactive play
│   └── arena.py          # Pit models against each other
├── tests/
├── speedrun.sh
└── pyproject.toml
```

## Configuration

All training is controlled via command-line arguments:

```bash
python -m scripts.train \
    --game=tictactoe \          # Game to train on
    --n_layer=4 \               # Transformer depth (THE complexity slider)
    --num_iterations=100 \      # Training iterations
    --games_per_iteration=100 \ # Self-play games per iteration
    --mcts_simulations=50 \     # MCTS simulations per move
    --batch_size=64 \           # Training batch size
    --lr=1e-3                   # Learning rate
```

## Extending to New Games

Implement the `Game` interface:

```python
from nanozero.game import Game
from nanozero.config import GameConfig

class MyGame(Game):
    def __init__(self, config: GameConfig):
        super().__init__(config)
    
    def initial_state(self) -> np.ndarray:
        # Return starting board
    
    def legal_actions(self, state) -> List[int]:
        # Return list of valid moves
    
    def next_state(self, state, action) -> np.ndarray:
        # Apply move, return new state
    
    def is_terminal(self, state) -> bool:
        # Is game over?
    
    def terminal_reward(self, state) -> float:
        # Who won? (-1, 0, or 1)
    
    def symmetries(self, state, policy) -> List[Tuple]:
        # Data augmentation via board symmetries
```

## References

- [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270) (AlphaGo Zero)
- [Mastering Chess and Shogi by Self-Play](https://arxiv.org/abs/1712.01815) (AlphaZero)
- [nanoGPT](https://github.com/karpathy/nanoGPT) / [nanochat](https://github.com/karpathy/nanochat)
- [alpha-zero-general](https://github.com/suragnair/alpha-zero-general)

## License

MIT

