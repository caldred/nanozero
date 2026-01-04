# NanoZero

A research-oriented AlphaZero implementation with a high-performance Rust MCTS backend.

## Overview

NanoZero combines a Transformer-based policy-value network with GPU-accelerated Monte Carlo Tree Search. The entire MCTS runs in Rust, calling back to Python only for neural network inference—minimizing boundary crossings for maximum throughput.

**Key Features:**
- **Rust MCTS backend** with virtual loss batching for parallel leaf evaluation
- **Symmetry-aware transposition table** for caching NN evaluations across searches
- **Two search algorithms**: PUCT (standard AlphaZero) and Bayesian MCTS (Thompson Sampling with IDS)
- **Transformer architecture** that generalizes across board games
- **Three games**: TicTacToe, Connect4, Go (9x9/19x19)

## Installation

```bash
git clone https://github.com/caldred/nanozero
cd nanozero
pip install -e .

# Build the Rust MCTS backend (required)
cd nanozero-mcts-rs
maturin develop --release
```

**Requirements:** Python 3.9+, PyTorch, Rust toolchain, maturin

## Project Structure

```
nanozero/
├── nanozero/                 # Python package
│   ├── game.py              # Game interface + TicTacToe, Connect4, Go
│   ├── model.py             # Transformer policy-value network
│   ├── mcts.py              # Python wrappers for Rust MCTS
│   ├── config.py            # Configuration dataclasses
│   ├── replay.py            # Experience replay buffer
│   └── common.py            # Utilities
│
├── nanozero-mcts-rs/        # Rust MCTS backend
│   └── src/
│       ├── search.rs        # PUCT search with virtual loss
│       ├── bayesian_search.rs  # Bayesian MCTS (TTTS + IDS)
│       ├── transposition_table.rs  # Symmetry-aware position cache
│       ├── game/            # Rust game implementations
│       │   ├── tictactoe.rs
│       │   ├── connect4.rs
│       │   └── go.rs
│       └── py_mcts.rs       # Python bindings
│
├── scripts/                 # Training and evaluation
│   ├── train.py
│   ├── train_bayesian.py
│   ├── arena.py
│   └── eval.py
│
└── notebooks/               # Experiments and benchmarks
```

## MCTS Algorithms

### PUCT (Standard AlphaZero)

Uses UCB with neural network priors:
```
UCB(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
```

```python
from nanozero.mcts import BatchedMCTS
from nanozero.config import MCTSConfig

config = MCTSConfig(num_simulations=100, c_puct=1.0)
mcts = BatchedMCTS(game, config, leaves_per_batch=64)
policies = mcts.search(states, model)
```

### Bayesian MCTS (Thompson Sampling)

Maintains Gaussian beliefs over action values, using Top-Two Thompson Sampling with Information-Directed Sampling for exploration:

```python
from nanozero.mcts import BayesianMCTS
from nanozero.config import BayesianMCTSConfig

config = BayesianMCTSConfig(
    num_simulations=100,
    sigma_0=0.5,        # Prior std
    obs_var=0.25,       # Observation variance
    ids_alpha=0.5,      # IDS exploration weight
)
mcts = BayesianMCTS(game, config, leaves_per_batch=64)
policies = mcts.search(states, model)
```

## Virtual Loss Batching

Both algorithms collect multiple leaves per neural network call using virtual loss to encourage diverse path exploration:

```python
# Collect 64 leaves before each NN inference
mcts = BatchedMCTS(game, config, leaves_per_batch=64, virtual_loss=1.0)
```

This dramatically improves GPU utilization by batching NN evaluations.

## Transposition Table

Both MCTS algorithms use a symmetry-aware transposition table to cache neural network evaluations:

- **Caches policy/value outputs** to avoid redundant NN calls for previously seen positions
- **Respects symmetries**: symmetric positions (rotations, reflections) share cache entries
- **Persists across searches** until explicitly cleared (typically when the model is retrained)

**Important**: Disable TT during training to maximize diversity in self-play data:

```python
# During TRAINING: disable TT for diverse self-play
train_mcts = BatchedMCTS(game, config, use_transposition_table=False)

# During EVALUATION/INFERENCE: enable TT for speed
eval_mcts = BatchedMCTS(game, config, use_transposition_table=True)
```

When TT is enabled during training, symmetric positions share cached evaluations, reducing diversity and potentially causing the model to plateau or regress.

```python
# Check cache statistics (when TT is enabled)
hits, misses, entries = mcts.cache_stats()
print(f"Cache hit rate: {hits / (hits + misses):.1%}, entries: {entries}")

# Clear cache if reusing MCTS instance after model update
mcts.clear_cache()
```

Symmetry-aware caching is particularly effective for games with many symmetries:
- **TicTacToe**: 8 symmetries (4 rotations × 2 reflections)
- **Connect4**: 2 symmetries (identity + horizontal flip)
- **Go**: 8 symmetries for square boards

## Games

| Game | Board | Actions | Notes |
|------|-------|---------|-------|
| TicTacToe | 3x3 | 9 | Good for quick experiments |
| Connect4 | 6x7 | 7 | Standard benchmark |
| Go | 9x9 or 19x19 | 82 or 362 | Includes ko rule, suicide prevention, area scoring |

Go games have a move limit (3x board size²) to prevent infinite games during training.

## Neural Network

The policy-value network is a standard Transformer:

```
Board State → Token Embedding → Transformer Blocks → Policy Head (actions)
                                                   → Value Head (win probability)
```

Token encoding: `0=opponent, 1=empty, 2=self`

Scale complexity with `--n_layer`:
- TicTacToe: 2 layers
- Connect4: 4 layers
- Go 9x9: 6+ layers

## Training

Training notebooks are in `notebooks/`. The basic loop:

```python
for iteration in range(num_iterations):
    # 1. Self-play with MCTS
    examples = self_play(model, mcts, num_games)

    # 2. Add to replay buffer
    replay_buffer.add(examples)

    # 3. Train on samples
    for batch in replay_buffer.sample():
        loss = train_step(model, batch)

    # 4. Evaluate periodically
    if iteration % eval_freq == 0:
        win_rate = evaluate_vs_random(model)
```

## Benchmarks

Run MCTS benchmarks:
```bash
python scripts/benchmark_mcts.py
```

Compare PUCT vs Bayesian:
```bash
python scripts/arena.py --model1 puct --model2 bayesian --games 100
```

## Development

Run Rust tests:
```bash
cd nanozero-mcts-rs
cargo test
```

Run Python tests:
```bash
python -m pytest tests/
```

## References

- [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270) (AlphaGo Zero)
- [Mastering Chess and Shogi by Self-Play](https://arxiv.org/abs/1712.01815) (AlphaZero)
- [Information-Directed Sampling](https://arxiv.org/abs/1403.5556) (IDS for bandits)

## License

MIT
