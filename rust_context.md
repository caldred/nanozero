# Rust MCTS Implementation Context

## Overview

A Rust backend for NanoZero MCTS was implemented using PyO3 bindings. The implementation covers:
- Standard MCTS with UCB selection
- BayesianMCTS with Thompson sampling and IDS allocation
- All game logic (TicTacToe, Connect4, Go)

## Project Structure

```
nanozero-mcts-rs/
├── Cargo.toml              # Dependencies: pyo3, numpy, rand, rand_distr, smallvec, libm
├── pyproject.toml          # Maturin build configuration
└── src/
    ├── lib.rs              # PyO3 module entry point with Python bindings
    ├── node.rs             # Standard MCTS Node struct
    ├── tree.rs             # TreeArena with arena allocation
    ├── ucb.rs              # UCB score calculation with virtual loss
    ├── math.rs             # normal_cdf, softmax, Dirichlet noise
    ├── search.rs           # Selection and backup for standard MCTS
    ├── batch.rs            # Batched MCTS coordination
    ├── bayesian_node.rs    # BayesianNode with aggregate_children
    ├── bayesian_search.rs  # Thompson sampling, Bayesian backup
    └── game/
        ├── mod.rs          # Game trait definition
        ├── state.rs        # GameState enum (TicTacToe, Connect4, Go states)
        ├── tictactoe.rs    # TicTacToe implementation (8 symmetries)
        ├── connect4.rs     # Connect4 implementation (2 symmetries)
        └── go.rs           # Go implementation (capture, ko, Chinese scoring)
```

## Key Rust Types

### Node (node.rs)
```rust
pub struct Node {
    pub prior: f32,
    pub visit_count: u32,
    pub value_sum: f32,
    pub children_start: u32,
    pub children_count: u8,
    pub virtual_loss: u8,
}
```

### BayesianNode (bayesian_node.rs)
```rust
pub struct BayesianNode {
    pub prior: f32,
    pub mu: f32,
    pub sigma_sq: f32,
    pub agg_mu: Option<f32>,
    pub agg_sigma_sq: Option<f32>,
    pub children_start: u32,
    pub children_count: u8,
}
```

### Game Trait (game/mod.rs)
```rust
pub trait Game: Send + Sync {
    fn initial_state(&self) -> GameState;
    fn current_player(&self, state: &GameState) -> i8;
    fn legal_actions(&self, state: &GameState) -> Vec<u16>;
    fn legal_actions_mask(&self, state: &GameState) -> Vec<bool>;
    fn next_state(&self, state: &GameState, action: u16) -> GameState;
    fn is_terminal(&self, state: &GameState) -> bool;
    fn terminal_reward(&self, state: &GameState) -> f32;
    fn canonical_state(&self, state: &GameState) -> GameState;
    fn to_tensor(&self, state: &GameState) -> Vec<i64>;
    fn symmetries(&self, state: &GameState, policy: &[f32]) -> Vec<(GameState, Vec<f32>)>;
    fn action_size(&self) -> usize;
    fn board_size(&self) -> (usize, usize);
    fn render(&self, state: &GameState) -> String;
}
```

## Python Integration

### Modified Files
- `nanozero/game.py` - Added `RustTicTacToeWrapper`, `RustConnect4Wrapper`, `RustGoWrapper` classes with automatic fallback to Python implementations
- `pyproject.toml` - Added `rust = ["nanozero-mcts-rs>=0.1.0"]` optional dependency

### Python Wrappers
```python
# In nanozero/game.py
try:
    from nanozero_mcts_rs import RustTicTacToe, RustConnect4, RustGo
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

# Usage
game = get_game('tictactoe', use_rust=True)  # Uses Rust if available
game = get_game('tictactoe', use_rust=False) # Forces Python
```

### Exposed Python Classes
- `RustTicTacToe` - 3x3 TicTacToe
- `RustConnect4` - 6x7 Connect4
- `RustGo(size)` - Go with configurable board size

## Build Instructions

After restarting terminal (to load Rust):

```bash
# Development build
cd nanozero-mcts-rs
maturin develop --release

# Or build wheel
maturin build --release
pip install target/wheels/nanozero_mcts_rs-*.whl

# Or with pip
pip install nanozero[rust]
```

## Tests

Test file: `tests/test_game.py`

Contains:
- Python implementation tests for TicTacToe, Connect4, Go
- Rust/Python equivalence tests (skipped if Rust unavailable)

Run tests:
```bash
python -m pytest tests/test_game.py -v
```

## Expected Speedup

| Component | Expected Speedup |
|-----------|------------------|
| MCTS operations | 10-20x |
| Game logic | ~100x |
| Overall (with NN inference) | 15-30x |
| Go (benefits most) | 20-30x |

## Plan File

Full implementation plan is at: `.claude/plans/imperative-munching-beaver.md`

## Next Steps After Build

1. Run `cargo check` to verify compilation
2. Run `maturin develop --release` to build and install
3. Run `python -m pytest tests/test_game.py -v` to verify equivalence
4. Benchmark performance: `python scripts/benchmark_mcts.py`
