# CLAUDE.MD - AI Assistant Guide for NanoZero

> This guide helps AI assistants (like Claude) understand the NanoZero codebase and contribute effectively.

## Project Philosophy

NanoZero is a **minimal, readable, hackable** AlphaZero implementation inspired by Karpathy's nanoGPT. The guiding principles:

1. **Simplicity over features** - Keep the core algorithm obvious (~1200 lines total)
2. **Minimal dependencies** - PyTorch + NumPy only
3. **Single complexity slider** - Just adjust `--n_layer` to scale up
4. **Code clarity** - Anyone should be able to read and understand the full implementation

## Quick Orientation

### Core Components (nanozero/)

| File | Lines | Purpose |
|------|-------|---------|
| `config.py` | ~80 | Dataclasses for all configuration |
| `game.py` | ~400 | Game interface + TicTacToe, Connect4, Go |
| `model.py` | ~150 | Transformer policy-value network |
| `mcts.py` | ~200 | Batched Monte Carlo Tree Search |
| `replay.py` | ~100 | Experience replay buffer |
| `common.py` | ~50 | Utilities (timing, logging) |

### Scripts (scripts/)

**Note:** The scripts/ directory doesn't exist yet. Training/eval scripts need to be implemented.

Planned scripts:
- `train.py` - Main training loop
- `eval.py` - Evaluate against baselines
- `play.py` - Interactive play
- `arena.py` - Pit models against each other

### Tests (tests/)

Use pytest for all testing:
```bash
python -m pytest tests/ -v
python -m pytest tests/test_game.py -v
python -m pytest tests/ --cov=nanozero
```

## Critical Conventions

### Board Representation
- **Values:** `-1` (player 2), `0` (empty), `1` (player 1)
- **Current player:** Always player whose turn it is
- **Canonical form:** Board flipped so current player sees their pieces as `+1`

### Canonical State Pattern
```python
# ALWAYS use canonical form for model input
canonical_state = game.canonical_state(state, game.current_player(state))
state_tensor = game.to_tensor(canonical_state)  # (board_size,) LongTensor
```

Token mapping for model:
- `0` → opponent's piece
- `1` → empty square
- `2` → your piece

### Game Interface Contract

Every game MUST implement:
```python
class Game:
    def initial_state(self) -> np.ndarray: ...
    def current_player(self, state) -> int: ...       # Returns 1 or -1
    def legal_actions(self, state) -> List[int]: ...
    def next_state(self, state, action) -> np.ndarray: ...
    def is_terminal(self, state) -> bool: ...
    def terminal_reward(self, state) -> float: ...    # From current player's view
    def canonical_state(self, state, player) -> np.ndarray: ...
    def to_tensor(self, state) -> torch.Tensor: ...
    def symmetries(self, state, policy) -> List[Tuple]: ...
```

## Common Tasks

### Adding a New Game

1. Inherit from `Game` in `nanozero/game.py`
2. Implement all required methods
3. Add to game registry at bottom of `game.py`
4. Add tests in `tests/test_game.py`
5. Update README with training recipe

Example skeleton:
```python
class MyGame(Game):
    def __init__(self, config: GameConfig):
        super().__init__(config)
        self.board_size = ...
        self.action_size = ...

    # Implement all required methods...
```

### Modifying the Model

The model is in `nanozero/model.py`. Key points:
- Uses standard Transformer blocks (RMSNorm, SelfAttention, MLP)
- Two heads: policy (action probabilities) and value (win/loss/draw prediction)
- Input: flattened board as token sequence
- **Important:** Model doesn't know game rules - MCTS handles that

### Training Loop (when implemented)

Expected structure:
```python
for iteration in range(num_iterations):
    # 1. Self-play: generate training data
    examples = self_play(model, mcts, games_per_iteration)

    # 2. Add to replay buffer
    replay_buffer.add(examples)

    # 3. Train on samples
    for batch in replay_buffer.sample_batches():
        loss = train_step(model, batch)

    # 4. Evaluate & checkpoint
    if iteration % eval_freq == 0:
        evaluate(model)
        save_checkpoint(model)
```

### MCTS Usage

```python
from nanozero.mcts import batched_mcts_search

# Single state
policies = batched_mcts_search(
    states=canonical_states,      # (B, ...) batch of canonical states
    model=model,                  # AlphaZeroTransformer
    game=game,                    # Game instance
    num_simulations=100,          # MCTS rollouts per move
    add_noise=True,               # Dirichlet noise for exploration (training only)
    temp=1.0                      # Temperature for policy sampling
)
# Returns: (B, action_size) improved policies
```

## Code Style (Karpathy-esque)

1. **Prefer functions over classes** - Only use classes for stateful objects
2. **Inline comments** - Explain WHY not WHAT for non-obvious code
3. **Type hints** - All function signatures should have types
4. **Docstrings** - Google-style with Args/Returns
5. **No hidden magic** - Keep control flow explicit and obvious
6. **Single file per concept** - Don't split unnecessarily
7. **Line budget** - Target ~1200 total lines (can go over if essential)

Example:
```python
def canonical_state(self, state: np.ndarray, player: int) -> np.ndarray:
    """
    Returns board from perspective of 'player' (their pieces become +1).

    Args:
        state: Current board state
        player: Player to canonicalize for (1 or -1)

    Returns:
        Board with signs flipped so 'player' sees their pieces as +1
    """
    return state * player
```

## Recent Updates

Based on commit history:
- **FP16 mixed precision training** - Added for faster training
- **Symmetry-aware transposition table** - Deduplicates symmetric positions in MCTS
- **IMPROVEMENTS.md** - Contains research ideas from AlphaZero/KataGo papers

## When Making Changes

### DO:
- Read existing code first - understand patterns before modifying
- Keep changes minimal and focused
- Add tests for new functionality
- Update docstrings and comments
- Verify total line count stays reasonable (~1200 target)
- Run tests: `python -m pytest tests/ -v`

### DON'T:
- Add dependencies beyond PyTorch/NumPy unless critical
- Over-engineer solutions - simple is better
- Break the Game interface contract
- Ignore the canonical state convention
- Add features "because they might be useful later"
- Create abstractions before you need them

## Debugging Tips

### Board State Issues
```python
# Always visualize state when debugging
print(game.render(state))

# Check if state is canonical
print(f"Current player: {game.current_player(state)}")
print(f"Canonical form:\n{game.canonical_state(state, game.current_player(state))}")
```

### MCTS Issues
```python
# Enable verbose mode (if implemented)
mcts_search(..., verbose=True)

# Check legal actions masking
legal_mask = game.legal_actions_mask(state)
assert legal_mask.sum() > 0, "No legal moves!"
```

### Model Issues
```python
# Check input shapes
print(f"State tensor: {state_tensor.shape}")  # Should be (batch, board_size)
print(f"Action mask: {action_mask.shape}")     # Should be (batch, action_size)

# Verify outputs
policy, value = model(state_tensor, action_mask)
assert policy.shape == (batch, action_size)
assert value.shape == (batch, 1)
assert torch.allclose(policy.sum(dim=-1), torch.ones(batch)), "Policy doesn't sum to 1"
```

## Key Metrics

When evaluating changes, track:
- **Win rate vs random** - Should be >95% for TicTacToe, >80% for Connect4
- **Training time** - TicTacToe ~5 min, Connect4 ~1 hour
- **Total lines of code** - Target ~1200, max ~1500
- **Dependencies** - Should only be PyTorch + NumPy
- **Test coverage** - Aim for >80% on core components

## Questions to Ask Before Contributing

1. Does this change align with the "minimal and hackable" philosophy?
2. Could this be simpler?
3. Are we adding dependencies or complexity unnecessarily?
4. Does this break any existing interfaces or conventions?
5. Can someone read the code and immediately understand it?

## Resources

- **AlphaZero Paper:** https://www.nature.com/articles/nature24270
- **nanoGPT:** https://github.com/karpathy/nanoGPT (style inspiration)
- **alpha-zero-general:** https://github.com/suragnair/alpha-zero-general (reference impl)

## Common Pitfalls

1. **Forgetting canonical form** - Always use `canonical_state()` before model input
2. **Wrong reward perspective** - `terminal_reward()` returns reward from current player's view
3. **Invalid action masking** - Must mask illegal moves before policy output
4. **Breaking symmetries** - New games should implement `symmetries()` for data augmentation
5. **Over-optimization** - Don't optimize before profiling. Clarity > speed initially.

---

**Remember:** This is meant to be a learning resource. Keep it simple, keep it readable, and make it easy for others to understand and modify. When in doubt, choose the simpler solution.
