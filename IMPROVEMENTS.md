# Potential Improvements for NanoZero

Based on research from AlphaGo Zero, AlphaZero, Leela Zero, and KataGo papers.

## Tier 1: Easy Wins (1-2 hours each)

### 1. History Planes Input
**Current**: Single board state (current positions only)
**Improvement**: Stack last N board positions as input channels

AlphaGo Zero uses 8 history planes per player (16 total) to help the network understand:
- Recent move sequences
- Ko situations implicitly
- Momentum and tactical patterns

**Implementation**:
```python
# State shape: (history_length * 2 + 1, height, width)
# Channel 0-7: current player's stones for last 8 moves
# Channel 8-15: opponent's stones for last 8 moves
# Channel 16: color to play (all 1s or all 0s)
```

**Impact**: Better tactical understanding, especially for ko fights
**Complexity**: Medium - requires changing state representation

---

### 2. Improved PUCT Constants
**Current**: Fixed c_puct = 1.0
**Improvement**: Use AlphaZero's adaptive exploration formula

AlphaZero uses:
```python
c_puct = log((1 + N + pb_c_base) / pb_c_base) + pb_c_init
# where pb_c_base = 19652, pb_c_init = 1.25
```

This increases exploration as search deepens.

**Impact**: Better exploration-exploitation balance
**Complexity**: Low - just change the UCB formula

---

### 3. Temperature Annealing
**Current**: Fixed temperature threshold (move 15)
**Improvement**: Gradual temperature decay

```python
temperature = max(0.1, 1.0 - move_num / 30)
```

Or use temperature schedule from the paper:
- τ=1 for first 30 moves
- τ→0 after move 30

**Impact**: Better early-game exploration, stronger late-game play
**Complexity**: Low

---

### 4. FPU (First Play Urgency)
**Current**: Unexplored nodes have Q=0
**Improvement**: Use parent's value minus a reduction

```python
fpu_reduction = 0.25
fpu_value = parent_value - fpu_reduction
# Use fpu_value for unexplored children instead of 0
```

This discourages exploring obviously bad moves.

**Impact**: Faster convergence, better move ordering
**Complexity**: Low

---

## Tier 2: Moderate Effort (2-4 hours each)

### 5. Playout Cap Randomization (KataGo)
**Current**: Fixed MCTS simulations per move
**Improvement**: Random mix of full/fast searches

```python
# 75% of moves: fast search (25 sims)
# 25% of moves: full search (100 sims) - used for policy target
if random.random() < 0.25:
    num_sims = 100  # full search
else:
    num_sims = 25   # fast search
```

**Impact**: 50%+ speedup in self-play while maintaining policy quality
**Complexity**: Low-Medium

---

### 6. Auxiliary Value Targets
**Current**: Single value target (game outcome: -1, 0, +1)
**Improvement**: Multiple auxiliary targets

- **Score prediction**: Predict final score difference (regression)
- **Ownership**: Predict which player owns each intersection at game end

```python
# Additional heads on the network:
score_head = nn.Linear(n_embd, 1)  # Predict score
ownership_head = nn.Linear(n_embd, board_size)  # Predict per-cell ownership
```

**Impact**: Better value network training, especially for Go
**Complexity**: Medium - requires network architecture change

---

### 7. Progressive Network Growing
**Current**: Fixed network size throughout training
**Improvement**: Start small, grow network as training progresses

```
Iterations 1-20: n_layer=2 (fast training)
Iterations 21-50: n_layer=3 (transfer weights)
Iterations 51+: n_layer=4 (full capacity)
```

**Impact**: Much faster initial training
**Complexity**: Medium - requires weight transfer logic

---

### 8. Policy Target Pruning
**Current**: MCTS visit distribution directly used as policy target
**Improvement**: Sharpen/filter the policy target

```python
# Remove low-probability moves from target
mask = mcts_policy > 0.01
target = mcts_policy * mask
target = target / target.sum()
```

**Impact**: Cleaner gradients, faster policy learning
**Complexity**: Low

---

### 9. Transposition Table / Graph Search
**Current**: Tree search (no position caching)
**Improvement**: Cache neural network evaluations

```python
# Simple zobrist hashing
position_hash = hash(board.tobytes())
if position_hash in cache:
    return cache[position_hash]
```

**Impact**: Significant speedup when same positions repeat
**Complexity**: Medium

---

## Tier 3: Significant Effort (4+ hours)

### 10. Residual Network Architecture
**Current**: Standard transformer
**Improvement**: Use residual convolutional network (ResNet)

AlphaGo Zero's architecture:
- 1 conv layer (256 filters, 3x3)
- 19 or 39 residual blocks
- Separate policy and value heads

**Why ResNet works well for board games**:
- Translation equivariance (same pattern anywhere on board)
- Efficient for 2D spatial data
- Well-studied architecture

**Impact**: Potentially better sample efficiency
**Complexity**: High - requires full architecture change

---

### 11. Global Pooling Layers (KataGo)
**Current**: Pure local convolutions/attention
**Improvement**: Add global pooling to capture non-local patterns

```python
# Add global average pool + broadcast back
global_features = x.mean(dim=spatial_dims, keepdim=True)
x = torch.cat([x, global_features.expand_as(x)], dim=-1)
```

Important for Go's ko fights and ladder reading.

**Impact**: Better long-range tactical understanding
**Complexity**: Medium

---

### 12. Cyclical Learning Rate
**Current**: Fixed learning rate
**Improvement**: Cosine annealing or cyclical LR

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)
```

**Impact**: Better convergence, escape local minima
**Complexity**: Low

---

### 13. Mixed Precision Training
**Current**: FP32 training
**Improvement**: FP16/BF16 mixed precision

```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    loss = compute_loss(...)
scaler.scale(loss).backward()
scaler.step(optimizer)
```

**Impact**: 2x memory efficiency, faster on modern GPUs
**Complexity**: Low-Medium

---

### 14. Gumbel AlphaZero (Policy Improvement)
**Current**: Standard MCTS + softmax sampling
**Improvement**: Use Gumbel-Top-k trick for better exploration

Recent work shows Gumbel-based action selection can improve policy:
```python
gumbel = -np.log(-np.log(np.random.uniform(size=policy.shape)))
scores = np.log(policy) + gumbel
action = np.argmax(scores)
```

**Impact**: Better exploration without explicit temperature
**Complexity**: Low

---

## Tier 4: Research-Level

### 15. MuZero-Style Learned Dynamics
Instead of using game rules for MCTS simulation, learn a dynamics model.
**Complexity**: Very High (essentially a different algorithm)

### 16. Opponent Modeling
Train against a pool of past versions rather than just self-play.
**Complexity**: High (infrastructure change)

### 17. Curriculum Learning
Start with simpler positions (endgames) and progress to full games.
**Complexity**: High (requires position generation)

---

## Recommended Priority for NanoZero

**Quick Wins (do first)**:
1. Improved PUCT constants (#2)
2. FPU reduction (#4)
3. Temperature annealing (#3)
4. Policy target pruning (#8)

**Medium Impact (do next)**:
5. Playout cap randomization (#5)
6. Transposition table (#9)
7. Cyclical learning rate (#12)

**If Time Permits**:
8. History planes (#1)
9. Auxiliary targets (#6)
10. Global pooling (#11)

---

## Sources

- [AlphaGo Zero Paper (Nature 2017)](https://www.nature.com/articles/nature24270)
- [AlphaZero Paper (DeepMind 2017)](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf)
- [Leela Zero GitHub](https://github.com/leela-zero/leela-zero)
- [KataGo Paper (arXiv 2019)](https://arxiv.org/abs/1902.10565)
- [KataGo GitHub](https://github.com/lightvector/KataGo)
- [Simple Alpha Zero Tutorial](https://suragnair.github.io/posts/alphazero.html)
- [PUCT Explanation](https://medium.com/oracledevs/lessons-from-alphazero-part-3-parameter-tweaking-4dceb78ed1e5)
