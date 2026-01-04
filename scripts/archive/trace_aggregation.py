"""
scripts/trace_aggregation.py - Trace exactly how aggregation produces action 3's value
"""
import math
import numpy as np
import torch
from nanozero.game import get_game
from nanozero.model import AlphaZeroTransformer
from nanozero.config import get_model_config, BayesianMCTSConfig
from nanozero.common import get_device, load_checkpoint
from nanozero.bayesian_mcts import BayesianMCTS, normal_cdf


def trace_aggregate_children(node, label):
    """Trace the aggregation calculation."""
    if not node.children:
        print(f"  No children")
        return

    visited = [(a, c) for a, c in node.children.items() if c.visits > 0]
    if not visited:
        print(f"  No visited children")
        return

    print(f"\n  {label} aggregation:")
    print(f"  {'Action':>7} | {'child.mu':>10} | {'-child.mu':>10} | {'visits':>7} | {'sigma_sq':>10}")
    print("  " + "-" * 55)

    mus = []
    sigma_sqs = []
    for a, c in sorted(visited):
        print(f"  {a:>7} | {c.mu:>+10.4f} | {-c.mu:>+10.4f} | {c.visits:>7} | {c.sigma_sq:>10.4f}")
        mus.append(-c.mu)  # From parent's perspective
        sigma_sqs.append(c.sigma_sq)

    mus = np.array(mus)
    sigma_sqs = np.array(sigma_sqs)
    n = len(mus)

    # Find leader and challenger
    sorted_idx = np.argsort(mus)[::-1]
    leader_idx = sorted_idx[0]
    challenger_idx = sorted_idx[1] if n > 1 else 0

    print(f"\n  Leader: action {list(visited)[leader_idx][0]} with -mu = {mus[leader_idx]:+.4f}")
    if n > 1:
        print(f"  Challenger: action {list(visited)[challenger_idx][0]} with -mu = {mus[challenger_idx]:+.4f}")

    # Compute scores
    scores = np.zeros(n)
    mu_L, sigma_sq_L = mus[leader_idx], sigma_sqs[leader_idx]
    mu_C, sigma_sq_C = mus[challenger_idx], sigma_sqs[challenger_idx]

    for i in range(n):
        if i == leader_idx:
            diff = mu_L - mu_C
            std = math.sqrt(sigma_sq_L + sigma_sq_C)
        else:
            diff = mus[i] - mu_L
            std = math.sqrt(sigma_sqs[i] + sigma_sq_L)

        if std > 1e-10:
            scores[i] = normal_cdf(diff / std)
        else:
            scores[i] = 1.0 if diff > 0 else 0.0

    print(f"\n  Scores (before pruning): {scores}")

    # Prune
    prune_threshold = 0.01
    scores[scores < prune_threshold] = 0.0
    print(f"  Scores (after prune 0.01): {scores}")

    # Normalize
    total = scores.sum()
    weights = scores / total if total > 1e-10 else np.ones(n) / n
    print(f"  Weights (normalized): {weights}")

    # Compute agg_mu (LINEAR weights)
    agg_mu = np.sum(weights * mus)
    print(f"\n  agg_mu (linear weights) = {agg_mu:+.4f} (from parent's perspective)")

    # Compute variance (SQUARED weights)
    disagreement = (mus - agg_mu) ** 2
    agg_sigma_sq_squared = np.sum(weights ** 2 * (sigma_sqs + disagreement))
    print(f"  agg_sigma_sq (squared weights) = {agg_sigma_sq_squared:.4f}")

    # Show contribution of each child to agg_mu
    print(f"\n  Contribution to agg_mu:")
    for i, (a, c) in enumerate(sorted(visited)):
        contrib = weights[i] * mus[i]
        print(f"    action {a}: weight={weights[i]:.4f} * -mu={mus[i]:+.4f} = {contrib:+.4f}")


def main():
    device = get_device()
    game = get_game('connect4')
    model_config = get_model_config(game.config, n_layer=4)
    model = AlphaZeroTransformer(model_config).to(device)
    load_checkpoint('checkpoints/connect4_iter150.pt', model)
    model.eval()

    # Use the midgame1 position
    state = game.initial_state()
    for m in [3, 4, 6, 2, 4, 4, 6, 1, 2, 6, 2]:
        state = game.next_state(state, m)

    config = BayesianMCTSConfig(num_simulations=50)
    ttts = BayesianMCTS(game, config)
    roots, _ = ttts._batch_expand_roots(state[np.newaxis, ...], model, device)
    root = roots[0]

    # Run some simulations
    np.random.seed(42)
    for sim in range(50):
        leaf_node, search_path, leaf_state, is_terminal = ttts._select_to_leaf(root, state)
        if is_terminal:
            leaf_value = game.terminal_reward(leaf_state)
        elif not leaf_node.expanded():
            values = ttts._batch_expand_leaves([leaf_node], [leaf_state], model, device)
            leaf_value = values[0]
        else:
            continue
        ttts._backup(search_path, leaf_value)

    # Now trace the aggregation for action 3
    a3 = root.children[3]
    print("=== ACTION 3 SUBTREE ===")
    print(f"\nAction 3 node: mu={a3.mu:+.4f}, visits={a3.visits}")
    print(f"From root's perspective: -mu = {-a3.mu:+.4f}")

    trace_aggregate_children(a3, "Action 3")

    # Trace root aggregation
    print("\n\n=== ROOT AGGREGATION ===")
    print(f"Root: mu={root.mu:+.4f}")
    trace_aggregate_children(root, "Root")

    # Show final values
    print("\n\n=== FINAL ROOT CHILDREN ===")
    print(f"{'Action':>7} | {'child.mu':>10} | {'-child.mu':>10} | {'visits':>7}")
    print("-" * 50)
    for a in sorted(root.children.keys()):
        c = root.children[a]
        print(f"{a:>7} | {c.mu:>+10.4f} | {-c.mu:>+10.4f} | {c.visits:>7}")

    # Show which is best
    best_action = max(root.children.keys(), key=lambda a: -root.children[a].mu)
    worst_action = min(root.children.keys(), key=lambda a: -root.children[a].mu)
    print(f"\nBest action (highest -child.mu): {best_action}")
    print(f"Worst action (lowest -child.mu): {worst_action}")


if __name__ == '__main__':
    main()
