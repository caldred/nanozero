"""
scripts/trace_aggregation_dilution.py - Check if siblings dilute the signal
"""
import math
import numpy as np
import torch
from nanozero.game import get_game
from nanozero.model import AlphaZeroTransformer
from nanozero.config import get_model_config, BayesianMCTSConfig
from nanozero.common import get_device, load_checkpoint
from nanozero.bayesian_mcts import BayesianMCTS, normal_cdf


def main():
    device = get_device()
    game = get_game('connect4')
    model_config = get_model_config(game.config, n_layer=4)
    model = AlphaZeroTransformer(model_config).to(device)
    load_checkpoint('checkpoints/connect4_iter150.pt', model)
    model.eval()

    moves = [3, 4, 6, 2, 4, 4, 6, 1, 2, 6, 2]
    state = game.initial_state()
    for m in moves:
        state = game.next_state(state, m)

    config = BayesianMCTSConfig(num_simulations=100, sigma_0=0.5, obs_var=0.25)
    ttts = BayesianMCTS(game, config)

    roots, _ = ttts._batch_expand_roots(state[np.newaxis, ...], model, device)
    root = roots[0]

    # Run some sims to build tree
    np.random.seed(42)
    for _ in range(30):
        leaf_node, search_path, leaf_state, is_terminal = ttts._select_to_leaf(root, state)
        if is_terminal:
            value = game.terminal_reward(leaf_state)
        elif not leaf_node.expanded():
            values = ttts._batch_expand_leaves([leaf_node], [leaf_state], model, device)
            value = values[0]
        else:
            continue
        ttts._backup(search_path, value)

    # Get child 1 (action 1 from root)
    child1 = root.children[1]
    print(f"Action 1 at root has {len(child1.children)} grandchildren")
    print("\nGrandchildren of action 1:")
    for a in sorted(child1.children.keys()):
        gc = child1.children[a]
        print(f"  Action {a}: gc.mu={gc.mu:.4f}, gc.sigma_sq={gc.sigma_sq:.4f}, precision={gc.precision():.1f}")

    # Trace aggregation at child1
    print("\n=== AGGREGATION AT CHILD1 ===")
    children = list(child1.children.values())
    actions = list(child1.children.keys())
    n = len(children)

    mus = np.array([-c.mu for c in children])
    sigma_sqs = np.array([c.sigma_sq for c in children])

    print(f"\nFrom child1's perspective (negated grandchild mus):")
    for i, a in enumerate(actions):
        print(f"  Action {a}: mus[{i}]={mus[i]:.4f}, sigma_sq={sigma_sqs[i]:.4f}")

    # Find leader/challenger
    sorted_idx = np.argsort(mus)[::-1]
    leader_idx = sorted_idx[0]
    challenger_idx = sorted_idx[1]

    print(f"\nLeader: action {actions[leader_idx]} with mu={mus[leader_idx]:.4f}")
    print(f"Challenger: action {actions[challenger_idx]} with mu={mus[challenger_idx]:.4f}")

    # Compute scores
    mu_L, sigma_sq_L = mus[leader_idx], sigma_sqs[leader_idx]
    mu_C, sigma_sq_C = mus[challenger_idx], sigma_sqs[challenger_idx]

    scores = np.zeros(n)
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

    print(f"\nOptimality scores:")
    for i, a in enumerate(actions):
        print(f"  Action {a}: score={scores[i]:.4f}")

    # Prune and normalize
    scores[scores < 0.01] = 0.0
    total = scores.sum()
    weights = scores / total if total > 1e-10 else np.ones(n) / n

    print(f"\nWeights after pruning:")
    for i, a in enumerate(actions):
        print(f"  Action {a}: weight={weights[i]:.4f}")

    # Compute aggregate
    agg_mu = np.sum(weights * mus)
    print(f"\nagg_mu = weighted average = {agg_mu:.4f}")

    # Compare to leader
    print(f"\nLeader's mu: {mus[leader_idx]:.4f}")
    print(f"Aggregate mu: {agg_mu:.4f}")
    print(f"Difference: {agg_mu - mus[leader_idx]:.4f}")

    if abs(agg_mu - mus[leader_idx]) > 0.01:
        print("\n*** Siblings are diluting the signal! ***")
        print("The aggregate value is pulled toward other children's beliefs.")


if __name__ == '__main__':
    main()
