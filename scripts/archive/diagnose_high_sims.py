"""
scripts/diagnose_high_sims.py - Diagnose why TTTS degrades at high simulation counts
"""
import math
import numpy as np
import torch
from nanozero.game import get_game
from nanozero.model import AlphaZeroTransformer
from nanozero.config import get_model_config, BayesianMCTSConfig, MCTSConfig
from nanozero.common import get_device, load_checkpoint
from nanozero.bayesian_mcts import BayesianMCTS, normal_cdf
from nanozero.mcts import BatchedMCTS, sample_action


def print_tree_stats(root, name=""):
    """Print statistics about the tree."""
    children = list(root.children.values())
    actions = list(root.children.keys())

    # Get child stats from parent's perspective
    mus = np.array([-c.mu for c in children])  # Parent perspective
    sigma_sqs = np.array([c.sigma_sq for c in children])
    precisions = np.array([c.precision() for c in children])

    # Total "visits" (precision-weighted)
    total_precision = sum(precisions) - len(children) * 4.0  # Subtract initial precision

    print(f"\n{name} Tree Stats:")
    print(f"  Total children: {len(children)}")
    print(f"  Approx total visits: {total_precision:.1f}")

    # Sort by mu for display
    sorted_idx = np.argsort(mus)[::-1]

    print(f"\n  Children (parent perspective, sorted by mu):")
    for i in sorted_idx[:5]:  # Top 5
        a = actions[i]
        prec = precisions[i]
        approx_visits = prec - 4.0  # Subtract initial
        print(f"    Action {a}: mu={mus[i]:.4f}, sigma_sq={sigma_sqs[i]:.6f}, visits≈{approx_visits:.1f}")

    # Compute optimality weights
    n = len(children)
    leader_idx = sorted_idx[0]
    challenger_idx = sorted_idx[1]

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

    total = scores.sum()
    weights = scores / total if total > 1e-10 else np.ones(n) / n

    print(f"\n  Optimality weights (top 5):")
    for i in sorted_idx[:5]:
        a = actions[i]
        print(f"    Action {a}: score={scores[i]:.4f}, weight={weights[i]:.4f}")

    # Best action by weight
    best_idx = np.argmax(weights)
    best_action = actions[best_idx]
    print(f"\n  Best action by weight: {best_action} (weight={weights[best_idx]:.4f})")
    print(f"  Best action by mu: {actions[leader_idx]} (mu={mus[leader_idx]:.4f})")

    return best_action, weights


def main():
    device = get_device()
    game = get_game('connect4')
    model_config = get_model_config(game.config, n_layer=4)
    model = AlphaZeroTransformer(model_config).to(device)
    load_checkpoint('checkpoints/connect4_iter150.pt', model)
    model.eval()

    # Use a fixed position
    state = game.initial_state()

    sim_counts = [50, 100, 200, 400]

    print("="*60)
    print("TTTS Tree Analysis at Different Simulation Counts")
    print("="*60)

    for sims in sim_counts:
        print(f"\n{'='*60}")
        print(f"Simulations: {sims}")
        print("="*60)

        # Run TTTS
        ttts_config = BayesianMCTSConfig(num_simulations=sims)
        ttts = BayesianMCTS(game, ttts_config)

        # Get internal state by running search
        roots, _ = ttts._batch_expand_roots(state[np.newaxis, ...], model, device)
        root = roots[0]

        np.random.seed(42)
        for sim in range(sims):
            leaf_node, search_path, leaf_state, is_terminal = ttts._select_to_leaf(root, state)

            if is_terminal:
                value = game.terminal_reward(leaf_state)
            elif not leaf_node.expanded():
                values = ttts._batch_expand_leaves([leaf_node], [leaf_state], model, device)
                value = values[0]
            else:
                continue

            ttts._backup(search_path, value)

        ttts_action, ttts_weights = print_tree_stats(root, "TTTS")

        # Compare to PUCT
        puct_config = MCTSConfig(num_simulations=sims)
        puct = BatchedMCTS(game, puct_config)
        puct_policy = puct.search(state[np.newaxis, ...], model, num_simulations=sims)[0]
        puct_action = sample_action(puct_policy, temperature=0)

        print(f"\n  PUCT best action: {puct_action} (policy={puct_policy[puct_action]:.4f})")
        print(f"  Actions agree: {ttts_action == puct_action}")

        # Check variance distribution
        children = list(root.children.values())
        sigma_sqs = np.array([c.sigma_sq for c in children])
        print(f"\n  Variance stats:")
        print(f"    Min sigma_sq: {sigma_sqs.min():.6f}")
        print(f"    Max sigma_sq: {sigma_sqs.max():.6f}")
        print(f"    Mean sigma_sq: {sigma_sqs.mean():.6f}")
        print(f"    Variance ratio (max/min): {sigma_sqs.max()/sigma_sqs.min():.1f}x")


if __name__ == '__main__':
    main()
