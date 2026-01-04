"""
scripts/trace_one_backup.py - Step-by-step trace of ONE backup

Shows exactly what value goes where with what sign.
"""
import math
import numpy as np
import torch
from nanozero.game import get_game
from nanozero.model import AlphaZeroTransformer
from nanozero.config import get_model_config, BayesianMCTSConfig
from nanozero.common import get_device, load_checkpoint
from nanozero.bayesian_mcts import BayesianMCTS, BayesianNode, normal_cdf


def manual_aggregate(node, prune_threshold=0.01):
    """Manual aggregate with detailed output."""
    children = list(node.children.values())
    actions = list(node.children.keys())
    n = len(children)

    if n == 1:
        child = children[0]
        return -child.mu, child.sigma_sq, {actions[0]: 1.0}

    mus = np.array([-c.mu for c in children])
    sigma_sqs = np.array([c.sigma_sq for c in children])

    sorted_idx = np.argsort(mus)[::-1]
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

    scores[scores < prune_threshold] = 0.0
    total = scores.sum()
    if total < 1e-10:
        weights = np.ones(n) / n
    else:
        weights = scores / total

    agg_mu = float(np.sum(weights * mus))
    disagreement = (mus - agg_mu) ** 2
    agg_sigma_sq = float(np.sum(weights * (sigma_sqs + disagreement)))

    weight_dict = {actions[i]: weights[i] for i in range(n)}
    return agg_mu, agg_sigma_sq, weight_dict


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

    root_player = game.current_player(state)
    print(f"Root player: {root_player}")
    print(f"Position:\n{game.display(state)}")

    config = BayesianMCTSConfig(num_simulations=100, sigma_0=0.5, obs_var=0.25)
    ttts = BayesianMCTS(game, config)

    # Expand root
    roots, _ = ttts._batch_expand_roots(state[np.newaxis, ...], model, device)
    root = roots[0]

    print("\n=== INITIAL STATE ===")
    print("Root children (child.mu is from CHILD's perspective, i.e., opponent of root):")
    for a in sorted(root.children.keys()):
        c = root.children[a]
        print(f"  Action {a}: child.mu={c.mu:.4f}, sigma_sq={c.sigma_sq:.4f}")

    # Force exploration of action 1 by manually setting up the path
    np.random.seed(12345)  # Seed to hopefully get action 1

    # Find a simulation that goes through action 1
    for attempt in range(100):
        leaf_node, search_path, leaf_state, is_terminal = ttts._select_to_leaf(root, state)
        if search_path and search_path[0][1] == 1:
            break
    else:
        print("Could not find action 1 path in 100 attempts")
        return

    print(f"\n=== FOUND PATH THROUGH ACTION 1 ===")
    print(f"Search path: {[(a, 'node') for _, a in search_path]}")
    print(f"Depth: {len(search_path)}")
    print(f"Is terminal: {is_terminal}")

    # Get leaf value
    if is_terminal:
        leaf_value = game.terminal_reward(leaf_state)
        print(f"Terminal value: {leaf_value}")
    else:
        values = ttts._batch_expand_leaves([leaf_node], [leaf_state], model, device)
        leaf_value = values[0]
        print(f"Leaf NN value: {leaf_value:.4f}")

    # Determine perspective at leaf
    depth = len(search_path)
    if depth % 2 == 1:
        leaf_perspective = "opponent of root"
    else:
        leaf_perspective = "same as root"
    print(f"Leaf is at depth {depth}, perspective: {leaf_perspective}")

    # Now manually trace the backup
    print("\n=== MANUAL BACKUP TRACE ===")

    value = leaf_value
    obs_var = config.obs_var

    print(f"\nStarting value: {value:.4f}")
    print(f"Starting obs_var: {obs_var:.4f}")

    for i, (parent, action) in enumerate(reversed(search_path)):
        level = depth - i  # Current tree level (1 = root's children)
        child = parent.children[action]

        if level % 2 == 1:
            level_perspective = "opponent of root"
        else:
            level_perspective = "same as root"

        print(f"\n--- Level {level} (action {action}) ---")
        print(f"This level's perspective: {level_perspective}")
        print(f"child.mu BEFORE update: {child.mu:.6f}")
        print(f"child.sigma_sq BEFORE: {child.sigma_sq:.6f}")
        print(f"Updating with value={value:.6f}, obs_var={obs_var:.6f}")

        # Check: does the value's perspective match the child's storage perspective?
        if i == 0:
            value_perspective = leaf_perspective
        else:
            # Value came from previous parent's agg_mu
            prev_level = level + 1
            if prev_level % 2 == 1:
                value_perspective = "opponent of root"
            else:
                value_perspective = "same as root"

        match = "✓ MATCH" if value_perspective == level_perspective else "✗ MISMATCH!"
        print(f"Value is from: {value_perspective}")
        print(f"Child stores: {level_perspective}")
        print(f"Sign check: {match}")

        # Do the update
        old_mu = child.mu
        old_sigma_sq = child.sigma_sq
        precision_prior = 1.0 / max(old_sigma_sq, 1e-6)
        precision_obs = 1.0 / max(obs_var, 1e-6)
        new_precision = precision_prior + precision_obs
        new_mu = (precision_prior * old_mu + precision_obs * value) / new_precision
        new_sigma_sq = max(1.0 / new_precision, 1e-6)

        child.mu = new_mu
        child.sigma_sq = new_sigma_sq

        print(f"child.mu AFTER update: {child.mu:.6f}")
        print(f"child.sigma_sq AFTER: {child.sigma_sq:.6f}")
        print(f"(mu moved from {old_mu:.4f} toward {value:.4f}, result {new_mu:.4f})")

        # Now aggregate at parent
        parent.aggregate_children(config.prune_threshold)

        print(f"\nParent aggregation:")
        print(f"  parent.agg_mu: {parent.agg_mu:.6f}")
        print(f"  parent.agg_sigma_sq: {parent.agg_sigma_sq:.6f}")

        # agg_mu is computed as weighted average of -child.mu
        # So agg_mu is from PARENT's perspective
        parent_level = level - 1
        if parent_level == 0:
            parent_perspective = "root"
        elif parent_level % 2 == 1:
            parent_perspective = "opponent of root"
        else:
            parent_perspective = "same as root"
        print(f"  parent.agg_mu is from: {parent_perspective}'s perspective")

        # For next iteration
        value = parent.agg_mu
        obs_var = parent.agg_sigma_sq

        print(f"\n  Next iteration will use:")
        print(f"    value = {value:.6f} (from {parent_perspective})")
        print(f"    obs_var = {obs_var:.6f}")

    print("\n=== FINAL STATE ===")
    print("Root children after this one backup:")
    for a in sorted(root.children.keys()):
        c = root.children[a]
        print(f"  Action {a}: child.mu={c.mu:.6f}, sigma_sq={c.sigma_sq:.6f}")


if __name__ == '__main__':
    main()
