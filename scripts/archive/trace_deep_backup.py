"""
scripts/trace_deep_backup.py - Step-by-step trace of a DEEP backup (depth 3+)
"""
import math
import numpy as np
import torch
from nanozero.game import get_game
from nanozero.model import AlphaZeroTransformer
from nanozero.config import get_model_config, BayesianMCTSConfig
from nanozero.common import get_device, load_checkpoint
from nanozero.bayesian_mcts import BayesianMCTS, BayesianNode, normal_cdf


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

    config = BayesianMCTSConfig(num_simulations=100, sigma_0=0.5, obs_var=0.25)
    ttts = BayesianMCTS(game, config)

    # Expand root
    roots, _ = ttts._batch_expand_roots(state[np.newaxis, ...], model, device)
    root = roots[0]

    # Run some sims to build the tree deeper
    np.random.seed(42)
    for _ in range(50):
        leaf_node, search_path, leaf_state, is_terminal = ttts._select_to_leaf(root, state)
        if is_terminal:
            value = game.terminal_reward(leaf_state)
        elif not leaf_node.expanded():
            values = ttts._batch_expand_leaves([leaf_node], [leaf_state], model, device)
            value = values[0]
        else:
            continue
        ttts._backup(search_path, value)

    # Now find a deep path through action 1
    print("Looking for deep path through action 1...")
    for attempt in range(200):
        leaf_node, search_path, leaf_state, is_terminal = ttts._select_to_leaf(root, state)
        if search_path and search_path[0][1] == 1 and len(search_path) >= 3:
            break
    else:
        print("Could not find deep action 1 path")
        return

    print(f"\n=== FOUND DEEP PATH THROUGH ACTION 1 ===")
    path_actions = [a for _, a in search_path]
    print(f"Path: root -> {' -> '.join(map(str, path_actions))}")
    print(f"Depth: {len(search_path)}")
    print(f"Is terminal: {is_terminal}")

    if is_terminal:
        leaf_value = game.terminal_reward(leaf_state)
        print(f"Terminal value: {leaf_value}")
    else:
        # Don't actually expand, just get what the value would be
        canonical = game.canonical_state(leaf_state)
        state_tensor = game.to_tensor(canonical).unsqueeze(0).to(device)
        action_mask = torch.from_numpy(game.legal_actions_mask(leaf_state)).unsqueeze(0).float().to(device)
        with torch.no_grad():
            _, v = model.predict(state_tensor, action_mask)
        leaf_value = v.cpu().item()
        print(f"Leaf NN value: {leaf_value:.4f}")

    depth = len(search_path)
    if depth % 2 == 1:
        leaf_perspective = "opponent of root"
    else:
        leaf_perspective = "same as root"
    print(f"Leaf perspective: {leaf_perspective}")

    print("\n=== BACKUP TRACE ===")
    print("For each level, showing:")
    print("  - What value is being used")
    print("  - What perspective that value is from")
    print("  - What perspective the child node stores")
    print("  - Whether these match")

    value = leaf_value
    obs_var = config.obs_var

    for i, (parent, action) in enumerate(reversed(search_path)):
        level = depth - i
        child = parent.children[action]

        # Determine perspectives
        if level % 2 == 1:
            child_stores = "opponent"
        else:
            child_stores = "root"

        if i == 0:
            # First iteration: value is leaf value
            if depth % 2 == 1:
                value_is_from = "opponent"
            else:
                value_is_from = "root"
        else:
            # Subsequent: value is prev parent's agg_mu
            prev_level = level + 1
            # agg_mu is from parent's perspective
            # parent at prev_level has perspective based on (prev_level - 1)
            parent_of_prev = prev_level - 1
            if parent_of_prev % 2 == 1:
                value_is_from = "opponent"
            elif parent_of_prev == 0:
                value_is_from = "root"
            else:
                value_is_from = "root"

        match = "✓" if value_is_from == child_stores else "✗ BUG!"

        print(f"\nLevel {level}, action {action}:")
        print(f"  value = {value:.4f}, from {value_is_from}'s perspective")
        print(f"  child stores {child_stores}'s perspective")
        print(f"  Match: {match}")

        # Do update
        old_mu = child.mu
        child.update(value, obs_var, config.min_variance)
        print(f"  child.mu: {old_mu:.4f} -> {child.mu:.4f}")

        # Aggregate
        parent.aggregate_children(config.prune_threshold)
        print(f"  parent.agg_mu = {parent.agg_mu:.4f}")

        # agg_mu is from parent's perspective (negated children)
        parent_level = level - 1
        if parent_level == 0:
            agg_is_from = "root"
        elif parent_level % 2 == 1:
            agg_is_from = "opponent"
        else:
            agg_is_from = "root"
        print(f"  agg_mu is from {agg_is_from}'s perspective")

        value = parent.agg_mu
        obs_var = parent.agg_sigma_sq


if __name__ == '__main__':
    main()
