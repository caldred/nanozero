"""
scripts/trace_action3_subtree.py - Trace action 3's subtree values to find why it gets wrong value
"""
import math
import numpy as np
import torch
from nanozero.game import get_game
from nanozero.model import AlphaZeroTransformer
from nanozero.config import get_model_config, BayesianMCTSConfig
from nanozero.common import get_device, load_checkpoint
from nanozero.bayesian_mcts import BayesianMCTS


def print_subtree(node, prefix="", max_depth=2, depth=0):
    """Print subtree with mu values."""
    if depth > max_depth or not node.children:
        return

    for a in sorted(node.children.keys()):
        child = node.children[a]
        visited = "✓" if child.visits > 0 else " "
        print(f"{prefix}a{a}{visited}: mu={child.mu:+.4f} (-mu={-child.mu:+.4f}), visits={child.visits}, σ²={child.sigma_sq:.4f}")
        print_subtree(child, prefix + "  ", max_depth, depth + 1)


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

    print("Position (midgame1):")
    print(game.display(state))

    # Use default config (sigma_0=0.5)
    config = BayesianMCTSConfig(num_simulations=50)
    ttts = BayesianMCTS(game, config)

    # Expand root
    roots, _ = ttts._batch_expand_roots(state[np.newaxis, ...], model, device)
    root = roots[0]

    print(f"\n=== INITIAL STATE ===")
    print(f"Root children:")
    for a in sorted(root.children.keys()):
        child = root.children[a]
        print(f"  a{a}: mu={child.mu:+.4f} (-mu={-child.mu:+.4f}), prior={child.prior:.4f}")

    # Track when action 3 is visited
    np.random.seed(42)
    for sim in range(100):
        leaf_node, search_path, leaf_state, is_terminal = ttts._select_to_leaf(root, state)

        # Check if action 3 was selected
        selected_action = search_path[0][1] if search_path else None

        if is_terminal:
            leaf_value = game.terminal_reward(leaf_state)
        elif not leaf_node.expanded():
            values = ttts._batch_expand_leaves([leaf_node], [leaf_state], model, device)
            leaf_value = values[0]
        else:
            continue

        ttts._backup(search_path, leaf_value)

        # Print when action 3 is visited
        if selected_action == 3:
            path_str = " -> ".join(f"a{a}" for _, a in search_path)
            print(f"\n=== SIM {sim+1}: Visited action 3 (path: {path_str}) ===")
            print(f"Leaf value: {leaf_value:.4f}")

            # Print action 3's children
            a3_node = root.children[3]
            print(f"\nAction 3 after backup:")
            print(f"  mu={a3_node.mu:+.4f} (-mu={-a3_node.mu:+.4f}), visits={a3_node.visits}")
            if a3_node.expanded():
                print(f"  agg_mu={a3_node.agg_mu:+.4f}, agg_sigma_sq={a3_node.agg_sigma_sq:.4f}")
                print(f"\n  Children of action 3:")
                for ca in sorted(a3_node.children.keys()):
                    c = a3_node.children[ca]
                    visited = "✓" if c.visits > 0 else " "
                    print(f"    a{ca}{visited}: mu={c.mu:+.4f} (-mu={-c.mu:+.4f}), visits={c.visits}")

        # Print summary at key points
        if (sim + 1) in [10, 25, 50, 75, 100]:
            print(f"\n=== SUMMARY AT SIM {sim+1} ===")
            print(f"{'Action':>7} | {'visits':>7} | {'child.mu':>10} | {'-child.mu':>10}")
            print("-" * 45)
            for a in sorted(root.children.keys()):
                child = root.children[a]
                print(f"{a:>7} | {child.visits:>7} | {child.mu:>+10.4f} | {-child.mu:>+10.4f}")

    # Final state
    print(f"\n=== FINAL STATE ===")
    print(f"\nAction 3 subtree (depth 2):")
    print_subtree(root.children[3], prefix="  ", max_depth=2)

    print(f"\nAction 4 subtree (depth 2):")
    print_subtree(root.children[4], prefix="  ", max_depth=2)


if __name__ == '__main__':
    main()
