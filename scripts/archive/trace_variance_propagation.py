"""
scripts/trace_variance_propagation.py - Trace how variance is propagated during backup
"""
import math
import numpy as np
import torch
from nanozero.game import get_game
from nanozero.model import AlphaZeroTransformer
from nanozero.config import get_model_config, BayesianMCTSConfig
from nanozero.common import get_device, load_checkpoint
from nanozero.bayesian_mcts import BayesianMCTS


def main():
    device = get_device()
    game = get_game('connect4')
    model_config = get_model_config(game.config, n_layer=4)
    model = AlphaZeroTransformer(model_config).to(device)
    load_checkpoint('checkpoints/connect4_iter150.pt', model)
    model.eval()

    state = game.initial_state()

    config = BayesianMCTSConfig(num_simulations=100, sigma_0=0.5, obs_var=0.25)
    ttts = BayesianMCTS(game, config)

    roots, _ = ttts._batch_expand_roots(state[np.newaxis, ...], model, device)
    root = roots[0]

    print(f"Initial obs_var: {config.obs_var}")
    print(f"Initial sigma_0: {config.sigma_0}")
    print(f"Initial precision: {1.0 / config.sigma_0**2}")

    # Run a few sims and track variance
    np.random.seed(42)
    for sim in range(10):
        leaf_node, search_path, leaf_state, is_terminal = ttts._select_to_leaf(root, state)

        if is_terminal:
            value = game.terminal_reward(leaf_state)
        elif not leaf_node.expanded():
            values = ttts._batch_expand_leaves([leaf_node], [leaf_state], model, device)
            value = values[0]
        else:
            continue

        depth = len(search_path)
        print(f"\n=== Simulation {sim + 1}, depth {depth} ===")
        print(f"Path: root -> {' -> '.join(str(a) for _, a in search_path)}")
        print(f"Leaf value: {value:.4f}")

        # Trace backup
        obs_var = config.obs_var
        for i, (parent, action) in enumerate(reversed(search_path)):
            level = depth - i
            child = parent.children[action]

            old_mu = child.mu
            old_sigma_sq = child.sigma_sq

            # Do the update
            child.update(value, obs_var, config.min_variance)

            print(f"\n  Level {level} (action {action}):")
            print(f"    Before: mu={old_mu:.4f}, sigma_sq={old_sigma_sq:.6f}, prec={1/old_sigma_sq:.1f}")
            print(f"    Update with value={value:.4f}, obs_var={obs_var:.6f}, prec_obs={1/obs_var:.1f}")
            print(f"    After: mu={child.mu:.4f}, sigma_sq={child.sigma_sq:.6f}, prec={1/child.sigma_sq:.1f}")

            # Aggregate
            initial_precision = 1.0 / (config.sigma_0 ** 2)
            parent.aggregate_children(
                config.prune_threshold,
                visited_only=True,
                initial_precision=initial_precision
            )

            if parent.agg_mu is not None:
                print(f"    Aggregated: agg_mu={parent.agg_mu:.4f}, agg_sigma_sq={parent.agg_sigma_sq:.6f}")
                value = parent.agg_mu
                obs_var = parent.agg_sigma_sq
            else:
                value = -child.mu
                obs_var = child.sigma_sq

            print(f"    Next level will use: value={value:.4f}, obs_var={obs_var:.6f}")


if __name__ == '__main__':
    main()
