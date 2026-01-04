"""
scripts/trace_detailed_backup.py - Detailed trace of TTTS backup to find sign errors
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

    # Use the midgame1 position from compare_search_evolution
    state = game.initial_state()
    for m in [3, 4, 6, 2, 4, 4, 6, 1, 2, 6, 2]:
        state = game.next_state(state, m)

    print("Position (midgame1):")
    print(game.display(state))
    print(f"Legal actions: {game.legal_actions(state)}")
    print(f"Current player: {game.current_player(state)}")

    # Get network's raw evaluation
    canonical = game.canonical_state(state)
    state_tensor = game.to_tensor(canonical).unsqueeze(0).to(device)
    action_mask = torch.from_numpy(game.legal_actions_mask(state)).unsqueeze(0).float().to(device)
    policy, value = model.predict(state_tensor, action_mask)
    policy = policy.cpu().numpy()[0]
    value_net = value.cpu().item()

    print(f"\nNetwork evaluation:")
    print(f"  Value (from current player's perspective): {value_net:.4f}")
    print(f"  Policy:")
    for a in game.legal_actions(state):
        print(f"    Action {a}: {policy[a]:.4f}")

    # Initialize TTTS
    config = BayesianMCTSConfig(num_simulations=50, sigma_0=1.0, obs_var=0.25)
    ttts = BayesianMCTS(game, config)

    # Expand root manually
    roots, _ = ttts._batch_expand_roots(state[np.newaxis, ...], model, device)
    root = roots[0]

    print(f"\n=== ROOT EXPANSION ===")
    print(f"Root has {len(root.children)} children")
    print(f"Initial child beliefs (child.mu is from child/opponent perspective):")
    print(f"{'Action':>7} | {'Prior':>7} | {'child.mu':>10} | {'-child.mu':>10} | {'sigma_sq':>10}")
    print("-" * 60)
    for a in sorted(root.children.keys()):
        child = root.children[a]
        print(f"{a:>7} | {child.prior:>7.4f} | {child.mu:>10.4f} | {-child.mu:>10.4f} | {child.sigma_sq:>10.4f}")

    print(f"\nRoot aggregated: agg_mu={root.agg_mu:.4f}, agg_sigma_sq={root.agg_sigma_sq:.4f}")

    # Run more simulations - print summary periodically
    np.random.seed(42)
    for sim in range(200):
        verbose = sim < 5 or (sim + 1) % 25 == 0

        if verbose:
            print(f"\n{'='*70}")
            print(f"SIMULATION {sim + 1}")
            print(f"{'='*70}")

        # Selection phase
        leaf_node, search_path, leaf_state, is_terminal = ttts._select_to_leaf(root, state)

        if verbose:
            path_str = " -> ".join(f"a{a}" for _, a in search_path)
            print(f"Selected path: root -> {path_str}")

        # Get leaf value
        if is_terminal:
            leaf_value = game.terminal_reward(leaf_state)
        elif not leaf_node.expanded():
            values = ttts._batch_expand_leaves([leaf_node], [leaf_state], model, device)
            leaf_value = values[0]
        else:
            continue

        # Backup (manual version of ttts._backup)
        for i, (parent, action) in enumerate(reversed(search_path)):
            child = parent.children[action]
            child.visits += 1

            if i == 0:
                if not child.expanded():
                    child.update(leaf_value, config.obs_var, config.min_variance)
                else:
                    if child.agg_mu is not None:
                        child.mu = child.agg_mu
                        child.sigma_sq = child.agg_sigma_sq

            parent.aggregate_children(config.prune_threshold, visited_only=True)

            if parent.agg_mu is not None:
                parent.mu = parent.agg_mu
                parent.sigma_sq = parent.agg_sigma_sq

        # Print root state periodically
        if verbose:
            print(f"\n--- ROOT STATE AFTER SIM {sim + 1} ---")
            print(f"{'Action':>7} | {'visits':>7} | {'child.mu':>10} | {'-child.mu':>10} | {'sigma_sq':>10}")
            print("-" * 60)
            for a in sorted(root.children.keys()):
                child = root.children[a]
                print(f"{a:>7} | {child.visits:>7} | {child.mu:>10.4f} | {-child.mu:>10.4f} | {child.sigma_sq:>10.4f}")

    # Final policy
    print(f"\n{'='*70}")
    print("FINAL POLICY")
    print(f"{'='*70}")
    policy_ttts = ttts._get_policy(root)
    print(f"{'Action':>7} | {'TTTS Prob':>10} | {'visits':>7} | {'-child.mu':>10}")
    print("-" * 50)
    for a in sorted(root.children.keys()):
        child = root.children[a]
        print(f"{a:>7} | {policy_ttts[a]:>10.4f} | {child.visits:>7} | {-child.mu:>10.4f}")


if __name__ == '__main__':
    main()
