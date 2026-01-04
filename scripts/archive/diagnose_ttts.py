"""
scripts/diagnose_ttts.py - Deep diagnostic of TTTS behavior

Inspects internal state of BayesianMCTS to understand:
1. How variance evolves during search
2. How Thompson samples compare to means
3. What the raw optimality scores look like before normalization
4. Why policy is collapsing to near-deterministic

Usage:
    python -m scripts.diagnose_ttts --game connect4 --checkpoint model.pt
"""
import argparse
import math
import numpy as np
import torch
from typing import Dict, List, Tuple

from nanozero.game import get_game, Game
from nanozero.model import AlphaZeroTransformer
from nanozero.bayesian_mcts import BayesianMCTS, BayesianNode, normal_cdf
from nanozero.config import get_model_config, BayesianMCTSConfig
from nanozero.common import get_device, load_checkpoint


def inspect_node(node: BayesianNode, label: str = "root"):
    """Print detailed info about a node and its children."""
    print(f"\n{'='*60}")
    print(f"Node: {label}")
    print(f"  mu={node.mu:.4f}, sigma_sq={node.sigma_sq:.6f}, precision={node.precision():.2f}")
    if node.agg_mu is not None:
        print(f"  agg_mu={node.agg_mu:.4f}, agg_sigma_sq={node.agg_sigma_sq:.6f}")

    if not node.children:
        print("  (no children)")
        return

    print(f"\n  Children ({len(node.children)}):")
    print(f"  {'Action':>6} | {'mu':>8} | {'sigma_sq':>10} | {'precision':>10} | {'-mu (parent view)':>16}")
    print("  " + "-"*60)

    for action, child in sorted(node.children.items()):
        print(f"  {action:>6} | {child.mu:>8.4f} | {child.sigma_sq:>10.6f} | {child.precision():>10.2f} | {-child.mu:>16.4f}")


def compute_optimality_scores(node: BayesianNode) -> Tuple[np.ndarray, List[int]]:
    """Compute raw optimality scores (before normalization) for children."""
    if not node.children:
        return np.array([]), []

    actions = list(node.children.keys())
    children = [node.children[a] for a in actions]
    n = len(children)

    if n == 1:
        return np.array([1.0]), actions

    # Get child beliefs from parent's perspective
    mus = np.array([-c.mu for c in children])
    sigma_sqs = np.array([c.sigma_sq for c in children])

    # Find leader and challenger by mean
    sorted_idx = np.argsort(mus)[::-1]
    leader_idx = sorted_idx[0]
    challenger_idx = sorted_idx[1]

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

    return scores, actions


def print_optimality_analysis(node: BayesianNode):
    """Print detailed analysis of optimality scoring."""
    if not node.children:
        return

    actions = list(node.children.keys())
    children = [node.children[a] for a in actions]
    n = len(children)

    mus = np.array([-c.mu for c in children])  # Parent perspective
    sigma_sqs = np.array([c.sigma_sq for c in children])
    stds = np.sqrt(sigma_sqs)

    sorted_idx = np.argsort(mus)[::-1]
    leader_idx = sorted_idx[0]
    challenger_idx = sorted_idx[1] if n > 1 else 0

    print(f"\n  Optimality Analysis:")
    print(f"  Leader: action {actions[leader_idx]} (mu={mus[leader_idx]:.4f})")
    print(f"  Challenger: action {actions[challenger_idx]} (mu={mus[challenger_idx]:.4f})")

    scores, _ = compute_optimality_scores(node)

    print(f"\n  {'Action':>6} | {'-mu':>8} | {'std':>8} | {'Raw Score':>10} | {'Normalized':>10}")
    print("  " + "-"*55)

    total = scores.sum()
    for i, action in enumerate(actions):
        is_leader = i == leader_idx
        is_challenger = i == challenger_idx
        marker = " L" if is_leader else (" C" if is_challenger else "  ")
        norm_score = scores[i] / total if total > 1e-10 else 1.0/n
        print(f"  {action:>6} | {mus[i]:>8.4f} | {stds[i]:>8.4f} | {scores[i]:>10.6f} | {norm_score:>10.4f}{marker}")


def sample_thompson_comparison(node: BayesianNode, n_samples: int = 1000):
    """Compare Thompson sampling distribution to means."""
    if not node.children:
        return

    actions = list(node.children.keys())
    children = [node.children[a] for a in actions]
    n = len(children)

    # Count how often each action wins under Thompson sampling
    win_counts = np.zeros(n)

    for _ in range(n_samples):
        samples = np.array([-c.sample() for c in children])  # Parent perspective
        winner = np.argmax(samples)
        win_counts[winner] += 1

    win_probs = win_counts / n_samples

    # Compare to mean ranking
    mus = np.array([-c.mu for c in children])
    mean_ranks = np.argsort(np.argsort(mus)[::-1])  # Rank by mean

    print(f"\n  Thompson Sampling Analysis ({n_samples} samples):")
    print(f"  {'Action':>6} | {'-mu':>8} | {'Mean Rank':>10} | {'Win Prob':>10}")
    print("  " + "-"*45)

    for i, action in enumerate(actions):
        print(f"  {action:>6} | {mus[i]:>8.4f} | {mean_ranks[i]:>10} | {win_probs[i]:>10.4f}")


def run_diagnostic_search(
    game: Game,
    model: torch.nn.Module,
    state: np.ndarray,
    config: BayesianMCTSConfig,
    checkpoints: List[int] = [1, 5, 10, 25, 50, 100]
):
    """Run search and inspect tree at various checkpoints."""
    device = next(model.parameters()).device

    mcts = BayesianMCTS(game, config)

    # Manually run search with inspection
    states = state[np.newaxis, ...]
    roots, _ = mcts._batch_expand_roots(states, model, device)
    root = roots[0]

    print("\n" + "="*70)
    print("INITIAL STATE (after root expansion)")
    print("="*70)
    inspect_node(root, "root (initial)")
    print_optimality_analysis(root)

    for sim in range(1, max(checkpoints) + 1):
        # Run one simulation
        leaf_node, search_path, leaf_state, is_terminal = mcts._select_to_leaf(root, state)

        if is_terminal:
            value = game.terminal_reward(leaf_state)
        elif not leaf_node.expanded():
            values = mcts._batch_expand_leaves([leaf_node], [leaf_state], model, device)
            value = values[0]
        else:
            continue  # Already expanded

        mcts._backup(search_path, value)

        if sim in checkpoints:
            print(f"\n{'='*70}")
            print(f"AFTER {sim} SIMULATIONS")
            print("="*70)
            inspect_node(root, f"root (sim={sim})")
            print_optimality_analysis(root)
            sample_thompson_comparison(root)

            # Get and print final policy
            policy = mcts._get_policy(root)
            legal_actions = game.legal_actions(state)
            print(f"\n  Final Policy:")
            for a in legal_actions:
                print(f"    action {a}: {policy[a]:.4f}")


def get_test_positions(game: Game, game_name: str) -> Dict[str, np.ndarray]:
    """Get test positions for a game."""
    positions = {}

    if game_name == 'connect4':
        positions['empty'] = game.initial_state()

        # Center opening
        state = game.initial_state()
        state = game.next_state(state, 3)
        positions['center_open'] = state

        # A few moves in (simple)
        state = game.initial_state()
        for move in [3, 3, 4, 4]:
            state = game.next_state(state, move)
        positions['midgame'] = state

        # Benchmark midgame_1 position (where TTTS showed entropy collapse)
        state = game.initial_state()
        for move in [4, 5, 2, 3, 6, 5, 6]:
            state = game.next_state(state, move)
        positions['midgame_1'] = state

    elif game_name == 'tictactoe':
        positions['empty'] = game.initial_state()

        state = game.initial_state()
        state = game.next_state(state, 4)  # X in center
        positions['x_center'] = state

    return positions


def main():
    parser = argparse.ArgumentParser(description='TTTS Diagnostic')
    parser.add_argument('--game', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--position', type=str, default='empty',
                        help='Position to analyze')
    parser.add_argument('--sigma_0', type=float, default=1.0,
                        help='Initial prior std')
    parser.add_argument('--obs_var', type=float, default=0.1,
                        help='Observation variance')

    args = parser.parse_args()

    device = get_device() if args.device == 'auto' else torch.device(args.device)

    # Load game and model
    game = get_game(args.game)
    print(f"Game: {args.game} (backend: {game.backend})")

    model_config = get_model_config(game.config, n_layer=args.n_layer)
    model = AlphaZeroTransformer(model_config).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    # Get positions
    positions = get_test_positions(game, args.game)

    if args.position not in positions:
        print(f"Unknown position: {args.position}")
        print(f"Available: {list(positions.keys())}")
        return

    state = positions[args.position]
    print(f"\nPosition: {args.position}")
    print(game.display(state))

    # Configure TTTS
    config = BayesianMCTSConfig(
        sigma_0=args.sigma_0,
        obs_var=args.obs_var,
    )
    print(f"\nConfig: sigma_0={config.sigma_0}, obs_var={config.obs_var}")

    # Run diagnostic
    run_diagnostic_search(
        game, model, state, config,
        checkpoints=[1, 5, 10, 25, 50, 100]
    )


if __name__ == '__main__':
    main()
