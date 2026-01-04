"""
scripts/compare_search_evolution.py - Compare TTTS vs PUCT search tree evolution

Diagnoses:
1. When is the best action first identified? (earlier/later)
2. How does confidence on best action evolve? (over/under confident)
3. Visit/selection distribution patterns
4. Belief/Q-value evolution over simulations

Usage:
    python -m scripts.compare_search_evolution --checkpoint checkpoints/connect4_iter150.pt
"""
import argparse
import math
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from nanozero.game import get_game, Game
from nanozero.model import AlphaZeroTransformer
from nanozero.mcts import BatchedMCTS
from nanozero.bayesian_mcts import BayesianMCTS, BayesianNode, normal_cdf
from nanozero.config import get_model_config, MCTSConfig, BayesianMCTSConfig
from nanozero.common import get_device, load_checkpoint


@dataclass
class SimSnapshot:
    """Snapshot of search state at a simulation count."""
    sim_count: int
    best_action: int
    confidence: float  # probability mass on best action
    entropy: float
    action_probs: np.ndarray
    # For TTTS: child beliefs; For PUCT: Q-values
    action_values: Dict[int, float] = field(default_factory=dict)
    action_visits: Dict[int, int] = field(default_factory=dict)


def entropy(probs: np.ndarray, eps: float = 1e-8) -> float:
    """Shannon entropy."""
    p = probs[probs > eps]
    return float(-np.sum(p * np.log(p)))


def run_puct_with_snapshots(
    game: Game,
    model: torch.nn.Module,
    state: np.ndarray,
    max_sims: int,
    snapshot_interval: int = 10,
) -> List[SimSnapshot]:
    """Run PUCT and capture snapshots at intervals."""
    device = next(model.parameters()).device
    snapshots = []

    config = MCTSConfig(num_simulations=max_sims)
    mcts = BatchedMCTS(game, config)

    for n_sims in range(snapshot_interval, max_sims + 1, snapshot_interval):
        mcts.clear_cache()
        policy = mcts.search(
            state[np.newaxis, ...],
            model,
            num_simulations=n_sims,
            add_noise=False
        )[0]

        best_action = int(np.argmax(policy))
        conf = float(policy[best_action])

        # Extract visit counts and Q-values from tree
        # Note: BatchedMCTS doesn't easily expose internals, so we approximate
        action_visits = {}
        action_values = {}
        legal_actions = game.legal_actions(state)
        for a in legal_actions:
            # Visits proportional to policy (approximately)
            action_visits[a] = int(policy[a] * n_sims)
            # Q-value not directly accessible, use policy as proxy
            action_values[a] = policy[a]

        snapshots.append(SimSnapshot(
            sim_count=n_sims,
            best_action=best_action,
            confidence=conf,
            entropy=entropy(policy),
            action_probs=policy.copy(),
            action_values=action_values,
            action_visits=action_visits,
        ))

    return snapshots


def run_ttts_with_snapshots(
    game: Game,
    model: torch.nn.Module,
    state: np.ndarray,
    max_sims: int,
    snapshot_interval: int = 10,
) -> List[SimSnapshot]:
    """Run TTTS and capture snapshots at intervals."""
    device = next(model.parameters()).device
    snapshots = []

    config = BayesianMCTSConfig(num_simulations=max_sims)
    mcts = BayesianMCTS(game, config)

    # Expand root
    roots, _ = mcts._batch_expand_roots(state[np.newaxis, ...], model, device)
    root = roots[0]

    np.random.seed(42)  # Reproducible

    for sim in range(1, max_sims + 1):
        # Run one simulation
        leaf_node, search_path, leaf_state, is_terminal = mcts._select_to_leaf(root, state)

        if is_terminal:
            value = game.terminal_reward(leaf_state)
        elif not leaf_node.expanded():
            values = mcts._batch_expand_leaves([leaf_node], [leaf_state], model, device)
            value = values[0]
        else:
            continue

        mcts._backup(search_path, value)

        # Take snapshot at intervals
        if sim % snapshot_interval == 0:
            policy = mcts._get_policy(root)
            best_action = int(np.argmax(policy))
            conf = float(policy[best_action])

            # Extract child beliefs and visits
            action_values = {}
            action_visits = {}
            for a, child in root.children.items():
                action_values[a] = -child.mu  # Parent perspective
                action_visits[a] = child.visits

            snapshots.append(SimSnapshot(
                sim_count=sim,
                best_action=best_action,
                confidence=conf,
                entropy=entropy(policy),
                action_probs=policy.copy(),
                action_values=action_values,
                action_visits=action_visits,
            ))

    return snapshots


def find_first_correct(snapshots: List[SimSnapshot], ground_truth_action: int) -> Optional[int]:
    """Find first simulation count where best_action matches ground truth."""
    for snap in snapshots:
        if snap.best_action == ground_truth_action:
            return snap.sim_count
    return None


def print_evolution_comparison(
    puct_snaps: List[SimSnapshot],
    ttts_snaps: List[SimSnapshot],
    ground_truth_action: int,
    legal_actions: List[int],
):
    """Print side-by-side comparison of search evolution."""
    print("\n" + "=" * 90)
    print("SEARCH EVOLUTION COMPARISON")
    print("=" * 90)
    print(f"Ground truth best action: {ground_truth_action}")

    puct_first = find_first_correct(puct_snaps, ground_truth_action)
    ttts_first = find_first_correct(ttts_snaps, ground_truth_action)

    print(f"\nFirst correct identification:")
    print(f"  PUCT: {puct_first if puct_first else 'Never'} sims")
    print(f"  TTTS: {ttts_first if ttts_first else 'Never'} sims")

    if puct_first and ttts_first:
        diff = ttts_first - puct_first
        if diff < 0:
            print(f"  --> TTTS finds it {-diff} sims EARLIER")
        elif diff > 0:
            print(f"  --> TTTS finds it {diff} sims LATER")
        else:
            print(f"  --> Same time")

    print("\n" + "-" * 90)
    print(f"{'Sims':>6} | {'PUCT Best':>9} | {'PUCT Conf':>9} | {'TTTS Best':>9} | {'TTTS Conf':>9} | {'PUCT H':>7} | {'TTTS H':>7}")
    print("-" * 90)

    for p, t in zip(puct_snaps, ttts_snaps):
        p_mark = "*" if p.best_action == ground_truth_action else " "
        t_mark = "*" if t.best_action == ground_truth_action else " "
        print(f"{p.sim_count:>6} | {p.best_action:>8}{p_mark} | {p.confidence:>9.3f} | {t.best_action:>8}{t_mark} | {t.confidence:>9.3f} | {p.entropy:>7.3f} | {t.entropy:>7.3f}")

    # Confidence on ground truth action over time
    print("\n" + "-" * 90)
    print(f"Confidence on ground truth action ({ground_truth_action}) over time:")
    print(f"{'Sims':>6} | {'PUCT':>10} | {'TTTS':>10} | {'Diff':>10}")
    print("-" * 40)

    for p, t in zip(puct_snaps, ttts_snaps):
        p_conf = p.action_probs[ground_truth_action]
        t_conf = t.action_probs[ground_truth_action]
        diff = t_conf - p_conf
        diff_str = f"+{diff:.3f}" if diff > 0 else f"{diff:.3f}"
        print(f"{p.sim_count:>6} | {p_conf:>10.3f} | {t_conf:>10.3f} | {diff_str:>10}")


def print_value_comparison(
    puct_snaps: List[SimSnapshot],
    ttts_snaps: List[SimSnapshot],
    legal_actions: List[int],
    sim_count: int,
):
    """Print value/belief comparison at a specific sim count."""
    puct_snap = next((s for s in puct_snaps if s.sim_count == sim_count), None)
    ttts_snap = next((s for s in ttts_snaps if s.sim_count == sim_count), None)

    if not puct_snap or not ttts_snap:
        return

    print(f"\n{'=' * 60}")
    print(f"VALUE/BELIEF COMPARISON AT {sim_count} SIMS")
    print("=" * 60)

    print(f"{'Action':>7} | {'PUCT Prob':>10} | {'TTTS Prob':>10} | {'TTTS Value':>11} | {'TTTS Visits':>12}")
    print("-" * 60)

    for a in sorted(legal_actions):
        p_prob = puct_snap.action_probs[a]
        t_prob = ttts_snap.action_probs[a]
        t_val = ttts_snap.action_values.get(a, 0.0)
        t_vis = ttts_snap.action_visits.get(a, 0)
        print(f"{a:>7} | {p_prob:>10.4f} | {t_prob:>10.4f} | {t_val:>11.4f} | {t_vis:>12}")


def get_test_positions(game: Game) -> Dict[str, Tuple[np.ndarray, str]]:
    """Get test positions with descriptions."""
    positions = {}

    # Empty board
    positions['empty'] = (game.initial_state(), "Empty board")

    # Center opening response
    state = game.initial_state()
    state = game.next_state(state, 3)
    positions['after_center'] = (state, "After center opening")

    # Midgame position 1
    state = game.initial_state()
    for m in [3, 4, 6, 2, 4, 4, 6, 1, 2, 6, 2]:
        state = game.next_state(state, m)
    positions['midgame1'] = (state, "Midgame (11 moves)")

    # Midgame position 2
    state = game.initial_state()
    for m in [4, 5, 2, 3, 6, 5, 6]:
        state = game.next_state(state, m)
    positions['midgame2'] = (state, "Midgame (7 moves)")

    return positions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='connect4')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/connect4_iter150.pt')
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--max_sims', type=int, default=200)
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--position', type=str, default='midgame1')
    parser.add_argument('--ground_truth_sims', type=int, default=2000)
    args = parser.parse_args()

    device = get_device()
    game = get_game(args.game)
    model_config = get_model_config(game.config, n_layer=args.n_layer)
    model = AlphaZeroTransformer(model_config).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    positions = get_test_positions(game)
    if args.position not in positions:
        print(f"Unknown position. Available: {list(positions.keys())}")
        return

    state, desc = positions[args.position]
    legal_actions = game.legal_actions(state)

    print(f"Position: {args.position} - {desc}")
    print(f"Legal actions: {legal_actions}")
    print(game.display(state))

    # Get ground truth from high-sim PUCT
    print(f"\nComputing ground truth with {args.ground_truth_sims} PUCT simulations...")
    gt_config = MCTSConfig(num_simulations=args.ground_truth_sims)
    gt_mcts = BatchedMCTS(game, gt_config)
    gt_policy = gt_mcts.search(state[np.newaxis, ...], model, num_simulations=args.ground_truth_sims, add_noise=False)[0]
    ground_truth_action = int(np.argmax(gt_policy))
    print(f"Ground truth best action: {ground_truth_action} (prob={gt_policy[ground_truth_action]:.3f})")

    # Run both algorithms with snapshots
    print(f"\nRunning PUCT with snapshots (max {args.max_sims} sims)...")
    puct_snaps = run_puct_with_snapshots(game, model, state, args.max_sims, args.interval)

    print(f"Running TTTS with snapshots (max {args.max_sims} sims)...")
    ttts_snaps = run_ttts_with_snapshots(game, model, state, args.max_sims, args.interval)

    # Print comparison
    print_evolution_comparison(puct_snaps, ttts_snaps, ground_truth_action, legal_actions)

    # Print value comparison at key points
    for sims in [50, 100, 200]:
        if sims <= args.max_sims:
            print_value_comparison(puct_snaps, ttts_snaps, legal_actions, sims)


if __name__ == '__main__':
    main()
