"""
scripts/benchmark_mcts.py - Compare MCTS algorithms on game positions

Compares BatchedMCTS (PUCT) vs BayesianMCTS (TTTS-IDS) on:
1. Policy quality (entropy, confidence)
2. Agreement between algorithms
3. Convergence speed

Usage:
    python -m scripts.benchmark_mcts --game tictactoe --checkpoint model.pt
    python -m scripts.benchmark_mcts --game connect4 --checkpoint model.pt --n_sims 200
"""
import argparse
import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from nanozero.game import get_game, Game
from nanozero.model import AlphaZeroTransformer
from nanozero.mcts import BatchedMCTS, BayesianMCTS
from nanozero.config import get_model_config, MCTSConfig, BayesianMCTSConfig
from nanozero.common import get_device, load_checkpoint


def entropy(probs: np.ndarray, eps: float = 1e-8) -> float:
    """Compute Shannon entropy of probability distribution."""
    probs = probs + eps
    probs = probs / probs.sum()
    return float(-np.sum(probs * np.log(probs)))


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    """Compute KL divergence D(p || q)."""
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


@dataclass
class PositionResult:
    """Results from comparing algorithms on a single position."""
    position_name: str
    n_sims: int
    puct_entropy: float
    ttts_entropy: float
    puct_confidence: float  # max(policy)
    ttts_confidence: float
    policy_kl: float
    puct_top1: int
    ttts_top1: int
    agree: bool


def get_test_positions(game: Game, game_name: str) -> Dict[str, np.ndarray]:
    """
    Get a set of test positions for a game.

    Returns dict of position_name -> state
    """
    positions = {}

    if game_name == 'tictactoe':
        # Empty board
        positions['empty'] = game.initial_state()

        # X in corner
        state = game.initial_state()
        state = game.next_state(state, 0)  # X in top-left
        positions['x_corner'] = state

        # X in center
        state = game.initial_state()
        state = game.next_state(state, 4)  # X in center
        positions['x_center'] = state

        # Midgame positions
        state = game.initial_state()
        state = game.next_state(state, 4)  # X center
        state = game.next_state(state, 0)  # O corner
        state = game.next_state(state, 8)  # X opposite corner
        positions['midgame_1'] = state

        # X about to win
        state = game.initial_state()
        state = game.next_state(state, 0)  # X
        state = game.next_state(state, 3)  # O
        state = game.next_state(state, 1)  # X
        state = game.next_state(state, 4)  # O
        # X can win by playing 2
        positions['x_winning'] = state

    elif game_name == 'connect4':
        # Empty board
        positions['empty'] = game.initial_state()

        # Center opening
        state = game.initial_state()
        state = game.next_state(state, 3)  # Center column
        positions['center_open'] = state

        # A few moves in
        state = game.initial_state()
        for move in [4, 5, 2, 3, 6, 5, 6]:
            state = game.next_state(state, move)
        positions['midgame_1'] = state

        # More developed
        state = game.initial_state()
        for move in [4, 5, 2, 3, 6, 5, 6]:
            state = game.next_state(state, move)
        positions['midgame_2'] = state

    elif game_name in ('go9x9', 'go19x19'):
        # Just empty board and a few moves
        positions['empty'] = game.initial_state()

        # One move
        state = game.initial_state()
        # Play near center
        if game_name == 'go9x9':
            state = game.next_state(state, 40)  # Near center of 9x9
        else:
            state = game.next_state(state, 180)  # Near center of 19x19
        positions['opening_1'] = state

    return positions


def compare_mcts_on_position(
    state: np.ndarray,
    game: Game,
    model: torch.nn.Module,
    puct_mcts: BatchedMCTS,
    ttts_mcts: BayesianMCTS,
    n_sims_list: List[int],
    position_name: str,
) -> List[PositionResult]:
    """
    Compare MCTS algorithms on a single position at various simulation counts.
    """
    results = []

    for n_sims in n_sims_list:
        # Clear caches for fair comparison
        puct_mcts.clear_cache()
        ttts_mcts.clear_cache()

        # Run both algorithms
        puct_policy = puct_mcts.search(
            state[np.newaxis, ...], model,
            num_simulations=n_sims, add_noise=False
        )[0]

        ttts_policy = ttts_mcts.search(
            state[np.newaxis, ...], model,
            num_simulations=n_sims
        )[0]

        result = PositionResult(
            position_name=position_name,
            n_sims=n_sims,
            puct_entropy=entropy(puct_policy),
            ttts_entropy=entropy(ttts_policy),
            puct_confidence=float(np.max(puct_policy)),
            ttts_confidence=float(np.max(ttts_policy)),
            policy_kl=kl_divergence(puct_policy, ttts_policy),
            puct_top1=int(np.argmax(puct_policy)),
            ttts_top1=int(np.argmax(ttts_policy)),
            agree=int(np.argmax(puct_policy)) == int(np.argmax(ttts_policy)),
        )
        results.append(result)

    return results


def measure_convergence(
    state: np.ndarray,
    game: Game,
    model: torch.nn.Module,
    mcts,  # Either BatchedMCTS or BayesianMCTS
    max_sims: int = 200,
    step: int = 10,
    mcts_type: str = 'puct',
) -> Dict:
    """
    Measure how policy converges as simulations increase.
    """
    entropies = []
    confidences = []
    sim_counts = list(range(step, max_sims + 1, step))

    for n_sims in sim_counts:
        mcts.clear_cache()

        if mcts_type == 'puct':
            policy = mcts.search(
                state[np.newaxis, ...], model,
                num_simulations=n_sims, add_noise=False
            )[0]
        else:
            policy = mcts.search(
                state[np.newaxis, ...], model,
                num_simulations=n_sims
            )[0]

        entropies.append(entropy(policy))
        confidences.append(float(np.max(policy)))

    return {
        'simulations': sim_counts,
        'entropy': entropies,
        'confidence': confidences,
    }


def print_position_results(results: List[PositionResult]):
    """Pretty print position comparison results."""
    print("\n" + "=" * 90)
    print("                         MCTS Position Comparison")
    print("=" * 90)

    # Group by position
    by_position = {}
    for r in results:
        if r.position_name not in by_position:
            by_position[r.position_name] = []
        by_position[r.position_name].append(r)

    print(f"\n{'Position':<15} {'n_sims':<8} {'PUCT H':<10} {'TTTS H':<10} {'PUCT Conf':<10} {'TTTS Conf':<10} {'KL Div':<10} {'Agree':<6}")
    print("-" * 90)

    for pos_name, pos_results in by_position.items():
        for r in pos_results:
            agree_str = "Yes" if r.agree else f"No ({r.puct_top1} vs {r.ttts_top1})"
            print(
                f"{r.position_name:<15} {r.n_sims:<8} "
                f"{r.puct_entropy:<10.3f} {r.ttts_entropy:<10.3f} "
                f"{r.puct_confidence:<10.3f} {r.ttts_confidence:<10.3f} "
                f"{r.policy_kl:<10.4f} {agree_str:<6}"
            )

    # Summary statistics
    agreement_rate = sum(1 for r in results if r.agree) / len(results) if results else 0
    avg_kl = np.mean([r.policy_kl for r in results]) if results else 0

    print("-" * 90)
    print(f"\nSummary: Agreement rate: {agreement_rate:.1%}, Avg KL divergence: {avg_kl:.4f}")


def print_convergence_comparison(
    puct_conv: Dict,
    ttts_conv: Dict,
    position_name: str
):
    """Print convergence comparison."""
    print(f"\n{'='*60}")
    print(f"Convergence Analysis: {position_name}")
    print(f"{'='*60}")
    print(f"\n{'Sims':<8} {'PUCT H':<10} {'TTTS H':<10} {'PUCT Conf':<10} {'TTTS Conf':<10}")
    print("-" * 60)

    for i, sims in enumerate(puct_conv['simulations']):
        print(
            f"{sims:<8} "
            f"{puct_conv['entropy'][i]:<10.3f} {ttts_conv['entropy'][i]:<10.3f} "
            f"{puct_conv['confidence'][i]:<10.3f} {ttts_conv['confidence'][i]:<10.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description='MCTS Algorithm Comparison')
    parser.add_argument('--game', type=str, required=True,
                        choices=['tictactoe', 'connect4', 'go9x9'])
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--n_sims', type=int, nargs='+', default=[25, 50, 100],
                        help='Simulation counts to test')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--convergence', action='store_true',
                        help='Run convergence analysis')
    parser.add_argument('--max_conv_sims', type=int, default=200,
                        help='Max simulations for convergence analysis')

    args = parser.parse_args()

    device = get_device() if args.device == 'auto' else torch.device(args.device)

    # Load game and model
    game = get_game(args.game)
    print(f"Game: {args.game} (backend: {game.backend})")
    model_config = get_model_config(game.config, n_layer=args.n_layer)
    model = AlphaZeroTransformer(model_config).to(device)

    load_checkpoint(args.checkpoint, model)
    model.eval()

    # Create MCTS instances
    puct_config = MCTSConfig()
    puct_mcts = BatchedMCTS(game, puct_config)

    ttts_config = BayesianMCTSConfig()
    ttts_mcts = BayesianMCTS(game, ttts_config)

    print(f"Comparing MCTS algorithms on {args.game}")
    print(f"Model: {args.checkpoint}")
    print(f"Simulations: {args.n_sims}")

    # Get test positions
    positions = get_test_positions(game, args.game)
    print(f"Test positions: {list(positions.keys())}")

    # Compare on each position
    all_results = []
    for pos_name, state in positions.items():
        results = compare_mcts_on_position(
            state, game, model, puct_mcts, ttts_mcts,
            args.n_sims, pos_name
        )
        all_results.extend(results)

    print_position_results(all_results)

    # Convergence analysis
    if args.convergence and positions:
        pos_name = list(positions.keys())[0]
        state = positions[pos_name]

        print(f"\nRunning convergence analysis on '{pos_name}'...")

        puct_conv = measure_convergence(
            state, game, model, puct_mcts,
            max_sims=args.max_conv_sims, mcts_type='puct'
        )
        ttts_conv = measure_convergence(
            state, game, model, ttts_mcts,
            max_sims=args.max_conv_sims, mcts_type='ttts'
        )

        print_convergence_comparison(puct_conv, ttts_conv, pos_name)


if __name__ == '__main__':
    main()
