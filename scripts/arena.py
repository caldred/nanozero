"""
scripts/arena.py - Pit two models or MCTS algorithms against each other

Usage:
    # Compare two models:
    python -m scripts.arena --game=connect4 \
        --model1=checkpoints/iter50.pt \
        --model2=checkpoints/iter100.pt

    # Compare MCTS algorithms (PUCT vs TTTS-IDS):
    python -m scripts.arena --game=connect4 \
        --model1=checkpoints/connect4_final.pt \
        --mcts_comparison
"""
import argparse
import numpy as np
import torch
from scipy import stats

from nanozero.game import get_game
from nanozero.model import AlphaZeroTransformer
from nanozero.mcts import BatchedMCTS, BayesianMCTS
from nanozero.config import get_model_config, MCTSConfig, BayesianMCTSConfig
from nanozero.common import sample_action
from nanozero.common import get_device, load_checkpoint, print0


@torch.inference_mode()
def arena(
    game,
    model1, mcts1,
    model2, mcts2,
    num_games: int = 100,
    is_bayesian1: bool = False,
    is_bayesian2: bool = False,
) -> dict:
    """
    Play games between two models in parallel with batched MCTS.

    All games run simultaneously. Each turn, we batch MCTS calls for
    all games where that player is to move.

    Returns results from model1's perspective.
    """
    model1.eval()
    model2.eval()

    # Initialize all games
    states = [game.initial_state() for _ in range(num_games)]
    p1_colors = [1 if i % 2 == 0 else -1 for i in range(num_games)]  # Alternate who goes first
    results = [None] * num_games

    while any(r is None for r in results):
        # Check for terminal states
        for i, (state, result) in enumerate(zip(states, results)):
            if result is not None:
                continue
            if game.is_terminal(state):
                reward = game.terminal_reward(state)
                final_player = game.current_player(state)
                p1_result = reward if final_player == p1_colors[i] else -reward
                results[i] = p1_result
                continue

        # Separate games by whose turn it is
        p1_turn_indices = []
        p2_turn_indices = []

        for i, (state, result) in enumerate(zip(states, results)):
            if result is not None:
                continue
            current = game.current_player(state)
            if current == p1_colors[i]:
                p1_turn_indices.append(i)
            else:
                p2_turn_indices.append(i)

        # Batch MCTS for player 1's moves
        if p1_turn_indices:
            p1_states = np.stack([states[i] for i in p1_turn_indices])
            if is_bayesian1:
                policies = mcts1.search(p1_states, model1)
            else:
                policies = mcts1.search(p1_states, model1, add_noise=False)

            for idx, game_idx in enumerate(p1_turn_indices):
                action = sample_action(policies[idx], temperature=0)
                states[game_idx] = game.next_state(states[game_idx], action)

        # Batch MCTS for player 2's moves
        if p2_turn_indices:
            p2_states = np.stack([states[i] for i in p2_turn_indices])
            if is_bayesian2:
                policies = mcts2.search(p2_states, model2)
            else:
                policies = mcts2.search(p2_states, model2, add_noise=False)

            for idx, game_idx in enumerate(p2_turn_indices):
                action = sample_action(policies[idx], temperature=0)
                states[game_idx] = game.next_state(states[game_idx], action)

    # Tally results
    wins = sum(1 for r in results if r > 0)
    losses = sum(1 for r in results if r < 0)
    draws = sum(1 for r in results if r == 0)

    return {
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'win_rate': wins / num_games,
    }


@torch.inference_mode()
def mcts_comparison_arena(
    game,
    model,
    puct_mcts: BatchedMCTS,
    ttts_mcts: BayesianMCTS,
    num_games: int = 100,
) -> dict:
    """
    Compare MCTS algorithms using the same model in parallel.

    All games run simultaneously with batched MCTS.
    TTTS-IDS plays as player 1.
    Returns results from TTTS-IDS perspective.
    """
    model.eval()

    # Initialize all games
    states = [game.initial_state() for _ in range(num_games)]
    ttts_colors = [1 if i % 2 == 0 else -1 for i in range(num_games)]  # Alternate who goes first
    results = [None] * num_games

    while any(r is None for r in results):
        # Check for terminal states
        for i, (state, result) in enumerate(zip(states, results)):
            if result is not None:
                continue
            if game.is_terminal(state):
                reward = game.terminal_reward(state)
                final_player = game.current_player(state)
                ttts_result = reward if final_player == ttts_colors[i] else -reward
                results[i] = ttts_result
                continue

        # Separate games by whose turn it is
        ttts_turn_indices = []
        puct_turn_indices = []

        for i, (state, result) in enumerate(zip(states, results)):
            if result is not None:
                continue
            current = game.current_player(state)
            if current == ttts_colors[i]:
                ttts_turn_indices.append(i)
            else:
                puct_turn_indices.append(i)

        # Batch MCTS for TTTS moves
        if ttts_turn_indices:
            ttts_states = np.stack([states[i] for i in ttts_turn_indices])
            policies = ttts_mcts.search(ttts_states, model)

            for idx, game_idx in enumerate(ttts_turn_indices):
                action = sample_action(policies[idx], temperature=0)
                states[game_idx] = game.next_state(states[game_idx], action)

        # Batch MCTS for PUCT moves
        if puct_turn_indices:
            puct_states = np.stack([states[i] for i in puct_turn_indices])
            policies = puct_mcts.search(puct_states, model, add_noise=False)

            for idx, game_idx in enumerate(puct_turn_indices):
                action = sample_action(policies[idx], temperature=0)
                states[game_idx] = game.next_state(states[game_idx], action)

    # Tally results
    wins = sum(1 for r in results if r > 0)
    losses = sum(1 for r in results if r < 0)
    draws = sum(1 for r in results if r == 0)

    # Binomial test for significance
    decisive_games = wins + losses
    if decisive_games > 0:
        result = stats.binomtest(wins, decisive_games, 0.5, alternative='two-sided')
        p_value = result.pvalue
    else:
        p_value = 1.0

    return {
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'win_rate': wins / num_games if num_games > 0 else 0,
        'decisive_win_rate': wins / decisive_games if decisive_games > 0 else 0,
        'p_value': p_value,
        'significant': p_value < 0.05,
    }


def main():
    parser = argparse.ArgumentParser(description='Arena: pit models or MCTS algorithms against each other')

    # Game settings
    parser.add_argument('--game', type=str, default='connect4',
                        help='Game to play (tictactoe, connect4, go9x9, go19x19)')

    # Model settings
    parser.add_argument('--model1', type=str, required=True,
                        help='Path to first model (or only model for MCTS comparison)')
    parser.add_argument('--model2', type=str, default=None,
                        help='Path to second model (not needed for MCTS comparison)')
    parser.add_argument('--n_layer', type=int, default=4,
                        help='Number of transformer layers')

    # Arena settings
    parser.add_argument('--num_games', type=int, default=100,
                        help='Number of games to play')
    parser.add_argument('--mcts_simulations', type=int, default=500,
                        help='MCTS simulations per move')
    parser.add_argument('--leaves_per_batch', type=int, default=64,
                        help='Leaves per NN batch for virtual loss batching')

    # PUCT settings
    parser.add_argument('--c_puct', type=float, default=1.5,
                        help='PUCT exploration constant')

    # Bayesian MCTS settings
    parser.add_argument('--sigma_0', type=float, default=1.0,
                        help='Prior standard deviation for Bayesian beliefs')
    parser.add_argument('--obs_var', type=float, default=0.5,
                        help='Observation variance for NN value updates')
    parser.add_argument('--ids_alpha', type=float, default=0.5,
                        help='IDS pseudocount for exploration')

    # Mode settings
    parser.add_argument('--mcts_comparison', action='store_true',
                        help='Compare PUCT vs TTTS-IDS using the same model')
    parser.add_argument('--bayesian1', action='store_true',
                        help='Use BayesianMCTS for model1')
    parser.add_argument('--bayesian2', action='store_true',
                        help='Use BayesianMCTS for model2')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, cuda, mps)')

    args = parser.parse_args()

    # Validate args
    if not args.mcts_comparison and args.model2 is None:
        parser.error("--model2 is required unless --mcts_comparison is set")

    device = get_device() if args.device == 'auto' else torch.device(args.device)
    print0(f"Using device: {device}")

    game = get_game(args.game)
    print0(f"Game: {args.game} (backend: {game.backend})")
    print0(f"Board size: {game.config.board_size}, Action size: {game.config.action_size}")

    model_config = get_model_config(game.config, n_layer=args.n_layer)

    if args.mcts_comparison:
        # MCTS algorithm comparison mode
        model = AlphaZeroTransformer(model_config).to(device)
        load_checkpoint(args.model1, model)
        model.eval()

        print0(f"Model: {model.count_parameters():,} parameters")

        puct_config = MCTSConfig(
            num_simulations=args.mcts_simulations,
            c_puct=args.c_puct,
        )
        puct_mcts = BatchedMCTS(game, puct_config, leaves_per_batch=args.leaves_per_batch)

        ttts_config = BayesianMCTSConfig(
            num_simulations=args.mcts_simulations,
            sigma_0=args.sigma_0,
            obs_var=args.obs_var,
            ids_alpha=args.ids_alpha,
        )
        ttts_mcts = BayesianMCTS(game, ttts_config, leaves_per_batch=args.leaves_per_batch)

        print0(f"\nMCTS Algorithm Arena")
        print0(f"Model: {args.model1}")
        print0(f"TTTS-IDS vs PUCT")
        print0(f"Simulations: {args.mcts_simulations}, Leaves/batch: {args.leaves_per_batch}")
        print0(f"Playing {args.num_games} games in parallel...\n")

        results = mcts_comparison_arena(
            game, model, puct_mcts, ttts_mcts,
            num_games=args.num_games,
        )

        print0(f"Results (TTTS-IDS perspective):")
        print0(f"  Wins: {results['wins']}")
        print0(f"  Draws: {results['draws']}")
        print0(f"  Losses: {results['losses']}")
        print0(f"  Win rate: {results['win_rate']:.1%}")
        print0(f"  Decisive win rate: {results['decisive_win_rate']:.1%}")
        print0(f"  p-value: {results['p_value']:.4f}")
        print0(f"  Significant (p < 0.05): {'Yes' if results['significant'] else 'No'}")

    else:
        # Model comparison mode
        model1 = AlphaZeroTransformer(model_config).to(device)
        model2 = AlphaZeroTransformer(model_config).to(device)

        load_checkpoint(args.model1, model1)
        load_checkpoint(args.model2, model2)

        model1.eval()
        model2.eval()

        print0(f"Model: {model1.count_parameters():,} parameters")

        # Create MCTS instances based on flags
        if args.bayesian1:
            mcts1_config = BayesianMCTSConfig(
                num_simulations=args.mcts_simulations,
                sigma_0=args.sigma_0,
                obs_var=args.obs_var,
                ids_alpha=args.ids_alpha,
            )
            mcts1 = BayesianMCTS(game, mcts1_config, leaves_per_batch=args.leaves_per_batch)
        else:
            mcts1_config = MCTSConfig(
                num_simulations=args.mcts_simulations,
                c_puct=args.c_puct,
            )
            mcts1 = BatchedMCTS(game, mcts1_config, leaves_per_batch=args.leaves_per_batch)

        if args.bayesian2:
            mcts2_config = BayesianMCTSConfig(
                num_simulations=args.mcts_simulations,
                sigma_0=args.sigma_0,
                obs_var=args.obs_var,
                ids_alpha=args.ids_alpha,
            )
            mcts2 = BayesianMCTS(game, mcts2_config, leaves_per_batch=args.leaves_per_batch)
        else:
            mcts2_config = MCTSConfig(
                num_simulations=args.mcts_simulations,
                c_puct=args.c_puct,
            )
            mcts2 = BatchedMCTS(game, mcts2_config, leaves_per_batch=args.leaves_per_batch)

        print0(f"\nModel Arena")
        print0(f"Model 1: {args.model1} ({'Bayesian' if args.bayesian1 else 'PUCT'})")
        print0(f"Model 2: {args.model2} ({'Bayesian' if args.bayesian2 else 'PUCT'})")
        print0(f"Simulations: {args.mcts_simulations}, Leaves/batch: {args.leaves_per_batch}")
        print0(f"Playing {args.num_games} games in parallel...\n")

        results = arena(
            game, model1, mcts1, model2, mcts2,
            num_games=args.num_games,
            is_bayesian1=args.bayesian1,
            is_bayesian2=args.bayesian2,
        )

        print0(f"Results (Model 1 perspective):")
        print0(f"  Wins: {results['wins']}")
        print0(f"  Draws: {results['draws']}")
        print0(f"  Losses: {results['losses']}")
        print0(f"  Win rate: {results['win_rate']:.1%}")


if __name__ == '__main__':
    main()
