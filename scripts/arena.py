"""
scripts/arena.py - Pit two models or MCTS algorithms against each other

Usage:
    # Compare two models:
    python -m scripts.arena --game=tictactoe \
        --model1=checkpoints/iter50.pt \
        --model2=checkpoints/iter100.pt

    # Compare MCTS algorithms (PUCT vs TTTS-IDS):
    python -m scripts.arena --game=tictactoe \
        --model1=checkpoints/tictactoe_final.pt \
        --mcts_comparison
"""
import argparse
import math
import numpy as np
import torch
from scipy import stats

from nanozero.game import get_game
from nanozero.model import AlphaZeroTransformer
from nanozero.mcts import BatchedMCTS, sample_action
from nanozero.bayesian_mcts import BayesianMCTS
from nanozero.config import get_model_config, MCTSConfig, BayesianMCTSConfig
from nanozero.common import get_device, load_checkpoint, print0


def arena(
    game,
    model1, mcts1,
    model2, mcts2,
    num_games: int = 100,
    mcts_simulations: int = 50
) -> dict:
    """
    Play games between two models.

    Returns results from model1's perspective.
    """

    def make_player(model, mcts):
        def play(state):
            policy = mcts.search(
                state[np.newaxis, ...],
                model,
                num_simulations=mcts_simulations,
                add_noise=False
            )[0]
            return sample_action(policy, temperature=0)
        return play

    player1 = make_player(model1, mcts1)
    player2 = make_player(model2, mcts2)

    wins, draws, losses = 0, 0, 0

    for i in range(num_games):
        state = game.initial_state()

        # Alternate who plays first
        if i % 2 == 0:
            p1_turn = 1
        else:
            p1_turn = -1

        while not game.is_terminal(state):
            current = game.current_player(state)
            if current == p1_turn:
                action = player1(state)
            else:
                action = player2(state)
            state = game.next_state(state, action)

        # Get result
        reward = game.terminal_reward(state)
        final_player = game.current_player(state)

        if final_player == p1_turn:
            p1_result = reward
        else:
            p1_result = -reward

        if p1_result > 0:
            wins += 1
        elif p1_result < 0:
            losses += 1
        else:
            draws += 1

        if (i + 1) % 20 == 0:
            print0(f"  Progress: {i+1}/{num_games}")

    return {
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'win_rate': wins / num_games,
    }


def mcts_comparison_arena(
    game,
    model,
    puct_mcts: BatchedMCTS,
    ttts_mcts: BayesianMCTS,
    num_games: int = 100,
    mcts_simulations: int = 50
) -> dict:
    """
    Compare MCTS algorithms using the same model.

    TTTS-IDS plays as player 1.
    Returns results from TTTS-IDS perspective.
    """

    def make_puct_player():
        def play(state):
            policy = puct_mcts.search(
                state[np.newaxis, ...],
                model,
                num_simulations=mcts_simulations,
                add_noise=False
            )[0]
            return sample_action(policy, temperature=0)
        return play

    def make_ttts_player():
        def play(state):
            policy = ttts_mcts.search(
                state[np.newaxis, ...],
                model,
                num_simulations=mcts_simulations
            )[0]
            return sample_action(policy, temperature=0)
        return play

    ttts_player = make_ttts_player()
    puct_player = make_puct_player()

    wins, draws, losses = 0, 0, 0

    for i in range(num_games):
        state = game.initial_state()

        # Alternate who plays first
        if i % 2 == 0:
            ttts_turn = 1  # TTTS goes first
        else:
            ttts_turn = -1  # PUCT goes first

        while not game.is_terminal(state):
            current = game.current_player(state)
            if current == ttts_turn:
                action = ttts_player(state)
            else:
                action = puct_player(state)
            state = game.next_state(state, action)

        # Get result from TTTS perspective
        reward = game.terminal_reward(state)
        final_player = game.current_player(state)

        if final_player == ttts_turn:
            ttts_result = reward
        else:
            ttts_result = -reward

        if ttts_result > 0:
            wins += 1
        elif ttts_result < 0:
            losses += 1
        else:
            draws += 1

        if (i + 1) % 20 == 0:
            print0(f"  Progress: {i+1}/{num_games}")

    # Binomial test for significance
    # H0: win probability = 0.5 (accounting for draws)
    decisive_games = wins + losses
    if decisive_games > 0:
        # Two-sided binomial test
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
    parser.add_argument('--game', type=str, required=True)
    parser.add_argument('--model1', type=str, required=True,
                        help='Path to first model (or only model for MCTS comparison)')
    parser.add_argument('--model2', type=str, default=None,
                        help='Path to second model (not needed for MCTS comparison)')
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--num_games', type=int, default=100)
    parser.add_argument('--mcts_simulations', type=int, default=50)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--mcts_comparison', action='store_true',
                        help='Compare PUCT vs TTTS-IDS using the same model')

    args = parser.parse_args()

    # Validate args
    if not args.mcts_comparison and args.model2 is None:
        parser.error("--model2 is required unless --mcts_comparison is set")

    device = get_device() if args.device == 'auto' else torch.device(args.device)

    game = get_game(args.game)
    model_config = get_model_config(game.config, n_layer=args.n_layer)

    if args.mcts_comparison:
        # MCTS algorithm comparison mode
        model = AlphaZeroTransformer(model_config).to(device)
        load_checkpoint(args.model1, model)
        model.eval()

        puct_config = MCTSConfig(num_simulations=args.mcts_simulations)
        puct_mcts = BatchedMCTS(game, puct_config)

        ttts_config = BayesianMCTSConfig(num_simulations=args.mcts_simulations)
        ttts_mcts = BayesianMCTS(game, ttts_config)

        print0(f"MCTS Algorithm Arena: {args.game}")
        print0(f"Model: {args.model1}")
        print0(f"TTTS-IDS vs PUCT")
        print0(f"Playing {args.num_games} games with {args.mcts_simulations} simulations each...\n")

        results = mcts_comparison_arena(
            game, model, puct_mcts, ttts_mcts,
            num_games=args.num_games,
            mcts_simulations=args.mcts_simulations
        )

        print0(f"\nResults (TTTS-IDS perspective):")
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

        mcts_config = MCTSConfig(num_simulations=args.mcts_simulations)
        mcts1 = BatchedMCTS(game, mcts_config)
        mcts2 = BatchedMCTS(game, mcts_config)

        print0(f"Arena: {args.game}")
        print0(f"Model 1: {args.model1}")
        print0(f"Model 2: {args.model2}")
        print0(f"Playing {args.num_games} games...\n")

        results = arena(
            game, model1, mcts1, model2, mcts2,
            num_games=args.num_games,
            mcts_simulations=args.mcts_simulations
        )

        print0(f"\nResults (Model 1 perspective):")
        print0(f"  Wins: {results['wins']}")
        print0(f"  Draws: {results['draws']}")
        print0(f"  Losses: {results['losses']}")
        print0(f"  Win rate: {results['win_rate']:.1%}")


if __name__ == '__main__':
    main()
