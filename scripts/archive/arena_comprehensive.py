"""
scripts/arena_comprehensive.py - Comprehensive arena between PUCT and TTTS at multiple sim counts
"""
import argparse
import numpy as np
import torch
from scipy import stats

from nanozero.game import get_game
from nanozero.model import AlphaZeroTransformer
from nanozero.mcts import BatchedMCTS, sample_action
from nanozero.bayesian_mcts import BayesianMCTS
from nanozero.config import get_model_config, MCTSConfig, BayesianMCTSConfig
from nanozero.common import get_device, load_checkpoint, print0


def run_arena(game, model, puct_mcts, ttts_mcts, num_games: int, mcts_simulations: int):
    """Run arena between PUCT and TTTS."""
    device = next(model.parameters()).device

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
        ttts_turn = 1 if i % 2 == 0 else -1

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

    # Stats
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
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='connect4')
    parser.add_argument('--model', type=str, default='checkpoints/connect4_iter150.pt')
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--num_games', type=int, default=50, help='Games per simulation count')
    parser.add_argument('--sim_counts', type=str, default='25,50,100,200,400',
                        help='Comma-separated simulation counts to test')
    args = parser.parse_args()

    device = get_device()
    game = get_game(args.game)
    model_config = get_model_config(game.config, n_layer=args.n_layer)

    model = AlphaZeroTransformer(model_config).to(device)
    load_checkpoint(args.model, model)
    model.eval()

    sim_counts = [int(x) for x in args.sim_counts.split(',')]

    print0(f"\n{'='*60}")
    print0(f"COMPREHENSIVE ARENA: TTTS vs PUCT")
    print0(f"Game: {args.game}")
    print0(f"Model: {args.model}")
    print0(f"Games per sim count: {args.num_games}")
    print0(f"Simulation counts: {sim_counts}")
    print0(f"{'='*60}\n")

    results = []

    for sims in sim_counts:
        print0(f"\n--- Testing with {sims} simulations ---")

        puct_config = MCTSConfig(num_simulations=sims)
        puct_mcts = BatchedMCTS(game, puct_config)

        ttts_config = BayesianMCTSConfig(num_simulations=sims)
        ttts_mcts = BayesianMCTS(game, ttts_config)

        r = run_arena(game, model, puct_mcts, ttts_mcts, args.num_games, sims)
        results.append((sims, r))

        print0(f"  TTTS: {r['wins']}W / {r['draws']}D / {r['losses']}L")
        print0(f"  Win rate: {r['win_rate']:.1%}, p-value: {r['p_value']:.4f}")

    # Summary table
    print0(f"\n{'='*60}")
    print0(f"SUMMARY")
    print0(f"{'='*60}")
    print0(f"{'Sims':>6} | {'TTTS W':>7} | {'Draw':>5} | {'TTTS L':>7} | {'Win%':>6} | {'p-val':>7}")
    print0(f"{'-'*6}-+-{'-'*7}-+-{'-'*5}-+-{'-'*7}-+-{'-'*6}-+-{'-'*7}")

    for sims, r in results:
        sig = '*' if r['p_value'] < 0.05 else ''
        print0(f"{sims:>6} | {r['wins']:>7} | {r['draws']:>5} | {r['losses']:>7} | {r['win_rate']*100:>5.1f}% | {r['p_value']:>6.4f}{sig}")

    # Overall
    total_wins = sum(r['wins'] for _, r in results)
    total_draws = sum(r['draws'] for _, r in results)
    total_losses = sum(r['losses'] for _, r in results)
    total_games = sum(args.num_games for _ in results)

    print0(f"{'-'*6}-+-{'-'*7}-+-{'-'*5}-+-{'-'*7}-+-{'-'*6}-+-{'-'*7}")
    print0(f"{'Total':>6} | {total_wins:>7} | {total_draws:>5} | {total_losses:>7} | {total_wins/total_games*100:>5.1f}% |")
    print0(f"\n{'='*60}")


if __name__ == '__main__':
    main()
