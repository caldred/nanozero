"""
Debug script to test TTTS vs PUCT arena with explicit Python implementation.
"""
import numpy as np
import torch
from nanozero.game import get_game
from nanozero.model import AlphaZeroTransformer
from nanozero.mcts import BatchedMCTS, sample_action
from nanozero.bayesian_mcts import BayesianMCTS
from nanozero.config import get_model_config, MCTSConfig, BayesianMCTSConfig
from nanozero.common import get_device, load_checkpoint


def run_arena(game, model, puct_mcts, ttts_mcts, num_games, mcts_simulations):
    """Run arena, return results from TTTS perspective."""

    def make_puct_player():
        def play(state):
            policy = puct_mcts.search(
                state[np.newaxis, ...], model,
                num_simulations=mcts_simulations, add_noise=False
            )[0]
            return sample_action(policy, temperature=0)
        return play

    def make_ttts_player():
        def play(state):
            policy = ttts_mcts.search(
                state[np.newaxis, ...], model,
                num_simulations=mcts_simulations
            )[0]
            return sample_action(policy, temperature=0)
        return play

    ttts_player = make_ttts_player()
    puct_player = make_puct_player()

    wins, draws, losses = 0, 0, 0

    for i in range(num_games):
        state = game.initial_state()
        ttts_turn = 1 if i % 2 == 0 else -1

        while not game.is_terminal(state):
            current = game.current_player(state)
            if current == ttts_turn:
                action = ttts_player(state)
            else:
                action = puct_player(state)
            state = game.next_state(state, action)

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

    return wins, draws, losses


def main():
    device = get_device()
    game = get_game('connect4')
    print(f"Game backend: {game.backend}")

    model_config = get_model_config(game.config, n_layer=4)
    model = AlphaZeroTransformer(model_config).to(device)
    load_checkpoint('checkpoints/connect4_iter150.pt', model)
    model.eval()

    # Print current config
    ttts_config = BayesianMCTSConfig()
    print(f"\nTTTS Config:")
    print(f"  optimality_weight: {ttts_config.optimality_weight}")
    print(f"  adaptive_weight: {ttts_config.adaptive_weight}")
    print(f"  visit_scale: {ttts_config.visit_scale}")
    print(f"  prune_threshold: {ttts_config.prune_threshold}")

    puct_config = MCTSConfig()
    puct_mcts = BatchedMCTS(game, puct_config)
    ttts_mcts = BayesianMCTS(game, ttts_config)

    num_games = 50

    for n_sims in [100, 200, 400]:
        print(f"\n=== {n_sims} simulations ===")
        np.random.seed(42)
        wins, draws, losses = run_arena(
            game, model, puct_mcts, ttts_mcts,
            num_games=num_games, mcts_simulations=n_sims
        )
        total = wins + losses
        win_rate = wins / total if total > 0 else 0
        print(f"TTTS: {wins}W / {draws}D / {losses}L")
        print(f"Decisive win rate: {win_rate:.1%}")


if __name__ == '__main__':
    main()
