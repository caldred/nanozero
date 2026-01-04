"""Quick arena test with different obs_var values."""
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
    wins, draws, losses = 0, 0, 0

    for i in range(num_games):
        state = game.initial_state()
        ttts_turn = 1 if i % 2 == 0 else -1

        while not game.is_terminal(state):
            current = game.current_player(state)
            if current == ttts_turn:
                policy = ttts_mcts.search(
                    state[np.newaxis, ...], model,
                    num_simulations=mcts_simulations
                )[0]
            else:
                policy = puct_mcts.search(
                    state[np.newaxis, ...], model,
                    num_simulations=mcts_simulations, add_noise=False
                )[0]
            action = sample_action(policy, temperature=0)
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
    game = get_game('connect4', use_rust=False)
    print(f"Game backend: {game.backend}")

    model_config = get_model_config(game.config, n_layer=4)
    model = AlphaZeroTransformer(model_config).to(device)
    load_checkpoint('checkpoints/connect4_iter150.pt', model)
    model.eval()

    puct_config = MCTSConfig()
    puct_mcts = BatchedMCTS(game, puct_config)

    num_games = 40

    # Test both=1.0 at different sim counts
    test_configs = [
        (1.0, 1.0, 100),
        (1.0, 1.0, 200),
        (1.0, 1.0, 400),
    ]

    print(f"\nArena test: {num_games} games, sigma_0=1.0, obs_var=1.0")
    print("Using pure optimality weights (no blending)")
    print("=" * 50)

    for sigma_0, obs_var, n_sims in test_configs:
        ttts_config = BayesianMCTSConfig(
            sigma_0=sigma_0,
            obs_var=obs_var,
        )
        ttts_mcts = BayesianMCTS(game, ttts_config)

        np.random.seed(42)
        wins, draws, losses = run_arena(
            game, model, puct_mcts, ttts_mcts,
            num_games=num_games, mcts_simulations=n_sims
        )

        decisive = wins + losses
        win_rate = wins / decisive if decisive > 0 else 0
        print(f"{n_sims} sims: {wins}W/{draws}D/{losses}L ({win_rate:.0%})")


if __name__ == '__main__':
    main()
