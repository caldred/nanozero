"""
scripts/eval.py - Evaluate trained model

Usage:
    python -m scripts.eval --game=connect4 --checkpoint=checkpoints/connect4_final.pt
"""
import argparse
import numpy as np
import torch

from nanozero.game import get_game
from nanozero.model import AlphaZeroTransformer
from nanozero.mcts import BatchedMCTS, BayesianMCTS
from nanozero.common import sample_action
from nanozero.config import get_model_config, MCTSConfig, BayesianMCTSConfig
from nanozero.common import get_device, print0, load_checkpoint


@torch.inference_mode()
def evaluate_vs_random(game, model, mcts, num_games=100, is_bayesian=False):
    """
    Evaluate model against random player.
    Runs all games in parallel with batched MCTS.

    Args:
        game: Game instance
        model: Neural network
        mcts: MCTS instance (BatchedMCTS or BayesianMCTS)
        num_games: Number of games to play
        is_bayesian: Whether mcts is BayesianMCTS

    Returns:
        Dict with win/draw/loss counts and rates
    """
    model.eval()

    # Initialize all games
    states = [game.initial_state() for _ in range(num_games)]
    model_players = [1 if i % 2 == 0 else -1 for i in range(num_games)]
    results = [None] * num_games

    while any(r is None for r in results):
        model_turn_indices = []
        random_turn_indices = []

        for i, (state, result) in enumerate(zip(states, results)):
            if result is not None:
                continue
            if game.is_terminal(state):
                reward = game.terminal_reward(state)
                final_player = game.current_player(state)
                model_result = reward if final_player == model_players[i] else -reward
                results[i] = model_result
                continue

            current = game.current_player(state)
            if current == model_players[i]:
                model_turn_indices.append(i)
            else:
                random_turn_indices.append(i)

        # Batch MCTS for all model moves
        if model_turn_indices:
            model_states = np.stack([states[i] for i in model_turn_indices])
            if is_bayesian:
                policies = mcts.search(model_states, model)
            else:
                policies = mcts.search(model_states, model, add_noise=False)

            for idx, game_idx in enumerate(model_turn_indices):
                action = sample_action(policies[idx], temperature=0)
                states[game_idx] = game.next_state(states[game_idx], action)

        # Random moves
        for game_idx in random_turn_indices:
            legal = game.legal_actions(states[game_idx])
            action = np.random.choice(legal)
            states[game_idx] = game.next_state(states[game_idx], action)

    wins = sum(1 for r in results if r > 0)
    losses = sum(1 for r in results if r < 0)
    draws = sum(1 for r in results if r == 0)

    return {
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'win_rate': wins / num_games,
        'draw_rate': draws / num_games,
        'loss_rate': losses / num_games,
    }


@torch.inference_mode()
def evaluate_vs_greedy(game, model, mcts, num_games=100, is_bayesian=False):
    """
    Evaluate model (with MCTS) against greedy model (no MCTS).
    Runs all games in parallel with batched inference.

    Args:
        game: Game instance
        model: Neural network
        mcts: MCTS instance
        num_games: Number of games to play
        is_bayesian: Whether mcts is BayesianMCTS

    Returns:
        Dict with win/draw/loss counts and rates
    """
    model.eval()
    device = next(model.parameters()).device

    # Initialize all games
    states = [game.initial_state() for _ in range(num_games)]
    mcts_players = [1 if i % 2 == 0 else -1 for i in range(num_games)]
    results = [None] * num_games

    while any(r is None for r in results):
        mcts_turn_indices = []
        greedy_turn_indices = []

        for i, (state, result) in enumerate(zip(states, results)):
            if result is not None:
                continue
            if game.is_terminal(state):
                reward = game.terminal_reward(state)
                final_player = game.current_player(state)
                mcts_result = reward if final_player == mcts_players[i] else -reward
                results[i] = mcts_result
                continue

            current = game.current_player(state)
            if current == mcts_players[i]:
                mcts_turn_indices.append(i)
            else:
                greedy_turn_indices.append(i)

        # Batch MCTS for MCTS player's moves
        if mcts_turn_indices:
            mcts_states = np.stack([states[i] for i in mcts_turn_indices])
            if is_bayesian:
                policies = mcts.search(mcts_states, model)
            else:
                policies = mcts.search(mcts_states, model, add_noise=False)

            for idx, game_idx in enumerate(mcts_turn_indices):
                action = sample_action(policies[idx], temperature=0)
                states[game_idx] = game.next_state(states[game_idx], action)

        # Batch greedy moves (raw network policy)
        if greedy_turn_indices:
            greedy_states = [states[i] for i in greedy_turn_indices]
            canonical_states = [game.canonical_state(s) for s in greedy_states]
            state_tensors = torch.stack([game.to_tensor(s) for s in canonical_states]).to(device)
            action_masks = torch.stack([
                torch.from_numpy(game.legal_actions_mask(s)).float()
                for s in canonical_states
            ]).to(device)

            policies, _ = model.predict(state_tensors, action_masks)

            for idx, game_idx in enumerate(greedy_turn_indices):
                action = int(policies[idx].argmax().item())
                states[game_idx] = game.next_state(states[game_idx], action)

    wins = sum(1 for r in results if r > 0)
    losses = sum(1 for r in results if r < 0)
    draws = sum(1 for r in results if r == 0)

    return {
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'win_rate': wins / num_games,
        'draw_rate': draws / num_games,
        'loss_rate': losses / num_games,
    }


@torch.inference_mode()
def evaluate_vs_mcts(game, model, mcts, num_games=100, is_bayesian=False):
    """
    Evaluate model against itself (MCTS vs MCTS).
    Runs all games in parallel with batched MCTS.

    This tests consistency - a well-trained model should win ~50% as both players.

    Args:
        game: Game instance
        model: Neural network
        mcts: MCTS instance
        num_games: Number of games to play
        is_bayesian: Whether mcts is BayesianMCTS

    Returns:
        Dict with win/draw/loss counts and rates (from player 1's perspective)
    """
    model.eval()

    # Initialize all games
    states = [game.initial_state() for _ in range(num_games)]
    results = [None] * num_games

    while any(r is None for r in results):
        p1_turn_indices = []
        p2_turn_indices = []

        for i, (state, result) in enumerate(zip(states, results)):
            if result is not None:
                continue
            if game.is_terminal(state):
                reward = game.terminal_reward(state)
                final_player = game.current_player(state)
                # Result from player 1's perspective
                results[i] = reward if final_player == 1 else -reward
                continue

            current = game.current_player(state)
            if current == 1:
                p1_turn_indices.append(i)
            else:
                p2_turn_indices.append(i)

        # Batch MCTS for player 1's moves
        if p1_turn_indices:
            p1_states = np.stack([states[i] for i in p1_turn_indices])
            if is_bayesian:
                policies = mcts.search(p1_states, model)
            else:
                policies = mcts.search(p1_states, model, add_noise=False)

            for idx, game_idx in enumerate(p1_turn_indices):
                action = sample_action(policies[idx], temperature=0)
                states[game_idx] = game.next_state(states[game_idx], action)

        # Batch MCTS for player 2's moves
        if p2_turn_indices:
            p2_states = np.stack([states[i] for i in p2_turn_indices])
            if is_bayesian:
                policies = mcts.search(p2_states, model)
            else:
                policies = mcts.search(p2_states, model, add_noise=False)

            for idx, game_idx in enumerate(p2_turn_indices):
                action = sample_action(policies[idx], temperature=0)
                states[game_idx] = game.next_state(states[game_idx], action)

    wins = sum(1 for r in results if r > 0)
    losses = sum(1 for r in results if r < 0)
    draws = sum(1 for r in results if r == 0)

    return {
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'win_rate': wins / num_games,
        'draw_rate': draws / num_games,
        'loss_rate': losses / num_games,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate NanoZero model')

    # Game settings
    parser.add_argument('--game', type=str, default='connect4',
                        help='Game to evaluate (tictactoe, connect4, go9x9, go19x19)')

    # Model settings
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--n_layer', type=int, default=4,
                        help='Number of transformer layers')

    # Evaluation settings
    parser.add_argument('--num_games', type=int, default=100,
                        help='Number of games per opponent')
    parser.add_argument('--mcts_simulations', type=int, default=500,
                        help='MCTS simulations per move')
    parser.add_argument('--leaves_per_batch', type=int, default=64,
                        help='Leaves per NN batch for virtual loss batching')

    # MCTS type
    parser.add_argument('--bayesian', action='store_true',
                        help='Use BayesianMCTS instead of PUCT')

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

    # Device
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, cuda, mps)')

    # What to evaluate
    parser.add_argument('--vs_random', action='store_true', default=True,
                        help='Evaluate vs random player')
    parser.add_argument('--vs_greedy', action='store_true', default=True,
                        help='Evaluate vs greedy player (no MCTS)')
    parser.add_argument('--vs_self', action='store_true', default=False,
                        help='Evaluate vs self (MCTS vs MCTS)')

    args = parser.parse_args()

    device = get_device() if args.device == 'auto' else torch.device(args.device)
    print0(f"Using device: {device}")

    # Load game and model
    game = get_game(args.game)
    print0(f"Game: {args.game} (backend: {game.backend})")
    print0(f"Board size: {game.config.board_size}, Action size: {game.config.action_size}")

    model_config = get_model_config(game.config, n_layer=args.n_layer)
    model = AlphaZeroTransformer(model_config).to(device)

    load_checkpoint(args.checkpoint, model)
    model.eval()

    print0(f"Model: {model.count_parameters():,} parameters")

    # Create MCTS
    if args.bayesian:
        mcts_config = BayesianMCTSConfig(
            num_simulations=args.mcts_simulations,
            sigma_0=args.sigma_0,
            obs_var=args.obs_var,
            ids_alpha=args.ids_alpha,
        )
        mcts = BayesianMCTS(game, mcts_config, leaves_per_batch=args.leaves_per_batch)
        mcts_type = "BayesianMCTS (TTTS-IDS)"
    else:
        mcts_config = MCTSConfig(
            num_simulations=args.mcts_simulations,
            c_puct=args.c_puct,
        )
        mcts = BatchedMCTS(game, mcts_config, leaves_per_batch=args.leaves_per_batch)
        mcts_type = "BatchedMCTS (PUCT)"

    print0(f"\nEvaluating: {args.checkpoint}")
    print0(f"MCTS: {mcts_type}")
    print0(f"Simulations: {args.mcts_simulations}, Leaves/batch: {args.leaves_per_batch}")
    print0(f"Playing {args.num_games} games per opponent in parallel\n")

    # Evaluate against different opponents
    if args.vs_random:
        print0("vs Random:")
        results = evaluate_vs_random(
            game, model, mcts,
            num_games=args.num_games,
            is_bayesian=args.bayesian
        )
        print0(f"  Wins: {results['wins']}, Draws: {results['draws']}, Losses: {results['losses']}")
        print0(f"  Win rate: {results['win_rate']:.1%}")
        print0()

    if args.vs_greedy:
        print0("vs Greedy (raw network, no MCTS):")
        results = evaluate_vs_greedy(
            game, model, mcts,
            num_games=args.num_games,
            is_bayesian=args.bayesian
        )
        print0(f"  Wins: {results['wins']}, Draws: {results['draws']}, Losses: {results['losses']}")
        print0(f"  Win rate: {results['win_rate']:.1%}")
        print0()

    if args.vs_self:
        print0("vs Self (MCTS vs MCTS):")
        results = evaluate_vs_mcts(
            game, model, mcts,
            num_games=args.num_games,
            is_bayesian=args.bayesian
        )
        print0(f"  Player 1 wins: {results['wins']}, Draws: {results['draws']}, Player 2 wins: {results['losses']}")
        print0(f"  Player 1 win rate: {results['win_rate']:.1%}")
        print0()


if __name__ == '__main__':
    main()
