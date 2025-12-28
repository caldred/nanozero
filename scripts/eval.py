"""
scripts/eval.py - Evaluate trained model

Usage:
    python -m scripts.eval --game=tictactoe --checkpoint=checkpoints/tictactoe_final.pt
"""
import argparse
import numpy as np
import torch

from nanozero.game import get_game
from nanozero.model import AlphaZeroTransformer
from nanozero.mcts import BatchedMCTS, sample_action
from nanozero.config import get_model_config, MCTSConfig
from nanozero.common import get_device, print0, load_checkpoint


def play_game(game, player1_fn, player2_fn):
    """
    Play a single game between two players.

    Args:
        game: Game instance
        player1_fn: Function(state) -> action for player 1
        player2_fn: Function(state) -> action for player 2

    Returns:
        1 if player1 wins, -1 if player2 wins, 0 for draw
    """
    state = game.initial_state()

    while not game.is_terminal(state):
        current = game.current_player(state)
        if current == 1:
            action = player1_fn(state)
        else:
            action = player2_fn(state)
        state = game.next_state(state, action)

    # Get result from player 1's perspective
    reward = game.terminal_reward(state)
    final_player = game.current_player(state)
    if final_player == 1:
        return -reward  # Player 2 just moved
    else:
        return reward  # Player 1 just moved


def make_random_player(game):
    """Create a random player function."""
    def play(state):
        legal = game.legal_actions(state)
        return np.random.choice(legal)
    return play


def make_mcts_player(game, model, mcts, num_simulations=50):
    """Create an MCTS player function."""
    def play(state):
        canonical = game.canonical_state(state)
        policy = mcts.search(
            canonical[np.newaxis, ...],
            model,
            num_simulations=num_simulations,
            add_noise=False
        )[0]
        return sample_action(policy, temperature=0)
    return play


def make_greedy_player(game, model):
    """Create a greedy player that uses raw network policy (no MCTS)."""
    def play(state):
        canonical = game.canonical_state(state)
        state_tensor = game.to_tensor(canonical).unsqueeze(0)
        action_mask = torch.from_numpy(
            game.legal_actions_mask(canonical)
        ).unsqueeze(0).float()

        device = next(model.parameters()).device
        state_tensor = state_tensor.to(device)
        action_mask = action_mask.to(device)

        policy, _ = model.predict(state_tensor, action_mask)
        return int(policy.argmax().item())
    return play


def evaluate(
    game,
    model,
    mcts,
    opponent_type: str = 'random',
    num_games: int = 100,
    mcts_simulations: int = 50
) -> dict:
    """
    Evaluate model against an opponent.

    Args:
        game: Game instance
        model: Neural network
        mcts: MCTS instance
        opponent_type: 'random', 'greedy', or 'mcts'
        num_games: Number of games to play
        mcts_simulations: Simulations for MCTS player

    Returns:
        Dict with win/draw/loss counts and rates
    """
    model.eval()

    # Create players
    model_player = make_mcts_player(game, model, mcts, mcts_simulations)

    if opponent_type == 'random':
        opponent = make_random_player(game)
    elif opponent_type == 'greedy':
        opponent = make_greedy_player(game, model)
    elif opponent_type == 'mcts':
        opponent = make_mcts_player(game, model, mcts, mcts_simulations)
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")

    wins, draws, losses = 0, 0, 0

    for i in range(num_games):
        # Alternate who plays first
        if i % 2 == 0:
            result = play_game(game, model_player, opponent)
        else:
            result = -play_game(game, opponent, model_player)

        if result > 0:
            wins += 1
        elif result < 0:
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
        'draw_rate': draws / num_games,
        'loss_rate': losses / num_games,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate NanoZero model')
    parser.add_argument('--game', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--num_games', type=int, default=100)
    parser.add_argument('--mcts_simulations', type=int, default=50)
    parser.add_argument('--device', type=str, default='auto')

    args = parser.parse_args()

    device = get_device() if args.device == 'auto' else torch.device(args.device)

    # Load game and model
    game = get_game(args.game)
    model_config = get_model_config(game.config, n_layer=args.n_layer)
    model = AlphaZeroTransformer(model_config).to(device)

    load_checkpoint(args.checkpoint, model)
    model.eval()

    mcts_config = MCTSConfig(num_simulations=args.mcts_simulations)
    mcts = BatchedMCTS(game, mcts_config)

    print0(f"Evaluating {args.game} model from {args.checkpoint}")
    print0(f"Playing {args.num_games} games against each opponent\n")

    # Evaluate against different opponents
    for opponent in ['random', 'greedy']:
        print0(f"vs {opponent}:")
        results = evaluate(
            game, model, mcts,
            opponent_type=opponent,
            num_games=args.num_games,
            mcts_simulations=args.mcts_simulations
        )
        print0(f"  Wins: {results['wins']}, Draws: {results['draws']}, Losses: {results['losses']}")
        print0(f"  Win rate: {results['win_rate']:.1%}")
        print0()


if __name__ == '__main__':
    main()
