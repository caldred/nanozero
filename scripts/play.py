"""
scripts/play.py - Play against trained model

Usage:
    python -m scripts.play --game=tictactoe --checkpoint=checkpoints/tictactoe_final.pt
"""
import argparse
import numpy as np
import torch

from nanozero.game import get_game
from nanozero.model import AlphaZeroTransformer
from nanozero.mcts import BatchedMCTS
from nanozero.common import sample_action
from nanozero.config import get_model_config, MCTSConfig
from nanozero.common import get_device, load_checkpoint


def get_human_action(game, state) -> int:
    """Get action from human player."""
    legal = game.legal_actions(state)

    print(f"\nLegal actions: {legal}")

    while True:
        try:
            action = int(input("Your move: "))
            if action in legal:
                return action
            else:
                print(f"Illegal move! Choose from {legal}")
        except ValueError:
            print("Please enter a number")


def play_interactive(game, model, mcts, human_first: bool = True):
    """Play an interactive game against the model."""

    state = game.initial_state()
    human_player = 1 if human_first else -1

    print("\n" + "="*40)
    print("New game!")
    print(f"You are: {'X (first)' if human_first else 'O (second)'}")
    print("="*40)

    while not game.is_terminal(state):
        print("\n" + game.display(state))

        current = game.current_player(state)

        if current == human_player:
            action = get_human_action(game, state)
        else:
            print("\nAI is thinking...")
            policy = mcts.search(
                state[np.newaxis, ...],
                model,
                num_simulations=100,
                add_noise=False
            )[0]
            action = sample_action(policy, temperature=0)
            print(f"AI plays: {action}")

        state = game.next_state(state, action)

    # Game over
    print("\n" + game.display(state))
    print("\n" + "="*40)

    reward = game.terminal_reward(state)
    final_player = game.current_player(state)

    # Determine winner from human's perspective
    if final_player == human_player:
        human_result = reward
    else:
        human_result = -reward

    if human_result > 0:
        print("You win!")
    elif human_result < 0:
        print("AI wins!")
    else:
        print("It's a draw!")

    print("="*40)


def main():
    parser = argparse.ArgumentParser(description='Play against NanoZero')
    parser.add_argument('--game', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--device', type=str, default='auto')

    args = parser.parse_args()

    device = get_device() if args.device == 'auto' else torch.device(args.device)

    # Load game and model
    game = get_game(args.game)
    print(f"Game: {args.game} (backend: {game.backend})")
    model_config = get_model_config(game.config, n_layer=args.n_layer)
    model = AlphaZeroTransformer(model_config).to(device)

    load_checkpoint(args.checkpoint, model)
    model.eval()

    mcts_config = MCTSConfig(num_simulations=100, temperature=0)
    mcts = BatchedMCTS(game, mcts_config)

    print(f"Loaded model from {args.checkpoint}")
    print(f"Model has {model.count_parameters():,} parameters")

    # Play games
    while True:
        first = input("\nDo you want to go first? (y/n): ").lower()
        human_first = first != 'n'

        play_interactive(game, model, mcts, human_first)

        again = input("\nPlay again? (y/n): ").lower()
        if again != 'y':
            break

    print("\nThanks for playing!")


if __name__ == '__main__':
    main()
