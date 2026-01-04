"""
scripts/trace_sign_error.py - Trace player alternation to verify sign conventions
"""
import numpy as np
import torch
from nanozero.game import get_game
from nanozero.model import AlphaZeroTransformer
from nanozero.config import get_model_config, BayesianMCTSConfig
from nanozero.common import get_device, load_checkpoint
from nanozero.bayesian_mcts import BayesianMCTS


def main():
    device = get_device()
    game = get_game('connect4')
    model_config = get_model_config(game.config, n_layer=4)
    model = AlphaZeroTransformer(model_config).to(device)
    load_checkpoint('checkpoints/connect4_iter150.pt', model)
    model.eval()

    # Use the midgame1 position
    state = game.initial_state()
    for m in [3, 4, 6, 2, 4, 4, 6, 1, 2, 6, 2]:
        state = game.next_state(state, m)

    print("=== TRACING PLAYER ALTERNATION ===\n")

    # Root position
    print(f"ROOT: player {game.current_player(state)} to move")
    root_player = game.current_player(state)

    # Get network value at root
    canonical = game.canonical_state(state)
    state_tensor = game.to_tensor(canonical).unsqueeze(0).to(device)
    action_mask = torch.from_numpy(game.legal_actions_mask(state)).unsqueeze(0).float().to(device)
    policy, value = model.predict(state_tensor, action_mask)
    root_value = value.cpu().item()
    print(f"  Network value: {root_value:.4f} (from player {root_player}'s perspective)")
    if root_value > 0:
        print(f"  -> Player {root_player} is WINNING")
    else:
        print(f"  -> Player {root_player} is LOSING")

    # After action 3
    state_a3 = game.next_state(state, 3)
    print(f"\nAfter action 3: player {game.current_player(state_a3)} to move")
    a3_player = game.current_player(state_a3)

    canonical_a3 = game.canonical_state(state_a3)
    state_tensor_a3 = game.to_tensor(canonical_a3).unsqueeze(0).to(device)
    action_mask_a3 = torch.from_numpy(game.legal_actions_mask(state_a3)).unsqueeze(0).float().to(device)
    policy_a3, value_a3 = model.predict(state_tensor_a3, action_mask_a3)
    a3_value = value_a3.cpu().item()
    print(f"  Network value: {a3_value:.4f} (from player {a3_player}'s perspective)")
    if a3_value > 0:
        print(f"  -> Player {a3_player} is WINNING")
    else:
        print(f"  -> Player {a3_player} is LOSING")

    # From root's perspective
    print(f"  From root's perspective (player {root_player}): {-a3_value:.4f}")
    if -a3_value > 0:
        print(f"  -> After action 3, player {root_player} is WINNING")
    else:
        print(f"  -> After action 3, player {root_player} is LOSING")

    # After action 3, 3
    state_a33 = game.next_state(state_a3, 3)
    print(f"\nAfter action 3, 3: player {game.current_player(state_a33)} to move")
    a33_player = game.current_player(state_a33)

    canonical_a33 = game.canonical_state(state_a33)
    state_tensor_a33 = game.to_tensor(canonical_a33).unsqueeze(0).to(device)
    action_mask_a33 = torch.from_numpy(game.legal_actions_mask(state_a33)).unsqueeze(0).float().to(device)
    policy_a33, value_a33 = model.predict(state_tensor_a33, action_mask_a33)
    a33_value = value_a33.cpu().item()
    print(f"  Network value: {a33_value:.4f} (from player {a33_player}'s perspective)")
    if a33_value > 0:
        print(f"  -> Player {a33_player} is WINNING")
    else:
        print(f"  -> Player {a33_player} is LOSING")

    # How this should propagate
    print("\n=== EXPECTED VALUE PROPAGATION ===")
    print(f"\nIf we backup from a33 to root:")
    print(f"  a33.mu initialized from network: {a33_value:.4f} (from player {a33_player}'s perspective)")

    # When a3 aggregates a33
    # a33.mu should be from a33's perspective (player {a33_player})
    # a3 aggregates: -a33.mu = from a3's perspective (player {a3_player})
    a3_sees_a33 = -a33_value  # assuming mu = value for simplicity
    print(f"\n  a3 aggregates a33:")
    print(f"    -a33.mu = {a3_sees_a33:.4f} (from player {a3_player}'s perspective)")
    print(f"    a3.mu = {a3_sees_a33:.4f} (from player {a3_player}'s perspective)")

    # When root aggregates a3
    root_sees_a3 = -a3_sees_a33
    print(f"\n  root aggregates a3:")
    print(f"    -a3.mu = {root_sees_a3:.4f} (from player {root_player}'s perspective)")

    if root_sees_a3 > 0:
        print(f"    -> Root thinks action 3 (leading to a33) is GOOD")
    else:
        print(f"    -> Root thinks action 3 (leading to a33) is BAD")

    print("\n=== CHECKING: Does a33_value sign match root's perspective? ===")
    if a33_player == root_player:
        print(f"a33 player ({a33_player}) == root player ({root_player})")
        print(f"So a33_value of {a33_value:.4f} IS from root's perspective!")
        print(f"After double negation: root sees {-(-a33_value):.4f} = {a33_value:.4f}")
        print(f"This is CORRECT: grandchild's value should match root's perspective after double negation.")
    else:
        print(f"a33 player ({a33_player}) != root player ({root_player})")
        print(f"So a33_value of {a33_value:.4f} is from opponent's perspective")
        print(f"After double negation: root sees {-(-a33_value):.4f} = {a33_value:.4f}")
        print(f"This is WRONG: grandchild's value should be negated for root!")


if __name__ == '__main__':
    main()
