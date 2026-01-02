"""
scripts/train_bayesian.py - Quick test training with BayesianMCTS

Verifies the BayesianMCTS implementation works end-to-end for training.

Usage:
    python -m scripts.train_bayesian
"""
import numpy as np
import torch
import torch.nn.functional as F

from nanozero.game import get_game
from nanozero.model import AlphaZeroTransformer
from nanozero.bayesian_mcts import BayesianMCTS, BayesianMCTSConfig
from nanozero.mcts import sample_action
from nanozero.replay import ReplayBuffer
from nanozero.config import get_model_config
from nanozero.common import get_device, set_seed, print0


@torch.inference_mode()
def self_play_games_bayesian(game, model, mcts, num_games, temperature_threshold=10, parallel_games=16):
    """
    Play multiple games of self-play using BayesianMCTS.

    Uses the new interleaved batching for GPU efficiency.
    """
    model.eval()
    all_examples = []
    games_completed = 0

    # Initialize parallel game states
    n_parallel = min(parallel_games, num_games)
    states = [game.initial_state() for _ in range(n_parallel)]
    move_counts = [0] * n_parallel
    game_examples = [[] for _ in range(n_parallel)]

    while games_completed < num_games:
        # Find active (non-terminal) games
        active_indices = [i for i, s in enumerate(states) if not game.is_terminal(s)]

        if not active_indices:
            break

        # Batch all active states for MCTS
        active_states = np.stack([states[i] for i in active_indices])

        # Use BayesianMCTS search (now with interleaved batching!)
        policies = mcts.search(active_states, model)

        # Process each active game
        for idx, game_idx in enumerate(active_indices):
            state = states[game_idx]
            policy = policies[idx]
            player = game.current_player(state)
            move_count = move_counts[game_idx]

            # Store example
            canonical = game.canonical_state(state)
            game_examples[game_idx].append((canonical.copy(), policy.copy(), player))

            # Sample action with temperature
            temperature = 1.0 if move_count < temperature_threshold else 0.0
            action = sample_action(policy, temperature=temperature)

            # Apply action
            states[game_idx] = game.next_state(state, action)
            move_counts[game_idx] += 1

        # Check for finished games
        for i in range(n_parallel):
            if game.is_terminal(states[i]) and game_examples[i]:
                reward = game.terminal_reward(states[i])
                final_player = game.current_player(states[i])

                for canonical, policy, player in game_examples[i]:
                    if player == final_player:
                        value = reward
                    else:
                        value = -reward

                    # Add symmetries for data augmentation
                    for sym_state, sym_policy in game.symmetries(canonical, policy):
                        all_examples.append((sym_state, sym_policy, value))

                games_completed += 1

                # Start a new game if we need more
                if games_completed < num_games:
                    states[i] = game.initial_state()
                    move_counts[i] = 0
                    game_examples[i] = []

    return all_examples


def train_step(model, optimizer, states, policies, values, action_masks, device):
    """Simple training step without mixed precision for simplicity."""
    model.train()

    states = states.to(device)
    policies = policies.to(device)
    values = values.to(device)
    action_masks = action_masks.to(device)

    optimizer.zero_grad()

    pred_log_policies, pred_values = model(states, action_masks)

    # Policy loss: cross-entropy
    policy_loss = -torch.mean(torch.sum(policies * pred_log_policies, dim=1))

    # Value loss: MSE
    value_loss = F.mse_loss(pred_values.squeeze(-1), values)

    loss = policy_loss + value_loss
    loss.backward()
    optimizer.step()

    return loss.item(), policy_loss.item(), value_loss.item()


@torch.inference_mode()
def evaluate_vs_random(game, model, mcts, num_games=20):
    """Quick evaluation against random player."""
    model.eval()
    wins = 0

    for i in range(num_games):
        state = game.initial_state()
        model_player = 1 if i % 2 == 0 else -1

        while not game.is_terminal(state):
            current = game.current_player(state)

            if current == model_player:
                policy = mcts.search(state[np.newaxis, ...], model, num_simulations=25)[0]
                action = sample_action(policy, temperature=0)
            else:
                legal = game.legal_actions(state)
                action = np.random.choice(legal)

            state = game.next_state(state, action)

        reward = game.terminal_reward(state)
        final_player = game.current_player(state)

        if final_player == model_player:
            model_result = reward
        else:
            model_result = -reward

        if model_result > 0:
            wins += 1

    return wins / num_games


def main():
    set_seed(42)
    device = get_device()
    print0(f"Using device: {device}")

    # Create game
    game = get_game('tictactoe')
    print0(f"Game: tictactoe (backend: {game.backend})")

    # Create small model for quick testing
    model_config = get_model_config(game.config, n_layer=2)
    model = AlphaZeroTransformer(model_config).to(device)
    print0(f"Model: {model.count_parameters():,} parameters")

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Create BayesianMCTS with reasonable settings for quick test
    mcts_config = BayesianMCTSConfig(
        num_simulations=50,
        early_stopping=True,
        confidence_threshold=0.9,
    )
    mcts = BayesianMCTS(game, mcts_config)
    print0(f"Using BayesianMCTS with {mcts_config.num_simulations} simulations")

    # Create replay buffer
    buffer = ReplayBuffer(10000)

    # Quick training loop
    num_iterations = 5
    games_per_iteration = 20
    training_steps = 50
    batch_size = 32

    print0(f"\nQuick test: {num_iterations} iterations, {games_per_iteration} games each\n")

    for iteration in range(num_iterations):
        print0(f"Iteration {iteration + 1}/{num_iterations}")

        # Self-play with BayesianMCTS
        print0("  Self-play...")
        examples = self_play_games_bayesian(
            game, model, mcts,
            num_games=games_per_iteration,
            parallel_games=8
        )

        # Add to buffer
        for state, policy, value in examples:
            buffer.push(state, policy, value)

        print0(f"  Collected {len(examples)} examples, buffer size: {len(buffer)}")

        # Training
        if len(buffer) >= batch_size:
            print0("  Training...")
            total_loss = 0

            for step in range(training_steps):
                states, policies, values = buffer.sample(batch_size)

                state_tensors = torch.stack([game.to_tensor(s) for s in states])
                policy_tensors = torch.from_numpy(policies).float()
                value_tensors = torch.from_numpy(values).float()
                action_mask_tensors = torch.stack([
                    torch.from_numpy(game.legal_actions_mask(s))
                    for s in states
                ]).float()

                loss, _, _ = train_step(
                    model, optimizer,
                    state_tensors, policy_tensors, value_tensors,
                    action_mask_tensors, device
                )
                total_loss += loss

            print0(f"  Avg loss: {total_loss / training_steps:.4f}")
            mcts.clear_cache()

        # Evaluation
        print0("  Evaluating vs random...")
        win_rate = evaluate_vs_random(game, model, mcts, num_games=20)
        print0(f"  Win rate vs random: {win_rate:.1%}")
        print0("")

    # Final evaluation
    print0("Final evaluation (more games):")
    win_rate = evaluate_vs_random(game, model, mcts, num_games=50)
    print0(f"  Win rate vs random: {win_rate:.1%}")

    print0("\nBayesianMCTS training test complete!")


if __name__ == '__main__':
    main()
