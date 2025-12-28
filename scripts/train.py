"""
scripts/train.py - Main training loop for NanoZero

Usage:
    python -m scripts.train --game=tictactoe --n_layer=2 --num_iterations=50
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from nanozero.game import get_game
from nanozero.model import AlphaZeroTransformer
from nanozero.mcts import BatchedMCTS, sample_action
from nanozero.replay import ReplayBuffer
from nanozero.config import get_game_config, get_model_config, MCTSConfig, TrainConfig
from nanozero.common import get_device, set_seed, print0, save_checkpoint, load_checkpoint, AverageMeter


def self_play_game(game, model, mcts, temperature_threshold=15):
    """
    Play a single game of self-play.

    Args:
        game: Game instance
        model: Neural network
        mcts: MCTS instance
        temperature_threshold: Move number after which temperature becomes 0

    Returns:
        List of (canonical_state, policy, current_player) tuples
    """
    state = game.initial_state()
    examples = []
    move_count = 0

    while not game.is_terminal(state):
        canonical = game.canonical_state(state)
        player = game.current_player(state)

        # Run MCTS to get policy
        temperature = 1.0 if move_count < temperature_threshold else 0.0
        policy = mcts.search(
            canonical[np.newaxis, ...],
            model,
            add_noise=(move_count == 0)  # Add noise only at root
        )[0]

        # Store example (without value yet - we'll add it at the end)
        examples.append((canonical.copy(), policy.copy(), player))

        # Sample action
        action = sample_action(policy, temperature=temperature)
        state = game.next_state(state, action)
        move_count += 1

    # Get terminal reward from perspective of final state
    reward = game.terminal_reward(state)
    final_player = game.current_player(state)

    # Assign values to all examples
    training_examples = []
    for canonical, policy, player in examples:
        # Value from this player's perspective (reward is from final player's view)
        if player == final_player:
            value = reward
        else:
            value = -reward
        training_examples.append((canonical, policy, value))

    return training_examples


def self_play_games(game, model, mcts, num_games, temperature_threshold=15):
    """
    Play multiple games of self-play.

    Args:
        game: Game instance
        model: Neural network
        mcts: MCTS instance
        num_games: Number of games to play
        temperature_threshold: Move number after which temperature becomes 0

    Returns:
        List of all (canonical_state, policy, value) tuples from all games
    """
    model.eval()
    all_examples = []

    for i in range(num_games):
        examples = self_play_game(game, model, mcts, temperature_threshold)

        # Add symmetries for data augmentation
        for state, policy, value in examples:
            symmetries = game.symmetries(state, policy)
            for sym_state, sym_policy in symmetries:
                all_examples.append((sym_state, sym_policy, value))

        if (i + 1) % 10 == 0:
            print0(f"  Self-play: {i+1}/{num_games} games")

    return all_examples


def train_step(model, optimizer, states, policies, values, device):
    """
    Perform a single training step.

    Args:
        model: Neural network
        optimizer: Optimizer
        states: Batch of state tensors (B, board_size)
        policies: Batch of policy targets (B, action_size)
        values: Batch of value targets (B,)
        device: Torch device

    Returns:
        Tuple of (total_loss, policy_loss, value_loss)
    """
    model.train()

    states = states.to(device)
    policies = policies.to(device)
    values = values.to(device)

    # Forward pass
    pred_log_policies, pred_values = model(states)

    # Policy loss: cross-entropy (using log-softmax output)
    policy_loss = -torch.mean(torch.sum(policies * pred_log_policies, dim=1))

    # Value loss: MSE
    value_loss = F.mse_loss(pred_values.squeeze(-1), values)

    # Total loss
    loss = policy_loss + value_loss

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), policy_loss.item(), value_loss.item()


def evaluate_vs_random(game, model, mcts, num_games=50, mcts_simulations=25):
    """
    Evaluate model against random player.

    Args:
        game: Game instance
        model: Neural network
        mcts: MCTS instance
        num_games: Number of games to play
        mcts_simulations: Number of MCTS simulations per move

    Returns:
        Win rate against random player
    """
    model.eval()
    wins = 0
    draws = 0

    for i in range(num_games):
        state = game.initial_state()

        # Alternate who goes first
        model_player = 1 if i % 2 == 0 else -1

        while not game.is_terminal(state):
            current = game.current_player(state)

            if current == model_player:
                # Model's turn
                canonical = game.canonical_state(state)
                policy = mcts.search(
                    canonical[np.newaxis, ...],
                    model,
                    num_simulations=mcts_simulations,
                    add_noise=False
                )[0]
                action = sample_action(policy, temperature=0)
            else:
                # Random player's turn
                legal = game.legal_actions(state)
                action = np.random.choice(legal)

            state = game.next_state(state, action)

        # Get result
        reward = game.terminal_reward(state)
        final_player = game.current_player(state)

        if final_player == model_player:
            model_result = reward
        else:
            model_result = -reward

        if model_result > 0:
            wins += 1
        elif model_result == 0:
            draws += 1

    return wins / num_games


def main():
    parser = argparse.ArgumentParser(description='Train NanoZero')

    # Game settings
    parser.add_argument('--game', type=str, default='tictactoe',
                        help='Game to train on (tictactoe, connect4)')

    # Model settings
    parser.add_argument('--n_layer', type=int, default=4,
                        help='Number of transformer layers')

    # Training settings
    parser.add_argument('--num_iterations', type=int, default=100,
                        help='Number of training iterations')
    parser.add_argument('--games_per_iteration', type=int, default=100,
                        help='Self-play games per iteration')
    parser.add_argument('--training_steps', type=int, default=100,
                        help='Training steps per iteration')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--buffer_size', type=int, default=100000,
                        help='Replay buffer size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')

    # MCTS settings
    parser.add_argument('--mcts_simulations', type=int, default=50,
                        help='MCTS simulations per move')
    parser.add_argument('--temperature_threshold', type=int, default=15,
                        help='Move number after which temperature is 0')

    # Checkpointing and evaluation
    parser.add_argument('--checkpoint_interval', type=int, default=10,
                        help='Save checkpoint every N iterations')
    parser.add_argument('--eval_interval', type=int, default=10,
                        help='Evaluate every N iterations')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory for checkpoints')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, cuda, mps)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Get device
    device = get_device() if args.device == 'auto' else torch.device(args.device)
    print0(f"Using device: {device}")

    # Create game
    game = get_game(args.game)
    print0(f"Game: {args.game}")
    print0(f"Board size: {game.config.board_size}, Action size: {game.config.action_size}")

    # Create model
    model_config = get_model_config(game.config, n_layer=args.n_layer)
    model = AlphaZeroTransformer(model_config).to(device)
    print0(f"Model: {model.count_parameters():,} parameters")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Create MCTS
    mcts_config = MCTSConfig(num_simulations=args.mcts_simulations)
    mcts = BatchedMCTS(game, mcts_config)

    # Create replay buffer
    buffer = ReplayBuffer(args.buffer_size)

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Resume from checkpoint if specified
    start_iteration = 0
    if args.resume:
        start_iteration = load_checkpoint(args.resume, model, optimizer)
        print0(f"Resumed from iteration {start_iteration}")

    print0(f"\nStarting training for {args.num_iterations} iterations")
    print0(f"  {args.games_per_iteration} games/iteration")
    print0(f"  {args.training_steps} training steps/iteration")
    print0(f"  {args.mcts_simulations} MCTS simulations/move")
    print0("")

    # Training loop
    for iteration in range(start_iteration, args.num_iterations):
        iter_start = time.time()

        print0(f"Iteration {iteration + 1}/{args.num_iterations}")

        # Self-play
        print0("  Self-play...")
        examples = self_play_games(
            game, model, mcts,
            num_games=args.games_per_iteration,
            temperature_threshold=args.temperature_threshold
        )

        # Add to buffer
        for state, policy, value in examples:
            buffer.push(state, policy, value)

        print0(f"  Collected {len(examples)} examples, buffer size: {len(buffer)}")

        # Training
        if len(buffer) >= args.batch_size:
            print0("  Training...")
            loss_meter = AverageMeter()
            policy_loss_meter = AverageMeter()
            value_loss_meter = AverageMeter()

            for step in range(args.training_steps):
                # Sample batch
                states, policies, values = buffer.sample(args.batch_size)

                # Convert to tensors
                state_tensors = torch.stack([
                    game.to_tensor(s) for s in states
                ])
                policy_tensors = torch.from_numpy(policies).float()
                value_tensors = torch.from_numpy(values).float()

                # Train step
                loss, policy_loss, value_loss = train_step(
                    model, optimizer,
                    state_tensors, policy_tensors, value_tensors,
                    device
                )

                loss_meter.update(loss)
                policy_loss_meter.update(policy_loss)
                value_loss_meter.update(value_loss)

            print0(f"  Loss: {loss_meter.avg:.4f} (policy: {policy_loss_meter.avg:.4f}, value: {value_loss_meter.avg:.4f})")

        # Evaluation
        if (iteration + 1) % args.eval_interval == 0:
            print0("  Evaluating vs random...")
            win_rate = evaluate_vs_random(game, model, mcts, num_games=50)
            print0(f"  Win rate vs random: {win_rate:.1%}")

        # Checkpoint
        if (iteration + 1) % args.checkpoint_interval == 0:
            ckpt_path = os.path.join(
                args.checkpoint_dir,
                f"{args.game}_iter{iteration + 1}.pt"
            )
            save_checkpoint(model, optimizer, iteration + 1, ckpt_path)
            print0(f"  Saved checkpoint: {ckpt_path}")

        iter_time = time.time() - iter_start
        print0(f"  Iteration time: {iter_time:.1f}s")
        print0("")

    # Save final checkpoint
    final_path = os.path.join(args.checkpoint_dir, f"{args.game}_final.pt")
    save_checkpoint(model, optimizer, args.num_iterations, final_path)
    print0(f"Saved final checkpoint: {final_path}")

    # Final evaluation
    print0("\nFinal evaluation:")
    win_rate = evaluate_vs_random(game, model, mcts, num_games=100)
    print0(f"  Win rate vs random: {win_rate:.1%}")

    print0("\nTraining complete!")


if __name__ == '__main__':
    main()
