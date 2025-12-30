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


@torch.inference_mode()
def self_play_games(game, model, mcts, num_games, temperature_threshold=15, parallel_games=64):
    """
    Play multiple games of self-play in parallel for GPU efficiency.

    Runs multiple games simultaneously, batching MCTS neural network calls
    across all active games for much better GPU utilization.

    Args:
        game: Game instance
        model: Neural network
        mcts: MCTS instance
        num_games: Total number of games to play
        temperature_threshold: Move number after which temperature becomes 0
        parallel_games: Number of games to run in parallel

    Returns:
        List of all (canonical_state, policy, value) tuples from all games
    """
    model.eval()
    all_examples = []
    games_completed = 0

    # Initialize parallel game states
    n_parallel = min(parallel_games, num_games)
    states = [game.initial_state() for _ in range(n_parallel)]
    move_counts = [0] * n_parallel
    game_examples = [[] for _ in range(n_parallel)]  # (canonical, policy, player) per game

    while games_completed < num_games:
        # Find active (non-terminal) games
        active_indices = [i for i, s in enumerate(states) if not game.is_terminal(s)]

        if not active_indices:
            # All current games finished, shouldn't happen but handle it
            break

        # Batch all active states for MCTS
        add_noise = [move_counts[i] == 0 for i in active_indices]

        # Apply Dirichlet noise only to roots at move 0 (avoid polluting all games)
        policies = np.zeros((len(active_indices), game.config.action_size), dtype=np.float32)
        active_pos = {game_idx: pos for pos, game_idx in enumerate(active_indices)}
        noise_indices = [i for i in active_indices if move_counts[i] == 0]
        no_noise_indices = [i for i in active_indices if move_counts[i] != 0]

        if noise_indices:
            noise_states = np.stack([states[i] for i in noise_indices])
            noise_policies = mcts.search(noise_states, model, add_noise=True)
            for local_idx, game_idx in enumerate(noise_indices):
                policies[active_pos[game_idx]] = noise_policies[local_idx]

        if no_noise_indices:
            no_noise_states = np.stack([states[i] for i in no_noise_indices])
            no_noise_policies = mcts.search(no_noise_states, model, add_noise=False)
            for local_idx, game_idx in enumerate(no_noise_indices):
                policies[active_pos[game_idx]] = no_noise_policies[local_idx]

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

        # Check for finished games and collect examples
        for i in range(n_parallel):
            if game.is_terminal(states[i]) and game_examples[i]:
                # Game finished - assign values and collect examples
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

                if games_completed % 10 == 0:
                    print0(f"  Self-play: {games_completed}/{num_games} games")

    return all_examples


def train_step(model, optimizer, scaler, states, policies, values, action_masks, device, use_amp):
    """
    Perform a single training step with optional mixed precision.

    Args:
        model: Neural network
        optimizer: Optimizer
        scaler: GradScaler for mixed precision (can be disabled)
        states: Batch of state tensors (B, board_size)
        policies: Batch of policy targets (B, action_size)
        values: Batch of value targets (B,)
        action_masks: Batch of legal action masks (B, action_size)
        device: Torch device
        use_amp: Whether to use automatic mixed precision

    Returns:
        Tuple of (total_loss, policy_loss, value_loss)
    """
    model.train()

    states = states.to(device)
    policies = policies.to(device)
    values = values.to(device)
    action_masks = action_masks.to(device)

    optimizer.zero_grad()

    # Forward pass with optional mixed precision
    with torch.amp.autocast('cuda', enabled=use_amp):
        pred_log_policies, pred_values = model(states, action_masks)

        # Policy loss: cross-entropy (using log-softmax output)
        policy_loss = -torch.mean(torch.sum(policies * pred_log_policies, dim=1))

        # Value loss: MSE
        value_loss = F.mse_loss(pred_values.squeeze(-1), values)

        # Total loss
        loss = policy_loss + value_loss

    # Backward pass with gradient scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return loss.item(), policy_loss.item(), value_loss.item()


@torch.inference_mode()
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
                # Model's turn - pass raw state, BatchedMCTS canonicalizes internally
                policy = mcts.search(
                    state[np.newaxis, ...],
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
                        help='Game to train on (tictactoe, connect4, go9x9, go19x19)')

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
    parser.add_argument('--parallel_games', type=int, default=64,
                        help='Number of games to run in parallel during self-play')
    parser.add_argument('--no_transposition_table', action='store_true',
                        help='Disable symmetry-aware transposition table in MCTS')

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

    # Mixed precision setup (only for CUDA)
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    if use_amp:
        print0("Using mixed precision training (FP16)")

    # Create MCTS
    mcts_config = MCTSConfig(num_simulations=args.mcts_simulations)
    use_tt = not args.no_transposition_table
    mcts = BatchedMCTS(game, mcts_config, use_transposition_table=use_tt)
    if not use_tt:
        print0("Transposition table disabled")

    # Create replay buffer
    buffer = ReplayBuffer(args.buffer_size)

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Resume from checkpoint if specified
    start_iteration = 0
    if args.resume:
        start_iteration = load_checkpoint(args.resume, model, optimizer, scaler=scaler)
        print0(f"Resumed from iteration {start_iteration}")

    print0(f"\nStarting training for {args.num_iterations} iterations")
    print0(f"  {args.games_per_iteration} games/iteration ({args.parallel_games} parallel)")
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
            temperature_threshold=args.temperature_threshold,
            parallel_games=args.parallel_games
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
                action_mask_tensors = torch.stack([
                    torch.from_numpy(game.legal_actions_mask(s))
                    for s in states
                ]).float()

                # Train step
                loss, policy_loss, value_loss = train_step(
                    model, optimizer, scaler,
                    state_tensors, policy_tensors, value_tensors,
                    action_mask_tensors, device, use_amp
                )

                loss_meter.update(loss)
                policy_loss_meter.update(policy_loss)
                value_loss_meter.update(value_loss)

            print0(f"  Loss: {loss_meter.avg:.4f} (policy: {policy_loss_meter.avg:.4f}, value: {value_loss_meter.avg:.4f})")

            # Clear MCTS cache after training (model weights changed)
            mcts.clear_cache()

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
            save_checkpoint(model, optimizer, iteration + 1, ckpt_path, scaler=scaler)
            print0(f"  Saved checkpoint: {ckpt_path}")

        iter_time = time.time() - iter_start
        print0(f"  Iteration time: {iter_time:.1f}s")
        print0("")

    # Save final checkpoint
    final_path = os.path.join(args.checkpoint_dir, f"{args.game}_final.pt")
    save_checkpoint(model, optimizer, args.num_iterations, final_path, scaler=scaler)
    print0(f"Saved final checkpoint: {final_path}")

    # Final evaluation
    print0("\nFinal evaluation:")
    win_rate = evaluate_vs_random(game, model, mcts, num_games=100)
    print0(f"  Win rate vs random: {win_rate:.1%}")

    print0("\nTraining complete!")


if __name__ == '__main__':
    main()
