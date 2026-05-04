"""
scripts/calibrate_bayesian.py - Bayesian MCTS confidence calibration on TicTacToe.

Uses exact minimax values to compare Bayesian root diagnostics against whether
the recommended action is truly optimal.
"""
import argparse
import os
import sys
from functools import lru_cache

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanozero.common import get_device, load_checkpoint, set_seed
from nanozero.config import BayesianMCTSConfig, get_model_config
from nanozero.game import get_game
from nanozero.mcts import BayesianMCTS
from nanozero.model import AlphaZeroTransformer


class UniformModel(torch.nn.Module):
    """Uniform policy, zero value baseline."""

    def __init__(self, action_size: int):
        super().__init__()
        self.action_size = action_size
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def predict(self, x, action_mask=None):
        mask = action_mask.float()
        probs = mask / mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        values = torch.zeros((x.shape[0], 1), device=x.device)
        return probs, values


def minimax_values(game):
    """Return an exact minimax value function for TicTacToe."""

    @lru_cache(maxsize=None)
    def value(flat_state):
        state = np.array(flat_state, dtype=np.int8).reshape(3, 3)
        if game.is_terminal(state):
            return game.terminal_reward(state)
        return max(-value(tuple(game.next_state(state, a).flatten())) for a in game.legal_actions(state))

    return value


def random_positions(game, n_positions: int, max_moves: int):
    positions = []
    seen = set()
    while len(positions) < n_positions:
        state = game.initial_state()
        for _ in range(np.random.randint(0, max_moves + 1)):
            if game.is_terminal(state):
                break
            state = game.next_state(state, int(np.random.choice(game.legal_actions(state))))
        key = tuple(state.flatten())
        if not game.is_terminal(state) and key not in seen:
            seen.add(key)
            positions.append(state)
    return positions


def bucket_index(confidence: float, n_buckets: int) -> int:
    return min(n_buckets - 1, max(0, int(confidence * n_buckets)))


def main():
    parser = argparse.ArgumentParser(description="Calibrate Bayesian MCTS on solved TicTacToe")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional model checkpoint; defaults to uniform model")
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--positions", type=int, default=200)
    parser.add_argument("--max_moves", type=int, default=7)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--buckets", type=int, default=10)
    parser.add_argument("--sigma_0", type=float, default=1.0)
    parser.add_argument("--obs_var", type=float, default=0.5)
    parser.add_argument("--ids_alpha", type=float, default=0.5)
    parser.add_argument("--ids_allocation", type=str, default="precision",
                        choices=["precision", "visits"])
    parser.add_argument("--final_policy", type=str, default="optimality",
                        choices=["optimality", "consensus"])
    parser.add_argument("--confidence_threshold", type=float, default=0.99)
    parser.add_argument("--epsilon_tie", type=float, default=0.02)
    parser.add_argument("--tie_sigma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    game = get_game("tictactoe")

    if args.checkpoint:
        model_config = get_model_config(game.config, n_layer=args.n_layer)
        model = AlphaZeroTransformer(model_config).to(device)
        load_checkpoint(args.checkpoint, model)
    else:
        model = UniformModel(game.config.action_size).to(device)
    model.eval()

    mcts = BayesianMCTS(
        game,
        BayesianMCTSConfig(
            num_simulations=args.simulations,
            sigma_0=args.sigma_0,
            obs_var=args.obs_var,
            ids_alpha=args.ids_alpha,
            ids_allocation=args.ids_allocation,
            final_policy=args.final_policy,
            confidence_threshold=args.confidence_threshold,
            epsilon_tie=args.epsilon_tie,
            tie_sigma=args.tie_sigma,
        ),
        leaves_per_batch=1,
        seed=args.seed,
        use_transposition_table=False,
    )

    exact_value = minimax_values(game)
    buckets = [{"n": 0, "correct": 0, "conf": 0.0} for _ in range(args.buckets)]
    stop_reasons = {}
    sims_used = []

    for state in random_positions(game, args.positions, args.max_moves):
        policy = mcts.search(state[np.newaxis, ...], model)[0]
        stats = mcts.search_stats()[0]
        action = int(np.argmax(policy))

        action_values = {
            a: -exact_value(tuple(game.next_state(state, a).flatten()))
            for a in game.legal_actions(state)
        }
        best_value = max(action_values.values())
        correct = action_values[action] == best_value
        confidence = float(stats["consensus_score"])

        b = buckets[bucket_index(confidence, args.buckets)]
        b["n"] += 1
        b["correct"] += int(correct)
        b["conf"] += confidence
        stop_reasons[stats["stop_reason"]] = stop_reasons.get(stats["stop_reason"], 0) + 1
        sims_used.append(stats["simulations_used"])

    ece = 0.0
    total = sum(b["n"] for b in buckets)
    print("bucket,count,avg_conf,accuracy")
    for i, b in enumerate(buckets):
        if b["n"] == 0:
            print(f"{i},0,na,na")
            continue
        avg_conf = b["conf"] / b["n"]
        acc = b["correct"] / b["n"]
        ece += (b["n"] / total) * abs(avg_conf - acc)
        print(f"{i},{b['n']},{avg_conf:.4f},{acc:.4f}")

    print(f"ece,{ece:.4f}")
    print(f"avg_sims,{np.mean(sims_used):.2f}")
    print("stops," + ",".join(f"{k}:{v}" for k, v in sorted(stop_reasons.items())))


if __name__ == "__main__":
    main()
