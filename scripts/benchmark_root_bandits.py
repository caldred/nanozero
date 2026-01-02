"""
scripts/benchmark_root_bandits.py - Root-level MCTS as a bandit problem

Tests selection algorithms using NN-derived priors and values.
This bridges pure synthetic bandits and full MCTS tree search.

The setup:
1. Expand root using NN -> get policy priors P(a) and value V(s)
2. For each "pull", select an action, get child state, evaluate with NN
3. The reward is the NN's value estimate V(s') (negated for opponent)
4. Compare how quickly algorithms identify the best action

This tests selection with:
- Realistic priors from the policy network
- Correlated, non-stationary rewards (NN values of game states)
- The actual action space of a game

Usage:
    python -m scripts.benchmark_root_bandits --game connect4 --checkpoint model.pt
"""
import argparse
import math
import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from nanozero.game import get_game, Game
from nanozero.model import AlphaZeroTransformer
from nanozero.config import get_model_config
from nanozero.common import get_device, load_checkpoint


@dataclass
class RootBanditResult:
    """Results from a single root-level bandit experiment."""
    position_name: str
    n_pulls: int
    algorithm: str
    recommended_action: int
    prob_correct: float  # Over multiple trials
    simple_regret: float  # V*(best) - V*(recommended)
    pulls_to_confidence: Optional[float]


class UCBRootBandit:
    """
    UCB selection at MCTS root level.

    Mimics PUCT but without tree expansion - just root-level selection.
    """

    def __init__(self, n_actions: int, priors: np.ndarray, legal_mask: np.ndarray, c: float = 1.0):
        self.n_actions = n_actions
        self.priors = priors  # Policy network priors P(a|s)
        self.legal_mask = legal_mask  # Boolean mask of legal actions
        self.c = c
        self.counts = np.zeros(n_actions)
        self.values = np.zeros(n_actions)
        self.total_pulls = 0

    def reset(self, priors: np.ndarray, legal_mask: np.ndarray):
        """Reset with new priors."""
        self.priors = priors
        self.legal_mask = legal_mask
        self.counts = np.zeros(self.n_actions)
        self.values = np.zeros(self.n_actions)
        self.total_pulls = 0

    def select(self) -> int:
        """Select action using PUCT formula."""
        legal_actions = np.where(self.legal_mask)[0]

        # PUCT: Q + c * P * sqrt(N_total) / (1 + N_action)
        if self.total_pulls == 0:
            # First pull: sample from prior over legal actions
            legal_priors = self.priors[legal_actions]
            legal_priors = legal_priors / legal_priors.sum()
            return int(np.random.choice(legal_actions, p=legal_priors))

        exploration = self.c * self.priors * math.sqrt(self.total_pulls) / (1 + self.counts)
        ucb_values = self.values + exploration

        # Mask illegal actions
        ucb_values[~self.legal_mask] = -np.inf

        return int(np.argmax(ucb_values))

    def update(self, action: int, reward: float):
        """Update with observed reward."""
        self.counts[action] += 1
        self.total_pulls += 1
        # Incremental mean
        self.values[action] += (reward - self.values[action]) / self.counts[action]

    def recommend(self) -> int:
        """Recommend best action by visit count."""
        return int(np.argmax(self.counts))

    def get_policy(self) -> np.ndarray:
        """Get policy from visit counts."""
        if self.counts.sum() == 0:
            return self.priors.copy()
        return self.counts / self.counts.sum()


class TTTSRootBandit:
    """
    TTTS-IDS selection at MCTS root level.

    Uses logit-shifted prior initialization and Bayesian updates.
    """

    def __init__(
        self,
        n_actions: int,
        priors: np.ndarray,
        root_value: float,
        legal_mask: np.ndarray,
        sigma_0: float = 1.0,
        obs_var: float = 0.5,
        alpha: float = 0.5,
        min_var: float = 1e-6
    ):
        self.n_actions = n_actions
        self.sigma_0 = sigma_0
        self.obs_var = obs_var
        self.alpha = alpha
        self.min_var = min_var

        # Initialize with logit-shifted priors
        self._init_beliefs(priors, root_value, legal_mask)

    def _init_beliefs(self, priors: np.ndarray, root_value: float, legal_mask: np.ndarray):
        """Initialize Gaussian beliefs using logit-shifted priors."""
        self.priors = priors
        self.legal_mask = legal_mask

        # Compute entropy of policy
        eps = 1e-8
        legal_probs = priors[self.legal_mask]
        legal_probs = legal_probs / (legal_probs.sum() + eps)
        log_probs = np.log(legal_probs + eps)
        entropy = -np.sum(legal_probs * log_probs)

        # Logit-shift scale
        scale = self.sigma_0 * (math.sqrt(6) / math.pi)

        # Initialize means and variances
        self.mu = np.zeros(self.n_actions)
        self.sigma_sq = np.full(self.n_actions, self.sigma_0 ** 2)

        legal_idx = 0
        for a in range(self.n_actions):
            if self.legal_mask[a]:
                self.mu[a] = root_value + scale * (log_probs[legal_idx] + entropy)
                legal_idx += 1
            else:
                self.mu[a] = -np.inf  # Illegal action

        self.total_pulls = 0

    def reset(self, priors: np.ndarray, root_value: float, legal_mask: np.ndarray):
        """Reset with new priors."""
        self._init_beliefs(priors, root_value, legal_mask)

    def select(self) -> int:
        """Select action using Top-Two Thompson Sampling with IDS."""
        legal_actions = np.where(self.legal_mask)[0]

        if len(legal_actions) == 1:
            return int(legal_actions[0])

        # Draw Thompson samples
        samples = np.random.normal(self.mu, np.sqrt(self.sigma_sq))
        samples[~self.legal_mask] = -np.inf

        # Find leader and challenger
        sorted_idx = np.argsort(samples)[::-1]
        leader = sorted_idx[0]
        challenger = sorted_idx[1]

        # IDS allocation
        precision_leader = 1.0 / max(self.sigma_sq[leader], self.min_var)
        precision_challenger = 1.0 / max(self.sigma_sq[challenger], self.min_var)

        beta = (precision_leader + self.alpha) / (
            precision_leader + precision_challenger + 2 * self.alpha
        )

        if np.random.random() < beta:
            return int(challenger)
        else:
            return int(leader)

    def update(self, action: int, reward: float):
        """Bayesian update with observed reward."""
        self.total_pulls += 1

        precision_prior = 1.0 / max(self.sigma_sq[action], self.min_var)
        precision_obs = 1.0 / max(self.obs_var, self.min_var)
        new_precision = precision_prior + precision_obs

        self.mu[action] = (
            precision_prior * self.mu[action] + precision_obs * reward
        ) / new_precision
        self.sigma_sq[action] = max(1.0 / new_precision, self.min_var)

    def recommend(self, n_samples: int = 100) -> int:
        """Recommend by probability of optimality."""
        wins = np.zeros(self.n_actions)

        for _ in range(n_samples):
            samples = np.random.normal(self.mu, np.sqrt(self.sigma_sq))
            samples[~self.legal_mask] = -np.inf
            winner = np.argmax(samples)
            wins[winner] += 1

        return int(np.argmax(wins))

    def get_policy(self, n_samples: int = 100) -> np.ndarray:
        """Get policy from probability of optimality."""
        wins = np.zeros(self.n_actions)

        for _ in range(n_samples):
            samples = np.random.normal(self.mu, np.sqrt(self.sigma_sq))
            samples[~self.legal_mask] = -np.inf
            winner = np.argmax(samples)
            wins[winner] += 1

        if wins.sum() > 0:
            return wins / wins.sum()
        return self.priors.copy()


class RootBanditEnv:
    """
    Environment that simulates MCTS root-level bandit.

    Each "pull" of an action:
    1. Takes the action to get child state
    2. Evaluates child state with NN
    3. Returns negated value (opponent's perspective)
    """

    def __init__(self, game: Game, model: torch.nn.Module, device: torch.device):
        self.game = game
        self.model = model
        self.device = device

    def setup(self, state: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """
        Set up bandit for a given state.

        Returns:
            priors: Policy network output P(a|s)
            root_value: Value network output V(s)
            true_values: Ground truth V(s') for each action (for evaluation)
            legal_mask: Boolean mask of legal actions
        """
        self.state = state
        self.legal_actions = self.game.legal_actions(state)
        self.legal_mask = self.game.legal_actions_mask(state).astype(bool)

        # Get root policy and value
        canonical = self.game.canonical_state(state)
        state_tensor = self.game.to_tensor(canonical).unsqueeze(0).to(self.device)
        action_mask = torch.from_numpy(self.legal_mask).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            policy, value = self.model.predict(state_tensor, action_mask)

        priors = policy.cpu().numpy()[0]
        root_value = value.cpu().item()

        # Compute true values for each action (for evaluation)
        true_values = np.full(self.game.config.action_size, -np.inf)
        for action in self.legal_actions:
            child_state = self.game.next_state(state, action)
            if self.game.is_terminal(child_state):
                # Terminal: use actual reward
                true_values[action] = -self.game.terminal_reward(child_state)
            else:
                # Non-terminal: use NN value
                child_canonical = self.game.canonical_state(child_state)
                child_tensor = self.game.to_tensor(child_canonical).unsqueeze(0).to(self.device)
                child_mask = torch.from_numpy(
                    self.game.legal_actions_mask(child_state)
                ).unsqueeze(0).float().to(self.device)

                with torch.no_grad():
                    _, child_value = self.model.predict(child_tensor, child_mask)
                true_values[action] = -child_value.cpu().item()

        self.true_values = true_values
        self.best_action = int(np.argmax(true_values))

        return priors, root_value, true_values, self.legal_mask

    def pull(self, action: int) -> float:
        """
        Pull an arm (take action) and get reward.

        Returns negated child value (from parent's perspective).
        """
        child_state = self.game.next_state(self.state, action)

        if self.game.is_terminal(child_state):
            return -self.game.terminal_reward(child_state)

        # Evaluate child with NN
        child_canonical = self.game.canonical_state(child_state)
        child_tensor = self.game.to_tensor(child_canonical).unsqueeze(0).to(self.device)
        child_mask = torch.from_numpy(
            self.game.legal_actions_mask(child_state)
        ).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            _, child_value = self.model.predict(child_tensor, child_mask)

        # Add small noise to simulate value uncertainty
        noise = np.random.normal(0, 0.1)
        return -child_value.cpu().item() + noise


def evaluate_root_bandit(
    env: RootBanditEnv,
    state: np.ndarray,
    n_pulls: int,
    n_trials: int = 50,
) -> Dict[str, RootBanditResult]:
    """
    Evaluate both algorithms on a root-level bandit problem.
    """
    results = {}

    for algo_name in ['UCB', 'TTTS-IDS']:
        correct_count = 0
        simple_regrets = []

        for trial in range(n_trials):
            # Setup environment
            priors, root_value, true_values, legal_mask = env.setup(state)
            best_action = env.best_action
            best_value = true_values[best_action]

            # Create algorithm
            if algo_name == 'UCB':
                algo = UCBRootBandit(env.game.config.action_size, priors, legal_mask)
            else:
                algo = TTTSRootBandit(
                    env.game.config.action_size, priors, root_value, legal_mask
                )

            # Run pulls
            for _ in range(n_pulls):
                action = algo.select()
                reward = env.pull(action)
                algo.update(action, reward)

            # Get recommendation
            recommended = algo.recommend()

            if recommended == best_action:
                correct_count += 1

            regret = best_value - true_values[recommended]
            simple_regrets.append(max(0, regret))  # Clamp to non-negative

        results[algo_name] = RootBanditResult(
            position_name="",
            n_pulls=n_pulls,
            algorithm=algo_name,
            recommended_action=-1,
            prob_correct=correct_count / n_trials,
            simple_regret=float(np.mean(simple_regrets)),
            pulls_to_confidence=None,
        )

    return results


def get_test_positions(game: Game, n_positions: int = 10) -> List[np.ndarray]:
    """Generate test positions by random play."""
    positions = []

    # Always include empty board
    positions.append(game.initial_state())

    # Generate random positions
    while len(positions) < n_positions:
        state = game.initial_state()
        n_moves = np.random.randint(1, 20)

        for _ in range(n_moves):
            if game.is_terminal(state):
                break
            legal = game.legal_actions(state)
            action = np.random.choice(legal)
            state = game.next_state(state, action)

        if not game.is_terminal(state):
            positions.append(state)

    return positions


def main():
    parser = argparse.ArgumentParser(description='Root-level MCTS bandit comparison')
    parser.add_argument('--game', type=str, required=True,
                        choices=['tictactoe', 'connect4', 'go9x9'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--n_pulls', type=int, nargs='+', default=[25, 50, 100],
                        help='Number of pulls to test')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Trials per position')
    parser.add_argument('--n_positions', type=int, default=10,
                        help='Number of positions to test')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = get_device() if args.device == 'auto' else torch.device(args.device)

    # Load game and model
    game = get_game(args.game)
    model_config = get_model_config(game.config, n_layer=args.n_layer)
    model = AlphaZeroTransformer(model_config).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    print(f"Root-Level MCTS Bandit Comparison: {args.game}")
    print(f"Model: {args.checkpoint}")
    print(f"Pulls: {args.n_pulls}, Trials: {args.n_trials}, Positions: {args.n_positions}")
    print()

    # Create environment
    env = RootBanditEnv(game, model, device)

    # Get test positions
    positions = get_test_positions(game, args.n_positions)

    # Aggregate results
    all_results = {n: {'UCB': [], 'TTTS-IDS': []} for n in args.n_pulls}

    for i, state in enumerate(positions):
        print(f"Position {i+1}/{len(positions)}...", end=" ", flush=True)

        for n_pulls in args.n_pulls:
            results = evaluate_root_bandit(env, state, n_pulls, args.n_trials)
            for algo_name, result in results.items():
                all_results[n_pulls][algo_name].append(result)

        print("done")

    # Print summary
    print("\n" + "=" * 70)
    print("              Root-Level MCTS Bandit Summary")
    print("=" * 70)
    print(f"\n{'Pulls':<10} {'Algorithm':<12} {'P(correct)':<15} {'Simple Regret':<15}")
    print("-" * 70)

    for n_pulls in args.n_pulls:
        for algo_name in ['UCB', 'TTTS-IDS']:
            results = all_results[n_pulls][algo_name]
            avg_correct = np.mean([r.prob_correct for r in results])
            avg_regret = np.mean([r.simple_regret for r in results])
            std_regret = np.std([r.simple_regret for r in results])

            print(f"{n_pulls:<10} {algo_name:<12} {avg_correct:<15.2%} {avg_regret:.4f} +/- {std_regret:.4f}")

    print("=" * 70)


if __name__ == '__main__':
    main()
