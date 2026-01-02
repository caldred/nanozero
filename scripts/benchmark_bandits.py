"""
scripts/benchmark_bandits.py - Pure Bandit BAI Comparison

Compare UCB (PUCT-style) vs TTTS-IDS (Bayesian) selection algorithms
in synthetic multi-armed bandit settings.

Usage:
    python -m scripts.benchmark_bandits
    python -m scripts.benchmark_bandits --n_pulls 200 --n_trials 500
"""
import argparse
import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class BanditConfig:
    """Configuration for a bandit problem."""
    name: str
    means: List[float]
    std: float = 1.0

    @property
    def n_arms(self) -> int:
        return len(self.means)

    @property
    def best_arm(self) -> int:
        return int(np.argmax(self.means))

    @property
    def gap(self) -> float:
        """Gap between best and second-best arm."""
        sorted_means = sorted(self.means, reverse=True)
        if len(sorted_means) < 2:
            return 0.0
        return sorted_means[0] - sorted_means[1]


# Bandit configurations to test
BANDIT_CONFIGS = {
    'easy': BanditConfig(
        name='easy',
        means=[1.0, 0.5, 0.5, 0.5],
        std=1.0,
    ),
    'medium': BanditConfig(
        name='medium',
        means=[1.0, 0.8, 0.6, 0.4],
        std=1.0,
    ),
    'hard': BanditConfig(
        name='hard',
        means=[1.0, 0.95, 0.5, 0.5],
        std=1.0,
    ),
    'many': BanditConfig(
        name='many',
        means=[1.0] + [0.5] * 19,
        std=1.0,
    ),
    'very_hard': BanditConfig(
        name='very_hard',
        means=[1.0, 0.99, 0.98, 0.97],
        std=1.0,
    ),
}


class GaussianBandit:
    """Multi-armed bandit with Gaussian rewards."""

    def __init__(self, config: BanditConfig):
        self.config = config
        self.means = np.array(config.means)
        self.std = config.std

    def pull(self, arm: int) -> float:
        """Pull an arm and get a reward."""
        return np.random.normal(self.means[arm], self.std)


class UCBSelection:
    """
    UCB1 selection algorithm (similar to PUCT without priors).

    UCB = Q + c * sqrt(ln(N) / n)

    For recommendation, uses empirical best arm.
    """

    def __init__(self, n_arms: int, c: float = 1.0):
        self.n_arms = n_arms
        self.c = c
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_pulls = 0

    def reset(self):
        """Reset algorithm state."""
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.total_pulls = 0

    def select(self) -> int:
        """Select next arm to pull using UCB."""
        # Pull each arm once first
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        # UCB selection
        ucb_values = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            exploration = self.c * math.sqrt(math.log(self.total_pulls) / self.counts[arm])
            ucb_values[arm] = self.values[arm] + exploration

        return int(np.argmax(ucb_values))

    def update(self, arm: int, reward: float):
        """Update arm statistics with observed reward."""
        self.counts[arm] += 1
        self.total_pulls += 1
        # Incremental mean update
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]

    def recommend(self) -> int:
        """Recommend best arm based on empirical mean."""
        return int(np.argmax(self.values))

    def confidence(self) -> float:
        """
        Estimate confidence in recommendation.

        Uses the gap between best and second-best empirical mean,
        normalized by uncertainty.
        """
        if self.total_pulls < 2 * self.n_arms:
            return 0.0

        sorted_indices = np.argsort(self.values)[::-1]
        best = sorted_indices[0]
        second = sorted_indices[1]

        gap = self.values[best] - self.values[second]
        # Approximate uncertainty
        std_best = 1.0 / math.sqrt(max(1, self.counts[best]))
        std_second = 1.0 / math.sqrt(max(1, self.counts[second]))
        combined_std = math.sqrt(std_best**2 + std_second**2)

        if combined_std < 1e-10:
            return 1.0 if gap > 0 else 0.5

        # Approximate P(best > second) using normal CDF
        z = gap / combined_std
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


class TTTSIDSSelection:
    """
    Top-Two Thompson Sampling with IDS allocation.

    Mirrors the selection logic in BayesianMCTS:
    1. Draw Thompson samples from Gaussian posteriors
    2. Identify leader and challenger
    3. Use IDS allocation to decide which to sample

    For recommendation, uses probability of optimality.
    """

    def __init__(
        self,
        n_arms: int,
        sigma_0: float = 1.0,
        obs_var: float = 0.5,
        alpha: float = 0.5,
        min_var: float = 1e-6
    ):
        self.n_arms = n_arms
        self.sigma_0 = sigma_0
        self.obs_var = obs_var
        self.alpha = alpha
        self.min_var = min_var

        # Gaussian posterior parameters
        self.mu = np.zeros(n_arms)
        self.sigma_sq = np.full(n_arms, sigma_0**2)
        self.total_pulls = 0

    def reset(self):
        """Reset algorithm state."""
        self.mu = np.zeros(self.n_arms)
        self.sigma_sq = np.full(self.n_arms, self.sigma_0**2)
        self.total_pulls = 0

    def select(self) -> int:
        """Select next arm using Top-Two Thompson Sampling with IDS."""
        if self.n_arms == 1:
            return 0

        # Draw Thompson samples
        samples = np.random.normal(self.mu, np.sqrt(self.sigma_sq))

        # Find leader and challenger
        sorted_indices = np.argsort(samples)[::-1]
        leader = sorted_indices[0]
        challenger = sorted_indices[1]

        # IDS allocation
        precision_leader = 1.0 / max(self.sigma_sq[leader], self.min_var)
        precision_challenger = 1.0 / max(self.sigma_sq[challenger], self.min_var)

        # beta = probability of selecting challenger
        # High leader precision -> explore challenger more
        beta = (precision_leader + self.alpha) / (
            precision_leader + precision_challenger + 2 * self.alpha
        )

        if np.random.random() < beta:
            return challenger
        else:
            return leader

    def update(self, arm: int, reward: float):
        """Bayesian update with observed reward."""
        self.total_pulls += 1

        precision_prior = 1.0 / max(self.sigma_sq[arm], self.min_var)
        precision_obs = 1.0 / max(self.obs_var, self.min_var)
        new_precision = precision_prior + precision_obs

        self.mu[arm] = (
            precision_prior * self.mu[arm] + precision_obs * reward
        ) / new_precision
        self.sigma_sq[arm] = max(1.0 / new_precision, self.min_var)

    def recommend(self, n_samples: int = 100) -> int:
        """
        Recommend best arm using probability of optimality.

        Draws Thompson samples and counts which arm wins most often.
        """
        wins = np.zeros(self.n_arms)

        for _ in range(n_samples):
            samples = np.random.normal(self.mu, np.sqrt(self.sigma_sq))
            winner = np.argmax(samples)
            wins[winner] += 1

        return int(np.argmax(wins))

    def confidence(self, n_samples: int = 100) -> float:
        """
        Estimate confidence as max probability of optimality.
        """
        wins = np.zeros(self.n_arms)

        for _ in range(n_samples):
            samples = np.random.normal(self.mu, np.sqrt(self.sigma_sq))
            winner = np.argmax(samples)
            wins[winner] += 1

        return float(np.max(wins) / n_samples)


@dataclass
class EvalResult:
    """Results from evaluating an algorithm."""
    simple_regret_mean: float
    simple_regret_std: float
    prob_correct: float
    samples_to_95: Optional[float]  # Average pulls to reach 95% confidence


def evaluate_algorithm(
    algo,
    bandit: GaussianBandit,
    n_pulls: int,
    n_trials: int = 100,
    confidence_threshold: float = 0.95,
) -> EvalResult:
    """
    Evaluate a bandit algorithm.

    Args:
        algo: Algorithm with select(), update(), recommend(), reset(), confidence()
        bandit: Bandit environment
        n_pulls: Number of pulls per trial
        n_trials: Number of independent trials
        confidence_threshold: Threshold for sample complexity metric

    Returns:
        EvalResult with metrics
    """
    simple_regrets = []
    correct_count = 0
    samples_to_confidence = []

    best_mean = bandit.means[bandit.config.best_arm]

    for trial in range(n_trials):
        algo.reset()
        reached_confidence = None

        for t in range(n_pulls):
            arm = algo.select()
            reward = bandit.pull(arm)
            algo.update(arm, reward)

            # Check if we've reached confidence threshold
            if reached_confidence is None:
                conf = algo.confidence()
                if conf >= confidence_threshold:
                    # Verify recommendation is correct
                    if algo.recommend() == bandit.config.best_arm:
                        reached_confidence = t + 1

        recommended = algo.recommend()
        regret = best_mean - bandit.means[recommended]
        simple_regrets.append(regret)

        if recommended == bandit.config.best_arm:
            correct_count += 1

        if reached_confidence is not None:
            samples_to_confidence.append(reached_confidence)

    avg_samples_to_95 = (
        np.mean(samples_to_confidence) if samples_to_confidence else None
    )

    return EvalResult(
        simple_regret_mean=float(np.mean(simple_regrets)),
        simple_regret_std=float(np.std(simple_regrets)),
        prob_correct=correct_count / n_trials,
        samples_to_95=avg_samples_to_95,
    )


def run_benchmark(
    configs: List[str],
    n_pulls: int = 100,
    n_trials: int = 200,
    ucb_c: float = 1.0,
    ttts_sigma_0: float = 1.0,
    ttts_obs_var: float = 0.5,
    ttts_alpha: float = 0.5,
) -> Dict[str, Dict[str, EvalResult]]:
    """
    Run benchmark across configurations and algorithms.

    Returns:
        Dict mapping config_name -> algorithm_name -> EvalResult
    """
    results = {}

    for config_name in configs:
        if config_name not in BANDIT_CONFIGS:
            print(f"Unknown config: {config_name}, skipping")
            continue

        config = BANDIT_CONFIGS[config_name]
        bandit = GaussianBandit(config)

        results[config_name] = {}

        # UCB
        ucb = UCBSelection(config.n_arms, c=ucb_c)
        results[config_name]['UCB'] = evaluate_algorithm(
            ucb, bandit, n_pulls, n_trials
        )

        # TTTS-IDS
        ttts = TTTSIDSSelection(
            config.n_arms,
            sigma_0=ttts_sigma_0,
            obs_var=ttts_obs_var,
            alpha=ttts_alpha,
        )
        results[config_name]['TTTS-IDS'] = evaluate_algorithm(
            ttts, bandit, n_pulls, n_trials
        )

    return results


def print_results(results: Dict[str, Dict[str, EvalResult]]):
    """Pretty print benchmark results."""
    print("\n" + "=" * 70)
    print("                    Pure Bandit BAI Comparison")
    print("=" * 70)

    for config_name, algo_results in results.items():
        config = BANDIT_CONFIGS[config_name]
        print(f"\nConfig: {config_name} (gap={config.gap:.2f}, {config.n_arms} arms)")
        print("-" * 70)
        print(f"  {'Algorithm':<12} {'Simple Regret':<20} {'P(correct)':<12} {'Samples@95%':<12}")
        print("-" * 70)

        for algo_name, result in algo_results.items():
            regret_str = f"{result.simple_regret_mean:.4f} +/- {result.simple_regret_std:.4f}"
            samples_str = (
                f"{result.samples_to_95:.0f}" if result.samples_to_95 else "n/a"
            )
            print(
                f"  {algo_name:<12} {regret_str:<20} {result.prob_correct:<12.2%} {samples_str:<12}"
            )

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Bandit BAI Benchmark')
    parser.add_argument('--configs', type=str, nargs='+',
                        default=['easy', 'medium', 'hard', 'many', 'very_hard'],
                        help='Bandit configurations to test')
    parser.add_argument('--n_pulls', type=int, default=100,
                        help='Number of pulls per trial')
    parser.add_argument('--n_trials', type=int, default=200,
                        help='Number of trials per configuration')
    parser.add_argument('--ucb_c', type=float, default=1.0,
                        help='UCB exploration constant')
    parser.add_argument('--ttts_sigma_0', type=float, default=1.0,
                        help='TTTS prior std')
    parser.add_argument('--ttts_obs_var', type=float, default=0.5,
                        help='TTTS observation variance')
    parser.add_argument('--ttts_alpha', type=float, default=0.5,
                        help='TTTS IDS alpha')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    print(f"Running bandit benchmark with {args.n_pulls} pulls, {args.n_trials} trials")

    results = run_benchmark(
        configs=args.configs,
        n_pulls=args.n_pulls,
        n_trials=args.n_trials,
        ucb_c=args.ucb_c,
        ttts_sigma_0=args.ttts_sigma_0,
        ttts_obs_var=args.ttts_obs_var,
        ttts_alpha=args.ttts_alpha,
    )

    print_results(results)


if __name__ == '__main__':
    main()
