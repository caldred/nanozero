"""
tests/test_bandit_algorithms.py - Tests for bandit selection algorithms

Tests the UCB and TTTS-IDS implementations used in benchmark_bandits.py
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.benchmark_bandits import (
    GaussianBandit,
    BanditConfig,
    UCBSelection,
    TTTSIDSSelection,
    evaluate_algorithm,
    BANDIT_CONFIGS,
)


class TestBanditConfig:
    def test_properties(self):
        config = BanditConfig(name='test', means=[1.0, 0.5, 0.3])
        assert config.n_arms == 3
        assert config.best_arm == 0
        assert config.gap == 0.5

    def test_gap_single_arm(self):
        config = BanditConfig(name='single', means=[1.0])
        assert config.gap == 0.0

    def test_predefined_configs(self):
        assert 'easy' in BANDIT_CONFIGS
        assert 'hard' in BANDIT_CONFIGS
        assert BANDIT_CONFIGS['easy'].n_arms == 4
        assert BANDIT_CONFIGS['many'].n_arms == 20


class TestGaussianBandit:
    def test_pull_returns_float(self):
        config = BanditConfig(name='test', means=[0.0], std=1.0)
        bandit = GaussianBandit(config)
        reward = bandit.pull(0)
        assert isinstance(reward, float)

    def test_pull_distribution(self):
        """Verify rewards follow expected Gaussian distribution."""
        np.random.seed(42)
        config = BanditConfig(name='test', means=[5.0], std=2.0)
        bandit = GaussianBandit(config)

        rewards = [bandit.pull(0) for _ in range(1000)]
        assert abs(np.mean(rewards) - 5.0) < 0.2
        assert abs(np.std(rewards) - 2.0) < 0.2


class TestUCBSelection:
    def test_initialization(self):
        ucb = UCBSelection(n_arms=4, c=1.5)
        assert ucb.n_arms == 4
        assert ucb.c == 1.5
        assert np.all(ucb.counts == 0)

    def test_initial_exploration(self):
        """UCB should try each arm once before using UCB formula."""
        ucb = UCBSelection(n_arms=3)
        selected = set()
        for _ in range(3):
            arm = ucb.select()
            selected.add(arm)
            ucb.update(arm, 0.0)
        assert selected == {0, 1, 2}

    def test_update_increments_counts(self):
        ucb = UCBSelection(n_arms=2)
        ucb.update(0, 1.0)
        ucb.update(0, 2.0)
        ucb.update(1, 0.5)
        assert ucb.counts[0] == 2
        assert ucb.counts[1] == 1
        assert ucb.total_pulls == 3

    def test_value_tracking(self):
        """Test that values track empirical mean correctly."""
        ucb = UCBSelection(n_arms=2)
        ucb.update(0, 1.0)
        ucb.update(0, 3.0)
        assert ucb.values[0] == 2.0  # Mean of 1.0 and 3.0

    def test_recommend_best_arm(self):
        ucb = UCBSelection(n_arms=3)
        # Arm 1 has highest empirical mean
        ucb.update(0, 0.0)
        ucb.update(1, 5.0)
        ucb.update(2, 2.0)
        assert ucb.recommend() == 1

    def test_reset(self):
        ucb = UCBSelection(n_arms=2)
        ucb.update(0, 1.0)
        ucb.reset()
        assert np.all(ucb.counts == 0)
        assert np.all(ucb.values == 0)
        assert ucb.total_pulls == 0

    def test_confidence_increases(self):
        """Confidence should increase with more samples."""
        np.random.seed(42)
        ucb = UCBSelection(n_arms=2)

        # Give arm 0 much higher rewards
        for _ in range(10):
            ucb.update(0, 2.0)
            ucb.update(1, 0.0)

        conf1 = ucb.confidence()

        for _ in range(40):
            ucb.update(0, 2.0)
            ucb.update(1, 0.0)

        conf2 = ucb.confidence()

        assert conf2 > conf1


class TestTTTSIDSSelection:
    def test_initialization(self):
        ttts = TTTSIDSSelection(n_arms=4, sigma_0=1.5, obs_var=0.3, alpha=0.7)
        assert ttts.n_arms == 4
        assert ttts.sigma_0 == 1.5
        assert ttts.obs_var == 0.3
        assert ttts.alpha == 0.7
        assert np.all(ttts.mu == 0)
        assert np.allclose(ttts.sigma_sq, 1.5**2)

    def test_bayesian_update(self):
        """Test that Bayesian updates follow precision weighting."""
        ttts = TTTSIDSSelection(n_arms=1, sigma_0=1.0, obs_var=1.0)

        # After one observation of 2.0:
        # precision_prior = 1.0, precision_obs = 1.0
        # new_precision = 2.0
        # mu = (1.0 * 0.0 + 1.0 * 2.0) / 2.0 = 1.0
        # sigma_sq = 1/2.0 = 0.5
        ttts.update(0, 2.0)

        assert abs(ttts.mu[0] - 1.0) < 1e-6
        assert abs(ttts.sigma_sq[0] - 0.5) < 1e-6

    def test_variance_decreases(self):
        """Variance should decrease with more observations."""
        ttts = TTTSIDSSelection(n_arms=1)
        initial_var = ttts.sigma_sq[0]

        for _ in range(10):
            ttts.update(0, 0.0)

        assert ttts.sigma_sq[0] < initial_var

    def test_select_returns_valid_arm(self):
        np.random.seed(42)
        ttts = TTTSIDSSelection(n_arms=4)
        for _ in range(100):
            arm = ttts.select()
            assert 0 <= arm < 4

    def test_recommend_returns_valid_arm(self):
        np.random.seed(42)
        ttts = TTTSIDSSelection(n_arms=4)
        for i in range(4):
            ttts.update(i, float(i))

        recommended = ttts.recommend()
        assert 0 <= recommended < 4

    def test_recommend_finds_best(self):
        """With clear winner, should recommend best arm."""
        np.random.seed(42)
        ttts = TTTSIDSSelection(n_arms=3, sigma_0=0.1, obs_var=0.1)

        # Give arm 2 much higher rewards
        for _ in range(50):
            ttts.update(0, 0.0)
            ttts.update(1, 0.5)
            ttts.update(2, 2.0)

        # With 500 samples, should reliably find arm 2
        assert ttts.recommend(n_samples=500) == 2

    def test_reset(self):
        ttts = TTTSIDSSelection(n_arms=2, sigma_0=1.0)
        ttts.update(0, 1.0)
        ttts.reset()
        assert np.all(ttts.mu == 0)
        assert np.allclose(ttts.sigma_sq, 1.0)
        assert ttts.total_pulls == 0

    def test_confidence_interpretation(self):
        """Confidence should reflect probability of optimality."""
        np.random.seed(42)
        ttts = TTTSIDSSelection(n_arms=2, sigma_0=0.1, obs_var=0.1)

        # Make arm 0 clearly better
        for _ in range(100):
            ttts.update(0, 1.0)
            ttts.update(1, 0.0)

        conf = ttts.confidence(n_samples=500)
        assert conf > 0.95  # Should be very confident

    def test_ids_allocation_logic(self):
        """Test that IDS allocation works as expected."""
        np.random.seed(42)
        ttts = TTTSIDSSelection(n_arms=2, sigma_0=1.0, obs_var=0.5, alpha=0.5)

        # Make arm 0 have lower variance (more samples)
        for _ in range(20):
            ttts.update(0, 0.5)
        ttts.update(1, 0.5)  # One sample for arm 1

        # Arm 0 has higher precision, so we should sample arm 1 (challenger) more
        selections = [ttts.select() for _ in range(100)]
        arm1_fraction = sum(1 for s in selections if s == 1) / len(selections)

        # With high precision on arm 0, IDS should explore arm 1 more
        # This is probabilistic, so allow some variance
        assert arm1_fraction > 0.3


class TestEvaluateAlgorithm:
    def test_evaluation_runs(self):
        """Basic smoke test that evaluation completes."""
        np.random.seed(42)
        config = BanditConfig(name='test', means=[1.0, 0.5])
        bandit = GaussianBandit(config)
        ucb = UCBSelection(n_arms=2)

        result = evaluate_algorithm(ucb, bandit, n_pulls=20, n_trials=5)

        assert 0 <= result.prob_correct <= 1
        assert result.simple_regret_mean >= 0

    def test_easy_problem_high_accuracy(self):
        """With a large gap, should achieve high accuracy."""
        np.random.seed(42)
        config = BanditConfig(name='easy', means=[2.0, 0.0], std=0.5)
        bandit = GaussianBandit(config)
        ttts = TTTSIDSSelection(n_arms=2, sigma_0=1.0, obs_var=0.5)

        result = evaluate_algorithm(ttts, bandit, n_pulls=50, n_trials=50)

        assert result.prob_correct > 0.8

    def test_simple_regret_interpretation(self):
        """Simple regret should be 0 when correct, gap when wrong."""
        np.random.seed(42)
        config = BanditConfig(name='test', means=[1.0, 0.5])
        bandit = GaussianBandit(config)
        ucb = UCBSelection(n_arms=2)

        result = evaluate_algorithm(ucb, bandit, n_pulls=100, n_trials=100)

        # Expected simple regret = (1 - prob_correct) * gap
        expected_regret = (1 - result.prob_correct) * config.gap
        assert abs(result.simple_regret_mean - expected_regret) < 0.1


class TestAlgorithmComparison:
    """Integration tests comparing algorithms."""

    def test_both_algorithms_reasonable_performance(self):
        """Both UCB and TTTS should achieve reasonable accuracy on easy problem."""
        np.random.seed(42)
        config = BANDIT_CONFIGS['easy']
        bandit = GaussianBandit(config)

        ucb = UCBSelection(n_arms=config.n_arms)
        ucb_result = evaluate_algorithm(ucb, bandit, n_pulls=100, n_trials=50)

        ttts = TTTSIDSSelection(n_arms=config.n_arms)
        ttts_result = evaluate_algorithm(ttts, bandit, n_pulls=100, n_trials=50)

        # Both should get > 70% accuracy on easy problem
        assert ucb_result.prob_correct > 0.7
        assert ttts_result.prob_correct > 0.7

    def test_algorithms_handle_many_arms(self):
        """Both algorithms should handle many arms gracefully."""
        np.random.seed(42)
        config = BANDIT_CONFIGS['many']
        bandit = GaussianBandit(config)

        ucb = UCBSelection(n_arms=config.n_arms)
        ucb_result = evaluate_algorithm(ucb, bandit, n_pulls=200, n_trials=30)

        ttts = TTTSIDSSelection(n_arms=config.n_arms)
        ttts_result = evaluate_algorithm(ttts, bandit, n_pulls=200, n_trials=30)

        # With 20 arms and only 200 pulls, accuracy will be lower
        # But both should do better than random (5%)
        assert ucb_result.prob_correct > 0.2
        assert ttts_result.prob_correct > 0.2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
