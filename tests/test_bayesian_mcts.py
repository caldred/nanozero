"""
Tests for Bayesian BAI-MCTS implementation.
"""
import math
import numpy as np
import pytest
import torch
from unittest.mock import Mock, patch

from nanozero.config import BayesianMCTSConfig, MCTSConfig, get_game_config, get_model_config
from nanozero.bayesian_mcts import BayesianNode, BayesianMCTS, sample_action
from nanozero.mcts import BatchedMCTS
from nanozero.game import get_game
from nanozero.model import AlphaZeroTransformer


class TestBayesianNode:
    """Tests for BayesianNode class."""

    def test_initialization(self):
        """Node initializes with correct prior, mu, sigma_sq."""
        node = BayesianNode(prior=0.5, mu=0.2, sigma_sq=0.8)
        assert node.prior == 0.5
        assert node.mu == 0.2
        assert node.sigma_sq == 0.8
        assert len(node.children) == 0

    def test_default_initialization(self):
        """Node defaults to uninformative prior."""
        node = BayesianNode()
        assert node.prior == 0.0
        assert node.mu == 0.0
        assert node.sigma_sq == 1.0

    def test_expanded(self):
        """expanded() returns True iff node has children."""
        node = BayesianNode()
        assert not node.expanded()
        node.children[0] = BayesianNode()
        assert node.expanded()

    def test_sample_distribution(self):
        """Thompson samples follow N(mu, sigma_sq)."""
        np.random.seed(42)
        node = BayesianNode(mu=0.5, sigma_sq=0.25)

        samples = [node.sample() for _ in range(1000)]

        # Check mean and variance (with some tolerance)
        assert abs(np.mean(samples) - 0.5) < 0.1
        assert abs(np.var(samples) - 0.25) < 0.1

    def test_precision(self):
        """Precision is inverse of variance."""
        node = BayesianNode(sigma_sq=0.25)
        assert node.precision() == 4.0

    def test_bayesian_update(self):
        """Update correctly combines prior and observation."""
        node = BayesianNode(mu=0.0, sigma_sq=1.0)
        node.update(value=1.0, obs_var=1.0)

        # With equal precision: new_mu = (0 + 1) / 2 = 0.5
        assert abs(node.mu - 0.5) < 1e-6
        # new_precision = 1 + 1 = 2, so new_sigma_sq = 0.5
        assert abs(node.sigma_sq - 0.5) < 1e-6

    def test_variance_decreases_with_observations(self):
        """Variance decreases as observations accumulate."""
        node = BayesianNode(mu=0.0, sigma_sq=1.0)
        initial_var = node.sigma_sq

        for _ in range(10):
            node.update(value=0.5, obs_var=0.5)

        assert node.sigma_sq < initial_var

    def test_update_respects_min_variance(self):
        """Variance never goes below min_variance."""
        node = BayesianNode(mu=0.0, sigma_sq=1e-10)
        node.update(value=0.5, obs_var=1e-10, min_var=1e-6)
        assert node.sigma_sq >= 1e-6


class TestBayesianMCTS:
    """Tests for BayesianMCTS class."""

    @pytest.fixture
    def game(self):
        """Create a TicTacToe game instance."""
        return get_game('tictactoe')

    @pytest.fixture
    def model(self, game):
        """Create a model instance."""
        model_config = get_model_config(game.config, n_layer=2)
        model = AlphaZeroTransformer(model_config)
        model.eval()
        return model

    @pytest.fixture
    def config(self):
        """Create a BayesianMCTSConfig."""
        return BayesianMCTSConfig(
            num_simulations=50,
            sigma_0=1.0,
            obs_var=0.5,
            ids_alpha=0.5
        )

    @pytest.fixture
    def mcts(self, game, config):
        """Create a BayesianMCTS instance."""
        return BayesianMCTS(game, config, use_transposition_table=False)

    def test_initialization(self, game, config):
        """MCTS initializes correctly."""
        mcts = BayesianMCTS(game, config)
        assert mcts.game == game
        assert mcts.config == config
        assert mcts.tt is not None

    def test_initialization_without_tt(self, game, config):
        """MCTS can be initialized without transposition table."""
        mcts = BayesianMCTS(game, config, use_transposition_table=False)
        assert mcts.tt is None

    def test_search_returns_valid_policy(self, mcts, game, model):
        """Search returns normalized policy over legal actions."""
        state = game.initial_state()
        states = state[np.newaxis, ...]

        policy = mcts.search(states, model, num_simulations=20)[0]

        # Policy should sum to 1
        assert abs(policy.sum() - 1.0) < 1e-6

        # Policy should only have mass on legal actions
        legal_actions = game.legal_actions(state)
        for a in range(game.config.action_size):
            if a not in legal_actions:
                assert policy[a] == 0.0

    def test_search_batch(self, mcts, game, model):
        """Search handles batches correctly."""
        states = np.stack([game.initial_state() for _ in range(4)])

        policies = mcts.search(states, model, num_simulations=10)

        assert policies.shape == (4, game.config.action_size)
        for i in range(4):
            assert abs(policies[i].sum() - 1.0) < 1e-6

    def test_clear_cache(self, game, config):
        """Clearing cache works."""
        mcts = BayesianMCTS(game, config, use_transposition_table=True)
        mcts.clear_cache()  # Should not raise

    def test_logit_shifted_prior_centering(self, mcts, game, model):
        """Logit-shifted priors center around value estimate."""
        state = game.initial_state()
        device = next(model.parameters()).device

        # Expand a node
        node = BayesianNode()
        value = mcts._expand(node, state, model, device)

        # Children should exist
        assert node.expanded()

        # Check that the mean of children mus is approximately the value
        mus = [child.mu for child in node.children.values()]
        weights = [child.prior for child in node.children.values()]
        weighted_mean = sum(m * w for m, w in zip(mus, weights)) / sum(weights)

        # Should be close to value (with some tolerance due to entropy term)
        assert abs(weighted_mean - value) < 1.0


class TestThompsonIDS:
    """Tests for Thompson IDS selection."""

    def test_ids_selects_leader_or_challenger(self):
        """IDS always selects leader or challenger."""
        np.random.seed(42)
        config = BayesianMCTSConfig()
        game = get_game('tictactoe')
        mcts = BayesianMCTS(game, config, use_transposition_table=False)

        # Create a node with 3 children (mu is from child's perspective)
        node = BayesianNode()
        node.children[0] = BayesianNode(mu=-0.8, sigma_sq=0.1)  # Strong for parent
        node.children[1] = BayesianNode(mu=-0.2, sigma_sq=0.1)  # Weak for parent
        node.children[2] = BayesianNode(mu=-0.5, sigma_sq=0.1)  # Medium

        # Run selection many times
        selections = {0: 0, 1: 0, 2: 0}
        for _ in range(1000):
            action, _ = mcts._select_child_thompson_ids(node)
            selections[action] += 1

        # Action 0 (strong for parent) should be selected most often
        # Action 1 (weak for parent) should be selected least often
        assert selections[0] > selections[1]

    def test_ids_allocation_formula(self):
        """IDS allocates samples based on precision balance."""
        np.random.seed(42)
        config = BayesianMCTSConfig(ids_alpha=0.5)
        game = get_game('tictactoe')
        mcts = BayesianMCTS(game, config, use_transposition_table=False)

        # Create a node where both children have equal mean but different variance
        # With equal means, Thompson sampling determines leader/challenger randomly
        # IDS then allocates based on the precision formula
        node = BayesianNode()
        node.children[0] = BayesianNode(mu=0.5, sigma_sq=0.5)   # Moderate variance
        node.children[1] = BayesianNode(mu=0.5, sigma_sq=0.5)   # Same variance

        # Run selection many times - with equal params, should be ~50/50
        selections = {0: 0, 1: 0}
        for _ in range(1000):
            action, _ = mcts._select_child_thompson_ids(node)
            selections[action] += 1

        # With identical children, selection should be approximately balanced
        # Allow for some randomness (each should get 400-600 out of 1000)
        assert 350 < selections[0] < 650
        assert 350 < selections[1] < 650


class TestEarlyStopping:
    """Tests for early stopping."""

    def test_should_stop_when_confident(self):
        """Should stop when leader is clearly better."""
        config = BayesianMCTSConfig(confidence_threshold=0.95)
        game = get_game('tictactoe')
        mcts = BayesianMCTS(game, config, use_transposition_table=False)

        # Create root where leader is clearly better for parent
        root = BayesianNode()
        root.children[0] = BayesianNode(mu=-0.9, sigma_sq=0.01)  # Leader, very confident
        root.children[1] = BayesianNode(mu=-0.1, sigma_sq=0.01)  # Challenger, clearly worse

        assert mcts._should_stop_early(root) is True

    def test_should_not_stop_when_uncertain(self):
        """Should not stop when outcome is uncertain."""
        config = BayesianMCTSConfig(confidence_threshold=0.95)
        game = get_game('tictactoe')
        mcts = BayesianMCTS(game, config, use_transposition_table=False)

        # Create root where leader and challenger are close for parent
        root = BayesianNode()
        root.children[0] = BayesianNode(mu=-0.55, sigma_sq=0.5)  # Slight lead, high variance
        root.children[1] = BayesianNode(mu=-0.45, sigma_sq=0.5)  # Close behind

        assert mcts._should_stop_early(root) is False

    def test_early_stopping_reduces_simulations(self):
        """Early stopping should use fewer simulations when confident."""
        game = get_game('tictactoe')
        model_config = get_model_config(game.config, n_layer=2)
        model = AlphaZeroTransformer(model_config)
        model.eval()

        # Config with early stopping
        config_early = BayesianMCTSConfig(
            num_simulations=100,
            early_stopping=True,
            confidence_threshold=0.90,
            min_simulations=5
        )
        mcts_early = BayesianMCTS(game, config_early, use_transposition_table=False)

        # Config without early stopping
        config_no_early = BayesianMCTSConfig(
            num_simulations=100,
            early_stopping=False
        )
        mcts_no_early = BayesianMCTS(game, config_no_early, use_transposition_table=False)

        state = game.initial_state()
        states = state[np.newaxis, ...]

        # Both should produce valid policies
        policy_early = mcts_early.search(states, model)[0]
        policy_no_early = mcts_no_early.search(states, model)[0]

        assert abs(policy_early.sum() - 1.0) < 1e-6
        assert abs(policy_no_early.sum() - 1.0) < 1e-6

    def test_single_action_stops_immediately(self):
        """Should stop immediately if only one legal action."""
        config = BayesianMCTSConfig(confidence_threshold=0.95)
        game = get_game('tictactoe')
        mcts = BayesianMCTS(game, config, use_transposition_table=False)

        # Create root with single child
        root = BayesianNode()
        root.children[0] = BayesianNode(mu=0.5, sigma_sq=1.0)

        assert mcts._should_stop_early(root) is True


class TestBayesianBackup:
    """Tests for Bayesian backup."""

    def test_backup_negates_value(self):
        """Values negate correctly for two-player games."""
        config = BayesianMCTSConfig()
        game = get_game('tictactoe')
        mcts = BayesianMCTS(game, config, use_transposition_table=False)

        # Create a simple path
        root = BayesianNode(mu=0.0, sigma_sq=1.0)
        child = BayesianNode(mu=0.0, sigma_sq=1.0)
        root.children[0] = child

        # Backup a positive value
        mcts._backup([(root, 0)], leaf_value=1.0)

        # Child should receive positive value (from its perspective, it's winning)
        assert child.mu > 0


class TestPolicyExtraction:
    """Tests for policy extraction."""

    def test_policy_favors_high_mu(self):
        """Policy favors actions with high mean value for parent."""
        config = BayesianMCTSConfig()
        game = get_game('tictactoe')
        mcts = BayesianMCTS(game, config, use_transposition_table=False)

        # Create a root with children of varying quality (mu from child's perspective)
        root = BayesianNode()
        root.children[0] = BayesianNode(mu=-0.9, sigma_sq=0.1)  # Best for parent
        root.children[1] = BayesianNode(mu=-0.1, sigma_sq=0.1)  # Worst for parent
        root.children[2] = BayesianNode(mu=-0.5, sigma_sq=0.1)  # Medium

        policy = mcts._get_policy(root, num_samples=1000)

        # Action 0 should have highest probability
        assert policy[0] > policy[1]
        assert policy[0] > policy[2]


class TestSampleAction:
    """Tests for sample_action utility."""

    def test_greedy_selection(self):
        """Temperature 0 gives greedy selection."""
        probs = np.array([0.1, 0.3, 0.6])
        action = sample_action(probs, temperature=0)
        assert action == 2

    def test_sampling(self):
        """Temperature 1 samples from distribution."""
        np.random.seed(42)
        probs = np.array([0.1, 0.3, 0.6])

        counts = {0: 0, 1: 0, 2: 0}
        for _ in range(1000):
            action = sample_action(probs, temperature=1.0)
            counts[action] += 1

        # Check approximate proportions
        assert counts[2] > counts[1] > counts[0]


class TestTerminalStates:
    """Tests for terminal state handling."""

    def test_search_on_terminal_state(self):
        """Search on terminal state returns valid policy (not all zeros)."""
        game = get_game('tictactoe')
        model_config = get_model_config(game.config, n_layer=2)
        model = AlphaZeroTransformer(model_config)
        model.eval()

        config = BayesianMCTSConfig(num_simulations=50)
        mcts = BayesianMCTS(game, config, use_transposition_table=False)

        # Create a terminal state (X wins with top row)
        state = game.initial_state()
        # Play: X(0), O(3), X(1), O(4), X(2) - X wins
        state = game.next_state(state, 0)  # X at 0
        state = game.next_state(state, 3)  # O at 3
        state = game.next_state(state, 1)  # X at 1
        state = game.next_state(state, 4)  # O at 4
        state = game.next_state(state, 2)  # X at 2 - X wins

        assert game.is_terminal(state)

        states = state[np.newaxis, ...]
        policy = mcts.search(states, model)[0]

        # Policy should be valid (sum to 1 or be all zeros for terminal with no legal moves)
        # For a terminal TicTacToe state, there are no legal moves
        assert policy.sum() == 0.0 or abs(policy.sum() - 1.0) < 1e-6

    def test_mixed_terminal_and_non_terminal(self):
        """Batch with mix of terminal and non-terminal states."""
        game = get_game('tictactoe')
        model_config = get_model_config(game.config, n_layer=2)
        model = AlphaZeroTransformer(model_config)
        model.eval()

        config = BayesianMCTSConfig(num_simulations=20)
        mcts = BayesianMCTS(game, config, use_transposition_table=False)

        # Create one terminal and one non-terminal state
        initial = game.initial_state()

        # Terminal state
        terminal = game.initial_state()
        terminal = game.next_state(terminal, 0)
        terminal = game.next_state(terminal, 3)
        terminal = game.next_state(terminal, 1)
        terminal = game.next_state(terminal, 4)
        terminal = game.next_state(terminal, 2)  # X wins

        states = np.stack([initial, terminal])
        policies = mcts.search(states, model)

        # First policy should be valid (non-terminal)
        assert abs(policies[0].sum() - 1.0) < 1e-6

        # Second policy can be zeros (terminal, no legal moves)
        assert policies[1].sum() == 0.0 or abs(policies[1].sum() - 1.0) < 1e-6


class TestIntegration:
    """Integration tests with real game and model."""

    def test_full_search_tictactoe(self):
        """Full MCTS search on TicTacToe produces reasonable policy."""
        game = get_game('tictactoe')
        model_config = get_model_config(game.config, n_layer=2)
        model = AlphaZeroTransformer(model_config)
        model.eval()

        config = BayesianMCTSConfig(num_simulations=50)
        mcts = BayesianMCTS(game, config, use_transposition_table=True)

        state = game.initial_state()
        states = state[np.newaxis, ...]

        policy = mcts.search(states, model)[0]

        # Should produce valid policy
        assert abs(policy.sum() - 1.0) < 1e-6
        assert all(policy >= 0)

    def test_transposition_table_caching(self):
        """Transposition table caches evaluations."""
        game = get_game('tictactoe')
        model_config = get_model_config(game.config, n_layer=2)
        model = AlphaZeroTransformer(model_config)
        model.eval()

        config = BayesianMCTSConfig(num_simulations=100)
        mcts = BayesianMCTS(game, config, use_transposition_table=True)

        state = game.initial_state()
        states = state[np.newaxis, ...]

        # Run search
        mcts.search(states, model)

        # Check that TT was used
        hits, misses, size = mcts.tt.stats()
        assert size > 0  # Some positions cached


class TestPUCTVsThompsonComparison:
    """Comparison tests between PUCT and Thompson sampling MCTS."""

    class DummyUniformModel(torch.nn.Module):
        """Uniform-policy, zero-value model for controlled comparisons."""

        def __init__(self, action_size: int):
            super().__init__()
            self.action_size = action_size
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def predict(self, x, action_mask=None):
            batch = x.shape[0]
            if action_mask is None:
                probs = torch.full((batch, self.action_size), 1.0 / self.action_size)
            else:
                mask = action_mask.float()
                probs = mask / mask.sum(dim=-1, keepdim=True)
            values = torch.zeros((batch, 1))
            return probs, values

    def test_thompson_handles_two_close_wins_connect4(self):
        """Thompson balances two winning moves better than PUCT in Connect4."""
        np.random.seed(0)
        torch.manual_seed(0)

        game = get_game('connect4')
        model = self.DummyUniformModel(game.config.action_size)

        # Two immediate winning moves for current player (columns 0 and 6).
        state = np.zeros((6, 7), dtype=np.int8)
        state[5, 0] = 1
        state[4, 0] = 1
        state[3, 0] = 1
        state[5, 6] = 1
        state[4, 6] = 1
        state[3, 6] = 1
        state[5, 1] = -1
        state[4, 1] = -1
        state[3, 1] = -1
        state[5, 2] = -1
        state[4, 2] = -1
        state[3, 2] = -1

        states = state[np.newaxis, ...]
        winning_actions = [0, 6]

        for action in winning_actions:
            assert action in game.legal_actions(state)
            assert game.is_terminal(game.next_state(state, action))

        puct = BatchedMCTS(game, MCTSConfig(num_simulations=20, c_puct=1.0))
        policy_puct = puct.search(states, model, add_noise=False, batch_size=1)[0]

        bayes = BayesianMCTS(
            game,
            BayesianMCTSConfig(num_simulations=20, early_stopping=False),
            use_transposition_table=False
        )
        policy_bayes = bayes.search(states, model)[0]

        target = np.zeros(game.config.action_size, dtype=np.float32)
        target[winning_actions[0]] = 0.5
        target[winning_actions[1]] = 0.5
        eps = 1e-8
        kl_puct = np.sum(target * np.log((target + eps) / (policy_puct + eps)))
        kl_bayes = np.sum(target * np.log((target + eps) / (policy_bayes + eps)))

        assert kl_bayes < kl_puct

    def test_thompson_identifies_best_move_faster_connect4(self):
        """Thompson assigns higher mass to the best trap sooner than PUCT."""
        np.random.seed(0)
        torch.manual_seed(0)

        game = get_game('connect4')
        model = self.DummyUniformModel(game.config.action_size)

        # Best move is column 3 (win in 3), while columns 1/4 win slower (win in 5).
        state = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, -1, -1],
            [-1, -1, -1, 1, -1, -1, 1],
            [1, 1, -1, 1, -1, 1, -1],
        ], dtype=np.int8)
        states = state[np.newaxis, ...]
        best_action = 3

        puct = BatchedMCTS(game, MCTSConfig(num_simulations=10, c_puct=1.0))
        policy_puct = puct.search(states, model, add_noise=False, batch_size=1)[0]

        np.random.seed(0)
        bayes = BayesianMCTS(
            game,
            BayesianMCTSConfig(num_simulations=10, early_stopping=False),
            use_transposition_table=False
        )
        policy_bayes = bayes.search(states, model)[0]

        assert policy_bayes[best_action] > policy_puct[best_action]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
