"""
Tests for Bayesian MCTS implementation (Rust backend).
"""
import numpy as np
import pytest
import torch

from nanozero.config import BayesianMCTSConfig, MCTSConfig, get_game_config, get_model_config
from nanozero.mcts import BayesianMCTS, BatchedMCTS
from nanozero.game import get_game
from nanozero.model import AlphaZeroTransformer
from nanozero.common import sample_action


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
        return BayesianMCTS(game, config)

    def test_initialization(self, game, config):
        """MCTS initializes correctly."""
        mcts = BayesianMCTS(game, config)
        assert mcts.game == game
        assert mcts.config == config
        assert mcts.backend == 'rust'

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
        """Clearing cache works (no-op for Rust)."""
        mcts = BayesianMCTS(game, config)
        mcts.clear_cache()  # Should not raise


class TestBayesianMCTSConnect4:
    """Tests for BayesianMCTS with Connect4."""

    @pytest.fixture
    def game(self):
        return get_game('connect4')

    @pytest.fixture
    def model(self, game):
        model_config = get_model_config(game.config, n_layer=2)
        model = AlphaZeroTransformer(model_config)
        model.eval()
        return model

    @pytest.fixture
    def config(self):
        return BayesianMCTSConfig(num_simulations=20)

    def test_search_connect4(self, game, model, config):
        """BayesianMCTS works with Connect4."""
        mcts = BayesianMCTS(game, config)
        state = game.initial_state()
        states = state[np.newaxis, ...]

        policy = mcts.search(states, model)[0]

        assert abs(policy.sum() - 1.0) < 1e-6
        # All 7 columns should be legal initially
        assert np.count_nonzero(policy) == 7


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
        mcts = BayesianMCTS(game, config)

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
        mcts = BayesianMCTS(game, config)

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
        mcts = BayesianMCTS(game, config)

        state = game.initial_state()
        states = state[np.newaxis, ...]

        policy = mcts.search(states, model)[0]

        # Should produce valid policy
        assert abs(policy.sum() - 1.0) < 1e-6
        assert all(policy >= 0)


class TestPUCTVsBayesianComparison:
    """Comparison tests between PUCT and Bayesian MCTS."""

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

    def test_both_produce_valid_policies(self):
        """Both PUCT and Bayesian produce valid policies."""
        np.random.seed(0)
        torch.manual_seed(0)

        game = get_game('connect4')
        model = self.DummyUniformModel(game.config.action_size)

        state = game.initial_state()
        states = state[np.newaxis, ...]

        puct = BatchedMCTS(game, MCTSConfig(num_simulations=20, c_puct=1.0))
        policy_puct = puct.search(states, model, add_noise=False)[0]

        bayes = BayesianMCTS(game, BayesianMCTSConfig(num_simulations=20))
        policy_bayes = bayes.search(states, model)[0]

        # Both should produce valid policies
        assert abs(policy_puct.sum() - 1.0) < 1e-6
        assert abs(policy_bayes.sum() - 1.0) < 1e-6

        # Both should have same legal actions
        assert np.count_nonzero(policy_puct) == np.count_nonzero(policy_bayes)


class TestBayesianMCTSGo:
    """Tests for BayesianMCTS with Go."""

    @pytest.fixture
    def game(self):
        return get_game('go9x9')

    @pytest.fixture
    def model(self, game):
        model_config = get_model_config(game.config, n_layer=2)
        model = AlphaZeroTransformer(model_config)
        model.eval()
        return model

    @pytest.fixture
    def config(self):
        return BayesianMCTSConfig(num_simulations=10)

    def test_search_go(self, game, model, config):
        """BayesianMCTS works with Go."""
        mcts = BayesianMCTS(game, config)
        state = game.initial_state()
        states = state[np.newaxis, ...]

        policy = mcts.search(states, model)[0]

        assert abs(policy.sum() - 1.0) < 1e-6
        # All 81 positions + pass should be legal initially
        assert np.count_nonzero(policy) == 82


class TestMCTSFactoryFunctions:
    """Tests for MCTS factory functions."""

    def test_get_batched_mcts(self):
        """get_batched_mcts returns BatchedMCTS instance."""
        from nanozero.mcts import get_batched_mcts, BatchedMCTS
        from nanozero.config import MCTSConfig

        game = get_game('tictactoe')
        config = MCTSConfig(num_simulations=10)
        mcts = get_batched_mcts(game, config)

        assert isinstance(mcts, BatchedMCTS)
        assert mcts.backend == 'rust'

    def test_get_bayesian_mcts(self):
        """get_bayesian_mcts returns BayesianMCTS instance."""
        from nanozero.mcts import get_bayesian_mcts, BayesianMCTS

        game = get_game('tictactoe')
        config = BayesianMCTSConfig(num_simulations=10)
        mcts = get_bayesian_mcts(game, config)

        assert isinstance(mcts, BayesianMCTS)
        assert mcts.backend == 'rust'

    def test_use_rust_param_ignored(self):
        """use_rust parameter is accepted but ignored."""
        from nanozero.mcts import get_batched_mcts, get_bayesian_mcts
        from nanozero.config import MCTSConfig

        game = get_game('tictactoe')

        mcts1 = get_batched_mcts(game, MCTSConfig(), use_rust=True)
        mcts2 = get_batched_mcts(game, MCTSConfig(), use_rust=False)
        assert mcts1.backend == 'rust'
        assert mcts2.backend == 'rust'

    def test_is_rust_mcts_available(self):
        """is_rust_mcts_available returns True."""
        from nanozero.mcts import is_rust_mcts_available
        assert is_rust_mcts_available() == True


class TestBatchedMCTS:
    """Tests for BatchedMCTS class."""

    @pytest.fixture
    def game(self):
        return get_game('tictactoe')

    @pytest.fixture
    def model(self, game):
        model_config = get_model_config(game.config, n_layer=2)
        model = AlphaZeroTransformer(model_config)
        model.eval()
        return model

    @pytest.fixture
    def config(self):
        return MCTSConfig(num_simulations=20)

    def test_initialization(self, game, config):
        """BatchedMCTS initializes correctly."""
        mcts = BatchedMCTS(game, config)
        assert mcts.game == game
        assert mcts.config == config
        assert mcts.backend == 'rust'

    def test_search_returns_valid_policy(self, game, config, model):
        """Search returns normalized policy."""
        mcts = BatchedMCTS(game, config)
        state = game.initial_state()
        states = state[np.newaxis, ...]

        policy = mcts.search(states, model, add_noise=False)[0]

        assert abs(policy.sum() - 1.0) < 1e-6
        assert all(policy >= 0)

    def test_search_with_noise(self, game, config, model):
        """Search with noise still produces valid policy."""
        mcts = BatchedMCTS(game, config)
        state = game.initial_state()
        states = state[np.newaxis, ...]

        policy = mcts.search(states, model, add_noise=True)[0]

        assert abs(policy.sum() - 1.0) < 1e-6

    def test_search_batch(self, game, config, model):
        """Search handles batches correctly."""
        mcts = BatchedMCTS(game, config)
        states = np.stack([game.initial_state() for _ in range(4)])

        policies = mcts.search(states, model, add_noise=False)

        assert policies.shape == (4, game.config.action_size)

    def test_clear_cache(self, game, config):
        """clear_cache doesn't raise."""
        mcts = BatchedMCTS(game, config)
        mcts.clear_cache()  # Should not raise


class TestBatchedMCTSConnect4:
    """Tests for BatchedMCTS with Connect4."""

    def test_search_connect4(self):
        """BatchedMCTS works with Connect4."""
        game = get_game('connect4')
        model_config = get_model_config(game.config, n_layer=2)
        model = AlphaZeroTransformer(model_config)
        model.eval()

        config = MCTSConfig(num_simulations=10)
        mcts = BatchedMCTS(game, config)

        state = game.initial_state()
        states = state[np.newaxis, ...]

        policy = mcts.search(states, model, add_noise=False)[0]

        assert abs(policy.sum() - 1.0) < 1e-6
        assert np.count_nonzero(policy) == 7  # All columns legal


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
