"""
Tests for MCTS transposition table integration.

Tests verify:
- cache_stats() and clear_cache() work correctly
- Symmetric positions share cache entries
- TT reduces NN calls by caching evaluations
"""
import numpy as np
import pytest
import torch

from nanozero.config import BayesianMCTSConfig, MCTSConfig, get_game_config, get_model_config
from nanozero.mcts import BayesianMCTS, BatchedMCTS
from nanozero.game import get_game
from nanozero.model import AlphaZeroTransformer


class CountingModel(torch.nn.Module):
    """Model wrapper that counts forward passes."""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.call_count = 0
        self.batch_sizes = []

    def predict(self, x, action_mask=None):
        self.call_count += 1
        self.batch_sizes.append(x.shape[0])
        return self.base_model.predict(x, action_mask)

    def reset_counts(self):
        self.call_count = 0
        self.batch_sizes = []

    @property
    def total_positions_evaluated(self):
        return sum(self.batch_sizes)


class TestCacheStatsAndClear:
    """Tests for cache_stats() and clear_cache() methods."""

    @pytest.fixture
    def game(self):
        return get_game('tictactoe')

    @pytest.fixture
    def model(self, game):
        model_config = get_model_config(game.config, n_layer=2)
        model = AlphaZeroTransformer(model_config)
        model.eval()
        return model

    def test_cache_stats_initial_empty(self, game):
        """Cache stats show empty initially."""
        mcts = BatchedMCTS(game, MCTSConfig(num_simulations=10))
        hits, misses, entries = mcts.cache_stats()

        assert hits == 0
        assert misses == 0
        assert entries == 0

    def test_cache_stats_after_search(self, game, model):
        """Cache stats show entries after search."""
        mcts = BatchedMCTS(game, MCTSConfig(num_simulations=20))
        state = game.initial_state()
        states = state[np.newaxis, ...]

        mcts.search(states, model, add_noise=False)

        hits, misses, entries = mcts.cache_stats()
        # After search, we should have some entries cached
        assert entries > 0

    def test_clear_cache_resets_stats(self, game, model):
        """clear_cache() resets entries."""
        mcts = BatchedMCTS(game, MCTSConfig(num_simulations=20))
        state = game.initial_state()
        states = state[np.newaxis, ...]

        mcts.search(states, model, add_noise=False)
        _, _, entries_before = mcts.cache_stats()
        assert entries_before > 0

        mcts.clear_cache()
        hits, misses, entries = mcts.cache_stats()
        assert entries == 0
        assert hits == 0
        assert misses == 0

    def test_bayesian_cache_stats(self, game, model):
        """Bayesian MCTS also has cache stats."""
        mcts = BayesianMCTS(game, BayesianMCTSConfig(num_simulations=20))
        state = game.initial_state()
        states = state[np.newaxis, ...]

        hits, misses, entries = mcts.cache_stats()
        assert entries == 0

        mcts.search(states, model)

        _, _, entries = mcts.cache_stats()
        assert entries > 0

        mcts.clear_cache()
        _, _, entries = mcts.cache_stats()
        assert entries == 0


class TestCacheReducesNNCalls:
    """Tests that transposition table reduces neural network calls."""

    @pytest.fixture
    def game(self):
        return get_game('tictactoe')

    @pytest.fixture
    def counting_model(self, game):
        model_config = get_model_config(game.config, n_layer=2)
        base_model = AlphaZeroTransformer(model_config)
        base_model.eval()
        return CountingModel(base_model)

    def test_same_state_uses_cache(self, game, counting_model):
        """Searching the same state twice should benefit from cache."""
        mcts = BatchedMCTS(game, MCTSConfig(num_simulations=30))
        state = game.initial_state()
        states = state[np.newaxis, ...]

        # First search - fills cache
        counting_model.reset_counts()
        mcts.search(states, counting_model, add_noise=False)
        first_positions = counting_model.total_positions_evaluated

        # Second search - should use cache
        counting_model.reset_counts()
        mcts.search(states, counting_model, add_noise=False)
        second_positions = counting_model.total_positions_evaluated

        # Second search should need fewer (or equal) NN evaluations
        # The root is already cached, so we start from cached priors
        assert second_positions <= first_positions

    def test_cache_persists_across_searches(self, game, counting_model):
        """Cache entries persist and accumulate."""
        mcts = BatchedMCTS(game, MCTSConfig(num_simulations=20))
        state = game.initial_state()
        states = state[np.newaxis, ...]

        mcts.search(states, counting_model, add_noise=False)
        _, _, entries1 = mcts.cache_stats()

        # Different state but overlapping positions
        state2 = game.next_state(state, 4)  # Center move
        states2 = state2[np.newaxis, ...]
        mcts.search(states2, counting_model, add_noise=False)
        _, _, entries2 = mcts.cache_stats()

        # Should have more entries now
        assert entries2 >= entries1

    def test_clear_cache_increases_nn_calls(self, game, counting_model):
        """Clearing cache should increase NN calls on re-search."""
        mcts = BatchedMCTS(game, MCTSConfig(num_simulations=30))
        state = game.initial_state()
        states = state[np.newaxis, ...]

        # First search
        counting_model.reset_counts()
        mcts.search(states, counting_model, add_noise=False)
        first_positions = counting_model.total_positions_evaluated

        # Second search with cache
        counting_model.reset_counts()
        mcts.search(states, counting_model, add_noise=False)
        cached_positions = counting_model.total_positions_evaluated

        # Clear cache and search again
        mcts.clear_cache()
        counting_model.reset_counts()
        mcts.search(states, counting_model, add_noise=False)
        cleared_positions = counting_model.total_positions_evaluated

        # After clearing, should need more evaluations than cached search
        assert cleared_positions > cached_positions


class TestSymmetryAwareCaching:
    """Tests that symmetric positions share cache entries."""

    @pytest.fixture
    def game(self):
        return get_game('tictactoe')

    @pytest.fixture
    def counting_model(self, game):
        model_config = get_model_config(game.config, n_layer=2)
        base_model = AlphaZeroTransformer(model_config)
        base_model.eval()
        return CountingModel(base_model)

    def test_symmetric_states_share_cache_tictactoe(self, game, counting_model):
        """Symmetric TicTacToe positions should share cache entries."""
        mcts = BatchedMCTS(game, MCTSConfig(num_simulations=20))

        # Position with X in top-left corner (position 0)
        state0 = game.next_state(game.initial_state(), 0)
        states0 = state0[np.newaxis, ...]

        # Search first state
        counting_model.reset_counts()
        mcts.search(states0, counting_model, add_noise=False)
        _, _, entries_after_first = mcts.cache_stats()

        # Position with X in top-right corner (position 2) - symmetric to position 0
        state2 = game.next_state(game.initial_state(), 2)
        states2 = state2[np.newaxis, ...]

        # Search symmetric state - should reuse cache
        counting_model.reset_counts()
        mcts.search(states2, counting_model, add_noise=False)
        second_positions = counting_model.total_positions_evaluated
        _, _, entries_after_second = mcts.cache_stats()

        # The second search should use cached entries from the symmetric position
        # Entry count might increase slightly but root should be cached
        # The key test is that we get cache hits
        hits, misses, _ = mcts.cache_stats()
        assert hits > 0, "Symmetric positions should generate cache hits"

    def test_corner_positions_share_cache(self, game, counting_model):
        """All corner opening positions should share cache."""
        mcts = BatchedMCTS(game, MCTSConfig(num_simulations=10))

        # Search with corner 0
        state0 = game.next_state(game.initial_state(), 0)
        mcts.search(state0[np.newaxis, ...], counting_model, add_noise=False)
        _, _, entries0 = mcts.cache_stats()

        # Search with corner 2, 6, 8 - all symmetric to 0
        for corner in [2, 6, 8]:
            state = game.next_state(game.initial_state(), corner)
            mcts.search(state[np.newaxis, ...], counting_model, add_noise=False)

        _, _, entries_all = mcts.cache_stats()
        hits, _, _ = mcts.cache_stats()

        # Should have cache hits from symmetric positions
        assert hits > 0, "All corners are symmetric, should have cache hits"

    def test_edge_positions_share_cache(self, game, counting_model):
        """All edge opening positions should share cache."""
        mcts = BatchedMCTS(game, MCTSConfig(num_simulations=10))

        # Search with edge 1
        state1 = game.next_state(game.initial_state(), 1)
        mcts.search(state1[np.newaxis, ...], counting_model, add_noise=False)

        # Search with edges 3, 5, 7 - all symmetric to 1
        for edge in [3, 5, 7]:
            state = game.next_state(game.initial_state(), edge)
            mcts.search(state[np.newaxis, ...], counting_model, add_noise=False)

        hits, _, _ = mcts.cache_stats()
        assert hits > 0, "All edges are symmetric, should have cache hits"


class TestSymmetryConnect4:
    """Tests for Connect4 symmetry caching (left-right flip)."""

    @pytest.fixture
    def game(self):
        return get_game('connect4')

    @pytest.fixture
    def counting_model(self, game):
        model_config = get_model_config(game.config, n_layer=2)
        base_model = AlphaZeroTransformer(model_config)
        base_model.eval()
        return CountingModel(base_model)

    def test_symmetric_columns_share_cache(self, game, counting_model):
        """Left and right side columns should share cache in Connect4."""
        mcts = BatchedMCTS(game, MCTSConfig(num_simulations=10))

        # Play in column 0 (leftmost)
        state0 = game.next_state(game.initial_state(), 0)
        mcts.search(state0[np.newaxis, ...], counting_model, add_noise=False)

        # Play in column 6 (rightmost) - symmetric to column 0
        state6 = game.next_state(game.initial_state(), 6)
        mcts.search(state6[np.newaxis, ...], counting_model, add_noise=False)

        hits, _, _ = mcts.cache_stats()
        assert hits > 0, "Columns 0 and 6 are symmetric, should have cache hits"

    def test_center_column_self_symmetric(self, game, counting_model):
        """Center column is self-symmetric."""
        mcts = BatchedMCTS(game, MCTSConfig(num_simulations=10))

        # Play in center column (3)
        state = game.next_state(game.initial_state(), 3)
        counting_model.reset_counts()
        mcts.search(state[np.newaxis, ...], counting_model, add_noise=False)
        first_positions = counting_model.total_positions_evaluated

        # Search same position again - should use cache heavily
        counting_model.reset_counts()
        mcts.search(state[np.newaxis, ...], counting_model, add_noise=False)
        second_positions = counting_model.total_positions_evaluated

        # Second search should need fewer (or equal) NN evaluations
        # because root and many children are already cached
        assert second_positions <= first_positions, \
            f"Cached search should need fewer evals: {second_positions} vs {first_positions}"


class TestBayesianMCTSTranspositionTable:
    """Tests for transposition table in Bayesian MCTS."""

    @pytest.fixture
    def game(self):
        return get_game('tictactoe')

    @pytest.fixture
    def counting_model(self, game):
        model_config = get_model_config(game.config, n_layer=2)
        base_model = AlphaZeroTransformer(model_config)
        base_model.eval()
        return CountingModel(base_model)

    def test_bayesian_uses_cache(self, game, counting_model):
        """Bayesian MCTS uses transposition table."""
        mcts = BayesianMCTS(game, BayesianMCTSConfig(num_simulations=20))
        state = game.initial_state()
        states = state[np.newaxis, ...]

        # First search
        counting_model.reset_counts()
        mcts.search(states, counting_model)
        first_positions = counting_model.total_positions_evaluated

        # Second search - should benefit from cache
        counting_model.reset_counts()
        mcts.search(states, counting_model)
        second_positions = counting_model.total_positions_evaluated

        assert second_positions <= first_positions

    def test_bayesian_symmetry_caching(self, game, counting_model):
        """Bayesian MCTS uses symmetry-aware caching."""
        mcts = BayesianMCTS(game, BayesianMCTSConfig(num_simulations=10))

        # Search corner 0
        state0 = game.next_state(game.initial_state(), 0)
        mcts.search(state0[np.newaxis, ...], counting_model)

        # Search symmetric corner 8
        state8 = game.next_state(game.initial_state(), 8)
        mcts.search(state8[np.newaxis, ...], counting_model)

        hits, _, _ = mcts.cache_stats()
        assert hits > 0, "Bayesian MCTS should have cache hits for symmetric positions"


class TestDisableTranspositionTable:
    """Tests for disabling transposition table."""

    @pytest.fixture
    def game(self):
        return get_game('tictactoe')

    @pytest.fixture
    def model(self, game):
        model_config = get_model_config(game.config, n_layer=2)
        model = AlphaZeroTransformer(model_config)
        model.eval()
        return CountingModel(model)

    def test_disable_tt_puct(self, game, model):
        """Can disable transposition table in BatchedMCTS."""
        mcts = BatchedMCTS(game, MCTSConfig(num_simulations=10),
                           use_transposition_table=False)
        state = game.initial_state()
        states = state[np.newaxis, ...]

        mcts.search(states, model, add_noise=False)

        # With TT disabled, cache should remain empty
        hits, misses, entries = mcts.cache_stats()
        assert entries == 0
        assert hits == 0

    def test_disable_tt_bayesian(self, game, model):
        """Can disable transposition table in BayesianMCTS."""
        mcts = BayesianMCTS(game, BayesianMCTSConfig(num_simulations=10),
                            use_transposition_table=False)
        state = game.initial_state()
        states = state[np.newaxis, ...]

        mcts.search(states, model)

        # With TT disabled, cache should remain empty
        hits, misses, entries = mcts.cache_stats()
        assert entries == 0
        assert hits == 0

    def test_disabled_tt_no_symmetry_benefit(self, game, model):
        """Without TT, symmetric positions don't share cache."""
        mcts = BatchedMCTS(game, MCTSConfig(num_simulations=10),
                           use_transposition_table=False)

        # Search corner 0
        state0 = game.next_state(game.initial_state(), 0)
        model.reset_counts()
        mcts.search(state0[np.newaxis, ...], model, add_noise=False)
        first_count = model.total_positions_evaluated

        # Search symmetric corner 2 - won't benefit from cache
        state2 = game.next_state(game.initial_state(), 2)
        model.reset_counts()
        mcts.search(state2[np.newaxis, ...], model, add_noise=False)
        second_count = model.total_positions_evaluated

        # Without TT, both searches should have similar eval counts
        # (neither benefits from caching)
        hits, _, _ = mcts.cache_stats()
        assert hits == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
