"""
Tests for Game implementations (TicTacToe, Connect4, Go).
Includes Python implementation tests and Rust/Python equivalence tests.
"""
import numpy as np
import pytest
import torch

from nanozero.config import get_game_config
from nanozero.game import (
    TicTacToe, Connect4, Go, get_game, RUST_AVAILABLE,
)


# ============================================================================
# TicTacToe Tests
# ============================================================================

class TestTicTacToe:
    """Tests for TicTacToe game."""

    @pytest.fixture
    def game(self):
        config = get_game_config('tictactoe')
        return TicTacToe(config)

    def test_initial_state(self, game):
        """Initial state is empty 3x3 board."""
        state = game.initial_state()
        assert state.shape == (3, 3)
        assert np.all(state == 0)

    def test_current_player_initial(self, game):
        """Player 1 goes first."""
        state = game.initial_state()
        assert game.current_player(state) == 1

    def test_current_player_alternates(self, game):
        """Players alternate turns."""
        state = game.initial_state()
        assert game.current_player(state) == 1

        state = game.next_state(state, 0)
        assert game.current_player(state) == -1

        state = game.next_state(state, 1)
        assert game.current_player(state) == 1

    def test_legal_actions_initial(self, game):
        """All 9 positions are legal initially."""
        state = game.initial_state()
        legal = game.legal_actions(state)
        assert len(legal) == 9
        assert set(legal) == set(range(9))

    def test_legal_actions_after_move(self, game):
        """Occupied positions are not legal."""
        state = game.initial_state()
        state = game.next_state(state, 4)  # Center
        legal = game.legal_actions(state)
        assert len(legal) == 8
        assert 4 not in legal

    def test_next_state_places_piece(self, game):
        """next_state places current player's piece."""
        state = game.initial_state()
        new_state = game.next_state(state, 0)
        assert new_state[0, 0] == 1
        assert np.sum(new_state) == 1

    def test_next_state_is_copy(self, game):
        """next_state returns a copy, not a reference."""
        state = game.initial_state()
        new_state = game.next_state(state, 0)
        assert state is not new_state
        assert np.all(state == 0)  # Original unchanged

    def test_is_terminal_false_initially(self, game):
        """Initial state is not terminal."""
        state = game.initial_state()
        assert not game.is_terminal(state)

    def test_is_terminal_win_row(self, game):
        """Detects horizontal win."""
        state = game.initial_state()
        # X plays 0, 1, 2 (top row); O plays 3, 4
        state = game.next_state(state, 0)  # X
        state = game.next_state(state, 3)  # O
        state = game.next_state(state, 1)  # X
        state = game.next_state(state, 4)  # O
        state = game.next_state(state, 2)  # X wins
        assert game.is_terminal(state)

    def test_is_terminal_win_col(self, game):
        """Detects vertical win."""
        state = game.initial_state()
        # X plays 0, 3, 6 (left column); O plays 1, 4
        state = game.next_state(state, 0)  # X
        state = game.next_state(state, 1)  # O
        state = game.next_state(state, 3)  # X
        state = game.next_state(state, 4)  # O
        state = game.next_state(state, 6)  # X wins
        assert game.is_terminal(state)

    def test_is_terminal_win_diagonal(self, game):
        """Detects diagonal win."""
        state = game.initial_state()
        # X plays 0, 4, 8; O plays 1, 2
        state = game.next_state(state, 0)  # X
        state = game.next_state(state, 1)  # O
        state = game.next_state(state, 4)  # X
        state = game.next_state(state, 2)  # O
        state = game.next_state(state, 8)  # X wins
        assert game.is_terminal(state)

    def test_is_terminal_draw(self, game):
        """Detects draw."""
        state = game.initial_state()
        # Fill board with no winner
        moves = [0, 1, 2, 4, 3, 5, 7, 6, 8]
        for move in moves:
            state = game.next_state(state, move)
        assert game.is_terminal(state)
        assert game.terminal_reward(state) == 0.0

    def test_terminal_reward_win(self, game):
        """Winner gets +1, loser gets -1 (from current player's view)."""
        state = game.initial_state()
        state = game.next_state(state, 0)  # X
        state = game.next_state(state, 3)  # O
        state = game.next_state(state, 1)  # X
        state = game.next_state(state, 4)  # O
        state = game.next_state(state, 2)  # X wins

        # Current player is O (player -1), X won
        assert game.current_player(state) == -1
        # From O's perspective, X winning is -1
        assert game.terminal_reward(state) == -1.0

    def test_canonical_state(self, game):
        """Canonical state flips for player -1."""
        state = game.initial_state()
        state = game.next_state(state, 0)  # X at (0,0)
        # Current player is O
        canonical = game.canonical_state(state)
        # Should flip the X to -1
        assert canonical[0, 0] == -1

    def test_to_tensor(self, game):
        """Tensor maps {-1,0,1} to {0,1,2}."""
        state = game.initial_state()
        state[0, 0] = 1
        state[0, 1] = -1
        tensor = game.to_tensor(state)
        assert tensor[0] == 2  # +1 -> 2
        assert tensor[1] == 0  # -1 -> 0
        assert tensor[2] == 1  # 0 -> 1

    def test_symmetries(self, game):
        """Returns 8 symmetric transformations."""
        state = game.initial_state()
        state[0, 0] = 1
        policy = np.zeros(9)
        policy[0] = 1.0

        symmetries = game.symmetries(state, policy)
        assert len(symmetries) == 8

        # All should have the same sum
        for s, p in symmetries:
            assert np.sum(s) == 1
            assert np.sum(p) == 1.0

    def test_legal_actions_mask(self, game):
        """Mask correctly identifies legal actions."""
        state = game.initial_state()
        state = game.next_state(state, 4)  # Center
        mask = game.legal_actions_mask(state)
        assert mask[4] == 0.0
        assert np.sum(mask) == 8


# ============================================================================
# Connect4 Tests
# ============================================================================

class TestConnect4:
    """Tests for Connect4 game."""

    @pytest.fixture
    def game(self):
        config = get_game_config('connect4')
        return Connect4(config)

    def test_initial_state(self, game):
        """Initial state is empty 6x7 board."""
        state = game.initial_state()
        assert state.shape == (6, 7)
        assert np.all(state == 0)

    def test_current_player_initial(self, game):
        """Player 1 goes first."""
        state = game.initial_state()
        assert game.current_player(state) == 1

    def test_legal_actions_initial(self, game):
        """All 7 columns are legal initially."""
        state = game.initial_state()
        legal = game.legal_actions(state)
        assert len(legal) == 7
        assert set(legal) == set(range(7))

    def test_next_state_drops_to_bottom(self, game):
        """Pieces drop to the bottom."""
        state = game.initial_state()
        state = game.next_state(state, 0)
        assert state[5, 0] == 1  # Bottom row

    def test_pieces_stack(self, game):
        """Pieces stack on top of each other."""
        state = game.initial_state()
        state = game.next_state(state, 0)  # X at (5, 0)
        state = game.next_state(state, 0)  # O at (4, 0)
        assert state[5, 0] == 1
        assert state[4, 0] == -1

    def test_full_column_illegal(self, game):
        """Full column becomes illegal."""
        state = game.initial_state()
        for _ in range(6):
            state = game.next_state(state, 0)
        legal = game.legal_actions(state)
        assert 0 not in legal

    def test_horizontal_win(self, game):
        """Detects horizontal win."""
        state = game.initial_state()
        # X: 0,1,2,3; O: 0,1,2 (stacking)
        state = game.next_state(state, 0)  # X
        state = game.next_state(state, 0)  # O
        state = game.next_state(state, 1)  # X
        state = game.next_state(state, 1)  # O
        state = game.next_state(state, 2)  # X
        state = game.next_state(state, 2)  # O
        state = game.next_state(state, 3)  # X wins
        assert game.is_terminal(state)

    def test_vertical_win(self, game):
        """Detects vertical win."""
        state = game.initial_state()
        # X: column 0 four times; O: column 1
        state = game.next_state(state, 0)  # X
        state = game.next_state(state, 1)  # O
        state = game.next_state(state, 0)  # X
        state = game.next_state(state, 1)  # O
        state = game.next_state(state, 0)  # X
        state = game.next_state(state, 1)  # O
        state = game.next_state(state, 0)  # X wins
        assert game.is_terminal(state)

    def test_diagonal_win(self, game):
        """Detects diagonal win."""
        state = game.initial_state()
        # Build a diagonal for X
        # Col 0: X
        # Col 1: O, X
        # Col 2: O, O, X
        # Col 3: O, O, O, X
        state = game.next_state(state, 0)  # X at (5,0)
        state = game.next_state(state, 1)  # O at (5,1)
        state = game.next_state(state, 1)  # X at (4,1)
        state = game.next_state(state, 2)  # O at (5,2)
        state = game.next_state(state, 2)  # X at (4,2)
        state = game.next_state(state, 3)  # O at (5,3)
        state = game.next_state(state, 2)  # X at (3,2)
        state = game.next_state(state, 3)  # O at (4,3)
        state = game.next_state(state, 3)  # X at (3,3)
        state = game.next_state(state, 4)  # O at (5,4)
        state = game.next_state(state, 3)  # X at (2,3) - wins diagonally
        assert game.is_terminal(state)

    def test_symmetries(self, game):
        """Returns 2 symmetric transformations."""
        state = game.initial_state()
        state[5, 0] = 1
        policy = np.zeros(7)
        policy[0] = 1.0

        symmetries = game.symmetries(state, policy)
        assert len(symmetries) == 2

        # Second should be flipped
        s2, p2 = symmetries[1]
        assert s2[5, 6] == 1  # Flipped horizontally
        assert p2[6] == 1.0


# ============================================================================
# Go Tests
# ============================================================================

class TestGo:
    """Tests for Go game."""

    @pytest.fixture
    def game(self):
        config = get_game_config('go9x9')
        return Go(config)

    def test_initial_state(self, game):
        """Initial state is empty 9x9 board with metadata row."""
        state = game.initial_state()
        assert state.shape == (10, 9)  # 9x9 + 1 metadata row
        assert np.all(state[:9, :] == 0)

    def test_current_player_initial(self, game):
        """Black (1) plays first."""
        state = game.initial_state()
        assert game.current_player(state) == 1

    def test_legal_actions_initial(self, game):
        """All 81 positions + pass are legal initially."""
        state = game.initial_state()
        legal = game.legal_actions(state)
        assert len(legal) == 82  # 81 positions + pass

    def test_pass_is_always_legal(self, game):
        """Pass (action 81) is always legal."""
        state = game.initial_state()
        legal = game.legal_actions(state)
        assert 81 in legal

    def test_next_state_places_stone(self, game):
        """Playing places a stone."""
        state = game.initial_state()
        state = game.next_state(state, 0)  # Black at (0,0)
        assert state[0, 0] == 1

    def test_capture_single_stone(self, game):
        """Can capture a single stone."""
        state = game.initial_state()
        # Surround white stone at (1,1)
        state = game.next_state(state, 0*9 + 1)   # Black at (0,1)
        state = game.next_state(state, 1*9 + 1)   # White at (1,1)
        state = game.next_state(state, 1*9 + 0)   # Black at (1,0)
        state = game.next_state(state, 81)        # White passes
        state = game.next_state(state, 1*9 + 2)   # Black at (1,2)
        state = game.next_state(state, 81)        # White passes
        state = game.next_state(state, 2*9 + 1)   # Black at (2,1) - captures

        assert state[1, 1] == 0  # White stone captured

    def test_suicide_illegal(self, game):
        """Suicide is illegal."""
        state = game.initial_state()
        # Create a situation where playing at (0,0) would be suicide
        state = game.next_state(state, 0*9 + 1)   # Black at (0,1)
        state = game.next_state(state, 81)        # White passes
        state = game.next_state(state, 1*9 + 0)   # Black at (1,0)
        state = game.next_state(state, 81)        # White passes

        # Now white cannot play at (0,0) - it would be suicide
        legal = game.legal_actions(state)
        # (0,0) = action 0
        # Actually, black just played, so it's white's turn
        # White at (0,0) would be suicide
        assert 0 not in legal

    def test_terminal_after_two_passes(self, game):
        """Game ends after two consecutive passes."""
        state = game.initial_state()
        assert not game.is_terminal(state)

        state = game.next_state(state, 81)  # Black passes
        assert not game.is_terminal(state)

        state = game.next_state(state, 81)  # White passes
        assert game.is_terminal(state)

    def test_symmetries(self, game):
        """Returns 8 symmetric transformations."""
        state = game.initial_state()
        state[0, 0] = 1
        policy = np.zeros(82)
        policy[0] = 0.5
        policy[81] = 0.5  # Pass

        symmetries = game.symmetries(state, policy)
        assert len(symmetries) == 8

        # Pass probability should be preserved
        for s, p in symmetries:
            assert p[81] == 0.5


# ============================================================================
# Rust/Python Equivalence Tests
# ============================================================================

@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestRustPythonEquivalence:
    """Tests that Rust and Python implementations produce identical results."""

    def test_tictactoe_equivalence(self):
        """TicTacToe: Rust matches Python."""
        py_game = get_game('tictactoe', use_rust=False)
        rs_game = get_game('tictactoe', use_rust=True)

        # Test initial state
        py_state = py_game.initial_state()
        rs_state = rs_game.initial_state()
        np.testing.assert_array_equal(py_state, rs_state)

        # Play a series of moves
        np.random.seed(42)
        for _ in range(100):
            py_state = py_game.initial_state()
            rs_state = rs_game.initial_state()

            while not py_game.is_terminal(py_state):
                # Check current player
                assert py_game.current_player(py_state) == rs_game.current_player(rs_state)

                # Check legal actions
                py_legal = set(py_game.legal_actions(py_state))
                rs_legal = set(rs_game.legal_actions(rs_state))
                assert py_legal == rs_legal

                # Check legal actions mask
                np.testing.assert_array_equal(
                    py_game.legal_actions_mask(py_state),
                    rs_game.legal_actions_mask(rs_state)
                )

                # Make a random move
                action = np.random.choice(list(py_legal))
                py_state = py_game.next_state(py_state, action)
                rs_state = rs_game.next_state(rs_state, action)
                np.testing.assert_array_equal(py_state, rs_state)

            # Check terminal state
            assert py_game.is_terminal(py_state) == rs_game.is_terminal(rs_state)
            assert py_game.terminal_reward(py_state) == rs_game.terminal_reward(rs_state)

    def test_connect4_equivalence(self):
        """Connect4: Rust matches Python."""
        py_game = get_game('connect4', use_rust=False)
        rs_game = get_game('connect4', use_rust=True)

        np.random.seed(42)
        for _ in range(50):
            py_state = py_game.initial_state()
            rs_state = rs_game.initial_state()

            while not py_game.is_terminal(py_state):
                assert py_game.current_player(py_state) == rs_game.current_player(rs_state)

                py_legal = set(py_game.legal_actions(py_state))
                rs_legal = set(rs_game.legal_actions(rs_state))
                assert py_legal == rs_legal

                action = np.random.choice(list(py_legal))
                py_state = py_game.next_state(py_state, action)
                rs_state = rs_game.next_state(rs_state, action)
                np.testing.assert_array_equal(py_state, rs_state)

            assert py_game.terminal_reward(py_state) == rs_game.terminal_reward(rs_state)

    def test_go_equivalence(self):
        """Go: Rust matches Python."""
        py_game = get_game('go9x9', use_rust=False)
        rs_game = get_game('go9x9', use_rust=True)

        np.random.seed(42)
        for _ in range(10):  # Fewer games, Go is slower
            py_state = py_game.initial_state()
            rs_state = rs_game.initial_state()

            move_count = 0
            while not py_game.is_terminal(py_state) and move_count < 100:
                assert py_game.current_player(py_state) == rs_game.current_player(rs_state)

                py_legal = set(py_game.legal_actions(py_state))
                rs_legal = set(rs_game.legal_actions(rs_state))
                assert py_legal == rs_legal, f"Move {move_count}: legal actions differ"

                action = np.random.choice(list(py_legal))
                py_state = py_game.next_state(py_state, action)
                rs_state = rs_game.next_state(rs_state, action)
                np.testing.assert_array_equal(py_state, rs_state)
                move_count += 1

            assert py_game.is_terminal(py_state) == rs_game.is_terminal(rs_state)
            if py_game.is_terminal(py_state):
                assert py_game.terminal_reward(py_state) == rs_game.terminal_reward(rs_state)

    def test_canonical_state_equivalence(self):
        """Canonical state computation matches."""
        for game_name in ['tictactoe', 'connect4']:
            py_game = get_game(game_name, use_rust=False)
            rs_game = get_game(game_name, use_rust=True)

            state = py_game.initial_state()
            # Make a move so we're player -1
            action = py_game.legal_actions(state)[0]
            py_state = py_game.next_state(state, action)
            rs_state = rs_game.next_state(state, action)

            py_canonical = py_game.canonical_state(py_state)
            rs_canonical = rs_game.canonical_state(rs_state)
            np.testing.assert_array_equal(py_canonical, rs_canonical)

    def test_to_tensor_equivalence(self):
        """Tensor conversion matches."""
        for game_name in ['tictactoe', 'connect4']:
            py_game = get_game(game_name, use_rust=False)
            rs_game = get_game(game_name, use_rust=True)

            state = py_game.initial_state()
            # Make some moves
            for _ in range(3):
                action = py_game.legal_actions(state)[0]
                state = py_game.next_state(state, action)

            py_tensor = py_game.to_tensor(state)
            rs_tensor = rs_game.to_tensor(state)
            torch.testing.assert_close(py_tensor, rs_tensor)


# ============================================================================
# get_game Factory Tests
# ============================================================================

class TestGetGame:
    """Tests for get_game factory function."""

    def test_get_tictactoe(self):
        """Can get TicTacToe game."""
        game = get_game('tictactoe', use_rust=False)
        assert isinstance(game, TicTacToe)

    def test_get_connect4(self):
        """Can get Connect4 game."""
        game = get_game('connect4', use_rust=False)
        assert isinstance(game, Connect4)

    def test_get_go9x9(self):
        """Can get 9x9 Go game."""
        game = get_game('go9x9', use_rust=False)
        assert isinstance(game, Go)
        assert game.height == 9
        assert game.width == 9

    def test_get_go19x19(self):
        """Can get 19x19 Go game."""
        game = get_game('go19x19', use_rust=False)
        assert isinstance(game, Go)
        assert game.height == 19
        assert game.width == 19

    def test_unknown_game_raises(self):
        """Unknown game name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown game"):
            get_game('unknown_game')

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
    def test_rust_preference(self):
        """With use_rust=True and Rust available, uses Rust implementation."""
        from nanozero.game import RustTicTacToeWrapper, RustConnect4Wrapper, RustGoWrapper

        game = get_game('tictactoe', use_rust=True)
        assert isinstance(game, RustTicTacToeWrapper)

        game = get_game('connect4', use_rust=True)
        assert isinstance(game, RustConnect4Wrapper)

        game = get_game('go9x9', use_rust=True)
        assert isinstance(game, RustGoWrapper)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
