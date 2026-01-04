//! TicTacToe game implementation.
//!
//! 3x3 board, 9 possible actions, 8 symmetries.

use super::state::{GameState, TicTacToeState};
use super::Game;

/// TicTacToe game.
pub struct TicTacToe;

impl TicTacToe {
    pub fn new() -> Self {
        Self
    }

    /// Check for a winner on the board.
    ///
    /// Returns the winner (1 or -1) or 0 if no winner.
    fn check_winner(state: &TicTacToeState) -> i8 {
        let b = &state.board;

        // Check rows
        for row in 0..3 {
            let start = row * 3;
            if b[start] != 0 && b[start] == b[start + 1] && b[start + 1] == b[start + 2] {
                return b[start];
            }
        }

        // Check columns
        for col in 0..3 {
            if b[col] != 0 && b[col] == b[col + 3] && b[col + 3] == b[col + 6] {
                return b[col];
            }
        }

        // Check diagonals
        if b[0] != 0 && b[0] == b[4] && b[4] == b[8] {
            return b[0];
        }
        if b[2] != 0 && b[2] == b[4] && b[4] == b[6] {
            return b[2];
        }

        0
    }

    /// Apply a rotation transformation to an action.
    fn rotate_action(action: usize, times: usize) -> usize {
        let mut row = action / 3;
        let mut col = action % 3;
        for _ in 0..times {
            let new_row = col;
            let new_col = 2 - row;
            row = new_row;
            col = new_col;
        }
        row * 3 + col
    }

    /// Apply a horizontal flip to an action.
    fn flip_action(action: usize) -> usize {
        let row = action / 3;
        let col = action % 3;
        row * 3 + (2 - col)
    }

    /// Rotate board 90 degrees clockwise.
    fn rotate_board(state: &TicTacToeState) -> TicTacToeState {
        let mut new_state = TicTacToeState::new();
        for row in 0..3 {
            for col in 0..3 {
                new_state.set(col, 2 - row, state.get(row, col));
            }
        }
        new_state
    }

    /// Flip board horizontally.
    fn flip_board(state: &TicTacToeState) -> TicTacToeState {
        let mut new_state = TicTacToeState::new();
        for row in 0..3 {
            for col in 0..3 {
                new_state.set(row, 2 - col, state.get(row, col));
            }
        }
        new_state
    }
}

impl Default for TicTacToe {
    fn default() -> Self {
        Self::new()
    }
}

impl Game for TicTacToe {
    fn initial_state(&self) -> GameState {
        GameState::TicTacToe(TicTacToeState::new())
    }

    fn current_player(&self, state: &GameState) -> i8 {
        let s = state.as_tictactoe();
        let x_count = s.board.iter().filter(|&&v| v == 1).count();
        let o_count = s.board.iter().filter(|&&v| v == -1).count();
        if x_count == o_count {
            1 // X goes first
        } else {
            -1
        }
    }

    fn legal_actions(&self, state: &GameState) -> Vec<u16> {
        let s = state.as_tictactoe();
        (0..9)
            .filter(|&i| s.board[i] == 0)
            .map(|i| i as u16)
            .collect()
    }

    fn next_state(&self, state: &GameState, action: u16) -> GameState {
        let s = state.as_tictactoe();
        let player = self.current_player(state);
        let mut new_state = s.clone();
        new_state.board[action as usize] = player;
        GameState::TicTacToe(new_state)
    }

    fn is_terminal(&self, state: &GameState) -> bool {
        let s = state.as_tictactoe();
        Self::check_winner(s) != 0 || s.board.iter().all(|&v| v != 0)
    }

    fn terminal_reward(&self, state: &GameState) -> f32 {
        let s = state.as_tictactoe();
        let winner = Self::check_winner(s);
        let current = self.current_player(state);
        if winner == 0 {
            0.0 // Draw
        } else if winner == current {
            1.0 // Current player won
        } else {
            -1.0 // Current player lost
        }
    }

    fn canonical_state(&self, state: &GameState) -> GameState {
        let s = state.as_tictactoe();
        let player = self.current_player(state);
        if player == 1 {
            state.clone()
        } else {
            // Flip all pieces
            let mut new_state = TicTacToeState::new();
            for i in 0..9 {
                new_state.board[i] = -s.board[i];
            }
            GameState::TicTacToe(new_state)
        }
    }

    fn to_tensor(&self, state: &GameState) -> Vec<i64> {
        let s = state.as_tictactoe();
        // Map {-1, 0, 1} -> {0, 1, 2}
        s.board.iter().map(|&v| (v + 1) as i64).collect()
    }

    fn symmetries(&self, state: &GameState, policy: &[f32]) -> Vec<(GameState, Vec<f32>)> {
        let s = state.as_tictactoe();
        let mut result = Vec::with_capacity(8);

        // Generate all 8 symmetries: 4 rotations × 2 (with/without flip)
        let mut current = s.clone();
        for rot in 0..4 {
            // Without flip
            let mut new_policy = vec![0.0f32; 9];
            for (i, &p) in policy.iter().enumerate() {
                let new_action = Self::rotate_action(i, rot);
                new_policy[new_action] = p;
            }
            result.push((GameState::TicTacToe(current.clone()), new_policy));

            // With flip
            let flipped = Self::flip_board(&current);
            let mut flipped_policy = vec![0.0f32; 9];
            for (i, &p) in policy.iter().enumerate() {
                let rotated = Self::rotate_action(i, rot);
                let new_action = Self::flip_action(rotated);
                flipped_policy[new_action] = p;
            }
            result.push((GameState::TicTacToe(flipped), flipped_policy));

            // Rotate for next iteration
            current = Self::rotate_board(&current);
        }

        result
    }

    fn action_size(&self) -> usize {
        9
    }

    fn board_size(&self) -> usize {
        9
    }

    fn board_dims(&self) -> (usize, usize) {
        (3, 3)
    }

    fn state_from_slice(&self, data: &[i8]) -> GameState {
        let mut state = TicTacToeState::new();
        for (i, &v) in data.iter().take(9).enumerate() {
            state.board[i] = v;
        }
        GameState::TicTacToe(state)
    }

    fn render(&self, state: &GameState) -> String {
        let s = state.as_tictactoe();
        let mut result = String::new();
        for row in 0..3 {
            for col in 0..3 {
                let c = match s.get(row, col) {
                    1 => 'X',
                    -1 => 'O',
                    _ => '.',
                };
                result.push(c);
                if col < 2 {
                    result.push(' ');
                }
            }
            result.push('\n');
        }
        result
    }

    fn num_symmetries(&self) -> usize {
        8 // 4 rotations × 2 (with/without flip)
    }

    fn map_action(&self, action: u16, symmetry_idx: usize) -> u16 {
        // Symmetry indices match symmetries() order:
        // 0,2,4,6 = rotations 0,1,2,3 without flip
        // 1,3,5,7 = rotations 0,1,2,3 with flip
        let rotations = symmetry_idx / 2;
        let flipped = symmetry_idx % 2 == 1;

        let mut a = action as usize;
        // Apply rotations
        a = Self::rotate_action(a, rotations);
        // Apply flip if needed
        if flipped {
            a = Self::flip_action(a);
        }
        a as u16
    }

    fn unmap_action(&self, action: u16, symmetry_idx: usize) -> u16 {
        // Inverse of map_action: undo flip, then undo rotation
        let rotations = symmetry_idx / 2;
        let flipped = symmetry_idx % 2 == 1;

        let mut a = action as usize;
        // Undo flip first (flip is self-inverse)
        if flipped {
            a = Self::flip_action(a);
        }
        // Undo rotation (rotate by 4 - rotations)
        a = Self::rotate_action(a, (4 - rotations) % 4);
        a as u16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let game = TicTacToe::new();
        let state = game.initial_state();
        let s = state.as_tictactoe();
        assert!(s.board.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_current_player() {
        let game = TicTacToe::new();
        let state = game.initial_state();
        assert_eq!(game.current_player(&state), 1); // X goes first

        let state = game.next_state(&state, 0);
        assert_eq!(game.current_player(&state), -1); // O's turn
    }

    #[test]
    fn test_legal_actions() {
        let game = TicTacToe::new();
        let state = game.initial_state();
        assert_eq!(game.legal_actions(&state).len(), 9);

        let state = game.next_state(&state, 0);
        assert_eq!(game.legal_actions(&state).len(), 8);
        assert!(!game.legal_actions(&state).contains(&0));
    }

    #[test]
    fn test_winner_detection() {
        let game = TicTacToe::new();
        let mut state = game.initial_state();

        // X plays: 0, 1, 2 (top row)
        // O plays: 3, 4
        state = game.next_state(&state, 0); // X
        state = game.next_state(&state, 3); // O
        state = game.next_state(&state, 1); // X
        state = game.next_state(&state, 4); // O
        state = game.next_state(&state, 2); // X wins

        assert!(game.is_terminal(&state));
        // Current player is O, X won, so reward is -1
        assert!((game.terminal_reward(&state) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_draw() {
        let game = TicTacToe::new();
        let mut state = game.initial_state();

        // X O X
        // X X O
        // O X O
        let moves = [0, 1, 2, 5, 3, 6, 4, 8, 7];
        for &m in &moves {
            state = game.next_state(&state, m);
        }

        assert!(game.is_terminal(&state));
        assert!((game.terminal_reward(&state) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_symmetries() {
        let game = TicTacToe::new();
        let state = game.initial_state();
        let policy = vec![0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.0];

        let syms = game.symmetries(&state, &policy);
        assert_eq!(syms.len(), 8);

        // All policies should sum to the same value
        let sum: f32 = policy.iter().sum();
        for (_, p) in &syms {
            let s: f32 = p.iter().sum();
            assert!((s - sum).abs() < 1e-6);
        }
    }

    #[test]
    fn test_canonical_state() {
        let game = TicTacToe::new();
        let mut state = game.initial_state();

        // X plays at 0
        state = game.next_state(&state, 0);
        // Now it's O's turn, canonical should flip
        let canonical = game.canonical_state(&state);
        let c = canonical.as_tictactoe();
        assert_eq!(c.board[0], -1); // X's piece becomes -1
    }

    #[test]
    fn test_map_unmap_action_roundtrip() {
        let game = TicTacToe::new();
        // For all symmetries and actions, map then unmap should give identity
        for sym_idx in 0..8 {
            for action in 0..9u16 {
                let mapped = game.map_action(action, sym_idx);
                let unmapped = game.unmap_action(mapped, sym_idx);
                assert_eq!(
                    unmapped, action,
                    "map_action/unmap_action roundtrip failed for sym={}, action={}",
                    sym_idx, action
                );
            }
        }
    }

    #[test]
    fn test_map_action_symmetry_consistency() {
        let game = TicTacToe::new();
        // Create a state with a piece at position 0 (top-left)
        let mut state = game.initial_state();
        state = game.next_state(&state, 0);

        let policy = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let syms = game.symmetries(&state, &policy);

        // For each symmetry, map_action(0) should point to where the piece moved
        for (sym_idx, (sym_state, sym_policy)) in syms.iter().enumerate() {
            let mapped_action = game.map_action(0, sym_idx);
            // The symmetric policy should have 1.0 at mapped_action
            assert!(
                (sym_policy[mapped_action as usize] - 1.0).abs() < 1e-6,
                "sym_idx={}: map_action(0)={} but policy has 1.0 at {:?}",
                sym_idx,
                mapped_action,
                sym_policy
                    .iter()
                    .position(|&p| (p - 1.0).abs() < 1e-6)
            );
            // The symmetric state should have the piece at mapped_action
            let s = sym_state.as_tictactoe();
            assert_eq!(
                s.board[mapped_action as usize], 1,
                "sym_idx={}: piece not at mapped_action={}",
                sym_idx, mapped_action
            );
        }
    }

    #[test]
    fn test_canonical_symmetry_index() {
        use crate::game::compute_hash;

        let game = TicTacToe::new();
        // Create a state with a piece at corner
        let mut state = game.initial_state();
        state = game.next_state(&state, 0);

        let (sym_idx, canonical_hash) = game.canonical_symmetry_index(&state);

        // Verify this is the minimum hash among symmetries
        let policy = vec![0.0f32; 9];
        let syms = game.symmetries(&state, &policy);
        for (idx, (sym_state, _)) in syms.iter().enumerate() {
            let hash = compute_hash(sym_state);
            if idx == sym_idx {
                assert_eq!(hash, canonical_hash);
            } else {
                assert!(hash >= canonical_hash, "Found smaller hash at idx {}", idx);
            }
        }
    }

    #[test]
    fn test_symmetric_states_same_canonical_hash() {
        let game = TicTacToe::new();

        // Create different starting positions that are symmetric
        // Corner positions: 0, 2, 6, 8 are all equivalent under symmetry
        for &action in &[0u16, 2, 6, 8] {
            let mut state = game.initial_state();
            state = game.next_state(&state, action);
            let (_, hash) = game.canonical_symmetry_index(&state);

            // All corner openings should have the same canonical hash
            let mut state0 = game.initial_state();
            state0 = game.next_state(&state0, 0);
            let (_, hash0) = game.canonical_symmetry_index(&state0);

            assert_eq!(
                hash, hash0,
                "Corner opening at {} should have same canonical hash as opening at 0",
                action
            );
        }

        // Edge positions: 1, 3, 5, 7 are all equivalent under symmetry
        for &action in &[1u16, 3, 5, 7] {
            let mut state = game.initial_state();
            state = game.next_state(&state, action);
            let (_, hash) = game.canonical_symmetry_index(&state);

            let mut state1 = game.initial_state();
            state1 = game.next_state(&state1, 1);
            let (_, hash1) = game.canonical_symmetry_index(&state1);

            assert_eq!(
                hash, hash1,
                "Edge opening at {} should have same canonical hash as opening at 1",
                action
            );
        }
    }

    #[test]
    fn test_distinct_positions_different_hashes() {
        let game = TicTacToe::new();

        // Corner opening (position 0)
        let mut corner = game.initial_state();
        corner = game.next_state(&corner, 0);
        let (_, corner_hash) = game.canonical_symmetry_index(&corner);

        // Edge opening (position 1)
        let mut edge = game.initial_state();
        edge = game.next_state(&edge, 1);
        let (_, edge_hash) = game.canonical_symmetry_index(&edge);

        // Center opening (position 4)
        let mut center = game.initial_state();
        center = game.next_state(&center, 4);
        let (_, center_hash) = game.canonical_symmetry_index(&center);

        // All three are truly distinct positions
        assert_ne!(corner_hash, edge_hash, "Corner and edge should have different hashes");
        assert_ne!(corner_hash, center_hash, "Corner and center should have different hashes");
        assert_ne!(edge_hash, center_hash, "Edge and center should have different hashes");
    }
}
