//! Connect4 game implementation.
//!
//! 6x7 board, 7 possible actions (columns), 2 symmetries.

use super::state::{Connect4State, GameState};
use super::Game;

/// Connect4 game.
pub struct Connect4;

impl Connect4 {
    pub fn new() -> Self {
        Self
    }

    /// Find the row where a piece would land in a column.
    ///
    /// Returns None if the column is full.
    fn find_drop_row(state: &Connect4State, col: usize) -> Option<usize> {
        for row in (0..Connect4State::HEIGHT).rev() {
            if state.get(row, col) == 0 {
                return Some(row);
            }
        }
        None
    }

    /// Check for a winner.
    ///
    /// Returns the winner (1 or -1) or 0 if no winner.
    fn check_winner(state: &Connect4State) -> i8 {
        // Check horizontal
        for row in 0..Connect4State::HEIGHT {
            for col in 0..Connect4State::WIDTH - 3 {
                let v = state.get(row, col);
                if v != 0
                    && v == state.get(row, col + 1)
                    && v == state.get(row, col + 2)
                    && v == state.get(row, col + 3)
                {
                    return v;
                }
            }
        }

        // Check vertical
        for row in 0..Connect4State::HEIGHT - 3 {
            for col in 0..Connect4State::WIDTH {
                let v = state.get(row, col);
                if v != 0
                    && v == state.get(row + 1, col)
                    && v == state.get(row + 2, col)
                    && v == state.get(row + 3, col)
                {
                    return v;
                }
            }
        }

        // Check diagonal (down-right)
        for row in 0..Connect4State::HEIGHT - 3 {
            for col in 0..Connect4State::WIDTH - 3 {
                let v = state.get(row, col);
                if v != 0
                    && v == state.get(row + 1, col + 1)
                    && v == state.get(row + 2, col + 2)
                    && v == state.get(row + 3, col + 3)
                {
                    return v;
                }
            }
        }

        // Check diagonal (down-left)
        for row in 0..Connect4State::HEIGHT - 3 {
            for col in 3..Connect4State::WIDTH {
                let v = state.get(row, col);
                if v != 0
                    && v == state.get(row + 1, col - 1)
                    && v == state.get(row + 2, col - 2)
                    && v == state.get(row + 3, col - 3)
                {
                    return v;
                }
            }
        }

        0
    }

    /// Flip the board horizontally.
    fn flip_board(state: &Connect4State) -> Connect4State {
        let mut new_state = Connect4State::new();
        for row in 0..Connect4State::HEIGHT {
            for col in 0..Connect4State::WIDTH {
                new_state.set(row, Connect4State::WIDTH - 1 - col, state.get(row, col));
            }
        }
        new_state
    }
}

impl Default for Connect4 {
    fn default() -> Self {
        Self::new()
    }
}

impl Game for Connect4 {
    fn initial_state(&self) -> GameState {
        GameState::Connect4(Connect4State::new())
    }

    fn current_player(&self, state: &GameState) -> i8 {
        let s = state.as_connect4();
        let p1_count = s.board.iter().filter(|&&v| v == 1).count();
        let p2_count = s.board.iter().filter(|&&v| v == -1).count();
        if p1_count == p2_count {
            1 // Player 1 goes first
        } else {
            -1
        }
    }

    fn legal_actions(&self, state: &GameState) -> Vec<u16> {
        let s = state.as_connect4();
        (0..Connect4State::WIDTH)
            .filter(|&col| s.get(0, col) == 0)
            .map(|col| col as u16)
            .collect()
    }

    fn next_state(&self, state: &GameState, action: u16) -> GameState {
        let s = state.as_connect4();
        let player = self.current_player(state);
        let col = action as usize;

        let mut new_state = s.clone();
        if let Some(row) = Self::find_drop_row(s, col) {
            new_state.set(row, col, player);
        }
        GameState::Connect4(new_state)
    }

    fn is_terminal(&self, state: &GameState) -> bool {
        let s = state.as_connect4();
        Self::check_winner(s) != 0 || self.legal_actions(state).is_empty()
    }

    fn terminal_reward(&self, state: &GameState) -> f32 {
        let s = state.as_connect4();
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
        let s = state.as_connect4();
        let player = self.current_player(state);
        if player == 1 {
            state.clone()
        } else {
            let mut new_state = Connect4State::new();
            for i in 0..42 {
                new_state.board[i] = -s.board[i];
            }
            GameState::Connect4(new_state)
        }
    }

    fn to_tensor(&self, state: &GameState) -> Vec<i64> {
        let s = state.as_connect4();
        s.board.iter().map(|&v| (v + 1) as i64).collect()
    }

    fn symmetries(&self, state: &GameState, policy: &[f32]) -> Vec<(GameState, Vec<f32>)> {
        let s = state.as_connect4();
        let mut result = Vec::with_capacity(2);

        // Identity
        result.push((state.clone(), policy.to_vec()));

        // Horizontal flip
        let flipped = Self::flip_board(s);
        let mut flipped_policy = vec![0.0f32; Connect4State::WIDTH];
        for (col, &p) in policy.iter().enumerate() {
            flipped_policy[Connect4State::WIDTH - 1 - col] = p;
        }
        result.push((GameState::Connect4(flipped), flipped_policy));

        result
    }

    fn action_size(&self) -> usize {
        Connect4State::WIDTH
    }

    fn board_size(&self) -> usize {
        Connect4State::HEIGHT * Connect4State::WIDTH
    }

    fn board_dims(&self) -> (usize, usize) {
        (Connect4State::HEIGHT, Connect4State::WIDTH)
    }

    fn state_from_slice(&self, data: &[i8]) -> GameState {
        let mut state = Connect4State::new();
        for (i, &v) in data.iter().take(42).enumerate() {
            state.board[i] = v;
        }
        GameState::Connect4(state)
    }

    fn render(&self, state: &GameState) -> String {
        let s = state.as_connect4();
        let mut result = String::new();

        // Column numbers
        for col in 0..Connect4State::WIDTH {
            result.push_str(&format!("{} ", col));
        }
        result.push('\n');

        // Board
        for row in 0..Connect4State::HEIGHT {
            for col in 0..Connect4State::WIDTH {
                let c = match s.get(row, col) {
                    1 => 'X',
                    -1 => 'O',
                    _ => '.',
                };
                result.push(c);
                result.push(' ');
            }
            result.push('\n');
        }
        result
    }

    fn num_symmetries(&self) -> usize {
        2 // Identity and horizontal flip
    }

    fn map_action(&self, action: u16, symmetry_idx: usize) -> u16 {
        if symmetry_idx == 0 {
            action // Identity
        } else {
            // Horizontal flip: column i -> column (WIDTH-1-i)
            (Connect4State::WIDTH - 1 - action as usize) as u16
        }
    }

    fn unmap_action(&self, action: u16, symmetry_idx: usize) -> u16 {
        // Flip is self-inverse
        self.map_action(action, symmetry_idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let game = Connect4::new();
        let state = game.initial_state();
        let s = state.as_connect4();
        assert!(s.board.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_current_player() {
        let game = Connect4::new();
        let state = game.initial_state();
        assert_eq!(game.current_player(&state), 1);

        let state = game.next_state(&state, 0);
        assert_eq!(game.current_player(&state), -1);
    }

    #[test]
    fn test_drop() {
        let game = Connect4::new();
        let mut state = game.initial_state();

        // Drop in column 0 three times
        state = game.next_state(&state, 0); // P1
        state = game.next_state(&state, 0); // P2
        state = game.next_state(&state, 0); // P1

        let s = state.as_connect4();
        assert_eq!(s.get(5, 0), 1); // Bottom
        assert_eq!(s.get(4, 0), -1);
        assert_eq!(s.get(3, 0), 1);
    }

    #[test]
    fn test_horizontal_win() {
        let game = Connect4::new();
        let mut state = game.initial_state();

        // P1 plays: 0, 1, 2, 3 (bottom row)
        // P2 plays: 0, 1, 2 (second row, stacking)
        state = game.next_state(&state, 0); // P1
        state = game.next_state(&state, 0); // P2
        state = game.next_state(&state, 1); // P1
        state = game.next_state(&state, 1); // P2
        state = game.next_state(&state, 2); // P1
        state = game.next_state(&state, 2); // P2
        state = game.next_state(&state, 3); // P1 wins

        assert!(game.is_terminal(&state));
        // Current player is P2, P1 won
        assert!((game.terminal_reward(&state) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_vertical_win() {
        let game = Connect4::new();
        let mut state = game.initial_state();

        // P1 plays column 0 four times, P2 plays column 1
        state = game.next_state(&state, 0); // P1
        state = game.next_state(&state, 1); // P2
        state = game.next_state(&state, 0); // P1
        state = game.next_state(&state, 1); // P2
        state = game.next_state(&state, 0); // P1
        state = game.next_state(&state, 1); // P2
        state = game.next_state(&state, 0); // P1 wins

        assert!(game.is_terminal(&state));
    }

    #[test]
    fn test_legal_actions() {
        let game = Connect4::new();
        let state = game.initial_state();
        assert_eq!(game.legal_actions(&state).len(), 7);

        // Fill column 0
        let mut state = state;
        for _ in 0..6 {
            if game.legal_actions(&state).contains(&0) {
                state = game.next_state(&state, 0);
            }
        }

        // Column 0 should be illegal
        let legal = game.legal_actions(&state);
        assert_eq!(legal.len(), 6);
        assert!(!legal.contains(&0));
    }

    #[test]
    fn test_symmetries() {
        let game = Connect4::new();
        let state = game.initial_state();
        let policy = vec![0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1];

        let syms = game.symmetries(&state, &policy);
        assert_eq!(syms.len(), 2);

        // Flipped policy should swap columns
        let (_, flipped_policy) = &syms[1];
        assert!((flipped_policy[0] - policy[6]).abs() < 1e-6);
        assert!((flipped_policy[3] - policy[3]).abs() < 1e-6);
    }

    #[test]
    fn test_map_unmap_action_roundtrip() {
        let game = Connect4::new();
        // For all symmetries and actions, map then unmap should give identity
        for sym_idx in 0..2 {
            for action in 0..7u16 {
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
    fn test_map_action_flip_is_self_inverse() {
        let game = Connect4::new();
        // Flip applied twice should give identity
        for action in 0..7u16 {
            let flipped_once = game.map_action(action, 1);
            let flipped_twice = game.map_action(flipped_once, 1);
            assert_eq!(flipped_twice, action, "Flip is not self-inverse for action={}", action);
        }
    }

    #[test]
    fn test_map_action_symmetry_consistency() {
        let game = Connect4::new();
        // Create a state with a piece at column 0
        let mut state = game.initial_state();
        state = game.next_state(&state, 0);

        let policy = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let syms = game.symmetries(&state, &policy);

        // For each symmetry, map_action(0) should point to where the policy mass moved
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
            // The symmetric state should have the piece in the mapped column
            let s = sym_state.as_connect4();
            assert_eq!(
                s.get(5, mapped_action as usize), 1,
                "sym_idx={}: piece not in column {}",
                sym_idx, mapped_action
            );
        }
    }

    #[test]
    fn test_canonical_symmetry_index() {
        use crate::game::compute_hash;

        let game = Connect4::new();
        // Create a state with a piece at left side
        let mut state = game.initial_state();
        state = game.next_state(&state, 0);

        let (sym_idx, canonical_hash) = game.canonical_symmetry_index(&state);

        // Verify this is the minimum hash among symmetries
        let policy = vec![0.0f32; 7];
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
        let game = Connect4::new();

        // Opening in column 0 and column 6 are symmetric
        let mut state0 = game.initial_state();
        state0 = game.next_state(&state0, 0);
        let (_, hash0) = game.canonical_symmetry_index(&state0);

        let mut state6 = game.initial_state();
        state6 = game.next_state(&state6, 6);
        let (_, hash6) = game.canonical_symmetry_index(&state6);

        assert_eq!(
            hash0, hash6,
            "Opening at column 0 and 6 should have same canonical hash"
        );

        // Similarly for columns 1 and 5, 2 and 4
        for (col_a, col_b) in [(1, 5), (2, 4)] {
            let mut sa = game.initial_state();
            sa = game.next_state(&sa, col_a);
            let (_, ha) = game.canonical_symmetry_index(&sa);

            let mut sb = game.initial_state();
            sb = game.next_state(&sb, col_b);
            let (_, hb) = game.canonical_symmetry_index(&sb);

            assert_eq!(
                ha, hb,
                "Opening at column {} and {} should have same canonical hash",
                col_a, col_b
            );
        }
    }

    #[test]
    fn test_center_column_self_symmetric() {
        let game = Connect4::new();

        // Opening at center column (3) should be self-symmetric
        let mut state = game.initial_state();
        state = game.next_state(&state, 3);

        let syms = game.symmetries(&state, &vec![0.0f32; 7]);
        let s0 = syms[0].0.as_connect4();
        let s1 = syms[1].0.as_connect4();

        // Both symmetries should produce the same board
        assert_eq!(s0.board, s1.board, "Center opening should be self-symmetric");
    }

    #[test]
    fn test_distinct_positions_different_hashes() {
        let game = Connect4::new();

        // Far left opening (column 0)
        let mut left = game.initial_state();
        left = game.next_state(&left, 0);
        let (_, left_hash) = game.canonical_symmetry_index(&left);

        // Center-left opening (column 2)
        let mut center_left = game.initial_state();
        center_left = game.next_state(&center_left, 2);
        let (_, center_left_hash) = game.canonical_symmetry_index(&center_left);

        // Center opening (column 3)
        let mut center = game.initial_state();
        center = game.next_state(&center, 3);
        let (_, center_hash) = game.canonical_symmetry_index(&center);

        // All three are truly distinct positions
        assert_ne!(left_hash, center_left_hash, "Left and center-left should have different hashes");
        assert_ne!(left_hash, center_hash, "Left and center should have different hashes");
        assert_ne!(center_left_hash, center_hash, "Center-left and center should have different hashes");
    }
}
