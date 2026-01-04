//! Go game implementation.
//!
//! Variable board size (typically 9x9, 13x13, or 19x19), action_size = board_size + 1 (for pass),
//! 8 symmetries for square boards.

use super::state::{GameState, GoState};
use super::{in_bounds, Game};
use std::collections::HashSet;

/// Go game with configurable board size.
pub struct Go {
    height: usize,
    width: usize,
    action_size: usize,
    komi: f32,
}

impl Go {
    pub fn new(size: usize) -> Self {
        Self::with_dimensions(size, size)
    }

    pub fn with_dimensions(height: usize, width: usize) -> Self {
        Self {
            height,
            width,
            action_size: height * width + 1, // +1 for pass
            komi: 7.5,
        }
    }

    pub fn with_komi(mut self, komi: f32) -> Self {
        self.komi = komi;
        self
    }

    /// Convert action to (row, col) or None for pass.
    fn action_to_pos(&self, action: u16) -> Option<(usize, usize)> {
        let a = action as usize;
        if a >= self.height * self.width {
            None // Pass
        } else {
            Some((a / self.width, a % self.width))
        }
    }

    /// Convert (row, col) to action.
    fn pos_to_action(&self, row: usize, col: usize) -> u16 {
        (row * self.width + col) as u16
    }

    /// Get the pass action.
    fn pass_action(&self) -> u16 {
        (self.height * self.width) as u16
    }

    /// Find all stones in the same group as the given position.
    fn find_group(&self, state: &GoState, row: usize, col: usize) -> HashSet<(usize, usize)> {
        let color = state.get(row, col);
        if color == 0 {
            return HashSet::new();
        }

        let mut group = HashSet::new();
        let mut stack = vec![(row, col)];

        while let Some((r, c)) = stack.pop() {
            if group.contains(&(r, c)) {
                continue;
            }
            if state.get(r, c) != color {
                continue;
            }

            group.insert((r, c));

            // Check neighbors
            for (dr, dc) in &[(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                let nr = r as i32 + dr;
                let nc = c as i32 + dc;
                if in_bounds(nr, nc, self.height, self.width) {
                    let nr = nr as usize;
                    let nc = nc as usize;
                    if state.get(nr, nc) == color && !group.contains(&(nr, nc)) {
                        stack.push((nr, nc));
                    }
                }
            }
        }

        group
    }

    /// Count liberties of a group.
    fn count_liberties(&self, state: &GoState, group: &HashSet<(usize, usize)>) -> usize {
        let mut liberties = HashSet::new();

        for &(r, c) in group {
            for (dr, dc) in &[(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                let nr = r as i32 + dr;
                let nc = c as i32 + dc;
                if in_bounds(nr, nc, self.height, self.width) {
                    let nr = nr as usize;
                    let nc = nc as usize;
                    if state.get(nr, nc) == 0 {
                        liberties.insert((nr, nc));
                    }
                }
            }
        }

        liberties.len()
    }

    /// Remove captured stones and return the number removed.
    fn remove_captured(&self, state: &mut GoState, player: i8) -> usize {
        let opponent = -player;
        let mut captured = 0;

        for row in 0..self.height {
            for col in 0..self.width {
                if state.get(row, col) == opponent {
                    let group = self.find_group(state, row, col);
                    if self.count_liberties(state, &group) == 0 {
                        for &(r, c) in &group {
                            state.set(r, c, 0);
                            captured += 1;
                        }
                    }
                }
            }
        }

        captured
    }

    /// Check if a move is suicide (would result in self-capture).
    fn is_suicide(&self, state: &GoState, row: usize, col: usize, player: i8) -> bool {
        // Place the stone temporarily
        let mut temp = state.clone();
        temp.set(row, col, player);

        // Check if any adjacent opponent groups are captured
        let opponent = -player;
        let mut would_capture = false;
        for (dr, dc) in &[(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
            let nr = row as i32 + dr;
            let nc = col as i32 + dc;
            if in_bounds(nr, nc, self.height, self.width) {
                let nr = nr as usize;
                let nc = nc as usize;
                if temp.get(nr, nc) == opponent {
                    let group = self.find_group(&temp, nr, nc);
                    if self.count_liberties(&temp, &group) == 0 {
                        would_capture = true;
                        break;
                    }
                }
            }
        }

        // If we would capture, it's not suicide
        if would_capture {
            return false;
        }

        // Check if our own group would have zero liberties
        let our_group = self.find_group(&temp, row, col);
        self.count_liberties(&temp, &our_group) == 0
    }

    /// Check if a move violates the ko rule.
    fn is_ko(&self, state: &GoState, row: usize, col: usize) -> bool {
        state.ko_point == (row as i8, col as i8)
    }

    /// Score the game using area scoring (Chinese rules).
    fn score(&self, state: &GoState) -> f32 {
        let mut black_score = 0.0f32;
        let mut white_score = self.komi;

        // Count stones and territory
        let mut visited = vec![vec![false; self.width]; self.height];

        for row in 0..self.height {
            for col in 0..self.width {
                if visited[row][col] {
                    continue;
                }

                let cell = state.get(row, col);
                if cell == 1 {
                    black_score += 1.0;
                    visited[row][col] = true;
                } else if cell == -1 {
                    white_score += 1.0;
                    visited[row][col] = true;
                } else {
                    // Empty - flood fill to find territory
                    let (territory, borders_black, borders_white) =
                        self.flood_fill_territory(state, row, col, &mut visited);

                    if borders_black && !borders_white {
                        black_score += territory as f32;
                    } else if borders_white && !borders_black {
                        white_score += territory as f32;
                    }
                    // Dame (neutral) doesn't count for either
                }
            }
        }

        black_score - white_score
    }

    /// Flood fill to find empty territory.
    fn flood_fill_territory(
        &self,
        state: &GoState,
        start_row: usize,
        start_col: usize,
        visited: &mut Vec<Vec<bool>>,
    ) -> (usize, bool, bool) {
        let mut count = 0;
        let mut borders_black = false;
        let mut borders_white = false;
        let mut stack = vec![(start_row, start_col)];

        while let Some((row, col)) = stack.pop() {
            if visited[row][col] {
                continue;
            }

            let cell = state.get(row, col);
            if cell == 1 {
                borders_black = true;
                continue;
            } else if cell == -1 {
                borders_white = true;
                continue;
            }

            visited[row][col] = true;
            count += 1;

            for (dr, dc) in &[(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                let nr = row as i32 + dr;
                let nc = col as i32 + dc;
                if in_bounds(nr, nc, self.height, self.width) {
                    stack.push((nr as usize, nc as usize));
                }
            }
        }

        (count, borders_black, borders_white)
    }

    /// Rotate position 90 degrees clockwise.
    fn rotate_pos(&self, row: usize, col: usize) -> (usize, usize) {
        (col, self.height - 1 - row)
    }

    /// Flip position horizontally.
    fn flip_pos(&self, row: usize, col: usize) -> (usize, usize) {
        (row, self.width - 1 - col)
    }

    /// Apply rotation to action.
    fn rotate_action(&self, action: u16) -> u16 {
        if let Some((row, col)) = self.action_to_pos(action) {
            let (nr, nc) = self.rotate_pos(row, col);
            self.pos_to_action(nr, nc)
        } else {
            action // Pass stays pass
        }
    }

    /// Apply flip to action.
    fn flip_action(&self, action: u16) -> u16 {
        if let Some((row, col)) = self.action_to_pos(action) {
            let (nr, nc) = self.flip_pos(row, col);
            self.pos_to_action(nr, nc)
        } else {
            action
        }
    }

    /// Rotate the board 90 degrees clockwise.
    fn rotate_board(&self, state: &GoState) -> GoState {
        let mut new_state = GoState::new(self.height, self.width);
        for row in 0..self.height {
            for col in 0..self.width {
                let (nr, nc) = self.rotate_pos(row, col);
                new_state.set(nr, nc, state.get(row, col));
            }
        }
        new_state.passes = state.passes;
        new_state.turn = state.turn;
        if state.ko_point.0 >= 0 {
            let (kr, kc) = self.rotate_pos(state.ko_point.0 as usize, state.ko_point.1 as usize);
            new_state.ko_point = (kr as i8, kc as i8);
        } else {
            new_state.ko_point = state.ko_point;
        }
        new_state
    }

    /// Flip the board horizontally.
    fn flip_board(&self, state: &GoState) -> GoState {
        let mut new_state = GoState::new(self.height, self.width);
        for row in 0..self.height {
            for col in 0..self.width {
                let (nr, nc) = self.flip_pos(row, col);
                new_state.set(nr, nc, state.get(row, col));
            }
        }
        new_state.passes = state.passes;
        new_state.turn = state.turn;
        if state.ko_point.0 >= 0 {
            let (kr, kc) = self.flip_pos(state.ko_point.0 as usize, state.ko_point.1 as usize);
            new_state.ko_point = (kr as i8, kc as i8);
        } else {
            new_state.ko_point = state.ko_point;
        }
        new_state
    }
}

impl Game for Go {
    fn initial_state(&self) -> GameState {
        GameState::Go(GoState::new(self.height, self.width))
    }

    fn current_player(&self, state: &GameState) -> i8 {
        state.as_go().current_player()
    }

    fn legal_actions(&self, state: &GameState) -> Vec<u16> {
        let s = state.as_go();
        let player = s.current_player();
        let mut actions = Vec::new();

        // Check board positions
        for row in 0..self.height {
            for col in 0..self.width {
                if s.get(row, col) == 0 {
                    // Check if legal (not suicide, not ko)
                    if !self.is_suicide(s, row, col, player) && !self.is_ko(s, row, col) {
                        actions.push(self.pos_to_action(row, col));
                    }
                }
            }
        }

        // Pass is always legal
        actions.push(self.pass_action());

        actions
    }

    fn next_state(&self, state: &GameState, action: u16) -> GameState {
        let s = state.as_go();
        let player = s.current_player();
        let mut new_state = s.clone();

        // Clear ko point
        new_state.ko_point = (-1, -1);

        if let Some((row, col)) = self.action_to_pos(action) {
            // Place stone
            new_state.set(row, col, player);
            new_state.passes = 0;

            // Remove captured stones
            let captured = self.remove_captured(&mut new_state, player);

            // Check for ko (single stone capture that could be immediately recaptured)
            if captured == 1 {
                // Find the captured position
                for (dr, dc) in &[(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let nr = row as i32 + dr;
                    let nc = col as i32 + dc;
                    if in_bounds(nr, nc, self.height, self.width) {
                        let nr = nr as usize;
                        let nc = nc as usize;
                        if new_state.get(nr, nc) == 0 {
                            // Check if playing there would recapture exactly our stone
                            let our_group = self.find_group(&new_state, row, col);
                            if our_group.len() == 1
                                && self.count_liberties(&new_state, &our_group) == 1
                            {
                                new_state.ko_point = (nr as i8, nc as i8);
                            }
                            break;
                        }
                    }
                }
            }
        } else {
            // Pass
            new_state.passes += 1;
        }

        // Switch turn
        new_state.turn = 1 - new_state.turn;

        GameState::Go(new_state)
    }

    fn is_terminal(&self, state: &GameState) -> bool {
        state.as_go().passes >= 2
    }

    fn terminal_reward(&self, state: &GameState) -> f32 {
        let s = state.as_go();
        let score = self.score(s);
        let current = s.current_player();

        // Score is from black's perspective
        if current == 1 {
            // Black's turn
            if score > 0.0 {
                1.0
            } else if score < 0.0 {
                -1.0
            } else {
                0.0
            }
        } else {
            // White's turn
            if score < 0.0 {
                1.0
            } else if score > 0.0 {
                -1.0
            } else {
                0.0
            }
        }
    }

    fn canonical_state(&self, state: &GameState) -> GameState {
        let s = state.as_go();
        let player = s.current_player();
        if player == 1 {
            state.clone()
        } else {
            let mut new_state = GoState::new(self.height, self.width);
            for i in 0..s.board.len() {
                new_state.board[i] = -s.board[i];
            }
            new_state.passes = s.passes;
            new_state.ko_point = s.ko_point;
            new_state.turn = s.turn;
            GameState::Go(new_state)
        }
    }

    fn to_tensor(&self, state: &GameState) -> Vec<i64> {
        let s = state.as_go();
        s.board.iter().map(|&v| (v + 1) as i64).collect()
    }

    fn symmetries(&self, state: &GameState, policy: &[f32]) -> Vec<(GameState, Vec<f32>)> {
        let s = state.as_go();
        let mut result = Vec::with_capacity(8);

        // Only generate 8 symmetries for square boards
        if self.height != self.width {
            result.push((state.clone(), policy.to_vec()));
            return result;
        }

        let mut current = s.clone();
        for rot in 0..4 {
            // Without flip
            let mut new_policy = vec![0.0f32; self.action_size];
            for (i, &p) in policy.iter().enumerate() {
                let mut a = i as u16;
                for _ in 0..rot {
                    a = self.rotate_action(a);
                }
                new_policy[a as usize] = p;
            }
            result.push((GameState::Go(current.clone()), new_policy));

            // With flip
            let flipped = self.flip_board(&current);
            let mut flipped_policy = vec![0.0f32; self.action_size];
            for (i, &p) in policy.iter().enumerate() {
                let mut a = i as u16;
                for _ in 0..rot {
                    a = self.rotate_action(a);
                }
                a = self.flip_action(a);
                flipped_policy[a as usize] = p;
            }
            result.push((GameState::Go(flipped), flipped_policy));

            // Rotate for next iteration
            current = self.rotate_board(&current);
        }

        result
    }

    fn action_size(&self) -> usize {
        self.action_size
    }

    fn board_size(&self) -> usize {
        self.height * self.width
    }

    fn board_dims(&self) -> (usize, usize) {
        (self.height, self.width)
    }

    fn state_from_slice(&self, data: &[i8]) -> GameState {
        let mut state = GoState::new(self.height, self.width);
        let board_size = self.height * self.width;

        // Copy board data
        for (i, &v) in data.iter().take(board_size).enumerate() {
            let row = i / self.width;
            let col = i % self.width;
            state.set(row, col, v);
        }

        // Parse metadata if present (flattened format includes extra row)
        // Metadata starts at index board_size: [passes, ko_row, ko_col, turn]
        if data.len() > board_size + 3 {
            state.passes = data[board_size] as u8;
            state.ko_point = (data[board_size + 1], data[board_size + 2]);
            state.turn = data[board_size + 3] as u8;
        }

        GameState::Go(state)
    }

    fn render(&self, state: &GameState) -> String {
        let s = state.as_go();
        let mut result = String::new();

        // Column labels
        result.push_str("  ");
        for col in 0..self.width {
            result.push_str(&format!("{} ", (b'A' + col as u8) as char));
        }
        result.push('\n');

        // Board
        for row in 0..self.height {
            result.push_str(&format!("{:2} ", self.height - row));
            for col in 0..self.width {
                let c = match s.get(row, col) {
                    1 => 'X',
                    -1 => 'O',
                    _ => '.',
                };
                result.push(c);
                result.push(' ');
            }
            result.push_str(&format!("{:2}", self.height - row));
            result.push('\n');
        }

        // Column labels again
        result.push_str("  ");
        for col in 0..self.width {
            result.push_str(&format!("{} ", (b'A' + col as u8) as char));
        }
        result.push('\n');

        // Status
        result.push_str(&format!(
            "Turn: {}, Passes: {}, Ko: {:?}\n",
            if s.current_player() == 1 {
                "Black"
            } else {
                "White"
            },
            s.passes,
            s.ko_point
        ));

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let game = Go::new(9);
        let state = game.initial_state();
        let s = state.as_go();
        assert!(s.board.iter().all(|&v| v == 0));
        assert_eq!(s.passes, 0);
        assert_eq!(game.current_player(&state), 1); // Black first
    }

    #[test]
    fn test_legal_actions() {
        let game = Go::new(9);
        let state = game.initial_state();
        let legal = game.legal_actions(&state);
        // All board positions + pass
        assert_eq!(legal.len(), 82);
    }

    #[test]
    fn test_pass() {
        let game = Go::new(9);
        let state = game.initial_state();

        let state = game.next_state(&state, game.pass_action());
        assert_eq!(state.as_go().passes, 1);
        assert_eq!(game.current_player(&state), -1); // White's turn

        let state = game.next_state(&state, game.pass_action());
        assert_eq!(state.as_go().passes, 2);
        assert!(game.is_terminal(&state));
    }

    #[test]
    fn test_capture() {
        let game = Go::new(9);
        let mut state = game.initial_state();

        // Surround a white stone
        // Place black at (0,1), (1,0), (1,2), (2,1)
        // Place white at (1,1)
        state = game.next_state(&state, game.pos_to_action(0, 1)); // Black
        state = game.next_state(&state, game.pos_to_action(1, 1)); // White
        state = game.next_state(&state, game.pos_to_action(1, 0)); // Black
        state = game.next_state(&state, game.pos_to_action(8, 8)); // White elsewhere
        state = game.next_state(&state, game.pos_to_action(1, 2)); // Black
        state = game.next_state(&state, game.pos_to_action(8, 7)); // White elsewhere
        state = game.next_state(&state, game.pos_to_action(2, 1)); // Black captures!

        // White stone at (1,1) should be captured
        assert_eq!(state.as_go().get(1, 1), 0);
    }

    #[test]
    fn test_suicide_prevention() {
        let game = Go::new(9);
        let mut state = game.initial_state();

        // Set up a situation where playing at (1,1) would be suicide
        // Black stones at (0,1), (1,0), (1,2), (2,1)
        state = game.next_state(&state, game.pos_to_action(0, 1)); // Black
        state = game.next_state(&state, game.pos_to_action(8, 8)); // White
        state = game.next_state(&state, game.pos_to_action(1, 0)); // Black
        state = game.next_state(&state, game.pos_to_action(8, 7)); // White
        state = game.next_state(&state, game.pos_to_action(1, 2)); // Black
        state = game.next_state(&state, game.pos_to_action(8, 6)); // White
        state = game.next_state(&state, game.pos_to_action(2, 1)); // Black

        // Now white at (1,1) would be suicide
        let legal = game.legal_actions(&state);
        assert!(!legal.contains(&game.pos_to_action(1, 1)));
    }

    #[test]
    fn test_symmetries() {
        let game = Go::new(9);
        let state = game.initial_state();
        let mut policy = vec![0.0f32; 82];
        policy[0] = 1.0; // Action at (0,0)

        let syms = game.symmetries(&state, &policy);
        assert_eq!(syms.len(), 8);

        // Check that the policy mass moves correctly
        for (i, (_, p)) in syms.iter().enumerate() {
            let sum: f32 = p.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "Symmetry {}: sum = {}",
                i,
                sum
            );
        }
    }
}
