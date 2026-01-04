//! Go game implementation.
//!
//! Variable board size (typically 9x9, 13x13, or 19x19), action_size = board_size + 1 (for pass),
//! 8 symmetries for square boards.
//!
//! Optimized for MCTS performance with fast legality checking.

use super::state::{GameState, GoState};
use super::{in_bounds, Game};

/// Neighbor offsets for the 4 cardinal directions.
const NEIGHBORS: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

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
    #[inline]
    fn action_to_pos(&self, action: u16) -> Option<(usize, usize)> {
        let a = action as usize;
        if a >= self.height * self.width {
            None // Pass
        } else {
            Some((a / self.width, a % self.width))
        }
    }

    /// Convert (row, col) to action.
    #[inline]
    fn pos_to_action(&self, row: usize, col: usize) -> u16 {
        (row * self.width + col) as u16
    }

    /// Get the pass action.
    #[inline]
    fn pass_action(&self) -> u16 {
        (self.height * self.width) as u16
    }

    /// Convert (row, col) to linear index.
    #[inline]
    fn pos_to_idx(&self, row: usize, col: usize) -> usize {
        row * self.width + col
    }

    /// Find all stones in the same group as the given position.
    /// Uses a reusable visited buffer passed in to avoid allocations.
    fn find_group_fast(
        &self,
        state: &GoState,
        row: usize,
        col: usize,
        group: &mut Vec<(usize, usize)>,
        visited: &mut [bool],
    ) {
        group.clear();
        let color = state.get(row, col);
        if color == 0 {
            return;
        }

        let mut stack_len = 1;
        let mut stack = [(row, col); 128]; // Fixed-size stack, sufficient for any reasonable board
        visited[self.pos_to_idx(row, col)] = true;

        while stack_len > 0 {
            stack_len -= 1;
            let (r, c) = stack[stack_len];
            group.push((r, c));

            for &(dr, dc) in &NEIGHBORS {
                let nr = r as i32 + dr;
                let nc = c as i32 + dc;
                if in_bounds(nr, nc, self.height, self.width) {
                    let nr = nr as usize;
                    let nc = nc as usize;
                    let idx = self.pos_to_idx(nr, nc);
                    if !visited[idx] && state.get(nr, nc) == color {
                        visited[idx] = true;
                        stack[stack_len] = (nr, nc);
                        stack_len += 1;
                    }
                }
            }
        }
    }

    /// Check if a group has any liberties (fast early exit).
    #[inline]
    fn group_has_liberties(&self, state: &GoState, group: &[(usize, usize)]) -> bool {
        for &(r, c) in group {
            for &(dr, dc) in &NEIGHBORS {
                let nr = r as i32 + dr;
                let nc = c as i32 + dc;
                if in_bounds(nr, nc, self.height, self.width) {
                    if state.get(nr as usize, nc as usize) == 0 {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Count liberties of a group (only used for scoring, not hot path).
    fn count_liberties_slow(&self, state: &GoState, group: &[(usize, usize)]) -> usize {
        let mut liberty_set = vec![false; self.height * self.width];
        let mut count = 0;

        for &(r, c) in group {
            for &(dr, dc) in &NEIGHBORS {
                let nr = r as i32 + dr;
                let nc = c as i32 + dc;
                if in_bounds(nr, nc, self.height, self.width) {
                    let nr = nr as usize;
                    let nc = nc as usize;
                    let idx = self.pos_to_idx(nr, nc);
                    if state.get(nr, nc) == 0 && !liberty_set[idx] {
                        liberty_set[idx] = true;
                        count += 1;
                    }
                }
            }
        }
        count
    }

    /// Remove captured opponent stones adjacent to (row, col) and return count.
    /// Only checks groups that touch the placed stone - much faster than scanning entire board.
    fn remove_captured_adjacent(&self, state: &mut GoState, row: usize, col: usize, player: i8) -> usize {
        let opponent = -player;
        let mut captured = 0;
        let board_size = self.height * self.width;
        let mut visited = vec![false; board_size];
        let mut group = Vec::with_capacity(32);

        for &(dr, dc) in &NEIGHBORS {
            let nr = row as i32 + dr;
            let nc = col as i32 + dc;
            if in_bounds(nr, nc, self.height, self.width) {
                let nr = nr as usize;
                let nc = nc as usize;
                let idx = self.pos_to_idx(nr, nc);

                // Only check opponent groups we haven't already processed
                if state.get(nr, nc) == opponent && !visited[idx] {
                    self.find_group_fast(state, nr, nc, &mut group, &mut visited);

                    if !self.group_has_liberties(state, &group) {
                        for &(gr, gc) in &group {
                            state.set(gr, gc, 0);
                            captured += 1;
                        }
                    }
                }
            }
        }

        captured
    }

    /// Fast check if placing a stone at (row, col) is legal (not suicide, not ko).
    /// Avoids cloning state by using direct checks.
    fn is_legal_move(&self, state: &GoState, row: usize, col: usize, player: i8) -> bool {
        // Ko check is trivial
        if state.ko_point == (row as i8, col as i8) {
            return false;
        }

        let opponent = -player;

        // Fast check 1: If the position has an empty neighbor, it's definitely not suicide
        for &(dr, dc) in &NEIGHBORS {
            let nr = row as i32 + dr;
            let nc = col as i32 + dc;
            if in_bounds(nr, nc, self.height, self.width) {
                if state.get(nr as usize, nc as usize) == 0 {
                    return true; // Has liberty, definitely legal
                }
            }
        }

        // Fast check 2: Would this move capture any opponent stones?
        // If so, it's legal (we gain liberties from the capture)
        let board_size = self.height * self.width;
        let mut visited = vec![false; board_size];
        let mut group = Vec::with_capacity(32);

        for &(dr, dc) in &NEIGHBORS {
            let nr = row as i32 + dr;
            let nc = col as i32 + dc;
            if in_bounds(nr, nc, self.height, self.width) {
                let nr = nr as usize;
                let nc = nc as usize;
                let idx = self.pos_to_idx(nr, nc);

                if state.get(nr, nc) == opponent && !visited[idx] {
                    self.find_group_fast(state, nr, nc, &mut group, &mut visited);

                    // Check if this opponent group has exactly one liberty (the position we're placing)
                    let mut liberties = 0;
                    let mut only_liberty_is_target = true;
                    'outer: for &(gr, gc) in &group {
                        for &(gdr, gdc) in &NEIGHBORS {
                            let gnr = gr as i32 + gdr;
                            let gnc = gc as i32 + gdc;
                            if in_bounds(gnr, gnc, self.height, self.width) {
                                let gnr = gnr as usize;
                                let gnc = gnc as usize;
                                if state.get(gnr, gnc) == 0 {
                                    if gnr != row || gnc != col {
                                        only_liberty_is_target = false;
                                        break 'outer;
                                    }
                                    liberties += 1;
                                }
                            }
                        }
                    }

                    if liberties > 0 && only_liberty_is_target {
                        return true; // Would capture, so legal
                    }
                }
            }
        }

        // Check 3: Would connecting to friendly groups give us liberties?
        // We need to check if any adjacent friendly group has >1 liberty
        visited.fill(false);
        for &(dr, dc) in &NEIGHBORS {
            let nr = row as i32 + dr;
            let nc = col as i32 + dc;
            if in_bounds(nr, nc, self.height, self.width) {
                let nr = nr as usize;
                let nc = nc as usize;
                let idx = self.pos_to_idx(nr, nc);

                if state.get(nr, nc) == player && !visited[idx] {
                    self.find_group_fast(state, nr, nc, &mut group, &mut visited);

                    // Count liberties of this friendly group (excluding our target position)
                    let mut other_liberties = 0;
                    for &(gr, gc) in &group {
                        for &(gdr, gdc) in &NEIGHBORS {
                            let gnr = gr as i32 + gdr;
                            let gnc = gc as i32 + gdc;
                            if in_bounds(gnr, gnc, self.height, self.width) {
                                let gnr = gnr as usize;
                                let gnc = gnc as usize;
                                if state.get(gnr, gnc) == 0 && (gnr != row || gnc != col) {
                                    other_liberties += 1;
                                    if other_liberties > 0 {
                                        return true; // Friendly group has other liberties
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // No liberties, no captures, no friendly groups with liberties = suicide
        false
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
        new_state.move_count = state.move_count;
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
        new_state.move_count = state.move_count;
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
        let mut actions = Vec::with_capacity(self.action_size);

        // Check board positions using optimized legality check
        for row in 0..self.height {
            for col in 0..self.width {
                if s.get(row, col) == 0 && self.is_legal_move(s, row, col, player) {
                    actions.push(self.pos_to_action(row, col));
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

            // Remove captured stones (only check adjacent groups)
            let captured = self.remove_captured_adjacent(&mut new_state, row, col, player);

            // Check for ko (single stone capture that could be immediately recaptured)
            if captured == 1 {
                // Find the captured position (empty neighbor)
                for &(dr, dc) in &NEIGHBORS {
                    let nr = row as i32 + dr;
                    let nc = col as i32 + dc;
                    if in_bounds(nr, nc, self.height, self.width) {
                        let nr = nr as usize;
                        let nc = nc as usize;
                        if new_state.get(nr, nc) == 0 {
                            // Check if our stone is a single stone with exactly one liberty
                            let board_size = self.height * self.width;
                            let mut visited = vec![false; board_size];
                            let mut group = Vec::with_capacity(8);
                            self.find_group_fast(&new_state, row, col, &mut group, &mut visited);
                            if group.len() == 1 {
                                let liberties = self.count_liberties_slow(&new_state, &group);
                                if liberties == 1 {
                                    new_state.ko_point = (nr as i8, nc as i8);
                                }
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

        // Switch turn and increment move count
        new_state.turn = 1 - new_state.turn;
        new_state.move_count += 1;

        GameState::Go(new_state)
    }

    fn is_terminal(&self, state: &GameState) -> bool {
        let s = state.as_go();
        // Game ends on two consecutive passes OR exceeding move limit
        s.passes >= 2 || s.move_count >= s.max_moves()
    }

    fn terminal_reward(&self, state: &GameState) -> f32 {
        let s = state.as_go();

        // If game ended due to move limit, it's a draw
        if s.move_count >= s.max_moves() {
            return 0.0;
        }

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
            new_state.move_count = s.move_count;
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
        // Metadata: [passes, ko_row, ko_col, turn, move_count_lo, move_count_hi]
        if data.len() > board_size + 3 {
            state.passes = data[board_size] as u8;
            state.ko_point = (data[board_size + 1], data[board_size + 2]);
            state.turn = data[board_size + 3] as u8;
            // Parse move_count if present (for backward compatibility)
            if data.len() > board_size + 5 {
                let lo = data[board_size + 4] as u8 as u16;
                let hi = data[board_size + 5] as u8 as u16;
                state.move_count = lo | (hi << 8);
            }
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

    #[test]
    fn test_move_limit_draw() {
        // Use a tiny 3x3 board to test move limit quickly
        // max_moves = 3 * 3 * 2 = 18
        let game = Go::new(3);
        let mut state = game.initial_state();
        let max_moves = state.as_go().max_moves();
        assert_eq!(max_moves, 18);

        // Play moves until we hit the limit
        for i in 0..max_moves {
            assert!(!game.is_terminal(&state), "Game ended early at move {}", i);
            // Just pass alternately
            state = game.next_state(&state, game.pass_action());
            // Reset passes to prevent ending via double pass
            if i < max_moves - 1 {
                state.as_go_mut().passes = 0;
            }
        }

        // Now should be terminal due to move limit
        assert!(game.is_terminal(&state));
        // Should be a draw
        assert_eq!(game.terminal_reward(&state), 0.0);
    }
}
