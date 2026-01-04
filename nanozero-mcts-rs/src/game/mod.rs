//! Game implementations for MCTS.
//!
//! Defines the Game trait and provides implementations for TicTacToe,
//! Connect4, and Go.

pub mod connect4;
pub mod go;
pub mod state;
pub mod tictactoe;

pub use connect4::Connect4;
pub use go::Go;
pub use state::GameState;
pub use tictactoe::TicTacToe;

/// Game trait defining the interface for all games.
///
/// All methods take and return owned state to simplify the interface.
/// Implementations should be optimized for performance as these methods
/// are called frequently during MCTS.
pub trait Game: Send + Sync {
    /// Get the initial game state.
    fn initial_state(&self) -> GameState;

    /// Get the current player (1 or -1).
    fn current_player(&self, state: &GameState) -> i8;

    /// Get list of legal actions.
    fn legal_actions(&self, state: &GameState) -> Vec<u16>;

    /// Get legal actions as a boolean mask.
    fn legal_actions_mask(&self, state: &GameState) -> Vec<bool> {
        let mut mask = vec![false; self.action_size()];
        for action in self.legal_actions(state) {
            mask[action as usize] = true;
        }
        mask
    }

    /// Apply an action and return the new state.
    fn next_state(&self, state: &GameState, action: u16) -> GameState;

    /// Check if the game is over.
    fn is_terminal(&self, state: &GameState) -> bool;

    /// Get the reward for the current player at a terminal state.
    ///
    /// Returns 1.0 for win, -1.0 for loss, 0.0 for draw.
    fn terminal_reward(&self, state: &GameState) -> f32;

    /// Get the canonical form of the state (from current player's perspective).
    ///
    /// Flips the board so the current player's pieces are always +1.
    fn canonical_state(&self, state: &GameState) -> GameState;

    /// Convert state to tensor representation for neural network.
    ///
    /// Maps {-1, 0, 1} -> {0, 1, 2} for embedding lookup.
    fn to_tensor(&self, state: &GameState) -> Vec<i64>;

    /// Generate symmetric variants of the state and policy.
    ///
    /// Used for data augmentation during training.
    fn symmetries(&self, state: &GameState, policy: &[f32]) -> Vec<(GameState, Vec<f32>)>;

    /// Get the number of possible actions.
    fn action_size(&self) -> usize;

    /// Get the total board size (number of cells).
    fn board_size(&self) -> usize;

    /// Get the board dimensions (height, width) for display.
    fn board_dims(&self) -> (usize, usize);

    /// Create a game state from a slice of i8 values.
    fn state_from_slice(&self, data: &[i8]) -> GameState;

    /// Render the state as a string for debugging.
    fn render(&self, state: &GameState) -> String;
}

/// Helper to check if a position is within bounds.
#[inline]
pub fn in_bounds(row: i32, col: i32, height: usize, width: usize) -> bool {
    row >= 0 && row < height as i32 && col >= 0 && col < width as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_in_bounds() {
        assert!(in_bounds(0, 0, 3, 3));
        assert!(in_bounds(2, 2, 3, 3));
        assert!(!in_bounds(-1, 0, 3, 3));
        assert!(!in_bounds(0, 3, 3, 3));
    }
}
