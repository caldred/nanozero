//! Game state representations.
//!
//! Each game has its own compact state representation optimized for
//! performance. The GameState enum provides a unified interface.

/// Unified game state enum.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GameState {
    TicTacToe(TicTacToeState),
    Connect4(Connect4State),
    Go(GoState),
}

/// TicTacToe state: 3x3 = 9 cells.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TicTacToeState {
    /// Board values: -1 (O), 0 (empty), 1 (X)
    pub board: [i8; 9],
}

impl TicTacToeState {
    pub fn new() -> Self {
        Self { board: [0; 9] }
    }

    #[inline]
    pub fn get(&self, row: usize, col: usize) -> i8 {
        self.board[row * 3 + col]
    }

    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: i8) {
        self.board[row * 3 + col] = value;
    }
}

impl Default for TicTacToeState {
    fn default() -> Self {
        Self::new()
    }
}

/// Connect4 state: 6x7 = 42 cells.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Connect4State {
    /// Board values: -1 (O), 0 (empty), 1 (X)
    pub board: [i8; 42],
}

impl Connect4State {
    pub const HEIGHT: usize = 6;
    pub const WIDTH: usize = 7;

    pub fn new() -> Self {
        Self { board: [0; 42] }
    }

    #[inline]
    pub fn get(&self, row: usize, col: usize) -> i8 {
        self.board[row * Self::WIDTH + col]
    }

    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: i8) {
        self.board[row * Self::WIDTH + col] = value;
    }
}

impl Default for Connect4State {
    fn default() -> Self {
        Self::new()
    }
}

/// Go state with board and metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GoState {
    /// Board values: -1 (white), 0 (empty), 1 (black)
    pub board: Vec<i8>,
    /// Board dimensions
    pub height: usize,
    pub width: usize,
    /// Consecutive pass count (2 = game over)
    pub passes: u8,
    /// Ko point (forbidden recapture), or (-1, -1) if none
    pub ko_point: (i8, i8),
    /// Current player's turn: 0 = black (1), 1 = white (-1)
    pub turn: u8,
    /// Move count for draw rule (game ends as draw after max_moves)
    pub move_count: u16,
}

impl GoState {
    pub fn new(height: usize, width: usize) -> Self {
        Self {
            board: vec![0; height * width],
            height,
            width,
            passes: 0,
            ko_point: (-1, -1),
            turn: 0,
            move_count: 0,
        }
    }

    /// Maximum moves before game is declared a draw (2 * board_size²)
    #[inline]
    pub fn max_moves(&self) -> u16 {
        (self.height * self.width * 2) as u16
    }

    #[inline]
    pub fn get(&self, row: usize, col: usize) -> i8 {
        self.board[row * self.width + col]
    }

    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: i8) {
        self.board[row * self.width + col] = value;
    }

    /// Get the current player (1 = black, -1 = white).
    #[inline]
    pub fn current_player(&self) -> i8 {
        if self.turn == 0 {
            1
        } else {
            -1
        }
    }
}

impl GameState {
    /// Check if this is a TicTacToe state.
    pub fn is_tictactoe(&self) -> bool {
        matches!(self, GameState::TicTacToe(_))
    }

    /// Check if this is a Connect4 state.
    pub fn is_connect4(&self) -> bool {
        matches!(self, GameState::Connect4(_))
    }

    /// Check if this is a Go state.
    pub fn is_go(&self) -> bool {
        matches!(self, GameState::Go(_))
    }

    /// Get the TicTacToe state, panicking if wrong type.
    pub fn as_tictactoe(&self) -> &TicTacToeState {
        match self {
            GameState::TicTacToe(s) => s,
            _ => panic!("Expected TicTacToe state"),
        }
    }

    /// Get the TicTacToe state mutably, panicking if wrong type.
    pub fn as_tictactoe_mut(&mut self) -> &mut TicTacToeState {
        match self {
            GameState::TicTacToe(s) => s,
            _ => panic!("Expected TicTacToe state"),
        }
    }

    /// Get the Connect4 state, panicking if wrong type.
    pub fn as_connect4(&self) -> &Connect4State {
        match self {
            GameState::Connect4(s) => s,
            _ => panic!("Expected Connect4 state"),
        }
    }

    /// Get the Connect4 state mutably, panicking if wrong type.
    pub fn as_connect4_mut(&mut self) -> &mut Connect4State {
        match self {
            GameState::Connect4(s) => s,
            _ => panic!("Expected Connect4 state"),
        }
    }

    /// Get the Go state, panicking if wrong type.
    pub fn as_go(&self) -> &GoState {
        match self {
            GameState::Go(s) => s,
            _ => panic!("Expected Go state"),
        }
    }

    /// Get the Go state mutably, panicking if wrong type.
    pub fn as_go_mut(&mut self) -> &mut GoState {
        match self {
            GameState::Go(s) => s,
            _ => panic!("Expected Go state"),
        }
    }
}
