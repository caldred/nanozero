//! nanozero-mcts-rs: High-performance Rust backend for NanoZero MCTS.
//!
//! Provides Python bindings via PyO3 for accelerated MCTS operations.

pub mod batch;
pub mod bayesian_node;
pub mod bayesian_search;
pub mod game;
pub mod math;
pub mod node;
pub mod search;
pub mod tree;
pub mod ucb;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

use crate::game::{Connect4, Game, GameState, Go, TicTacToe};

// ============================================================================
// Game Python Wrappers
// ============================================================================

/// Python wrapper for TicTacToe game.
#[pyclass(name = "RustTicTacToe")]
pub struct PyTicTacToe {
    game: TicTacToe,
}

#[pymethods]
impl PyTicTacToe {
    #[new]
    fn new() -> Self {
        Self {
            game: TicTacToe::new(),
        }
    }

    fn initial_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i8>> {
        match self.game.initial_state() {
            GameState::TicTacToe(s) => s.board.to_vec().into_pyarray_bound(py),
            _ => unreachable!(),
        }
    }

    fn current_player(&self, state: PyReadonlyArray1<i8>) -> i8 {
        let board: [i8; 9] = state.as_slice().unwrap().try_into().unwrap();
        let gs = GameState::TicTacToe(game::state::TicTacToeState { board });
        self.game.current_player(&gs)
    }

    fn legal_actions(&self, state: PyReadonlyArray1<i8>) -> Vec<u16> {
        let board: [i8; 9] = state.as_slice().unwrap().try_into().unwrap();
        let gs = GameState::TicTacToe(game::state::TicTacToeState { board });
        self.game.legal_actions(&gs)
    }

    fn legal_actions_mask(&self, state: PyReadonlyArray1<i8>) -> Vec<bool> {
        let board: [i8; 9] = state.as_slice().unwrap().try_into().unwrap();
        let gs = GameState::TicTacToe(game::state::TicTacToeState { board });
        self.game.legal_actions_mask(&gs)
    }

    fn next_state<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<i8>,
        action: u16,
    ) -> Bound<'py, PyArray1<i8>> {
        let board: [i8; 9] = state.as_slice().unwrap().try_into().unwrap();
        let gs = GameState::TicTacToe(game::state::TicTacToeState { board });
        match self.game.next_state(&gs, action) {
            GameState::TicTacToe(s) => s.board.to_vec().into_pyarray_bound(py),
            _ => unreachable!(),
        }
    }

    fn is_terminal(&self, state: PyReadonlyArray1<i8>) -> bool {
        let board: [i8; 9] = state.as_slice().unwrap().try_into().unwrap();
        let gs = GameState::TicTacToe(game::state::TicTacToeState { board });
        self.game.is_terminal(&gs)
    }

    fn terminal_reward(&self, state: PyReadonlyArray1<i8>) -> f32 {
        let board: [i8; 9] = state.as_slice().unwrap().try_into().unwrap();
        let gs = GameState::TicTacToe(game::state::TicTacToeState { board });
        self.game.terminal_reward(&gs)
    }

    fn canonical_state<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<i8>,
    ) -> Bound<'py, PyArray1<i8>> {
        let board: [i8; 9] = state.as_slice().unwrap().try_into().unwrap();
        let gs = GameState::TicTacToe(game::state::TicTacToeState { board });
        match self.game.canonical_state(&gs) {
            GameState::TicTacToe(s) => s.board.to_vec().into_pyarray_bound(py),
            _ => unreachable!(),
        }
    }

    fn to_tensor<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<i8>,
    ) -> Bound<'py, PyArray1<i64>> {
        let board: [i8; 9] = state.as_slice().unwrap().try_into().unwrap();
        let gs = GameState::TicTacToe(game::state::TicTacToeState { board });
        self.game.to_tensor(&gs).into_pyarray_bound(py)
    }

    fn action_size(&self) -> usize {
        self.game.action_size()
    }

    fn board_size(&self) -> (usize, usize) {
        self.game.board_size()
    }

    fn render(&self, state: PyReadonlyArray1<i8>) -> String {
        let board: [i8; 9] = state.as_slice().unwrap().try_into().unwrap();
        let gs = GameState::TicTacToe(game::state::TicTacToeState { board });
        self.game.render(&gs)
    }
}

/// Python wrapper for Connect4 game.
#[pyclass(name = "RustConnect4")]
pub struct PyConnect4 {
    game: Connect4,
}

#[pymethods]
impl PyConnect4 {
    #[new]
    fn new() -> Self {
        Self {
            game: Connect4::new(),
        }
    }

    fn initial_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i8>> {
        match self.game.initial_state() {
            GameState::Connect4(s) => s.board.to_vec().into_pyarray_bound(py),
            _ => unreachable!(),
        }
    }

    fn current_player(&self, state: PyReadonlyArray1<i8>) -> i8 {
        let board: [i8; 42] = state.as_slice().unwrap().try_into().unwrap();
        let gs = GameState::Connect4(game::state::Connect4State { board });
        self.game.current_player(&gs)
    }

    fn legal_actions(&self, state: PyReadonlyArray1<i8>) -> Vec<u16> {
        let board: [i8; 42] = state.as_slice().unwrap().try_into().unwrap();
        let gs = GameState::Connect4(game::state::Connect4State { board });
        self.game.legal_actions(&gs)
    }

    fn legal_actions_mask(&self, state: PyReadonlyArray1<i8>) -> Vec<bool> {
        let board: [i8; 42] = state.as_slice().unwrap().try_into().unwrap();
        let gs = GameState::Connect4(game::state::Connect4State { board });
        self.game.legal_actions_mask(&gs)
    }

    fn next_state<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<i8>,
        action: u16,
    ) -> Bound<'py, PyArray1<i8>> {
        let board: [i8; 42] = state.as_slice().unwrap().try_into().unwrap();
        let gs = GameState::Connect4(game::state::Connect4State { board });
        match self.game.next_state(&gs, action) {
            GameState::Connect4(s) => s.board.to_vec().into_pyarray_bound(py),
            _ => unreachable!(),
        }
    }

    fn is_terminal(&self, state: PyReadonlyArray1<i8>) -> bool {
        let board: [i8; 42] = state.as_slice().unwrap().try_into().unwrap();
        let gs = GameState::Connect4(game::state::Connect4State { board });
        self.game.is_terminal(&gs)
    }

    fn terminal_reward(&self, state: PyReadonlyArray1<i8>) -> f32 {
        let board: [i8; 42] = state.as_slice().unwrap().try_into().unwrap();
        let gs = GameState::Connect4(game::state::Connect4State { board });
        self.game.terminal_reward(&gs)
    }

    fn canonical_state<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<i8>,
    ) -> Bound<'py, PyArray1<i8>> {
        let board: [i8; 42] = state.as_slice().unwrap().try_into().unwrap();
        let gs = GameState::Connect4(game::state::Connect4State { board });
        match self.game.canonical_state(&gs) {
            GameState::Connect4(s) => s.board.to_vec().into_pyarray_bound(py),
            _ => unreachable!(),
        }
    }

    fn to_tensor<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<i8>,
    ) -> Bound<'py, PyArray1<i64>> {
        let board: [i8; 42] = state.as_slice().unwrap().try_into().unwrap();
        let gs = GameState::Connect4(game::state::Connect4State { board });
        self.game.to_tensor(&gs).into_pyarray_bound(py)
    }

    fn action_size(&self) -> usize {
        self.game.action_size()
    }

    fn board_size(&self) -> (usize, usize) {
        self.game.board_size()
    }

    fn render(&self, state: PyReadonlyArray1<i8>) -> String {
        let board: [i8; 42] = state.as_slice().unwrap().try_into().unwrap();
        let gs = GameState::Connect4(game::state::Connect4State { board });
        self.game.render(&gs)
    }
}

/// Python wrapper for Go game.
#[pyclass(name = "RustGo")]
pub struct PyGo {
    game: Go,
    height: usize,
    width: usize,
}

#[pymethods]
impl PyGo {
    #[new]
    #[pyo3(signature = (size=9))]
    fn new(size: usize) -> Self {
        Self {
            game: Go::new(size),
            height: size,
            width: size,
        }
    }

    fn initial_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<i8>> {
        match self.game.initial_state() {
            GameState::Go(s) => {
                // Return as 2D array with metadata row
                let mut data = vec![vec![0i8; self.width]; self.height + 1];
                for row in 0..self.height {
                    for col in 0..self.width {
                        data[row][col] = s.get(row, col);
                    }
                }
                // Metadata row
                data[self.height][0] = s.passes as i8;
                data[self.height][1] = s.ko_point.0;
                data[self.height][2] = s.ko_point.1;
                data[self.height][3] = s.turn as i8;

                let flat: Vec<i8> = data.into_iter().flatten().collect();
                PyArray1::from_vec_bound(py, flat)
                    .reshape([self.height + 1, self.width])
                    .unwrap()
            }
            _ => unreachable!(),
        }
    }

    fn current_player(&self, state: PyReadonlyArray2<i8>) -> i8 {
        let gs = self.array_to_state(&state);
        self.game.current_player(&gs)
    }

    fn legal_actions(&self, state: PyReadonlyArray2<i8>) -> Vec<u16> {
        let gs = self.array_to_state(&state);
        self.game.legal_actions(&gs)
    }

    fn legal_actions_mask(&self, state: PyReadonlyArray2<i8>) -> Vec<bool> {
        let gs = self.array_to_state(&state);
        self.game.legal_actions_mask(&gs)
    }

    fn next_state<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray2<i8>,
        action: u16,
    ) -> Bound<'py, PyArray2<i8>> {
        let gs = self.array_to_state(&state);
        match self.game.next_state(&gs, action) {
            GameState::Go(s) => self.state_to_array(py, &s),
            _ => unreachable!(),
        }
    }

    fn is_terminal(&self, state: PyReadonlyArray2<i8>) -> bool {
        let gs = self.array_to_state(&state);
        self.game.is_terminal(&gs)
    }

    fn terminal_reward(&self, state: PyReadonlyArray2<i8>) -> f32 {
        let gs = self.array_to_state(&state);
        self.game.terminal_reward(&gs)
    }

    fn canonical_state<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray2<i8>,
    ) -> Bound<'py, PyArray2<i8>> {
        let gs = self.array_to_state(&state);
        match self.game.canonical_state(&gs) {
            GameState::Go(s) => self.state_to_array(py, &s),
            _ => unreachable!(),
        }
    }

    fn to_tensor<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray2<i8>,
    ) -> Bound<'py, PyArray1<i64>> {
        let gs = self.array_to_state(&state);
        self.game.to_tensor(&gs).into_pyarray_bound(py)
    }

    fn action_size(&self) -> usize {
        self.game.action_size()
    }

    fn board_size(&self) -> (usize, usize) {
        self.game.board_size()
    }

    fn render(&self, state: PyReadonlyArray2<i8>) -> String {
        let gs = self.array_to_state(&state);
        self.game.render(&gs)
    }
}

impl PyGo {
    fn array_to_state(&self, state: &PyReadonlyArray2<i8>) -> GameState {
        let shape = state.shape();
        let height = shape[0] - 1;
        let width = shape[1];

        let mut go_state = game::state::GoState::new(height, width);
        let slice = state.as_slice().unwrap();

        for row in 0..height {
            for col in 0..width {
                go_state.set(row, col, slice[row * width + col]);
            }
        }

        // Metadata
        let meta_row = height * width;
        go_state.passes = slice[meta_row] as u8;
        go_state.ko_point = (slice[meta_row + 1], slice[meta_row + 2]);
        go_state.turn = slice[meta_row + 3] as u8;

        GameState::Go(go_state)
    }

    fn state_to_array<'py>(
        &self,
        py: Python<'py>,
        s: &game::state::GoState,
    ) -> Bound<'py, PyArray2<i8>> {
        let mut data = vec![0i8; (self.height + 1) * self.width];

        for row in 0..self.height {
            for col in 0..self.width {
                data[row * self.width + col] = s.get(row, col);
            }
        }

        // Metadata
        let meta_row = self.height * self.width;
        data[meta_row] = s.passes as i8;
        data[meta_row + 1] = s.ko_point.0;
        data[meta_row + 2] = s.ko_point.1;
        data[meta_row + 3] = s.turn as i8;

        PyArray1::from_vec_bound(py, data)
            .reshape([self.height + 1, self.width])
            .unwrap()
    }
}

// ============================================================================
// Module Definition
// ============================================================================

/// High-performance Rust backend for NanoZero MCTS.
#[pymodule]
fn nanozero_mcts_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Games
    m.add_class::<PyTicTacToe>()?;
    m.add_class::<PyConnect4>()?;
    m.add_class::<PyGo>()?;

    // Version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
