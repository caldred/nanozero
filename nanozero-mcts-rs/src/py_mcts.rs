//! Python bindings for batched MCTS.
//!
//! Exposes Rust MCTS that runs the full search loop internally,
//! only calling back to Python for neural network inference.

use crate::game::{Game, GameState};
use crate::search::{
    apply_virtual_loss, backup_with_virtual_loss_removal, select_child_with_virtual_loss,
    SearchPath,
};
use crate::tree::TreeArena;
use crate::math::add_dirichlet_noise;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rand::SeedableRng;

/// Python wrapper for batched MCTS with PUCT selection.
///
/// Runs the entire MCTS search in Rust, only calling back to Python
/// for neural network inference. This minimizes Python-Rust boundary
/// crossings for maximum performance.
#[pyclass(name = "RustBatchedMCTS")]
pub struct PyBatchedMCTS {
    /// Exploration constant for UCB
    c_puct: f32,
    /// Dirichlet noise alpha
    dirichlet_alpha: f32,
    /// Dirichlet noise mixing weight
    dirichlet_epsilon: f32,
    /// Number of simulations per search
    num_simulations: u32,
    /// Number of leaves to collect per NN call (virtual loss batching)
    leaves_per_batch: u32,
    /// Virtual loss value (how much to penalize in-flight paths)
    virtual_loss_value: f32,
    /// Random number generator
    rng: rand::rngs::StdRng,
}

#[pymethods]
impl PyBatchedMCTS {
    /// Create a new batched MCTS instance.
    ///
    /// Args:
    ///     c_puct: Exploration constant for UCB (default 1.0)
    ///     dirichlet_alpha: Dirichlet noise alpha (default 0.3)
    ///     dirichlet_epsilon: Dirichlet noise mixing weight (default 0.25)
    ///     num_simulations: Number of simulations per search (default 100)
    ///     leaves_per_batch: Number of leaves to collect per NN call (default 0 = auto)
    ///     virtual_loss_value: Virtual loss penalty value (default 1.0)
    ///     seed: Random seed (optional)
    #[new]
    #[pyo3(signature = (c_puct=1.0, dirichlet_alpha=0.3, dirichlet_epsilon=0.25, num_simulations=100, leaves_per_batch=0, virtual_loss_value=1.0, seed=None))]
    fn new(
        c_puct: f32,
        dirichlet_alpha: f32,
        dirichlet_epsilon: f32,
        num_simulations: u32,
        leaves_per_batch: u32,
        virtual_loss_value: f32,
        seed: Option<u64>,
    ) -> Self {
        let rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };

        Self {
            c_puct,
            dirichlet_alpha,
            dirichlet_epsilon,
            num_simulations,
            leaves_per_batch,
            virtual_loss_value,
            rng,
        }
    }

    /// Run MCTS search on a batch of TicTacToe states.
    ///
    /// Runs the entire search in Rust, only calling the nn_callback
    /// for neural network inference.
    ///
    /// Args:
    ///     states: Batch of game states, shape (batch_size, 9)
    ///     nn_callback: Python callable that takes (states_tensor, legal_masks)
    ///                  and returns (policies, values)
    ///     num_simulations: Override number of simulations (optional)
    ///     add_noise: Whether to add Dirichlet noise at roots (default True)
    ///
    /// Returns:
    ///     Policies array of shape (batch_size, action_size)
    #[pyo3(signature = (states, nn_callback, num_simulations=None, add_noise=true))]
    fn search_tictactoe<'py>(
        &mut self,
        py: Python<'py>,
        states: PyReadonlyArray2<i8>,
        nn_callback: PyObject,
        num_simulations: Option<u32>,
        add_noise: bool,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let num_sims = num_simulations.unwrap_or(self.num_simulations);
        let game = crate::game::TicTacToe::new();

        self.search_internal(py, states, nn_callback, num_sims, add_noise, &game)
    }

    /// Run MCTS search on a batch of Connect4 states.
    #[pyo3(signature = (states, nn_callback, num_simulations=None, add_noise=true))]
    fn search_connect4<'py>(
        &mut self,
        py: Python<'py>,
        states: PyReadonlyArray2<i8>,
        nn_callback: PyObject,
        num_simulations: Option<u32>,
        add_noise: bool,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let num_sims = num_simulations.unwrap_or(self.num_simulations);
        let game = crate::game::Connect4::new();

        self.search_internal(py, states, nn_callback, num_sims, add_noise, &game)
    }
}

impl PyBatchedMCTS {
    /// Internal search implementation generic over game type.
    fn search_internal<'py, G: Game>(
        &mut self,
        py: Python<'py>,
        states: PyReadonlyArray2<i8>,
        nn_callback: PyObject,
        num_simulations: u32,
        add_noise: bool,
        game: &G,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let states_arr = states.as_array();
        let num_states = states_arr.nrows();
        let action_size = game.action_size();

        // Convert input states to GameStates
        let mut game_states: Vec<GameState> = Vec::with_capacity(num_states);
        for i in 0..num_states {
            let row = states_arr.row(i);
            let state = game.state_from_slice(row.as_slice().unwrap());
            game_states.push(state);
        }

        // Initialize search trees
        let mut arenas: Vec<TreeArena> = Vec::with_capacity(num_states);
        for _ in 0..num_states {
            let mut arena = TreeArena::new(1024);
            arena.new_root();
            arenas.push(arena);
        }

        // Expand roots with NN
        let (root_policies, _root_values) = self.call_nn(
            py, &nn_callback, &game_states, game
        )?;

        for (state_idx, arena) in arenas.iter_mut().enumerate() {
            let state = &game_states[state_idx];
            let policy = &root_policies[state_idx];

            // Get legal actions and priors
            let legal_actions = game.legal_actions(state);
            let mut actions: Vec<u16> = Vec::new();
            let mut priors: Vec<f32> = Vec::new();

            for action in legal_actions {
                actions.push(action);
                priors.push(policy[action as usize]);
            }

            // Renormalize
            let sum: f32 = priors.iter().sum();
            if sum > 0.0 {
                for p in priors.iter_mut() {
                    *p /= sum;
                }
            }

            arena.add_children(0, &actions, &priors);
        }

        // Add Dirichlet noise to roots
        if add_noise && self.dirichlet_alpha > 0.0 && self.dirichlet_epsilon > 0.0 {
            for arena in &mut arenas {
                let root_idx = 0u32;
                let child_indices: Vec<u32> = arena
                    .get_children(root_idx)
                    .iter()
                    .map(|c| c.node_idx)
                    .collect();

                if child_indices.is_empty() {
                    continue;
                }

                let mut priors: Vec<f32> = child_indices
                    .iter()
                    .map(|&idx| arena.get(idx).prior)
                    .collect();

                add_dirichlet_noise(
                    &mut priors,
                    self.dirichlet_alpha,
                    self.dirichlet_epsilon,
                    &mut self.rng,
                );

                for (&child_idx, &new_prior) in child_indices.iter().zip(priors.iter()) {
                    arena.get_mut(child_idx).prior = new_prior;
                }
            }
        }

        // Track which root states are terminal
        let root_terminal: Vec<bool> = game_states
            .iter()
            .map(|s| game.is_terminal(s))
            .collect();

        // Determine leaves per batch:
        // - If user specified > 0, use that
        // - Otherwise, use num_states * 8 (8 leaves per game per batch)
        // Higher batching reduces NN callback overhead significantly
        let leaves_per_batch = if self.leaves_per_batch > 0 {
            self.leaves_per_batch as usize
        } else {
            num_states * 8
        };

        // Run simulations with virtual loss batching
        let mut sims_completed = 0u32;
        while sims_completed < num_simulations {
            // Collect leaves with virtual loss until we have enough or run out
            let mut leaves_to_expand: Vec<(usize, SearchPath, GameState)> = Vec::new();
            let mut terminal_backups: Vec<(usize, SearchPath, f32)> = Vec::new();

            // Track how many leaves we've collected from each game tree
            let mut game_leaf_counts: Vec<u32> = vec![0; num_states];

            // Collect leaves across all games, cycling through them
            let mut attempts = 0;
            let max_attempts = leaves_per_batch * 2; // Prevent infinite loop

            while leaves_to_expand.len() < leaves_per_batch
                && attempts < max_attempts
                && (sims_completed + leaves_to_expand.len() as u32 + terminal_backups.len() as u32)
                    < num_simulations
            {
                for state_idx in 0..num_states {
                    if root_terminal[state_idx] {
                        continue;
                    }

                    if leaves_to_expand.len() >= leaves_per_batch {
                        break;
                    }

                    let arena = &mut arenas[state_idx];
                    let root_idx = 0u32;
                    let mut path = SearchPath::from_root(root_idx);
                    let mut node_idx = root_idx;
                    let mut current_state = game_states[state_idx].clone();

                    // SELECT: traverse tree using virtual loss
                    loop {
                        // Check if current state is terminal
                        if game.is_terminal(&current_state) {
                            let value = game.terminal_reward(&current_state);
                            terminal_backups.push((state_idx, path.clone(), value));
                            break;
                        }

                        let node = arena.get(node_idx);

                        // Check if needs expansion
                        if !node.expanded() {
                            // Apply virtual loss to this path
                            apply_virtual_loss(arena, &path);
                            leaves_to_expand.push((state_idx, path, current_state));
                            game_leaf_counts[state_idx] += 1;
                            break;
                        }

                        // Select best child using virtual loss
                        let (action, child_idx) = select_child_with_virtual_loss(
                            arena,
                            node_idx,
                            self.c_puct,
                            self.virtual_loss_value,
                        );
                        path.push(action, child_idx);
                        node_idx = child_idx;

                        // Apply action to state
                        current_state = game.next_state(&current_state, action);
                    }
                }
                attempts += 1;
            }

            // Backup terminal leaves immediately (no NN needed)
            for (state_idx, path, value) in terminal_backups.drain(..) {
                // Terminal paths don't have virtual loss applied, so use regular backup
                use crate::search::backup;
                backup(&mut arenas[state_idx], &path, value);
                sims_completed += 1;
            }

            // Expand non-terminal leaves with batched NN call
            if !leaves_to_expand.is_empty() {
                let leaf_states: Vec<GameState> = leaves_to_expand
                    .iter()
                    .map(|(_, _, s)| s.clone())
                    .collect();

                let (policies, values) = self.call_nn(py, &nn_callback, &leaf_states, game)?;

                for (i, (state_idx, path, leaf_state)) in leaves_to_expand.iter().enumerate() {
                    let arena = &mut arenas[*state_idx];
                    let leaf_idx = path.leaf();
                    let policy = &policies[i];
                    let value = values[i];

                    // Add children to leaf
                    let legal_actions = game.legal_actions(leaf_state);
                    let mut actions: Vec<u16> = Vec::new();
                    let mut priors: Vec<f32> = Vec::new();

                    for action in legal_actions {
                        actions.push(action);
                        priors.push(policy[action as usize]);
                    }

                    // Renormalize
                    let sum: f32 = priors.iter().sum();
                    if sum > 0.0 {
                        for p in priors.iter_mut() {
                            *p /= sum;
                        }
                    }

                    arena.add_children(leaf_idx, &actions, &priors);

                    // Backup value and remove virtual loss
                    backup_with_virtual_loss_removal(arena, path, value);
                    sims_completed += 1;
                }
            }

            // Safety: if nothing was collected, break to avoid infinite loop
            if leaves_to_expand.is_empty() && terminal_backups.is_empty() {
                break;
            }
        }

        // Extract policies from visit counts
        let mut result = vec![0.0f32; num_states * action_size];
        for (state_idx, arena) in arenas.iter().enumerate() {
            let root_idx = 0u32;
            let children = arena.get_children(root_idx);

            let total_visits: u32 = children
                .iter()
                .map(|c| arena.get(c.node_idx).visit_count)
                .sum();

            if total_visits > 0 {
                for child_entry in children {
                    let child = arena.get(child_entry.node_idx);
                    let offset = state_idx * action_size + child_entry.action as usize;
                    result[offset] = child.visit_count as f32 / total_visits as f32;
                }
            }
        }

        Ok(PyArray1::from_vec_bound(py, result)
            .reshape([num_states, action_size])
            .unwrap())
    }

    /// Call the Python NN callback with batched states.
    fn call_nn<G: Game>(
        &self,
        py: Python<'_>,
        nn_callback: &PyObject,
        states: &[GameState],
        game: &G,
    ) -> PyResult<(Vec<Vec<f32>>, Vec<f32>)> {
        let batch_size = states.len();
        let action_size = game.action_size();

        // Prepare canonical states as i64 tensors (for model input)
        let mut state_data: Vec<i64> = Vec::with_capacity(batch_size * game.board_size());
        for state in states {
            let canonical = game.canonical_state(state);
            let tensor = game.to_tensor(&canonical);
            state_data.extend(tensor);
        }

        // Prepare legal action masks
        let mut mask_data: Vec<bool> = Vec::with_capacity(batch_size * action_size);
        for state in states {
            let mask = game.legal_actions_mask(state);
            mask_data.extend(mask);
        }

        // Create numpy arrays
        let states_array = PyArray1::from_vec_bound(py, state_data)
            .reshape([batch_size, game.board_size()])
            .unwrap();
        let masks_array = PyArray1::from_vec_bound(py, mask_data)
            .reshape([batch_size, action_size])
            .unwrap();

        // Call Python callback
        let result = nn_callback.call1(py, (states_array, masks_array))?;
        let tuple = result.downcast_bound::<PyTuple>(py)?;

        // Extract policies and values
        let policies_arr: PyReadonlyArray2<f32> = tuple.get_item(0)?.extract()?;
        let values_arr: PyReadonlyArray1<f32> = tuple.get_item(1)?.extract()?;

        let policies_view = policies_arr.as_array();
        let values_view = values_arr.as_array();

        let mut policies: Vec<Vec<f32>> = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            policies.push(policies_view.row(i).to_vec());
        }

        let values: Vec<f32> = values_view.to_vec();

        Ok((policies, values))
    }
}

// For backwards compatibility - keep the old simpler structs
/// Information about a leaf that needs expansion, returned to Python.
#[pyclass]
#[derive(Clone)]
pub struct PyLeafInfo {
    #[pyo3(get)]
    pub state_idx: usize,
    #[pyo3(get)]
    pub actions: Vec<u16>,
}

/// Result of batch selection, returned to Python.
#[pyclass]
pub struct PySelectResult {
    #[pyo3(get)]
    pub leaves: Vec<PyLeafInfo>,
    #[pyo3(get)]
    pub num_leaves: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_py_batched_mcts_creation() {
        let mcts = PyBatchedMCTS::new(1.0, 0.3, 0.25, 100, Some(42));
        assert_eq!(mcts.c_puct, 1.0);
        assert_eq!(mcts.num_simulations, 100);
    }
}
