//! Python bindings for batched MCTS.
//!
//! Exposes Rust MCTS that runs the full search loop internally,
//! only calling back to Python for neural network inference.

use crate::bayesian_node::create_bayesian_children;
use crate::bayesian_search::{
    apply_bayesian_virtual_loss, bayesian_backup_with_virtual_loss_removal, get_bayesian_policy,
    select_child_thompson_ids_with_virtual_loss, should_stop_early, BayesianSearchPath,
    BayesianTreeArena,
};
use crate::game::{compute_hash, Game, GameState};
use crate::math::add_dirichlet_noise;
use crate::search::{
    apply_virtual_loss, backup_with_virtual_loss_removal, select_child_with_virtual_loss,
    SearchPath,
};
use crate::transposition_table::{ChildStats, TranspositionTable};
use crate::tree::TreeArena;
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
    /// Transposition table for caching positions (persists across searches)
    transposition_table: TranspositionTable,
    /// Whether to use the transposition table
    use_transposition_table: bool,
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
    ///     use_transposition_table: Whether to use the transposition table (default true)
    #[new]
    #[pyo3(signature = (c_puct=1.0, dirichlet_alpha=0.3, dirichlet_epsilon=0.25, num_simulations=100, leaves_per_batch=0, virtual_loss_value=1.0, seed=None, use_transposition_table=true))]
    fn new(
        c_puct: f32,
        dirichlet_alpha: f32,
        dirichlet_epsilon: f32,
        num_simulations: u32,
        leaves_per_batch: u32,
        virtual_loss_value: f32,
        seed: Option<u64>,
        use_transposition_table: bool,
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
            transposition_table: TranspositionTable::with_capacity(10000),
            use_transposition_table,
        }
    }

    /// Clear the transposition table.
    ///
    /// Call this when the model is retrained to invalidate cached evaluations.
    fn clear_cache(&mut self) {
        self.transposition_table.clear();
    }

    /// Get transposition table statistics.
    ///
    /// Returns: (hits, misses, num_entries)
    fn cache_stats(&self) -> (u64, u64, usize) {
        self.transposition_table.stats()
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

    /// Run MCTS search on a batch of Go states.
    ///
    /// States should be flattened Go boards with metadata appended.
    /// For a 9x9 board: 81 board cells + 4 metadata = 85 elements per state.
    #[pyo3(signature = (states, nn_callback, board_size=9, num_simulations=None, add_noise=true))]
    fn search_go<'py>(
        &mut self,
        py: Python<'py>,
        states: PyReadonlyArray2<i8>,
        nn_callback: PyObject,
        board_size: usize,
        num_simulations: Option<u32>,
        add_noise: bool,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let num_sims = num_simulations.unwrap_or(self.num_simulations);
        let game = crate::game::Go::new(board_size);

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

        // Convert input states to GameStates and compute canonical hashes
        let mut game_states: Vec<GameState> = Vec::with_capacity(num_states);
        let mut root_hashes: Vec<u64> = Vec::with_capacity(num_states);
        let mut root_sym_indices: Vec<usize> = Vec::with_capacity(num_states);

        for i in 0..num_states {
            let row = states_arr.row(i);
            let state = game.state_from_slice(row.as_slice().unwrap());
            let canonical = game.canonical_state(&state);

            // Compute canonical symmetry hash
            if self.use_transposition_table {
                let (sym_idx, hash) = game.canonical_symmetry_index(&canonical);
                root_hashes.push(hash);
                root_sym_indices.push(sym_idx);
            } else {
                root_hashes.push(0);
                root_sym_indices.push(0);
            }

            game_states.push(state);
        }

        // Initialize search trees
        let mut arenas: Vec<TreeArena> = Vec::with_capacity(num_states);
        for _ in 0..num_states {
            let mut arena = TreeArena::new(1024);
            arena.new_root();
            arenas.push(arena);
        }

        // Check transposition table for cached root evaluations
        let mut roots_need_nn: Vec<usize> = Vec::new();
        let mut cached_policies: Vec<Option<Vec<f32>>> = vec![None; num_states];

        if self.use_transposition_table {
            for state_idx in 0..num_states {
                let hash = root_hashes[state_idx];
                let sym_idx = root_sym_indices[state_idx];

                if let Some(entry) = self.transposition_table.get(hash) {
                    if entry.expanded() {
                        // Use cached policy, mapping actions through symmetry
                        let mut policy = vec![0.0f32; action_size];
                        for child in &entry.children {
                            // Unmap action from canonical space to original space
                            let orig_action = game.unmap_action(child.action, sym_idx);
                            policy[orig_action as usize] = child.prior;
                        }
                        cached_policies[state_idx] = Some(policy);
                        continue;
                    }
                }
                roots_need_nn.push(state_idx);
            }
        } else {
            roots_need_nn = (0..num_states).collect();
        }

        // Call NN only for states not in cache
        if !roots_need_nn.is_empty() {
            let states_for_nn: Vec<GameState> = roots_need_nn
                .iter()
                .map(|&i| game_states[i].clone())
                .collect();

            let (policies, _values) = self.call_nn(py, &nn_callback, &states_for_nn, game)?;

            for (nn_idx, &state_idx) in roots_need_nn.iter().enumerate() {
                cached_policies[state_idx] = Some(policies[nn_idx].clone());

                // Store in transposition table
                if self.use_transposition_table {
                    let hash = root_hashes[state_idx];
                    let sym_idx = root_sym_indices[state_idx];
                    let state = &game_states[state_idx];
                    let policy = &policies[nn_idx];

                    let entry = self.transposition_table.get_or_insert(hash);
                    let legal_actions = game.legal_actions(state);

                    for action in legal_actions {
                        // Map action to canonical space for storage
                        let canon_action = game.map_action(action, sym_idx);
                        entry.children.push(ChildStats {
                            action: canon_action,
                            prior: policy[action as usize],
                            visits: 0,
                            value_sum: 0.0,
                        });
                    }
                }
            }
        }

        // Initialize trees with policies
        for (state_idx, arena) in arenas.iter_mut().enumerate() {
            let state = &game_states[state_idx];
            let policy = cached_policies[state_idx].as_ref().unwrap();

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
                // Compute canonical hashes for all leaves
                let mut leaf_hashes: Vec<u64> = Vec::with_capacity(leaves_to_expand.len());
                let mut leaf_sym_indices: Vec<usize> = Vec::with_capacity(leaves_to_expand.len());
                let mut leaves_need_nn: Vec<usize> = Vec::new();
                let mut leaf_policies: Vec<Option<(Vec<f32>, f32)>> = vec![None; leaves_to_expand.len()];

                for (leaf_idx, (_, _, leaf_state)) in leaves_to_expand.iter().enumerate() {
                    let canonical = game.canonical_state(leaf_state);

                    if self.use_transposition_table {
                        let (sym_idx, hash) = game.canonical_symmetry_index(&canonical);
                        leaf_hashes.push(hash);
                        leaf_sym_indices.push(sym_idx);

                        // Check if cached
                        if let Some(entry) = self.transposition_table.get(hash) {
                            if entry.expanded() {
                                let mut policy = vec![0.0f32; action_size];
                                for child in &entry.children {
                                    let orig_action = game.unmap_action(child.action, sym_idx);
                                    policy[orig_action as usize] = child.prior;
                                }
                                leaf_policies[leaf_idx] = Some((policy, entry.value()));
                                continue;
                            }
                        }
                    } else {
                        leaf_hashes.push(0);
                        leaf_sym_indices.push(0);
                    }
                    leaves_need_nn.push(leaf_idx);
                }

                // Call NN only for uncached leaves
                if !leaves_need_nn.is_empty() {
                    let states_for_nn: Vec<GameState> = leaves_need_nn
                        .iter()
                        .map(|&i| leaves_to_expand[i].2.clone())
                        .collect();

                    let (policies, values) = self.call_nn(py, &nn_callback, &states_for_nn, game)?;

                    for (nn_idx, &leaf_idx) in leaves_need_nn.iter().enumerate() {
                        let policy = policies[nn_idx].clone();
                        let value = values[nn_idx];
                        leaf_policies[leaf_idx] = Some((policy.clone(), value));

                        // Store in transposition table
                        if self.use_transposition_table {
                            let hash = leaf_hashes[leaf_idx];
                            let sym_idx = leaf_sym_indices[leaf_idx];
                            let leaf_state = &leaves_to_expand[leaf_idx].2;

                            let entry = self.transposition_table.get_or_insert(hash);
                            entry.visits = 1;
                            entry.value_sum = value;

                            let legal_actions = game.legal_actions(leaf_state);
                            for action in legal_actions {
                                let canon_action = game.map_action(action, sym_idx);
                                entry.children.push(ChildStats {
                                    action: canon_action,
                                    prior: policy[action as usize],
                                    visits: 0,
                                    value_sum: 0.0,
                                });
                            }
                        }
                    }
                }

                // Expand all leaves
                for (i, (state_idx, path, leaf_state)) in leaves_to_expand.iter().enumerate() {
                    let arena = &mut arenas[*state_idx];
                    let leaf_node_idx = path.leaf();
                    let (policy, value) = leaf_policies[i].as_ref().unwrap();

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

                    arena.add_children(leaf_node_idx, &actions, &priors);

                    // Backup value and remove virtual loss
                    backup_with_virtual_loss_removal(arena, path, *value);
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

// ============================================================================
// Bayesian MCTS (TTTS) Python Bindings
// ============================================================================

/// Python wrapper for Bayesian MCTS with Thompson Sampling.
///
/// Uses Gaussian beliefs instead of visit counts, and Top-Two Thompson
/// Sampling with IDS allocation for selection. Optimizes for best arm
/// identification rather than cumulative regret.
#[pyclass(name = "RustBayesianMCTS")]
pub struct PyBayesianMCTS {
    /// Number of simulations per search
    num_simulations: u32,
    /// Prior std for logit-shifted initialization
    sigma_0: f32,
    /// Observation variance (NN value uncertainty)
    obs_var: f32,
    /// IDS pseudocount
    ids_alpha: f32,
    /// Soft-prune threshold for aggregation
    prune_threshold: f32,
    /// Whether to stop early when confident
    early_stopping: bool,
    /// P(leader is optimal) threshold for early stopping
    confidence_threshold: f32,
    /// Minimum simulations before early stopping
    min_simulations: u32,
    /// Floor for variance (numerical stability)
    min_variance: f32,
    /// Number of leaves to collect per NN call
    leaves_per_batch: u32,
    /// Virtual loss value for parallel selection
    virtual_loss_value: f32,
    /// Random number generator
    rng: rand::rngs::StdRng,
    /// Transposition table for caching positions (persists across searches)
    transposition_table: TranspositionTable,
    /// Whether to use the transposition table
    use_transposition_table: bool,
}

#[pymethods]
impl PyBayesianMCTS {
    /// Create a new Bayesian MCTS instance.
    ///
    /// Args:
    ///     num_simulations: Number of simulations per search (default 1000)
    ///     sigma_0: Prior std for logit-shifted init (default 1.0)
    ///     obs_var: Observation variance (default 1.0)
    ///     ids_alpha: IDS pseudocount (default 0.0)
    ///     prune_threshold: Soft-prune threshold (default 0.01)
    ///     early_stopping: Whether to stop early (default True)
    ///     confidence_threshold: P(optimal) threshold for stopping (default 0.95)
    ///     min_simulations: Min sims before early stopping (default 10)
    ///     min_variance: Variance floor (default 1e-6)
    ///     leaves_per_batch: Leaves per NN call, 0 = auto (default 0)
    ///     virtual_loss_value: Virtual loss penalty for parallel selection (default 1.0)
    ///     seed: Random seed (optional)
    ///     use_transposition_table: Whether to use the transposition table (default true)
    #[new]
    #[pyo3(signature = (
        num_simulations=1000,
        sigma_0=1.0,
        obs_var=1.0,
        ids_alpha=0.0,
        prune_threshold=0.01,
        early_stopping=true,
        confidence_threshold=0.95,
        min_simulations=10,
        min_variance=1e-6,
        leaves_per_batch=0,
        virtual_loss_value=1.0,
        seed=None,
        use_transposition_table=true
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_simulations: u32,
        sigma_0: f32,
        obs_var: f32,
        ids_alpha: f32,
        prune_threshold: f32,
        early_stopping: bool,
        confidence_threshold: f32,
        min_simulations: u32,
        min_variance: f32,
        leaves_per_batch: u32,
        virtual_loss_value: f32,
        seed: Option<u64>,
        use_transposition_table: bool,
    ) -> Self {
        let rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };

        Self {
            num_simulations,
            sigma_0,
            obs_var,
            ids_alpha,
            prune_threshold,
            early_stopping,
            confidence_threshold,
            min_simulations,
            min_variance,
            leaves_per_batch,
            virtual_loss_value,
            rng,
            transposition_table: TranspositionTable::with_capacity(10000),
            use_transposition_table,
        }
    }

    /// Clear the transposition table.
    ///
    /// Call this when the model is retrained to invalidate cached evaluations.
    fn clear_cache(&mut self) {
        self.transposition_table.clear();
    }

    /// Get transposition table statistics.
    ///
    /// Returns: (hits, misses, num_entries)
    fn cache_stats(&self) -> (u64, u64, usize) {
        self.transposition_table.stats()
    }

    /// Run Bayesian MCTS on a batch of TicTacToe states.
    #[pyo3(signature = (states, nn_callback, num_simulations=None))]
    fn search_tictactoe<'py>(
        &mut self,
        py: Python<'py>,
        states: PyReadonlyArray2<i8>,
        nn_callback: PyObject,
        num_simulations: Option<u32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let num_sims = num_simulations.unwrap_or(self.num_simulations);
        let game = crate::game::TicTacToe::new();
        self.search_internal(py, states, nn_callback, num_sims, &game)
    }

    /// Run Bayesian MCTS on a batch of Connect4 states.
    #[pyo3(signature = (states, nn_callback, num_simulations=None))]
    fn search_connect4<'py>(
        &mut self,
        py: Python<'py>,
        states: PyReadonlyArray2<i8>,
        nn_callback: PyObject,
        num_simulations: Option<u32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let num_sims = num_simulations.unwrap_or(self.num_simulations);
        let game = crate::game::Connect4::new();
        self.search_internal(py, states, nn_callback, num_sims, &game)
    }

    /// Run Bayesian MCTS on a batch of Go states.
    #[pyo3(signature = (states, nn_callback, board_size=9, num_simulations=None))]
    fn search_go<'py>(
        &mut self,
        py: Python<'py>,
        states: PyReadonlyArray2<i8>,
        nn_callback: PyObject,
        board_size: usize,
        num_simulations: Option<u32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let num_sims = num_simulations.unwrap_or(self.num_simulations);
        let game = crate::game::Go::new(board_size);
        self.search_internal(py, states, nn_callback, num_sims, &game)
    }
}

impl PyBayesianMCTS {
    /// Internal search implementation generic over game type.
    fn search_internal<'py, G: Game>(
        &mut self,
        py: Python<'py>,
        states: PyReadonlyArray2<i8>,
        nn_callback: PyObject,
        num_simulations: u32,
        game: &G,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let states_arr = states.as_array();
        let num_states = states_arr.nrows();
        let action_size = game.action_size();

        // Convert input states to GameStates and compute canonical hashes
        let mut game_states: Vec<GameState> = Vec::with_capacity(num_states);
        let mut root_hashes: Vec<u64> = Vec::with_capacity(num_states);
        let mut root_sym_indices: Vec<usize> = Vec::with_capacity(num_states);

        for i in 0..num_states {
            let row = states_arr.row(i);
            let state = game.state_from_slice(row.as_slice().unwrap());
            let canonical = game.canonical_state(&state);

            if self.use_transposition_table {
                let (sym_idx, hash) = game.canonical_symmetry_index(&canonical);
                root_hashes.push(hash);
                root_sym_indices.push(sym_idx);
            } else {
                root_hashes.push(0);
                root_sym_indices.push(0);
            }

            game_states.push(state);
        }

        // Initialize Bayesian search trees
        let mut arenas: Vec<BayesianTreeArena> = Vec::with_capacity(num_states);
        for _ in 0..num_states {
            let mut arena = BayesianTreeArena::new(1024);
            arena.new_root();
            arenas.push(arena);
        }

        // Check transposition table for cached root evaluations
        let mut roots_need_nn: Vec<usize> = Vec::new();
        let mut cached_policies: Vec<Option<(Vec<f32>, f32)>> = vec![None; num_states];

        if self.use_transposition_table {
            for state_idx in 0..num_states {
                let hash = root_hashes[state_idx];
                let sym_idx = root_sym_indices[state_idx];

                if let Some(entry) = self.transposition_table.get(hash) {
                    if entry.expanded() {
                        let mut policy = vec![0.0f32; action_size];
                        for child in &entry.children {
                            let orig_action = game.unmap_action(child.action, sym_idx);
                            policy[orig_action as usize] = child.prior;
                        }
                        cached_policies[state_idx] = Some((policy, entry.value()));
                        continue;
                    }
                }
                roots_need_nn.push(state_idx);
            }
        } else {
            roots_need_nn = (0..num_states).collect();
        }

        // Call NN only for states not in cache
        if !roots_need_nn.is_empty() {
            let states_for_nn: Vec<GameState> = roots_need_nn
                .iter()
                .map(|&i| game_states[i].clone())
                .collect();

            let (policies, values) = self.call_nn(py, &nn_callback, &states_for_nn, game)?;

            for (nn_idx, &state_idx) in roots_need_nn.iter().enumerate() {
                let policy = policies[nn_idx].clone();
                let value = values[nn_idx];
                cached_policies[state_idx] = Some((policy.clone(), value));

                // Store in transposition table
                if self.use_transposition_table {
                    let hash = root_hashes[state_idx];
                    let sym_idx = root_sym_indices[state_idx];
                    let state = &game_states[state_idx];

                    let entry = self.transposition_table.get_or_insert(hash);
                    entry.visits = 1;
                    entry.value_sum = value;

                    let legal_actions = game.legal_actions(state);
                    for action in legal_actions {
                        let canon_action = game.map_action(action, sym_idx);
                        entry.children.push(ChildStats {
                            action: canon_action,
                            prior: policy[action as usize],
                            visits: 0,
                            value_sum: 0.0,
                        });
                    }
                }
            }
        }

        // Initialize trees with policies
        for (state_idx, arena) in arenas.iter_mut().enumerate() {
            let state = &game_states[state_idx];
            let (policy, value) = cached_policies[state_idx].as_ref().unwrap();

            let legal_actions = game.legal_actions(state);
            let mut actions: Vec<u16> = Vec::new();
            let mut priors: Vec<f32> = Vec::new();

            for action in legal_actions {
                actions.push(action);
                priors.push(policy[action as usize]);
            }

            // Renormalize priors
            let sum: f32 = priors.iter().sum();
            if sum > 0.0 {
                for p in priors.iter_mut() {
                    *p /= sum;
                }
            }

            // Create children with logit-shifted initialization
            let children_params = create_bayesian_children(*value, &priors, self.sigma_0);
            arena.add_children(0, &actions, &children_params);

            // Initialize aggregated belief at root
            arena.update_aggregated(0, self.prune_threshold, false);
        }

        // Track which root states are terminal
        let root_terminal: Vec<bool> = game_states
            .iter()
            .map(|s| game.is_terminal(s))
            .collect();

        // Track which states are still active (not stopped early)
        let mut active_mask: Vec<bool> = vec![true; num_states];

        // Determine leaves per batch
        let leaves_per_batch = if self.leaves_per_batch > 0 {
            self.leaves_per_batch as usize
        } else {
            num_states * 8
        };

        // Run simulations
        let mut sims_completed = 0u32;
        while sims_completed < num_simulations {
            // Collect leaves with virtual loss batching
            let mut leaves_to_expand: Vec<(usize, BayesianSearchPath, GameState)> = Vec::new();
            let mut terminal_backups: Vec<(usize, BayesianSearchPath, f32)> = Vec::new();

            // Collect leaves from all active, non-terminal states
            let mut attempts = 0;
            let max_attempts = leaves_per_batch * 2;

            while leaves_to_expand.len() < leaves_per_batch
                && attempts < max_attempts
                && (sims_completed + leaves_to_expand.len() as u32 + terminal_backups.len() as u32)
                    < num_simulations
            {
                for state_idx in 0..num_states {
                    if root_terminal[state_idx] || !active_mask[state_idx] {
                        continue;
                    }

                    if leaves_to_expand.len() >= leaves_per_batch {
                        break;
                    }

                    let arena = &mut arenas[state_idx];
                    let root_idx = 0u32;
                    let mut path = BayesianSearchPath::from_root(root_idx);
                    let mut node_idx = root_idx;
                    let mut current_state = game_states[state_idx].clone();

                    // SELECT: traverse tree using Thompson sampling
                    loop {
                        // Check if current state is terminal
                        if game.is_terminal(&current_state) {
                            let value = game.terminal_reward(&current_state);
                            // Apply virtual loss before storing (will be removed during backup)
                            apply_bayesian_virtual_loss(arena, &path);
                            terminal_backups.push((state_idx, path.clone(), value));
                            break;
                        }

                        let node = arena.get(node_idx);

                        // Check if needs expansion
                        if !node.expanded() {
                            // Apply virtual loss to path before storing
                            apply_bayesian_virtual_loss(arena, &path);
                            leaves_to_expand.push((state_idx, path, current_state));
                            break;
                        }

                        // Select using Thompson sampling with IDS and virtual loss
                        let (action, child_idx) = select_child_thompson_ids_with_virtual_loss(
                            arena,
                            node_idx,
                            self.ids_alpha,
                            self.virtual_loss_value,
                            &mut self.rng,
                        );
                        path.push(action, child_idx);
                        node_idx = child_idx;

                        // Apply action to state
                        current_state = game.next_state(&current_state, action);
                    }
                }
                attempts += 1;
            }

            // Backup terminal leaves immediately (also removes virtual loss)
            for (state_idx, path, value) in terminal_backups.drain(..) {
                bayesian_backup_with_virtual_loss_removal(
                    &mut arenas[state_idx],
                    &path,
                    value,
                    self.obs_var,
                    self.min_variance,
                    self.prune_threshold,
                );
                sims_completed += 1;
            }

            // Expand non-terminal leaves with batched NN call
            if !leaves_to_expand.is_empty() {
                // Compute canonical hashes for all leaves
                let mut leaf_hashes: Vec<u64> = Vec::with_capacity(leaves_to_expand.len());
                let mut leaf_sym_indices: Vec<usize> = Vec::with_capacity(leaves_to_expand.len());
                let mut leaves_need_nn: Vec<usize> = Vec::new();
                let mut leaf_policies: Vec<Option<(Vec<f32>, f32)>> = vec![None; leaves_to_expand.len()];

                for (leaf_idx, (_, _, leaf_state)) in leaves_to_expand.iter().enumerate() {
                    let canonical = game.canonical_state(leaf_state);

                    if self.use_transposition_table {
                        let (sym_idx, hash) = game.canonical_symmetry_index(&canonical);
                        leaf_hashes.push(hash);
                        leaf_sym_indices.push(sym_idx);

                        // Check if cached
                        if let Some(entry) = self.transposition_table.get(hash) {
                            if entry.expanded() {
                                let mut policy = vec![0.0f32; action_size];
                                for child in &entry.children {
                                    let orig_action = game.unmap_action(child.action, sym_idx);
                                    policy[orig_action as usize] = child.prior;
                                }
                                leaf_policies[leaf_idx] = Some((policy, entry.value()));
                                continue;
                            }
                        }
                    } else {
                        leaf_hashes.push(0);
                        leaf_sym_indices.push(0);
                    }
                    leaves_need_nn.push(leaf_idx);
                }

                // Call NN only for uncached leaves
                if !leaves_need_nn.is_empty() {
                    let states_for_nn: Vec<GameState> = leaves_need_nn
                        .iter()
                        .map(|&i| leaves_to_expand[i].2.clone())
                        .collect();

                    let (policies, values) = self.call_nn(py, &nn_callback, &states_for_nn, game)?;

                    for (nn_idx, &leaf_idx) in leaves_need_nn.iter().enumerate() {
                        let policy = policies[nn_idx].clone();
                        let value = values[nn_idx];
                        leaf_policies[leaf_idx] = Some((policy.clone(), value));

                        // Store in transposition table
                        if self.use_transposition_table {
                            let hash = leaf_hashes[leaf_idx];
                            let sym_idx = leaf_sym_indices[leaf_idx];
                            let leaf_state = &leaves_to_expand[leaf_idx].2;

                            let entry = self.transposition_table.get_or_insert(hash);
                            entry.visits = 1;
                            entry.value_sum = value;

                            let legal_actions = game.legal_actions(leaf_state);
                            for action in legal_actions {
                                let canon_action = game.map_action(action, sym_idx);
                                entry.children.push(ChildStats {
                                    action: canon_action,
                                    prior: policy[action as usize],
                                    visits: 0,
                                    value_sum: 0.0,
                                });
                            }
                        }
                    }
                }

                // Expand all leaves
                for (i, (state_idx, path, leaf_state)) in leaves_to_expand.iter().enumerate() {
                    let arena = &mut arenas[*state_idx];
                    let leaf_node_idx = path.leaf();
                    let (policy, value) = leaf_policies[i].as_ref().unwrap();

                    // Get legal actions and priors
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

                    // Create children with logit-shifted initialization
                    let children_params = create_bayesian_children(*value, &priors, self.sigma_0);
                    arena.add_children(leaf_node_idx, &actions, &children_params);

                    // Initialize aggregated belief for the expanded node
                    arena.update_aggregated(leaf_node_idx, self.prune_threshold, false);

                    // Backup (also removes virtual loss)
                    bayesian_backup_with_virtual_loss_removal(
                        arena,
                        path,
                        *value,
                        self.obs_var,
                        self.min_variance,
                        self.prune_threshold,
                    );
                    sims_completed += 1;
                }
            }

            // Early stopping check
            if self.early_stopping && sims_completed >= self.min_simulations {
                for state_idx in 0..num_states {
                    if active_mask[state_idx]
                        && !root_terminal[state_idx]
                        && should_stop_early(&arenas[state_idx], 0, self.confidence_threshold)
                    {
                        active_mask[state_idx] = false;
                    }
                }

                // If all states stopped early, exit loop
                if !active_mask.iter().any(|&a| a) {
                    break;
                }
            }

            // Safety: if nothing was collected, break
            if leaves_to_expand.is_empty() && terminal_backups.is_empty() {
                break;
            }
        }

        // Extract policies from optimality weights
        let mut result = vec![0.0f32; num_states * action_size];
        for (state_idx, arena) in arenas.iter().enumerate() {
            let policy = get_bayesian_policy(arena, 0, action_size);
            for (a, &p) in policy.iter().enumerate() {
                result[state_idx * action_size + a] = p;
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

        // Prepare canonical states as i64 tensors
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
        let mcts = PyBatchedMCTS::new(1.0, 0.3, 0.25, 100, 0, 1.0, Some(42), true);
        assert_eq!(mcts.c_puct, 1.0);
        assert_eq!(mcts.num_simulations, 100);
        assert!(mcts.use_transposition_table);
    }

    #[test]
    fn test_py_bayesian_mcts_creation() {
        let mcts = PyBayesianMCTS::new(
            1000,  // num_simulations
            1.0,   // sigma_0
            1.0,   // obs_var
            0.0,   // ids_alpha
            0.01,  // prune_threshold
            true,  // early_stopping
            0.95,  // confidence_threshold
            10,    // min_simulations
            1e-6,  // min_variance
            0,     // leaves_per_batch
            1.0,   // virtual_loss_value
            Some(42),
            true,  // use_transposition_table
        );
        assert_eq!(mcts.num_simulations, 1000);
        assert_eq!(mcts.sigma_0, 1.0);
    }
}
