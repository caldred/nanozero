//! Batch coordination for parallel MCTS.
//!
//! Handles multiple search trees and coordinates batched neural network
//! evaluation across all of them.

use crate::math::add_dirichlet_noise;
use crate::search::{backup, select_child, SearchPath};
use crate::tree::TreeArena;
use rand::SeedableRng;
use smallvec::SmallVec;

/// Configuration for batched MCTS.
#[derive(Clone, Debug)]
pub struct MCTSConfig {
    /// Exploration constant for UCB
    pub c_puct: f32,
    /// Virtual loss value for parallel search
    pub virtual_loss: f32,
    /// Dirichlet noise alpha (0 to disable)
    pub dirichlet_alpha: f32,
    /// Dirichlet noise mixing weight
    pub dirichlet_epsilon: f32,
    /// Number of simulations per search
    pub num_simulations: u32,
}

impl Default for MCTSConfig {
    fn default() -> Self {
        Self {
            c_puct: 1.0,
            virtual_loss: 1.0,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
            num_simulations: 100,
        }
    }
}

/// Information about a leaf that needs expansion.
#[derive(Clone, Debug)]
pub struct LeafInfo {
    /// Index of the state this leaf belongs to
    pub state_idx: usize,
    /// Path from root to this leaf
    pub path: SearchPath,
    /// Actions taken from root state to reach leaf state
    pub actions: SmallVec<[u16; 128]>,
}

/// Result of batch selection.
pub struct BatchSelectResult {
    /// Leaves that need neural network evaluation
    pub leaves_to_expand: Vec<LeafInfo>,
    /// Leaves that are terminal (just need backup)
    pub terminal_leaves: Vec<(usize, SearchPath, f32)>,
}

/// Manages multiple MCTS search trees for batched search.
pub struct BatchedMCTS {
    /// One tree arena per state in the batch
    arenas: Vec<TreeArena>,
    /// Configuration
    config: MCTSConfig,
    /// Random number generator
    rng: rand::rngs::StdRng,
}

impl BatchedMCTS {
    /// Create a new batched MCTS with the given configuration.
    pub fn new(config: MCTSConfig) -> Self {
        Self {
            arenas: Vec::new(),
            config,
            rng: rand::rngs::StdRng::from_entropy(),
        }
    }

    /// Create with a specific seed for reproducibility.
    pub fn with_seed(config: MCTSConfig, seed: u64) -> Self {
        Self {
            arenas: Vec::new(),
            config,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }

    /// Initialize new searches for a batch of states.
    ///
    /// Clears any existing search trees and creates fresh roots.
    pub fn new_search(&mut self, num_states: usize) {
        self.arenas.clear();
        self.arenas.reserve(num_states);
        for _ in 0..num_states {
            let mut arena = TreeArena::new(1024);
            arena.new_root();
            self.arenas.push(arena);
        }
    }

    /// Expand root nodes with policies from neural network.
    ///
    /// # Arguments
    /// * `policies` - Policy for each state, shape (num_states, action_size)
    /// * `legal_masks` - Legal action masks for each state
    pub fn expand_roots(&mut self, policies: &[Vec<f32>], legal_masks: &[Vec<bool>]) {
        debug_assert_eq!(policies.len(), self.arenas.len());
        debug_assert_eq!(legal_masks.len(), self.arenas.len());

        for (arena, (policy, legal_mask)) in self
            .arenas
            .iter_mut()
            .zip(policies.iter().zip(legal_masks.iter()))
        {
            let root_idx = 0; // Root is always at index 0

            // Extract legal actions and their priors
            let mut actions = Vec::new();
            let mut priors = Vec::new();
            for (action, (&is_legal, &prior)) in legal_mask.iter().zip(policy.iter()).enumerate() {
                if is_legal {
                    actions.push(action as u16);
                    priors.push(prior);
                }
            }

            // Renormalize priors over legal actions
            let sum: f32 = priors.iter().sum();
            if sum > 0.0 {
                for p in priors.iter_mut() {
                    *p /= sum;
                }
            }

            arena.add_children(root_idx, &actions, &priors);
        }
    }

    /// Add Dirichlet noise to root nodes.
    pub fn add_dirichlet_noise(&mut self) {
        if self.config.dirichlet_alpha <= 0.0 || self.config.dirichlet_epsilon <= 0.0 {
            return;
        }

        for arena in &mut self.arenas {
            let root_idx = 0u32;

            // Collect child indices and priors first to avoid borrow conflicts
            let child_indices: Vec<u32> = arena
                .get_children(root_idx)
                .iter()
                .map(|c| c.node_idx)
                .collect();

            let mut priors: Vec<f32> = child_indices
                .iter()
                .map(|&idx| arena.get(idx).prior)
                .collect();

            // Add noise
            add_dirichlet_noise(
                &mut priors,
                self.config.dirichlet_alpha,
                self.config.dirichlet_epsilon,
                &mut self.rng,
            );

            // Update node priors
            for (&child_idx, &new_prior) in child_indices.iter().zip(priors.iter()) {
                arena.get_mut(child_idx).prior = new_prior;
            }
        }
    }

    /// Select leaves from all trees for batched evaluation.
    ///
    /// # Arguments
    /// * `terminal_masks` - Which states are terminal (skip selection for these)
    /// * `get_terminal_value` - Function to get terminal value for a state
    /// * `is_terminal_from_actions` - Function to check if actions lead to terminal state
    pub fn select_batch<F, G>(
        &mut self,
        terminal_masks: &[bool],
        get_terminal_value: F,
        is_terminal_from_actions: G,
    ) -> BatchSelectResult
    where
        F: Fn(usize) -> f32,
        G: Fn(usize, &[u16]) -> bool,
    {
        let mut leaves_to_expand = Vec::new();
        let mut terminal_leaves = Vec::new();

        for (state_idx, (arena, &is_root_terminal)) in
            self.arenas.iter().zip(terminal_masks.iter()).enumerate()
        {
            // Skip terminal states
            if is_root_terminal {
                continue;
            }

            let root_idx = 0u32;
            let mut path = SearchPath::from_root(root_idx);
            let mut node_idx = root_idx;

            loop {
                let node = arena.get(node_idx);

                // Check if we reached a terminal state
                if is_terminal_from_actions(state_idx, &path.actions) {
                    let value = get_terminal_value(state_idx);
                    terminal_leaves.push((state_idx, path, value));
                    break;
                }

                // Check if needs expansion
                if !node.expanded() {
                    let actions = path.actions.clone();
                    leaves_to_expand.push(LeafInfo {
                        state_idx,
                        path,
                        actions,
                    });
                    break;
                }

                // Select best child
                let (action, child_idx) = select_child(arena, node_idx, self.config.c_puct);
                path.push(action, child_idx);
                node_idx = child_idx;
            }
        }

        BatchSelectResult {
            leaves_to_expand,
            terminal_leaves,
        }
    }

    /// Expand leaves with policies and values from neural network.
    ///
    /// # Arguments
    /// * `leaf_infos` - Information about leaves to expand
    /// * `policies` - Policies for each leaf
    /// * `legal_masks` - Legal action masks for each leaf
    pub fn expand_leaves(
        &mut self,
        leaf_infos: &[LeafInfo],
        policies: &[Vec<f32>],
        legal_masks: &[Vec<bool>],
    ) {
        for (leaf_info, (policy, legal_mask)) in leaf_infos
            .iter()
            .zip(policies.iter().zip(legal_masks.iter()))
        {
            let arena = &mut self.arenas[leaf_info.state_idx];
            let leaf_idx = leaf_info.path.leaf();

            // Extract legal actions and priors
            let mut actions = Vec::new();
            let mut priors = Vec::new();
            for (action, (&is_legal, &prior)) in legal_mask.iter().zip(policy.iter()).enumerate() {
                if is_legal {
                    actions.push(action as u16);
                    priors.push(prior);
                }
            }

            // Renormalize
            let sum: f32 = priors.iter().sum();
            if sum > 0.0 {
                for p in priors.iter_mut() {
                    *p /= sum;
                }
            }

            arena.add_children(leaf_idx, &actions, &priors);
        }
    }

    /// Backup values through the search trees.
    ///
    /// # Arguments
    /// * `leaf_infos` - Information about leaves
    /// * `values` - Values to backup for each leaf
    pub fn backup_batch(&mut self, leaf_infos: &[LeafInfo], values: &[f32]) {
        for (leaf_info, &value) in leaf_infos.iter().zip(values.iter()) {
            let arena = &mut self.arenas[leaf_info.state_idx];
            backup(arena, &leaf_info.path, value);
        }
    }

    /// Backup terminal leaves.
    pub fn backup_terminal(&mut self, terminal_leaves: &[(usize, SearchPath, f32)]) {
        for (state_idx, path, value) in terminal_leaves {
            let arena = &mut self.arenas[*state_idx];
            backup(arena, path, *value);
        }
    }

    /// Get policies from all search trees.
    ///
    /// Returns visit count distributions for each state.
    pub fn get_policies(&self, action_size: usize) -> Vec<Vec<f32>> {
        self.arenas
            .iter()
            .map(|arena| {
                let mut policy = vec![0.0f32; action_size];
                let root_idx = 0u32;
                let children = arena.get_children(root_idx);

                let total_visits: u32 = children
                    .iter()
                    .map(|c| arena.get(c.node_idx).visit_count)
                    .sum();

                if total_visits > 0 {
                    for child_entry in children {
                        let child = arena.get(child_entry.node_idx);
                        policy[child_entry.action as usize] =
                            child.visit_count as f32 / total_visits as f32;
                    }
                }

                policy
            })
            .collect()
    }

    /// Get the number of states being searched.
    pub fn num_states(&self) -> usize {
        self.arenas.len()
    }

    /// Get visit counts for debugging.
    pub fn get_root_visit_count(&self, state_idx: usize) -> u32 {
        self.arenas[state_idx].get(0).visit_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_search() {
        let mut mcts = BatchedMCTS::new(MCTSConfig::default());
        mcts.new_search(4);
        assert_eq!(mcts.num_states(), 4);
    }

    #[test]
    fn test_expand_roots() {
        let mut mcts = BatchedMCTS::new(MCTSConfig::default());
        mcts.new_search(2);

        let policies = vec![vec![0.5, 0.3, 0.2], vec![0.4, 0.4, 0.2]];
        let legal_masks = vec![vec![true, true, true], vec![true, false, true]];

        mcts.expand_roots(&policies, &legal_masks);

        // Check first state
        let arena = &mcts.arenas[0];
        assert!(arena.get(0).expanded());
        assert_eq!(arena.get_children(0).len(), 3);

        // Check second state (middle action illegal)
        let arena = &mcts.arenas[1];
        assert!(arena.get(0).expanded());
        assert_eq!(arena.get_children(0).len(), 2);
    }

    #[test]
    fn test_get_policies() {
        let mut mcts = BatchedMCTS::new(MCTSConfig::default());
        mcts.new_search(1);

        let policies = vec![vec![0.5, 0.3, 0.2]];
        let legal_masks = vec![vec![true, true, true]];
        mcts.expand_roots(&policies, &legal_masks);

        // Manually add visits
        let arena = &mut mcts.arenas[0];
        let children = arena.get_children(0);
        let child0_idx = children[0].node_idx;
        let child1_idx = children[1].node_idx;
        arena.get_mut(child0_idx).visit_count = 60;
        arena.get_mut(child1_idx).visit_count = 40;

        let result_policies = mcts.get_policies(3);
        assert_eq!(result_policies.len(), 1);
        assert!((result_policies[0][0] - 0.6).abs() < 1e-6);
        assert!((result_policies[0][1] - 0.4).abs() < 1e-6);
        assert!((result_policies[0][2] - 0.0).abs() < 1e-6);
    }
}
