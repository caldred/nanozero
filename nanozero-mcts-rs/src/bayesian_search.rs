//! Bayesian MCTS search operations.
//!
//! Implements Top-Two Thompson Sampling with IDS allocation and
//! variance-propagating backup.

use crate::bayesian_node::{aggregate_children, BayesianNode};
use crate::tree::ChildEntry;
use smallvec::SmallVec;

/// Maximum tree depth for stack allocation
const MAX_DEPTH: usize = 128;

/// Arena for Bayesian MCTS nodes.
#[derive(Debug)]
pub struct BayesianTreeArena {
    nodes: Vec<BayesianNode>,
    children: Vec<ChildEntry>,
}

impl BayesianTreeArena {
    pub fn new(estimated_nodes: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(estimated_nodes),
            children: Vec::with_capacity(estimated_nodes * 4),
        }
    }

    pub fn new_root(&mut self) -> u32 {
        self.allocate_node(1.0, 0.0, 1.0)
    }

    pub fn allocate_node(&mut self, prior: f32, mu: f32, sigma_sq: f32) -> u32 {
        let idx = self.nodes.len() as u32;
        self.nodes.push(BayesianNode::new(prior, mu, sigma_sq));
        idx
    }

    #[inline]
    pub fn get(&self, idx: u32) -> &BayesianNode {
        &self.nodes[idx as usize]
    }

    #[inline]
    pub fn get_mut(&mut self, idx: u32) -> &mut BayesianNode {
        &mut self.nodes[idx as usize]
    }

    pub fn add_children(
        &mut self,
        parent_idx: u32,
        actions: &[u16],
        children_params: &[(f32, f32, f32)], // (prior, mu, sigma_sq)
    ) {
        debug_assert_eq!(actions.len(), children_params.len());

        let children_start = self.children.len() as u32;
        let children_count = actions.len() as u16;

        for (&action, &(prior, mu, sigma_sq)) in actions.iter().zip(children_params.iter()) {
            let child_idx = self.allocate_node(prior, mu, sigma_sq);
            self.children.push(ChildEntry {
                action,
                node_idx: child_idx,
            });
        }

        let parent = self.get_mut(parent_idx);
        parent.children_start = children_start;
        parent.children_count = children_count;
    }

    pub fn get_children(&self, node_idx: u32) -> &[ChildEntry] {
        let node = self.get(node_idx);
        if node.children_count == 0 {
            return &[];
        }
        let start = node.children_start as usize;
        let end = start + node.children_count as usize;
        &self.children[start..end]
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.children.clear();
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Update aggregated beliefs for a node from its children.
    pub fn update_aggregated(&mut self, node_idx: u32, prune_threshold: f32) {
        let children = self.get_children(node_idx);
        if children.is_empty() {
            return;
        }

        // Collect child beliefs from parent's perspective (negate child values)
        let child_beliefs: Vec<(f32, f32)> = children
            .iter()
            .map(|c| {
                let child = self.get(c.node_idx);
                (-child.mu, child.sigma_sq)
            })
            .collect();

        let (agg_mu, agg_sigma_sq) = aggregate_children(&child_beliefs, prune_threshold);

        let node = self.get_mut(node_idx);
        node.agg_mu = Some(agg_mu);
        node.agg_sigma_sq = Some(agg_sigma_sq);
    }
}

impl Default for BayesianTreeArena {
    fn default() -> Self {
        Self::new(1024)
    }
}

/// Path through a Bayesian search tree.
#[derive(Clone, Debug)]
pub struct BayesianSearchPath {
    pub nodes: SmallVec<[u32; MAX_DEPTH]>,
    pub actions: SmallVec<[u16; MAX_DEPTH]>,
}

impl BayesianSearchPath {
    pub fn new() -> Self {
        Self {
            nodes: SmallVec::new(),
            actions: SmallVec::new(),
        }
    }

    pub fn from_root(root_idx: u32) -> Self {
        let mut path = Self::new();
        path.nodes.push(root_idx);
        path
    }

    pub fn push(&mut self, action: u16, node_idx: u32) {
        self.actions.push(action);
        self.nodes.push(node_idx);
    }

    pub fn leaf(&self) -> u32 {
        *self.nodes.last().unwrap()
    }

    pub fn depth(&self) -> usize {
        self.actions.len()
    }
}

impl Default for BayesianSearchPath {
    fn default() -> Self {
        Self::new()
    }
}

/// Top-Two Thompson Sampling with IDS allocation.
///
/// 1. Draw Thompson sample from each child's posterior
/// 2. Leader I = argmax of samples
/// 3. Challenger J = second highest
/// 4. Compute allocation: beta = (precision_I + alpha) / (precision_I + precision_J + 2*alpha)
/// 5. Select Challenger with probability beta, else Leader
pub fn select_child_thompson_ids<R: rand::Rng>(
    arena: &BayesianTreeArena,
    node_idx: u32,
    ids_alpha: f32,
    rng: &mut R,
) -> (u16, u32) {
    let children = arena.get_children(node_idx);
    debug_assert!(!children.is_empty());

    if children.len() == 1 {
        return (children[0].action, children[0].node_idx);
    }

    // Draw Thompson samples (from parent's perspective: negate child values)
    let mut samples: Vec<(u16, u32, f32)> = children
        .iter()
        .map(|c| {
            let child = arena.get(c.node_idx);
            let sample = -child.sample(rng); // Negate for parent's perspective
            (c.action, c.node_idx, sample)
        })
        .collect();

    // Sort by sample value (descending)
    samples.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    let (leader_action, leader_idx, _) = samples[0];
    let (challenger_action, challenger_idx, _) = samples[1];

    // IDS allocation probability
    let leader_precision = arena.get(leader_idx).precision();
    let challenger_precision = arena.get(challenger_idx).precision();

    // beta = probability of selecting challenger
    // High leader precision → explore challenger more
    let beta = (leader_precision + ids_alpha)
        / (leader_precision + challenger_precision + 2.0 * ids_alpha);

    // Select challenger with probability beta
    if rng.gen::<f32>() < beta {
        (challenger_action, challenger_idx)
    } else {
        (leader_action, leader_idx)
    }
}

/// Bayesian backup with variance aggregation.
///
/// For each level (from leaf to root):
/// 1. Update visited child with observed value and propagated variance
/// 2. Recompute parent's aggregated belief from all children
/// 3. Propagate parent's aggregated value AND variance up
pub fn bayesian_backup(
    arena: &mut BayesianTreeArena,
    path: &BayesianSearchPath,
    leaf_value: f32,
    initial_obs_var: f32,
    min_variance: f32,
    prune_threshold: f32,
) {
    if path.nodes.len() < 2 {
        return; // No backup needed for root-only path
    }

    let mut value = leaf_value;
    let mut obs_var = initial_obs_var;

    // Iterate from leaf to root (skip root for update, it has no parent)
    for i in (1..path.nodes.len()).rev() {
        let child_idx = path.nodes[i];
        let parent_idx = path.nodes[i - 1];

        // Update child's belief
        arena.get_mut(child_idx).update(value, obs_var, min_variance);

        // Recompute parent's aggregated belief
        arena.update_aggregated(parent_idx, prune_threshold);

        // Propagate aggregated value and variance up
        let parent = arena.get(parent_idx);
        if let (Some(agg_mu), Some(agg_sigma_sq)) = (parent.agg_mu, parent.agg_sigma_sq) {
            value = agg_mu;
            obs_var = agg_sigma_sq;
        }
    }
}

/// Select to leaf for Bayesian MCTS.
pub fn bayesian_select_to_leaf<R: rand::Rng, F>(
    arena: &BayesianTreeArena,
    root_idx: u32,
    ids_alpha: f32,
    rng: &mut R,
    mut is_terminal_fn: F,
) -> (BayesianSearchPath, bool)
where
    F: FnMut(&[u16]) -> bool,
{
    let mut path = BayesianSearchPath::from_root(root_idx);
    let mut node_idx = root_idx;

    loop {
        let node = arena.get(node_idx);

        // Check if terminal
        if is_terminal_fn(&path.actions) {
            return (path, true);
        }

        // Check if needs expansion
        if !node.expanded() {
            return (path, false);
        }

        // Select using Thompson sampling with IDS
        let (action, child_idx) = select_child_thompson_ids(arena, node_idx, ids_alpha, rng);
        path.push(action, child_idx);
        node_idx = child_idx;
    }
}

/// Get policy from optimality weights.
///
/// Computes P(each child is optimal) using pairwise Gaussian CDF comparisons.
pub fn get_bayesian_policy(arena: &BayesianTreeArena, root_idx: u32, action_size: usize) -> Vec<f32> {
    use crate::math::normal_cdf;

    let mut policy = vec![0.0f32; action_size];
    let children = arena.get_children(root_idx);

    if children.is_empty() {
        return policy;
    }

    if children.len() == 1 {
        policy[children[0].action as usize] = 1.0;
        return policy;
    }

    // Get beliefs from parent's perspective
    let beliefs: Vec<(f32, f32)> = children
        .iter()
        .map(|c| {
            let child = arena.get(c.node_idx);
            (-child.mu, child.sigma_sq) // Negate for parent's perspective
        })
        .collect();

    // Find leader and challenger
    let mut sorted_indices: Vec<usize> = (0..beliefs.len()).collect();
    sorted_indices.sort_by(|&a, &b| {
        beliefs[b]
            .0
            .partial_cmp(&beliefs[a].0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let leader_idx = sorted_indices[0];
    let challenger_idx = sorted_indices[1];

    let (mu_l, sigma_sq_l) = beliefs[leader_idx];
    let (mu_c, sigma_sq_c) = beliefs[challenger_idx];

    // Compute scores
    let mut scores = vec![0.0f32; beliefs.len()];
    for i in 0..beliefs.len() {
        let (mu_i, sigma_sq_i) = beliefs[i];

        let (diff, combined_var) = if i == leader_idx {
            (mu_l - mu_c, sigma_sq_l + sigma_sq_c)
        } else {
            (mu_i - mu_l, sigma_sq_i + sigma_sq_l)
        };

        let std = combined_var.sqrt();
        scores[i] = if std > 1e-10 {
            normal_cdf(diff / std)
        } else if diff > 0.0 {
            1.0
        } else {
            0.0
        };
    }

    // Normalize
    let total: f32 = scores.iter().sum();
    if total < 1e-10 {
        // Uniform fallback
        for child in children {
            policy[child.action as usize] = 1.0 / children.len() as f32;
        }
    } else {
        for (i, child) in children.iter().enumerate() {
            policy[child.action as usize] = scores[i] / total;
        }
    }

    policy
}

/// Check if early stopping condition is met.
///
/// Uses P(leader > challenger) as lower bound on P(leader is optimal).
pub fn should_stop_early(
    arena: &BayesianTreeArena,
    root_idx: u32,
    confidence_threshold: f32,
) -> bool {
    use crate::math::normal_cdf;

    let children = arena.get_children(root_idx);
    if children.len() <= 1 {
        return true;
    }

    // Find leader and challenger by mean value
    let mut beliefs: Vec<(u32, f32, f32)> = children
        .iter()
        .map(|c| {
            let child = arena.get(c.node_idx);
            (c.node_idx, -child.mu, child.sigma_sq) // Negate for parent's perspective
        })
        .collect();

    beliefs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let (_, mu_leader, sigma_sq_leader) = beliefs[0];
    let (_, mu_challenger, sigma_sq_challenger) = beliefs[1];

    let std_diff = (sigma_sq_leader + sigma_sq_challenger).sqrt();
    if std_diff < 1e-10 {
        return mu_leader > mu_challenger;
    }

    let prob_leader_better = normal_cdf((mu_leader - mu_challenger) / std_diff);
    prob_leader_better >= confidence_threshold
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_bayesian_arena() {
        let mut arena = BayesianTreeArena::new(100);
        let root = arena.new_root();
        assert_eq!(root, 0);
        assert!(!arena.get(root).expanded());
    }

    #[test]
    fn test_thompson_selection() {
        let mut arena = BayesianTreeArena::new(100);
        let root = arena.new_root();

        // Add children with different beliefs
        let actions = vec![0, 1, 2];
        let params = vec![
            (0.33, -0.5, 0.1), // Good child (high value = -0.5 negated to 0.5)
            (0.33, 0.0, 0.1),  // Neutral
            (0.33, 0.5, 0.1),  // Bad child
        ];
        arena.add_children(root, &actions, &params);

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut action_counts = [0u32; 3];

        // Run many selections
        for _ in 0..1000 {
            let (action, _) = select_child_thompson_ids(&arena, root, 1.0, &mut rng);
            action_counts[action as usize] += 1;
        }

        // First child should be selected most often (best value)
        assert!(action_counts[0] > action_counts[2]);
    }

    #[test]
    fn test_bayesian_backup() {
        let mut arena = BayesianTreeArena::new(100);
        let root = arena.new_root();

        let actions = vec![0, 1];
        let params = vec![(0.5, 0.0, 1.0), (0.5, 0.0, 1.0)];
        arena.add_children(root, &actions, &params);

        let child_idx = arena.get_children(root)[0].node_idx;
        let mut path = BayesianSearchPath::from_root(root);
        path.push(0, child_idx);

        // Backup a positive value
        bayesian_backup(&mut arena, &path, 1.0, 0.5, 1e-6, 0.01);

        // Child should have updated belief
        let child = arena.get(child_idx);
        assert!(child.mu > 0.0);

        // Root should have aggregated belief
        let root_node = arena.get(root);
        assert!(root_node.agg_mu.is_some());
    }

    #[test]
    fn test_get_bayesian_policy() {
        let mut arena = BayesianTreeArena::new(100);
        let root = arena.new_root();

        // Add children with clearly different values
        let actions = vec![0, 1, 2];
        let params = vec![
            (0.33, -0.9, 0.01), // Best (high value from parent's view)
            (0.33, 0.0, 0.01),
            (0.33, 0.9, 0.01), // Worst
        ];
        arena.add_children(root, &actions, &params);

        let policy = get_bayesian_policy(&arena, root, 5);

        // Best action should have highest probability
        assert!(policy[0] > policy[1]);
        assert!(policy[0] > policy[2]);

        // Should sum to ~1
        let sum: f32 = policy.iter().sum();
        assert!((sum - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_early_stopping() {
        let mut arena = BayesianTreeArena::new(100);
        let root = arena.new_root();

        // Clear winner
        let actions = vec![0, 1];
        let params = vec![
            (0.5, -1.0, 0.01), // Very confident good
            (0.5, 1.0, 0.01),  // Very confident bad
        ];
        arena.add_children(root, &actions, &params);

        assert!(should_stop_early(&arena, root, 0.95));

        // Unclear - high variance
        arena.clear();
        let root = arena.new_root();
        let params = vec![
            (0.5, -0.1, 1.0), // Uncertain
            (0.5, 0.1, 1.0),  // Uncertain
        ];
        arena.add_children(root, &actions, &params);

        assert!(!should_stop_early(&arena, root, 0.95));
    }
}
