//! Bayesian MCTS search operations.
//!
//! Implements Top-Two Thompson Sampling with IDS allocation and
//! variance-propagating backup.

use crate::bayesian_node::{aggregate_children, pairwise_optimality_weights, BayesianNode};
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
    ///
    /// If `visited_only` is true, only children with visits > 0
    /// are included in the aggregation.
    pub fn update_aggregated(&mut self, node_idx: u32, prune_threshold: f32, visited_only: bool) {
        let children = self.get_children(node_idx);
        if children.is_empty() {
            return;
        }

        // Collect child beliefs from parent's perspective (negate child values)
        let child_beliefs: Vec<(f32, f32)> = children
            .iter()
            .filter_map(|c| {
                let child = self.get(c.node_idx);
                if visited_only && child.visits == 0 {
                    None
                } else {
                    Some((-child.mu, child.sigma_sq))
                }
            })
            .collect();

        if child_beliefs.is_empty() {
            return;
        }

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

/// Root child belief from the parent's perspective.
#[derive(Clone, Debug)]
pub struct RootChildBelief {
    pub action: u16,
    pub prior: f32,
    pub mu: f32,
    pub sigma_sq: f32,
    pub weight: f32,
}

/// Root-level stopping decision and diagnostics.
#[derive(Clone, Debug)]
pub struct BayesianRootDecision {
    pub should_stop: bool,
    pub stop_reason: &'static str,
    pub consensus_score: f32,
    pub tie_gap: f32,
    pub leader_action: Option<u16>,
    pub challenger_action: Option<u16>,
    pub recommended_action: Option<u16>,
}

/// IDS allocation signal for top-two sampling.
#[derive(Clone, Copy, Debug)]
pub enum IdsAllocation {
    Precision,
    Visits,
}

/// Final root policy exposed to training/self-play.
#[derive(Clone, Copy, Debug)]
pub enum BayesianFinalPolicy {
    Optimality,
    Consensus,
}

fn better_root_recommendation(candidate: &RootChildBelief, best: &RootChildBelief) -> bool {
    const EPS: f32 = 1e-7;

    if candidate.weight > best.weight + EPS {
        return true;
    }
    if best.weight > candidate.weight + EPS {
        return false;
    }

    if candidate.mu > best.mu + EPS {
        return true;
    }
    if best.mu > candidate.mu + EPS {
        return false;
    }

    if candidate.prior > best.prior + EPS {
        return true;
    }
    if best.prior > candidate.prior + EPS {
        return false;
    }

    candidate.action < best.action
}

/// Compute root optimality weights from child Gaussian beliefs.
///
/// This is the single source of truth for the Bayesian root policy and
/// root-level stopping diagnostics. Child values are converted to the
/// parent's perspective by negating the stored child belief mean.
pub fn root_optimality_weights(arena: &BayesianTreeArena, root_idx: u32) -> Vec<RootChildBelief> {
    let children = arena.get_children(root_idx);
    let n = children.len();
    if n == 0 {
        return Vec::new();
    }

    let mut beliefs: Vec<RootChildBelief> = children
        .iter()
        .map(|c| {
            let child = arena.get(c.node_idx);
            RootChildBelief {
                action: c.action,
                prior: child.prior,
                mu: -child.mu,
                sigma_sq: child.sigma_sq,
                weight: 0.0,
            }
        })
        .collect();

    let child_beliefs: Vec<(f32, f32)> = beliefs
        .iter()
        .map(|belief| (belief.mu, belief.sigma_sq))
        .collect();
    let weights = pairwise_optimality_weights(&child_beliefs, 0.0);
    for (belief, weight) in beliefs.iter_mut().zip(weights.iter()) {
        belief.weight = *weight;
    }

    beliefs
}

/// Root policy derived from Bayesian optimality weights and optional prior pooling.
pub fn get_bayesian_policy_with_mode(
    arena: &BayesianTreeArena,
    root_idx: u32,
    action_size: usize,
    final_policy: BayesianFinalPolicy,
) -> Vec<f32> {
    let mut policy = vec![0.0f32; action_size];
    let beliefs = root_optimality_weights(arena, root_idx);
    if beliefs.is_empty() {
        return policy;
    }

    match final_policy {
        BayesianFinalPolicy::Optimality => {
            for belief in beliefs {
                policy[belief.action as usize] = belief.weight;
            }
        }
        BayesianFinalPolicy::Consensus => {
            let mut total = 0.0f32;
            for belief in &beliefs {
                let pooled = (belief.prior.max(0.0) * belief.weight.max(0.0)).sqrt();
                policy[belief.action as usize] = pooled;
                total += pooled;
            }
            if total > 1e-10 {
                for p in policy.iter_mut() {
                    *p /= total;
                }
            } else {
                for belief in beliefs {
                    policy[belief.action as usize] = belief.weight;
                }
            }
        }
    }

    policy
}

/// Analyze whether root search can stop early.
pub fn root_stop_decision(
    arena: &BayesianTreeArena,
    root_idx: u32,
    confidence_threshold: f32,
    epsilon_tie: f32,
    tie_sigma: f32,
) -> BayesianRootDecision {
    let beliefs = root_optimality_weights(arena, root_idx);

    if beliefs.is_empty() {
        return BayesianRootDecision {
            should_stop: true,
            stop_reason: "terminal",
            consensus_score: 0.0,
            tie_gap: 0.0,
            leader_action: None,
            challenger_action: None,
            recommended_action: None,
        };
    }

    let recommended_action = beliefs
        .iter()
        .reduce(|best, candidate| {
            if better_root_recommendation(candidate, best) {
                candidate
            } else {
                best
            }
        })
        .map(|b| b.action);

    if beliefs.len() == 1 {
        return BayesianRootDecision {
            should_stop: true,
            stop_reason: "forced",
            consensus_score: 1.0,
            tie_gap: 0.0,
            leader_action: Some(beliefs[0].action),
            challenger_action: None,
            recommended_action,
        };
    }

    let mut sorted_indices: Vec<usize> = (0..beliefs.len()).collect();
    sorted_indices.sort_by(|&a, &b| {
        beliefs[b]
            .mu
            .partial_cmp(&beliefs[a].mu)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| beliefs[a].action.cmp(&beliefs[b].action))
    });

    let leader = &beliefs[sorted_indices[0]];
    let challenger = &beliefs[sorted_indices[1]];
    let tie_gap = (leader.mu - challenger.mu).abs()
        + tie_sigma.max(0.0) * (leader.sigma_sq + challenger.sigma_sq).sqrt();

    let mut pooled_total = 0.0f32;
    let mut pooled_max = 0.0f32;
    for belief in &beliefs {
        let pooled = (belief.prior.max(0.0) * belief.weight.max(0.0)).sqrt();
        pooled_total += pooled;
        pooled_max = pooled_max.max(pooled);
    }

    let normalized_consensus = if pooled_total > 1e-10 {
        pooled_max / pooled_total
    } else {
        0.0
    };
    // The normalized geometric score alone can be spuriously high when the
    // prior/search overlap is tiny, so gate it by the Hellinger affinity.
    let consensus_score = normalized_consensus.min(pooled_total);

    let (should_stop, stop_reason) = if consensus_score >= confidence_threshold {
        (true, "consensus")
    } else if epsilon_tie > 0.0 && tie_gap <= epsilon_tie {
        (true, "epsilon_tie")
    } else {
        (false, "none")
    };

    BayesianRootDecision {
        should_stop,
        stop_reason,
        consensus_score,
        tie_gap,
        leader_action: Some(leader.action),
        challenger_action: Some(challenger.action),
        recommended_action,
    }
}

fn ids_allocation_signal(node: &BayesianNode, allocation: IdsAllocation) -> f32 {
    match allocation {
        IdsAllocation::Precision => node.precision(),
        IdsAllocation::Visits => node.visits as f32,
    }
}

fn challenger_probability(leader_signal: f32, challenger_signal: f32, ids_alpha: f32) -> f32 {
    let denom = leader_signal + challenger_signal + 2.0 * ids_alpha;
    if denom > 1e-10 {
        (leader_signal + ids_alpha) / denom
    } else {
        0.5
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
    ids_allocation: IdsAllocation,
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
    let leader_signal = ids_allocation_signal(arena.get(leader_idx), ids_allocation);
    let challenger_signal = ids_allocation_signal(arena.get(challenger_idx), ids_allocation);

    // beta = probability of selecting challenger
    // High leader precision → explore challenger more
    let beta = challenger_probability(leader_signal, challenger_signal, ids_alpha);

    // Select challenger with probability beta
    if rng.gen::<f32>() < beta {
        (challenger_action, challenger_idx)
    } else {
        (leader_action, leader_idx)
    }
}

/// Top-Two Thompson Sampling with IDS allocation and virtual loss.
///
/// Same as `select_child_thompson_ids` but uses virtual loss adjusted samples
/// to discourage re-selecting in-flight paths.
pub fn select_child_thompson_ids_with_virtual_loss<R: rand::Rng>(
    arena: &BayesianTreeArena,
    node_idx: u32,
    ids_alpha: f32,
    ids_allocation: IdsAllocation,
    virtual_loss_value: f32,
    rng: &mut R,
) -> (u16, u32) {
    let children = arena.get_children(node_idx);
    debug_assert!(!children.is_empty());

    if children.len() == 1 {
        return (children[0].action, children[0].node_idx);
    }

    // Draw Thompson samples with virtual loss adjustment
    // (from parent's perspective: negate child values)
    let mut samples: Vec<(u16, u32, f32)> = children
        .iter()
        .map(|c| {
            let child = arena.get(c.node_idx);
            // Use virtual loss adjusted sample - higher adjusted mu = worse for parent
            let sample = -child.sample_with_virtual_loss(rng, virtual_loss_value);
            (c.action, c.node_idx, sample)
        })
        .collect();

    // Sort by sample value (descending)
    samples.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    let (leader_action, leader_idx, _) = samples[0];
    let (challenger_action, challenger_idx, _) = samples[1];

    // IDS allocation probability
    let leader_signal = ids_allocation_signal(arena.get(leader_idx), ids_allocation);
    let challenger_signal = ids_allocation_signal(arena.get(challenger_idx), ids_allocation);

    // beta = probability of selecting challenger
    let beta = challenger_probability(leader_signal, challenger_signal, ids_alpha);

    // Select challenger with probability beta
    if rng.gen::<f32>() < beta {
        (challenger_action, challenger_idx)
    } else {
        (leader_action, leader_idx)
    }
}

/// Apply virtual loss to all nodes in a Bayesian search path.
pub fn apply_bayesian_virtual_loss(arena: &mut BayesianTreeArena, path: &BayesianSearchPath) {
    for &node_idx in path.nodes.iter() {
        arena.get_mut(node_idx).apply_virtual_loss();
    }
}

/// Remove virtual loss from all nodes in a Bayesian search path.
pub fn remove_bayesian_virtual_loss(arena: &mut BayesianTreeArena, path: &BayesianSearchPath) {
    for &node_idx in path.nodes.iter() {
        arena.get_mut(node_idx).remove_virtual_loss();
    }
}

/// Backup with posterior propagation (not Bayesian updates at each level).
///
/// Only the leaf node receives a true Bayesian update with obs_var.
/// Intermediate nodes just aggregate their children's posteriors and
/// copy the aggregated belief to their own belief.
///
/// This prevents precision from compounding at each level, since we're
/// propagating posteriors rather than treating aggregated values as
/// new observations.
pub fn bayesian_backup(
    arena: &mut BayesianTreeArena,
    path: &BayesianSearchPath,
    leaf_value: f32,
    obs_var: f32,
    min_variance: f32,
    prune_threshold: f32,
    _optimality_weight: f32,
    _adaptive: bool,
    _visit_scale: f32,
) {
    if path.nodes.len() < 2 {
        return; // No backup needed for root-only path
    }

    // Iterate from leaf to root (skip root for update, it has no parent)
    for (iteration, i) in (1..path.nodes.len()).rev().enumerate() {
        let child_idx = path.nodes[i];
        let parent_idx = path.nodes[i - 1];

        // Mark child as visited and increment visit count
        arena.get_mut(child_idx).visits += 1;

        if iteration == 0 {
            // First iteration: child is the leaf
            let child = arena.get(child_idx);
            if !child.expanded() {
                // Terminal or unexpanded: do Bayesian update with observation
                arena
                    .get_mut(child_idx)
                    .update(leaf_value, obs_var, min_variance);
            } else {
                // Just expanded: its agg_mu/agg_sigma_sq were set during expansion
                // Copy aggregated belief to own belief
                let child = arena.get(child_idx);
                if let (Some(agg_mu), Some(agg_sigma_sq)) = (child.agg_mu, child.agg_sigma_sq) {
                    let child_mut = arena.get_mut(child_idx);
                    child_mut.mu = agg_mu;
                    child_mut.sigma_sq = agg_sigma_sq;
                }
            }
        }
        // else: child's mu/sigma_sq were updated in previous iteration

        // Aggregate parent's children (visited only)
        arena.update_aggregated(parent_idx, prune_threshold, true);

        // Copy aggregated belief to own belief (no Bayesian update!)
        // This is what the grandparent will see when it aggregates
        let parent = arena.get(parent_idx);
        if let (Some(agg_mu), Some(agg_sigma_sq)) = (parent.agg_mu, parent.agg_sigma_sq) {
            let parent_mut = arena.get_mut(parent_idx);
            parent_mut.mu = agg_mu;
            parent_mut.sigma_sq = agg_sigma_sq;
        }
    }
}

/// Backup with posterior propagation and virtual loss removal.
///
/// Same as `bayesian_backup` but also removes virtual loss from the path.
pub fn bayesian_backup_with_virtual_loss_removal(
    arena: &mut BayesianTreeArena,
    path: &BayesianSearchPath,
    leaf_value: f32,
    obs_var: f32,
    min_variance: f32,
    prune_threshold: f32,
) {
    // First remove virtual loss from entire path
    remove_bayesian_virtual_loss(arena, path);

    // Then do normal backup
    if path.nodes.len() < 2 {
        return;
    }

    for (iteration, i) in (1..path.nodes.len()).rev().enumerate() {
        let child_idx = path.nodes[i];
        let parent_idx = path.nodes[i - 1];

        arena.get_mut(child_idx).visits += 1;

        if iteration == 0 {
            let child = arena.get(child_idx);
            if !child.expanded() {
                arena
                    .get_mut(child_idx)
                    .update(leaf_value, obs_var, min_variance);
            } else {
                let child = arena.get(child_idx);
                if let (Some(agg_mu), Some(agg_sigma_sq)) = (child.agg_mu, child.agg_sigma_sq) {
                    let child_mut = arena.get_mut(child_idx);
                    child_mut.mu = agg_mu;
                    child_mut.sigma_sq = agg_sigma_sq;
                }
            }
        }

        arena.update_aggregated(parent_idx, prune_threshold, true);

        let parent = arena.get(parent_idx);
        if let (Some(agg_mu), Some(agg_sigma_sq)) = (parent.agg_mu, parent.agg_sigma_sq) {
            let parent_mut = arena.get_mut(parent_idx);
            parent_mut.mu = agg_mu;
            parent_mut.sigma_sq = agg_sigma_sq;
        }
    }
}

/// Select to leaf for Bayesian MCTS.
pub fn bayesian_select_to_leaf<R: rand::Rng, F>(
    arena: &BayesianTreeArena,
    root_idx: u32,
    ids_alpha: f32,
    ids_allocation: IdsAllocation,
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
        let (action, child_idx) =
            select_child_thompson_ids(arena, node_idx, ids_alpha, ids_allocation, rng);
        path.push(action, child_idx);
        node_idx = child_idx;
    }
}

/// Get policy from optimality weights.
///
/// Computes P(each child is optimal) using pairwise Gaussian CDF comparisons.
pub fn get_bayesian_policy(
    arena: &BayesianTreeArena,
    root_idx: u32,
    action_size: usize,
) -> Vec<f32> {
    get_bayesian_policy_with_mode(
        arena,
        root_idx,
        action_size,
        BayesianFinalPolicy::Optimality,
    )
}

/// Check if early stopping condition is met.
///
/// Uses geometric consensus between the policy prior and Bayesian root
/// optimality weights. Kept as a small compatibility wrapper for tests and
/// callers that only need a boolean decision.
pub fn should_stop_early(
    arena: &BayesianTreeArena,
    root_idx: u32,
    confidence_threshold: f32,
) -> bool {
    root_stop_decision(arena, root_idx, confidence_threshold, 0.0, 1.0).should_stop
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
            let (action, _) =
                select_child_thompson_ids(&arena, root, 1.0, IdsAllocation::Precision, &mut rng);
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
        bayesian_backup(&mut arena, &path, 1.0, 0.5, 1e-6, 0.01, 0.3, true, 50.0);

        // Child should have updated belief and visit count
        let child = arena.get(child_idx);
        assert!(child.mu > 0.0);
        assert_eq!(child.visits, 1);

        // Root should have aggregated belief (now only includes visited child)
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
    fn test_root_optimality_weights_are_normalized() {
        let mut arena = BayesianTreeArena::new(100);
        let root = arena.new_root();

        let actions = vec![1, 3, 4];
        let params = vec![(0.3, -0.4, 0.2), (0.4, 0.0, 0.2), (0.3, 0.2, 0.2)];
        arena.add_children(root, &actions, &params);

        let weights = root_optimality_weights(&arena, root);
        let sum: f32 = weights.iter().map(|w| w.weight).sum();

        assert_eq!(weights.len(), 3);
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(weights.iter().all(|w| actions.contains(&w.action)));
    }

    #[test]
    fn test_root_decision_tie_breaks_recommended_action_deterministically() {
        let mut arena = BayesianTreeArena::new(100);
        let root = arena.new_root();

        let actions = vec![0, 1, 2];
        let params = vec![
            (1.0 / 3.0, 0.0, 1.0),
            (1.0 / 3.0, 0.0, 1.0),
            (1.0 / 3.0, 0.0, 1.0),
        ];
        arena.add_children(root, &actions, &params);

        let decision = root_stop_decision(&arena, root, 0.99, 0.0, 1.0);

        assert_eq!(decision.recommended_action, Some(0));
        assert_eq!(decision.leader_action, Some(0));
        assert_eq!(decision.challenger_action, Some(1));
    }

    #[test]
    fn test_early_stopping() {
        let mut arena = BayesianTreeArena::new(100);
        let root = arena.new_root();

        // Clear winner
        let actions = vec![0, 1];
        let params = vec![
            (0.99, -1.0, 0.01), // Prior and search agree on a clear winner
            (0.01, 1.0, 0.01),
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

    #[test]
    fn test_consensus_stop_requires_prior_search_agreement() {
        let actions = vec![0, 1];

        let mut arena = BayesianTreeArena::new(100);
        let root = arena.new_root();
        let agreeing_params = vec![
            (0.99, -1.0, 0.01), // Prior and search both favor action 0
            (0.01, 1.0, 0.01),
        ];
        arena.add_children(root, &actions, &agreeing_params);

        let decision = root_stop_decision(&arena, root, 0.95, 0.0, 1.0);
        assert!(decision.should_stop);
        assert_eq!(decision.stop_reason, "consensus");
        assert!(decision.consensus_score > 0.95);

        arena.clear();
        let root = arena.new_root();
        let disagreeing_params = vec![
            (0.01, -1.0, 0.01), // Search favors action 0, prior favors action 1
            (0.99, 1.0, 0.01),
        ];
        arena.add_children(root, &actions, &disagreeing_params);

        let decision = root_stop_decision(&arena, root, 0.95, 0.0, 1.0);
        assert!(!decision.should_stop);
        assert!(decision.consensus_score < 0.95);
    }

    #[test]
    fn test_epsilon_tie_stop() {
        let actions = vec![0, 1];

        let mut arena = BayesianTreeArena::new(100);
        let root = arena.new_root();
        let close_low_variance = vec![(0.5, -0.01, 0.0001), (0.5, 0.01, 0.0001)];
        arena.add_children(root, &actions, &close_low_variance);

        let decision = root_stop_decision(&arena, root, 0.99, 0.05, 1.0);
        assert!(decision.should_stop);
        assert_eq!(decision.stop_reason, "epsilon_tie");

        arena.clear();
        let root = arena.new_root();
        let close_high_variance = vec![(0.5, -0.01, 1.0), (0.5, 0.01, 1.0)];
        arena.add_children(root, &actions, &close_high_variance);

        let decision = root_stop_decision(&arena, root, 0.99, 0.05, 1.0);
        assert!(!decision.should_stop);

        arena.clear();
        let root = arena.new_root();
        let clear_gap = vec![(0.5, -0.5, 0.0001), (0.5, 0.5, 0.0001)];
        arena.add_children(root, &actions, &clear_gap);

        let decision = root_stop_decision(&arena, root, 0.99, 0.05, 1.0);
        assert!(!decision.should_stop);
    }
}
