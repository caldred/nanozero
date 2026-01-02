//! Bayesian MCTS node with Gaussian beliefs.
//!
//! Instead of tracking visit counts and value sums, maintains Gaussian
//! posteriors (mu, sigma_sq) over node values.

use crate::math::normal_cdf;

/// Bayesian MCTS node with Gaussian belief over value.
#[derive(Clone, Debug)]
pub struct BayesianNode {
    /// Prior probability from policy network P(a|s)
    pub prior: f32,
    /// Mean of value belief
    pub mu: f32,
    /// Variance of value belief
    pub sigma_sq: f32,
    /// Aggregated mean from children (computed after expansion/backup)
    pub agg_mu: Option<f32>,
    /// Aggregated variance from children
    pub agg_sigma_sq: Option<f32>,
    /// Index into children array
    pub children_start: u32,
    /// Number of children
    pub children_count: u16,
}

impl BayesianNode {
    /// Create a new Bayesian node.
    pub fn new(prior: f32, mu: f32, sigma_sq: f32) -> Self {
        Self {
            prior,
            mu,
            sigma_sq,
            agg_mu: None,
            agg_sigma_sq: None,
            children_start: 0,
            children_count: 0,
        }
    }

    /// Create a node with default prior initialization.
    pub fn with_prior(prior: f32) -> Self {
        Self::new(prior, 0.0, 1.0)
    }

    /// Check if this node has been expanded.
    #[inline]
    pub fn expanded(&self) -> bool {
        self.children_count > 0
    }

    /// Get precision (inverse variance) - proxy for visit count.
    #[inline]
    pub fn precision(&self) -> f32 {
        1.0 / self.sigma_sq.max(1e-8)
    }

    /// Bayesian update with an observed value.
    ///
    /// Uses precision-weighted combination:
    /// precision_new = precision_prior + precision_obs
    /// mu_new = (precision_prior * mu + precision_obs * value) / precision_new
    pub fn update(&mut self, value: f32, obs_var: f32, min_var: f32) {
        let precision_prior = 1.0 / self.sigma_sq.max(min_var);
        let precision_obs = 1.0 / obs_var.max(min_var);
        let new_precision = precision_prior + precision_obs;

        self.mu = (precision_prior * self.mu + precision_obs * value) / new_precision;
        self.sigma_sq = (1.0 / new_precision).max(min_var);
    }

    /// Draw a Thompson sample from the posterior.
    pub fn sample<R: rand::Rng>(&self, rng: &mut R) -> f32 {
        use rand_distr::{Distribution, Normal};
        let normal = Normal::new(self.mu, self.sigma_sq.sqrt()).unwrap();
        normal.sample(rng)
    }
}

impl Default for BayesianNode {
    fn default() -> Self {
        Self::with_prior(0.0)
    }
}

/// Aggregate children beliefs into parent's aggregated belief.
///
/// Uses optimality weights (probability each child is best) and variance
/// aggregation (ensemble effect + disagreement).
///
/// # Arguments
/// * `children` - Slice of (mu, sigma_sq) for each child (from parent's perspective, i.e., negated)
/// * `prune_threshold` - Children with P(optimal) < threshold get weight 0
///
/// # Returns
/// * `(agg_mu, agg_sigma_sq)` - Aggregated mean and variance
pub fn aggregate_children(children: &[(f32, f32)], prune_threshold: f32) -> (f32, f32) {
    let n = children.len();
    if n == 0 {
        return (0.0, 1.0);
    }
    if n == 1 {
        return (children[0].0, children[0].1);
    }

    // Find leader and challenger by mean
    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| {
        children[b]
            .0
            .partial_cmp(&children[a].0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let leader_idx = sorted_indices[0];
    let challenger_idx = sorted_indices[1];

    let (mu_l, sigma_sq_l) = children[leader_idx];
    let (mu_c, sigma_sq_c) = children[challenger_idx];

    // Compute optimality scores via pairwise Gaussian CDF comparisons
    let mut scores = vec![0.0f32; n];

    for i in 0..n {
        let (mu_i, sigma_sq_i) = children[i];

        let (diff, combined_var) = if i == leader_idx {
            // P(leader > challenger)
            (mu_l - mu_c, sigma_sq_l + sigma_sq_c)
        } else {
            // P(child > leader)
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

    // Soft prune and normalize to get weights
    for score in scores.iter_mut() {
        if *score < prune_threshold {
            *score = 0.0;
        }
    }

    let total: f32 = scores.iter().sum();
    let weights: Vec<f32> = if total < 1e-10 {
        // Fallback: uniform weights
        vec![1.0 / n as f32; n]
    } else {
        scores.iter().map(|s| s / total).collect()
    };

    // Aggregated mean (weighted average of children)
    let agg_mu: f32 = weights
        .iter()
        .zip(children.iter())
        .map(|(&w, &(mu, _))| w * mu)
        .sum();

    // Aggregated variance: Σ w²_a [σ²_a + (μ_a - V_parent)²]
    let agg_sigma_sq: f32 = weights
        .iter()
        .zip(children.iter())
        .map(|(&w, &(mu, sigma_sq))| {
            let disagreement = (mu - agg_mu).powi(2);
            w * w * (sigma_sq + disagreement)
        })
        .sum();

    (agg_mu, agg_sigma_sq)
}

/// Create children with logit-shifted prior initialization.
///
/// For each legal action a:
/// H = entropy of policy = -sum(p * log(p))
/// mu_a = -V(s) - sigma_0 * (sqrt(6)/pi) * [ln(p_a) + H]
/// sigma_sq_a = sigma_0^2
///
/// Note: We negate the value because children store values from child's
/// (opponent's) perspective.
pub fn create_bayesian_children(
    value: f32,
    priors: &[f32],
    sigma_0: f32,
) -> Vec<(f32, f32, f32)> {
    // (prior, mu, sigma_sq) for each child
    if priors.is_empty() {
        return Vec::new();
    }

    let eps = 1e-8;
    let scale = sigma_0 * (6.0f32.sqrt() / std::f32::consts::PI);

    // Compute entropy
    let log_priors: Vec<f32> = priors.iter().map(|&p| (p + eps).ln()).collect();
    let entropy: f32 = -priors
        .iter()
        .zip(log_priors.iter())
        .map(|(&p, &lp)| p * lp)
        .sum::<f32>();

    // Create children with logit-shifted means
    priors
        .iter()
        .zip(log_priors.iter())
        .map(|(&prior, &log_prior)| {
            // Negate value: children store values from opponent's perspective
            let mu = -value - scale * (log_prior + entropy);
            let sigma_sq = sigma_0 * sigma_0;
            (prior, mu, sigma_sq)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bayesian_node_update() {
        let mut node = BayesianNode::new(0.5, 0.0, 1.0);

        // Update with a positive observation
        node.update(1.0, 0.5, 1e-6);

        // Mean should shift towards observed value
        assert!(node.mu > 0.0);
        // Variance should decrease
        assert!(node.sigma_sq < 1.0);
    }

    #[test]
    fn test_aggregate_single_child() {
        let children = vec![(0.5, 0.1)];
        let (mu, sigma_sq) = aggregate_children(&children, 0.01);
        assert!((mu - 0.5).abs() < 1e-6);
        assert!((sigma_sq - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_aggregate_two_children() {
        // Leader clearly better
        let children = vec![(0.8, 0.1), (0.2, 0.1)];
        let (mu, sigma_sq) = aggregate_children(&children, 0.01);

        // Should be close to leader's value
        assert!(mu > 0.5);
        assert!(sigma_sq > 0.0);
    }

    #[test]
    fn test_aggregate_with_pruning() {
        // One child clearly dominates
        let children = vec![(0.9, 0.01), (0.1, 0.01)];
        let (mu, _) = aggregate_children(&children, 0.1);

        // With high pruning threshold, should be very close to leader
        assert!(mu > 0.8);
    }

    #[test]
    fn test_create_bayesian_children() {
        let priors = vec![0.5, 0.3, 0.2];
        let value = 0.5;
        let sigma_0 = 1.0;

        let children = create_bayesian_children(value, &priors, sigma_0);

        assert_eq!(children.len(), 3);

        // All should have same variance
        for (_, _, sigma_sq) in &children {
            assert!((sigma_sq - 1.0).abs() < 1e-6);
        }

        // Higher prior should have higher initial mean
        // (because log(higher_prior) is less negative)
        assert!(children[0].1 > children[2].1);
    }

    #[test]
    fn test_precision() {
        let node = BayesianNode::new(0.5, 0.0, 0.25);
        assert!((node.precision() - 4.0).abs() < 1e-6);
    }
}
