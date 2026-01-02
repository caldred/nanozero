//! Mathematical utilities for MCTS.
//!
//! Contains functions like normal CDF used in Bayesian MCTS.

use std::f32::consts::FRAC_1_SQRT_2;

/// Compute the standard normal CDF (cumulative distribution function).
///
/// Uses the error function: Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
#[inline]
pub fn normal_cdf(x: f32) -> f32 {
    0.5 * (1.0 + libm::erff(x * FRAC_1_SQRT_2))
}

/// Compute softmax of a slice in-place.
///
/// Modifies the input slice to contain softmax probabilities.
pub fn softmax_inplace(values: &mut [f32]) {
    if values.is_empty() {
        return;
    }

    // Find max for numerical stability
    let max_val = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp and sum
    let mut sum = 0.0f32;
    for v in values.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }

    // Normalize
    if sum > 0.0 {
        for v in values.iter_mut() {
            *v /= sum;
        }
    }
}

/// Sample from a categorical distribution.
///
/// # Arguments
/// * `probs` - Probability distribution (must sum to ~1.0)
/// * `rand_val` - Random value in [0, 1)
///
/// Returns the index of the sampled element.
pub fn sample_categorical(probs: &[f32], rand_val: f32) -> usize {
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if rand_val < cumsum {
            return i;
        }
    }
    // Fallback to last element (handles floating point errors)
    probs.len().saturating_sub(1)
}

/// Add Dirichlet noise to a probability distribution.
///
/// # Arguments
/// * `probs` - Original probabilities (modified in-place)
/// * `alpha` - Dirichlet concentration parameter
/// * `epsilon` - Mixing weight (0 = no noise, 1 = all noise)
/// * `rng` - Random number generator
pub fn add_dirichlet_noise<R: rand::Rng>(probs: &mut [f32], alpha: f32, epsilon: f32, rng: &mut R) {
    use rand_distr::{Dirichlet, Distribution};

    if probs.is_empty() || epsilon == 0.0 {
        return;
    }

    // Generate Dirichlet noise
    let dirichlet = Dirichlet::new_with_size(alpha, probs.len()).unwrap();
    let noise: Vec<f32> = dirichlet.sample(rng);

    // Mix original with noise
    for (p, n) in probs.iter_mut().zip(noise.iter()) {
        *p = (1.0 - epsilon) * *p + epsilon * n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_cdf() {
        // Φ(0) = 0.5
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);

        // Φ(-∞) → 0, Φ(+∞) → 1
        assert!(normal_cdf(-10.0) < 0.001);
        assert!(normal_cdf(10.0) > 0.999);

        // Φ(1.96) ≈ 0.975
        assert!((normal_cdf(1.96) - 0.975).abs() < 0.01);

        // Φ(-x) = 1 - Φ(x)
        let x = 1.5;
        assert!((normal_cdf(-x) + normal_cdf(x) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax() {
        let mut vals = vec![1.0, 2.0, 3.0];
        softmax_inplace(&mut vals);

        // Should sum to 1
        let sum: f32 = vals.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Should be monotonic
        assert!(vals[0] < vals[1]);
        assert!(vals[1] < vals[2]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let mut vals = vec![1000.0, 1001.0, 1002.0];
        softmax_inplace(&mut vals);

        // Should not overflow
        assert!(vals.iter().all(|&v| v.is_finite()));
        let sum: f32 = vals.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sample_categorical() {
        let probs = vec![0.2, 0.3, 0.5];

        // Sample at different points
        assert_eq!(sample_categorical(&probs, 0.1), 0);
        assert_eq!(sample_categorical(&probs, 0.3), 1);
        assert_eq!(sample_categorical(&probs, 0.6), 2);
        assert_eq!(sample_categorical(&probs, 0.99), 2);
    }

    #[test]
    fn test_dirichlet_noise() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let mut probs = vec![0.25, 0.25, 0.25, 0.25];
        add_dirichlet_noise(&mut probs, 0.3, 0.25, &mut rng);

        // Should still sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Should be different from uniform (with high probability)
        assert!(probs.iter().any(|&p| (p - 0.25).abs() > 0.01));
    }
}
