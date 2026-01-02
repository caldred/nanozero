//! UCB (Upper Confidence Bound) score calculation for MCTS.
//!
//! Implements the PUCT formula used in AlphaZero:
//! UCB(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

/// Calculate the UCB score for a child node.
///
/// Uses the PUCT formula from AlphaZero:
/// UCB = -Q(child) + c_puct * prior * sqrt(parent_visits) / (1 + child_visits)
///
/// Note: We negate Q(child) because the value is from the child's perspective
/// (opponent), but we want it from the parent's perspective.
///
/// # Arguments
/// * `parent_visit_count` - Total visits to parent node N(s)
/// * `child_prior` - Prior probability P(s,a) from policy network
/// * `child_visit_count` - Number of visits to child node N(s,a)
/// * `child_value` - Mean value of child Q(s,a) from child's perspective
/// * `c_puct` - Exploration constant (typically 1.0-2.0)
#[inline]
pub fn ucb_score(
    parent_visit_count: u32,
    child_prior: f32,
    child_visit_count: u32,
    child_value: f32,
    c_puct: f32,
) -> f32 {
    let exploration = c_puct * child_prior * (parent_visit_count as f32).sqrt()
        / (1.0 + child_visit_count as f32);
    let exploitation = -child_value; // Negate: child value is from opponent's view
    exploitation + exploration
}

/// Calculate UCB score with virtual loss.
///
/// Same as `ucb_score` but uses effective visit count that includes virtual loss.
#[inline]
pub fn ucb_score_with_virtual_loss(
    parent_visit_count: u32,
    parent_virtual_loss: u8,
    child_prior: f32,
    child_visit_count: u32,
    child_virtual_loss: u8,
    child_value: f32,
    c_puct: f32,
    virtual_loss_value: f32,
) -> f32 {
    let effective_parent_visits = parent_visit_count + parent_virtual_loss as u32;
    let effective_child_visits = child_visit_count + child_virtual_loss as u32;

    // Adjust child value for virtual loss
    // Virtual loss is treated as a loss (-virtual_loss_value) for each virtual visit
    let adjusted_value = if child_visit_count > 0 {
        let total_value =
            child_value * child_visit_count as f32 - virtual_loss_value * child_virtual_loss as f32;
        total_value / effective_child_visits as f32
    } else if child_virtual_loss > 0 {
        -virtual_loss_value
    } else {
        0.0
    };

    let exploration = c_puct * child_prior * (effective_parent_visits as f32).sqrt()
        / (1.0 + effective_child_visits as f32);
    let exploitation = -adjusted_value;
    exploitation + exploration
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ucb_unvisited_child() {
        // Unvisited child should have high UCB due to exploration term
        let score = ucb_score(100, 0.5, 0, 0.0, 1.0);
        // sqrt(100) * 0.5 / 1 = 5.0
        assert!((score - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_ucb_visited_child() {
        // Visited child with positive value (from child's perspective)
        // means negative exploitation (opponent is winning)
        let score = ucb_score(100, 0.5, 10, 0.5, 1.0);
        // exploitation = -0.5
        // exploration = 1.0 * 0.5 * 10 / 11 ≈ 0.4545
        // total ≈ -0.045
        assert!(score < 0.0);
    }

    #[test]
    fn test_ucb_negative_child_value() {
        // Negative child value means we are winning
        let score = ucb_score(100, 0.5, 10, -0.5, 1.0);
        // exploitation = 0.5
        // exploration ≈ 0.4545
        // total ≈ 0.955
        assert!(score > 0.5);
    }

    #[test]
    fn test_ucb_exploration_decreases_with_visits() {
        let score_low_visits = ucb_score(100, 0.5, 1, 0.0, 1.0);
        let score_high_visits = ucb_score(100, 0.5, 50, 0.0, 1.0);
        assert!(score_low_visits > score_high_visits);
    }

    #[test]
    fn test_virtual_loss_reduces_score() {
        let score_no_vl = ucb_score(100, 0.5, 10, 0.0, 1.0);
        let score_with_vl =
            ucb_score_with_virtual_loss(100, 0, 0.5, 10, 3, 0.0, 1.0, 1.0);
        // Virtual loss should reduce the score
        assert!(score_with_vl < score_no_vl);
    }
}
