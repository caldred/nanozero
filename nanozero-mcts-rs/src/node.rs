//! Node struct for standard MCTS.
//!
//! Each node represents a game state in the search tree and tracks
//! visit counts and value sums for UCB-based selection.

/// MCTS tree node with visit count and value statistics.
///
/// Uses arena allocation - children are stored as indices into a separate
/// children array rather than as owned pointers.
#[derive(Clone, Debug)]
pub struct Node {
    /// Prior probability from policy network P(a|s)
    pub prior: f32,
    /// Number of times this node has been visited
    pub visit_count: u32,
    /// Sum of values backpropagated through this node
    pub value_sum: f32,
    /// Index into the arena's children array where this node's children start
    pub children_start: u32,
    /// Number of children (max 362 for Go 19x19 + pass)
    pub children_count: u16,
    /// Virtual loss counter for parallel search
    pub virtual_loss: u8,
}

impl Node {
    /// Create a new node with the given prior probability.
    #[inline]
    pub fn new(prior: f32) -> Self {
        Self {
            prior,
            visit_count: 0,
            value_sum: 0.0,
            children_start: 0,
            children_count: 0,
            virtual_loss: 0,
        }
    }

    /// Get the mean value of this node.
    ///
    /// Returns 0.0 if the node hasn't been visited yet.
    #[inline]
    pub fn value(&self) -> f32 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.value_sum / self.visit_count as f32
        }
    }

    /// Check if this node has been expanded (has children).
    #[inline]
    pub fn expanded(&self) -> bool {
        self.children_count > 0
    }

    /// Get the effective visit count including virtual loss.
    #[inline]
    pub fn effective_visit_count(&self) -> u32 {
        self.visit_count + self.virtual_loss as u32
    }

    /// Apply virtual loss to discourage parallel threads from visiting same path.
    #[inline]
    pub fn apply_virtual_loss(&mut self) {
        self.virtual_loss = self.virtual_loss.saturating_add(1);
    }

    /// Remove virtual loss after backpropagation.
    #[inline]
    pub fn remove_virtual_loss(&mut self) {
        self.virtual_loss = self.virtual_loss.saturating_sub(1);
    }
}

impl Default for Node {
    fn default() -> Self {
        Self::new(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation() {
        let node = Node::new(0.5);
        assert_eq!(node.prior, 0.5);
        assert_eq!(node.visit_count, 0);
        assert_eq!(node.value_sum, 0.0);
        assert!(!node.expanded());
    }

    #[test]
    fn test_node_value() {
        let mut node = Node::new(0.5);
        assert_eq!(node.value(), 0.0);

        node.visit_count = 10;
        node.value_sum = 5.0;
        assert!((node.value() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_virtual_loss() {
        let mut node = Node::new(0.5);
        assert_eq!(node.virtual_loss, 0);
        assert_eq!(node.effective_visit_count(), 0);

        node.apply_virtual_loss();
        assert_eq!(node.virtual_loss, 1);
        assert_eq!(node.effective_visit_count(), 1);

        node.visit_count = 10;
        assert_eq!(node.effective_visit_count(), 11);

        node.remove_virtual_loss();
        assert_eq!(node.virtual_loss, 0);
        assert_eq!(node.effective_visit_count(), 10);
    }
}
