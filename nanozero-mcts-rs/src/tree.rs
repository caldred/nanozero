//! Tree arena for efficient MCTS node allocation.
//!
//! Uses contiguous memory allocation for better cache locality during
//! tree traversal. Nodes reference their children via indices rather
//! than pointers.

use crate::node::Node;

/// A child entry storing the action and the index of the child node.
#[derive(Clone, Copy, Debug)]
pub struct ChildEntry {
    /// Action that leads to this child
    pub action: u16,
    /// Index of the child node in the arena
    pub node_idx: u32,
}

/// Arena-based tree storage for MCTS.
///
/// Provides O(1) node allocation and efficient memory layout for
/// tree traversal. Each search tree has its own arena.
#[derive(Debug)]
pub struct TreeArena {
    /// All nodes in this tree
    nodes: Vec<Node>,
    /// Children entries: (action, node_idx) pairs
    children: Vec<ChildEntry>,
}

impl TreeArena {
    /// Create a new arena with estimated capacity.
    pub fn new(estimated_nodes: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(estimated_nodes),
            children: Vec::with_capacity(estimated_nodes * 4), // Assume avg 4 children
        }
    }

    /// Allocate a new root node with prior 1.0.
    pub fn new_root(&mut self) -> u32 {
        self.allocate_node(1.0)
    }

    /// Allocate a new node with the given prior.
    ///
    /// Returns the index of the new node.
    #[inline]
    pub fn allocate_node(&mut self, prior: f32) -> u32 {
        let idx = self.nodes.len() as u32;
        self.nodes.push(Node::new(prior));
        idx
    }

    /// Get a reference to a node by index.
    #[inline]
    pub fn get(&self, idx: u32) -> &Node {
        &self.nodes[idx as usize]
    }

    /// Get a mutable reference to a node by index.
    #[inline]
    pub fn get_mut(&mut self, idx: u32) -> &mut Node {
        &mut self.nodes[idx as usize]
    }

    /// Add children to a node from policy and legal actions.
    ///
    /// # Arguments
    /// * `parent_idx` - Index of the parent node
    /// * `actions` - Legal actions at this state
    /// * `priors` - Prior probabilities for each action (same length as actions)
    pub fn add_children(&mut self, parent_idx: u32, actions: &[u16], priors: &[f32]) {
        debug_assert_eq!(actions.len(), priors.len());

        let children_start = self.children.len() as u32;
        let children_count = actions.len() as u16;

        // Allocate child nodes and add entries
        for (&action, &prior) in actions.iter().zip(priors.iter()) {
            let child_idx = self.allocate_node(prior);
            self.children.push(ChildEntry {
                action,
                node_idx: child_idx,
            });
        }

        // Update parent
        let parent = self.get_mut(parent_idx);
        parent.children_start = children_start;
        parent.children_count = children_count;
    }

    /// Get the children of a node.
    #[inline]
    pub fn get_children(&self, node_idx: u32) -> &[ChildEntry] {
        let node = self.get(node_idx);
        if node.children_count == 0 {
            return &[];
        }
        let start = node.children_start as usize;
        let end = start + node.children_count as usize;
        &self.children[start..end]
    }

    /// Get a specific child by action.
    ///
    /// Returns None if the action is not a valid child.
    pub fn get_child(&self, node_idx: u32, action: u16) -> Option<u32> {
        self.get_children(node_idx)
            .iter()
            .find(|c| c.action == action)
            .map(|c| c.node_idx)
    }

    /// Clear the arena for reuse.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.children.clear();
    }

    /// Get the number of nodes in the arena.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the arena is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

impl Default for TreeArena {
    fn default() -> Self {
        Self::new(1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_allocation() {
        let mut arena = TreeArena::new(100);
        let root = arena.new_root();
        assert_eq!(root, 0);
        assert_eq!(arena.len(), 1);

        let node = arena.get(root);
        assert_eq!(node.prior, 1.0);
        assert!(!node.expanded());
    }

    #[test]
    fn test_add_children() {
        let mut arena = TreeArena::new(100);
        let root = arena.new_root();

        let actions = vec![0, 1, 2];
        let priors = vec![0.5, 0.3, 0.2];
        arena.add_children(root, &actions, &priors);

        assert!(arena.get(root).expanded());
        assert_eq!(arena.get(root).children_count, 3);

        let children = arena.get_children(root);
        assert_eq!(children.len(), 3);
        assert_eq!(children[0].action, 0);
        assert_eq!(children[1].action, 1);
        assert_eq!(children[2].action, 2);

        // Check child nodes
        let child0 = arena.get(children[0].node_idx);
        assert!((child0.prior - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_get_child() {
        let mut arena = TreeArena::new(100);
        let root = arena.new_root();

        let actions = vec![5, 10, 15];
        let priors = vec![0.33, 0.33, 0.34];
        arena.add_children(root, &actions, &priors);

        assert!(arena.get_child(root, 5).is_some());
        assert!(arena.get_child(root, 10).is_some());
        assert!(arena.get_child(root, 15).is_some());
        assert!(arena.get_child(root, 20).is_none());
    }

    #[test]
    fn test_clear() {
        let mut arena = TreeArena::new(100);
        let root = arena.new_root();
        arena.add_children(root, &[0, 1], &[0.5, 0.5]);

        assert!(!arena.is_empty());
        arena.clear();
        assert!(arena.is_empty());
    }
}
