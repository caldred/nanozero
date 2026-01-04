//! MCTS search operations: selection and backup.
//!
//! Implements the core tree traversal and value propagation logic.

use crate::tree::TreeArena;
use crate::ucb::ucb_score;
use smallvec::SmallVec;

/// Maximum expected tree depth (for stack allocation)
const MAX_DEPTH: usize = 128;

/// A path through the search tree.
///
/// Stores node indices and actions taken from root to leaf.
#[derive(Clone, Debug)]
pub struct SearchPath {
    /// Node indices from root to leaf
    pub nodes: SmallVec<[u32; MAX_DEPTH]>,
    /// Actions taken (one fewer than nodes)
    pub actions: SmallVec<[u16; MAX_DEPTH]>,
}

impl SearchPath {
    /// Create a new empty path.
    pub fn new() -> Self {
        Self {
            nodes: SmallVec::new(),
            actions: SmallVec::new(),
        }
    }

    /// Create a path starting from a root node.
    pub fn from_root(root_idx: u32) -> Self {
        let mut path = Self::new();
        path.nodes.push(root_idx);
        path
    }

    /// Add a step to the path.
    #[inline]
    pub fn push(&mut self, action: u16, node_idx: u32) {
        self.actions.push(action);
        self.nodes.push(node_idx);
    }

    /// Get the leaf node index.
    #[inline]
    pub fn leaf(&self) -> u32 {
        *self.nodes.last().unwrap()
    }

    /// Get the depth of the path (number of actions taken).
    #[inline]
    pub fn depth(&self) -> usize {
        self.actions.len()
    }

    /// Check if the path is empty (just root).
    #[inline]
    pub fn is_root_only(&self) -> bool {
        self.actions.is_empty()
    }
}

impl Default for SearchPath {
    fn default() -> Self {
        Self::new()
    }
}

/// Select the best child according to UCB score.
///
/// Returns (action, child_node_idx) of the best child.
pub fn select_child(arena: &TreeArena, node_idx: u32, c_puct: f32) -> (u16, u32) {
    let parent = arena.get(node_idx);
    let parent_visits = parent.visit_count;
    let children = arena.get_children(node_idx);

    debug_assert!(!children.is_empty(), "select_child called on unexpanded node");

    let mut best_score = f32::NEG_INFINITY;
    let mut best_action = 0u16;
    let mut best_child_idx = 0u32;

    for child_entry in children {
        let child = arena.get(child_entry.node_idx);
        let score = ucb_score(
            parent_visits,
            child.prior,
            child.visit_count,
            child.value(),
            c_puct,
        );

        if score > best_score {
            best_score = score;
            best_action = child_entry.action;
            best_child_idx = child_entry.node_idx;
        }
    }

    (best_action, best_child_idx)
}

/// Select the best child according to UCB score with virtual loss.
///
/// Uses effective visit counts that include virtual loss to discourage
/// selecting the same path multiple times before backup.
///
/// Returns (action, child_node_idx) of the best child.
pub fn select_child_with_virtual_loss(
    arena: &TreeArena,
    node_idx: u32,
    c_puct: f32,
    virtual_loss_value: f32,
) -> (u16, u32) {
    use crate::ucb::ucb_score_with_virtual_loss;

    let parent = arena.get(node_idx);
    let parent_visits = parent.visit_count;
    let parent_vl = parent.virtual_loss;
    let children = arena.get_children(node_idx);

    debug_assert!(!children.is_empty(), "select_child called on unexpanded node");

    let mut best_score = f32::NEG_INFINITY;
    let mut best_action = 0u16;
    let mut best_child_idx = 0u32;

    for child_entry in children {
        let child = arena.get(child_entry.node_idx);
        let score = ucb_score_with_virtual_loss(
            parent_visits,
            parent_vl,
            child.prior,
            child.visit_count,
            child.virtual_loss,
            child.value(),
            c_puct,
            virtual_loss_value,
        );

        if score > best_score {
            best_score = score;
            best_action = child_entry.action;
            best_child_idx = child_entry.node_idx;
        }
    }

    (best_action, best_child_idx)
}

/// Result of selecting to a leaf node.
pub struct SelectResult {
    /// Path from root to leaf
    pub path: SearchPath,
    /// Whether the leaf is a terminal state
    pub is_terminal: bool,
    /// Whether the leaf needs expansion (not yet expanded and not terminal)
    pub needs_expansion: bool,
}

/// Select from root to an unexpanded or terminal leaf.
///
/// # Arguments
/// * `arena` - The tree arena
/// * `root_idx` - Index of the root node
/// * `c_puct` - Exploration constant
/// * `is_terminal_fn` - Function to check if current state is terminal
///
/// Returns the path from root to leaf and whether the leaf is terminal.
pub fn select_to_leaf<F>(
    arena: &TreeArena,
    root_idx: u32,
    c_puct: f32,
    mut is_terminal_fn: F,
) -> SelectResult
where
    F: FnMut(&[u16]) -> bool,
{
    let mut path = SearchPath::from_root(root_idx);
    let mut node_idx = root_idx;

    loop {
        let node = arena.get(node_idx);

        // Check if terminal
        if is_terminal_fn(&path.actions) {
            return SelectResult {
                path,
                is_terminal: true,
                needs_expansion: false,
            };
        }

        // Check if needs expansion
        if !node.expanded() {
            return SelectResult {
                path,
                is_terminal: false,
                needs_expansion: true,
            };
        }

        // Select best child
        let (action, child_idx) = select_child(arena, node_idx, c_puct);
        path.push(action, child_idx);
        node_idx = child_idx;
    }
}

/// Backup a value through the search path.
///
/// Updates visit counts and value sums for all nodes in the path.
/// Values are negated at each level since the game is zero-sum.
///
/// # Arguments
/// * `arena` - The tree arena
/// * `path` - Path from root to leaf
/// * `leaf_value` - Value at the leaf from the leaf player's perspective
pub fn backup(arena: &mut TreeArena, path: &SearchPath, leaf_value: f32) {
    let mut value = leaf_value;

    // Iterate from leaf to root
    for &node_idx in path.nodes.iter().rev() {
        let node = arena.get_mut(node_idx);
        node.visit_count += 1;
        node.value_sum += value;
        value = -value; // Negate for opponent's perspective
    }
}

/// Backup with virtual loss removal.
///
/// Same as `backup` but also removes virtual loss applied during selection.
pub fn backup_with_virtual_loss_removal(arena: &mut TreeArena, path: &SearchPath, leaf_value: f32) {
    let mut value = leaf_value;

    for &node_idx in path.nodes.iter().rev() {
        let node = arena.get_mut(node_idx);
        node.remove_virtual_loss();
        node.visit_count += 1;
        node.value_sum += value;
        value = -value;
    }
}

/// Apply virtual loss to all nodes in a path.
pub fn apply_virtual_loss(arena: &mut TreeArena, path: &SearchPath) {
    for &node_idx in path.nodes.iter() {
        arena.get_mut(node_idx).apply_virtual_loss();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_path() {
        let mut path = SearchPath::from_root(0);
        assert_eq!(path.leaf(), 0);
        assert!(path.is_root_only());

        path.push(3, 1);
        assert_eq!(path.leaf(), 1);
        assert_eq!(path.depth(), 1);
        assert!(!path.is_root_only());

        path.push(5, 2);
        assert_eq!(path.leaf(), 2);
        assert_eq!(path.depth(), 2);
    }

    #[test]
    fn test_select_child() {
        let mut arena = TreeArena::new(100);
        let root = arena.new_root();

        // Add children with different priors
        let actions = vec![0, 1, 2];
        let priors = vec![0.7, 0.2, 0.1];
        arena.add_children(root, &actions, &priors);

        // First selection should choose highest prior (action 0)
        let (action, _) = select_child(&arena, root, 1.0);
        assert_eq!(action, 0);
    }

    #[test]
    fn test_backup() {
        let mut arena = TreeArena::new(100);
        let root = arena.new_root();
        arena.add_children(root, &[0, 1], &[0.5, 0.5]);

        let child_idx = arena.get_child(root, 0).unwrap();

        let mut path = SearchPath::from_root(root);
        path.push(0, child_idx);

        // Backup a win (1.0) from leaf's perspective
        backup(&mut arena, &path, 1.0);

        // Leaf should have value 1.0
        assert_eq!(arena.get(child_idx).visit_count, 1);
        assert!((arena.get(child_idx).value() - 1.0).abs() < 1e-6);

        // Root should have value -1.0 (opponent lost)
        assert_eq!(arena.get(root).visit_count, 1);
        assert!((arena.get(root).value() - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_select_to_leaf() {
        let mut arena = TreeArena::new(100);
        let root = arena.new_root();
        arena.add_children(root, &[0, 1], &[0.5, 0.5]);

        // Select should go to child
        let result = select_to_leaf(&arena, root, 1.0, |_| false);
        assert!(!result.is_terminal);
        assert!(result.needs_expansion);
        assert_eq!(result.path.depth(), 1);
    }
}
