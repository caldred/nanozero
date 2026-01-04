//! Transposition table for MCTS with symmetry support.
//!
//! Caches NN policy/value evaluations to reuse across searches and symmetric positions.
//! Used by both PUCT and Bayesian MCTS - both cache the same NN output (policy priors + value).
//! Entries persist until explicitly cleared (typically when the model is retrained).

use std::collections::HashMap;

/// Entry storing MCTS statistics for a position.
#[derive(Clone, Debug)]
pub struct TranspositionEntry {
    /// Total visits to this node
    pub visits: u32,
    /// Sum of values (for computing mean)
    pub value_sum: f32,
    /// Child statistics: (action, prior, visits, value_sum)
    pub children: Vec<ChildStats>,
    /// Whether root Dirichlet noise has been added
    pub noise_added: bool,
}

/// Statistics for a child action.
#[derive(Clone, Debug)]
pub struct ChildStats {
    pub action: u16,
    pub prior: f32,
    pub visits: u32,
    pub value_sum: f32,
}

impl TranspositionEntry {
    /// Create a new entry with initial values.
    pub fn new() -> Self {
        Self {
            visits: 0,
            value_sum: 0.0,
            children: Vec::new(),
            noise_added: false,
        }
    }

    /// Get mean value.
    #[inline]
    pub fn value(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }

    /// Check if this node has been expanded.
    #[inline]
    pub fn expanded(&self) -> bool {
        !self.children.is_empty()
    }
}

impl Default for TranspositionEntry {
    fn default() -> Self {
        Self::new()
    }
}


/// Transposition table for standard PUCT MCTS.
///
/// Stores position statistics keyed by canonical hash (smallest among symmetries).
/// All symmetric positions share the same entry.
pub struct TranspositionTable {
    /// Map from canonical hash to entry
    entries: HashMap<u64, TranspositionEntry>,
    /// Statistics
    hits: u64,
    misses: u64,
}

impl TranspositionTable {
    /// Create a new empty transposition table.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            hits: 0,
            misses: 0,
        }
    }

    /// Create with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(capacity),
            hits: 0,
            misses: 0,
        }
    }

    /// Look up an entry by canonical hash.
    /// Returns the entry and whether it was a hit.
    pub fn get(&mut self, hash: u64) -> Option<&TranspositionEntry> {
        if let Some(entry) = self.entries.get(&hash) {
            self.hits += 1;
            Some(entry)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Get mutable reference to an entry, creating if needed.
    pub fn get_or_insert(&mut self, hash: u64) -> &mut TranspositionEntry {
        self.entries.entry(hash).or_insert_with(TranspositionEntry::new)
    }

    /// Get mutable reference to existing entry.
    pub fn get_mut(&mut self, hash: u64) -> Option<&mut TranspositionEntry> {
        self.entries.get_mut(&hash)
    }

    /// Insert or update an entry.
    pub fn insert(&mut self, hash: u64, entry: TranspositionEntry) {
        self.entries.insert(hash, entry);
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.hits = 0;
        self.misses = 0;
    }

    /// Get the number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get hit rate.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> (u64, u64, usize) {
        (self.hits, self.misses, self.entries.len())
    }
}

impl Default for TranspositionTable {
    fn default() -> Self {
        Self::new()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transposition_table() {
        let mut tt = TranspositionTable::new();

        // Miss
        assert!(tt.get(12345).is_none());

        // Insert
        let entry = tt.get_or_insert(12345);
        entry.visits = 10;
        entry.value_sum = 5.0;

        // Hit
        assert!(tt.get(12345).is_some());
        assert_eq!(tt.get(12345).unwrap().visits, 10);

        // Stats
        assert_eq!(tt.len(), 1);
        assert!(tt.hit_rate() > 0.0);
    }

    #[test]
    fn test_entry_value() {
        let mut entry = TranspositionEntry::new();
        assert_eq!(entry.value(), 0.0);

        entry.visits = 10;
        entry.value_sum = 5.0;
        assert!((entry.value() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_clear() {
        let mut tt = TranspositionTable::new();
        tt.get_or_insert(1);
        tt.get_or_insert(2);
        assert_eq!(tt.len(), 2);

        tt.clear();
        assert_eq!(tt.len(), 0);
        assert!(tt.is_empty());
    }
}
