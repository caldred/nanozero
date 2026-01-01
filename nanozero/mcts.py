"""
nanozero/mcts.py - Monte Carlo Tree Search implementation

Implements both single-threaded MCTS and batched MCTS for GPU efficiency.
"""
import math
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from nanozero.config import MCTSConfig
from nanozero.game import Game


class TranspositionTable:
    """
    Cache for neural network evaluations with symmetry awareness.

    Stores (policy, value) pairs keyed by canonical board state.
    When looking up, checks all symmetric variants and transforms
    the cached policy back to the original orientation.
    """

    def __init__(self, game: Game):
        """
        Initialize transposition table.

        Args:
            game: Game instance (used for symmetry operations)
        """
        self.game = game
        self.cache: Dict[bytes, Tuple[np.ndarray, float]] = {}
        self.hits = 0
        self.misses = 0

    def _get_board(self, state: np.ndarray) -> np.ndarray:
        """Extract board portion from state (handles Go's metadata row)."""
        if hasattr(self.game, 'height') and state.shape[0] > self.game.height:
            return state[:self.game.height, :]
        return state

    def _get_metadata(self, state: np.ndarray) -> Optional[bytes]:
        """Extract metadata portion from state (Go's turn/ko/passes row)."""
        if hasattr(self.game, 'height') and state.shape[0] > self.game.height:
            return state[self.game.height:, :].tobytes()
        return None

    def _canonical_key(self, state: np.ndarray) -> Tuple[bytes, int]:
        """
        Find canonical (smallest) symmetric form and which symmetry index it is.

        For games with metadata (like Go), the key includes both the canonical
        board AND the metadata, so different turns/ko/passes don't collide.

        Returns:
            Tuple of (canonical_key, symmetry_index)
        """
        board = self._get_board(state)
        canonical_board = board.tobytes()
        canonical_idx = 0

        # Generate all symmetries and find the smallest (by board only)
        dummy_policy = np.zeros(self.game.config.action_size)
        symmetries = self.game.symmetries(state, dummy_policy)

        for i, (sym_state, _) in enumerate(symmetries):
            sym_board = self._get_board(sym_state)
            sym_key = sym_board.tobytes()
            if sym_key < canonical_board:
                canonical_board = sym_key
                canonical_idx = i

        # Include metadata in the key (if present) to avoid collisions
        metadata = self._get_metadata(state)
        if metadata is not None:
            canonical_key = canonical_board + metadata
        else:
            canonical_key = canonical_board

        return canonical_key, canonical_idx

    def _transform_policy_to_original(
        self,
        cached_policy: np.ndarray,
        sym_idx: int,
        original_state: np.ndarray
    ) -> np.ndarray:
        """
        Transform cached policy from canonical form back to original orientation.

        The approach: generate symmetries from the canonical state with the cached
        policy, then find which symmetric state matches our original and return
        that policy.
        """
        if sym_idx == 0:
            return cached_policy

        # Get the canonical state
        dummy_policy = np.zeros(self.game.config.action_size)
        original_symmetries = self.game.symmetries(original_state, dummy_policy)
        canonical_state = original_symmetries[sym_idx][0]

        # Generate all symmetries from canonical state with the cached policy
        canonical_symmetries = self.game.symmetries(canonical_state, cached_policy)

        # Find which one matches our original state
        orig_board = self._get_board(original_state)
        orig_key = orig_board.tobytes()

        for sym_state, sym_policy in canonical_symmetries:
            sym_board = self._get_board(sym_state)
            if sym_board.tobytes() == orig_key:
                return sym_policy

        # Fallback (shouldn't happen if symmetries are implemented correctly)
        return cached_policy

    def get(self, state: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        """
        Look up cached evaluation, checking symmetries.

        Args:
            state: Game state to look up

        Returns:
            Tuple of (policy, value) if found, None otherwise
        """
        canonical_key, sym_idx = self._canonical_key(state)

        if canonical_key in self.cache:
            self.hits += 1
            cached_policy, cached_value = self.cache[canonical_key]
            # Transform policy back to original orientation
            policy = self._transform_policy_to_original(cached_policy, sym_idx, state)
            return policy, cached_value

        self.misses += 1
        return None

    def put(self, state: np.ndarray, policy: np.ndarray, value: float):
        """
        Store evaluation in canonical form.

        Args:
            state: Game state
            policy: Policy from neural network
            value: Value from neural network
        """
        canonical_key, sym_idx = self._canonical_key(state)

        # Transform policy to canonical orientation before storing
        if sym_idx != 0:
            symmetries = self.game.symmetries(state, policy)
            canonical_policy = symmetries[sym_idx][1]
        else:
            canonical_policy = policy

        self.cache[canonical_key] = (canonical_policy.copy(), value)

    def clear(self):
        """Clear cache (call between searches during training)."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> Tuple[int, int, int]:
        """Return (hits, misses, cache_size)."""
        return self.hits, self.misses, len(self.cache)


class Node:
    """
    MCTS tree node.

    Stores visit statistics and child nodes for a single game state.
    """

    def __init__(self, prior: float = 0.0):
        """
        Initialize a node.

        Args:
            prior: Prior probability from policy network P(a|s)
        """
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[int, Node] = {}

    def value(self) -> float:
        """
        Get mean value Q = W / N.

        Returns:
            Average value, or 0 if unvisited
        """
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expanded(self) -> bool:
        """Check if node has been expanded (has children)."""
        return len(self.children) > 0


def ucb_score(parent: Node, child: Node, c_puct: float) -> float:
    """
    Compute UCB score for action selection.

    UCB = Q + c_puct * P * sqrt(N_parent) / (1 + N_child)

    Args:
        parent: Parent node
        child: Child node for this action
        c_puct: Exploration constant

    Returns:
        UCB score for selecting this child
    """
    # Exploration term: P * sqrt(N_parent) / (1 + N_child)
    exploration = c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
    # Exploitation term: Q value (from child's perspective, negate for parent's view)
    exploitation = -child.value()
    return exploitation + exploration


class MCTS:
    """
    Single-threaded Monte Carlo Tree Search.

    Used for reference implementation and debugging.
    """

    def __init__(self, game: Game, config: MCTSConfig):
        """
        Initialize MCTS.

        Args:
            game: Game instance
            config: MCTS configuration
        """
        self.game = game
        self.config = config

    def run(
        self,
        state: np.ndarray,
        model: torch.nn.Module,
        num_simulations: Optional[int] = None,
        add_noise: bool = True
    ) -> np.ndarray:
        """
        Run MCTS from a given state.

        Args:
            state: Game state (raw board values)
            model: Neural network for policy/value prediction
            num_simulations: Number of MCTS simulations (default from config)
            add_noise: Whether to add Dirichlet noise at root (for exploration)

        Returns:
            Policy array of shape (action_size,) with visit count distribution
        """
        if num_simulations is None:
            num_simulations = self.config.num_simulations

        root = Node()
        device = next(model.parameters()).device

        # Expand root node (canonicalize internally)
        self._expand(root, state, model, device)

        # Add Dirichlet noise to root for exploration
        if add_noise:
            self._add_dirichlet_noise(root)

        # Run simulations
        for _ in range(num_simulations):
            node = root
            search_path = [node]
            current_state = state.copy()

            # SELECT: traverse tree until we reach an unexpanded node or terminal
            while node.expanded() and not self.game.is_terminal(current_state):
                action, node = self._select_child(node)
                current_state = self.game.next_state(current_state, action)
                search_path.append(node)

            # Get value for leaf
            if self.game.is_terminal(current_state):
                # Terminal state: get actual reward
                value = self.game.terminal_reward(current_state)
            else:
                # Non-terminal leaf: expand and use network value
                value = self._expand(node, current_state, model, device)

            # BACKUP: propagate value up the tree
            self._backup(search_path, value)

        # Return visit count distribution as policy
        return self._get_policy(root)

    def _expand(
        self,
        node: Node,
        state: np.ndarray,
        model: torch.nn.Module,
        device: torch.device
    ) -> float:
        """
        Expand a node using the neural network.

        Args:
            node: Node to expand
            state: Game state at this node
            model: Neural network
            device: Device for inference

        Returns:
            Value estimate from neural network
        """
        # Get legal actions
        legal_actions = self.game.legal_actions(state)

        # Get policy and value from network (canonical perspective)
        canonical = self.game.canonical_state(state)
        state_tensor = self.game.to_tensor(canonical).unsqueeze(0).to(device)
        action_mask = torch.from_numpy(
            self.game.legal_actions_mask(state)
        ).unsqueeze(0).float().to(device)

        policy, value = model.predict(state_tensor, action_mask)
        policy = policy.cpu().numpy()[0]
        value = value.cpu().item()

        # Create child nodes for each legal action
        for action in legal_actions:
            node.children[action] = Node(prior=policy[action])

        return value

    def _add_dirichlet_noise(self, node: Node) -> None:
        """
        Add Dirichlet noise to root node priors for exploration.

        Args:
            node: Root node to add noise to
        """
        actions = list(node.children.keys())
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(actions))

        for i, action in enumerate(actions):
            child = node.children[action]
            child.prior = (
                (1 - self.config.dirichlet_epsilon) * child.prior +
                self.config.dirichlet_epsilon * noise[i]
            )

    def _select_child(self, node: Node) -> Tuple[int, Node]:
        """
        Select the child with highest UCB score.

        Args:
            node: Parent node

        Returns:
            Tuple of (action, child_node)
        """
        best_score = float('-inf')
        best_action = -1
        best_child = None

        for action, child in node.children.items():
            score = ucb_score(node, child, self.config.c_puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _backup(self, search_path: List[Node], value: float) -> None:
        """
        Backup value through the tree.

        Args:
            search_path: List of nodes from root to leaf
            value: Value to propagate (from leaf's perspective)
        """
        # Value alternates sign as we go up (opponent's gain is our loss)
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip for opponent's perspective

    def _get_policy(self, root: Node) -> np.ndarray:
        """
        Get policy from visit counts.

        Args:
            root: Root node

        Returns:
            Policy array of shape (action_size,)
        """
        policy = np.zeros(self.game.config.action_size, dtype=np.float32)

        for action, child in root.children.items():
            policy[action] = child.visit_count

        # Normalize
        if policy.sum() > 0:
            policy /= policy.sum()

        return policy


class BatchedMCTS:
    """
    Batched MCTS for GPU-efficient search.

    Runs MCTS on multiple game states in parallel, batching neural network
    evaluations for better GPU utilization. Uses virtual loss to enable
    parallel simulations within each search.
    """

    def __init__(self, game: Game, config: MCTSConfig, virtual_loss: float = 1.0,
                 use_transposition_table: bool = True):
        """
        Initialize batched MCTS.

        Args:
            game: Game instance
            config: MCTS configuration
            virtual_loss: Virtual loss value to apply during parallel selection
            use_transposition_table: Whether to cache NN evaluations with symmetry awareness
        """
        self.game = game
        self.config = config
        self.virtual_loss = virtual_loss
        self.use_tt = use_transposition_table
        self.tt = TranspositionTable(game) if use_transposition_table else None

    def clear_cache(self):
        """Clear the transposition table. Call this after model weights change."""
        if self.tt:
            self.tt.clear()

    def search(
        self,
        states: np.ndarray,
        model: torch.nn.Module,
        num_simulations: Optional[int] = None,
        add_noise: bool = True,
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Run batched MCTS on multiple states with virtual loss parallelism.

        Uses virtual loss to run multiple simulations in parallel within each
        search, batching their neural network calls together.

        Note: batch_size controls parallelism within each search. If too large
        relative to num_simulations, the tree won't grow deep enough. Default
        is num_simulations // 8, capped at 256.

        Args:
            states: Batch of game states (shape is game-defined)
            model: Neural network for policy/value prediction
            num_simulations: Number of MCTS simulations per state
            add_noise: Whether to add Dirichlet noise at roots
            batch_size: Number of simulations to batch together per state

        Returns:
            Policy array of shape (B, action_size)
        """
        if num_simulations is None:
            num_simulations = self.config.num_simulations

        if batch_size is None:
            batch_size = max(1, min(num_simulations // 8, 256))

        num_states = states.shape[0]
        device = next(model.parameters()).device

        # Create root nodes for each state
        roots = [Node() for _ in range(num_states)]

        # Initial expansion of all roots
        self._batch_expand(roots, states, model, device)

        # Add Dirichlet noise to roots
        if add_noise:
            for root in roots:
                if root.expanded():
                    self._add_dirichlet_noise(root)

        # Run simulations in batches
        sims_done = 0
        while sims_done < num_simulations:
            # How many simulations to run this batch
            sims_this_batch = min(batch_size, num_simulations - sims_done)

            # Collect leaves from all states and all parallel sims
            all_search_paths = []  # (state_idx, search_path)
            all_leaf_states = []
            all_leaf_info = []  # (state_idx, path_idx)
            all_terminal_info = []  # (state_idx, path_idx, value)

            for state_idx, (root, state) in enumerate(zip(roots, states)):
                for sim_idx in range(sims_this_batch):
                    node = root
                    search_path = [node]
                    current_state = state.copy()

                    # Apply virtual loss to root for proper UCB exploration term
                    root.visit_count += 1

                    # SELECT with virtual loss
                    while node.expanded() and not self.game.is_terminal(current_state):
                        action, node = self._select_child(node)
                        # Apply virtual loss (add to make value() positive, so -value() is negative)
                        node.visit_count += 1
                        node.value_sum += self.virtual_loss
                        current_state = self.game.next_state(current_state, action)
                        search_path.append(node)

                    path_idx = len(all_search_paths)
                    all_search_paths.append((state_idx, search_path))

                    if self.game.is_terminal(current_state):
                        all_terminal_info.append((state_idx, path_idx, self.game.terminal_reward(current_state)))
                    elif not node.expanded():
                        all_leaf_states.append(current_state)
                        all_leaf_info.append((state_idx, path_idx))

            # Batch expand all non-terminal leaves at once
            if all_leaf_states:
                leaf_states_arr = np.stack(all_leaf_states)
                leaf_nodes = [all_search_paths[info[1]][1][-1] for info in all_leaf_info]
                values = self._batch_expand_leaves(leaf_nodes, leaf_states_arr, model, device)

                # Backup and remove virtual loss
                for (state_idx, path_idx), value in zip(all_leaf_info, values):
                    search_path = all_search_paths[path_idx][1]
                    self._backup_with_virtual_loss_removal(search_path, value)

            # Backup terminal states and remove virtual loss
            for state_idx, path_idx, value in all_terminal_info:
                search_path = all_search_paths[path_idx][1]
                self._backup_with_virtual_loss_removal(search_path, value)

            # Handle paths that hit already-expanded nodes (no leaf to expand)
            expanded_paths = set(range(len(all_search_paths)))
            expanded_paths -= {info[1] for info in all_leaf_info}
            expanded_paths -= {info[1] for info in all_terminal_info}
            for path_idx in expanded_paths:
                search_path = all_search_paths[path_idx][1]
                # Just remove virtual loss, backup with 0 (already expanded)
                self._remove_virtual_loss(search_path)

            sims_done += sims_this_batch

        # Collect policies
        policies = np.zeros((num_states, self.game.config.action_size), dtype=np.float32)
        for i, root in enumerate(roots):
            policies[i] = self._get_policy(root)

        return policies

    def _backup_with_virtual_loss_removal(self, search_path: List[Node], value: float) -> None:
        """Backup value and remove virtual loss from path."""
        for node in reversed(search_path):
            # Remove virtual loss from all nodes including root
            node.visit_count -= 1
            if node != search_path[0]:
                node.value_sum -= self.virtual_loss
            # Apply real backup
            node.visit_count += 1
            node.value_sum += value
            value = -value

    def _remove_virtual_loss(self, search_path: List[Node]) -> None:
        """Remove virtual loss from path without backup."""
        # Remove root's visit count VL
        search_path[0].visit_count -= 1
        # Remove VL from other nodes
        for node in search_path[1:]:
            node.visit_count -= 1
            node.value_sum -= self.virtual_loss

    def _batch_expand(
        self,
        nodes: List[Node],
        states: np.ndarray,
        model: torch.nn.Module,
        device: torch.device
    ) -> np.ndarray:
        """
        Expand multiple nodes in a batch.

        Args:
            nodes: List of nodes to expand
            states: Batch of states (shape is game-defined)
            model: Neural network
            device: Device for inference

        Returns:
            Array of values from network, shape (B,)
        """
        batch_size = len(nodes)

        # Prepare batch tensors (canonical perspective for each state)
        state_tensors = torch.stack([
            self.game.to_tensor(self.game.canonical_state(states[i])) for i in range(batch_size)
        ]).to(device)

        action_masks = torch.stack([
            torch.from_numpy(self.game.legal_actions_mask(states[i]))
            for i in range(batch_size)
        ]).float().to(device)

        # Get predictions
        policies, values = model.predict(state_tensors, action_masks)
        policies = policies.cpu().numpy()
        values = values.cpu().numpy().flatten()

        # Create children for each node
        for i, node in enumerate(nodes):
            legal_actions = self.game.legal_actions(states[i])
            for action in legal_actions:
                node.children[action] = Node(prior=policies[i, action])

        return values

    def _batch_expand_leaves(
        self,
        nodes: List[Node],
        states: np.ndarray,
        model: torch.nn.Module,
        device: torch.device
    ) -> List[float]:
        """
        Expand leaf nodes in a batch, using transposition table if enabled.

        Args:
            nodes: List of leaf nodes to expand
            states: Batch of leaf states (shape is game-defined)
            model: Neural network
            device: Device for inference

        Returns:
            List of values for each leaf
        """
        batch_size = len(nodes)
        results = [None] * batch_size

        if not self.tt:
            # No cache - just evaluate all
            return self._batch_expand_leaves_no_cache(nodes, states, model, device)

        # Check transposition table for cache hits
        miss_indices = []
        miss_canonical_keys = []

        for i in range(batch_size):
            state = states[i]
            node = nodes[i]

            # Check cache
            cached = self.tt.get(state)
            if cached is not None:
                policy, value = cached
                # Create children from cached policy
                legal_actions = self.game.legal_actions(state)
                for action in legal_actions:
                    node.children[action] = Node(prior=policy[action])
                results[i] = value
            else:
                # Cache miss - record canonical key for deduplication
                canonical_key, _ = self.tt._canonical_key(state)
                miss_indices.append(i)
                miss_canonical_keys.append(canonical_key)

        # Deduplicate misses by canonical key - only evaluate unique positions
        if miss_indices:
            # Group misses by canonical key
            unique_keys = {}  # canonical_key -> first index in miss_indices
            for j, (idx, key) in enumerate(zip(miss_indices, miss_canonical_keys)):
                if key not in unique_keys:
                    unique_keys[key] = j

            # Only evaluate unique positions
            unique_indices = list(unique_keys.values())
            unique_states = [states[miss_indices[j]] for j in unique_indices]

            # Prepare batch tensors
            state_tensors = torch.stack([
                self.game.to_tensor(self.game.canonical_state(s)) for s in unique_states
            ]).to(device)

            action_masks = torch.stack([
                torch.from_numpy(self.game.legal_actions_mask(s))
                for s in unique_states
            ]).float().to(device)

            # Get predictions for unique positions only
            policies, values = model.predict(state_tensors, action_masks)
            policies = policies.cpu().numpy()
            values = values.cpu().numpy().flatten()

            # Store unique results in cache
            for j, state in enumerate(unique_states):
                self.tt.put(state, policies[j], values[j])

            # Now expand all miss nodes using cached results
            for idx in miss_indices:
                state = states[idx]
                node = nodes[idx]

                # Get from cache (guaranteed hit now)
                policy, value = self.tt.get(state)

                # Create children
                legal_actions = self.game.legal_actions(state)
                for action in legal_actions:
                    node.children[action] = Node(prior=policy[action])

                results[idx] = value

        return results

    def _batch_expand_leaves_no_cache(
        self,
        nodes: List[Node],
        states: np.ndarray,
        model: torch.nn.Module,
        device: torch.device
    ) -> List[float]:
        """Expand leaves without using cache."""
        batch_size = len(nodes)

        state_tensors = torch.stack([
            self.game.to_tensor(self.game.canonical_state(states[i])) for i in range(batch_size)
        ]).to(device)

        action_masks = torch.stack([
            torch.from_numpy(self.game.legal_actions_mask(states[i]))
            for i in range(batch_size)
        ]).float().to(device)

        policies, values = model.predict(state_tensors, action_masks)
        policies = policies.cpu().numpy()
        values = values.cpu().numpy().flatten()

        for i, node in enumerate(nodes):
            legal_actions = self.game.legal_actions(states[i])
            for action in legal_actions:
                node.children[action] = Node(prior=policies[i, action])

        return values.tolist()

    def _add_dirichlet_noise(self, node: Node) -> None:
        """Add Dirichlet noise to root node priors for exploration."""
        actions = list(node.children.keys())
        if len(actions) == 0:
            return
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(actions))

        for i, action in enumerate(actions):
            child = node.children[action]
            child.prior = (
                (1 - self.config.dirichlet_epsilon) * child.prior +
                self.config.dirichlet_epsilon * noise[i]
            )

    def _select_child(self, node: Node) -> Tuple[int, Node]:
        """Select the child with highest UCB score."""
        best_score = float('-inf')
        best_action = -1
        best_child = None

        for action, child in node.children.items():
            score = ucb_score(node, child, self.config.c_puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _backup(self, search_path: List[Node], value: float) -> None:
        """Backup value through the tree with alternating sign."""
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    def _get_policy(self, root: Node) -> np.ndarray:
        """Get policy from visit counts."""
        policy = np.zeros(self.game.config.action_size, dtype=np.float32)

        for action, child in root.children.items():
            policy[action] = child.visit_count

        # Normalize
        if policy.sum() > 0:
            policy /= policy.sum()

        return policy


def sample_action(probs: np.ndarray, temperature: float = 1.0) -> int:
    """
    Sample an action from probability distribution.

    Args:
        probs: Probability distribution over actions, shape (action_size,)
        temperature: Temperature for sampling.
                    0 = greedy (argmax)
                    1 = sample from distribution
                    >1 = more uniform
                    <1 = more peaked

    Returns:
        Sampled action index
    """
    if temperature == 0:
        # Greedy selection
        return int(np.argmax(probs))

    # Apply temperature
    if temperature != 1.0:
        # Use log to avoid numerical issues with power
        log_probs = np.log(probs + 1e-4)
        log_probs = log_probs / temperature
        probs = np.exp(log_probs - np.max(log_probs))  # Subtract max for numerical stability
        probs = probs / probs.sum()

    # Sample from distribution
    return int(np.random.choice(len(probs), p=probs))
