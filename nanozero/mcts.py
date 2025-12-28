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
    evaluations for better GPU utilization.
    """

    def __init__(self, game: Game, config: MCTSConfig):
        """
        Initialize batched MCTS.

        Args:
            game: Game instance
            config: MCTS configuration
        """
        self.game = game
        self.config = config

    def search(
        self,
        states: np.ndarray,
        model: torch.nn.Module,
        num_simulations: Optional[int] = None,
        add_noise: bool = True
    ) -> np.ndarray:
        """
        Run batched MCTS on multiple states.

        Args:
            states: Batch of game states, shape (B, H, W), raw board values
            model: Neural network for policy/value prediction
            num_simulations: Number of MCTS simulations per state
            add_noise: Whether to add Dirichlet noise at roots

        Returns:
            Policy array of shape (B, action_size)
        """
        if num_simulations is None:
            num_simulations = self.config.num_simulations

        batch_size = states.shape[0]
        device = next(model.parameters()).device

        # Create root nodes for each state
        roots = [Node() for _ in range(batch_size)]

        # Initial expansion of all roots
        self._batch_expand(roots, states, model, device)

        # Add Dirichlet noise to roots
        if add_noise:
            for root in roots:
                if root.expanded():
                    self._add_dirichlet_noise(root)

        # Run simulations
        for _ in range(num_simulations):
            # For each state in batch, traverse to leaf
            nodes = []
            search_paths = []
            leaf_states = []
            leaf_indices = []  # Track which batch element this leaf belongs to
            terminal_values = []  # Store terminal values for terminal states
            terminal_indices = []  # Track which batch elements hit terminal

            for i, (root, state) in enumerate(zip(roots, states)):
                node = root
                search_path = [node]
                current_state = state.copy()

                # SELECT: traverse to leaf
                while node.expanded() and not self.game.is_terminal(current_state):
                    action, node = self._select_child(node)
                    current_state = self.game.next_state(current_state, action)
                    search_path.append(node)

                search_paths.append(search_path)

                if self.game.is_terminal(current_state):
                    # Terminal state
                    terminal_values.append(self.game.terminal_reward(current_state))
                    terminal_indices.append(i)
                elif not node.expanded():
                    # Need to expand this node
                    nodes.append(node)
                    leaf_states.append(current_state)
                    leaf_indices.append(i)

            # Batch expand non-terminal leaves
            if leaf_states:
                leaf_states_arr = np.stack(leaf_states)
                values = self._batch_expand_leaves(nodes, leaf_states_arr, model, device)

                # Backup non-terminal leaves
                for idx, value in zip(leaf_indices, values):
                    self._backup(search_paths[idx], value)

            # Backup terminal leaves
            for idx, value in zip(terminal_indices, terminal_values):
                self._backup(search_paths[idx], value)

        # Collect policies
        policies = np.zeros((batch_size, self.game.config.action_size), dtype=np.float32)
        for i, root in enumerate(roots):
            policies[i] = self._get_policy(root)

        return policies

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
            states: Batch of states, shape (B, H, W)
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
        Expand leaf nodes in a batch.

        Args:
            nodes: List of leaf nodes to expand
            states: Batch of states at these leaves, shape (B, H, W)
            model: Neural network
            device: Device for inference

        Returns:
            List of values for each leaf
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
        log_probs = np.log(probs + 1e-10)
        log_probs = log_probs / temperature
        probs = np.exp(log_probs - np.max(log_probs))  # Subtract max for numerical stability
        probs = probs / probs.sum()

    # Sample from distribution
    return int(np.random.choice(len(probs), p=probs))
