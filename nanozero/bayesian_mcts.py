"""
nanozero/bayesian_mcts.py - Bayesian Best Arm Identification MCTS

Implements MCTS using Gaussian beliefs and Thompson sampling instead of PUCT.
Optimizes for identifying the best action rather than cumulative regret.
"""
import math
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from nanozero.config import BayesianMCTSConfig
from nanozero.game import Game
from nanozero.mcts import TranspositionTable


def normal_cdf(x: float) -> float:
    """Standard normal CDF using error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class BayesianNode:
    """
    MCTS tree node with Gaussian belief over value.

    Instead of tracking (visit_count, value_sum) like standard MCTS,
    maintains a Gaussian posterior (mu, sigma_sq) over the node's value.
    """

    def __init__(self, prior: float = 0.0, mu: float = 0.0, sigma_sq: float = 1.0):
        """
        Initialize a Bayesian node.

        Args:
            prior: Prior probability from policy network P(a|s)
            mu: Mean of value belief
            sigma_sq: Variance of value belief
        """
        self.prior = prior
        self.mu = mu
        self.sigma_sq = sigma_sq
        self.children: Dict[int, BayesianNode] = {}

    def expanded(self) -> bool:
        """Check if node has been expanded (has children)."""
        return len(self.children) > 0

    def sample(self) -> float:
        """Draw a Thompson sample from the posterior."""
        return np.random.normal(self.mu, math.sqrt(self.sigma_sq))

    def update(self, value: float, obs_var: float, min_var: float = 1e-6) -> None:
        """
        Bayesian update with an observed value.

        Uses precision-weighted combination:
            precision_new = precision_prior + precision_obs
            mu_new = (precision_prior * mu + precision_obs * value) / precision_new

        Args:
            value: Observed value to incorporate
            obs_var: Observation variance (noise in the value estimate)
            min_var: Minimum variance to maintain (numerical stability)
        """
        precision_prior = 1.0 / max(self.sigma_sq, min_var)
        precision_obs = 1.0 / max(obs_var, min_var)
        new_precision = precision_prior + precision_obs

        self.mu = (precision_prior * self.mu + precision_obs * value) / new_precision
        self.sigma_sq = max(1.0 / new_precision, min_var)

    def precision(self) -> float:
        """Return precision (inverse variance) - proxy for visit count."""
        return 1.0 / self.sigma_sq


class BayesianMCTS:
    """
    Bayesian Best Arm Identification MCTS.

    Key differences from standard BatchedMCTS:
    1. Nodes track Gaussian beliefs (mu, sigma_sq) instead of (N, W)
    2. Selection uses Top-Two Thompson Sampling with IDS allocation
    3. Expansion uses logit-shifted prior initialization
    4. Backup uses Bayesian updates with precision weighting
    5. Runs simulations sequentially (Thompson sampling provides exploration)
    """

    def __init__(
        self,
        game: Game,
        config: BayesianMCTSConfig,
        use_transposition_table: bool = True
    ):
        """
        Initialize Bayesian MCTS.

        Args:
            game: Game instance
            config: Bayesian MCTS configuration
            use_transposition_table: Whether to cache NN evaluations
        """
        self.game = game
        self.config = config
        self.use_tt = use_transposition_table
        self.tt = TranspositionTable(game) if use_transposition_table else None

    def clear_cache(self):
        """Clear the transposition table. Call after model weights change."""
        if self.tt:
            self.tt.clear()

    def search(
        self,
        states: np.ndarray,
        model: torch.nn.Module,
        num_simulations: Optional[int] = None
    ) -> np.ndarray:
        """
        Run Bayesian MCTS on a batch of states.

        Unlike BatchedMCTS, simulations run sequentially per state since
        Thompson sampling provides natural exploration without virtual loss.
        No Dirichlet noise is needed - Thompson sampling explores naturally.

        Supports early stopping when confident about the best action.

        Args:
            states: Batch of game states, shape (B, ...)
            model: Neural network for policy/value prediction
            num_simulations: Number of MCTS simulations per state

        Returns:
            Policy array of shape (B, action_size)
        """
        if num_simulations is None:
            num_simulations = self.config.num_simulations

        num_states = states.shape[0]
        device = next(model.parameters()).device
        policies = np.zeros((num_states, self.game.config.action_size), dtype=np.float32)

        # Handle terminal states: return uniform policy over legal actions (or zeros if none)
        non_terminal_indices = []
        non_terminal_states = []
        for i in range(num_states):
            if self.game.is_terminal(states[i]):
                # Terminal state: put uniform mass on legal actions (if any)
                legal = self.game.legal_actions(states[i])
                if legal:
                    for a in legal:
                        policies[i, a] = 1.0 / len(legal)
            else:
                non_terminal_indices.append(i)
                non_terminal_states.append(states[i])

        if not non_terminal_states:
            return policies

        # Batch expand roots for non-terminal states
        non_terminal_states = np.stack(non_terminal_states)
        roots, _ = self._batch_expand_roots(non_terminal_states, model, device)

        # Run simulations sequentially per state
        for idx, (state_idx, root) in enumerate(zip(non_terminal_indices, roots)):
            state = states[state_idx]

            for sim in range(num_simulations):
                self._run_simulation(root, state, model, device)

                # Early stopping check
                if (self.config.early_stopping and
                    sim >= self.config.min_simulations - 1 and
                    self._should_stop_early(root)):
                    break

            policies[state_idx] = self._get_policy(root)

        return policies

    def _should_stop_early(self, root: BayesianNode) -> bool:
        """
        Check if we should stop early based on confidence about the best action.

        Uses P(leader > challenger) as a lower bound on P(leader is optimal).
        For Gaussian beliefs: P(X > Y) = Φ((μ_X - μ_Y) / sqrt(σ_X² + σ_Y²))

        Returns:
            True if P(leader is optimal) > confidence_threshold
        """
        if len(root.children) <= 1:
            return True  # Only one action, we're done

        # Find leader and challenger by mean value (from parent's perspective)
        children = list(root.children.values())
        children.sort(key=lambda c: -c.mu, reverse=True)
        leader = children[0]
        challenger = children[1]

        # P(leader > challenger) using Gaussian CDF
        mu_diff = (-leader.mu) - (-challenger.mu)
        std_diff = math.sqrt(leader.sigma_sq + challenger.sigma_sq)

        if std_diff < 1e-10:
            # Degenerate case: both have near-zero variance
            return mu_diff > 0

        prob_leader_better = normal_cdf(mu_diff / std_diff)

        return prob_leader_better >= self.config.confidence_threshold

    def _run_simulation(
        self,
        root: BayesianNode,
        state: np.ndarray,
        model: torch.nn.Module,
        device: torch.device
    ) -> None:
        """
        Run a single MCTS simulation from root.

        Args:
            root: Root node to start from
            state: Game state at root
            model: Neural network
            device: Device for inference
        """
        node = root
        search_path: List[Tuple[BayesianNode, int]] = []  # (parent, action)
        current_state = state.copy()

        # SELECT: traverse until unexpanded node or terminal
        while node.expanded() and not self.game.is_terminal(current_state):
            action, child = self._select_child_thompson_ids(node)
            search_path.append((node, action))
            current_state = self.game.next_state(current_state, action)
            node = child

        # Get leaf value
        if self.game.is_terminal(current_state):
            value = self.game.terminal_reward(current_state)
        else:
            # Expand leaf node
            value = self._expand(node, current_state, model, device)

        # BACKUP: propagate value up the tree
        self._backup(search_path, value)

    def _expand(
        self,
        node: BayesianNode,
        state: np.ndarray,
        model: torch.nn.Module,
        device: torch.device
    ) -> float:
        """
        Expand a node using logit-shifted prior initialization.

        For each legal action a:
            H = entropy of policy = -sum(p * log(p))
            mu_a = V(s) + sigma_0 * (sqrt(6)/pi) * [ln(p_a) + H]
            sigma_sq_a = sigma_0^2

        Args:
            node: Node to expand
            state: Game state at this node
            model: Neural network
            device: Device for inference

        Returns:
            Value estimate from neural network
        """
        # Check transposition table first
        if self.tt:
            cached = self.tt.get(state)
            if cached is not None:
                policy, value = cached
                self._create_children_from_policy(node, state, policy, value)
                return value

        # Get policy and value from network
        canonical = self.game.canonical_state(state)
        state_tensor = self.game.to_tensor(canonical).unsqueeze(0).to(device)
        action_mask = torch.from_numpy(
            self.game.legal_actions_mask(state)
        ).unsqueeze(0).float().to(device)

        policy, value = model.predict(state_tensor, action_mask)
        policy = policy.cpu().numpy()[0]
        value = value.cpu().item()

        # Store in transposition table
        if self.tt:
            self.tt.put(state, policy, value)

        # Create children with logit-shifted priors
        self._create_children_from_policy(node, state, policy, value)

        return value

    def _create_children_from_policy(
        self,
        node: BayesianNode,
        state: np.ndarray,
        policy: np.ndarray,
        value: float
    ) -> None:
        """
        Create child nodes with logit-shifted prior initialization.

        Args:
            node: Parent node to add children to
            state: Game state
            policy: Policy probabilities from network
            value: Value estimate from network
        """
        legal_actions = self.game.legal_actions(state)
        sigma_0 = self.config.sigma_0

        # Compute policy entropy for centering
        eps = 1e-8
        legal_probs = np.array([policy[a] for a in legal_actions])
        legal_probs = legal_probs / (legal_probs.sum() + eps)  # Renormalize over legal
        log_probs = np.log(legal_probs + eps)
        entropy = -np.sum(legal_probs * log_probs)

        # Logit-shift scale factor
        scale = sigma_0 * (math.sqrt(6) / math.pi)

        for i, action in enumerate(legal_actions):
            # Logit-shifted prior mean
            mu = value + scale * (log_probs[i] + entropy)
            # Prior variance
            sigma_sq = sigma_0 ** 2

            node.children[action] = BayesianNode(
                prior=policy[action],
                mu=mu,
                sigma_sq=sigma_sq
            )

    def _batch_expand_roots(
        self,
        states: np.ndarray,
        model: torch.nn.Module,
        device: torch.device
    ) -> Tuple[List[BayesianNode], np.ndarray]:
        """
        Batch expand all root nodes at once.

        Args:
            states: Batch of states
            model: Neural network
            device: Device for inference

        Returns:
            Tuple of (root nodes, values)
        """
        batch_size = states.shape[0]
        roots = [BayesianNode() for _ in range(batch_size)]

        # Check cache for hits
        cache_hits = [None] * batch_size
        miss_indices = []

        if self.tt:
            for i in range(batch_size):
                cached = self.tt.get(states[i])
                if cached is not None:
                    cache_hits[i] = cached
                else:
                    miss_indices.append(i)
        else:
            miss_indices = list(range(batch_size))

        # Batch evaluate cache misses
        if miss_indices:
            miss_states = np.stack([states[i] for i in miss_indices])

            state_tensors = torch.stack([
                self.game.to_tensor(self.game.canonical_state(s)) for s in miss_states
            ]).to(device)

            action_masks = torch.stack([
                torch.from_numpy(self.game.legal_actions_mask(s))
                for s in miss_states
            ]).float().to(device)

            policies, values = model.predict(state_tensors, action_masks)
            policies = policies.cpu().numpy()
            values = values.cpu().numpy().flatten()

            # Store in cache and fill in results
            for j, idx in enumerate(miss_indices):
                if self.tt:
                    self.tt.put(states[idx], policies[j], values[j])
                cache_hits[idx] = (policies[j], values[j])

        # Create children for all roots
        all_values = np.zeros(batch_size, dtype=np.float32)
        for i, root in enumerate(roots):
            policy, value = cache_hits[i]
            self._create_children_from_policy(root, states[i], policy, value)
            all_values[i] = value

        return roots, all_values

    def _select_child_thompson_ids(
        self,
        node: BayesianNode
    ) -> Tuple[int, BayesianNode]:
        """
        Top-Two Thompson Sampling with IDS allocation.

        1. Draw Thompson sample from each child's posterior
        2. Leader I = argmax of samples
        3. Challenger J = second highest
        4. Compute allocation: beta = (precision_I + alpha) / (precision_I + precision_J + 2*alpha)
        5. Select Challenger with probability beta, else Leader

        Args:
            node: Parent node with children

        Returns:
            Tuple of (action, child_node)
        """
        children = list(node.children.items())
        if len(children) == 1:
            return children[0]

        # Draw Thompson samples (from parent's perspective)
        samples = [(action, child, -child.sample()) for action, child in children]
        samples.sort(key=lambda x: x[2], reverse=True)

        # Leader and Challenger
        leader_action, leader_node, _ = samples[0]
        challenger_action, challenger_node, _ = samples[1]

        # IDS allocation probability
        alpha = self.config.ids_alpha
        precision_i = leader_node.precision()
        precision_j = challenger_node.precision()

        # beta = probability of selecting challenger
        # High leader precision → we're confident about leader → explore challenger more
        # High challenger precision → we're confident about challenger → explore leader more
        beta = (precision_i + alpha) / (precision_i + precision_j + 2 * alpha)

        # Select challenger with probability beta
        if np.random.random() < beta:
            return challenger_action, challenger_node
        else:
            return leader_action, leader_node

    def _backup(
        self,
        search_path: List[Tuple[BayesianNode, int]],
        leaf_value: float
    ) -> None:
        """
        Bayesian backup through the tree.

        Updates each child node on the path with the observed value,
        negating at each level for two-player games.

        Args:
            search_path: List of (parent_node, action) pairs from root to leaf
            leaf_value: Value at the leaf (from leaf's perspective)
        """
        value = leaf_value

        for parent, action in reversed(search_path):
            child = parent.children[action]
            # Update child's belief with observed value
            child.update(value, self.config.obs_var, self.config.min_variance)
            # Negate for opponent's perspective
            value = -value

    def _get_policy(self, root: BayesianNode, num_samples: int = 100) -> np.ndarray:
        """
        Get policy from probability of optimality.

        Draws many Thompson samples and counts how often each child wins.

        Args:
            root: Root node
            num_samples: Number of Thompson samples for policy estimation

        Returns:
            Policy array of shape (action_size,)
        """
        policy = np.zeros(self.game.config.action_size, dtype=np.float32)

        if not root.expanded():
            return policy

        actions = list(root.children.keys())
        children = [root.children[a] for a in actions]

        # Draw samples and count wins (from parent's perspective)
        for _ in range(num_samples):
            samples = [-child.sample() for child in children]
            winner = np.argmax(samples)
            policy[actions[winner]] += 1

        # Normalize
        if policy.sum() > 0:
            policy /= policy.sum()

        return policy


def sample_action(probs: np.ndarray, temperature: float = 1.0) -> int:
    """
    Sample an action from probability distribution.

    Args:
        probs: Probability distribution over actions, shape (action_size,)
        temperature: Temperature for sampling (0 = greedy)

    Returns:
        Sampled action index
    """
    if temperature == 0:
        return int(np.argmax(probs))

    if temperature != 1.0:
        log_probs = np.log(probs + 1e-8)
        log_probs = log_probs / temperature
        probs = np.exp(log_probs - np.max(log_probs))
        probs = probs / probs.sum()

    return int(np.random.choice(len(probs), p=probs))
