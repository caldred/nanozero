"""
scripts/test_ttts_controlled.py - Controlled test of TTTS with mock values

Creates a minimal environment where we fully control the NN outputs
to understand and debug TTTS behavior step by step.

Usage:
    python -m scripts.test_ttts_controlled
"""
import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


def normal_cdf(x: float) -> float:
    """Standard normal CDF using error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass
class MockConfig:
    """Minimal config for testing."""
    sigma_0: float = 1.0
    obs_var: float = 0.25
    prune_threshold: float = 0.01


class BayesianNode:
    """Copy of BayesianNode for isolated testing."""

    def __init__(self, prior: float = 0.0, mu: float = 0.0, sigma_sq: float = 1.0):
        self.prior = prior
        self.mu = mu
        self.sigma_sq = sigma_sq
        self.children: Dict[int, 'BayesianNode'] = {}
        self.agg_mu: Optional[float] = None
        self.agg_sigma_sq: Optional[float] = None

    def expanded(self) -> bool:
        return len(self.children) > 0

    def sample(self) -> float:
        return np.random.normal(self.mu, math.sqrt(self.sigma_sq))

    def precision(self) -> float:
        return 1.0 / self.sigma_sq

    def update(self, value: float, obs_var: float, min_var: float = 1e-6) -> None:
        precision_prior = 1.0 / max(self.sigma_sq, min_var)
        precision_obs = 1.0 / max(obs_var, min_var)
        new_precision = precision_prior + precision_obs

        self.mu = (precision_prior * self.mu + precision_obs * value) / new_precision
        self.sigma_sq = max(1.0 / new_precision, min_var)


def create_children_with_logit_shift(
    parent_value: float,
    priors: List[float],
    sigma_0: float
) -> Dict[int, BayesianNode]:
    """Create children using logit-shifted initialization."""
    children = {}
    eps = 1e-8
    scale = sigma_0 * (math.sqrt(6) / math.pi)

    # Compute entropy
    log_priors = [math.log(p + eps) for p in priors]
    entropy = -sum(p * lp for p, lp in zip(priors, log_priors))

    for action, (prior, log_prior) in enumerate(zip(priors, log_priors)):
        # Children store values from child's (opponent's) perspective
        mu = -parent_value - scale * (log_prior + entropy)
        sigma_sq = sigma_0 ** 2
        children[action] = BayesianNode(prior=prior, mu=mu, sigma_sq=sigma_sq)

    return children


def aggregate_children(node: BayesianNode, prune_threshold: float = 0.01) -> None:
    """Compute aggregated belief from children."""
    if not node.children:
        return

    children = list(node.children.values())
    n = len(children)

    if n == 1:
        c = children[0]
        node.agg_mu = -c.mu
        node.agg_sigma_sq = c.sigma_sq
        return

    # Get child beliefs from parent's perspective
    mus = np.array([-c.mu for c in children])
    sigma_sqs = np.array([c.sigma_sq for c in children])

    # Find leader and challenger
    sorted_idx = np.argsort(mus)[::-1]
    leader_idx = sorted_idx[0]
    challenger_idx = sorted_idx[1]

    mu_L, sigma_sq_L = mus[leader_idx], sigma_sqs[leader_idx]
    mu_C, sigma_sq_C = mus[challenger_idx], sigma_sqs[challenger_idx]

    # Compute optimality scores
    scores = np.zeros(n)
    for i in range(n):
        if i == leader_idx:
            diff = mu_L - mu_C
            std = math.sqrt(sigma_sq_L + sigma_sq_C)
        else:
            diff = mus[i] - mu_L
            std = math.sqrt(sigma_sqs[i] + sigma_sq_L)

        if std > 1e-10:
            scores[i] = normal_cdf(diff / std)
        else:
            scores[i] = 1.0 if diff > 0 else 0.0

    # Soft prune and normalize
    scores[scores < prune_threshold] = 0.0
    total = scores.sum()
    weights = scores / total if total > 1e-10 else np.ones(n) / n

    # Aggregated mean
    node.agg_mu = float(np.sum(weights * mus))

    # Aggregated variance (linear weights)
    disagreement = (mus - node.agg_mu) ** 2
    node.agg_sigma_sq = float(np.sum(weights * (sigma_sqs + disagreement)))


def get_policy(node: BayesianNode) -> np.ndarray:
    """Get policy from optimality weights."""
    if not node.expanded():
        return np.array([])

    actions = list(node.children.keys())
    children = [node.children[a] for a in actions]
    n = len(children)

    if n == 1:
        policy = np.zeros(n)
        policy[0] = 1.0
        return policy

    mus = np.array([-c.mu for c in children])
    sigma_sqs = np.array([c.sigma_sq for c in children])

    sorted_idx = np.argsort(mus)[::-1]
    leader_idx = sorted_idx[0]
    challenger_idx = sorted_idx[1]

    scores = np.zeros(n)
    mu_L, sigma_sq_L = mus[leader_idx], sigma_sqs[leader_idx]
    mu_C, sigma_sq_C = mus[challenger_idx], sigma_sqs[challenger_idx]

    for i in range(n):
        if i == leader_idx:
            diff = mu_L - mu_C
            std = math.sqrt(sigma_sq_L + sigma_sq_C)
        else:
            diff = mus[i] - mu_L
            std = math.sqrt(sigma_sqs[i] + sigma_sq_L)

        if std > 1e-10:
            scores[i] = normal_cdf(diff / std)
        else:
            scores[i] = 1.0 if diff > 0 else 0.0

    total = scores.sum()
    return scores / total if total > 1e-10 else np.ones(n) / n


def thompson_select(node: BayesianNode) -> int:
    """Select action using Thompson sampling."""
    samples = {a: -c.sample() for a, c in node.children.items()}  # Parent perspective
    return max(samples, key=samples.get)


def print_node(node: BayesianNode, label: str = "Node"):
    """Print node state."""
    print(f"\n{label}:")
    print(f"  mu={node.mu:.4f}, sigma_sq={node.sigma_sq:.6f}, prec={node.precision():.2f}")
    if node.agg_mu is not None:
        print(f"  agg_mu={node.agg_mu:.4f}, agg_sigma_sq={node.agg_sigma_sq:.6f}")

    if node.children:
        print(f"  Children ({len(node.children)}):")
        for a, c in sorted(node.children.items()):
            print(f"    a={a}: prior={c.prior:.3f}, mu={c.mu:.4f}, σ²={c.sigma_sq:.6f}, -mu(parent view)={-c.mu:.4f}")


def test_scenario_1():
    """
    Scenario 1: Uniform priors, single NN evaluation

    Network says: value=0.5, uniform policy [0.33, 0.33, 0.33]
    This tests logit-shifted initialization.
    """
    print("\n" + "="*70)
    print("SCENARIO 1: Uniform priors, single evaluation")
    print("="*70)

    config = MockConfig(sigma_0=1.0, obs_var=0.25)

    # Root node
    root = BayesianNode(mu=0.0, sigma_sq=1.0)

    # Simulate NN evaluation
    parent_value = 0.5
    priors = [1/3, 1/3, 1/3]

    print(f"\nNN output: value={parent_value}, priors={priors}")

    # Create children with logit-shifted initialization
    root.children = create_children_with_logit_shift(parent_value, priors, config.sigma_0)

    # Update root with observed value
    root.update(parent_value, config.obs_var)

    # Aggregate children
    aggregate_children(root, config.prune_threshold)

    print_node(root, "Root after expansion")

    # Get policy
    policy = get_policy(root)
    print(f"\n  Policy: {[f'{p:.4f}' for p in policy]}")

    # With uniform priors and uniform logits, all children should have same mu
    # So policy should be uniform (each ~0.5 probability for pairwise comparison)


def test_scenario_2():
    """
    Scenario 2: Non-uniform priors, single NN evaluation

    Network says: value=0.5, policy [0.6, 0.3, 0.1]
    Tests how logit shift creates mean differences.
    """
    print("\n" + "="*70)
    print("SCENARIO 2: Non-uniform priors, single evaluation")
    print("="*70)

    config = MockConfig(sigma_0=1.0, obs_var=0.25)

    root = BayesianNode(mu=0.0, sigma_sq=1.0)

    parent_value = 0.5
    priors = [0.6, 0.3, 0.1]

    print(f"\nNN output: value={parent_value}, priors={priors}")

    root.children = create_children_with_logit_shift(parent_value, priors, config.sigma_0)
    root.update(parent_value, config.obs_var)
    aggregate_children(root, config.prune_threshold)

    print_node(root, "Root after expansion")

    policy = get_policy(root)
    print(f"\n  Policy: {[f'{p:.4f}' for p in policy]}")
    print(f"  Expected: Action 0 should have highest prob (highest prior)")


def test_scenario_3():
    """
    Scenario 3: Multiple simulations with controlled leaf values

    Tests backup and variance reduction.
    """
    print("\n" + "="*70)
    print("SCENARIO 3: Multiple simulations with controlled values")
    print("="*70)

    config = MockConfig(sigma_0=1.0, obs_var=0.25)

    root = BayesianNode(mu=0.0, sigma_sq=1.0)

    # Initial expansion
    parent_value = 0.0
    priors = [0.5, 0.3, 0.2]
    root.children = create_children_with_logit_shift(parent_value, priors, config.sigma_0)
    root.update(parent_value, config.obs_var)
    aggregate_children(root, config.prune_threshold)

    print_node(root, "Initial state")
    print(f"  Initial policy: {[f'{p:.4f}' for p in get_policy(root)]}")

    # Simulate 10 searches where each child leads to a fixed value
    # Values are from CHILD's perspective (what the child node evaluates to)
    # Positive = child is winning, which means parent should AVOID this action
    # We want action 2 to be best for parent, so:
    #   - Action 2 child value = -0.8 (child losing = parent winning)
    #   - Action 0 child value = +0.5 (child winning = parent losing)
    leaf_values = {0: 0.5, 1: 0.0, 2: -0.8}

    print("\n  Leaf values (child perspective): ", leaf_values)
    print("  Expected: Action 2 best for parent (-(-0.8)=+0.8)")
    print("\n  Simulating 10 backups...")
    for sim in range(10):
        # Thompson select
        action = thompson_select(root)

        # Get leaf value for this action (from child's perspective)
        leaf_value = leaf_values[action]

        # Backup: update child with value directly (no negation here!)
        # The negation happens in aggregate_children via mus = [-c.mu for c]
        child = root.children[action]
        child.update(leaf_value, config.obs_var)  # NO negation - matches real backup

        # Re-aggregate
        aggregate_children(root, config.prune_threshold)

        if sim in [0, 4, 9]:
            print(f"\n  After sim {sim+1} (visited action {action}, value={leaf_value}):")
            for a, c in root.children.items():
                print(f"    a={a}: mu={c.mu:.4f}, σ²={c.sigma_sq:.6f}, prec={c.precision():.2f}")
            print(f"    Policy: {[f'{p:.4f}' for p in get_policy(root)]}")


def test_scenario_4():
    """
    Scenario 4: Compare Thompson sampling selection statistics

    With fixed beliefs, how often does each action get selected?
    """
    print("\n" + "="*70)
    print("SCENARIO 4: Thompson sampling selection statistics")
    print("="*70)

    config = MockConfig(sigma_0=1.0, obs_var=0.25)

    root = BayesianNode(mu=0.0, sigma_sq=1.0)

    # Create children with DIFFERENT means (simulating after some updates)
    root.children[0] = BayesianNode(prior=0.5, mu=-0.3, sigma_sq=0.2)  # Best from parent view (+0.3)
    root.children[1] = BayesianNode(prior=0.3, mu=0.1, sigma_sq=0.2)   # Mid (-0.1)
    root.children[2] = BayesianNode(prior=0.2, mu=0.5, sigma_sq=0.2)   # Worst (-0.5)

    aggregate_children(root, config.prune_threshold)
    print_node(root, "Fixed beliefs")

    # Run Thompson sampling many times
    n_samples = 10000
    counts = {0: 0, 1: 0, 2: 0}
    for _ in range(n_samples):
        action = thompson_select(root)
        counts[action] += 1

    print(f"\n  Thompson selection over {n_samples} samples:")
    for a in sorted(counts.keys()):
        print(f"    Action {a}: {counts[a]} ({counts[a]/n_samples*100:.1f}%)")

    print(f"\n  Policy (optimality weights):")
    policy = get_policy(root)
    for a, p in enumerate(policy):
        print(f"    Action {a}: {p:.4f} ({p*100:.1f}%)")

    print("\n  The policy should roughly match selection frequencies!")


def test_scenario_5():
    """
    Scenario 5: Effect of sigma_0 on prior influence

    How does sigma_0 affect policy with strongly non-uniform priors?
    """
    print("\n" + "="*70)
    print("SCENARIO 5: Effect of sigma_0 on prior influence")
    print("="*70)

    priors = [0.7, 0.2, 0.1]
    parent_value = 0.0

    for sigma_0 in [0.1, 0.5, 1.0, 2.0]:
        config = MockConfig(sigma_0=sigma_0, obs_var=0.25)

        root = BayesianNode(mu=0.0, sigma_sq=1.0)
        root.children = create_children_with_logit_shift(parent_value, priors, sigma_0)
        root.update(parent_value, config.obs_var)
        aggregate_children(root, config.prune_threshold)

        policy = get_policy(root)

        # Child means (from parent view)
        mus = [-c.mu for c in root.children.values()]

        print(f"\n  sigma_0={sigma_0}:")
        print(f"    Child mus (parent view): {[f'{m:.4f}' for m in mus]}")
        print(f"    Policy: {[f'{p:.4f}' for p in policy]}")


def test_scenario_6():
    """
    Scenario 6: Variance collapse with multiple updates

    What happens to variance as we accumulate evidence?
    """
    print("\n" + "="*70)
    print("SCENARIO 6: Variance evolution with updates")
    print("="*70)

    config = MockConfig(sigma_0=1.0, obs_var=0.25)

    node = BayesianNode(mu=0.0, sigma_sq=1.0)

    print(f"  Initial: mu={node.mu:.4f}, σ²={node.sigma_sq:.6f}, prec={node.precision():.2f}")

    # Apply updates with consistent value
    for i in range(10):
        node.update(0.5, config.obs_var)
        if i in [0, 2, 4, 9]:
            print(f"  After {i+1} updates: mu={node.mu:.4f}, σ²={node.sigma_sq:.6f}, prec={node.precision():.2f}")

    print("\n  Precision increases linearly with updates (like visit counts)")


def test_scenario_7():
    """
    Scenario 7: Run multiple "full searches" and compare policies

    This mimics running MCTS multiple times and checking policy variance.
    """
    print("\n" + "="*70)
    print("SCENARIO 7: Policy variance across multiple searches")
    print("="*70)

    config = MockConfig(sigma_0=1.0, obs_var=0.25)

    priors = [0.4, 0.35, 0.25]
    parent_value = 0.2
    n_sims = 50
    n_trials = 5

    # Fixed leaf values from CHILD's perspective
    # Lower = child losing = parent winning = better for parent
    # So action 0 should be best since child value is most negative
    leaf_values = {0: -0.6, 1: -0.2, 2: 0.1}

    policies = []

    for trial in range(n_trials):
        np.random.seed(trial * 123)  # Different seed each trial

        root = BayesianNode(mu=0.0, sigma_sq=1.0)
        root.children = create_children_with_logit_shift(parent_value, priors, config.sigma_0)
        root.update(parent_value, config.obs_var)
        aggregate_children(root, config.prune_threshold)

        # Run simulations
        for _ in range(n_sims):
            action = thompson_select(root)
            child = root.children[action]
            child.update(leaf_values[action], config.obs_var)  # NO negation
            aggregate_children(root, config.prune_threshold)

        policy = get_policy(root)
        policies.append(policy)

        top_action = np.argmax(policy)
        print(f"  Trial {trial+1}: top={top_action}, policy={[f'{p:.3f}' for p in policy]}")

    policies = np.array(policies)
    mean_policy = policies.mean(axis=0)
    std_policy = policies.std(axis=0)

    print(f"\n  Mean policy: {[f'{p:.3f}' for p in mean_policy]}")
    print(f"  Std policy:  {[f'{s:.3f}' for s in std_policy]}")
    print(f"\n  Expected: Action 0 should dominate (highest value)")


def test_scenario_8():
    """
    Scenario 8: Noisy leaf values (simulating NN variance)

    What happens when leaf values are stochastic rather than fixed?
    This mimics what happens with a real NN that has some variance.
    """
    print("\n" + "="*70)
    print("SCENARIO 8: Noisy leaf values (simulating NN)")
    print("="*70)

    config = MockConfig(sigma_0=1.0, obs_var=0.25)

    priors = [0.4, 0.35, 0.25]
    parent_value = 0.0
    n_sims = 100
    n_trials = 5

    # Mean values (child perspective) - action 0 is best for parent
    # All values in [-1, 1] range
    mean_values = {0: -0.5, 1: -0.1, 2: 0.2}
    noise_std = 0.2  # Simulates NN evaluation noise (reduced to keep values in range)

    policies = []

    for trial in range(n_trials):
        np.random.seed(trial * 456)

        root = BayesianNode(mu=0.0, sigma_sq=1.0)
        root.children = create_children_with_logit_shift(parent_value, priors, config.sigma_0)
        root.update(parent_value, config.obs_var)
        aggregate_children(root, config.prune_threshold)

        for _ in range(n_sims):
            action = thompson_select(root)
            # Add noise to leaf value, clip to [-1, 1]
            leaf_value = mean_values[action] + np.random.normal(0, noise_std)
            leaf_value = np.clip(leaf_value, -1.0, 1.0)
            child = root.children[action]
            child.update(leaf_value, config.obs_var)
            aggregate_children(root, config.prune_threshold)

        policy = get_policy(root)
        policies.append(policy)

        top_action = np.argmax(policy)
        print(f"  Trial {trial+1}: top={top_action}, policy={[f'{p:.3f}' for p in policy]}")

    policies = np.array(policies)
    mean_policy = policies.mean(axis=0)
    std_policy = policies.std(axis=0)

    print(f"\n  Mean policy: {[f'{p:.3f}' for p in mean_policy]}")
    print(f"  Std policy:  {[f'{s:.3f}' for s in std_policy]}")
    print(f"\n  With noise_std={noise_std}, variance should still be reasonable")


def test_scenario_9():
    """
    Scenario 9: Two-level tree with grandchildren

    Simulates deeper search where backup propagates through 2 levels.
    """
    print("\n" + "="*70)
    print("SCENARIO 9: Two-level tree")
    print("="*70)

    config = MockConfig(sigma_0=1.0, obs_var=0.25)

    # Create root with 3 children
    root = BayesianNode(mu=0.0, sigma_sq=1.0)
    root.children = create_children_with_logit_shift(0.0, [0.4, 0.35, 0.25], config.sigma_0)
    root.update(0.0, config.obs_var)

    # Expand each child with grandchildren
    grandchild_priors = [0.5, 0.5]
    for action, child in root.children.items():
        child.children = create_children_with_logit_shift(-child.mu, grandchild_priors, config.sigma_0)
        child.update(-child.mu, config.obs_var)  # Update with "NN value"
        aggregate_children(child, config.prune_threshold)

    aggregate_children(root, config.prune_threshold)

    print_node(root, "Root (2-level tree)")
    for a, c in root.children.items():
        print(f"\n  Child {a} grandchildren:")
        for ga, gc in c.children.items():
            print(f"    gc {ga}: mu={gc.mu:.4f}")

    # True terminal values at grandchildren (restricted to [-1, 1])
    # Sign convention trace:
    #   gc_value = +0.6 → grandchild winning → child losing → root winning
    #   gc_value = -0.6 → grandchild losing → child winning → root losing
    gc_values = {
        0: {0: -0.8, 1: -0.6},  # gc losing → child winning → root LOSING (bad for root)
        1: {0: 0.0, 1: 0.0},    # Neutral
        2: {0: 0.6, 1: 0.4},    # gc winning → child losing → root WINNING (good for root)
    }
    # All values already in [-1, 1] range

    print(f"\n  Grandchild values (gc perspective): {gc_values}")
    print("  Expected: Action 2 best for root (gc winning = root winning)")

    n_sims = 50
    n_trials = 3

    for trial in range(n_trials):
        np.random.seed(trial * 789)

        # Reset tree
        root = BayesianNode(mu=0.0, sigma_sq=1.0)
        root.children = create_children_with_logit_shift(0.0, [0.4, 0.35, 0.25], config.sigma_0)
        root.update(0.0, config.obs_var)

        for action, child in root.children.items():
            child.children = create_children_with_logit_shift(0.0, grandchild_priors, config.sigma_0)
            aggregate_children(child, config.prune_threshold)

        aggregate_children(root, config.prune_threshold)

        for sim in range(n_sims):
            # Select child via Thompson
            child_action = thompson_select(root)
            child = root.children[child_action]

            # Select grandchild via Thompson
            gc_action = thompson_select(child)
            gc = child.children[gc_action]

            # Get terminal value
            leaf_value = gc_values[child_action][gc_action]

            # Backup mimics real BayesianMCTS:
            # Path: root -> child -> gc
            # Reversed: [(child, gc_action), (root, child_action)]

            # First: update grandchild, aggregate child
            gc.update(leaf_value, config.obs_var)
            aggregate_children(child, config.prune_threshold)
            value = child.agg_mu
            obs_var = child.agg_sigma_sq

            # Second: update child.mu with aggregated value, aggregate root
            # This is the key step - child.mu gets updated with info from grandchildren
            child.update(value, obs_var)
            aggregate_children(root, config.prune_threshold)

        policy = get_policy(root)
        top = np.argmax(policy)
        print(f"\n  Trial {trial+1}: top={top}, policy={[f'{p:.3f}' for p in policy]}")


def test_scenario_10():
    """
    Scenario 10: Realistic game scenario

    Simulates a Connect4-like situation where:
    - 7 actions (columns)
    - NN gives a strong preference for center column
    - True best action differs slightly from NN prior
    - Compare TTTS convergence behavior
    """
    print("\n" + "="*70)
    print("SCENARIO 10: Realistic 7-action game")
    print("="*70)

    config = MockConfig(sigma_0=0.5, obs_var=0.25)

    # NN priors favor center (action 3) strongly
    priors = [0.08, 0.10, 0.15, 0.27, 0.22, 0.12, 0.06]
    parent_value = 0.3  # Position slightly good for current player

    # True values (child perspective): action 4 is actually slightly better than 3
    # but NN prior doesn't know this
    true_values = {
        0: 0.1,   # child slightly winning
        1: 0.0,   # neutral
        2: -0.1,  # child slightly losing
        3: -0.25, # child losing (good for parent)
        4: -0.35, # child losing MORE (best for parent!)
        5: 0.05,  # child slightly winning
        6: 0.15,  # child winning
    }

    print(f"\n  Priors: {priors}")
    print(f"  True values (child persp): {true_values}")
    print(f"  Best for parent: action 4 (true val=-0.35)")
    print(f"  NN prior prefers: action 3 (prior=0.27)")

    n_sims = 100
    n_trials = 5

    all_policies = []

    for trial in range(n_trials):
        np.random.seed(trial * 999)

        root = BayesianNode(mu=0.0, sigma_sq=1.0)
        root.children = create_children_with_logit_shift(parent_value, priors, config.sigma_0)
        root.update(parent_value, config.obs_var)
        aggregate_children(root, config.prune_threshold)

        # Track selection counts
        selection_counts = {a: 0 for a in range(7)}

        for sim in range(n_sims):
            action = thompson_select(root)
            selection_counts[action] += 1

            # Add small noise to true value
            leaf_value = true_values[action] + np.random.normal(0, 0.1)
            leaf_value = np.clip(leaf_value, -1.0, 1.0)

            child = root.children[action]
            child.update(leaf_value, config.obs_var)
            aggregate_children(root, config.prune_threshold)

        policy = get_policy(root)
        all_policies.append(policy)

        top = np.argmax(policy)
        print(f"\n  Trial {trial+1}: top={top}")
        print(f"    Policy: {[f'{p:.3f}' for p in policy]}")
        print(f"    Selections: {[selection_counts[a] for a in range(7)]}")

    policies = np.array(all_policies)
    mean_policy = policies.mean(axis=0)
    std_policy = policies.std(axis=0)

    print(f"\n  Mean policy: {[f'{p:.3f}' for p in mean_policy]}")
    print(f"  Std policy:  {[f'{s:.3f}' for s in std_policy]}")

    # Check if we're finding the true best (action 4)
    best_action = np.argmax(mean_policy)
    print(f"\n  Converged to action {best_action} (expected: 4)")


def main():
    print("TTTS Controlled Testing Environment")
    print("====================================")
    print("\nThis script tests TTTS components with controlled inputs")
    print("to understand and verify behavior.\n")

    test_scenario_1()
    test_scenario_2()
    test_scenario_3()
    test_scenario_4()
    test_scenario_5()
    test_scenario_6()
    test_scenario_7()
    test_scenario_8()
    test_scenario_9()
    test_scenario_10()

    print("\n" + "="*70)
    print("ALL SCENARIOS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
