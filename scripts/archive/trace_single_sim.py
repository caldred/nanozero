"""
scripts/trace_single_sim.py - Trace a single TTTS simulation step-by-step

Saves results to /tmp/trace_single_sim.txt
"""
import math
import numpy as np
import torch
from nanozero.game import get_game
from nanozero.model import AlphaZeroTransformer
from nanozero.config import get_model_config, BayesianMCTSConfig
from nanozero.common import get_device, load_checkpoint
from nanozero.bayesian_mcts import BayesianMCTS, BayesianNode, normal_cdf


def trace_policy_extraction(root: BayesianNode, game, output: list):
    """Trace through _get_policy step by step."""
    output.append("\n=== POLICY EXTRACTION TRACE ===")

    actions = list(root.children.keys())
    children = [root.children[a] for a in actions]
    n = len(children)

    output.append(f"Actions in root.children.keys(): {actions}")
    output.append(f"Number of children: {n}")

    # Show raw child values
    output.append("\nRaw child values (child.mu = from child/opponent perspective):")
    for i, a in enumerate(actions):
        c = children[i]
        output.append(f"  actions[{i}]={a}: child.mu={c.mu:.6f}, sigma_sq={c.sigma_sq:.8f}")

    # Compute mus (from parent perspective)
    mus = np.array([-c.mu for c in children])
    sigma_sqs = np.array([c.sigma_sq for c in children])

    output.append("\nParent-perspective values (mus = -child.mu):")
    for i, a in enumerate(actions):
        output.append(f"  actions[{i}]={a}: mus[{i}]={mus[i]:.6f}")

    # Find leader and challenger
    sorted_idx = np.argsort(mus)[::-1]
    leader_idx = sorted_idx[0]
    challenger_idx = sorted_idx[1]

    output.append(f"\nSorted indices (descending by mus): {sorted_idx.tolist()}")
    output.append(f"leader_idx = {leader_idx} (action {actions[leader_idx]})")
    output.append(f"challenger_idx = {challenger_idx} (action {actions[challenger_idx]})")

    mu_L, sigma_sq_L = mus[leader_idx], sigma_sqs[leader_idx]
    mu_C, sigma_sq_C = mus[challenger_idx], sigma_sqs[challenger_idx]

    output.append(f"\nLeader stats: mu_L={mu_L:.6f}, sigma_sq_L={sigma_sq_L:.8f}")
    output.append(f"Challenger stats: mu_C={mu_C:.6f}, sigma_sq_C={sigma_sq_C:.8f}")

    # Compute scores
    output.append("\nScore computation:")
    scores = np.zeros(n)
    for i in range(n):
        if i == leader_idx:
            diff = mu_L - mu_C
            std = math.sqrt(sigma_sq_L + sigma_sq_C)
            comparison = "P(leader > challenger)"
        else:
            diff = mus[i] - mu_L
            std = math.sqrt(sigma_sqs[i] + sigma_sq_L)
            comparison = f"P(action {actions[i]} > leader)"

        if std > 1e-10:
            scores[i] = normal_cdf(diff / std)
        else:
            scores[i] = 1.0 if diff > 0 else 0.0

        output.append(f"  i={i} (action {actions[i]}): {comparison}")
        output.append(f"    diff={diff:.6f}, std={std:.6f}, z={diff/std if std > 1e-10 else 'inf':.4f}")
        output.append(f"    score={scores[i]:.6f}")

    # Normalize
    total = scores.sum()
    policy = np.zeros(game.config.action_size, dtype=np.float32)

    output.append(f"\nNormalization:")
    output.append(f"  total score = {total:.6f}")

    for i, action in enumerate(actions):
        policy[action] = scores[i] / total
        output.append(f"  policy[{action}] = scores[{i}]/total = {scores[i]:.6f}/{total:.6f} = {policy[action]:.4f}")

    output.append(f"\nFinal policy: {[f'{p:.4f}' for p in policy]}")
    output.append(f"Best action: {np.argmax(policy)} with prob {np.max(policy):.4f}")

    return policy


def trace_aggregate_children(parent: BayesianNode, output: list, label: str = ""):
    """Trace through aggregate_children step by step."""
    output.append(f"\n=== AGGREGATE CHILDREN TRACE {label} ===")

    children = list(parent.children.values())
    actions = list(parent.children.keys())
    n = len(children)

    if n == 1:
        child = children[0]
        agg_mu = -child.mu
        agg_sigma_sq = child.sigma_sq
        output.append(f"Single child: agg_mu = -child.mu = {agg_mu:.6f}")
        return

    # Get child beliefs from parent's perspective
    mus = np.array([-c.mu for c in children])
    sigma_sqs = np.array([c.sigma_sq for c in children])

    output.append(f"Children (from parent's perspective):")
    for i, a in enumerate(actions):
        output.append(f"  action {a}: mus[{i}]={mus[i]:.6f}, sigma_sqs[{i}]={sigma_sqs[i]:.8f}")

    # Find leader and challenger
    sorted_idx = np.argsort(mus)[::-1]
    leader_idx = sorted_idx[0]
    challenger_idx = sorted_idx[1]

    mu_L = mus[leader_idx]
    mu_C = mus[challenger_idx]
    sigma_sq_L = sigma_sqs[leader_idx]
    sigma_sq_C = sigma_sqs[challenger_idx]

    output.append(f"Leader: action {actions[leader_idx]}, mu={mu_L:.6f}")
    output.append(f"Challenger: action {actions[challenger_idx]}, mu={mu_C:.6f}")


def main():
    device = get_device()
    game = get_game('connect4')
    model_config = get_model_config(game.config, n_layer=4)
    model = AlphaZeroTransformer(model_config).to(device)
    load_checkpoint('checkpoints/connect4_iter150.pt', model)
    model.eval()

    # Test position
    moves = [3, 4, 6, 2, 4, 4, 6, 1, 2, 6, 2]
    state = game.initial_state()
    for m in moves:
        state = game.next_state(state, m)

    output = []
    output.append("Single Simulation Trace")
    output.append("=" * 60)
    output.append(f"\nPosition:\n{game.display(state)}")
    output.append(f"Current player: {game.current_player(state)}")
    output.append(f"Legal actions: {game.legal_actions(state)}")

    # Set up TTTS
    config = BayesianMCTSConfig(num_simulations=100, sigma_0=0.5, obs_var=0.25)
    ttts = BayesianMCTS(game, config)

    # Expand root
    output.append("\n=== INITIAL ROOT EXPANSION ===")
    roots, values = ttts._batch_expand_roots(state[np.newaxis, ...], model, device)
    root = roots[0]
    root_value = values[0]

    output.append(f"Root NN value: {root_value:.6f}")
    output.append(f"\nInitial child beliefs (after expansion):")

    for a in sorted(root.children.keys()):
        c = root.children[a]
        output.append(f"  action {a}: prior={c.prior:.4f}, child.mu={c.mu:.6f}, sigma_sq={c.sigma_sq:.4f}")
        output.append(f"           (parent perspective: -child.mu = {-c.mu:.6f})")

    output.append(f"\nRoot aggregated: agg_mu={root.agg_mu:.6f}, agg_sigma_sq={root.agg_sigma_sq:.6f}")

    # Extract initial policy
    output.append("\n=== INITIAL POLICY (before any simulations) ===")
    initial_policy = trace_policy_extraction(root, game, output)

    # Run a few simulations and trace each
    output.append("\n" + "=" * 60)
    output.append("RUNNING SIMULATIONS")
    output.append("=" * 60)

    np.random.seed(42)  # For reproducibility

    for sim in range(5):
        output.append(f"\n--- Simulation {sim+1} ---")

        # Selection
        leaf_node, search_path, leaf_state, is_terminal = ttts._select_to_leaf(root, state)

        output.append(f"Search path: {[(a, 'node') for _, a in search_path]}")
        output.append(f"Is terminal: {is_terminal}")

        if search_path:
            first_action = search_path[0][1]
            output.append(f"First action selected: {first_action}")

        # Expansion/Terminal
        if is_terminal:
            value = game.terminal_reward(leaf_state)
            output.append(f"Terminal value: {value}")
        elif not leaf_node.expanded():
            values = ttts._batch_expand_leaves([leaf_node], [leaf_state], model, device)
            value = values[0]
            output.append(f"Expansion value: {value:.6f}")
        else:
            output.append("Leaf already expanded (shouldn't happen often)")
            continue

        # Backup trace
        output.append(f"\nBackup with value={value:.6f}:")
        obs_var = config.obs_var

        for parent, action in reversed(search_path):
            child = parent.children[action]
            old_mu = child.mu
            old_sigma_sq = child.sigma_sq

            # Update child
            child.update(value, obs_var, config.min_variance)

            output.append(f"  action {action}: child.mu: {old_mu:.6f} -> {child.mu:.6f}")
            output.append(f"              child.sigma_sq: {old_sigma_sq:.6f} -> {child.sigma_sq:.6f}")

            # Aggregate parent
            old_agg_mu = parent.agg_mu
            old_agg_sigma_sq = parent.agg_sigma_sq
            parent.aggregate_children(config.prune_threshold)

            output.append(f"              parent.agg_mu: {old_agg_mu:.6f} -> {parent.agg_mu:.6f}")
            output.append(f"              parent.agg_sigma_sq: {old_agg_sigma_sq:.6f} -> {parent.agg_sigma_sq:.6f}")

            # Prepare for next level
            value = parent.agg_mu
            obs_var = parent.agg_sigma_sq
            output.append(f"              next value={value:.6f}, obs_var={obs_var:.6f}")

    # Show state after 5 simulations
    output.append("\n=== STATE AFTER 5 SIMULATIONS ===")
    output.append("Child beliefs:")
    for a in sorted(root.children.keys()):
        c = root.children[a]
        output.append(f"  action {a}: child.mu={c.mu:.6f}, sigma_sq={c.sigma_sq:.8f}, prec={c.precision():.0f}")
        output.append(f"           (parent perspective: -child.mu = {-c.mu:.6f})")

    # Run more simulations
    output.append("\n=== RUNNING 95 MORE SIMULATIONS ===")
    for sim in range(95):
        leaf_node, search_path, leaf_state, is_terminal = ttts._select_to_leaf(root, state)
        if is_terminal:
            value = game.terminal_reward(leaf_state)
        elif not leaf_node.expanded():
            values = ttts._batch_expand_leaves([leaf_node], [leaf_state], model, device)
            value = values[0]
        else:
            continue
        ttts._backup(search_path, value)

    # Show final state
    output.append("\n=== FINAL STATE AFTER 100 SIMULATIONS ===")
    output.append("Child beliefs:")
    for a in sorted(root.children.keys()):
        c = root.children[a]
        output.append(f"  action {a}: child.mu={c.mu:.6f}, sigma_sq={c.sigma_sq:.8f}, prec={c.precision():.0f}")

    output.append("\n=== FINAL POLICY ===")
    final_policy = trace_policy_extraction(root, game, output)

    # Compare to ground truth
    output.append("\n=== COMPARISON TO GROUND TRUTH ===")
    from nanozero.mcts import BatchedMCTS
    from nanozero.config import MCTSConfig

    puct_10k = BatchedMCTS(game, MCTSConfig(num_simulations=10000))
    gt_policy = puct_10k.search(state[np.newaxis, ...], model, num_simulations=10000, add_noise=False)[0]

    output.append(f"Ground truth (PUCT 10k): best action = {np.argmax(gt_policy)}")
    output.append(f"TTTS 100 sims: best action = {np.argmax(final_policy)}")
    output.append(f"Match: {np.argmax(gt_policy) == np.argmax(final_policy)}")

    # Save to file
    with open('/tmp/trace_single_sim.txt', 'w') as f:
        f.write('\n'.join(output))

    print('\n'.join(output))
    print("\nResults saved to /tmp/trace_single_sim.txt")


if __name__ == '__main__':
    main()
