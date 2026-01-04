"""
scripts/compare_nn_values.py - Compare NN values received by PUCT vs TTTS

Tracks what values each algorithm receives from the NN for actions 1 and 3.
Saves results to /tmp/compare_nn_values.txt
"""
import math
import numpy as np
import torch
from collections import defaultdict
from nanozero.game import get_game
from nanozero.model import AlphaZeroTransformer
from nanozero.config import get_model_config, MCTSConfig, BayesianMCTSConfig
from nanozero.common import get_device, load_checkpoint
from nanozero.mcts import BatchedMCTS
from nanozero.bayesian_mcts import BayesianMCTS, BayesianNode


class TrackedBayesianMCTS(BayesianMCTS):
    """BayesianMCTS that tracks NN values received."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nn_calls = []  # List of (state_hash, action_path, value)
        self.action_values = defaultdict(list)  # action -> [values received]

    def _expand(self, node, state, model, device):
        value = super()._expand(node, state, model, device)
        return value

    def search_with_tracking(self, states, model, root_state, num_simulations=None):
        """Search while tracking which values go to which root actions."""
        if num_simulations is None:
            num_simulations = self.config.num_simulations

        device = next(model.parameters()).device

        # Expand root
        roots, root_values = self._batch_expand_roots(states, model, device)
        root = roots[0]

        self.nn_calls.append(('root', [], root_values[0]))

        # Track values by first action taken
        for sim in range(num_simulations):
            leaf_node, search_path, leaf_state, is_terminal = self._select_to_leaf(root, root_state)

            if search_path:
                first_action = search_path[0][1]
            else:
                first_action = None

            if is_terminal:
                value = self.game.terminal_reward(leaf_state)
                self.action_values[first_action].append(('terminal', value))
            elif not leaf_node.expanded():
                values = self._batch_expand_leaves([leaf_node], [leaf_state], model, device)
                value = values[0]
                depth = len(search_path)
                self.action_values[first_action].append((f'depth_{depth}', value))
            else:
                continue

            self._backup(search_path, value)

        return self._get_policy(root), root


class TrackedPUCT(BatchedMCTS):
    """BatchedMCTS that tracks NN values received."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_values = defaultdict(list)

    def search_with_tracking(self, states, model, num_simulations, add_noise=False):
        """Search while tracking values."""
        # We need to instrument the internal search
        # For now, let's just run search and then inspect the tree
        policies = self.search(states, model, num_simulations, add_noise)
        return policies


def get_puct_subtree_values(game, state, model, device, action, num_sims=1000):
    """Get all NN values in PUCT's subtree for a given action."""
    from nanozero.mcts import MCTSNode

    # Create root and expand
    puct = BatchedMCTS(game, MCTSConfig(num_simulations=num_sims))

    # We need to track the actual values - let's use transposition table
    if puct.tt:
        puct.tt.clear()

    # Run search
    policies = puct.search(state[np.newaxis, ...], model, num_simulations=num_sims, add_noise=False)

    # Now inspect the cache to see what values were computed
    # The TT stores (policy, value) for each state
    values_in_subtree = []

    # We need to traverse the tree to find values under action
    # This is tricky because PUCT doesn't expose its tree structure easily
    # Let's use a different approach - track during search

    return policies[0], None


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
    output.append("NN Value Comparison: PUCT vs TTTS")
    output.append("=" * 60)
    output.append(f"\nPosition:\n{game.display(state)}")

    # Get raw NN evaluation at root
    canonical = game.canonical_state(state)
    state_tensor = game.to_tensor(canonical).unsqueeze(0).to(device)
    action_mask = torch.from_numpy(game.legal_actions_mask(state)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        policy, value = model.predict(state_tensor, action_mask)
    root_policy = policy.cpu().numpy()[0]
    root_value = value.cpu().item()

    output.append(f"\nRoot NN evaluation:")
    output.append(f"  Value: {root_value:.4f}")
    output.append(f"  Policy: {[f'{p:.3f}' for p in root_policy]}")
    output.append(f"  Action 1 prior: {root_policy[1]:.4f}")
    output.append(f"  Action 3 prior: {root_policy[3]:.4f}")

    # Get NN evaluations for states after action 1 and action 3
    output.append("\n" + "=" * 60)
    output.append("NN VALUES AFTER EACH ROOT ACTION")
    output.append("=" * 60)

    for action in [1, 3]:
        next_state = game.next_state(state, action)
        canonical = game.canonical_state(next_state)
        state_tensor = game.to_tensor(canonical).unsqueeze(0).to(device)
        action_mask = torch.from_numpy(game.legal_actions_mask(next_state)).unsqueeze(0).float().to(device)

        with torch.no_grad():
            policy, value = model.predict(state_tensor, action_mask)
        child_value = value.cpu().item()
        child_policy = policy.cpu().numpy()[0]

        output.append(f"\nAfter action {action}:")
        output.append(f"  NN value (from next player's perspective): {child_value:.4f}")
        output.append(f"  Negated (from root player's perspective): {-child_value:.4f}")
        output.append(f"  Child policy: {[f'{p:.3f}' for p in child_policy]}")

    # Run TTTS with tracking
    output.append("\n" + "=" * 60)
    output.append("TTTS VALUE TRACKING (500 sims)")
    output.append("=" * 60)

    np.random.seed(42)
    config = BayesianMCTSConfig(num_simulations=500, sigma_0=0.5, obs_var=0.25)
    ttts = TrackedBayesianMCTS(game, config)
    ttts_policy, ttts_root = ttts.search_with_tracking(
        state[np.newaxis, ...], model, state, num_simulations=500
    )

    output.append(f"\nTTTS final policy: {[f'{p:.3f}' for p in ttts_policy]}")
    output.append(f"TTTS best action: {np.argmax(ttts_policy)}")

    for action in [1, 3]:
        values = ttts.action_values[action]
        output.append(f"\nAction {action} received {len(values)} values:")

        # Separate by type
        terminal_vals = [v for t, v in values if t == 'terminal']
        depth_vals = [(t, v) for t, v in values if t != 'terminal']

        if terminal_vals:
            output.append(f"  Terminal values: {len(terminal_vals)} total")
            output.append(f"    Mean: {np.mean(terminal_vals):.4f}")
            output.append(f"    Min/Max: {min(terminal_vals):.4f} / {max(terminal_vals):.4f}")

        if depth_vals:
            output.append(f"  NN expansion values: {len(depth_vals)} total")
            nn_vals = [v for _, v in depth_vals]
            output.append(f"    Mean: {np.mean(nn_vals):.4f}")
            output.append(f"    Std: {np.std(nn_vals):.4f}")
            output.append(f"    Min/Max: {min(nn_vals):.4f} / {max(nn_vals):.4f}")

            # Show by depth
            by_depth = defaultdict(list)
            for t, v in depth_vals:
                by_depth[t].append(v)
            for depth in sorted(by_depth.keys()):
                vals = by_depth[depth]
                output.append(f"    {depth}: n={len(vals)}, mean={np.mean(vals):.4f}")

    # Show final TTTS beliefs
    output.append("\n" + "=" * 60)
    output.append("TTTS FINAL BELIEFS")
    output.append("=" * 60)

    for action in sorted(ttts_root.children.keys()):
        c = ttts_root.children[action]
        output.append(f"Action {action}: child.mu={c.mu:.4f} (parent perspective: {-c.mu:.4f}), sigma_sq={c.sigma_sq:.6f}, precision={c.precision():.0f}")

    # Run PUCT for comparison
    output.append("\n" + "=" * 60)
    output.append("PUCT COMPARISON (500 sims)")
    output.append("=" * 60)

    puct = BatchedMCTS(game, MCTSConfig(num_simulations=500))
    puct_policy = puct.search(state[np.newaxis, ...], model, num_simulations=500, add_noise=False)[0]

    output.append(f"\nPUCT policy: {[f'{p:.3f}' for p in puct_policy]}")
    output.append(f"PUCT best action: {np.argmax(puct_policy)}")

    # Ground truth
    output.append("\n" + "=" * 60)
    output.append("GROUND TRUTH (PUCT 10k)")
    output.append("=" * 60)

    puct_10k = BatchedMCTS(game, MCTSConfig(num_simulations=10000))
    gt_policy = puct_10k.search(state[np.newaxis, ...], model, num_simulations=10000, add_noise=False)[0]

    output.append(f"\nGround truth policy: {[f'{p:.3f}' for p in gt_policy]}")
    output.append(f"Ground truth best action: {np.argmax(gt_policy)}")

    # Key insight: compare the Q-values PUCT computes vs what TTTS believes
    output.append("\n" + "=" * 60)
    output.append("KEY COMPARISON")
    output.append("=" * 60)

    # For TTTS, parent-perspective value is -child.mu
    ttts_q1 = -ttts_root.children[1].mu
    ttts_q3 = -ttts_root.children[3].mu

    output.append(f"\nTTTS believes (parent perspective):")
    output.append(f"  Action 1 value: {ttts_q1:.4f}")
    output.append(f"  Action 3 value: {ttts_q3:.4f}")
    output.append(f"  Difference (3-1): {ttts_q3 - ttts_q1:.4f}")

    # What values did they actually receive?
    vals_1 = [v for _, v in ttts.action_values[1]]
    vals_3 = [v for _, v in ttts.action_values[3]]

    if vals_1 and vals_3:
        output.append(f"\nActual NN values received:")
        output.append(f"  Action 1: mean={np.mean(vals_1):.4f}, n={len(vals_1)}")
        output.append(f"  Action 3: mean={np.mean(vals_3):.4f}, n={len(vals_3)}")

    # Save to file
    with open('/tmp/compare_nn_values.txt', 'w') as f:
        f.write('\n'.join(output))

    print('\n'.join(output))
    print("\nResults saved to /tmp/compare_nn_values.txt")


if __name__ == '__main__':
    main()
