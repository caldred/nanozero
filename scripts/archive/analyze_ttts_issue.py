"""
scripts/analyze_ttts_issue.py - Analyze TTTS vs PUCT discrepancy

Saves results to /tmp/ttts_analysis.txt
"""
import numpy as np
import torch
from nanozero.game import get_game
from nanozero.model import AlphaZeroTransformer
from nanozero.mcts import BatchedMCTS
from nanozero.bayesian_mcts import BayesianMCTS
from nanozero.config import get_model_config, MCTSConfig, BayesianMCTSConfig
from nanozero.common import get_device, load_checkpoint

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
    output.append("TTTS vs PUCT Analysis")
    output.append("=" * 60)
    output.append(f"\nPosition:\n{game.display(state)}")

    # Ground truth
    output.append("\n--- Ground Truth (PUCT 10k) ---")
    puct_10k = BatchedMCTS(game, MCTSConfig(num_simulations=10000))
    gt_policy = puct_10k.search(state[np.newaxis, ...], model, num_simulations=10000, add_noise=False)[0]
    output.append(f"Best: action {np.argmax(gt_policy)} ({gt_policy[np.argmax(gt_policy)]:.1%})")
    output.append(f"Policy: {[f'{p:.3f}' for p in gt_policy]}")

    # Compare at various sim counts
    output.append("\n--- Comparison at various sim counts ---")
    for sims in [100, 200, 500, 1000, 2000]:
        puct = BatchedMCTS(game, MCTSConfig(num_simulations=sims))
        ttts = BayesianMCTS(game, BayesianMCTSConfig(num_simulations=sims, sigma_0=0.5, obs_var=0.25))

        puct_pol = puct.search(state[np.newaxis, ...], model, num_simulations=sims, add_noise=False)[0]
        ttts_pol = ttts.search(state[np.newaxis, ...], model)[0]

        output.append(f"\n{sims} sims:")
        output.append(f"  PUCT: best={np.argmax(puct_pol)}, policy={[f'{p:.2f}' for p in puct_pol]}")
        output.append(f"  TTTS: best={np.argmax(ttts_pol)}, policy={[f'{p:.2f}' for p in ttts_pol]}")

    # Detailed TTTS analysis at 1000 sims
    output.append("\n--- TTTS internals at 1000 sims ---")
    ttts = BayesianMCTS(game, BayesianMCTSConfig(num_simulations=1000, sigma_0=0.5, obs_var=0.25))
    roots, _ = ttts._batch_expand_roots(state[np.newaxis, ...], model, device)
    root = roots[0]

    # Track selections
    selection_counts = {a: 0 for a in range(7)}
    for _ in range(1000):
        leaf_node, search_path, leaf_state, is_terminal = ttts._select_to_leaf(root, state)
        if search_path:
            selection_counts[search_path[0][1]] += 1
        if is_terminal:
            value = game.terminal_reward(leaf_state)
        elif not leaf_node.expanded():
            values = ttts._batch_expand_leaves([leaf_node], [leaf_state], model, device)
            value = values[0]
        else:
            continue
        ttts._backup(search_path, value)

    output.append(f"Selection counts: {[selection_counts[a] for a in range(7)]}")
    output.append(f"Child beliefs (parent perspective):")
    for a in range(7):
        c = root.children[a]
        output.append(f"  {a}: mu={-c.mu:.3f}, sigma_sq={c.sigma_sq:.4f}, prec={c.precision():.0f}")

    # Save to file
    with open('/tmp/ttts_analysis.txt', 'w') as f:
        f.write('\n'.join(output))

    print('\n'.join(output))
    print("\nResults saved to /tmp/ttts_analysis.txt")

if __name__ == '__main__':
    main()
