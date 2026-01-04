"""
Test different obs_var and sigma_0 values for BayesianMCTS.

Theory: If these are too low, beliefs become overconfident, making
optimality weights too aggressive during backup.
"""
import numpy as np
import torch
from nanozero.game import get_game
from nanozero.model import AlphaZeroTransformer
from nanozero.bayesian_mcts import BayesianMCTS
from nanozero.config import get_model_config, BayesianMCTSConfig
from nanozero.common import get_device, load_checkpoint


def analyze_beliefs(game, model, config, state, n_sims=100):
    """Run MCTS and analyze resulting beliefs."""
    mcts = BayesianMCTS(game, config)
    policy = mcts.search(state[np.newaxis, ...], model, num_simulations=n_sims)[0]

    # Policy analysis
    legal = game.legal_actions(state)
    legal_policy = policy[legal]

    results = {
        'top_action': int(np.argmax(policy)),
        'top_prob': float(np.max(policy)),
        'entropy': float(-np.sum(legal_policy * np.log(legal_policy + 1e-8))),
        'policy': {a: float(policy[a]) for a in legal},
    }

    return results


def main():
    device = get_device()
    game = get_game('connect4', use_rust=False)
    print(f"Game backend: {game.backend}")

    model_config = get_model_config(game.config, n_layer=4)
    model = AlphaZeroTransformer(model_config).to(device)
    load_checkpoint('checkpoints/connect4_iter150.pt', model)
    model.eval()

    # Test position (midgame)
    state = game.initial_state()
    for m in [3, 4, 6, 2, 4, 4, 6, 1, 2, 6, 2]:
        state = game.next_state(state, m)

    print("\nTest position: midgame after [3, 4, 6, 2, 4, 4, 6, 1, 2, 6, 2]")

    # Parameter combinations to test
    # No weight blending - pure optimality weights
    param_sets = [
        # (sigma_0, obs_var, description)
        (0.5, 0.25, "default (sigma_0=0.5, obs_var=0.25)"),
        (1.0, 0.25, "higher sigma_0=1.0"),
        (0.5, 1.0, "higher obs_var=1.0"),
        (1.0, 1.0, "both high (sigma_0=1.0, obs_var=1.0)"),
        (2.0, 2.0, "very high (sigma_0=2.0, obs_var=2.0)"),
        (0.5, 0.5, "moderate (sigma_0=0.5, obs_var=0.5)"),
    ]

    n_sims = 100
    print(f"\nTesting with {n_sims} simulations, NO weight blending (optimality_weight=1.0)")
    print("=" * 80)

    for sigma_0, obs_var, desc in param_sets:
        config = BayesianMCTSConfig(
            num_simulations=n_sims,
            sigma_0=sigma_0,
            obs_var=obs_var,
            optimality_weight=1.0,  # Pure optimality weights
            adaptive_weight=False,
            prune_threshold=0.01,
        )

        np.random.seed(42)
        results = analyze_beliefs(game, model, config, state, n_sims)

        print(f"\n{desc}")
        print(f"  Top action: {results['top_action']} (prob={results['top_prob']:.3f})")
        print(f"  Policy entropy: {results['entropy']:.3f}")
        print(f"  Policy: {results['policy']}")

    # Now test at higher sims to see if it degrades
    print("\n" + "=" * 80)
    print("Testing best candidates at 200 sims")
    print("=" * 80)

    best_params = [
        (1.0, 1.0, "both=1.0"),
        (2.0, 2.0, "both=2.0"),
    ]

    for sigma_0, obs_var, desc in best_params:
        config = BayesianMCTSConfig(
            num_simulations=200,
            sigma_0=sigma_0,
            obs_var=obs_var,
            optimality_weight=1.0,
            adaptive_weight=False,
            prune_threshold=0.01,
        )

        np.random.seed(42)
        results = analyze_beliefs(game, model, config, state, n_sims=200)

        print(f"\n{desc} @ 200 sims")
        print(f"  Top action: {results['top_action']} (prob={results['top_prob']:.3f})")
        print(f"  Policy entropy: {results['entropy']:.3f}")
        print(f"  Policy: {results['policy']}")


if __name__ == '__main__':
    main()
