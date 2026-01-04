"""
scripts/trace_perspectives.py - Trace value perspectives through backup

Verify signs are correct at each level of the tree.
"""
import numpy as np
import torch
from nanozero.game import get_game
from nanozero.model import AlphaZeroTransformer
from nanozero.config import get_model_config, BayesianMCTSConfig
from nanozero.common import get_device, load_checkpoint
from nanozero.bayesian_mcts import BayesianMCTS


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

    root_player = game.current_player(state)
    print(f"Root player: {root_player} ({'X' if root_player == 1 else 'O'})")
    print(f"Position:\n{game.display(state)}")

    # Manual trace: get NN values at each depth for action 1
    print("\n=== NN VALUES AT EACH DEPTH FOR ACTION 1 ===")

    s = state.copy()
    for depth in range(1, 5):
        # Play action 1 first, then random legal moves
        if depth == 1:
            s = game.next_state(state, 1)  # Action 1
        else:
            # Pick first legal action
            legal = game.legal_actions(s)
            if legal and not game.is_terminal(s):
                s = game.next_state(s, legal[0])
            else:
                break

        if game.is_terminal(s):
            val = game.terminal_reward(s)
            perspective = "current player at terminal"
        else:
            canonical = game.canonical_state(s)
            st = game.to_tensor(canonical).unsqueeze(0).to(device)
            am = torch.from_numpy(game.legal_actions_mask(s)).unsqueeze(0).float().to(device)
            with torch.no_grad():
                _, v = model.predict(st, am)
            val = v.cpu().item()
            perspective = f"player {game.current_player(s)}"

        # Determine who is playing at this depth
        # Root is player X (-1 based on output), depth 1 is opponent, etc.
        if depth % 2 == 1:
            who = "OPPONENT (of root)"
        else:
            who = "ROOT's ally"

        print(f"Depth {depth}: NN value = {val:.4f} from {perspective}'s POV")
        print(f"         At this depth, it's {who}'s turn")
        print(f"         For root: {'good' if (depth % 2 == 0 and val > 0) or (depth % 2 == 1 and val < 0) else 'bad'}")

    # Now trace what TTTS believes
    print("\n=== TTTS BELIEFS ===")

    config = BayesianMCTSConfig(num_simulations=200, sigma_0=0.5, obs_var=0.25)
    ttts = BayesianMCTS(game, config)

    np.random.seed(42)
    roots, _ = ttts._batch_expand_roots(state[np.newaxis, ...], model, device)
    root = roots[0]

    # Run search
    for sim in range(200):
        leaf_node, search_path, leaf_state, is_terminal = ttts._select_to_leaf(root, state)
        if is_terminal:
            value = game.terminal_reward(leaf_state)
        elif not leaf_node.expanded():
            values = ttts._batch_expand_leaves([leaf_node], [leaf_state], model, device)
            value = values[0]
        else:
            continue
        ttts._backup(search_path, value)

    print("\nFinal child beliefs at root:")
    print("(child.mu is stored from CHILD's perspective, i.e., opponent of root)")
    for a in sorted(root.children.keys()):
        c = root.children[a]
        print(f"  Action {a}: child.mu = {c.mu:.4f} (opponent's POV)")
        print(f"             From ROOT's POV: {-c.mu:.4f}")
        if -c.mu > 0:
            print(f"             Interpretation: ROOT is WINNING")
        else:
            print(f"             Interpretation: ROOT is LOSING")

    print("\n=== VERIFICATION ===")
    # Ground truth
    from nanozero.mcts import BatchedMCTS
    from nanozero.config import MCTSConfig

    puct = BatchedMCTS(game, MCTSConfig(num_simulations=5000))
    gt_policy = puct.search(state[np.newaxis, ...], model, num_simulations=5000, add_noise=False)[0]

    print(f"\nPUCT 5k says best action is: {np.argmax(gt_policy)}")
    print(f"TTTS says best action is: {np.argmax([-c.mu for c in [root.children[a] for a in sorted(root.children.keys())]])}")

    print(f"\n(Note: if PUCT says action X is best, root should be WINNING after action X)")


if __name__ == '__main__':
    main()
