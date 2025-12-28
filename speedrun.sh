#!/bin/bash
# NanoZero Speedrun: Train TicTacToe in ~5 minutes
set -e

echo "==================================="
echo "  NanoZero Speedrun - TicTacToe"
echo "==================================="

# Setup environment
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found!"
    exit 1
fi

# Create virtual environment if needed
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q torch numpy

# Train
echo ""
echo "Starting training..."
echo "Game: TicTacToe"
echo "Config: n_layer=2, 50 iterations, 100 games/iter"
echo ""

python -m scripts.train \
    --game=tictactoe \
    --n_layer=2 \
    --num_iterations=50 \
    --games_per_iteration=100 \
    --training_steps=50 \
    --mcts_simulations=25 \
    --batch_size=64 \
    --eval_interval=10 \
    --checkpoint_interval=25

# Evaluate
echo ""
echo "Final evaluation..."
python -m scripts.eval \
    --game=tictactoe \
    --checkpoint=checkpoints/tictactoe_final.pt \
    --n_layer=2 \
    --num_games=50

echo ""
echo "==================================="
echo "  Training complete!"
echo "  Play: python -m scripts.play --game=tictactoe --checkpoint=checkpoints/tictactoe_final.pt --n_layer=2"
echo "==================================="
