"""
Tests for common utilities.
"""
import numpy as np
import pytest
import torch
import tempfile
import os

from nanozero.common import (
    get_device,
    set_seed,
    print0,
    AverageMeter,
    save_checkpoint,
    load_checkpoint,
    sample_action,
)


class TestGetDevice:
    """Tests for get_device function."""

    def test_returns_device(self):
        """get_device returns a torch.device."""
        device = get_device()
        assert isinstance(device, torch.device)

    def test_returns_valid_device_type(self):
        """get_device returns cuda, mps, or cpu."""
        device = get_device()
        assert device.type in ['cuda', 'mps', 'cpu']


class TestSetSeed:
    """Tests for set_seed function."""

    def test_reproducible_numpy(self):
        """Setting seed makes numpy random reproducible."""
        set_seed(42)
        a = np.random.rand(10)

        set_seed(42)
        b = np.random.rand(10)

        np.testing.assert_array_equal(a, b)

    def test_reproducible_torch(self):
        """Setting seed makes torch random reproducible."""
        set_seed(42)
        a = torch.rand(10)

        set_seed(42)
        b = torch.rand(10)

        torch.testing.assert_close(a, b)

    def test_different_seeds_different_results(self):
        """Different seeds produce different results."""
        set_seed(42)
        a = np.random.rand(10)

        set_seed(123)
        b = np.random.rand(10)

        assert not np.array_equal(a, b)


class TestPrint0:
    """Tests for print0 function."""

    def test_print0_works(self, capsys):
        """print0 outputs to stdout."""
        print0("Hello, world!")
        captured = capsys.readouterr()
        assert "Hello, world!" in captured.out


class TestAverageMeter:
    """Tests for AverageMeter class."""

    def test_initialization(self):
        """AverageMeter initializes to zero."""
        meter = AverageMeter()
        assert meter.val == 0.0
        assert meter.sum == 0.0
        assert meter.count == 0
        assert meter.avg == 0.0

    def test_update_single(self):
        """Single update tracks value correctly."""
        meter = AverageMeter()
        meter.update(5.0)

        assert meter.val == 5.0
        assert meter.sum == 5.0
        assert meter.count == 1
        assert meter.avg == 5.0

    def test_update_multiple(self):
        """Multiple updates compute average correctly."""
        meter = AverageMeter()
        meter.update(2.0)
        meter.update(4.0)
        meter.update(6.0)

        assert meter.val == 6.0  # Last value
        assert meter.sum == 12.0
        assert meter.count == 3
        assert meter.avg == 4.0

    def test_update_with_count(self):
        """Update with count > 1 works correctly."""
        meter = AverageMeter()
        meter.update(5.0, n=10)  # Average of 5.0 over 10 samples

        assert meter.sum == 50.0
        assert meter.count == 10
        assert meter.avg == 5.0

    def test_reset(self):
        """Reset clears all values."""
        meter = AverageMeter()
        meter.update(10.0)
        meter.update(20.0)

        meter.reset()

        assert meter.val == 0.0
        assert meter.sum == 0.0
        assert meter.count == 0
        assert meter.avg == 0.0


class TestSampleAction:
    """Tests for sample_action function."""

    def test_greedy_selection(self):
        """Temperature 0 always returns argmax."""
        probs = np.array([0.1, 0.2, 0.7])

        for _ in range(10):
            action = sample_action(probs, temperature=0)
            assert action == 2

    def test_greedy_with_ties(self):
        """Temperature 0 with ties returns first max."""
        probs = np.array([0.5, 0.5, 0.0])
        action = sample_action(probs, temperature=0)
        assert action == 0  # First of the tied maxes

    def test_sampling_temperature_1(self):
        """Temperature 1 samples from distribution."""
        np.random.seed(42)
        probs = np.array([0.1, 0.3, 0.6])

        counts = {0: 0, 1: 0, 2: 0}
        for _ in range(1000):
            action = sample_action(probs, temperature=1.0)
            counts[action] += 1

        # Action 2 should be most common, action 0 least
        assert counts[2] > counts[1] > counts[0]

    def test_high_temperature_more_uniform(self):
        """High temperature makes distribution more uniform."""
        np.random.seed(42)
        probs = np.array([0.05, 0.05, 0.9])

        counts = {0: 0, 1: 0, 2: 0}
        for _ in range(1000):
            action = sample_action(probs, temperature=2.0)
            counts[action] += 1

        # With high temperature, distribution is more uniform
        # All actions should get significant counts
        assert counts[0] > 50
        assert counts[1] > 50
        assert counts[2] > counts[0]  # But 2 still more likely

    def test_low_temperature_more_peaked(self):
        """Low temperature makes distribution more peaked."""
        np.random.seed(42)
        probs = np.array([0.3, 0.3, 0.4])

        counts = {0: 0, 1: 0, 2: 0}
        for _ in range(1000):
            action = sample_action(probs, temperature=0.1)
            counts[action] += 1

        # With low temperature, most samples should be action 2
        assert counts[2] > 800  # At least 80%
        assert counts[2] > counts[0]
        assert counts[2] > counts[1]

    def test_returns_valid_index(self):
        """Always returns valid action index."""
        probs = np.array([0.25, 0.25, 0.25, 0.25])

        for _ in range(100):
            action = sample_action(probs, temperature=1.0)
            assert 0 <= action < len(probs)


class TestSaveLoadCheckpoint:
    """Tests for checkpoint save/load functions."""

    def test_save_and_load_model(self):
        """Can save and load model state."""
        # Create a simple model
        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())

        # Save initial weights
        initial_weight = model.weight.data.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")

            # Save checkpoint
            save_checkpoint(model, optimizer, iteration=42, path=path)

            # Modify weights
            model.weight.data.fill_(0)
            assert not torch.equal(model.weight.data, initial_weight)

            # Load checkpoint
            iteration = load_checkpoint(path, model, optimizer)

            # Verify
            assert iteration == 42
            torch.testing.assert_close(model.weight.data, initial_weight)

    def test_load_model_only(self):
        """Can load just the model without optimizer."""
        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")

            save_checkpoint(model, optimizer, iteration=10, path=path)

            # Create new model and load
            new_model = torch.nn.Linear(10, 5)
            iteration = load_checkpoint(path, new_model)  # No optimizer

            assert iteration == 10

    def test_save_with_scaler(self):
        """Can save and load with gradient scaler."""
        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())
        scaler = torch.amp.GradScaler('cuda', enabled=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")

            save_checkpoint(model, optimizer, iteration=5, path=path, scaler=scaler)

            # Load with scaler
            new_scaler = torch.amp.GradScaler('cuda', enabled=False)
            load_checkpoint(path, model, optimizer, scaler=new_scaler)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
