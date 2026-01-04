"""
Tests for replay buffer.
"""
import numpy as np
import pytest
from nanozero.replay import ReplayBuffer


class TestReplayBuffer:
    """Tests for ReplayBuffer class."""

    def test_initialization(self):
        """Buffer initializes with correct capacity."""
        buffer = ReplayBuffer(100)
        assert buffer.capacity == 100
        assert len(buffer) == 0

    def test_push_single_example(self):
        """Can push a single example."""
        buffer = ReplayBuffer(100)
        state = np.zeros((3, 3), dtype=np.float32)
        policy = np.ones(9) / 9
        value = 0.5

        buffer.push(state, policy, value)

        assert len(buffer) == 1

    def test_push_multiple_examples(self):
        """Can push multiple examples."""
        buffer = ReplayBuffer(100)

        for i in range(10):
            state = np.full((3, 3), i, dtype=np.float32)
            policy = np.ones(9) / 9
            value = float(i) / 10
            buffer.push(state, policy, value)

        assert len(buffer) == 10

    def test_extend(self):
        """Can extend buffer with list of examples."""
        buffer = ReplayBuffer(100)

        examples = []
        for i in range(5):
            state = np.full((3, 3), i, dtype=np.float32)
            policy = np.ones(9) / 9
            value = float(i)
            examples.append((state, policy, value))

        buffer.extend(examples)
        assert len(buffer) == 5

    def test_circular_overflow(self):
        """Buffer overwrites oldest when full."""
        buffer = ReplayBuffer(5)

        # Fill buffer
        for i in range(5):
            state = np.full((3, 3), i, dtype=np.float32)
            policy = np.ones(9) / 9
            buffer.push(state, policy, float(i))

        assert len(buffer) == 5

        # Push one more - should overwrite index 0
        state = np.full((3, 3), 99, dtype=np.float32)
        policy = np.ones(9) / 9
        buffer.push(state, policy, 99.0)

        assert len(buffer) == 5  # Still 5

        # Verify the old value at index 0 was overwritten
        # Sample all to check
        states, policies, values = buffer.sample(5)
        assert 99.0 in values  # New value should be present
        assert 0.0 not in values  # Old value should be gone

    def test_sample_returns_correct_shapes(self):
        """Sample returns arrays with correct shapes."""
        buffer = ReplayBuffer(100)

        for i in range(20):
            state = np.zeros((3, 3), dtype=np.float32)
            policy = np.ones(9) / 9
            buffer.push(state, policy, 0.0)

        states, policies, values = buffer.sample(8)

        assert states.shape == (8, 3, 3)
        assert policies.shape == (8, 9)
        assert values.shape == (8,)
        assert values.dtype == np.float32

    def test_sample_respects_batch_size(self):
        """Sample returns at most batch_size examples."""
        buffer = ReplayBuffer(100)

        for i in range(50):
            state = np.zeros((3, 3), dtype=np.float32)
            policy = np.ones(9) / 9
            buffer.push(state, policy, 0.0)

        states, policies, values = buffer.sample(10)
        assert len(states) == 10

    def test_sample_caps_at_buffer_size(self):
        """Sample returns buffer size if batch_size is larger."""
        buffer = ReplayBuffer(100)

        for i in range(5):
            state = np.zeros((3, 3), dtype=np.float32)
            policy = np.ones(9) / 9
            buffer.push(state, policy, 0.0)

        states, policies, values = buffer.sample(20)
        assert len(states) == 5  # Only 5 examples in buffer

    def test_sample_preserves_data(self):
        """Sampled data matches what was pushed."""
        buffer = ReplayBuffer(10)

        # Push specific data
        state = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        policy = np.array([0.1, 0.2, 0.3, 0.05, 0.05, 0.1, 0.05, 0.05, 0.1])
        value = 0.75

        buffer.push(state, policy, value)

        # Sample it back
        states, policies, values = buffer.sample(1)

        np.testing.assert_array_equal(states[0], state)
        np.testing.assert_array_almost_equal(policies[0], policy)
        assert values[0] == pytest.approx(value)

    def test_push_copies_data(self):
        """Buffer stores copies, not references."""
        buffer = ReplayBuffer(10)

        state = np.zeros((3, 3), dtype=np.float32)
        policy = np.ones(9) / 9
        buffer.push(state, policy, 0.0)

        # Modify original
        state[0, 0] = 999
        policy[0] = 0.999

        # Sample and verify it's unchanged
        states, policies, _ = buffer.sample(1)
        assert states[0, 0, 0] == 0.0
        assert policies[0, 0] == pytest.approx(1/9)

    def test_clear(self):
        """Clear removes all examples."""
        buffer = ReplayBuffer(100)

        for i in range(20):
            state = np.zeros((3, 3), dtype=np.float32)
            policy = np.ones(9) / 9
            buffer.push(state, policy, 0.0)

        assert len(buffer) == 20

        buffer.clear()

        assert len(buffer) == 0
        assert buffer.position == 0

    def test_clear_and_reuse(self):
        """Buffer can be reused after clear."""
        buffer = ReplayBuffer(10)

        # Fill once
        for i in range(5):
            state = np.full((3, 3), i, dtype=np.float32)
            policy = np.ones(9) / 9
            buffer.push(state, policy, float(i))

        buffer.clear()

        # Fill again
        for i in range(3):
            state = np.full((3, 3), i + 10, dtype=np.float32)
            policy = np.ones(9) / 9
            buffer.push(state, policy, float(i + 10))

        assert len(buffer) == 3
        states, _, values = buffer.sample(3)
        assert all(v >= 10 for v in values)


class TestReplayBufferEdgeCases:
    """Edge case tests for ReplayBuffer."""

    def test_sample_from_single_example(self):
        """Can sample when only one example exists."""
        buffer = ReplayBuffer(100)
        state = np.zeros((3, 3), dtype=np.float32)
        policy = np.ones(9) / 9
        buffer.push(state, policy, 1.0)

        states, policies, values = buffer.sample(1)
        assert len(states) == 1

    def test_large_capacity(self):
        """Works with large capacity."""
        buffer = ReplayBuffer(100000)
        state = np.zeros((3, 3), dtype=np.float32)
        policy = np.ones(9) / 9

        for i in range(1000):
            buffer.push(state, policy, float(i))

        assert len(buffer) == 1000

    def test_different_state_shapes(self):
        """Works with different state shapes."""
        buffer = ReplayBuffer(10)

        # Connect4 shape
        state = np.zeros((6, 7), dtype=np.float32)
        policy = np.ones(7) / 7
        buffer.push(state, policy, 0.0)

        states, policies, _ = buffer.sample(1)
        assert states.shape == (1, 6, 7)
        assert policies.shape == (1, 7)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
