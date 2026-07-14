# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for distance calculation.

These tests verify that the Python implementation matches the Go implementation
in gonka/decentralized-api/internal/validation/inference_validation.go.

The distance metric uses: |a - b| / (eps + |a| + |b|) / 2
Aggregated across all positions and normalized.
"""

import pytest
import math

from vllm.validation_logic import (
    position_distance,
    compute_distance,
    DEFAULT_FRAUD_DISTANCE_THRESHOLD,
)


class TestPositionDistance:
    """Test single position distance calculation."""

    def test_identical_logprobs_zero_distance(self):
        """Identical logprobs should have zero distance."""
        logprobs = {"100": -0.5, "200": -1.2, "300": -2.1}
        dist = position_distance(logprobs, logprobs)
        assert dist == 0.0

    def test_different_logprobs_nonzero_distance(self):
        """Different logprobs should have non-zero distance."""
        inf_lp = {"100": -0.5, "200": -1.2}
        val_lp = {"100": -0.8, "200": -1.5}
        
        dist = position_distance(inf_lp, val_lp)
        assert 0.0 < dist < 1.0
        
    def test_symmetric_distance(self):
        """Distance should be symmetric: d(a,b) == d(b,a)."""
        inf_lp = {"100": -0.5, "200": -1.2, "300": -2.5}
        val_lp = {"100": -0.7, "200": -1.0, "300": -2.8}
        
        dist1 = position_distance(inf_lp, val_lp)
        dist2 = position_distance(val_lp, inf_lp)
        
        # Should be equal (symmetric)
        assert abs(dist1 - dist2) < 1e-10

    def test_missing_token_estimation(self):
        """Test distance when validator is missing a token."""
        inf_lp = {"100": -0.5, "200": -1.2, "300": -2.1}
        val_lp = {"100": -0.5, "200": -1.2}  # Missing 300
        
        dist = position_distance(inf_lp, val_lp)
        
        # Should have non-zero distance due to estimated missing token
        assert dist > 0.0

    def test_empty_logprobs(self):
        """Empty logprobs should return zero distance."""
        assert position_distance({}, {}) == 0.0
        assert position_distance({"100": -0.5}, {}) == 0.0
        assert position_distance({}, {"100": -0.5}) == 0.0

    def test_single_token(self):
        """Test with single token."""
        inf_lp = {"100": -0.5}
        val_lp = {"100": -0.5}
        assert position_distance(inf_lp, val_lp) == 0.0
        
        val_lp = {"100": -1.0}
        assert position_distance(inf_lp, val_lp) > 0.0

    def test_distance_formula_correctness(self):
        """Verify the distance formula matches Go implementation."""
        # Test case with known values
        inf_lp = {"100": -1.0, "200": -2.0}
        val_lp = {"100": -1.5, "200": -2.5}
        
        dist = position_distance(inf_lp, val_lp)
        
        # Manual calculation:
        # Token 100: |(-1.0) - (-1.5)| / (1e-10 + |-1.0| + |-1.5|) / 2
        #          = 0.5 / (1e-10 + 1.0 + 1.5) / 2
        #          = 0.5 / 2.5 / 2 = 0.1
        # Token 200: |(-2.0) - (-2.5)| / (1e-10 + |-2.0| + |-2.5|) / 2
        #          = 0.5 / (1e-10 + 2.0 + 2.5) / 2
        #          = 0.5 / 4.5 / 2 ≈ 0.0556
        # Total ≈ 0.1 + 0.0556 ≈ 0.1556
        
        expected = 0.5 / 2.5 / 2 + 0.5 / 4.5 / 2
        assert abs(dist - expected) < 1e-6


class TestComputeDistance:
    """Test full sequence distance calculation."""

    def test_identical_sequences_zero_distance(self):
        """Identical sequences should have low distance."""
        logprobs = [
            {"100": -0.5, "200": -1.2},
            {"300": -0.8, "400": -2.1},
            {"500": -0.3, "600": -1.5},
        ]
        
        dist = compute_distance(logprobs, logprobs)
        
        # Distance should be close to normalization baseline
        # (total_dist=0 + 1.0) / (max(100, 3) * 2 + 1.0)
        expected_baseline = 1.0 / (100 * 2 + 1.0)
        assert abs(dist - expected_baseline) < 1e-6

    def test_different_sequences_nonzero_distance(self):
        """Different sequences should have higher distance."""
        inf_logprobs = [
            {"100": -0.5, "200": -1.2},
            {"300": -0.8, "400": -2.1},
        ]
        val_logprobs = [
            {"100": -0.8, "200": -1.5},
            {"300": -1.2, "400": -2.5},
        ]
        
        dist = compute_distance(inf_logprobs, val_logprobs)
        
        # Should be higher than baseline
        baseline = 1.0 / (100 * 2 + 1.0)
        assert dist > baseline

    def test_length_mismatch_max_distance(self):
        """Mismatched sequence lengths should return max fraud distance."""
        inf_logprobs = [{"100": -0.5}, {"200": -1.0}]
        val_logprobs = [{"100": -0.5}]
        
        dist = compute_distance(inf_logprobs, val_logprobs)
        assert dist == 10.0

    def test_empty_sequences(self):
        """Empty sequences should return zero distance."""
        assert compute_distance([], []) == 0.0

    def test_normalization(self):
        """Test normalization matches Go implementation."""
        # With 150 positions and top-5 logprobs
        n_positions = 150
        n_tokens = 5
        
        # Create identical sequences
        logprobs = [
            {str(i): -float(i) for i in range(100, 100 + n_tokens)}
            for _ in range(n_positions)
        ]
        
        dist = compute_distance(logprobs, logprobs)
        
        # Normalization: (0 + 1.0) / (max(100, 150) * 5 + 1.0)
        expected = 1.0 / (150 * 5 + 1.0)
        assert abs(dist - expected) < 1e-6

    def test_fraud_threshold(self):
        """Test that significant differences exceed fraud threshold."""
        # Simulate very different distributions (wrong model)
        inf_logprobs = [
            {"100": -0.1, "200": -5.0, "300": -10.0}  # Token 100 dominates
            for _ in range(100)
        ]
        val_logprobs = [
            {"100": -5.0, "200": -0.1, "300": -10.0}  # Token 200 dominates
            for _ in range(100)
        ]
        
        dist = compute_distance(inf_logprobs, val_logprobs)
        
        # Should exceed fraud threshold
        assert dist > DEFAULT_FRAUD_DISTANCE_THRESHOLD


class TestGoParity:
    """
    Tests to verify parity with Go implementation.
    
    These use specific test cases with known expected values from
    the Go implementation in inference_validation.go.
    """

    def test_position_distance_go_example_1(self):
        """Test case from Go implementation."""
        # Example logprobs
        inf_lp = {
            "16": -0.0008104139124043286,
            "15": -7.125810623168945,
            "785": -13.875810623168945,
        }
        val_lp = {
            "16": -0.0008104139124043286,
            "15": -7.125810623168945,
            "785": -13.875810623168945,
        }
        
        dist = position_distance(inf_lp, val_lp)
        assert dist == 0.0

    def test_honest_execution_passes_threshold(self):
        """Honest execution (same model) should pass threshold."""
        # Simulated honest logprobs with small FP variations
        inf_logprobs = []
        val_logprobs = []
        
        for i in range(100):
            # Small variations due to numerical precision
            inf = {
                "100": -0.5 + i * 0.001,
                "200": -1.2 - i * 0.0005,
                "300": -2.1 + i * 0.0008,
            }
            val = {
                "100": -0.5 + i * 0.001 + 0.0001,  # Tiny variation
                "200": -1.2 - i * 0.0005 - 0.0002,
                "300": -2.1 + i * 0.0008 + 0.0001,
            }
            inf_logprobs.append(inf)
            val_logprobs.append(val)
        
        dist = compute_distance(inf_logprobs, val_logprobs)
        
        # Should be below fraud threshold
        assert dist < DEFAULT_FRAUD_DISTANCE_THRESHOLD, \
            f"Honest execution distance {dist} exceeds threshold"


class TestEdgeCases:
    """Test edge cases and corner cases."""

    def test_very_negative_logprobs(self):
        """Test with very negative logprobs (rare tokens)."""
        inf_lp = {"100": -0.1, "999": -50.0}
        val_lp = {"100": -0.1, "999": -50.0}
        
        dist = position_distance(inf_lp, val_lp)
        assert dist == 0.0

    def test_near_zero_logprobs(self):
        """Test with logprobs near zero (dominant token)."""
        inf_lp = {"100": -0.0001, "200": -10.0}
        val_lp = {"100": -0.0002, "200": -10.0}
        
        dist = position_distance(inf_lp, val_lp)
        assert dist > 0.0  # Should be small but non-zero

    def test_large_vocabulary(self):
        """Test with large number of top-k tokens."""
        n_tokens = 50
        
        inf_lp = {str(i): -float(i) / 10 for i in range(100, 100 + n_tokens)}
        val_lp = {str(i): -float(i) / 10 for i in range(100, 100 + n_tokens)}
        
        dist = position_distance(inf_lp, val_lp)
        assert dist == 0.0
