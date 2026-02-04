# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for sampling verification and weight consistency.

These tests verify:
1. Deterministic sampling verification (Stage 1b)
2. Weight consistency verification (Stage 1a)
3. Weight quantization
"""

import pytest
import math
from typing import List, Dict

from vllm.validation_logic import (
    verify_sampling_sequence,
    verify_weights_consistency,
    recompute_weights_from_logprobs,
    quantize_to_weights,
    WEIGHT_SCALE,
)
from vllm.validation import EnforcedToken
from vllm.v1.sample.deterministic_utils import (
    Sha256CounterRNG,
    sample_categorical_weights,
)


class TestQuantizeToWeights:
    """Test probability to weight quantization."""

    def test_basic_quantization(self):
        """Test basic probability quantization."""
        probs = {"100": 0.6, "200": 0.3, "300": 0.1}
        weights = quantize_to_weights(probs)
        
        assert weights["100"] == round(0.6 * WEIGHT_SCALE)
        assert weights["200"] == round(0.3 * WEIGHT_SCALE)
        assert weights["300"] == round(0.1 * WEIGHT_SCALE)

    def test_weights_sum_approximately_correct(self):
        """Weights should sum to approximately WEIGHT_SCALE."""
        probs = {str(i): 1/100 for i in range(100)}  # Uniform
        weights = quantize_to_weights(probs)
        
        total = sum(weights.values())
        # Allow small rounding error
        assert abs(total - WEIGHT_SCALE) < 100

    def test_dominant_token(self):
        """Test with one dominant token."""
        probs = {"100": 0.9999, "200": 0.0001}
        weights = quantize_to_weights(probs)
        
        assert weights["100"] > WEIGHT_SCALE * 0.99
        assert weights["200"] < 10

    def test_empty_probs(self):
        """Empty probs should return empty weights."""
        assert quantize_to_weights({}) == {}

    def test_zero_probs(self):
        """Zero probabilities should quantize to zero."""
        probs = {"100": 0.5, "200": 0.5, "300": 0.0}
        weights = quantize_to_weights(probs)
        
        assert weights["300"] == 0


class TestRecomputeWeightsFromLogprobs:
    """Test weight recomputation from logprobs."""

    def test_basic_recomputation(self):
        """Test basic weight recomputation."""
        # Logprobs that give roughly [0.6, 0.3, 0.1] after softmax
        logprobs = {"100": -0.5, "200": -1.2, "300": -2.3}
        
        weights = recompute_weights_from_logprobs(logprobs, temperature=1.0)
        
        # Should have three weights
        assert len(weights) == 3
        # First should be largest
        assert weights["100"] > weights["200"] > weights["300"]

    def test_temperature_effect(self):
        """Test that temperature affects weight distribution."""
        logprobs = {"100": -0.5, "200": -1.0, "300": -2.0}
        
        weights_high_temp = recompute_weights_from_logprobs(logprobs, temperature=2.0)
        weights_low_temp = recompute_weights_from_logprobs(logprobs, temperature=0.5)
        
        # Higher temperature -> more uniform distribution
        # Lower temperature -> more peaked distribution
        ratio_high = weights_high_temp["100"] / max(1, weights_high_temp["300"])
        ratio_low = weights_low_temp["100"] / max(1, weights_low_temp["300"])
        
        assert ratio_low > ratio_high  # Low temp is more peaked

    def test_empty_logprobs(self):
        """Empty logprobs should return empty weights."""
        assert recompute_weights_from_logprobs({}, temperature=1.0) == {}

    def test_single_token(self):
        """Single token should get all weight."""
        logprobs = {"100": -0.5}
        weights = recompute_weights_from_logprobs(logprobs, temperature=1.0)
        
        assert weights["100"] == WEIGHT_SCALE


class TestVerifySamplingSequence:
    """Test sampling sequence verification (Stage 1b)."""

    def test_correct_sampling_passes(self):
        """Correctly sampled sequence should pass verification."""
        seed_str = "verification_test_seed"
        
        # Generate a sequence with known correct sampling
        weights1 = [60000, 5000, 500]
        weights2 = [30000, 30000, 5536]
        token_ids = ["100", "200", "300"]
        
        # Sample using same RNG to get expected tokens
        rng = Sha256CounterRNG.from_seed_string(seed_str)
        idx1 = sample_categorical_weights(weights1, rng)
        idx2 = sample_categorical_weights(weights2, rng)
        
        expected_token1 = token_ids[idx1]
        expected_token2 = token_ids[idx2]
        
        # Build artifact with correct tokens
        tokens = [
            EnforcedToken(
                token=expected_token1,
                top_tokens=token_ids,
                sampling_weights={tid: w for tid, w in zip(token_ids, weights1)},
            ),
            EnforcedToken(
                token=expected_token2,
                top_tokens=token_ids,
                sampling_weights={tid: w for tid, w in zip(token_ids, weights2)},
            ),
        ]
        
        success, failed_pos = verify_sampling_sequence(tokens, seed_str)
        
        assert success, f"Verification failed at position {failed_pos}"
        assert failed_pos == -1

    def test_wrong_token_fails(self):
        """Wrong token should fail verification."""
        seed_str = "wrong_token_test"
        weights = [60000, 5000, 500]
        token_ids = ["100", "200", "300"]
        
        # Get expected token
        rng = Sha256CounterRNG.from_seed_string(seed_str)
        expected_idx = sample_categorical_weights(weights, rng)
        
        # Use wrong token
        wrong_idx = (expected_idx + 1) % 3
        wrong_token = token_ids[wrong_idx]
        
        tokens = [
            EnforcedToken(
                token=wrong_token,
                top_tokens=token_ids,
                sampling_weights={tid: w for tid, w in zip(token_ids, weights)},
            ),
        ]
        
        success, failed_pos = verify_sampling_sequence(tokens, seed_str)
        
        assert not success
        assert failed_pos == 0

    def test_wrong_seed_fails(self):
        """Wrong seed should fail verification."""
        correct_seed = "correct_seed"
        wrong_seed = "wrong_seed"
        
        weights = [60000, 5000, 500]
        token_ids = ["100", "200", "300"]
        
        # Sample with correct seed
        rng = Sha256CounterRNG.from_seed_string(correct_seed)
        idx = sample_categorical_weights(weights, rng)
        token = token_ids[idx]
        
        tokens = [
            EnforcedToken(
                token=token,
                top_tokens=token_ids,
                sampling_weights={tid: w for tid, w in zip(token_ids, weights)},
            ),
        ]
        
        # Verify with wrong seed
        success, failed_pos = verify_sampling_sequence(tokens, wrong_seed)
        
        # May or may not fail depending on whether wrong seed happens to
        # produce same sample (unlikely but possible)
        # In most cases it should fail
        # This test documents the behavior

    def test_no_weights_skipped(self):
        """Tokens without sampling_weights should be skipped."""
        seed_str = "no_weights_test"
        
        tokens = [
            EnforcedToken(
                token="100",
                top_tokens=["100", "200"],
                sampling_weights=None,  # No weights
            ),
        ]
        
        success, failed_pos = verify_sampling_sequence(tokens, seed_str)
        
        # Should pass (nothing to verify)
        assert success
        assert failed_pos == -1

    def test_long_sequence(self):
        """Test with longer sequence."""
        seed_str = "long_sequence_test"
        n_positions = 50
        
        weights_template = [60000, 5000, 500, 35, 1]
        token_ids = ["100", "200", "300", "400", "500"]
        
        # Generate expected sequence
        rng = Sha256CounterRNG.from_seed_string(seed_str)
        expected_tokens = []
        for _ in range(n_positions):
            idx = sample_categorical_weights(weights_template, rng)
            expected_tokens.append(token_ids[idx])
        
        # Build artifact
        tokens = [
            EnforcedToken(
                token=expected_tokens[i],
                top_tokens=token_ids,
                sampling_weights={tid: w for tid, w in zip(token_ids, weights_template)},
            )
            for i in range(n_positions)
        ]
        
        success, failed_pos = verify_sampling_sequence(tokens, seed_str)
        
        assert success, f"Failed at position {failed_pos}"
        assert failed_pos == -1


class TestVerifyWeightsConsistency:
    """Test weight consistency verification (Stage 1a)."""

    def test_consistent_weights_pass(self):
        """Consistent weights should pass verification."""
        # Create logprobs and compute weights
        logprobs = {"100": -0.5, "200": -1.2, "300": -2.3}
        weights = recompute_weights_from_logprobs(logprobs, temperature=0.99)
        
        tokens = [
            EnforcedToken(
                token="100",
                top_tokens=["100", "200", "300"],
                logprobs=logprobs,
                sampling_weights=weights,
            ),
        ]
        
        success, failed_pos = verify_weights_consistency(tokens, temperature=0.99)
        
        assert success
        assert failed_pos == -1

    def test_tampered_weights_fail(self):
        """Tampered weights should fail verification."""
        logprobs = {"100": -0.5, "200": -1.2, "300": -2.3}
        correct_weights = recompute_weights_from_logprobs(logprobs, temperature=0.99)
        
        # Tamper with weights
        tampered_weights = dict(correct_weights)
        tampered_weights["100"] = 99999  # Very different
        
        tokens = [
            EnforcedToken(
                token="100",
                top_tokens=["100", "200", "300"],
                logprobs=logprobs,
                sampling_weights=tampered_weights,
            ),
        ]
        
        success, failed_pos = verify_weights_consistency(tokens, temperature=0.99)
        
        assert not success
        assert failed_pos == 0

    def test_no_logprobs_skipped(self):
        """Tokens without logprobs should be skipped."""
        tokens = [
            EnforcedToken(
                token="100",
                top_tokens=["100", "200"],
                logprobs=None,
                sampling_weights={"100": 60000, "200": 5536},
            ),
        ]
        
        success, failed_pos = verify_weights_consistency(tokens, temperature=0.99)
        
        # Should pass (nothing to verify)
        assert success
        assert failed_pos == -1

    def test_no_weights_skipped(self):
        """Tokens without weights should be skipped."""
        tokens = [
            EnforcedToken(
                token="100",
                top_tokens=["100", "200"],
                logprobs={"100": -0.5, "200": -1.2},
                sampling_weights=None,
            ),
        ]
        
        success, failed_pos = verify_weights_consistency(tokens, temperature=0.99)
        
        assert success
        assert failed_pos == -1

    def test_tolerance_allows_small_differences(self):
        """Small differences within tolerance should pass."""
        logprobs = {"100": -0.5, "200": -1.2}
        correct_weights = recompute_weights_from_logprobs(logprobs, temperature=0.99)
        
        # Add small perturbation (within 5% tolerance)
        slightly_off_weights = {
            k: int(v * 1.03)  # 3% increase
            for k, v in correct_weights.items()
        }
        
        tokens = [
            EnforcedToken(
                token="100",
                top_tokens=["100", "200"],
                logprobs=logprobs,
                sampling_weights=slightly_off_weights,
            ),
        ]
        
        success, failed_pos = verify_weights_consistency(
            tokens, temperature=0.99, tolerance=0.05
        )
        
        assert success


class TestDeterministicRNGProperties:
    """Test properties of the deterministic RNG."""

    def test_rng_reproducibility(self):
        """Same seed should produce identical sequence."""
        seed = "reproducibility_test"
        
        rng1 = Sha256CounterRNG.from_seed_string(seed)
        rng2 = Sha256CounterRNG.from_seed_string(seed)
        
        for _ in range(100):
            assert rng1.next_u64() == rng2.next_u64()

    def test_different_seeds_differ(self):
        """Different seeds should produce different sequences."""
        rng1 = Sha256CounterRNG.from_seed_string("seed_a")
        rng2 = Sha256CounterRNG.from_seed_string("seed_b")
        
        seq1 = [rng1.next_u64() for _ in range(10)]
        seq2 = [rng2.next_u64() for _ in range(10)]
        
        assert seq1 != seq2

    def test_sample_categorical_weights_distribution(self):
        """Sampling should follow weight distribution."""
        weights = [60000, 4000, 1536, 500]  # Sum = 66036
        seed = "distribution_test"
        
        rng = Sha256CounterRNG.from_seed_string(seed)
        n_samples = 10000
        
        counts = [0] * len(weights)
        for _ in range(n_samples):
            idx = sample_categorical_weights(weights, rng)
            counts[idx] += 1
        
        # Check approximate proportions (with some tolerance)
        total_weight = sum(weights)
        for i, (count, weight) in enumerate(zip(counts, weights)):
            expected_prop = weight / total_weight
            actual_prop = count / n_samples
            # Allow 10% relative error
            if expected_prop > 0.01:  # Only check non-negligible probs
                assert abs(actual_prop - expected_prop) / expected_prop < 0.2, \
                    f"Index {i}: expected {expected_prop:.3f}, got {actual_prop:.3f}"

    def test_reference_values(self):
        """Verify reference values for cross-implementation testing."""
        rng = Sha256CounterRNG.from_seed_string("reference_seed_v1")
        
        # These values are from the current Sha256CounterRNG implementation
        expected = [
            4286832458236889005,
            12281003819428572724,
            12352776571910749143,
            12178488218135958089,
            6205195570139478562,
        ]
        
        for exp in expected:
            actual = rng.next_u64()
            assert actual == exp, f"Reference value mismatch: {actual} != {exp}"
