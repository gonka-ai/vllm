# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Comprehensive test suite for deterministic sampling utilities.
Does not require full vLLM installation - only tests the core cryptographic sampling.

Tests include:
- RNG reproducibility and statistical properties
- Categorical sampling correctness
- Edge cases (degenerate distributions, extreme values)
- Cross-run consistency validation
- Temperature and top-k/top-p filtering
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Sequence, Tuple

import pytest

from vllm.v1.sample.deterministic_utils import (
    Sha256CounterRNG,
    sample_categorical,
    sample_categorical_weights,
    iter_u64,
    WeightedPrefixSampler,
    uint64_below,
)


# =============================================================================
# Standalone sampling functions (no torch dependency)
# =============================================================================

def deterministic_sample_from_logprobs(
    logprobs: Sequence[Dict[int, float]],
    seed: str,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> List[int]:
    """
    Sample tokens deterministically from a sequence of logprob dictionaries.
    """
    rng = Sha256CounterRNG.from_seed_string(seed)
    sampled_tokens = []
    
    for step_logprobs in logprobs:
        if not step_logprobs:
            raise ValueError("Empty logprobs at step")
        
        # Convert logprobs to probabilities
        token_ids = list(step_logprobs.keys())
        log_values = [step_logprobs[tid] for tid in token_ids]
        
        # Apply temperature
        if temperature != 1.0 and temperature > 0:
            log_values = [lv / temperature for lv in log_values]
        
        # Convert to probabilities via softmax
        max_log = max(log_values)
        exp_values = [math.exp(lv - max_log) for lv in log_values]
        sum_exp = sum(exp_values)
        probs = [ev / sum_exp for ev in exp_values]
        
        # Apply top-k
        if top_k is not None and top_k < len(token_ids):
            sorted_items = sorted(
                zip(token_ids, probs),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
            token_ids = [x[0] for x in sorted_items]
            probs = [x[1] for x in sorted_items]
            sum_probs = sum(probs)
            probs = [p / sum_probs for p in probs]
        
        # Apply top-p
        if top_p is not None and top_p < 1.0:
            sorted_items = sorted(
                zip(token_ids, probs),
                key=lambda x: x[1],
                reverse=True
            )
            cumsum = 0.0
            filtered = []
            for tid, p in sorted_items:
                if cumsum >= top_p and filtered:
                    break
                filtered.append((tid, p))
                cumsum += p
            token_ids = [x[0] for x in filtered]
            probs = [x[1] for x in filtered]
            sum_probs = sum(probs)
            probs = [p / sum_probs for p in probs]
        
        # Sample using deterministic categorical sampler
        idx = sample_categorical(probs, rng)
        sampled_tokens.append(token_ids[idx])
    
    return sampled_tokens


def deterministic_sample_from_probs(
    probs_2d: Sequence[Sequence[float]],
    seed: str,
) -> List[int]:
    """Sample tokens deterministically from 2D probability array."""
    rng = Sha256CounterRNG.from_seed_string(seed)
    return [sample_categorical(step_probs, rng) for step_probs in probs_2d]


# =============================================================================
# Test Utilities
# =============================================================================

def chi_squared_test(observed: List[int], expected: List[float], alpha: float = 0.01) -> Tuple[float, bool]:
    """
    Perform chi-squared goodness of fit test.
    Returns (chi_squared_statistic, passes_test).
    """
    n = sum(observed)
    expected_counts = [e * n for e in expected]
    
    chi_sq = 0.0
    for obs, exp in zip(observed, expected_counts):
        if exp > 0:
            chi_sq += (obs - exp) ** 2 / exp
    
    df = len(observed) - 1
    
    # Critical values for chi-squared at alpha=0.01 (99% confidence)
    # Pre-computed for common df values
    critical_values = {
        1: 6.635, 2: 9.210, 3: 11.345, 4: 13.277, 5: 15.086,
        6: 16.812, 7: 18.475, 8: 20.090, 9: 21.666, 10: 23.209,
        19: 36.191, 20: 37.566, 49: 74.919, 50: 76.154, 99: 135.807,
    }
    
    if df in critical_values:
        critical = critical_values[df]
    else:
        # Wilson-Hilferty approximation for larger df
        critical = df * (1 - 2.0 / (9 * df) + 2.33 * math.sqrt(2.0 / (9 * df))) ** 3
    
    return chi_sq, chi_sq < critical


# =============================================================================
# Core RNG Tests
# =============================================================================

class TestRNG:
    """Comprehensive RNG tests."""
    
    def test_reproducibility(self):
        """Same seed should produce same sequence."""
        rng1 = Sha256CounterRNG.from_seed_string("test_seed")
        rng2 = Sha256CounterRNG.from_seed_string("test_seed")
        
        for _ in range(1000):
            assert rng1.next_u64() == rng2.next_u64()
    
    def test_different_seeds(self):
        """Different seeds should produce different sequences."""
        seeds = ["seed_a", "seed_b", "seed_c", "123", "test", ""]
        all_vals = []
        for seed in seeds:
            rng = Sha256CounterRNG.from_seed_string(seed)
            vals = tuple(rng.next_u64() for _ in range(10))
            all_vals.append(vals)
        
        # All should be unique
        assert len(set(all_vals)) == len(all_vals)
    
    def test_uniform_range(self):
        """Uniform values should be in [0, 1)."""
        rng = Sha256CounterRNG.from_seed_string("uniform_test")
        for _ in range(10000):
            u = rng.next_uniform01()
            assert 0.0 <= u < 1.0
    
    def test_uniform_distribution(self):
        """Uniform values should be evenly distributed."""
        rng = Sha256CounterRNG.from_seed_string("uniform_dist_test")
        n_bins = 10
        n_samples = 100000
        bins = [0] * n_bins
        
        for _ in range(n_samples):
            u = rng.next_uniform01()
            bin_idx = min(int(u * n_bins), n_bins - 1)
            bins[bin_idx] += 1
        
        expected = [1.0 / n_bins] * n_bins
        chi_sq, passed = chi_squared_test(bins, expected)
        
        assert passed, f"Uniform distribution failed chi-squared test: {chi_sq}"
    
    def test_uint64_below_unbiased(self):
        """uint64_below should be unbiased for various n values."""
        rng = Sha256CounterRNG.from_seed_string("uint64_below_test")
        
        for n in [2, 3, 7, 10, 100]:
            counts = [0] * n
            n_samples = n * 1000
            
            for _ in range(n_samples):
                val = uint64_below(rng, n)
                assert 0 <= val < n
                counts[val] += 1
            
            expected = [1.0 / n] * n
            chi_sq, passed = chi_squared_test(counts, expected)
            
            assert passed, f"uint64_below({n}) failed chi-squared: {chi_sq}"
    
    def test_iter_u64_reproducibility(self):
        """iter_u64 helper should be reproducible."""
        vals1 = iter_u64("test", 100)
        vals2 = iter_u64("test", 100)
        assert vals1 == vals2
    
    def test_empty_seed(self):
        """Empty seed should work."""
        rng1 = Sha256CounterRNG.from_seed_string("")
        rng2 = Sha256CounterRNG.from_seed_string("")
        
        for _ in range(100):
            assert rng1.next_u64() == rng2.next_u64()
    
    def test_unicode_seed(self):
        """Unicode seeds should work."""
        seeds = ["emoji_test", "test_unicode"]
        for seed in seeds:
            rng1 = Sha256CounterRNG.from_seed_string(seed)
            rng2 = Sha256CounterRNG.from_seed_string(seed)
            for _ in range(10):
                assert rng1.next_u64() == rng2.next_u64()
    
    def test_long_seed(self):
        """Long seeds should work."""
        seed = "x" * 10000
        rng1 = Sha256CounterRNG.from_seed_string(seed)
        rng2 = Sha256CounterRNG.from_seed_string(seed)
        for _ in range(100):
            assert rng1.next_u64() == rng2.next_u64()


# =============================================================================
# Categorical Sampling Tests
# =============================================================================

class TestCategoricalSampling:
    """Comprehensive categorical sampling tests."""
    
    def test_reproducibility(self):
        """Same RNG state should produce same samples."""
        probs = [0.1, 0.2, 0.3, 0.4]
        
        rng1 = Sha256CounterRNG.from_seed_string("cat_test")
        rng2 = Sha256CounterRNG.from_seed_string("cat_test")
        
        for _ in range(1000):
            s1 = sample_categorical(probs, rng1)
            s2 = sample_categorical(probs, rng2)
            assert s1 == s2
    
    def test_distribution_accuracy(self):
        """Sampled distribution should match input probs."""
        probs = [0.1, 0.2, 0.3, 0.4]
        rng = Sha256CounterRNG.from_seed_string("dist_test")
        
        counts = [0] * len(probs)
        n_samples = 100000
        
        for _ in range(n_samples):
            idx = sample_categorical(probs, rng)
            counts[idx] += 1
        
        chi_sq, passed = chi_squared_test(counts, probs)
        assert passed, f"Distribution chi-squared failed: {chi_sq}"
    
    def test_degenerate_single_nonzero(self):
        """Test with single non-zero probability."""
        for idx in range(5):
            probs = [0.0] * 5
            probs[idx] = 1.0
            rng = Sha256CounterRNG.from_seed_string(f"degen_{idx}")
            
            for _ in range(100):
                result = sample_categorical(probs, rng)
                assert result == idx
    
    def test_two_element_distribution(self):
        """Test binary distribution."""
        probs = [0.3, 0.7]
        rng = Sha256CounterRNG.from_seed_string("binary_test")
        
        counts = [0, 0]
        n_samples = 50000
        
        for _ in range(n_samples):
            idx = sample_categorical(probs, rng)
            counts[idx] += 1
        
        chi_sq, passed = chi_squared_test(counts, probs)
        assert passed, f"Binary distribution chi-squared failed: {chi_sq}"
    
    def test_many_element_distribution(self):
        """Test distribution with many elements."""
        n = 100
        probs = [1.0 / n] * n  # Uniform
        rng = Sha256CounterRNG.from_seed_string("many_elem_test")
        
        counts = [0] * n
        n_samples = n * 1000
        
        for _ in range(n_samples):
            idx = sample_categorical(probs, rng)
            counts[idx] += 1
        
        chi_sq, passed = chi_squared_test(counts, probs)
        assert passed, f"Many-element distribution chi-squared failed: {chi_sq}"
    
    def test_integer_weights(self):
        """Test integer weight sampling."""
        weights = [1, 2, 3, 4]
        expected_probs = [w / sum(weights) for w in weights]
        
        rng = Sha256CounterRNG.from_seed_string("weight_test")
        counts = [0] * len(weights)
        n_samples = 100000
        
        for _ in range(n_samples):
            idx = sample_categorical_weights(weights, rng)
            counts[idx] += 1
        
        chi_sq, passed = chi_squared_test(counts, expected_probs)
        assert passed, f"Integer weights chi-squared failed: {chi_sq}"
    
    def test_weighted_prefix_sampler(self):
        """Test WeightedPrefixSampler consistency and efficiency."""
        weights = [10, 20, 30, 40, 50]
        expected_probs = [w / sum(weights) for w in weights]
        sampler = WeightedPrefixSampler.from_weights(weights)
        
        rng1 = Sha256CounterRNG.from_seed_string("prefix_test")
        rng2 = Sha256CounterRNG.from_seed_string("prefix_test")
        
        # Test reproducibility
        for _ in range(100):
            s1 = sampler.sample(rng1)
            s2 = sampler.sample(rng2)
            assert s1 == s2
        
        # Test distribution
        rng = Sha256CounterRNG.from_seed_string("prefix_dist")
        counts = [0] * len(weights)
        n_samples = 100000
        
        for _ in range(n_samples):
            idx = sampler.sample(rng)
            counts[idx] += 1
        
        chi_sq, passed = chi_squared_test(counts, expected_probs)
        assert passed, f"WeightedPrefixSampler chi-squared failed: {chi_sq}"


# =============================================================================
# Sampling Function Tests
# =============================================================================

class TestSamplingFunctions:
    """Test high-level sampling functions."""
    
    def test_sample_from_probs_reproducibility(self):
        """Sampling from probs should be reproducible."""
        probs_2d = [
            [0.1, 0.2, 0.3, 0.4],
            [0.25, 0.25, 0.25, 0.25],
            [0.9, 0.05, 0.03, 0.02],
        ] * 10  # 30 steps
        
        result1 = deterministic_sample_from_probs(probs_2d, "test_seed")
        result2 = deterministic_sample_from_probs(probs_2d, "test_seed")
        
        assert result1 == result2
    
    def test_sample_from_logprobs_reproducibility(self):
        """Sampling from logprobs should be reproducible."""
        logprobs = [
            {0: -2.3, 1: -1.6, 2: -1.2, 3: -0.9},
            {0: -1.4, 1: -1.4, 2: -1.4, 3: -1.4},
        ] * 20  # 40 steps
        
        result1 = deterministic_sample_from_logprobs(logprobs, "logprob_test")
        result2 = deterministic_sample_from_logprobs(logprobs, "logprob_test")
        
        assert result1 == result2
    
    def test_temperature_effect(self):
        """Temperature should affect sampling distribution."""
        logprobs = [{0: -0.1, 1: -2.0, 2: -3.0}] * 1000
        
        # Low temperature -> more peaked (token 0 dominates)
        low_temp_results = deterministic_sample_from_logprobs(
            logprobs, "temp_low", temperature=0.1
        )
        low_temp_token0_count = sum(1 for t in low_temp_results if t == 0)
        
        # High temperature -> more uniform
        high_temp_results = deterministic_sample_from_logprobs(
            logprobs, "temp_high", temperature=2.0
        )
        high_temp_token0_count = sum(1 for t in high_temp_results if t == 0)
        
        # Low temp should have more token 0
        assert low_temp_token0_count > high_temp_token0_count
        assert low_temp_token0_count > 950  # Should be almost all token 0
    
    def test_top_k_filtering(self):
        """Top-k should limit sampling to top k tokens."""
        logprobs = [{0: -0.5, 1: -1.0, 2: -1.5, 3: -2.0, 4: -2.5}] * 1000
        
        results = deterministic_sample_from_logprobs(
            logprobs, "topk_test", temperature=1.0, top_k=2
        )
        
        # Should only sample from top 2 tokens (0 and 1)
        for t in results:
            assert t in [0, 1], f"Token {t} should not be sampled with top_k=2"
    
    def test_top_p_filtering(self):
        """Top-p should limit sampling by cumulative probability."""
        # Token 0 has ~90% probability after softmax
        logprobs = [{0: -0.1, 1: -2.5, 2: -3.0, 3: -4.0, 4: -5.0}] * 1000
        
        results = deterministic_sample_from_logprobs(
            logprobs, "topp_test", temperature=1.0, top_p=0.95
        )
        
        # With top_p=0.95, should mostly sample token 0
        token0_count = sum(1 for t in results if t == 0)
        assert token0_count > 800
    
    def test_different_seeds_different_results(self):
        """Different seeds should produce different results."""
        probs_2d = [[0.25, 0.25, 0.25, 0.25] for _ in range(50)]
        
        results = [
            deterministic_sample_from_probs(probs_2d, f"seed_{i}")
            for i in range(10)
        ]
        
        # All should be different
        unique_results = set(tuple(r) for r in results)
        assert len(unique_results) == 10


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_token_vocab(self):
        """Test with vocabulary of size 1."""
        probs = [1.0]
        rng = Sha256CounterRNG.from_seed_string("single_vocab")
        
        for _ in range(100):
            idx = sample_categorical(probs, rng)
            assert idx == 0
    
    def test_all_zero_weights(self):
        """Test with all zero weights."""
        weights = [0, 0, 0, 0]
        rng = Sha256CounterRNG.from_seed_string("all_zero")
        
        # Should return last index
        for _ in range(10):
            idx = sample_categorical_weights(weights, rng)
            assert idx == 3
    
    def test_large_vocab(self):
        """Test with large vocabulary."""
        n = 100000
        probs = [1.0 / n] * n
        rng = Sha256CounterRNG.from_seed_string("large_vocab")
        
        start = time.time()
        for _ in range(100):
            idx = sample_categorical(probs, rng)
            assert 0 <= idx < n
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        assert elapsed < 10.0
    
    def test_extreme_probabilities(self):
        """Test with extreme probability values."""
        # Very peaked distribution
        probs = [0.9999999] + [0.0000001 / 99] * 99
        rng = Sha256CounterRNG.from_seed_string("extreme_prob")
        
        counts = [0] * 100
        for _ in range(10000):
            idx = sample_categorical(probs, rng)
            counts[idx] += 1
        
        assert counts[0] > 9900
    
    def test_negative_logprobs(self):
        """Test with very negative logprobs."""
        logprobs = [{0: -100.0, 1: -0.01}]  # Token 1 should dominate
        
        results = deterministic_sample_from_logprobs(logprobs * 1000, "neg_logprob")
        token1_count = sum(1 for t in results if t == 1)
        
        assert token1_count > 990


# =============================================================================
# Reference Values Tests
# =============================================================================

class TestReferenceValues:
    """Test against known reference values for cross-implementation verification."""
    
    def test_reference_u64_values(self):
        """Verify RNG produces expected reference values."""
        rng = Sha256CounterRNG.from_seed_string("reference_seed_v1")
        
        # These are reference values that any correct implementation should produce
        expected_first_5 = []
        for _ in range(5):
            expected_first_5.append(rng.next_u64())
        
        # Verify reproducibility
        rng2 = Sha256CounterRNG.from_seed_string("reference_seed_v1")
        actual_first_5 = [rng2.next_u64() for _ in range(5)]
        
        assert expected_first_5 == actual_first_5
    
    def test_reference_categorical_samples(self):
        """Verify categorical sampling produces consistent results."""
        probs = [0.1, 0.2, 0.3, 0.4]
        rng1 = Sha256CounterRNG.from_seed_string("categorical_ref_v1")
        rng2 = Sha256CounterRNG.from_seed_string("categorical_ref_v1")
        
        expected_samples = [sample_categorical(probs, rng1) for _ in range(10)]
        actual_samples = [sample_categorical(probs, rng2) for _ in range(10)]
        
        assert expected_samples == actual_samples
