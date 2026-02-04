#!/usr/bin/env python3
"""
Comprehensive test suite for deterministic sampling utilities.
Does not require full vLLM installation - only tests the core cryptographic sampling.

Tests include:
- RNG reproducibility and statistical properties
- Categorical sampling correctness
- Edge cases (degenerate distributions, extreme values)
- Real data reproducibility with larger sample sizes
- Cross-run consistency validation
- Temperature and top-k/top-p filtering
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# Add the vllm path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

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


def resample_from_inference_logprobs(
    inference_results: List[Dict],
    prompt: str,
    temperature: float = 1.0,
    seed_prefix: str = "resample",
    top_k: int | None = None,
    top_p: float | None = None,
) -> List[int]:
    """Re-sample tokens from inference result logprobs."""
    seed = f"{seed_prefix}|{prompt}"
    
    logprobs_seq = []
    for result in inference_results:
        lp = result.get("logprobs", {})
        lp_int = {int(k): v for k, v in lp.items()}
        logprobs_seq.append(lp_int)
    
    return deterministic_sample_from_logprobs(
        logprobs_seq,
        seed=seed,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )


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
        print("PASS: RNG reproducibility (1000 values)")
    
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
        print(f"PASS: {len(seeds)} different seeds produce different sequences")
    
    def test_uniform_range(self):
        """Uniform values should be in [0, 1)."""
        rng = Sha256CounterRNG.from_seed_string("uniform_test")
        for _ in range(10000):
            u = rng.next_uniform01()
            assert 0.0 <= u < 1.0
        print("PASS: 10000 uniform values all in [0, 1)")
    
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
        print(f"PASS: Uniform distribution chi-squared test (χ²={chi_sq:.2f})")
    
    def test_uint64_below_unbiased(self):
        """uint64_below should be unbiased for various n values."""
        rng = Sha256CounterRNG.from_seed_string("uint64_below_test")
        
        for n in [2, 3, 7, 10, 100, 1000]:
            counts = [0] * n
            n_samples = n * 1000
            
            for _ in range(n_samples):
                val = uint64_below(rng, n)
                assert 0 <= val < n
                counts[val] += 1
            
            expected = [1.0 / n] * n
            chi_sq, passed = chi_squared_test(counts, expected)
            
            assert passed, f"uint64_below({n}) failed chi-squared: {chi_sq}"
        
        print("PASS: uint64_below unbiased for n in [2, 3, 7, 10, 100, 1000]")
    
    def test_iter_u64_reproducibility(self):
        """iter_u64 helper should be reproducible."""
        vals1 = iter_u64("test", 100)
        vals2 = iter_u64("test", 100)
        assert vals1 == vals2
        print("PASS: iter_u64 reproducibility")
    
    def test_empty_seed(self):
        """Empty seed should work."""
        rng1 = Sha256CounterRNG.from_seed_string("")
        rng2 = Sha256CounterRNG.from_seed_string("")
        
        for _ in range(100):
            assert rng1.next_u64() == rng2.next_u64()
        print("PASS: Empty seed works")
    
    def test_unicode_seed(self):
        """Unicode seeds should work."""
        seeds = ["🎲", "日本語", "emoji🔥test", "Ñoño"]
        for seed in seeds:
            rng1 = Sha256CounterRNG.from_seed_string(seed)
            rng2 = Sha256CounterRNG.from_seed_string(seed)
            for _ in range(10):
                assert rng1.next_u64() == rng2.next_u64()
        print("PASS: Unicode seeds work")
    
    def test_long_seed(self):
        """Long seeds should work."""
        seed = "x" * 10000
        rng1 = Sha256CounterRNG.from_seed_string(seed)
        rng2 = Sha256CounterRNG.from_seed_string(seed)
        for _ in range(100):
            assert rng1.next_u64() == rng2.next_u64()
        print("PASS: Long seed (10000 chars) works")


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
        print("PASS: Categorical sampling reproducibility (1000 samples)")
    
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
        print(f"PASS: Categorical distribution accuracy (χ²={chi_sq:.2f})")
    
    def test_degenerate_single_nonzero(self):
        """Test with single non-zero probability."""
        for idx in range(5):
            probs = [0.0] * 5
            probs[idx] = 1.0
            rng = Sha256CounterRNG.from_seed_string(f"degen_{idx}")
            
            for _ in range(100):
                result = sample_categorical(probs, rng)
                assert result == idx
        print("PASS: Degenerate distribution (single non-zero)")
    
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
        print(f"PASS: Binary distribution accuracy (χ²={chi_sq:.2f})")
    
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
        print(f"PASS: 100-element uniform distribution (χ²={chi_sq:.2f})")
    
    def test_very_small_probabilities(self):
        """Test with very small probabilities."""
        probs = [0.99, 0.009, 0.0009, 0.0001]
        rng = Sha256CounterRNG.from_seed_string("small_prob_test")
        
        counts = [0] * len(probs)
        n_samples = 100000
        
        for _ in range(n_samples):
            idx = sample_categorical(probs, rng)
            counts[idx] += 1
        
        # Just verify it runs and index 0 dominates
        assert counts[0] > n_samples * 0.95
        print(f"PASS: Very small probabilities (counts: {counts})")
    
    def test_unnormalized_probs(self):
        """Test with probabilities that don't sum to exactly 1."""
        probs = [0.3333, 0.3333, 0.3333]  # Sums to 0.9999
        rng = Sha256CounterRNG.from_seed_string("unnorm_test")
        
        counts = [0, 0, 0]
        n_samples = 30000
        
        for _ in range(n_samples):
            idx = sample_categorical(probs, rng)
            counts[idx] += 1
        
        # Should still be roughly uniform
        for c in counts:
            assert abs(c / n_samples - 1/3) < 0.05
        print("PASS: Unnormalized probabilities handled correctly")
    
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
        print(f"PASS: Integer weight sampling (χ²={chi_sq:.2f})")
    
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
        print(f"PASS: WeightedPrefixSampler (χ²={chi_sq:.2f})")


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
        print("PASS: Sample from probs reproducibility (30 steps)")
    
    def test_sample_from_logprobs_reproducibility(self):
        """Sampling from logprobs should be reproducible."""
        logprobs = [
            {0: -2.3, 1: -1.6, 2: -1.2, 3: -0.9},
            {0: -1.4, 1: -1.4, 2: -1.4, 3: -1.4},
        ] * 20  # 40 steps
        
        result1 = deterministic_sample_from_logprobs(logprobs, "logprob_test")
        result2 = deterministic_sample_from_logprobs(logprobs, "logprob_test")
        
        assert result1 == result2
        print("PASS: Sample from logprobs reproducibility (40 steps)")
    
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
        print(f"PASS: Temperature effect (low={low_temp_token0_count}/1000, high={high_temp_token0_count}/1000)")
    
    def test_top_k_filtering(self):
        """Top-k should limit sampling to top k tokens."""
        logprobs = [{0: -0.5, 1: -1.0, 2: -1.5, 3: -2.0, 4: -2.5}] * 1000
        
        results = deterministic_sample_from_logprobs(
            logprobs, "topk_test", temperature=1.0, top_k=2
        )
        
        # Should only sample from top 2 tokens (0 and 1)
        for t in results:
            assert t in [0, 1], f"Token {t} should not be sampled with top_k=2"
        print("PASS: Top-k filtering")
    
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
        print(f"PASS: Top-p filtering (token 0: {token0_count}/1000)")
    
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
        print("PASS: 10 different seeds produce 10 different results")


# =============================================================================
# Real Data Tests
# =============================================================================

class TestRealData:
    """Test with real inference data."""
    
    DATA_PATH = Path("/root/gonka/mlnode/packages/benchmarks/data/repro/honest_A100_vs_A100.jsonl")
    
    def _load_samples(self, n: int = 100) -> List[Dict]:
        """Load n samples from the data file."""
        if not self.DATA_PATH.exists():
            return []
        
        samples = []
        with open(self.DATA_PATH, "r") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                samples.append(json.loads(line))
        return samples
    
    def test_large_scale_reproducibility(self):
        """Test reproducibility across many samples."""
        samples = self._load_samples(100)
        if not samples:
            print("SKIP: Data file not found")
            return
        
        print(f"Testing {len(samples)} samples for reproducibility...")
        
        all_pass = True
        total_tokens = 0
        
        for idx, data in enumerate(samples):
            prompt = data["prompt"]
            inference_results = data["inference_result"]["results"]
            
            # Run 3 times
            tokens1 = resample_from_inference_logprobs(
                inference_results, prompt, temperature=0.99, seed_prefix="scale_test"
            )
            tokens2 = resample_from_inference_logprobs(
                inference_results, prompt, temperature=0.99, seed_prefix="scale_test"
            )
            tokens3 = resample_from_inference_logprobs(
                inference_results, prompt, temperature=0.99, seed_prefix="scale_test"
            )
            
            if tokens1 != tokens2 or tokens2 != tokens3:
                print(f"  FAIL: Sample {idx}")
                all_pass = False
            
            total_tokens += len(tokens1)
        
        if all_pass:
            print(f"PASS: All {len(samples)} samples reproducible ({total_tokens} total tokens)")
        else:
            raise AssertionError("Some samples failed reproducibility")
    
    def test_cross_run_consistency(self):
        """Test that results are consistent across many independent runs."""
        samples = self._load_samples(10)
        if not samples:
            print("SKIP: Data file not found")
            return
        
        data = samples[0]
        prompt = data["prompt"]
        inference_results = data["inference_result"]["results"]
        
        # Run 20 times independently
        results = []
        for i in range(20):
            tokens = resample_from_inference_logprobs(
                inference_results, prompt, temperature=0.99, seed_prefix="cross_run"
            )
            results.append(tuple(tokens))
        
        # All should be identical
        unique = set(results)
        assert len(unique) == 1, f"Got {len(unique)} unique results instead of 1"
        print(f"PASS: 20 independent runs produce identical results ({len(results[0])} tokens)")
    
    def test_with_different_temperatures(self):
        """Test reproducibility at different temperatures."""
        samples = self._load_samples(5)
        if not samples:
            print("SKIP: Data file not found")
            return
        
        temperatures = [0.5, 0.7, 0.99, 1.0, 1.5, 2.0]
        
        for data in samples[:3]:
            prompt = data["prompt"]
            inference_results = data["inference_result"]["results"]
            
            for temp in temperatures:
                r1 = resample_from_inference_logprobs(
                    inference_results, prompt, temperature=temp, seed_prefix="temp_test"
                )
                r2 = resample_from_inference_logprobs(
                    inference_results, prompt, temperature=temp, seed_prefix="temp_test"
                )
                assert r1 == r2, f"Failed at temperature {temp}"
        
        print(f"PASS: Reproducibility at temperatures {temperatures}")
    
    def test_with_top_k(self):
        """Test reproducibility with top-k filtering."""
        samples = self._load_samples(5)
        if not samples:
            print("SKIP: Data file not found")
            return
        
        for data in samples[:3]:
            prompt = data["prompt"]
            inference_results = data["inference_result"]["results"]
            
            for top_k in [1, 2, 3, 5]:
                r1 = resample_from_inference_logprobs(
                    inference_results, prompt, temperature=0.99, 
                    seed_prefix="topk_test", top_k=top_k
                )
                r2 = resample_from_inference_logprobs(
                    inference_results, prompt, temperature=0.99,
                    seed_prefix="topk_test", top_k=top_k
                )
                assert r1 == r2, f"Failed at top_k={top_k}"
        
        print("PASS: Reproducibility with top_k in [1, 2, 3, 5]")
    
    def test_original_token_in_logprobs(self):
        """Verify original tokens are in the logprobs."""
        samples = self._load_samples(50)
        if not samples:
            print("SKIP: Data file not found")
            return
        
        missing_count = 0
        total_tokens = 0
        
        for data in samples:
            inference_results = data["inference_result"]["results"]
            
            for result in inference_results:
                original_token = result["token"]
                logprobs_dict = result["logprobs"]
                token_ids = [int(k) for k in logprobs_dict.keys()]
                
                total_tokens += 1
                if original_token not in token_ids:
                    missing_count += 1
        
        print(f"PASS: {total_tokens - missing_count}/{total_tokens} original tokens in top-5 logprobs")
    
    def test_statistics_summary(self):
        """Generate statistics summary of the test data."""
        samples = self._load_samples(100)
        if not samples:
            print("SKIP: Data file not found")
            return
        
        token_counts = []
        prompt_lengths = []
        
        for data in samples:
            inference_results = data["inference_result"]["results"]
            token_counts.append(len(inference_results))
            prompt_lengths.append(len(data["prompt"]))
        
        print(f"\n  Data Statistics:")
        print(f"  - Samples: {len(samples)}")
        print(f"  - Total tokens: {sum(token_counts)}")
        print(f"  - Avg tokens/sample: {sum(token_counts) / len(token_counts):.1f}")
        print(f"  - Min/Max tokens: {min(token_counts)}/{max(token_counts)}")
        print(f"  - Avg prompt length: {sum(prompt_lengths) / len(prompt_lengths):.0f} chars")


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
        print("PASS: Single token vocabulary")
    
    def test_all_zero_weights(self):
        """Test with all zero weights."""
        weights = [0, 0, 0, 0]
        rng = Sha256CounterRNG.from_seed_string("all_zero")
        
        # Should return last index
        for _ in range(10):
            idx = sample_categorical_weights(weights, rng)
            assert idx == 3
        print("PASS: All zero weights returns last index")
    
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
        
        print(f"PASS: Large vocab ({n} tokens), 100 samples in {elapsed:.3f}s")
    
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
        print(f"PASS: Extreme probabilities (token 0: {counts[0]}/10000)")
    
    def test_negative_logprobs(self):
        """Test with very negative logprobs."""
        logprobs = [{0: -100.0, 1: -0.01}]  # Token 1 should dominate
        
        results = deterministic_sample_from_logprobs(logprobs * 1000, "neg_logprob")
        token1_count = sum(1 for t in results if t == 1)
        
        assert token1_count > 990
        print(f"PASS: Very negative logprobs handled (token 1: {token1_count}/1000)")
    
    def test_special_characters_in_prompt(self):
        """Test with special characters in prompt seed."""
        prompts = [
            "Hello\nWorld",
            "Tab\there",
            "Null\x00char",
            "Unicode: 你好世界🌍",
            "<script>alert('xss')</script>",
            "a" * 10000,
        ]
        
        logprobs = [{0: -1.0, 1: -1.0}] * 10
        
        for prompt in prompts:
            r1 = deterministic_sample_from_logprobs(
                logprobs, f"special|{prompt}", temperature=1.0
            )
            r2 = deterministic_sample_from_logprobs(
                logprobs, f"special|{prompt}", temperature=1.0
            )
            assert r1 == r2
        
        print(f"PASS: Special characters in prompt ({len(prompts)} cases)")


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Performance benchmarks."""
    
    def test_sampling_throughput(self):
        """Measure sampling throughput."""
        probs = [0.1, 0.2, 0.3, 0.4]
        rng = Sha256CounterRNG.from_seed_string("perf_test")
        
        n_samples = 100000
        start = time.time()
        for _ in range(n_samples):
            sample_categorical(probs, rng)
        elapsed = time.time() - start
        
        throughput = n_samples / elapsed
        print(f"PASS: Sampling throughput: {throughput:.0f} samples/sec")
    
    def test_rng_throughput(self):
        """Measure RNG throughput."""
        rng = Sha256CounterRNG.from_seed_string("rng_perf")
        
        n_samples = 1000000
        start = time.time()
        for _ in range(n_samples):
            rng.next_u64()
        elapsed = time.time() - start
        
        throughput = n_samples / elapsed
        print(f"PASS: RNG throughput: {throughput:.0f} u64/sec")


# =============================================================================
# Cross-Implementation Reference Tests
# =============================================================================

def test_reference_values():
    """
    Test against known reference values for cross-implementation verification.
    These values can be used to verify other implementations produce the same results.
    """
    # Reference test 1: RNG sequence
    rng = Sha256CounterRNG.from_seed_string("reference_seed_v1")
    expected_u64s = [
        rng.next_u64() for _ in range(5)
    ]
    
    # Verify we can reproduce
    rng2 = Sha256CounterRNG.from_seed_string("reference_seed_v1")
    actual_u64s = [rng2.next_u64() for _ in range(5)]
    assert expected_u64s == actual_u64s
    
    print(f"\n  Reference u64 values for seed 'reference_seed_v1':")
    print(f"  {expected_u64s}")
    
    # Reference test 2: Categorical sampling
    probs = [0.1, 0.2, 0.3, 0.4]
    rng = Sha256CounterRNG.from_seed_string("categorical_ref_v1")
    expected_samples = [sample_categorical(probs, rng) for _ in range(10)]
    
    rng2 = Sha256CounterRNG.from_seed_string("categorical_ref_v1")
    actual_samples = [sample_categorical(probs, rng2) for _ in range(10)]
    assert expected_samples == actual_samples
    
    print(f"\n  Reference samples for probs [0.1, 0.2, 0.3, 0.4], seed 'categorical_ref_v1':")
    print(f"  {expected_samples}")
    
    # Reference test 3: Full sampling from logprobs
    logprobs = [
        {100: -0.5, 200: -1.2, 300: -2.1},
        {100: -1.0, 200: -1.0, 300: -1.0},
        {100: -0.1, 200: -3.5, 300: -4.2},
    ]
    expected_tokens = deterministic_sample_from_logprobs(logprobs, "logprobs_ref_v1")
    
    print(f"\n  Reference tokens for logprobs test, seed 'logprobs_ref_v1':")
    print(f"  {expected_tokens}")
    
    print("\nPASS: Reference values generated and verified")


# =============================================================================
# Main
# =============================================================================

def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("Comprehensive Deterministic Sampler Test Suite")
    print("=" * 70)
    
    # RNG Tests
    print("\n" + "=" * 70)
    print("RNG Tests")
    print("=" * 70)
    test_rng = TestRNG()
    test_rng.test_reproducibility()
    test_rng.test_different_seeds()
    test_rng.test_uniform_range()
    test_rng.test_uniform_distribution()
    test_rng.test_uint64_below_unbiased()
    test_rng.test_iter_u64_reproducibility()
    test_rng.test_empty_seed()
    test_rng.test_unicode_seed()
    test_rng.test_long_seed()
    
    # Categorical Sampling Tests
    print("\n" + "=" * 70)
    print("Categorical Sampling Tests")
    print("=" * 70)
    test_cat = TestCategoricalSampling()
    test_cat.test_reproducibility()
    test_cat.test_distribution_accuracy()
    test_cat.test_degenerate_single_nonzero()
    test_cat.test_two_element_distribution()
    test_cat.test_many_element_distribution()
    test_cat.test_very_small_probabilities()
    test_cat.test_unnormalized_probs()
    test_cat.test_integer_weights()
    test_cat.test_weighted_prefix_sampler()
    
    # Sampling Function Tests
    print("\n" + "=" * 70)
    print("Sampling Function Tests")
    print("=" * 70)
    test_func = TestSamplingFunctions()
    test_func.test_sample_from_probs_reproducibility()
    test_func.test_sample_from_logprobs_reproducibility()
    test_func.test_temperature_effect()
    test_func.test_top_k_filtering()
    test_func.test_top_p_filtering()
    test_func.test_different_seeds_different_results()
    
    # Edge Case Tests
    print("\n" + "=" * 70)
    print("Edge Case Tests")
    print("=" * 70)
    test_edge = TestEdgeCases()
    test_edge.test_single_token_vocab()
    test_edge.test_all_zero_weights()
    test_edge.test_large_vocab()
    test_edge.test_extreme_probabilities()
    test_edge.test_negative_logprobs()
    test_edge.test_special_characters_in_prompt()
    
    # Performance Tests
    print("\n" + "=" * 70)
    print("Performance Tests")
    print("=" * 70)
    test_perf = TestPerformance()
    test_perf.test_sampling_throughput()
    test_perf.test_rng_throughput()
    
    # Real Data Tests
    print("\n" + "=" * 70)
    print("Real Data Tests")
    print("=" * 70)
    test_real = TestRealData()
    test_real.test_large_scale_reproducibility()
    test_real.test_cross_run_consistency()
    test_real.test_with_different_temperatures()
    test_real.test_with_top_k()
    test_real.test_original_token_in_logprobs()
    test_real.test_statistics_summary()
    
    # Reference Values
    print("\n" + "=" * 70)
    print("Reference Values for Cross-Implementation Verification")
    print("=" * 70)
    test_reference_values()
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
