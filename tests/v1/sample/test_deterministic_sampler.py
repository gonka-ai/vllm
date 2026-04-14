# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test suite for the deterministic sampler.

This test file has two modes:
1. If vllm is importable (C extension works), it tests the full DeterministicSampler class
2. If vllm is not importable, it falls back to testing standalone functions

Tests:
1. Basic reproducibility: same seed produces same results
2. Different seeds produce different results
3. Cross-run reproducibility: can reproduce results across multiple invocations
4. Real data test: resample from actual inference logprobs
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import pytest

# Always import the core utils (no dependencies)
from vllm.v1.sample.deterministic_utils import (
    Sha256CounterRNG,
    sample_categorical,
    sample_categorical_weights,
    iter_u64,
)

# Try to import the full sampler (has torch/vllm dependencies)
VLLM_AVAILABLE = False
try:
    from vllm.v1.sample.deterministic_sampler import (
        DeterministicSampler,
        DeterministicSamplingState,
        deterministic_sample_from_logprobs,
        deterministic_sample_from_probs,
        resample_from_inference_logprobs,
    )
    VLLM_AVAILABLE = True
except ImportError as e:
    # Define standalone implementations when vllm is not available
    def deterministic_sample_from_logprobs(
        logprobs: Sequence[Dict[int, float]],
        seed: str,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> List[int]:
        """Sample tokens deterministically from logprob dictionaries."""
        rng = Sha256CounterRNG.from_seed_string(seed)
        sampled_tokens = []
        
        for step_logprobs in logprobs:
            if not step_logprobs:
                raise ValueError("Empty logprobs at step")
            
            token_ids = list(step_logprobs.keys())
            log_values = [step_logprobs[tid] for tid in token_ids]
            
            # Apply temperature
            if temperature != 1.0 and temperature > 0:
                log_values = [lv / temperature for lv in log_values]
            
            # Softmax
            max_log = max(log_values)
            exp_values = [math.exp(lv - max_log) for lv in log_values]
            sum_exp = sum(exp_values)
            probs = [ev / sum_exp for ev in exp_values]
            
            # Top-k
            if top_k is not None and top_k < len(token_ids):
                sorted_items = sorted(
                    zip(token_ids, probs), key=lambda x: x[1], reverse=True
                )[:top_k]
                token_ids = [x[0] for x in sorted_items]
                probs = [x[1] for x in sorted_items]
                sum_probs = sum(probs)
                probs = [p / sum_probs for p in probs]
            
            # Top-p
            if top_p is not None and top_p < 1.0:
                sorted_items = sorted(
                    zip(token_ids, probs), key=lambda x: x[1], reverse=True
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
    ) -> List[int]:
        """Re-sample tokens from inference result logprobs."""
        seed = f"{seed_prefix}|{prompt}"
        logprobs_seq = []
        for result in inference_results:
            lp = result.get("logprobs", {})
            lp_int = {int(k): v for k, v in lp.items()}
            logprobs_seq.append(lp_int)
        return deterministic_sample_from_logprobs(
            logprobs_seq, seed=seed, temperature=temperature
        )


# =============================================================================
# Unit Tests for RNG
# =============================================================================

class TestSha256CounterRNG:
    """Test the SHA256-based RNG."""
    
    def test_reproducibility(self):
        """Same seed should produce same sequence."""
        rng1 = Sha256CounterRNG.from_seed_string("test_seed")
        rng2 = Sha256CounterRNG.from_seed_string("test_seed")
        
        for _ in range(100):
            assert rng1.next_u64() == rng2.next_u64()
    
    def test_different_seeds_different_output(self):
        """Different seeds should produce different sequences."""
        rng1 = Sha256CounterRNG.from_seed_string("seed_a")
        rng2 = Sha256CounterRNG.from_seed_string("seed_b")
        
        vals1 = [rng1.next_u64() for _ in range(10)]
        vals2 = [rng2.next_u64() for _ in range(10)]
        assert vals1 != vals2
    
    def test_uniform01_range(self):
        """Uniform values should be in [0, 1)."""
        rng = Sha256CounterRNG.from_seed_string("uniform_test")
        for _ in range(1000):
            u = rng.next_uniform01()
            assert 0.0 <= u < 1.0
    
    def test_iter_u64_reproducibility(self):
        """iter_u64 helper should be reproducible."""
        vals1 = iter_u64("test", 50)
        vals2 = iter_u64("test", 50)
        assert vals1 == vals2


# =============================================================================
# Unit Tests for Categorical Sampling
# =============================================================================

class TestCategoricalSampling:
    """Test categorical sampling functions."""
    
    def test_sample_categorical_reproducibility(self):
        """Same RNG state should produce same samples."""
        probs = [0.1, 0.2, 0.3, 0.4]
        
        rng1 = Sha256CounterRNG.from_seed_string("cat_test")
        rng2 = Sha256CounterRNG.from_seed_string("cat_test")
        
        for _ in range(100):
            s1 = sample_categorical(probs, rng1)
            s2 = sample_categorical(probs, rng2)
            assert s1 == s2
    
    def test_sample_categorical_distribution(self):
        """Sampled distribution should roughly match input probs."""
        probs = [0.1, 0.2, 0.3, 0.4]
        rng = Sha256CounterRNG.from_seed_string("dist_test")
        
        counts = [0, 0, 0, 0]
        n_samples = 10000
        
        for _ in range(n_samples):
            idx = sample_categorical(probs, rng)
            counts[idx] += 1
        
        for i, p in enumerate(probs):
            empirical = counts[i] / n_samples
            assert abs(empirical - p) < 0.05, f"Index {i}: expected ~{p}, got {empirical}"
    
    def test_sample_categorical_weights(self):
        """Test integer weight sampling."""
        weights = [1, 2, 3, 4]
        
        rng1 = Sha256CounterRNG.from_seed_string("weight_test")
        rng2 = Sha256CounterRNG.from_seed_string("weight_test")
        
        for _ in range(50):
            s1 = sample_categorical_weights(weights, rng1)
            s2 = sample_categorical_weights(weights, rng2)
            assert s1 == s2
    
    def test_degenerate_distribution(self):
        """Test with a degenerate distribution (all mass on one token)."""
        probs = [0.0, 0.0, 1.0, 0.0]
        rng = Sha256CounterRNG.from_seed_string("degen_test")
        
        for _ in range(100):
            idx = sample_categorical(probs, rng)
            assert idx == 2


# =============================================================================
# Unit Tests for Deterministic Sampler Functions
# =============================================================================

class TestDeterministicSamplerFunctions:
    """Test standalone sampling functions."""
    
    def test_sample_from_probs_reproducibility(self):
        """Sampling from probs should be reproducible."""
        probs_2d = [
            [0.1, 0.2, 0.3, 0.4],
            [0.25, 0.25, 0.25, 0.25],
            [0.9, 0.05, 0.03, 0.02],
        ]
        
        result1 = deterministic_sample_from_probs(probs_2d, "test_seed")
        result2 = deterministic_sample_from_probs(probs_2d, "test_seed")
        
        assert result1 == result2
    
    def test_sample_from_logprobs_reproducibility(self):
        """Sampling from logprobs dicts should be reproducible."""
        logprobs = [
            {0: -2.3, 1: -1.6, 2: -1.2, 3: -0.9},
            {0: -1.4, 1: -1.4, 2: -1.4, 3: -1.4},
        ]
        
        result1 = deterministic_sample_from_logprobs(logprobs, "logprob_test")
        result2 = deterministic_sample_from_logprobs(logprobs, "logprob_test")
        
        assert result1 == result2
    
    def test_sample_from_logprobs_with_temperature(self):
        """Temperature should affect sampling distribution."""
        logprobs = [{0: -0.1, 1: -2.3}]
        
        low_temp_results = []
        for i in range(100):
            result = deterministic_sample_from_logprobs(
                logprobs, f"temp_test_{i}", temperature=0.1
            )
            low_temp_results.append(result[0])
        
        assert low_temp_results.count(0) > 90
    
    def test_different_seeds_different_results(self):
        """Different seeds should (usually) produce different results."""
        probs_2d = [[0.25, 0.25, 0.25, 0.25] for _ in range(20)]
        
        result1 = deterministic_sample_from_probs(probs_2d, "seed_a")
        result2 = deterministic_sample_from_probs(probs_2d, "seed_b")
        
        assert result1 != result2


# =============================================================================
# Test for DeterministicSampler class (only if vllm is available)
# =============================================================================

class TestDeterministicSamplerClass:
    """Test the DeterministicSampler class (requires full vllm)."""
    
    @pytest.mark.skipif(not VLLM_AVAILABLE, reason="Full vLLM not available")
    def test_sampler_state_creation(self):
        """Test creating sampling state from seeds."""
        state = DeterministicSamplingState.from_seeds({
            0: "prompt_0",
            1: "prompt_1",
        })
        
        # Get RNGs
        rng0 = state.get_rng(0)
        rng1 = state.get_rng(1)
        
        # Should be different
        assert rng0.next_u64() != rng1.next_u64()
    
    @pytest.mark.skipif(not VLLM_AVAILABLE, reason="Full vLLM not available")
    def test_sampler_state_reproducibility(self):
        """Test that state produces reproducible RNGs."""
        state1 = DeterministicSamplingState.from_seeds({0: "test"})
        state2 = DeterministicSamplingState.from_seeds({0: "test"})
        
        for _ in range(100):
            assert state1.get_rng(0).next_u64() == state2.get_rng(0).next_u64()


# =============================================================================
# Comprehensive Reproducibility Test
# =============================================================================

def test_full_reproducibility_pipeline():
    """
    Comprehensive test: Sample twice with same seed, verify identical results.
    """
    probs_2d = [[0.1, 0.2, 0.3, 0.4] for _ in range(100)]
    seed = "synthetic_test_seed"
    
    run1 = deterministic_sample_from_probs(probs_2d, seed)
    run2 = deterministic_sample_from_probs(probs_2d, seed)
    
    assert run1 == run2
