#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test that integer weight sampling is perfectly reproducible.

This script verifies the core assumption of deterministic sampling:
given the same integer weights and seed, sampling will be deterministic
across any machine and any run.

Test procedure:
1. Generate random raw logprobs (simulating model output)
2. Apply processing (temperature) to get final probabilities
3. Quantize to integer weights (2^16 scale)
4. Sample sequences using same seed
5. Verify all sequences are identical

Run:
    python scripts/test_integer_weight_reproducibility.py
"""

import random
import math
from typing import List, Dict, Tuple

from vllm.v1.sample.deterministic_utils import (
    Sha256CounterRNG,
    sample_categorical_weights,
    sample_categorical,
)
from vllm.validation_logic import (
    quantize_to_weights,
    recompute_weights_from_logprobs,
    WEIGHT_SCALE,
)


def generate_random_logprobs(
    vocab_size: int = 100,
    top_k: int = 5,
) -> Dict[str, float]:
    """Generate random logprobs for testing."""
    # Generate random logprobs (mostly very negative, a few dominant)
    top_k_ids = random.sample(range(vocab_size), top_k)
    
    logprobs = {}
    # Dominant token
    logprobs[str(top_k_ids[0])] = -random.uniform(0.1, 2.0)
    
    # Other top-k tokens
    for tid in top_k_ids[1:]:
        logprobs[str(tid)] = -random.uniform(3.0, 10.0)
    
    return logprobs


def test_rng_reproducibility():
    """Test that RNG produces identical sequences with same seed."""
    print("=" * 60)
    print("Test 1: RNG Reproducibility")
    print("=" * 60)
    
    seed = "test_reproducibility_seed"
    n_values = 100
    
    # Generate sequence twice
    rng1 = Sha256CounterRNG.from_seed_string(seed)
    seq1 = [rng1.next_u64() for _ in range(n_values)]
    
    rng2 = Sha256CounterRNG.from_seed_string(seed)
    seq2 = [rng2.next_u64() for _ in range(n_values)]
    
    assert seq1 == seq2, "RNG sequences differ!"
    print(f"✓ Generated {n_values} identical u64 values with same seed")
    
    # Test uniform01 reproducibility
    rng1 = Sha256CounterRNG.from_seed_string(seed)
    rng2 = Sha256CounterRNG.from_seed_string(seed)
    
    for i in range(100):
        v1 = rng1.next_uniform01()
        v2 = rng2.next_uniform01()
        assert v1 == v2, f"Uniform values differ at position {i}"
    
    print(f"✓ Generated 100 identical uniform01 values with same seed")


def test_weight_sampling_reproducibility():
    """Test that sampling from integer weights is reproducible."""
    print("\n" + "=" * 60)
    print("Test 2: Weight Sampling Reproducibility")
    print("=" * 60)
    
    # Fixed weights for testing
    weights = [60000, 5000, 500, 35, 1]  # ~91%, ~8%, ~0.8%, ~0.05%, ~0.001%
    seed = "weight_sampling_test"
    n_samples = 1000
    
    # Sample sequence twice
    results1 = []
    rng1 = Sha256CounterRNG.from_seed_string(seed)
    for _ in range(n_samples):
        idx = sample_categorical_weights(weights, rng1)
        results1.append(idx)
    
    results2 = []
    rng2 = Sha256CounterRNG.from_seed_string(seed)
    for _ in range(n_samples):
        idx = sample_categorical_weights(weights, rng2)
        results2.append(idx)
    
    assert results1 == results2, "Weight sampling sequences differ!"
    print(f"✓ Sampled {n_samples} identical indices with same seed")
    
    # Print distribution
    from collections import Counter
    counter = Counter(results1)
    print(f"  Distribution: {dict(counter)}")


def test_full_pipeline_reproducibility():
    """Test full pipeline: logprobs -> weights -> sampling."""
    print("\n" + "=" * 60)
    print("Test 3: Full Pipeline Reproducibility")
    print("=" * 60)
    
    # Generate random logprobs (simulating model output)
    random.seed(42)  # Fix numpy/random seed for reproducible test data
    
    logprobs_sequence = []
    for _ in range(20):  # 20 token positions
        logprobs_sequence.append(generate_random_logprobs(vocab_size=100, top_k=5))
    
    temperature = 0.99
    seed = "full_pipeline_test"
    
    def sample_sequence_from_logprobs(
        logprobs_seq: List[Dict[str, float]],
        temp: float,
        seed_str: str,
    ) -> Tuple[List[int], List[Dict[str, int]]]:
        """Sample a sequence and return tokens + weights used."""
        rng = Sha256CounterRNG.from_seed_string(seed_str)
        tokens = []
        weights_used = []
        
        for logprobs in logprobs_seq:
            # Recompute weights from logprobs (same as vLLM does)
            weights = recompute_weights_from_logprobs(logprobs, temp)
            weights_used.append(weights)
            
            # Sort by token ID for deterministic ordering
            sorted_items = sorted(weights.items(), key=lambda x: int(x[0]))
            weight_list = [w for _, w in sorted_items]
            token_list = [int(t) for t, _ in sorted_items]
            
            # Sample
            idx = sample_categorical_weights(weight_list, rng)
            token = token_list[idx]
            tokens.append(token)
        
        return tokens, weights_used
    
    # Run multiple times
    n_runs = 5
    all_results = []
    
    for run in range(n_runs):
        tokens, weights = sample_sequence_from_logprobs(
            logprobs_sequence, temperature, seed
        )
        all_results.append(tokens)
        print(f"  Run {run + 1}: first 5 tokens = {tokens[:5]}")
    
    # All runs should be identical
    for i, result in enumerate(all_results[1:], 2):
        assert result == all_results[0], f"Run {i} differs from Run 1!"
    
    print(f"✓ All {n_runs} runs produced identical {len(all_results[0])}-token sequences")


def test_cross_seed_differs():
    """Test that different seeds produce different sequences."""
    print("\n" + "=" * 60)
    print("Test 4: Different Seeds Produce Different Sequences")
    print("=" * 60)
    
    weights = [50000, 10000, 5000, 500, 36]
    n_samples = 100
    
    results = {}
    for seed in ["seed_a", "seed_b", "seed_c"]:
        rng = Sha256CounterRNG.from_seed_string(seed)
        seq = [sample_categorical_weights(weights, rng) for _ in range(n_samples)]
        results[seed] = seq
    
    # All seeds should produce different sequences
    assert results["seed_a"] != results["seed_b"], "seed_a == seed_b"
    assert results["seed_b"] != results["seed_c"], "seed_b == seed_c"
    assert results["seed_a"] != results["seed_c"], "seed_a == seed_c"
    
    print("✓ Different seeds produce different sequences")


def test_reference_values():
    """
    Generate and verify reference values for cross-implementation testing.
    
    These exact values should be reproducible by any correct implementation.
    """
    print("\n" + "=" * 60)
    print("Test 5: Reference Values (for cross-implementation verification)")
    print("=" * 60)
    
    # Reference RNG values
    rng = Sha256CounterRNG.from_seed_string("reference_seed_v1")
    reference_u64s = [rng.next_u64() for _ in range(10)]
    
    print("  Reference u64 values for seed 'reference_seed_v1':")
    for i, v in enumerate(reference_u64s):
        print(f"    [{i}] = {v}")
    
    # Reference sampling
    weights = [60000, 5000, 500]
    rng = Sha256CounterRNG.from_seed_string("reference_sampling_v1")
    reference_samples = [sample_categorical_weights(weights, rng) for _ in range(10)]
    
    print(f"\n  Reference samples from weights {weights}:")
    print(f"    Samples = {reference_samples}")
    
    # Verify expected values (update these if algorithm changes)
    # These values are from the current Sha256CounterRNG implementation
    expected_u64s = [
        4286832458236889005,
        12281003819428572724,
        12352776571910749143,
        12178488218135958089,
        6205195570139478562,
        16961475390381133449,
        4266954777775371921,
        13066482787726221110,
        16734088885020042614,
        3747751605064727020,
    ]
    
    assert reference_u64s == expected_u64s, \
        f"Reference u64 values changed!\nExpected: {expected_u64s}\nGot: {reference_u64s}"
    
    print("\n✓ Reference values match expected (cross-implementation compatible)")


def test_weight_quantization_consistency():
    """Test that weight quantization is consistent."""
    print("\n" + "=" * 60)
    print("Test 6: Weight Quantization Consistency")
    print("=" * 60)
    
    # Test quantization
    probs = {"100": 0.6, "200": 0.3, "300": 0.1}
    weights = quantize_to_weights(probs)
    
    expected = {
        "100": round(0.6 * WEIGHT_SCALE),
        "200": round(0.3 * WEIGHT_SCALE),
        "300": round(0.1 * WEIGHT_SCALE),
    }
    
    assert weights == expected, f"Quantization mismatch: {weights} != {expected}"
    
    # Verify sum is approximately WEIGHT_SCALE
    total = sum(weights.values())
    assert abs(total - WEIGHT_SCALE) < 100, f"Weight sum {total} far from {WEIGHT_SCALE}"
    
    print(f"  Quantized weights: {weights}")
    print(f"  Sum: {total} (expected ~{WEIGHT_SCALE})")
    print("✓ Weight quantization is consistent")


def test_edge_cases():
    """Test edge cases in sampling."""
    print("\n" + "=" * 60)
    print("Test 7: Edge Cases")
    print("=" * 60)
    
    seed = "edge_case_test"
    
    # Single token (should always return 0)
    rng = Sha256CounterRNG.from_seed_string(seed)
    for _ in range(100):
        idx = sample_categorical_weights([65536], rng)
        assert idx == 0, "Single token should always return 0"
    print("✓ Single token always returns 0")
    
    # Dominant token (should almost always return 0)
    rng = Sha256CounterRNG.from_seed_string(seed)
    weights = [65530, 1, 1, 1, 1, 1, 1]  # 99.99% probability for first
    counts = [0] * len(weights)
    for _ in range(1000):
        idx = sample_categorical_weights(weights, rng)
        counts[idx] += 1
    
    # First token should dominate
    assert counts[0] > 990, f"Dominant token should appear >99% of time, got {counts[0]/10}%"
    print(f"✓ Dominant token (99.99%) appeared {counts[0]/10}% of time")
    
    # All zeros (should return last index)
    rng = Sha256CounterRNG.from_seed_string(seed)
    idx = sample_categorical_weights([0, 0, 0], rng)
    assert idx == 2, f"All-zero weights should return last index, got {idx}"
    print("✓ All-zero weights returns last index")


def main():
    print("=" * 60)
    print("Integer Weight Reproducibility Tests")
    print("=" * 60)
    print("\nThese tests verify that deterministic sampling produces")
    print("identical results across multiple runs with the same seed.")
    print("This is fundamental for cross-platform inference validation.\n")
    
    try:
        test_rng_reproducibility()
        test_weight_sampling_reproducibility()
        test_full_pipeline_reproducibility()
        test_cross_seed_differs()
        test_reference_values()
        test_weight_quantization_consistency()
        test_edge_cases()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nDeterministic sampling is working correctly.")
        print("The same seed + weights will produce identical samples")
        print("across any machine and any number of runs.")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
