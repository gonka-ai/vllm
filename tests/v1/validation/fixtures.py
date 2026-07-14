# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test fixtures and data generators for validation tests.

These fixtures provide synthetic data for testing the validation logic
without requiring a running vLLM server.
"""

import random
import math
from typing import List, Dict, Optional, Tuple

from vllm.validation import EnforcedToken, EnforcedTokens
from vllm.validation_logic import (
    recompute_weights_from_logprobs,
    quantize_to_weights,
    WEIGHT_SCALE,
)
from vllm.v1.sample.deterministic_utils import (
    Sha256CounterRNG,
    sample_categorical_weights,
)


def generate_random_logprobs(
    vocab_size: int = 32000,
    top_k: int = 5,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Generate random logprobs simulating model output.
    
    Args:
        vocab_size: Total vocabulary size for token ID generation
        top_k: Number of top tokens to include
        seed: Random seed for reproducibility
        
    Returns:
        Dict mapping token ID (str) to logprob
    """
    if seed is not None:
        random.seed(seed)
    
    # Select random token IDs
    token_ids = random.sample(range(vocab_size), top_k)
    
    # Generate logprobs (dominant token has highest prob)
    logprobs = {}
    
    # Dominant token: logprob between -0.1 and -2.0
    logprobs[str(token_ids[0])] = -random.uniform(0.1, 2.0)
    
    # Other tokens: progressively lower logprobs
    for i, tid in enumerate(token_ids[1:], 1):
        base = logprobs[str(token_ids[0])]
        logprobs[str(tid)] = base - random.uniform(1.0 * i, 3.0 * i)
    
    return logprobs


def generate_logprobs_sequence(
    n_positions: int,
    top_k: int = 5,
    seed: Optional[int] = None,
) -> List[Dict[str, float]]:
    """Generate a sequence of logprobs for multiple positions."""
    if seed is not None:
        random.seed(seed)
    
    return [
        generate_random_logprobs(top_k=top_k, seed=None)
        for _ in range(n_positions)
    ]


def build_artifact_from_logprobs(
    logprobs_sequence: List[Dict[str, float]],
    seed_str: str,
    temperature: float = 0.99,
) -> Tuple[EnforcedTokens, List[int]]:
    """
    Build an artifact from logprobs using deterministic sampling.
    
    Returns:
        Tuple of (EnforcedTokens artifact, list of sampled token IDs)
    """
    rng = Sha256CounterRNG.from_seed_string(seed_str)
    
    tokens = []
    sampled_ids = []
    
    for logprobs in logprobs_sequence:
        # Compute weights
        weights = recompute_weights_from_logprobs(logprobs, temperature)
        
        # Sort by token ID for deterministic ordering
        sorted_items = sorted(weights.items(), key=lambda x: int(x[0]))
        weight_list = [w for _, w in sorted_items]
        token_list = [t for t, _ in sorted_items]
        
        # Sample
        idx = sample_categorical_weights(weight_list, rng)
        sampled_token = token_list[idx]
        sampled_ids.append(int(sampled_token))
        
        tokens.append(EnforcedToken(
            token=sampled_token,
            top_tokens=list(logprobs.keys()),
            logprobs=logprobs,
            sampling_weights=weights,
        ))
    
    return EnforcedTokens(tokens=tokens), sampled_ids


def build_honest_artifact(
    prompt: str,
    seed: int,
    n_tokens: int = 10,
    temperature: float = 0.99,
    top_k: int = 5,
    logprobs_seed: Optional[int] = None,
) -> EnforcedTokens:
    """
    Build an honest artifact with correct sampling.
    
    This simulates what an honest executor would produce.
    
    Args:
        prompt: The prompt string
        seed: The user-provided seed
        n_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_k: Number of top logprobs per position
        logprobs_seed: Seed for generating synthetic logprobs
        
    Returns:
        EnforcedTokens artifact
    """
    seed_str = f"{seed}|{prompt}"
    logprobs_seq = generate_logprobs_sequence(n_tokens, top_k, logprobs_seed)
    artifact, _ = build_artifact_from_logprobs(logprobs_seq, seed_str, temperature)
    return artifact


def build_prefill_attack_artifact(
    prompt: str,
    seed: int,
    n_tokens: int = 10,
    temperature: float = 0.99,
    top_k: int = 5,
    logprobs_seed: Optional[int] = None,
) -> EnforcedTokens:
    """
    Build an artifact simulating a pre-fill attack.
    
    The tokens don't match what would be sampled from the logprobs/weights.
    
    Args:
        prompt: The prompt string
        seed: The user-provided seed
        n_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_k: Number of top logprobs per position
        logprobs_seed: Seed for generating synthetic logprobs
        
    Returns:
        EnforcedTokens artifact with wrong tokens
    """
    seed_str = f"{seed}|{prompt}"
    logprobs_seq = generate_logprobs_sequence(n_tokens, top_k, logprobs_seed)
    
    tokens = []
    for logprobs in logprobs_seq:
        weights = recompute_weights_from_logprobs(logprobs, temperature)
        
        # ATTACK: Pick the LEAST likely token (wrong choice)
        sorted_items = sorted(weights.items(), key=lambda x: x[1])
        wrong_token = sorted_items[0][0]  # Lowest weight token
        
        tokens.append(EnforcedToken(
            token=wrong_token,
            top_tokens=list(logprobs.keys()),
            logprobs=logprobs,
            sampling_weights=weights,
        ))
    
    return EnforcedTokens(tokens=tokens)


def build_tampered_weights_artifact(
    prompt: str,
    seed: int,
    n_tokens: int = 10,
    temperature: float = 0.99,
    top_k: int = 5,
    logprobs_seed: Optional[int] = None,
) -> EnforcedTokens:
    """
    Build an artifact with tampered weights.
    
    The sampling_weights are modified to be inconsistent with logprobs.
    
    Args:
        prompt: The prompt string
        seed: The user-provided seed
        n_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_k: Number of top logprobs per position
        logprobs_seed: Seed for generating synthetic logprobs
        
    Returns:
        EnforcedTokens artifact with tampered weights
    """
    seed_str = f"{seed}|{prompt}"
    rng = Sha256CounterRNG.from_seed_string(seed_str)
    logprobs_seq = generate_logprobs_sequence(n_tokens, top_k, logprobs_seed)
    
    tokens = []
    for logprobs in logprobs_seq:
        # Create fake weights (all weight on arbitrary token)
        token_ids = list(logprobs.keys())
        fake_weights = {tid: 0 for tid in token_ids}
        fake_weights[token_ids[-1]] = WEIGHT_SCALE  # All weight on last
        
        # Sample from fake weights
        sorted_items = sorted(fake_weights.items(), key=lambda x: int(x[0]))
        weight_list = [w for _, w in sorted_items]
        token_list = [t for t, _ in sorted_items]
        
        idx = sample_categorical_weights(weight_list, rng)
        sampled_token = token_list[idx]
        
        tokens.append(EnforcedToken(
            token=sampled_token,
            top_tokens=token_ids,
            logprobs=logprobs,
            sampling_weights=fake_weights,  # TAMPERED
        ))
    
    return EnforcedTokens(tokens=tokens)


def build_wrong_model_artifact(
    prompt: str,
    seed: int,
    n_tokens: int = 100,
    temperature: float = 0.99,
    top_k: int = 5,
    logprob_shift: float = 2.0,
) -> Tuple[EnforcedTokens, List[Dict[str, float]]]:
    """
    Build artifact simulating wrong model (for Stage 2 testing).
    
    Returns:
        Tuple of:
        - EnforcedTokens from "executor" (honest sampling from executor's logprobs)
        - Validator's logprobs (shifted to simulate different model)
    """
    # Generate executor's honest artifact
    honest_artifact = build_honest_artifact(
        prompt, seed, n_tokens, temperature, top_k
    )
    
    # Validator's logprobs are shifted (different model)
    validator_logprobs = []
    for token in honest_artifact.tokens:
        if token.logprobs:
            shifted = {k: v + logprob_shift for k, v in token.logprobs.items()}
            validator_logprobs.append(shifted)
    
    return honest_artifact, validator_logprobs


def perturb_logprobs(
    logprobs: Dict[str, float],
    noise_scale: float = 0.001,
) -> Dict[str, float]:
    """
    Add small perturbation to logprobs (simulating numerical precision).
    
    Args:
        logprobs: Original logprobs
        noise_scale: Scale of noise to add
        
    Returns:
        Perturbed logprobs
    """
    return {
        k: v + random.gauss(0, noise_scale)
        for k, v in logprobs.items()
    }


def simulate_cross_gpu_logprobs(
    logprobs_seq: List[Dict[str, float]],
    noise_scale: float = 0.0001,
) -> List[Dict[str, float]]:
    """
    Simulate logprobs from a different GPU (small numerical differences).
    
    Args:
        logprobs_seq: Original logprobs sequence
        noise_scale: Scale of noise to add
        
    Returns:
        Perturbed logprobs sequence (simulating different GPU)
    """
    return [perturb_logprobs(lp, noise_scale) for lp in logprobs_seq]


# Pytest fixtures

import pytest


@pytest.fixture
def honest_artifact():
    """Fixture providing an honest artifact."""
    return build_honest_artifact(
        prompt="Test prompt",
        seed=42,
        n_tokens=10,
        temperature=0.99,
        logprobs_seed=123,
    )


@pytest.fixture
def prefill_attack_artifact():
    """Fixture providing a pre-fill attack artifact."""
    return build_prefill_attack_artifact(
        prompt="Test prompt",
        seed=42,
        n_tokens=10,
        temperature=0.99,
        logprobs_seed=123,
    )


@pytest.fixture
def tampered_weights_artifact():
    """Fixture providing a tampered weights artifact."""
    return build_tampered_weights_artifact(
        prompt="Test prompt",
        seed=42,
        n_tokens=10,
        temperature=0.99,
        logprobs_seed=123,
    )


@pytest.fixture
def sample_logprobs_sequence():
    """Fixture providing a sample logprobs sequence."""
    return generate_logprobs_sequence(n_positions=20, top_k=5, seed=42)
