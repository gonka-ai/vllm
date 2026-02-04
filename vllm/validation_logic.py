# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Validation logic for decentralized inference networks.

This module implements the validation protocol ported from the Go implementation
in gonka/decentralized-api/internal/validation/inference_validation.go.

The validation has two stages:
1. Pre-model validation (Stage 1):
   a. Weight consistency: Verify sampling_weights match logprobs
   b. Sampling verification: Verify tokens were sampled correctly from weights

2. Post-model validation (Stage 2):
   - Distance calculation: Compare executor vs validator logprob distributions
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.validation import EnforcedToken, EnforcedTokens
    from vllm.entrypoints.openai.protocol import ValidationResult

# Constants matching Go implementation
DEFAULT_FRAUD_DISTANCE_THRESHOLD = 0.01
WEIGHT_SCALE = 65536  # 2^16 for integer weight quantization
EPSILON = 1e-10  # Small value to avoid division by zero


# =============================================================================
# Distance Calculation (Stage 2)
# =============================================================================

def position_distance(
    inf_lp: Dict[str, float],
    val_lp: Dict[str, float],
) -> float:
    """
    Calculate distance for a single position.
    
    Uses formula: |a - b| / (eps + |a| + |b|) / 2
    Summed across all tokens in the position.
    
    Args:
        inf_lp: Executor's logprobs for this position {token_id: logprob}
        val_lp: Validator's logprobs for this position {token_id: logprob}
    
    Returns:
        Distance metric for this position
    """
    if not inf_lp or not val_lp:
        return 0.0
    
    total_dist = 0.0
    
    # Process all tokens from executor's logprobs
    for token_id, inf_logprob in inf_lp.items():
        if token_id in val_lp:
            val_logprob = val_lp[token_id]
        else:
            # Estimate missing token as minimum of validator's values minus offset
            if val_lp:
                val_logprob = min(val_lp.values()) - 5.0
            else:
                val_logprob = inf_logprob
        
        # Distance formula: |a - b| / (eps + |a| + |b|) / 2
        diff = abs(inf_logprob - val_logprob)
        denom = EPSILON + abs(inf_logprob) + abs(val_logprob)
        total_dist += diff / denom / 2.0
    
    return total_dist


def compute_distance(
    inf_logprobs: List[Dict[str, float]],
    val_logprobs: List[Dict[str, float]],
) -> float:
    """
    Calculate normalized distance between two logprob sequences.
    
    Args:
        inf_logprobs: Executor's logprobs per position
        val_logprobs: Validator's logprobs per position
    
    Returns:
        Normalized distance (0.0 = identical, higher = more different)
        Returns 10.0 (max fraud) if sequence lengths don't match
    """
    if len(inf_logprobs) != len(val_logprobs):
        return 10.0  # Max fraud distance for length mismatch
    
    if not inf_logprobs:
        return 0.0
    
    # Sum distances across all positions
    total_dist = 0.0
    total_tokens = 0
    
    for inf_lp, val_lp in zip(inf_logprobs, val_logprobs):
        total_dist += position_distance(inf_lp, val_lp)
        total_tokens += len(inf_lp)
    
    # Normalization matching Go implementation:
    # (total_dist + 1.0) / (max(100, n_positions) * avg_tokens_per_pos + 1.0)
    n_positions = len(inf_logprobs)
    avg_tokens = total_tokens / n_positions if n_positions > 0 else 1
    
    normalized = (total_dist + 1.0) / (max(100, n_positions) * avg_tokens + 1.0)
    
    return normalized


# =============================================================================
# Weight Quantization and Recomputation
# =============================================================================

def quantize_to_weights(probs: Dict[str, float]) -> Dict[str, int]:
    """
    Quantize probabilities to integer weights at WEIGHT_SCALE.
    
    Args:
        probs: Probability distribution {token_id: probability}
    
    Returns:
        Integer weights {token_id: weight}
    """
    if not probs:
        return {}
    
    return {
        token_id: round(prob * WEIGHT_SCALE)
        for token_id, prob in probs.items()
    }


def recompute_weights_from_logprobs(
    logprobs: Dict[str, float],
    temperature: float,
) -> Dict[str, int]:
    """
    Recompute integer weights from logprobs using softmax with temperature.
    
    Args:
        logprobs: Raw logprobs {token_id: logprob}
        temperature: Temperature for softmax
    
    Returns:
        Integer weights {token_id: weight}
    """
    if not logprobs:
        return {}
    
    if len(logprobs) == 1:
        # Single token gets all weight
        token_id = list(logprobs.keys())[0]
        return {token_id: WEIGHT_SCALE}
    
    # Apply temperature and compute softmax
    # logprob = log(prob), so prob = exp(logprob)
    # With temperature: prob = exp(logprob / temperature)
    
    # For numerical stability, subtract max
    values = list(logprobs.values())
    max_lp = max(values)
    
    exp_values = {}
    for token_id, lp in logprobs.items():
        # Scale by temperature
        scaled = (lp - max_lp) / temperature
        exp_values[token_id] = math.exp(scaled)
    
    total = sum(exp_values.values())
    
    if total <= 0:
        # Fallback: uniform weights
        n = len(logprobs)
        return {token_id: WEIGHT_SCALE // n for token_id in logprobs}
    
    # Normalize to probabilities and quantize to weights
    weights = {}
    for token_id, exp_val in exp_values.items():
        prob = exp_val / total
        weights[token_id] = round(prob * WEIGHT_SCALE)
    
    return weights


# =============================================================================
# Sampling Verification (Stage 1b)
# =============================================================================

def verify_sampling_sequence(
    tokens: List["EnforcedToken"],
    seed_str: str,
) -> Tuple[bool, int]:
    """
    Verify that tokens were sampled correctly from their weights.
    
    Uses the deterministic Sha256CounterRNG to replay sampling and verify
    each token matches what would have been sampled.
    
    Args:
        tokens: List of EnforcedToken with sampling_weights
        seed_str: Seed string for the RNG
    
    Returns:
        Tuple of (success, failed_position)
        If success is True, failed_position is -1
        If success is False, failed_position is the first failing position
    """
    from vllm.v1.sample.deterministic_utils import (
        Sha256CounterRNG,
        sample_categorical_weights,
    )
    
    rng = Sha256CounterRNG.from_seed_string(seed_str)
    
    for pos, token in enumerate(tokens):
        # Skip tokens without sampling weights
        if token.sampling_weights is None:
            continue
        
        # Get sorted weights (consistent ordering)
        sorted_items = sorted(token.sampling_weights.items(), key=lambda x: int(x[0]))
        weight_list = [w for _, w in sorted_items]
        token_list = [t for t, _ in sorted_items]
        
        # Sample using RNG
        idx = sample_categorical_weights(weight_list, rng)
        expected_token = token_list[idx]
        
        # Compare
        if token.token != expected_token:
            return False, pos
    
    return True, -1


# =============================================================================
# Weight Consistency Verification (Stage 1a)
# =============================================================================

def verify_weights_consistency(
    tokens: List["EnforcedToken"],
    temperature: float,
    tolerance: float = 0.05,
) -> Tuple[bool, int]:
    """
    Verify that sampling_weights are consistent with logprobs.
    
    Recomputes weights from logprobs and compares against claimed weights.
    
    Args:
        tokens: List of EnforcedToken with logprobs and sampling_weights
        temperature: Temperature used for weight computation
        tolerance: Relative tolerance for weight comparison (default 5%)
    
    Returns:
        Tuple of (success, failed_position)
        If success is True, failed_position is -1
        If success is False, failed_position is the first failing position
    """
    for pos, token in enumerate(tokens):
        # Skip if missing either logprobs or weights
        if token.logprobs is None or token.sampling_weights is None:
            continue
        
        # Recompute weights from logprobs
        expected_weights = recompute_weights_from_logprobs(
            token.logprobs, temperature
        )
        
        # Compare weights
        for token_id, claimed_weight in token.sampling_weights.items():
            if token_id not in expected_weights:
                return False, pos
            
            expected = expected_weights[token_id]
            
            # Allow tolerance for rounding differences
            if expected == 0 and claimed_weight == 0:
                continue
            
            if expected == 0:
                # Expected zero but got non-zero
                if claimed_weight > 10:  # Small tolerance
                    return False, pos
                continue
            
            relative_diff = abs(claimed_weight - expected) / max(expected, 1)
            if relative_diff > tolerance:
                return False, pos
    
    return True, -1


# =============================================================================
# High-Level Validation Functions
# =============================================================================

def validate_before_model(
    artifact: "EnforcedTokens",
    seed_str: str,
    temperature: float,
) -> "ValidationResult":
    """
    Stage 1 validation (before running model).
    
    Verifies:
    1a. Weight consistency: sampling_weights match logprobs
    1b. Sampling verification: tokens were sampled correctly from weights
    
    Args:
        artifact: EnforcedTokens containing tokens with weights and logprobs
        seed_str: Seed string for deterministic sampling
        temperature: Temperature used for weight computation
    
    Returns:
        ValidationResult with fraud indicators set
    """
    from vllm.entrypoints.openai.protocol import ValidationResult
    
    result = ValidationResult()
    
    # Stage 1a: Verify weight consistency
    weights_ok, failed_pos = verify_weights_consistency(
        artifact.tokens, temperature
    )
    
    if not weights_ok:
        result.fraud = True
        result.correct_processed_logprobs = False
        result.distance = 10.0
        return result
    
    # Stage 1b: Verify sampling
    sampling_ok, failed_pos = verify_sampling_sequence(
        artifact.tokens, seed_str
    )
    
    if not sampling_ok:
        result.fraud = True
        result.correct_sampling = False
        result.distance = 10.0
        return result
    
    return result


def validate_after_model(
    executor_logprobs: List[Dict[str, float]],
    validator_logprobs: List[Dict[str, float]],
) -> "ValidationResult":
    """
    Stage 2 validation (after running model).
    
    Compares logprob distributions between executor and validator.
    
    Args:
        executor_logprobs: Logprobs from the executor's artifact
        validator_logprobs: Logprobs from validator's model run
    
    Returns:
        ValidationResult with distance and fraud indicators
    """
    from vllm.entrypoints.openai.protocol import ValidationResult
    
    result = ValidationResult()
    
    distance = compute_distance(executor_logprobs, validator_logprobs)
    result.distance = distance
    
    if distance > DEFAULT_FRAUD_DISTANCE_THRESHOLD:
        result.fraud = True
        result.correct_raw_logprobs = False
    
    return result


def validate_full(
    artifact: "EnforcedTokens",
    validator_logprobs: List[Dict[str, float]],
    seed_str: str,
    temperature: float,
) -> "ValidationResult":
    """
    Full two-stage validation.
    
    Runs both Stage 1 (pre-model) and Stage 2 (post-model) validation.
    
    Args:
        artifact: EnforcedTokens from executor
        validator_logprobs: Logprobs from validator's model run
        seed_str: Seed string for deterministic sampling
        temperature: Temperature used for weight computation
    
    Returns:
        ValidationResult with all fraud indicators
    """
    from vllm.entrypoints.openai.protocol import ValidationResult
    
    # Stage 1: Pre-model validation
    result = validate_before_model(artifact, seed_str, temperature)
    
    if result.fraud:
        return result
    
    # Stage 2: Post-model validation
    executor_logprobs = [
        t.logprobs for t in artifact.tokens
        if t.logprobs is not None
    ]
    
    stage2_result = validate_after_model(executor_logprobs, validator_logprobs)
    
    # Merge results
    result.distance = stage2_result.distance
    result.correct_raw_logprobs = stage2_result.correct_raw_logprobs
    
    if stage2_result.fraud:
        result.fraud = True
    
    return result
