# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Deterministic sampling utilities for cross-platform reproducibility.

This module provides:
1. A portable SHA256 counter-mode RNG that produces identical sequences
   across Python and Go implementations.
2. Integer-weight categorical sampling (cross-language reproducible).
3. A decimal-arithmetic pipeline that converts logprob strings to integer
   weights, guaranteeing bit-identical results on any CPython 3.3+ machine.

The decimal pipeline is the core of the deterministic sampling scheme:
  logprob strings -> Decimal temperature scaling -> Decimal softmax ->
  filtering (top_k, top_p, min_p) -> quantize to int weights (2^16 scale) ->
  sample with SHA256 RNG on integer weights.

See docs/DETERMINISTIC_SAMPLING_VALIDATION.md for full design.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from decimal import Decimal, getcontext, ROUND_HALF_EVEN
from typing import Dict, List, Optional, Sequence

# Configure decimal context for reproducible arithmetic.
# Precision=10 gives ~3 guard digits beyond float32's ~7 significant digits.
# ROUND_HALF_EVEN is the IEEE 754-2008 default (banker's rounding).
getcontext().prec = 10
getcontext().rounding = ROUND_HALF_EVEN

# Scale factor for integer weight quantization (2^16 = 65536).
WEIGHT_SCALE = 2 ** 16


# =============================================================================
# SHA256 Counter-Mode RNG
# =============================================================================

def _u64_be(x: int) -> bytes:
    return struct.pack(">Q", x & 0xFFFFFFFFFFFFFFFF)


@dataclass
class Sha256CounterRNG:
    """
    Portable, fully-specified RNG:
      u64 = first_8_bytes(SHA256(seed_bytes || counter_be_u64))

    Produces identical sequences in Python and Go given the same seed.
    The counter advances by 1 on each call to next_u64().
    """

    seed_bytes: bytes
    counter: int = 0

    @classmethod
    def from_seed_string(cls, seed: str) -> "Sha256CounterRNG":
        """Create RNG from a string seed (UTF-8 encoded)."""
        return cls(seed.encode("utf-8"), 0)

    def next_u64(self) -> int:
        """Return next pseudorandom uint64."""
        h = hashlib.sha256(self.seed_bytes + _u64_be(self.counter)).digest()
        self.counter += 1
        return int.from_bytes(h[:8], byteorder="big", signed=False)


def iter_u64(seed: str, count: int) -> List[int]:
    """
    Return a list of `count` u64 values from a fresh RNG seeded with `seed`.
    Useful for testing RNG reproducibility across implementations.
    """
    rng = Sha256CounterRNG.from_seed_string(seed)
    return [rng.next_u64() for _ in range(count)]


# =============================================================================
# Integer Sampling Primitives
# =============================================================================

def uint64_below(rng: Sha256CounterRNG, n: int) -> int:
    """
    Unbiased draw in [0, n) from 64-bit uniform values via rejection sampling.
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    two64 = 1 << 64
    limit = two64 - (two64 % n)  # largest multiple of n below 2^64
    while True:
        x = rng.next_u64()
        if x < limit:
            return x % n


def sample_categorical_weights(
    weights: Sequence[int], rng: Sha256CounterRNG
) -> int:
    """
    Deterministic categorical sampler on non-negative integer weights.

    Uses unbiased rejection sampling (uint64_below) then linear scan
    over the cumulative weight distribution. This is the primary sampling
    function used in the deterministic pipeline.

    Args:
        weights: Non-negative integer weights (must sum > 0).
        rng: SHA256 counter-mode RNG instance (state is advanced).

    Returns:
        Index of the sampled category.
    """
    if not weights:
        raise ValueError("weights is empty")
    total = 0
    last_nonzero = -1
    for i, w in enumerate(weights):
        if w < 0:
            raise ValueError(f"Negative weight at index {i}: {w}")
        if w > 0:
            last_nonzero = i
        total += w
    if total <= 0:
        return len(weights) - 1

    r = uint64_below(rng, total)
    cum = 0
    for i, w in enumerate(weights):
        cum += w
        if r < cum:
            return i
    return last_nonzero if last_nonzero >= 0 else len(weights) - 1



# =============================================================================
# Decimal Pipeline: logprob strings -> integer weights
# =============================================================================

def logprobs_to_weights(
    logprob_strings: Dict[str, str],
    temperature: str,
    top_p: Optional[str] = None,
    top_k: Optional[int] = None,
    min_p: Optional[str] = None,
) -> Dict[str, int]:
    """
    Deterministic logprobs -> integer weights pipeline.

    Both executor and validator call this with identical inputs.
    Produces bit-identical results on any machine running CPython 3.3+
    (backed by libmpdec with IEEE 754-2008 decimal arithmetic).

    All token iteration uses a fixed order: sorted by token ID string
    (lexicographic). This eliminates accumulation-order ambiguity in
    Decimal sums.

    Args:
        logprob_strings: {token_id_str: logprob_str} -- post-penalty logprobs
            as string values (e.g. {"791": "-0.05000000074505806"}).
        temperature: Temperature as string (e.g. "0.7"). Must be > 0.
        top_p: Optional nucleus sampling threshold as string (e.g. "0.9").
        top_k: Optional top-k filter count.
        min_p: Optional min-p threshold as string (e.g. "0.05").

    Returns:
        {token_id_str: int_weight} -- integer weights summing to exactly
        WEIGHT_SCALE (2^16 = 65536).
    """
    T = Decimal(temperature)
    sorted_tids = sorted(logprob_strings.keys())

    # Temperature scaling
    scaled = {tid: Decimal(logprob_strings[tid]) / T for tid in sorted_tids}

    # Softmax with log-sum-exp stability shift
    max_val = max(scaled[tid] for tid in sorted_tids)
    exps = {tid: (scaled[tid] - max_val).exp() for tid in sorted_tids}
    total_exp = sum(exps[tid] for tid in sorted_tids)
    probs = {tid: exps[tid] / total_exp for tid in sorted_tids}

    # top_k filtering
    if top_k is not None and top_k < len(sorted_tids):
        top_k_tids = sorted(
            sorted_tids, key=lambda t: probs[t], reverse=True
        )[:top_k]
        probs = {tid: probs[tid] for tid in top_k_tids}
        sorted_tids = sorted(top_k_tids)

    # top_p filtering
    if top_p is not None:
        tp = Decimal(top_p)
        sorted_by_prob = sorted(
            sorted_tids, key=lambda t: probs[t], reverse=True
        )
        cumsum = Decimal(0)
        kept: List[str] = []
        for tid in sorted_by_prob:
            cumsum += probs[tid]
            kept.append(tid)
            if cumsum >= tp:
                break
        probs = {tid: probs[tid] for tid in kept}
        sorted_tids = sorted(kept)

    # min_p filtering
    if min_p is not None:
        mp = Decimal(min_p)
        max_prob = max(probs[tid] for tid in sorted_tids)
        threshold = max_prob * mp
        kept = [tid for tid in sorted_tids if probs[tid] >= threshold]
        if not kept:
            kept = [max(sorted_tids, key=lambda t: probs[t])]
        probs = {tid: probs[tid] for tid in kept}
        sorted_tids = sorted(kept)

    # Re-normalize after filtering
    kept_total = sum(probs[tid] for tid in sorted_tids)
    norm_probs = {tid: probs[tid] / kept_total for tid in sorted_tids}

    # Quantize to integer weights
    D_SCALE = Decimal(WEIGHT_SCALE)
    weights = {
        tid: int((norm_probs[tid] * D_SCALE).to_integral_value())
        for tid in sorted_tids
    }

    # Fix total to exactly WEIGHT_SCALE (deterministic residual assignment).
    # Ties broken by token ID string (lexicographic) for determinism.
    residual = WEIGHT_SCALE - sum(weights.values())
    max_tid = max(sorted_tids, key=lambda t: (weights[t], t))
    weights[max_tid] += residual

    return weights


def decimal_sample_from_logprobs(
    logprob_strings: Dict[str, str],
    rng: Sha256CounterRNG,
    temperature: str,
    top_p: Optional[str] = None,
    top_k: Optional[int] = None,
    min_p: Optional[str] = None,
) -> str:
    """
    Full decimal pipeline + sample: logprob strings -> sampled token ID.

    Calls logprobs_to_weights() to derive integer weights, then
    sample_categorical_weights() to pick a token. Returns the sampled
    token ID as a string.

    Args:
        logprob_strings: {token_id_str: logprob_str}
        rng: SHA256 counter-mode RNG (state is advanced by one sample).
        temperature: Temperature as string (e.g. "0.7").
        top_p: Optional nucleus sampling threshold as string.
        top_k: Optional top-k filter count.
        min_p: Optional min-p threshold as string.

    Returns:
        Sampled token ID as string (e.g. "791").
    """
    weights = logprobs_to_weights(
        logprob_strings, temperature,
        top_p=top_p, top_k=top_k, min_p=min_p,
    )

    # Build parallel lists in deterministic order (sorted by token ID string)
    sorted_tids = sorted(weights.keys())
    weight_list = [weights[tid] for tid in sorted_tids]

    idx = sample_categorical_weights(weight_list, rng)
    return sorted_tids[idx]
