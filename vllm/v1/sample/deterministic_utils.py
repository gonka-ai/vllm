# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Deterministic sampling utilities for cross-platform reproducibility.

This module provides a portable, fully-specified RNG based on SHA256
that produces identical sequences across Python and Go implementations.
"""

from __future__ import annotations

import bisect
import hashlib
import struct
from dataclasses import dataclass
from typing import List, Sequence


def _u64_be(x: int) -> bytes:
    return struct.pack(">Q", x & 0xFFFFFFFFFFFFFFFF)


@dataclass
class Sha256CounterRNG:
    """
    Portable, fully-specified RNG:
      u64 = first_8_bytes(SHA256(seed_bytes || counter_be_u64))
    """

    seed_bytes: bytes
    counter: int = 0

    @classmethod
    def from_seed_string(cls, seed: str) -> "Sha256CounterRNG":
        # Specify UTF-8 for portability.
        return cls(seed.encode("utf-8"), 0)

    def next_u64(self) -> int:
        h = hashlib.sha256(self.seed_bytes + _u64_be(self.counter)).digest()
        self.counter += 1
        return int.from_bytes(h[:8], byteorder="big", signed=False)

    def next_uniform01(self) -> float:
        """
        Uniform in [0,1) using top 53 bits (exactly representable in float64):
          u = (u64 >> 11) / 2^53
        """
        x = self.next_u64()
        return (x >> 11) * (1.0 / (1 << 53))


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


def sample_categorical_weights(weights: Sequence[int], rng: Sha256CounterRNG) -> int:
    """
    Deterministic categorical sampler on integer weights (recommended for
    cross-language reproducibility when probs may differ in float rounding).
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


@dataclass(frozen=True)
class WeightedPrefixSampler:
    """
    Fast categorical sampler for *fixed* non-negative integer weights.
    Build once: O(vocab). Sample: O(log vocab) via binary search on prefix sums.
    """

    prefix: List[int]  # strictly increasing at non-zero weights
    total: int
    last_nonzero: int

    @classmethod
    def from_weights(cls, weights: Sequence[int]) -> "WeightedPrefixSampler":
        if not weights:
            raise ValueError("weights is empty")
        prefix: List[int] = [0] * len(weights)
        total = 0
        last_nonzero = -1
        for i, w in enumerate(weights):
            if w < 0:
                raise ValueError(f"Negative weight at index {i}: {w}")
            if w > 0:
                last_nonzero = i
            total += w
            prefix[i] = total
        return cls(prefix=prefix, total=total, last_nonzero=last_nonzero)

    def sample(self, rng: Sha256CounterRNG) -> int:
        if self.total <= 0:
            return len(self.prefix) - 1
        r = uint64_below(rng, self.total)  # in [0,total)
        # find first i with prefix[i] > r
        i = bisect.bisect_right(self.prefix, r)
        if i >= len(self.prefix):
            return self.last_nonzero if self.last_nonzero >= 0 else len(self.prefix) - 1
        return i


def sample_categorical(probs: Sequence[float], rng: Sha256CounterRNG) -> int:
    """
    Deterministic categorical sampler.

    Requirements:
      - probs[i] >= 0
      - sum(probs) ~ 1 (tolerate small float error)
    """
    u = rng.next_uniform01()

    cum = 0.0
    last_nonzero = -1
    for i, p in enumerate(probs):
        if p < 0.0:
            raise ValueError(f"Negative probability at index {i}: {p}")
        if p > 0.0:
            last_nonzero = i
        cum += p
        if cum > u:
            return i

    # If u is very close to 1 or probs sum to slightly < 1 due to rounding,
    # return the last non-zero (or last index if all zeros).
    if last_nonzero >= 0:
        return last_nonzero
    return len(probs) - 1


def sample_sequence(
    probs_2d: Sequence[Sequence[float]], seed: str
) -> List[int]:
    """
    Sample one token per time step.
    probs_2d: shape [seq_len][vocab_size]
    """
    rng = Sha256CounterRNG.from_seed_string(seed)
    return [sample_categorical(step_probs, rng) for step_probs in probs_2d]
