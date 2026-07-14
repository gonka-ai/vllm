# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Deterministic sampling utilities for cross-platform reproducibility.

This module provides:
1. A portable, fully-specified RNG based on SHA256 that produces identical
   sequences across Python and Go implementations.
2. A decimal-arithmetic pipeline that converts logprob strings to integer
   weights, guaranteeing bit-identical results on any CPython 3.3+ machine.

The decimal context (prec=10, ROUND_HALF_EVEN) is applied *locally* inside
the pipeline functions via ``localcontext()`` -- importing this module does
NOT mutate the process-wide default Decimal context.
"""

from __future__ import annotations

import bisect
import hashlib
import struct
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from typing import Dict, List, Optional, Sequence

# Scale factor for integer weight quantization (2^16 = 65536).
WEIGHT_SCALE = 2**16


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


def iter_u64(seed: str, count: int) -> List[int]:
    """
    Return a list of `count` u64 values from a fresh RNG seeded with `seed`.
    """
    rng = Sha256CounterRNG.from_seed_string(seed)
    return [rng.next_u64() for _ in range(count)]


# =============================================================================
# Chain-bound seed derivation (from gonka-ai/vllm#56, @neuron7xLab)
# =============================================================================

# Versioned domain separator for chain-bound deterministic sampling seeds.
_SEED_DOMAIN_TAG = "gonka-deterministic-sampling-v1"

# Canonical contract bounds. These make the derivation reproduce identically
# across languages (Python executor, Go/Rust validator): the accept/reject
# boundary must not depend on any runtime's Unicode or integer semantics.
#
# Chain id: printable ASCII only (0x21..0x7E). This is deliberately stricter
# than "non-whitespace": a whitespace/`strip()` predicate rejects a different
# set of code points in Python vs Go vs JS (e.g. U+001C-1F, U+0085, U+00A0),
# which would split consensus on validity itself. Printable ASCII removes the
# whole ambiguity class and also rules out NUL/control bytes.
_ID_MIN_ORD = 0x21
_ID_MAX_ORD = 0x7E
_MAX_INFERENCE_ID_LEN = 256

# user_seed: pin to signed 64-bit range. A Python executor hashes an arbitrary
# big int fine, but a Go/Rust validator on int64 would overflow -> divergent
# or crashed validation. Pinning the range keeps the contract portable.
_MIN_USER_SEED = -(2**63)
_MAX_USER_SEED = 2**63 - 1


def derive_chain_bound_seed(
    user_seed: int,
    inference_id_from_chain: str,
) -> str:
    """Derive a chain-bound deterministic sampling seed.

    Stage-1 replay must be bound to chain provenance; prompt-derived seed
    material is request-controlled and must not be used for this path.

    Args:
        user_seed: Integer sampling seed (vLLM's public seed type is
            ``Optional[int]``; a str would collide with its int form).
            Must be within the signed 64-bit range for cross-language
            reproducibility.
        inference_id_from_chain: Chain-provided inference identifier.
            Must be printable ASCII (``0x21``..``0x7E``), non-empty, and at
            most ``_MAX_INFERENCE_ID_LEN`` characters.

    Returns:
        Lowercase SHA256 hex digest usable as a Sha256CounterRNG seed.

    Raises:
        ValueError: If inference_id_from_chain is missing, empty, too long,
            or contains a non-printable-ASCII character; or if user_seed is
            outside the signed 64-bit range.
        TypeError: If inference_id_from_chain is not a str, or if user_seed
            is not an exact int (bool and int subclasses excluded).
    """
    # Chain id must be canonical chain-provided text: reject None and any
    # non-str so arbitrary objects/bytes/ints are never stringified into
    # provenance material.
    if inference_id_from_chain is None:
        raise ValueError(
            "inference_id_from_chain is required for chain-bound "
            "deterministic sampling; refusing to fall back to "
            "request-controlled seed material."
        )
    if not isinstance(inference_id_from_chain, str):
        raise TypeError(
            "inference_id_from_chain must be a str (chain-provided text), "
            f"got {type(inference_id_from_chain).__name__}."
        )
    if inference_id_from_chain == "":
        raise ValueError("inference_id_from_chain must be non-empty.")
    if len(inference_id_from_chain) > _MAX_INFERENCE_ID_LEN:
        raise ValueError(
            "inference_id_from_chain too long "
            f"(>{_MAX_INFERENCE_ID_LEN} characters)."
        )
    # Language-invariant charset: reject anything outside printable ASCII so
    # the accept/reject boundary is identical in Python/Go/Rust/JS and no
    # control/NUL/whitespace/non-ASCII byte enters the hashed material.
    for ch in inference_id_from_chain:
        if not (_ID_MIN_ORD <= ord(ch) <= _ID_MAX_ORD):
            raise ValueError(
                "inference_id_from_chain must be printable ASCII "
                "(0x21..0x7E); no whitespace/control/NUL/non-ASCII. "
                f"Found U+{ord(ch):04X}."
            )

    # vLLM's seed is int-only (SamplingParams.seed / OpenAI seed are
    # Optional[int]). Require an EXACT int: `type(...) is int` rejects bool
    # (True == 1) and any int subclass whose `__str__` is overridden (which
    # would hash the repr, not the value). This also stops "7" colliding with
    # int 7 and any object being stringified into the seed.
    if type(user_seed) is not int:
        raise TypeError(
            "user_seed must be an exact int deterministic seed value "
            "(bool and int subclasses excluded), "
            f"got {type(user_seed).__name__}."
        )
    if not (_MIN_USER_SEED <= user_seed <= _MAX_USER_SEED):
        raise ValueError(
            "user_seed must be within the signed 64-bit range "
            "for cross-language reproducibility."
        )

    # Byte-length-prefixed framing over UTF-8 bytes: prefixes count BYTES (not
    # Python codepoints) so the contract reproduces in Go/Rust/JS. The chain id
    # is hashed byte-exact (no stripping) to preserve provenance.
    seed_bytes = bytes(str(user_seed), "utf-8")
    inference_id_bytes = bytes(inference_id_from_chain, "utf-8")
    material = b"".join([
        _SEED_DOMAIN_TAG.encode("ascii"),
        b"\nuser_seed_len=",
        str(len(seed_bytes)).encode("ascii"),
        b"\n",
        seed_bytes,
        b"\ninference_id_len=",
        str(len(inference_id_bytes)).encode("ascii"),
        b"\n",
        inference_id_bytes,
    ])
    return hashlib.sha256(material).hexdigest()


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


# =============================================================================
# Decimal Pipeline: logprob strings -> integer weights
#
# The decimal context is applied locally via ``localcontext()`` so importing
# this module never mutates the process-wide default. All Decimal
# arithmetic in the pipeline must run inside the ``with`` block; anything left
# outside would fall back to the default precision and break reproducibility.
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

    Both executor and validator call this with identical inputs. Produces
    bit-identical results on any machine running CPython 3.3+ (backed by
    libmpdec with IEEE 754-2008 decimal arithmetic).

    All token iteration uses a fixed order: sorted by token ID string
    (lexicographic). This eliminates accumulation-order ambiguity in Decimal
    sums.

    Note: the final weight list order and residual tie-break both rely on this
    lexicographic order. The executor production path must build its weight list
    in the *same* order for Check 2 to agree.

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
    with localcontext() as ctx:
        # prec=10 gives ~3 guard digits beyond float32's ~7 significant
        # digits; ROUND_HALF_EVEN is the IEEE 754-2008 default.
        ctx.prec = 10
        ctx.rounding = ROUND_HALF_EVEN

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
        d_scale = Decimal(WEIGHT_SCALE)
        weights = {
            tid: int((norm_probs[tid] * d_scale).to_integral_value())
            for tid in sorted_tids
        }

        # Fix total to exactly WEIGHT_SCALE (deterministic residual
        # assignment). Ties broken by token ID string (lexicographic).
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
    sample_categorical_weights() to pick a token. Returns the sampled token ID
    as a string.

    The weight list is built in lexicographic token-ID-string order; the
    returned index maps back through the same order.

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

    # Build parallel lists in deterministic order (sorted by token ID string).
    sorted_tids = sorted(weights.keys())
    weight_list = [weights[tid] for tid in sorted_tids]

    idx = sample_categorical_weights(weight_list, rng)
    return sorted_tids[idx]
