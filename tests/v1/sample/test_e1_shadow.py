# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-CPU tests for the E1 shadow helper ``decimal_token_from_probs``
(gonka-ai/gonka#1199). It computes the token the decimal validator pipeline
would sample from a post-filter distribution, for comparison against the float
executor path. No torch needed."""

import math

from vllm.v1.sample.deterministic_utils import (
    Sha256CounterRNG,
    decimal_sample_from_logprobs,
    decimal_token_from_probs,
)

_PROBS = {5: 0.6, 10: 0.3, 2: 0.08, 100: 0.02}


def test_matches_decimal_pipeline_on_equivalent_logprobs():
    """The shadow token equals what the decimal pipeline draws from the
    equivalent logprobs (temperature 1, no filter) under the same RNG."""
    ids = list(_PROBS)
    vals = [_PROBS[t] for t in ids]
    seed = "e1shadow|0"

    shadow_tok = decimal_token_from_probs(
        ids, vals, Sha256CounterRNG.from_seed_string(seed))

    logprobs = {str(t): repr(math.log(p)) for t, p in _PROBS.items()}
    pipeline_tok = decimal_sample_from_logprobs(
        logprobs, Sha256CounterRNG.from_seed_string(seed), "1.0")

    assert shadow_tok == pipeline_tok


def test_deterministic():
    ids = list(_PROBS)
    vals = [_PROBS[t] for t in ids]
    a = decimal_token_from_probs(ids, vals, Sha256CounterRNG.from_seed_string("s"))
    b = decimal_token_from_probs(ids, vals, Sha256CounterRNG.from_seed_string("s"))
    assert a == b


def test_peaked_distribution_returns_peak():
    probs = {7: 0.999, 1: 0.001}
    tok = decimal_token_from_probs(
        list(probs), list(probs.values()),
        Sha256CounterRNG.from_seed_string("x"))
    assert tok == "7"


def test_zero_probability_support_is_dropped():
    # tokens with prob 0 (masked by top_k/top_p) must not be sampled.
    ids = [5, 10, 2, 0]
    vals = [0.6, 0.4, 0.0, 0.0]
    tok = decimal_token_from_probs(
        ids, vals, Sha256CounterRNG.from_seed_string("z"))
    assert tok in ("5", "10")
