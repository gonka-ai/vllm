# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-CPU smoke test for Check 2 sampling replay (no torch).

Asserts that ``verify_sampling_from_logprobs`` correctly accepts a self-
consistent position and rejects a tampered one.

Scope:
- This proves the *replay logic* is correct, NOT that the executor is honest
  (that needs Check 1 / GPU), and NOT that it accepts a real v011 executor
  artifact (the production weight path is not yet reproducible). The artifact
  here is self-consistent: the "reported" token is computed by the same decimal
  pipeline the validator replays.
"""

from vllm.v1.sample.deterministic_utils import (
    Sha256CounterRNG,
    decimal_sample_from_logprobs,
)
from vllm.validation_sampling import verify_sampling_from_logprobs

# A position's logprobs as JSON floats (as they live on EnforcedToken.logprobs).
_LOGPROBS = {
    "791": -0.05000000074505806,
    "1": -3.0,
    "2": -3.5,
    "10": -4.0,
}
_SEED = "42|[1,2,3]"
_TEMPERATURE = "1.0"


def _self_consistent_token() -> str:
    """The token the validator's own pipeline draws for this position -- i.e.
    what an honest executor sharing the decimal path would report."""
    logprob_strings = {tid: repr(f) for tid, f in _LOGPROBS.items()}
    return decimal_sample_from_logprobs(
        logprob_strings,
        Sha256CounterRNG.from_seed_string(_SEED),
        _TEMPERATURE,
    )


def test_replay_accepts_self_consistent_token():
    honest_token = _self_consistent_token()
    assert verify_sampling_from_logprobs(
        _LOGPROBS, _SEED, _TEMPERATURE,
        top_p=None, top_k=None, min_p=None,
        reported_token=honest_token,
    ) is True


def test_replay_rejects_tampered_token():
    honest_token = _self_consistent_token()
    tampered = next(t for t in _LOGPROBS if t != honest_token)
    assert verify_sampling_from_logprobs(
        _LOGPROBS, _SEED, _TEMPERATURE,
        top_p=None, top_k=None, min_p=None,
        reported_token=tampered,
    ) is False
