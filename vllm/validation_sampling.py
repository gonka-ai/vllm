# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Check 2 -- Sampling Replay (pure CPU, no torch).

The validator replays the executor's per-position sampling from the reported
logprobs and compares against the reported token. Zero tolerance: any mismatch
is fraud. Background and contract: gonka-ai/gonka#1199.

Scope:
- This module implements Check 2 only. Full honesty also requires Check 1
  (logprob distance), which re-runs the model on GPU and is out of scope here.
- ``verify_sampling_from_logprobs`` verifies a *single* position and returns a
  bool. The sequence-level loop belongs to the serving-layer orchestrator, not
  here.
- The seed is passed in already-composed as ``seed_str``; this module does not
  derive it. Seed hardening is a separate concern.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from vllm.v1.sample.deterministic_utils import (
    Sha256CounterRNG,
    logprobs_to_weights,
    sample_categorical_weights,
)

# The contract version this validator can faithfully replay. An artifact
# declaring anything else is Inconclusive (version-unsupported), not Fraud.
# Mirrors detsample.SupportedContractVersion (contract §0).
SUPPORTED_CONTRACT_VERSION = "1.0.0"


def verify_sampling_from_logprobs(
    logprobs: Dict[str, float],
    seed_str: str,
    temperature: str,
    top_p: Optional[str],
    top_k: Optional[int],
    min_p: Optional[str],
    reported_token: str,
) -> bool:
    """Replay Check 2 for a single token position.

    Args:
        logprobs: {token_id_str: float} -- the executor's post-penalty logprobs
            for this position (matches ``EnforcedToken.logprobs``). Converted to
            canonical decimal strings via ``repr(f)`` before entering the
            pipeline (the single float->string conversion point).
            ``Decimal(repr(f))`` is the documented canonicalization.
        seed_str: The already-composed RNG seed string (this function does not
            derive it).
        temperature: Temperature as string (e.g. "0.7"). Must be > 0.
        top_p: Optional nucleus sampling threshold as string.
        top_k: Optional top-k filter count.
        min_p: Optional min-p threshold as string.
        reported_token: The token ID string the executor claims it sampled.

    Returns:
        True if the replayed token matches ``reported_token`` (honest), else
        False (fraud). Zero tolerance.

    Note:
        A True result means the sampling step is consistent with *these*
        logprobs -- it does not prove the logprobs themselves are what the model
        produced (that is Check 1, §8 U8), nor that the executor's production
        weight path is reproducible (§8 U11/U12).
    """
    # Single float->string conversion point (§4). repr() is CPython's
    # shortest round-tripping representation.
    logprob_strings = {tid: repr(f) for tid, f in logprobs.items()}

    weights = logprobs_to_weights(
        logprob_strings, temperature,
        top_p=top_p, top_k=top_k, min_p=min_p,
    )

    # Weight list built in lexicographic token-ID-string order; the returned
    # index maps back through the same order (§8 U12).
    sorted_tids = sorted(weights.keys())
    weight_list = [weights[tid] for tid in sorted_tids]

    rng = Sha256CounterRNG.from_seed_string(seed_str)
    idx = sample_categorical_weights(weight_list, rng)
    replayed_token = sorted_tids[idx]

    return replayed_token == reported_token


class Verdict(str, Enum):
    """Classified outcome of a single-position replay.

    Mirrors detsample.Verdict (verify.go). A replay mismatch is only FRAUD when
    the validator could faithfully reproduce the executor's computation; an
    unsupported version, a greedy position, or a validator-side replay error is
    INCONCLUSIVE, never FRAUD.
    """

    HONEST = "honest"
    FRAUD = "fraud"
    INCONCLUSIVE = "inconclusive"


@dataclass
class PositionResult:
    """A verdict plus a human-readable reason (empty for HONEST)."""

    verdict: Verdict
    reason: str = ""


def _inconclusive(reason: str) -> PositionResult:
    return PositionResult(verdict=Verdict.INCONCLUSIVE, reason=reason)


def verify_position(
    logprobs: Dict[str, float],
    seed_str: str,
    temperature: str,
    top_p: Optional[str],
    top_k: Optional[int],
    min_p: Optional[str],
    reported_token: str,
    *,
    contract_version: str = SUPPORTED_CONTRACT_VERSION,
    greedy: bool = False,
) -> PositionResult:
    """Replay one artifact position and classify it (Honest/Fraud/Inconclusive).

    Version gating and the greedy/temperature exemptions run *before* any fraud
    verdict, clause-for-clause with detsample.VerifyPosition (verify.go):

    1. contract-version mismatch -> Inconclusive (version-unsupported, §0)
    2. greedy (temperature 0)    -> Inconclusive (§7: argmax bypasses the RNG,
       so the sequence check carries no signal)
    3. non-positive/unparseable temperature with greedy unset -> Inconclusive
       (an inconsistent artifact, not fraud)
    4. replay error              -> Inconclusive (validator-side inability)
    5. replay ok, token differs  -> Fraud (zero tolerance)
    6. token matches             -> Honest

    Scope: single position. Sequence/response-level aggregation is out of scope
    (not defined on the Go side yet).
    """
    if contract_version != SUPPORTED_CONTRACT_VERSION:
        return _inconclusive(
            f"unsupported contract version {contract_version!r} "
            f"(validator supports {SUPPORTED_CONTRACT_VERSION!r})")

    if greedy:
        return _inconclusive(
            "greedy position (temperature 0): sequence check not applicable")

    try:
        temp_val = float(temperature)
    except (TypeError, ValueError):
        return _inconclusive(
            f"unparseable temperature {temperature!r} with greedy flag unset")
    if temp_val <= 0:
        return _inconclusive(
            f"non-positive temperature {temperature!r} with greedy flag unset")

    try:
        honest = verify_sampling_from_logprobs(
            logprobs, seed_str, temperature,
            top_p=top_p, top_k=top_k, min_p=min_p,
            reported_token=reported_token,
        )
    except Exception as e:  # noqa: BLE001 — any replay error is validator-side
        return _inconclusive(f"replay error: {e}")

    if not honest:
        return PositionResult(
            verdict=Verdict.FRAUD,
            reason=f"replayed token differs from reported {reported_token!r}")
    return PositionResult(verdict=Verdict.HONEST)


def result_for_validator_error(error: BaseException) -> None:
    """A validator-side error yields no verdict (None), never a silent
    honest/fraud judgement about the executor. Mirrors detsample.VerifyPosition
    treating a validator-side problem as inconclusive. Always None; `error` is
    for the caller to log."""
    return None
