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
from typing import Dict, List, Optional

from vllm.v1.sample.deterministic_utils import (
    Sha256CounterRNG,
    logprobs_to_weights,
    sample_categorical_weights,
)

# The contract version this validator can faithfully replay. An artifact
# declaring anything else is Inconclusive (version-unsupported), not Fraud.
# Mirrors detsample.SupportedContractVersion (contract §0).
SUPPORTED_CONTRACT_VERSION = "1.0.0"

# The seed-derivation domain this validator can replay. An artifact declaring a
# different domain is Inconclusive (version-unsupported), not Fraud. Must match
# deterministic_utils._SEED_DOMAIN_TAG and detsample.SupportedSeedDomain (Go);
# the conformance vectors' seed_derivation.domain_tag pins it, so drift is caught.
SUPPORTED_SEED_DOMAIN = "gonka-deterministic-sampling-v1"


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
    seed_domain: str = SUPPORTED_SEED_DOMAIN,
    greedy: bool = False,
) -> PositionResult:
    """Replay one artifact position and classify it (Honest/Fraud/Inconclusive).

    Version gating and the greedy/temperature exemptions run *before* any fraud
    verdict, clause-for-clause with detsample.VerifyPosition (verify.go):

    1. contract-version mismatch -> Inconclusive (version-unsupported, §0)
    2. seed-domain mismatch      -> Inconclusive (version-unsupported, §8)
    3. greedy (temperature 0)    -> Inconclusive (§7: argmax bypasses the RNG,
       so the sequence check carries no signal)
    4. non-positive/unparseable temperature with greedy unset -> Inconclusive
       (an inconsistent artifact, not fraud)
    5. replay error              -> Inconclusive (validator-side inability)
    6. replay ok, token differs  -> Fraud (zero tolerance)
    7. token matches             -> Honest

    Scope: single position. Sequence/response-level aggregation is out of scope
    (not defined on the Go side yet).
    """
    if contract_version != SUPPORTED_CONTRACT_VERSION:
        return _inconclusive(
            f"unsupported contract version {contract_version!r} "
            f"(validator supports {SUPPORTED_CONTRACT_VERSION!r})")

    if seed_domain != SUPPORTED_SEED_DOMAIN:
        return _inconclusive(
            f"unsupported seed domain {seed_domain!r} "
            f"(validator supports {SUPPORTED_SEED_DOMAIN!r})")

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


# =============================================================================
# Sequence-level orchestration (Stage-1 replay) + Stage-2 distance
#
# Converges the end-to-end validation onto this contract-faithful path, retiring
# the legacy recompute path in ``validation_logic.py`` (gonka-ai/gonka#1199
# follow-up items #1/#2/#3):
#   - Stage-1 replays the SIGNED logprobs per position through the decimal
#     pipeline (it never recomputes weights with an unfiltered float softmax).
#   - RNG semantics: one seed per position (Decision B = per-position), so the
#     variable draw-count of rejection sampling at one position cannot desync the
#     replay of the rest of the sequence.
#   - Stage-2 distance is MAE over the signed top-K support (Experiment 4), not
#     the legacy relative-diff ratio.
#   - Unbounded-support requests (no top_k and no min_p) carry no cheap Stage-1
#     signal and are Inconclusive, deferred to Stage-2 (Experiment 2).
# =============================================================================

# Stage-2 fraud threshold on ``mae_distance``. PLACEHOLDER pending Decision D
# (#1199): Experiments 4/5 put the honest floor near ~0.01 MAE and a clearly
# different model near ~0.4-0.8, with int8 quantization a gray zone around
# ~0.1-0.2. 0.25 accepts int8 as "the model" while still catching int4 and
# cheaper models. The final value — and whether int8 counts as the model — is a
# policy decision, NOT a measurement. Do not treat this as final/enforcing.
STAGE2_MAE_FRAUD_THRESHOLD = 0.25


@dataclass
class SequenceResult:
    """Aggregate verdict over a whole response."""

    verdict: Verdict
    fraud_position: int = -1
    n_honest: int = 0
    n_inconclusive: int = 0
    reason: str = ""


def _support_is_bounded(top_k: Optional[int], min_p: Optional[str]) -> bool:
    """A request has bounded support iff ``top_k`` or ``min_p`` is active.

    ``top_p`` alone (especially at high temperature) and pure-temperature
    sampling have unbounded support — the nucleus can exceed the signed top-K, so
    the reported set cannot faithfully reproduce the filter (Experiment 2). Such
    requests get no cheap Stage-1 signal and are deferred to Stage-2.
    """
    if top_k is not None and top_k > 0:
        return True
    if min_p is not None:
        try:
            return float(min_p) > 0.0
        except (TypeError, ValueError):
            return False
    return False


def verify_sequence(
    tokens: List["EnforcedToken"],  # noqa: F821 — avoid importing torch-side model
    base_seed_str: str,
    temperature: str,
    top_p: Optional[str],
    top_k: Optional[int],
    min_p: Optional[str],
    *,
    contract_version: str = SUPPORTED_CONTRACT_VERSION,
    seed_domain: str = SUPPORTED_SEED_DOMAIN,
    greedy: bool = False,
) -> SequenceResult:
    """Replay a whole response position-by-position and aggregate the verdict.

    Aggregation (zero tolerance):
      - any position FRAUD  -> FRAUD (reports the first such position)
      - else >= 1 HONEST    -> HONEST (Inconclusive positions defer to Stage-2)
      - else all Inconclusive -> INCONCLUSIVE (Stage-1 carries no signal)

    The per-position seed is ``f"{base_seed_str}|{pos}"`` (Decision B resolved to
    per-position: independent, O(1) replay, and immune to rejection-sampling
    draw-count desync across positions). Each ``token`` is an ``EnforcedToken``
    with ``.logprobs`` (signed) and ``.token`` (reported); ``top_p/top_k/min_p``
    are the resolved request-level params.
    """
    if contract_version != SUPPORTED_CONTRACT_VERSION:
        return SequenceResult(
            Verdict.INCONCLUSIVE,
            reason=f"unsupported contract version {contract_version!r}")
    if seed_domain != SUPPORTED_SEED_DOMAIN:
        return SequenceResult(
            Verdict.INCONCLUSIVE,
            reason=f"unsupported seed domain {seed_domain!r}")
    if greedy:
        return SequenceResult(
            Verdict.INCONCLUSIVE,
            reason="greedy (temperature 0): sequence check not applicable")
    if not _support_is_bounded(top_k, min_p):
        return SequenceResult(
            Verdict.INCONCLUSIVE,
            reason="unbounded support (no top_k/min_p): deferred to distance check")

    n_honest = 0
    n_inconclusive = 0
    for pos, token in enumerate(tokens):
        if token.logprobs is None:
            n_inconclusive += 1
            continue
        seed_pos = f"{base_seed_str}|{pos}"
        pr = verify_position(
            token.logprobs, seed_pos, temperature,
            top_p=top_p, top_k=top_k, min_p=min_p,
            reported_token=token.token,
            contract_version=contract_version,
            seed_domain=seed_domain,
        )
        if pr.verdict is Verdict.FRAUD:
            return SequenceResult(
                Verdict.FRAUD, fraud_position=pos,
                n_honest=n_honest, n_inconclusive=n_inconclusive,
                reason=pr.reason)
        if pr.verdict is Verdict.HONEST:
            n_honest += 1
        else:
            n_inconclusive += 1

    if n_honest > 0:
        return SequenceResult(
            Verdict.HONEST, n_honest=n_honest, n_inconclusive=n_inconclusive)
    return SequenceResult(
        Verdict.INCONCLUSIVE, n_inconclusive=n_inconclusive,
        reason="no replayable position (all inconclusive)")


def mae_distance(
    executor_logprobs: List[Dict[str, float]],
    validator_logprobs: List[Dict[str, float]],
) -> float:
    """Stage-2 distance: mean over positions of the mean absolute logprob
    difference over the executor's reported top-K support.

    Replaces the legacy relative-diff ratio (``validation_logic.compute_distance``).
    Experiment 4 (#1199 follow-up) showed MAE over the top-K support separates
    honest from a wrong model by ~40-65x, while the sampled-token delta overlaps
    and cannot gate. A token missing on the validator side is charged a large
    penalty so a truncated/mismatched distribution reads as distant.
    """
    if not executor_logprobs:
        return 0.0
    if len(executor_logprobs) != len(validator_logprobs):
        return 10.0  # length mismatch -> maximally distant

    penalty = 10.0
    per_position = []
    for exec_lp, val_lp in zip(executor_logprobs, validator_logprobs):
        if not exec_lp:
            continue
        # Sum over sorted token IDs so the float accumulation order is identical
        # to the Go validator's (byte-exact cross-language distance).
        diffs = [
            abs(exec_lp[tid] - val_lp[tid]) if tid in val_lp else penalty
            for tid in sorted(exec_lp)
        ]
        if diffs:
            per_position.append(sum(diffs) / len(diffs))

    if not per_position:
        return 0.0
    return sum(per_position) / len(per_position)
