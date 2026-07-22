# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-CPU tests for the sequence-level Stage-1 replay orchestrator
(``verify_sequence``) and the Stage-2 ``mae_distance`` metric — the contract-
faithful path that replaces the legacy ``validation_logic.validate_full``
(gonka-ai/gonka#1199 follow-up).

Scope: replay/aggregation logic only. The "reported" tokens are computed by the
same decimal pipeline the validator replays (self-consistent artifacts), so an
HONEST verdict proves the orchestration is correct, not that a real executor is
honest. No torch: ``verify_sequence`` reads only ``.token`` / ``.logprobs``, so a
minimal stand-in stands in for ``EnforcedToken``.
"""

from vllm.v1.sample.deterministic_utils import (
    Sha256CounterRNG,
    logprobs_to_weights,
    sample_categorical_weights,
)
from vllm.validation_sampling import (
    STAGE2_MAE_FRAUD_THRESHOLD,
    Verdict,
    _support_is_bounded,
    mae_distance,
    verify_sequence,
)

_BASE_SEED = "42|[1,2,3]"
_TEMP = "1.0"
_TOP_K = 20  # bounded support (> the 5 tokens per position, so clamps to all)

# Four positions of self-consistent per-position logprobs.
_POS = [
    {"5": -0.5, "10": -1.2, "2": -2.5, "100": -3.9, "7": -4.1},
    {"3": -0.3, "42": -1.1, "9": -2.2, "11": -3.0, "1": -4.5},
    {"8": -0.7, "6": -1.5, "4": -2.0, "20": -2.9, "15": -3.3},
    {"12": -0.9, "13": -1.8, "14": -2.4, "16": -3.1, "17": -3.7},
]


class _Tok:
    """Duck-typed stand-in for EnforcedToken (verify_sequence only reads
    ``.token`` and ``.logprobs``)."""

    def __init__(self, token, logprobs):
        self.token = token
        self.logprobs = logprobs


def _honest_token(logprobs, pos, *, top_p=None, top_k=_TOP_K, min_p=None):
    """The token an honest executor sharing the decimal path would report at
    ``pos`` — the same pipeline + per-position seed verify_sequence replays."""
    lp = {tid: repr(f) for tid, f in logprobs.items()}
    weights = logprobs_to_weights(lp, _TEMP, top_p=top_p, top_k=top_k, min_p=min_p)
    tids = sorted(weights)
    weight_list = [weights[t] for t in tids]
    rng = Sha256CounterRNG.from_seed_string(f"{_BASE_SEED}|{pos}")
    return tids[sample_categorical_weights(weight_list, rng)]


def _honest_artifact():
    return [_Tok(_honest_token(p, i), p) for i, p in enumerate(_POS)]


# --------------------------------------------------------------------------- #
# Stage-1: verify_sequence
# --------------------------------------------------------------------------- #

def test_sequence_accepts_honest():
    r = verify_sequence(_honest_artifact(), _BASE_SEED, _TEMP,
                        top_p=None, top_k=_TOP_K, min_p=None)
    assert r.verdict is Verdict.HONEST
    assert r.n_honest == len(_POS)
    assert r.fraud_position == -1


def test_sequence_flags_first_tampered_position():
    toks = _honest_artifact()
    toks[2] = _Tok("999999", _POS[2])  # a token the RNG would not have drawn
    r = verify_sequence(toks, _BASE_SEED, _TEMP,
                        top_p=None, top_k=_TOP_K, min_p=None)
    assert r.verdict is Verdict.FRAUD
    assert r.fraud_position == 2


def test_unbounded_support_is_inconclusive():
    # top_p only, no top_k / min_p -> support may exceed the signed top-K.
    r = verify_sequence(_honest_artifact(), _BASE_SEED, _TEMP,
                        top_p="0.9", top_k=None, min_p=None)
    assert r.verdict is Verdict.INCONCLUSIVE
    assert "unbounded" in r.reason


def test_greedy_is_inconclusive():
    r = verify_sequence(_honest_artifact(), _BASE_SEED, "0",
                        top_p=None, top_k=_TOP_K, min_p=None, greedy=True)
    assert r.verdict is Verdict.INCONCLUSIVE


def test_unsupported_version_is_inconclusive():
    r = verify_sequence(_honest_artifact(), _BASE_SEED, _TEMP,
                        top_p=None, top_k=_TOP_K, min_p=None,
                        contract_version="9.9.9")
    assert r.verdict is Verdict.INCONCLUSIVE


def test_missing_logprobs_position_is_inconclusive_not_fraud():
    toks = _honest_artifact()
    toks[1] = _Tok(toks[1].token, None)  # no replay data at this position
    r = verify_sequence(toks, _BASE_SEED, _TEMP,
                        top_p=None, top_k=_TOP_K, min_p=None)
    assert r.verdict is Verdict.HONEST  # >=1 honest; missing != fraud
    assert r.n_inconclusive == 1
    assert r.n_honest == len(_POS) - 1


def test_unsupported_seed_domain_is_inconclusive():
    # mirrors the Go validator's seed-domain gate (detsample.VerifyPosition).
    r = verify_sequence(_honest_artifact(), _BASE_SEED, _TEMP,
                        top_p=None, top_k=_TOP_K, min_p=None,
                        seed_domain="some-other-domain")
    assert r.verdict is Verdict.INCONCLUSIVE
    assert "seed domain" in r.reason


def test_zero_weights_raises_not_silent_fallback():
    # §6.9: a degenerate all-zero weight vector must raise, not silently return
    # the last index (matches the Go validator, which errors).
    import pytest
    with pytest.raises(ValueError):
        sample_categorical_weights([0, 0, 0], Sha256CounterRNG.from_seed_string("z"))


def test_support_boundedness_rule():
    assert _support_is_bounded(_TOP_K, None) is True
    assert _support_is_bounded(None, "0.02") is True
    assert _support_is_bounded(None, None) is False   # pure temperature
    assert _support_is_bounded(0, "0") is False       # disabled sentinels


# --------------------------------------------------------------------------- #
# Stage-2: mae_distance
# --------------------------------------------------------------------------- #

def test_mae_distance_identical_is_zero():
    assert mae_distance([_POS[0]], [_POS[0]]) == 0.0


def test_mae_distance_uniform_shift():
    shifted = {tid: v - 0.5 for tid, v in _POS[0].items()}
    assert abs(mae_distance([_POS[0]], [shifted]) - 0.5) < 1e-9


def test_mae_distance_length_mismatch_is_max():
    assert mae_distance([_POS[0], _POS[1]], [_POS[0]]) == 10.0


def test_mae_distance_missing_token_penalized():
    partial = dict(list(_POS[0].items())[:-1])  # validator missing one token
    assert mae_distance([_POS[0]], [partial]) > STAGE2_MAE_FRAUD_THRESHOLD
