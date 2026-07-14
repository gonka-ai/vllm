# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Three-valued verdict for single-position replay (Python side of Go's
``detsample.VerifyPosition``).

A replay mismatch is only *fraud* when the validator could faithfully reproduce
the executor's computation. An unsupported contract version, a greedy position,
a non-positive temperature, or a validator-side replay error is *inconclusive*,
never fraud — so a validator-side or version problem does not punish an honest
executor. Mirrors ``verify.go`` clause-for-clause (gonka
decentralized-api/internal/validation/detsample/verify.go).

Scope: single position only. Sequence/response-level aggregation is intentionally
out of scope (not yet defined on the Go side either).

Pure CPU, no torch/GPU — loaded by path.
"""

from __future__ import annotations

import importlib.util
import os
import sys

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _load_by_path(mod_name: str, rel_path: str):
    path = os.path.join(_REPO_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_du = _load_by_path("deterministic_utils",
                    "vllm/v1/sample/deterministic_utils.py")
sys.modules["vllm.v1.sample.deterministic_utils"] = _du
_vs = _load_by_path("validation_sampling", "vllm/validation_sampling.py")


# An honest position: token "1" is what the pipeline samples for these
# logprobs / seed / temperature (computed with the reference pipeline).
_LOGPROBS = {"1": -0.5, "2": -1.2, "3": -2.0}
_SEED = "reference_seed_v1"
_TEMP = "1.0"
_HONEST_TOKEN = "1"
_SUPPORTED_VERSION = "1.0.0"


def _pos(**overrides):
    """Build a keyword dict for verify_position with honest defaults."""
    kw = dict(
        contract_version=_SUPPORTED_VERSION,
        logprobs=_LOGPROBS,
        seed_str=_SEED,
        temperature=_TEMP,
        top_p=None,
        top_k=None,
        min_p=None,
        reported_token=_HONEST_TOKEN,
        greedy=False,
    )
    kw.update(overrides)
    return kw


def test_honest_position_returns_honest():
    assert _vs.verify_position(**_pos()).verdict == _vs.Verdict.HONEST


def test_tampered_token_returns_fraud():
    assert _vs.verify_position(
        **_pos(reported_token="3")).verdict == _vs.Verdict.FRAUD


def test_unsupported_contract_version_is_inconclusive_not_fraud():
    v = _vs.verify_position(**_pos(contract_version="9.9.9",
                                   reported_token="3")).verdict
    assert v == _vs.Verdict.INCONCLUSIVE  # version skew, not fraud


def test_greedy_position_is_inconclusive():
    # temperature 0 / greedy bypasses the RNG (contract §7): no sequence signal.
    assert _vs.verify_position(
        **_pos(greedy=True)).verdict == _vs.Verdict.INCONCLUSIVE


def test_non_positive_temperature_without_greedy_is_inconclusive():
    # An inconsistent artifact (temp<=0 but greedy flag unset) is not fraud.
    assert _vs.verify_position(
        **_pos(temperature="0")).verdict == _vs.Verdict.INCONCLUSIVE
