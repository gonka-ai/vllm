# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Regression test for U12 — token iteration order (contract §3).

Two Python validators must agree on the same artifact:
- ``validation_sampling.py`` sorts the weight list by token-ID **string**
  (lexicographic, contract §3 — the correct one).
- ``validation_logic.py::verify_sampling_sequence`` historically sorted by
  **numeric** token id, so ``"10" < "2"`` lexicographically but ``2 < 10``
  numerically. The two orderings map the same RNG draw to different tokens and
  false-reject an honest executor.

Contract §3 pins lexicographic string order; the numeric-sort path must match.

Pure CPU, no torch/GPU — modules are loaded by path.
"""

from __future__ import annotations

import importlib.util
import os
import sys

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _load_by_path(mod_name: str, rel_path: str):
    """Load a stdlib-only module by file path, bypassing ``import vllm``
    (which pulls torch). Registers in sys.modules first so dataclasses in the
    module resolve their own module namespace."""
    path = os.path.join(_REPO_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# deterministic_utils must be importable as its bare name because
# validation_logic imports it as `from vllm.v1.sample.deterministic_utils`.
# We shim that import path by pre-registering the by-path module under the
# fully-qualified name the function uses.
_du = _load_by_path("deterministic_utils",
                    "vllm/v1/sample/deterministic_utils.py")
sys.modules["vllm.v1.sample.deterministic_utils"] = _du

_vl = _load_by_path("validation_logic", "vllm/validation_logic.py")


class _StubToken:
    """Minimal stand-in for EnforcedToken: only the two attributes
    verify_sampling_sequence touches."""

    def __init__(self, token: str, sampling_weights):
        self.token = token
        self.sampling_weights = sampling_weights


# Chosen so lexicographic and numeric order diverge:
#   lexicographic ("10" < "2") -> RNG draws token "2"
#   numeric       (2 < 10)     -> RNG draws token "10"
_SEED = "42|1,2,3"
_WEIGHTS = {"2": 30000, "10": 35536}  # sums to 65536
_HONEST_TOKEN = "2"  # what the contract's lexicographic order actually samples


def test_verify_sampling_sequence_accepts_lexicographically_honest_artifact():
    """An artifact whose token was sampled under contract §3 (lexicographic
    string order) must verify as honest. Under the legacy numeric sort it
    false-rejects."""
    tokens = [_StubToken(_HONEST_TOKEN, _WEIGHTS)]

    success, failed_pos = _vl.verify_sampling_sequence(tokens, _SEED)

    assert success is True, (
        f"honest lexicographic artifact false-rejected at pos {failed_pos}; "
        "verify_sampling_sequence is not using contract §3 string order")
    assert failed_pos == -1


def test_verify_sampling_sequence_still_rejects_tampered_token():
    """The order fix must not weaken detection: a token that was NOT the one
    sampled under contract §3 order must still be flagged as fraud."""
    # Under lexicographic order the honest token is "2"; claiming "10" is a
    # tampered artifact.
    tokens = [_StubToken("10", _WEIGHTS)]

    success, failed_pos = _vl.verify_sampling_sequence(tokens, _SEED)

    assert success is False
    assert failed_pos == 0
