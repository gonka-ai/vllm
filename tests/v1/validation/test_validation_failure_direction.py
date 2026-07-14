# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Failure direction of the serving-side validator (issue #1199, review Point 6).

When the validator itself cannot produce a verdict — a bug, an unexpected
exception, a version it cannot replay — it must NOT translate that into
"executor honest" (fraud=False). A validator-side failure yields *no verdict*,
never a silent pass. This mirrors detsample.VerifyPosition's principle that a
validator-side problem is inconclusive, never a fraud judgement about the
executor.

Tests the pure decision function, not the full serving stack (which needs torch
+ fastapi). Loaded by path.
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


def test_validator_error_yields_no_verdict_not_pass():
    """A validator-side error must yield None (no verdict), never a silent
    fraud=False pass."""
    err = RuntimeError("weights consistency check blew up")

    result = _vs.result_for_validator_error(err)

    assert result is None, (
        "a validator-side error must not be translated into a verdict; "
        "returning a fraud=False result silently passes the executor")
