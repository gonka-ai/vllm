# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Drift guard for the deterministic-sampling cross-language contract.

Re-runs the reference Python pipeline over the committed conformance vectors and
asserts the outputs still match. If this fails, either the pipeline changed
(bump contract_version and regenerate) or a regression slipped in. The gonka Go
validator consumes the same JSON and must produce identical results.

Pure CPU, no torch/GPU — deterministic_utils is loaded by path.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_DU_PATH = os.path.join(_REPO_ROOT, "vllm", "v1", "sample", "deterministic_utils.py")
_VECTORS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "conformance_vectors.json")


def _load_du():
    spec = importlib.util.spec_from_file_location("deterministic_utils", _DU_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["deterministic_utils"] = mod
    spec.loader.exec_module(mod)
    return mod


du = _load_du()

with open(_VECTORS) as fh:
    VECTORS = json.load(fh)

CASES = VECTORS["cases"]


def test_weight_scale_matches_contract():
    assert VECTORS["weight_scale"] == du.WEIGHT_SCALE == 65536


def test_uint64_below_vectors():
    ub = VECTORS["uint64_below"]
    for case in ub["cases"]:
        rng = du.Sha256CounterRNG.from_seed_string(ub["seed"])
        got = [du.uint64_below(rng, case["n"]) for _ in case["draws"]]
        assert got == case["draws"], case["n"]


def test_rng_reference_vector():
    ref = VECTORS["rng_reference"]
    assert du.iter_u64(ref["seed"], len(ref["first_u64"])) == ref["first_u64"]
    # The pinned reference value from the contract (§5).
    assert ref["first_u64"][0] == 4286832458236889005


def test_float_to_string_canonicalization():
    import struct
    f32_neg005 = struct.unpack("f", struct.pack("f", -0.05))[0]
    assert repr(-0.05) == "-0.05"
    assert repr(f32_neg005) == "-0.05000000074505806"


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_case_reproduces(case):
    logprobs = case["logprobs"]
    kw = dict(top_p=case["top_p"], top_k=case["top_k"], min_p=case["min_p"])

    weights = du.logprobs_to_weights(logprobs, case["temperature"], **kw)
    weights = {tid: int(w) for tid, w in weights.items()}
    assert weights == case["expected_weights"], case["name"]
    assert sum(weights.values()) == case["expected_weight_sum"] == 65536

    rng = du.Sha256CounterRNG.from_seed_string(case["seed_str"])
    token = du.decimal_sample_from_logprobs(logprobs, rng, case["temperature"], **kw)
    assert token == case["expected_token"], case["name"]


def test_seed_derivation_accept_vectors():
    # The committed accept digests must still reproduce through
    # derive_chain_bound_seed, and the domain tag must match the impl. The Go
    # validator consumes the same block and must derive identical seeds.
    sd = VECTORS["seed_derivation"]
    assert sd["domain_tag"] == du._SEED_DOMAIN_TAG
    for case in sd["accept"]:
        got = du.derive_chain_bound_seed(case["user_seed"], case["inference_id"])
        assert got == case["expected_seed"], case["inference_id"]


def test_seed_derivation_reject_vectors():
    # Every inference_id the contract marks invalid must fail closed (never
    # silently derive a seed) on the Python side too.
    sd = VECTORS["seed_derivation"]
    for case in sd["reject_inference_id"]:
        with pytest.raises(ValueError):
            du.derive_chain_bound_seed(7, case["inference_id"])


def test_reproducible_across_runs():
    # Same seed twice -> identical token (determinism, not just fixture match).
    c = CASES[0]
    kw = dict(top_p=c["top_p"], top_k=c["top_k"], min_p=c["min_p"])
    a = du.decimal_sample_from_logprobs(
        c["logprobs"], du.Sha256CounterRNG.from_seed_string(c["seed_str"]),
        c["temperature"], **kw)
    b = du.decimal_sample_from_logprobs(
        c["logprobs"], du.Sha256CounterRNG.from_seed_string(c["seed_str"]),
        c["temperature"], **kw)
    assert a == b
