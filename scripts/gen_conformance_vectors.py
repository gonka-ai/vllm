# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Generate the cross-language conformance vectors for deterministic sampling.

These vectors are the executable form of the deterministic-sampling contract
(gonka-ai/gonka#1199). The vLLM Python pipeline and the gonka Go validator must
both reproduce them bit-for-bit. Run:

    python scripts/gen_conformance_vectors.py

Writes tests/v1/validation/conformance_vectors.json (deterministic output;
diff-review it like any other checked-in artifact). Pure CPU, no torch/GPU:
the module is loaded by file path so a built vLLM env is not required.
"""

from __future__ import annotations

import importlib.util
import json
import os
import struct
import sys
from typing import Dict, Optional

CONTRACT_VERSION = "1.0.0"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DU_PATH = os.path.join(_REPO_ROOT, "vllm", "v1", "sample", "deterministic_utils.py")
_OUT_PATH = os.path.join(
    _REPO_ROOT, "tests", "v1", "validation", "conformance_vectors.json"
)


def _load_du():
    """Load deterministic_utils by path (avoids importing the vllm package)."""
    spec = importlib.util.spec_from_file_location("deterministic_utils", _DU_PATH)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass can resolve cls.__module__.
    sys.modules["deterministic_utils"] = mod
    spec.loader.exec_module(mod)
    return mod


du = _load_du()


def _canonical(f: float) -> str:
    """The one float->string conversion (contract §1)."""
    return repr(f)


def _f32_widened(f: float) -> float:
    """float32 value widened to float64 (what the model actually emits)."""
    return struct.unpack("f", struct.pack("f", f))[0]


# Each case: fixed inputs -> the reference pipeline computes the outputs.
# logprobs are already canonical strings (contract §1).
_CASES = [
    {
        "name": "single_token",
        "logprobs": {"791": "-0.05"},
        "temperature": "1.0",
    },
    {
        "name": "three_token_plain",
        "logprobs": {"791": "-0.05", "1": "-3.2", "5": "-2.1"},
        "temperature": "1.0",
    },
    {
        "name": "two_token_near_tie",
        # ln(0.5) for both -> equal probabilities; RNG breaks the tie.
        "logprobs": {"5": "-0.6931471805599453", "7": "-0.6931471805599453"},
        "temperature": "1.0",
    },
    {
        "name": "top_k_truncation",
        "logprobs": {"791": "-0.05", "1": "-3.2", "5": "-2.1", "9": "-1.0"},
        "temperature": "1.0",
        "top_k": 2,
    },
    {
        "name": "top_p_nucleus",
        "logprobs": {"791": "-0.05", "1": "-3.2", "5": "-2.1", "9": "-1.0"},
        "temperature": "1.0",
        "top_p": "0.9",
    },
    {
        "name": "min_p_filter",
        "logprobs": {"791": "-0.05", "1": "-3.2", "5": "-2.1", "9": "-1.0"},
        "temperature": "1.0",
        "min_p": "0.05",
    },
    {
        "name": "temperature_small",
        "logprobs": {"791": "-0.05", "1": "-3.2", "5": "-2.1"},
        "temperature": "0.1",
    },
    {
        "name": "temperature_large",
        "logprobs": {"791": "-0.05", "1": "-3.2", "5": "-2.1"},
        "temperature": "2.0",
    },
    {
        "name": "float32_widened_logprob",
        # The producer would emit repr(f32->f64); pin that exact string here.
        "logprobs": {"791": _canonical(_f32_widened(-0.05)), "1": "-3.0"},
        "temperature": "0.7",
    },
    {
        "name": "ten_tokens",
        "logprobs": {
            str(t): _canonical(-0.1 * (i + 1))
            for i, t in enumerate([3, 10, 42, 100, 256, 512, 777, 1024, 2048, 4096])
        },
        "temperature": "0.8",
        "top_p": "0.95",
    },
]

# Every case is replayed under this fixed seed string (seed composition is S1,
# out of scope for this contract; here it is an opaque input).
_SEED_STR = "42|[1,2,3]"


def _run_case(case: Dict) -> Dict:
    logprobs: Dict[str, str] = case["logprobs"]
    temperature: str = case["temperature"]
    top_p: Optional[str] = case.get("top_p")
    top_k: Optional[int] = case.get("top_k")
    min_p: Optional[str] = case.get("min_p")

    weights = du.logprobs_to_weights(
        logprobs, temperature, top_p=top_p, top_k=top_k, min_p=min_p
    )
    rng = du.Sha256CounterRNG.from_seed_string(_SEED_STR)
    token = du.decimal_sample_from_logprobs(
        logprobs, rng, temperature, top_p=top_p, top_k=top_k, min_p=min_p
    )
    out = {
        "name": case["name"],
        "logprobs": logprobs,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "seed_str": _SEED_STR,
        "expected_weights": {tid: int(w) for tid, w in sorted(weights.items())},
        "expected_weight_sum": int(sum(weights.values())),
        "expected_token": token,
    }
    return out


def _iter_uint64_below(seed: str, n: int, count: int) -> list:
    """A sequence of unbiased [0, n) draws from a fresh RNG (contract §6)."""
    rng = du.Sha256CounterRNG.from_seed_string(seed)
    return [du.uint64_below(rng, n) for _ in range(count)]


def build() -> Dict:
    rng_ref_seed = "reference_seed_v1"
    doc = {
        "contract_version": CONTRACT_VERSION,
        "decimal": {"prec": 10, "rounding": "ROUND_HALF_EVEN"},
        "weight_scale": du.WEIGHT_SCALE,
        "rng_reference": {
            "seed": rng_ref_seed,
            "first_u64": du.iter_u64(rng_ref_seed, 5),
        },
        "float_to_string": [
            {
                "description": "plain -0.05 as float64",
                "repr": _canonical(-0.05),
            },
            {
                "description": "-0.05 as float32 widened to float64",
                "repr": _canonical(_f32_widened(-0.05)),
            },
            {
                "description": "0.1 as float32 widened to float64",
                "repr": _canonical(_f32_widened(0.1)),
            },
        ],
        "cases": [_run_case(c) for c in _CASES],
        # Unbiased categorical draw (contract §6). The pipeline always produces
        # weights summing to 2^16 (a power of 2), so uint64_below never rejects
        # there; these non-power-of-2 moduli exercise the limit + modulo path and
        # pin it cross-language.
        "uint64_below": {
            "seed": rng_ref_seed,
            "count": 8,
            "cases": [
                {"n": n, "draws": _iter_uint64_below(rng_ref_seed, n, 8)}
                for n in [3, 10, 1000, 65537, 999983]
            ],
        },
        # Chain-bound seed derivation (gonka-ai/vllm#56). Independently versioned
        # via its own domain tag; the Go validator must reproduce these digests
        # and reject the same invalid inference ids.
        "seed_derivation": {
            # Pinned contract value (must match deterministic_utils._SEED_DOMAIN_TAG;
            # the accept digests below are derived through it, so drift is caught).
            "domain_tag": "gonka-deterministic-sampling-v1",
            "accept": [
                {"user_seed": us, "inference_id": iid,
                 "expected_seed": du.derive_chain_bound_seed(us, iid)}
                for us, iid in [
                    (7, "chain-abc"),               # pinned golden vector
                    (7, "chain-xyz"),               # domain separation vs above
                    (42, "devshard-escrow1-100"),   # devshard-style id
                    (-1, "chain-abc"),              # negative seed
                    (2**63 - 1, "x"),               # int64 max boundary
                ]
            ],
            # Must fail closed on both sides (Go seed is int64, so only the
            # inference-id rules are cross-language-relevant here).
            "reject_inference_id": [
                {"inference_id": "", "reason": "empty"},
                {"inference_id": " chain-abc ", "reason": "whitespace (space is 0x20, below 0x21)"},
                {"inference_id": "chain\tabc", "reason": "control char (tab)"},
                {"inference_id": "chain-é", "reason": "non-ASCII"},
                {"inference_id": "x" * 257, "reason": "too long (>256)"},
            ],
        },
    }
    return doc


def main() -> None:
    doc = build()
    os.makedirs(os.path.dirname(_OUT_PATH), exist_ok=True)
    with open(_OUT_PATH, "w") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"wrote {_OUT_PATH}")
    print(f"  contract_version={doc['contract_version']} cases={len(doc['cases'])}")
    for c in doc["cases"]:
        assert c["expected_weight_sum"] == du.WEIGHT_SCALE, c["name"]
        print(f"  {c['name']:24} -> token={c['expected_token']}")


if __name__ == "__main__":
    main()
