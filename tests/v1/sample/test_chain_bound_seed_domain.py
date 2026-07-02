# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Invariant tests for chain-bound deterministic sampling seed derivation.

Guards the security boundary required by the inference-validation proposal:

    run_seed = SHA256(user_seed || inference_id_from_chain)

Stage-1 sampling replay is only sound if the RNG domain is bound to the
chain inference instance and cannot be steered by request-controlled material.
These tests pin that contract; they need no GPU, torch, or running server.
"""
import pytest

from vllm.v1.sample.deterministic_utils import (
    Sha256CounterRNG,
    derive_chain_bound_seed,
)


def _stream(seed_str: str, n: int = 8):
    rng = Sha256CounterRNG.from_seed_string(seed_str)
    return [rng.next_u64() for _ in range(n)]


def test_domain_separation_different_inference_id():
    # Same user_seed + (implicitly) same prompt, different chain inference id
    # MUST produce a different seed and therefore a different RNG stream.
    seed_a = derive_chain_bound_seed(42, "chain-inference-0001")
    seed_b = derive_chain_bound_seed(42, "chain-inference-0002")
    assert seed_a != seed_b
    assert _stream(seed_a) != _stream(seed_b)


def test_determinism_same_inputs():
    # Same (user_seed, inference_id) MUST be bit-identical across calls.
    seed_1 = derive_chain_bound_seed(42, "chain-inference-0001")
    seed_2 = derive_chain_bound_seed(42, "chain-inference-0001")
    assert seed_1 == seed_2
    assert _stream(seed_1) == _stream(seed_2)


def test_fail_closed_on_missing_inference_id():
    # No silent fallback to request-controlled material.
    for bad in (None, "", "   "):
        with pytest.raises(ValueError):
            derive_chain_bound_seed(42, bad)  # type: ignore[arg-type]


def test_no_concatenation_collision():
    # Length-prefixed framing must prevent ambiguous concatenation, e.g.
    # naive f"{user_seed}{inference_id}" collides for (4,"2x") and (42,"x").
    assert (
        derive_chain_bound_seed(4, "2x")
        != derive_chain_bound_seed(42, "x")
    )


def test_output_is_stable_hex_digest():
    seed = derive_chain_bound_seed(7, "chain-abc")
    assert isinstance(seed, str)
    assert len(seed) == 64
    assert seed == seed.lower()
    int(seed, 16)  # must be valid hex


def test_chain_bound_seed_golden_vector():
    # Pins the exact digest so a future framing/encoding change is caught even
    # when the property-level tests still pass.
    assert derive_chain_bound_seed(7, "chain-abc") == (
        "910b688db5b2061e66385acf0ee665682d5e01bab5d1d8d2cdde9a2612a6e6c2"
    )


def test_non_string_chain_id_rejected():
    # Provenance material must be chain-provided text, never a stringified
    # object/bytes/int.
    for bad in (123, b"chain-abc", object()):
        with pytest.raises(TypeError):
            derive_chain_bound_seed(7, bad)  # type: ignore[arg-type]


def test_whitespace_in_chain_id_not_normalized():
    # The chain id is hashed byte-exact; surrounding whitespace is significant.
    assert (
        derive_chain_bound_seed(7, "chain-abc")
        != derive_chain_bound_seed(7, " chain-abc ")
    )


def test_unicode_chain_id_byte_framing_stable():
    # Unicode in the chain id exercises byte-length (not codepoint) framing.
    seed = derive_chain_bound_seed(7, "chain-β")
    assert len(seed) == 64
    assert seed == seed.lower()
    int(seed, 16)


def test_user_seed_type_rejected():
    # vLLM's seed is int-only; a str "7" would collide with int 7, so only int
    # is accepted. bool is a subclass of int and is rejected explicitly.
    for bad in (None, True, False, 1.5, "7", b"7", object()):
        with pytest.raises(TypeError):
            derive_chain_bound_seed(bad, "chain-abc")  # type: ignore[arg-type]
