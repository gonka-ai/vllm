import pytest
from vllm.entrypoints.openai.validation_utils import (
    generate_run_seed,
    compute_derived_seed,
    derive_seed_from_run_seed,
)


def test_generate_run_seed():
    run_seed = generate_run_seed(42, "chatcmpl-abc123")
    assert run_seed != ""
    assert len(run_seed) == 64


def test_generate_run_seed_deterministic():
    run_seed1 = generate_run_seed(42, "chatcmpl-abc123")
    run_seed2 = generate_run_seed(42, "chatcmpl-abc123")
    assert run_seed1 == run_seed2


def test_generate_run_seed_different_inference_ids():
    run_seed1 = generate_run_seed(42, "chatcmpl-abc123")
    run_seed2 = generate_run_seed(42, "chatcmpl-xyz789")
    assert run_seed1 != run_seed2


def test_generate_run_seed_none():
    run_seed = generate_run_seed(None, "chatcmpl-abc123")
    assert run_seed == ""


def test_compute_derived_seed():
    derived_seed, run_seed = compute_derived_seed(42, "chatcmpl-abc123")
    assert derived_seed is not None
    assert isinstance(derived_seed, int)
    assert derived_seed > 0
    assert len(run_seed) == 64


def test_compute_derived_seed_deterministic():
    derived_seed1, run_seed1 = compute_derived_seed(42, "chatcmpl-abc123")
    derived_seed2, run_seed2 = compute_derived_seed(42, "chatcmpl-abc123")
    assert derived_seed1 == derived_seed2
    assert run_seed1 == run_seed2


def test_compute_derived_seed_none():
    derived_seed, run_seed = compute_derived_seed(None, "chatcmpl-abc123")
    assert derived_seed is None
    assert run_seed == ""


def test_compute_derived_seed_with_run_seed():
    original_run_seed = "4ef75adbd0ec3fba28ade0cf79ba5d4afea742e476567517d348836bb1876c9b"
    derived_seed, returned_run_seed = compute_derived_seed(
        42, "chatcmpl-different-id", run_seed=original_run_seed)
    assert returned_run_seed == original_run_seed
    assert derived_seed == derive_seed_from_run_seed(original_run_seed)


def test_compute_derived_seed_run_seed_overrides_user_seed():
    original_run_seed = "4ef75adbd0ec3fba28ade0cf79ba5d4afea742e476567517d348836bb1876c9b"
    derived_seed1, run_seed1 = compute_derived_seed(42, "chatcmpl-abc123")
    derived_seed2, run_seed2 = compute_derived_seed(
        42, "chatcmpl-abc123", run_seed=original_run_seed)
    assert run_seed1 != run_seed2
    assert run_seed2 == original_run_seed
    assert derived_seed1 != derived_seed2


def test_compute_derived_seed_run_seed_without_user_seed():
    original_run_seed = "4ef75adbd0ec3fba28ade0cf79ba5d4afea742e476567517d348836bb1876c9b"
    derived_seed, returned_run_seed = compute_derived_seed(
        None, "chatcmpl-abc123", run_seed=original_run_seed)
    assert returned_run_seed == original_run_seed
    assert derived_seed == derive_seed_from_run_seed(original_run_seed)


def test_derive_seed_from_run_seed():
    run_seed = "4ef75adbd0ec3fba28ade0cf79ba5d4afea742e476567517d348836bb1876c9b"
    derived = derive_seed_from_run_seed(run_seed)
    expected = int(run_seed[:16], 16) & 0x7FFFFFFFFFFFFFFF
    assert derived == expected
