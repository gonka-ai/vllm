import pytest
from vllm.entrypoints.openai.validation_utils import (
    generate_run_seed,
    compute_derived_seed,
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
