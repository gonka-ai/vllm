# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Integration tests for validation flow.

These tests verify the full validation flow with synthetic/mocked data,
without requiring a running vLLM server.
"""

import pytest
from typing import List, Dict

from vllm.validation import EnforcedToken, EnforcedTokens
from vllm.validation_logic import (
    validate_before_model,
    validate_after_model,
    validate_full,
    recompute_weights_from_logprobs,
    WEIGHT_SCALE,
    DEFAULT_FRAUD_DISTANCE_THRESHOLD,
)
from vllm.v1.sample.deterministic_utils import (
    Sha256CounterRNG,
    sample_categorical_weights,
)


def build_honest_artifact(
    prompt: str,
    seed: int,
    n_tokens: int = 10,
    temperature: float = 0.99,
    top_k: int = 5,
) -> EnforcedTokens:
    """
    Build an honest artifact with correct sampling.
    
    This simulates what an honest executor would produce.
    """
    seed_str = f"{seed}|{prompt}"
    rng = Sha256CounterRNG.from_seed_string(seed_str)
    
    tokens = []
    for i in range(n_tokens):
        # Generate synthetic logprobs
        # Dominant token has high prob, others lower
        logprobs = {}
        token_ids = [str(100 + j) for j in range(top_k)]
        
        # Create descending logprobs
        for j, tid in enumerate(token_ids):
            logprobs[tid] = -(0.5 + j * 1.5)  # -0.5, -2.0, -3.5, ...
        
        # Compute weights from logprobs
        weights = recompute_weights_from_logprobs(logprobs, temperature)
        
        # Sample token using RNG
        sorted_items = sorted(weights.items(), key=lambda x: int(x[0]))
        weight_list = [w for _, w in sorted_items]
        token_list = [t for t, _ in sorted_items]
        
        idx = sample_categorical_weights(weight_list, rng)
        sampled_token = token_list[idx]
        
        tokens.append(EnforcedToken(
            token=sampled_token,
            top_tokens=token_ids,
            logprobs=logprobs,
            sampling_weights=weights,
        ))
    
    return EnforcedTokens(tokens=tokens)


def build_prefill_attack_artifact(
    prompt: str,
    seed: int,
    n_tokens: int = 10,
    temperature: float = 0.99,
    top_k: int = 5,
) -> EnforcedTokens:
    """
    Build an artifact simulating a pre-fill attack.
    
    The attacker used a cheap model to generate tokens, then computed
    logprobs from the expensive model. The tokens don't match what
    would be sampled from the logprobs.
    """
    seed_str = f"{seed}|{prompt}"
    
    tokens = []
    for i in range(n_tokens):
        # Generate synthetic logprobs
        logprobs = {}
        token_ids = [str(100 + j) for j in range(top_k)]
        
        for j, tid in enumerate(token_ids):
            logprobs[tid] = -(0.5 + j * 1.5)
        
        # Compute weights
        weights = recompute_weights_from_logprobs(logprobs, temperature)
        
        # ATTACK: Use a WRONG token (not sampled from weights)
        # Pick the least likely token instead
        wrong_token = token_ids[-1]
        
        tokens.append(EnforcedToken(
            token=wrong_token,
            top_tokens=token_ids,
            logprobs=logprobs,
            sampling_weights=weights,
        ))
    
    return EnforcedTokens(tokens=tokens)


def build_tampered_weights_artifact(
    prompt: str,
    seed: int,
    n_tokens: int = 10,
    temperature: float = 0.99,
    top_k: int = 5,
) -> EnforcedTokens:
    """
    Build an artifact with tampered weights.
    
    The attacker modified the sampling_weights to match their desired tokens,
    but the weights are inconsistent with the logprobs.
    """
    seed_str = f"{seed}|{prompt}"
    rng = Sha256CounterRNG.from_seed_string(seed_str)
    
    tokens = []
    for i in range(n_tokens):
        logprobs = {}
        token_ids = [str(100 + j) for j in range(top_k)]
        
        for j, tid in enumerate(token_ids):
            logprobs[tid] = -(0.5 + j * 1.5)
        
        # ATTACK: Create fake weights that would sample the wrong token
        # Put all weight on the last token
        fake_weights = {tid: 0 for tid in token_ids}
        fake_weights[token_ids[-1]] = WEIGHT_SCALE  # All weight on last
        
        # Sample from fake weights (will always pick last)
        sorted_items = sorted(fake_weights.items(), key=lambda x: int(x[0]))
        weight_list = [w for _, w in sorted_items]
        token_list = [t for t, _ in sorted_items]
        
        idx = sample_categorical_weights(weight_list, rng)
        sampled_token = token_list[idx]
        
        tokens.append(EnforcedToken(
            token=sampled_token,
            top_tokens=token_ids,
            logprobs=logprobs,
            sampling_weights=fake_weights,
        ))
    
    return EnforcedTokens(tokens=tokens)


class TestValidateBeforeModel:
    """Test pre-model validation (Stage 1)."""

    def test_honest_artifact_passes(self):
        """Honest artifact should pass Stage 1."""
        artifact = build_honest_artifact("What is 2+2?", seed=42)
        seed_str = "42|What is 2+2?"
        
        result = validate_before_model(
            artifact,
            seed_str,
            temperature=0.99,
        )
        
        assert not result.fraud
        assert result.correct_sampling
        assert result.correct_processed_logprobs

    def test_prefill_attack_detected(self):
        """Pre-fill attack should be detected in Stage 1b."""
        artifact = build_prefill_attack_artifact("What is 2+2?", seed=42)
        seed_str = "42|What is 2+2?"
        
        result = validate_before_model(
            artifact,
            seed_str,
            temperature=0.99,
        )
        
        assert result.fraud
        assert not result.correct_sampling
        assert result.distance == 10.0

    def test_tampered_weights_detected(self):
        """Tampered weights should be detected in Stage 1a."""
        artifact = build_tampered_weights_artifact("What is 2+2?", seed=42)
        seed_str = "42|What is 2+2?"
        
        result = validate_before_model(
            artifact,
            seed_str,
            temperature=0.99,
        )
        
        assert result.fraud
        assert not result.correct_processed_logprobs
        assert result.distance == 10.0

    def test_no_weights_skips_validation(self):
        """Artifact without weights should skip Stage 1."""
        tokens = [
            EnforcedToken(
                token="100",
                top_tokens=["100", "200"],
                logprobs={"100": -0.5, "200": -1.2},
                sampling_weights=None,  # No weights
            )
        ]
        artifact = EnforcedTokens(tokens=tokens)
        
        result = validate_before_model(
            artifact,
            "test_seed",
            temperature=0.99,
        )
        
        # Should pass (nothing to verify)
        assert not result.fraud


class TestValidateAfterModel:
    """Test post-model validation (Stage 2)."""

    def test_identical_logprobs_pass(self):
        """Identical logprobs should pass with low distance."""
        logprobs = [
            {"100": -0.5, "200": -1.2, "300": -2.1},
            {"100": -0.8, "200": -0.9, "300": -2.5},
        ]
        
        result = validate_after_model(logprobs, logprobs)
        
        assert not result.fraud
        assert result.correct_raw_logprobs
        assert result.distance < DEFAULT_FRAUD_DISTANCE_THRESHOLD

    def test_small_differences_pass(self):
        """Small differences (numerical precision) should pass."""
        executor_lp = [
            {"100": -0.5, "200": -1.2},
            {"100": -0.8, "200": -1.5},
        ]
        validator_lp = [
            {"100": -0.5001, "200": -1.2002},  # Tiny difference
            {"100": -0.8001, "200": -1.5001},
        ]
        
        result = validate_after_model(executor_lp, validator_lp)
        
        assert not result.fraud
        assert result.correct_raw_logprobs

    def test_large_differences_fail(self):
        """Large differences (wrong model) should fail."""
        executor_lp = [
            {"100": -0.1, "200": -5.0},  # Token 100 dominates
        ] * 100
        validator_lp = [
            {"100": -5.0, "200": -0.1},  # Token 200 dominates
        ] * 100
        
        result = validate_after_model(executor_lp, validator_lp)
        
        assert result.fraud
        assert not result.correct_raw_logprobs
        assert result.distance > DEFAULT_FRAUD_DISTANCE_THRESHOLD

    def test_length_mismatch_fails(self):
        """Mismatched sequence lengths should fail."""
        executor_lp = [{"100": -0.5}] * 10
        validator_lp = [{"100": -0.5}] * 5  # Different length
        
        result = validate_after_model(executor_lp, validator_lp)
        
        assert result.fraud
        assert result.distance == 10.0


class TestValidateFull:
    """Test full two-stage validation."""

    def test_honest_execution_passes_all(self):
        """Honest execution should pass all checks."""
        artifact = build_honest_artifact("Hello world", seed=123, n_tokens=20)
        seed_str = "123|Hello world"
        
        # Validator's logprobs (same as executor's in honest case)
        validator_logprobs = [t.logprobs for t in artifact.tokens]
        
        result = validate_full(
            artifact,
            validator_logprobs,
            seed_str,
            temperature=0.99,
        )
        
        assert not result.fraud
        assert result.correct_sampling
        assert result.correct_processed_logprobs
        assert result.correct_raw_logprobs
        assert result.distance < DEFAULT_FRAUD_DISTANCE_THRESHOLD

    def test_prefill_attack_fails_stage1(self):
        """Pre-fill attack should fail at Stage 1."""
        artifact = build_prefill_attack_artifact("Hello world", seed=123)
        seed_str = "123|Hello world"
        
        # Even with matching validator logprobs, should fail Stage 1
        validator_logprobs = [t.logprobs for t in artifact.tokens]
        
        result = validate_full(
            artifact,
            validator_logprobs,
            seed_str,
            temperature=0.99,
        )
        
        assert result.fraud
        assert not result.correct_sampling  # Failed Stage 1b

    def test_wrong_model_fails_stage2(self):
        """Wrong model should pass Stage 1 but fail Stage 2."""
        artifact = build_honest_artifact("Hello world", seed=123, n_tokens=100)
        seed_str = "123|Hello world"
        
        # Validator's logprobs are different (simulating wrong model)
        validator_logprobs = []
        for t in artifact.tokens:
            # Shift all logprobs
            shifted = {k: v - 2.0 for k, v in t.logprobs.items()}
            validator_logprobs.append(shifted)
        
        result = validate_full(
            artifact,
            validator_logprobs,
            seed_str,
            temperature=0.99,
        )
        
        assert result.fraud
        assert result.correct_sampling  # Stage 1 passed
        assert result.correct_processed_logprobs  # Stage 1a passed
        assert not result.correct_raw_logprobs  # Stage 2 failed


class TestValidationResultFields:
    """Test ValidationResult field semantics."""

    def test_fraud_false_by_default(self):
        """Result should not indicate fraud by default."""
        from vllm.entrypoints.openai.protocol import ValidationResult
        
        result = ValidationResult()
        
        assert not result.fraud
        assert result.correct_raw_logprobs
        assert result.correct_processed_logprobs
        assert result.correct_sampling

    def test_fraud_indicators_set_correctly(self):
        """Fraud indicators should be set based on which check failed."""
        # Pre-fill attack
        artifact = build_prefill_attack_artifact("test", seed=1)
        result = validate_before_model(artifact, "1|test", temperature=0.99)
        
        assert result.fraud
        assert not result.correct_sampling
        assert result.correct_processed_logprobs  # Didn't fail this

        # Tampered weights
        artifact = build_tampered_weights_artifact("test", seed=1)
        result = validate_before_model(artifact, "1|test", temperature=0.99)
        
        assert result.fraud
        assert not result.correct_processed_logprobs


class TestEnforcedTokensHelpers:
    """Test EnforcedTokens helper methods."""

    def test_has_validation_data_with_logprobs(self):
        """Should return True when logprobs present."""
        tokens = [
            EnforcedToken(
                token="100",
                top_tokens=["100", "200"],
                logprobs={"100": -0.5, "200": -1.2},
            )
        ]
        artifact = EnforcedTokens(tokens=tokens)
        
        assert artifact.has_validation_data()

    def test_has_validation_data_with_weights(self):
        """Should return True when weights present."""
        tokens = [
            EnforcedToken(
                token="100",
                top_tokens=["100", "200"],
                sampling_weights={"100": 60000, "200": 5536},
            )
        ]
        artifact = EnforcedTokens(tokens=tokens)
        
        assert artifact.has_validation_data()

    def test_has_validation_data_without_data(self):
        """Should return False when no validation data."""
        tokens = [
            EnforcedToken(
                token="100",
                top_tokens=["100", "200"],
            )
        ]
        artifact = EnforcedTokens(tokens=tokens)
        
        assert not artifact.has_validation_data()

    def test_from_content_with_logprobs(self):
        """Test creating from OpenAI-format content."""
        content = [
            {
                "token": "100",
                "logprob": -0.5,
                "top_logprobs": [
                    {"token": "100", "logprob": -0.5},
                    {"token": "200", "logprob": -1.2},
                ],
            },
        ]
        
        artifact = EnforcedTokens.from_content(
            content,
            include_logprobs=True,
            include_sampling_weights=False,
        )
        
        assert len(artifact.tokens) == 1
        assert artifact.tokens[0].token == "100"
        assert artifact.tokens[0].logprobs == {"100": -0.5, "200": -1.2}

    def test_from_content_with_weights(self):
        """Test creating with sampling weights."""
        content = [
            {
                "token": "100",
                "logprob": -0.5,
                "top_logprobs": [
                    {"token": "100", "logprob": -0.5},
                    {"token": "200", "logprob": -1.2},
                ],
                "sampling_weights": {"100": 60000, "200": 5536},
            },
        ]
        
        artifact = EnforcedTokens.from_content(
            content,
            include_logprobs=True,
            include_sampling_weights=True,
        )
        
        assert artifact.tokens[0].sampling_weights == {"100": 60000, "200": 5536}
