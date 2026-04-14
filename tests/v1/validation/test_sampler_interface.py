# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Integration tests for the sampler to topk_topp_sampler interface.

These tests verify that:
1. SamplingMetadata is correctly passed between sampler and topk_topp_sampler
2. The interface signatures match
3. The sampling chain works correctly with deterministic sampling

These tests catch interface mismatches like passing a SamplingMetadata object
where individual parameters are expected.
"""

import pytest
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch


class TestSamplerInterface:
    """Test the interface between Sampler and TopKTopPSampler."""

    def test_topk_topp_sampler_accepts_sampling_metadata(self):
        """
        Verify that TopKTopPSampler.forward methods accept SamplingMetadata.
        
        This test verifies the fix for the critical interface mismatch where
        sampler.py passes SamplingMetadata but topk_topp_sampler expected
        individual parameters (generators, k, p, deterministic_rngs).
        """
        import inspect
        from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler
        
        # Check forward_native signature
        sig = inspect.signature(TopKTopPSampler.forward_native)
        params = list(sig.parameters.keys())
        
        # Should have: self, logits, sampling_metadata
        assert 'self' in params
        assert 'logits' in params
        assert 'sampling_metadata' in params
        
        # Should NOT have the old individual parameters
        assert 'generators' not in params
        assert 'k' not in params
        assert 'p' not in params
        assert 'deterministic_rngs' not in params
        
    def test_forward_cuda_signature(self):
        """Verify forward_cuda has correct signature."""
        import inspect
        from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler
        
        sig = inspect.signature(TopKTopPSampler.forward_cuda)
        params = list(sig.parameters.keys())
        
        assert 'sampling_metadata' in params
        assert 'generators' not in params
        
    def test_forward_cpu_signature(self):
        """Verify forward_cpu has correct signature."""
        import inspect
        from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler
        
        sig = inspect.signature(TopKTopPSampler.forward_cpu)
        params = list(sig.parameters.keys())
        
        assert 'sampling_metadata' in params
        assert 'generators' not in params


class TestSamplingMetadataFields:
    """Test that SamplingMetadata has required fields for sampling."""
    
    def test_sampling_metadata_has_required_fields(self):
        """Verify SamplingMetadata has all fields needed by topk_topp_sampler."""
        from vllm.v1.sample.metadata import SamplingMetadata
        import dataclasses
        
        field_names = {f.name for f in dataclasses.fields(SamplingMetadata)}
        
        # Required fields for topk_topp_sampler
        assert 'generators' in field_names
        assert 'top_k' in field_names
        assert 'top_p' in field_names
        assert 'deterministic_rngs' in field_names
        
    def test_deterministic_rngs_default(self):
        """Verify deterministic_rngs has correct default (empty dict)."""
        from vllm.v1.sample.metadata import SamplingMetadata
        import dataclasses
        
        for f in dataclasses.fields(SamplingMetadata):
            if f.name == 'deterministic_rngs':
                # Should have a default_factory
                assert f.default_factory is not None
                # Default should be empty dict
                default = f.default_factory()
                assert default == {}


class TestDeterministicSamplingIntegration:
    """Test deterministic sampling integration."""
    
    def test_deterministic_sample_function_exists(self):
        """Verify deterministic_sample function exists and is importable."""
        from vllm.v1.sample.ops.topk_topp_sampler import deterministic_sample
        assert callable(deterministic_sample)
        
    def test_sha256_counter_rng_works(self):
        """Test that Sha256CounterRNG produces reproducible results."""
        from vllm.v1.sample.deterministic_utils import (
            Sha256CounterRNG,
            sample_categorical_weights,
        )
        
        # Same seed should produce same sequence
        rng1 = Sha256CounterRNG.from_seed_string("test_seed")
        rng2 = Sha256CounterRNG.from_seed_string("test_seed")
        
        for _ in range(100):
            assert rng1.next_u64() == rng2.next_u64()
            
    def test_sample_categorical_weights_deterministic(self):
        """Test that sample_categorical_weights is deterministic."""
        from vllm.v1.sample.deterministic_utils import (
            Sha256CounterRNG,
            sample_categorical_weights,
        )
        
        weights = [60000, 5000, 500, 36]
        seed = "categorical_test"
        
        # Sample 100 times with same seed - should get same results
        results1 = []
        rng1 = Sha256CounterRNG.from_seed_string(seed)
        for _ in range(100):
            results1.append(sample_categorical_weights(weights, rng1))
            
        results2 = []
        rng2 = Sha256CounterRNG.from_seed_string(seed)
        for _ in range(100):
            results2.append(sample_categorical_weights(weights, rng2))
            
        assert results1 == results2


class TestValidationLogicIntegration:
    """Test that validation logic works with the sampling chain."""
    
    def test_verify_sampling_sequence_uses_correct_rng(self):
        """
        Verify that verify_sampling_sequence uses the same RNG logic
        as the actual sampling.
        """
        from vllm.v1.sample.deterministic_utils import (
            Sha256CounterRNG,
            sample_categorical_weights,
        )
        from vllm.validation import EnforcedToken
        from vllm.validation_logic import verify_sampling_sequence
        
        seed_str = "integration_test_seed"
        weights = [60000, 5000, 500]
        token_ids = ["100", "200", "300"]
        
        # Generate expected sequence using same RNG
        rng = Sha256CounterRNG.from_seed_string(seed_str)
        expected_tokens = []
        for _ in range(10):
            idx = sample_categorical_weights(weights, rng)
            expected_tokens.append(token_ids[idx])
            
        # Build artifact with expected tokens
        tokens = [
            EnforcedToken(
                token=expected_tokens[i],
                top_tokens=token_ids,
                sampling_weights={tid: w for tid, w in zip(token_ids, weights)},
            )
            for i in range(10)
        ]
        
        # Verify should pass
        success, failed_pos = verify_sampling_sequence(tokens, seed_str)
        assert success, f"Verification failed at position {failed_pos}"
        assert failed_pos == -1


class TestEnvVariableIntegration:
    """Test VLLM_DETERMINISTIC_SAMPLING environment variable integration."""
    
    def test_env_variable_defined(self):
        """Verify VLLM_DETERMINISTIC_SAMPLING is defined in envs."""
        from vllm import envs
        
        # The attribute should exist
        assert hasattr(envs, 'VLLM_DETERMINISTIC_SAMPLING')
        
    def test_env_variable_is_boolean(self):
        """Verify VLLM_DETERMINISTIC_SAMPLING returns a boolean."""
        from vllm import envs
        
        # Should be a boolean (False by default)
        value = envs.VLLM_DETERMINISTIC_SAMPLING
        assert isinstance(value, bool)
