#!/usr/bin/env python3
"""
Test script to verify deterministic hash sampling works via OpenAI API.
This script tests the API parameter integration.
"""

import json

def test_api_parameter_acceptance():
    print("Testing API Parameter Acceptance...")
    print("=" * 70)
    
    try:
        from vllm.entrypoints.openai.protocol import ChatCompletionRequest
        
        request1 = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "test"}]
        )
        assert request1.use_deterministic_hash == False, "Default should be False"
        print("✓ Default value is False")
        
        request2 = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
            use_deterministic_hash=True
        )
        assert request2.use_deterministic_hash == True
        print("✓ Can set to True")
        
        request3 = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
            seed=42,
            use_deterministic_hash=True,
            temperature=1.0
        )
        assert request3.seed == 42
        assert request3.use_deterministic_hash == True
        print("✓ Works with seed parameter")
        
        request_dict = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "seed": 42,
            "use_deterministic_hash": True,
            "temperature": 1.0
        }
        request4 = ChatCompletionRequest(**request_dict)
        assert request4.use_deterministic_hash == True
        print("✓ Works with dict/JSON input")
        
        print("\n✓ All API parameter tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sampling_params_conversion():
    print("\n" + "=" * 70)
    print("Testing SamplingParams Conversion...")
    print("=" * 70)
    
    try:
        from vllm.entrypoints.openai.protocol import ChatCompletionRequest
        from vllm.sampling_params import SamplingType
        
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
            seed=42,
            use_deterministic_hash=True,
            temperature=1.0,
            max_tokens=100
        )
        
        sampling_params = request.to_sampling_params(
            default_max_tokens=512,
            logits_processor_pattern=None
        )
        
        assert sampling_params.use_deterministic_hash == True
        print("✓ use_deterministic_hash passed to SamplingParams")
        
        assert sampling_params.seed == 42
        print("✓ seed passed correctly")
        
        assert sampling_params.temperature == 1.0
        print("✓ temperature passed correctly")
        
        # Verify sampling type
        assert sampling_params.sampling_type == SamplingType.DETERMINISTIC_HASH
        print(f"✓ sampling_type is {sampling_params.sampling_type}")
        
        # Test without deterministic hash
        request2 = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
            seed=42,
            use_deterministic_hash=False,
            temperature=1.0
        )
        
        sampling_params2 = request2.to_sampling_params(
            default_max_tokens=512,
            logits_processor_pattern=None
        )
        
        assert sampling_params2.sampling_type == SamplingType.RANDOM_SEED
        print(f"✓ Without deterministic_hash: {sampling_params2.sampling_type}")
        
        print("\n✓ All SamplingParams conversion tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_completion_request():
    print("\n" + "=" * 70)
    print("Testing CompletionRequest...")
    print("=" * 70)
    
    try:
        from vllm.entrypoints.openai.protocol import CompletionRequest
        from vllm.sampling_params import SamplingType
        
        request = CompletionRequest(
            model="test-model",
            prompt="Test prompt",
            seed=42,
            use_deterministic_hash=True,
            temperature=1.0,
            max_tokens=100
        )
        
        assert request.use_deterministic_hash == True
        print("✓ CompletionRequest accepts use_deterministic_hash")
        
        sampling_params = request.to_sampling_params(
            default_max_tokens=512,
            logits_processor_pattern=None
        )
        
        assert sampling_params.use_deterministic_hash == True
        assert sampling_params.sampling_type == SamplingType.DETERMINISTIC_HASH
        print(f"✓ Converts to {sampling_params.sampling_type}")
        
        print("\n✓ All CompletionRequest tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_json_examples():
    print("\n" + "=" * 70)
    print("Testing JSON API Payloads...")
    print("=" * 70)
    
    try:
        from vllm.entrypoints.openai.protocol import ChatCompletionRequest
        
        json_payload = {
            "model": "meta-llama/Llama-3.2-3B-Instruct",
            "messages": [
                {"role": "user", "content": "Write a short story"}
            ],
            "temperature": 1.0,
            "seed": 42,
            "use_deterministic_hash": True,
            "max_tokens": 100,
            "logprobs": True,
            "top_logprobs": 5
        }
        
        request = ChatCompletionRequest(**json_payload)
        assert request.use_deterministic_hash == True
        assert request.seed == 42
        assert request.logprobs == True
        print("✓ Chat completion JSON payload accepted")
        
        json_payload2 = {
            "model": "meta-llama/Llama-3.2-3B-Instruct",
            "messages": [
                {"role": "user", "content": "Write a short story"}
            ],
            "temperature": 0.7,
            "seed": 42,
            # use_deterministic_hash omitted - should default to False
        }
        
        request2 = ChatCompletionRequest(**json_payload2)
        assert request2.use_deterministic_hash == False
        print("✓ Backward compatibility maintained (defaults to False)")
        
        # Example 3: Pretty print for documentation
        example_json = json.dumps({
            "model": "meta-llama/Llama-3.2-3B-Instruct",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 1.0,
            "seed": 42,
            "use_deterministic_hash": True,
        }, indent=2)
        
        print("\n Example API Request:")
        print(example_json)
        
        print("\n✓ All JSON payload tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("vLLM Deterministic Sampling - API Integration Tests")
    print("=" * 70)
    print()
    
    all_passed = True
    
    all_passed &= test_api_parameter_acceptance()
    all_passed &= test_sampling_params_conversion()
    all_passed &= test_completion_request()
    all_passed &= test_json_examples()
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nAPI integration is ready!")
        print("\nNext steps:")
        print("  1. Build Docker image: docker build -t ghcr.io/gonka-ai/vllm:v0.9.2 .")
        print("  2. Test with actual server: python -m vllm.entrypoints.openai.api_server --model <model>")
        print("  3. Make API call with use_deterministic_hash: true")
        print("  4. Verify reproducibility")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
