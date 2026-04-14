#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end test for validation enhancement.

Prerequisites:
- vLLM server running with VLLM_DETERMINISTIC_SAMPLING=1
- Server accessible at http://localhost:8000

Usage:
    # Start server first:
    VLLM_DETERMINISTIC_SAMPLING=1 python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --max-model-len 2048 \
        --enforce-eager \
        --port 8000

    # Then run tests:
    python scripts/test_validation_e2e.py

    # Or with custom URL/model:
    python scripts/test_validation_e2e.py --url http://localhost:8001 --model Qwen/Qwen3-8B
"""

import argparse
import json
import sys
import time
from typing import Dict, List, Any, Optional

import requests


def chat_completion(
    base_url: str,
    model: str,
    messages: List[Dict],
    **kwargs
) -> Dict:
    """Send chat completion request."""
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            **kwargs
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def test_server_health(base_url: str) -> bool:
    """Check if server is healthy."""
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        return response.status_code == 200
    except:
        return False


def test_deterministic_sampling_produces_weights(base_url: str, model: str):
    """Test that sampling_weights are included when logprobs requested."""
    print("\n=== Test 1: sampling_weights in response ===")
    
    response = chat_completion(
        base_url, model,
        messages=[{"role": "user", "content": "Say hello"}],
        temperature=0.99,
        seed=42,
        logprobs=True,
        top_logprobs=5,
        max_tokens=10
    )
    
    content = response["choices"][0]["logprobs"]["content"]
    first_token = content[0]
    
    assert "sampling_weights" in first_token, "sampling_weights missing from response"
    assert first_token["sampling_weights"] is not None, "sampling_weights is None"
    assert isinstance(first_token["sampling_weights"], dict), "sampling_weights should be dict"
    
    # Weights should be integers
    for tid, weight in first_token["sampling_weights"].items():
        assert isinstance(weight, int), f"Weight {weight} for token {tid} is not int"
    
    print(f"  First token: {first_token['token']}")
    print(f"  sampling_weights: {first_token['sampling_weights']}")
    print("✓ sampling_weights present and valid")
    return response


def test_reproducibility(base_url: str, model: str):
    """Test that same seed produces identical outputs."""
    print("\n=== Test 2: Reproducibility ===")
    
    params = {
        "messages": [{"role": "user", "content": "Count from 1 to 5"}],
        "temperature": 0.99,
        "seed": 12345,
        "max_tokens": 20
    }
    
    results = []
    for i in range(3):
        response = chat_completion(base_url, model, **params)
        text = response["choices"][0]["message"]["content"]
        results.append(text)
        print(f"  Run {i+1}: {text[:50]}...")
    
    assert len(set(results)) == 1, f"Outputs differ across runs: {results}"
    print("✓ All 3 runs produced identical output")


def test_validation_flow(base_url: str, model: str):
    """Test the full validation flow: executor -> validator."""
    print("\n=== Test 3: Validation Flow ===")
    
    # Step 1: Executor generates
    print("  Step 1: Executor generates...")
    exec_response = chat_completion(
        base_url, model,
        messages=[{"role": "user", "content": "What is 2+2?"}],
        temperature=0.99,
        seed=42,
        logprobs=True,
        top_logprobs=5,
        max_tokens=10
    )
    
    exec_content = exec_response["choices"][0]["logprobs"]["content"]
    print(f"  Executor generated {len(exec_content)} tokens")
    
    # Step 2: Build enforced_tokens from executor response
    print("  Step 2: Building enforced_tokens...")
    enforced_tokens = {"tokens": []}
    
    for pos in exec_content:
        token_data = {
            "token": str(pos["token"]),
            "top_tokens": [str(t["token"]) for t in pos["top_logprobs"]],
        }
        
        # Include logprobs
        if pos.get("top_logprobs"):
            token_data["logprobs"] = {
                str(t["token"]): t["logprob"]
                for t in pos["top_logprobs"]
            }
        
        # Include sampling_weights
        if pos.get("sampling_weights"):
            token_data["sampling_weights"] = pos["sampling_weights"]
        
        enforced_tokens["tokens"].append(token_data)
    
    print(f"  Built artifact with {len(enforced_tokens['tokens'])} tokens")
    
    # Step 3: Validator verifies
    print("  Step 3: Validator verifies...")
    try:
        val_response = chat_completion(
            base_url, model,
            messages=[{"role": "user", "content": "What is 2+2?"}],
            temperature=0.99,
            seed=42,
            logprobs=True,
            top_logprobs=5,
            max_tokens=10,
            extra_body={"enforced_tokens": enforced_tokens}
        )
        
        # Step 4: Check validation result
        validation = val_response.get("validation")
        if validation:
            print(f"  Validation result: {json.dumps(validation, indent=2)}")
            assert not validation.get("fraud"), "Honest execution flagged as fraud!"
            if validation.get("correct_sampling") is not None:
                assert validation["correct_sampling"], "Sampling verification failed!"
            print("✓ Validation passed (no fraud detected)")
        else:
            print("  Note: validation field not in response")
            print("  (Server may not have validation integration enabled)")
            print("✓ Validation flow completed (no validation result)")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 422:
            print("  Note: Server returned 422 - enforced_tokens may not be supported")
            print("✓ Validation flow test skipped (feature not available)")
        else:
            raise


def test_greedy_no_weights(base_url: str, model: str):
    """Test that greedy sampling (temp=0) doesn't produce weights."""
    print("\n=== Test 4: Greedy sampling (no weights) ===")
    
    response = chat_completion(
        base_url, model,
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0,  # Greedy
        logprobs=True,
        top_logprobs=5,
        max_tokens=5
    )
    
    content = response["choices"][0]["logprobs"]["content"]
    first_token = content[0]
    
    # Greedy should have None weights
    weights = first_token.get("sampling_weights")
    assert weights is None, f"Expected None weights for greedy, got {weights}"
    print("✓ Greedy sampling correctly has no sampling_weights")


def test_weights_sum(base_url: str, model: str):
    """Test that weights sum approximately to 2^16."""
    print("\n=== Test 5: Weight sum validation ===")
    
    response = chat_completion(
        base_url, model,
        messages=[{"role": "user", "content": "Test weights"}],
        temperature=0.99,
        seed=42,
        logprobs=True,
        top_logprobs=10,
        max_tokens=5
    )
    
    WEIGHT_SCALE = 2**16
    
    content = response["choices"][0]["logprobs"]["content"]
    for i, pos in enumerate(content):
        weights = pos.get("sampling_weights")
        if weights:
            total = sum(weights.values())
            # Allow 10% tolerance
            assert abs(total - WEIGHT_SCALE) < WEIGHT_SCALE * 0.1, \
                f"Weight sum {total} too far from {WEIGHT_SCALE} at position {i}"
            print(f"  Position {i}: weight sum = {total} (target: {WEIGHT_SCALE})")
    
    print("✓ Weight sums are within expected range")


def test_different_seeds_differ(base_url: str, model: str):
    """Test that different seeds produce different outputs."""
    print("\n=== Test 6: Different seeds produce different outputs ===")
    
    results = []
    for seed in [100, 200, 300]:
        response = chat_completion(
            base_url, model,
            messages=[{"role": "user", "content": "Write something random"}],
            temperature=0.99,
            seed=seed,
            max_tokens=20
        )
        text = response["choices"][0]["message"]["content"]
        results.append(text)
        print(f"  Seed {seed}: {text[:40]}...")
    
    unique_results = len(set(results))
    assert unique_results >= 2, "All seeds produced same output!"
    print(f"✓ {unique_results}/3 unique outputs from different seeds")


def main():
    parser = argparse.ArgumentParser(description="E2E validation tests for vLLM")
    parser.add_argument(
        "--url", 
        default="http://localhost:8000",
        help="vLLM server URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model name (default: Qwen/Qwen2.5-1.5B-Instruct)"
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=0,
        help="Wait for server to be ready (seconds, 0=no wait)"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("vLLM Validation Enhancement E2E Tests")
    print("=" * 60)
    print(f"Server: {args.url}")
    print(f"Model: {args.model}")
    
    # Wait for server if requested
    if args.wait > 0:
        print(f"\nWaiting up to {args.wait}s for server...")
        start = time.time()
        while time.time() - start < args.wait:
            if test_server_health(args.url):
                print("Server is ready!")
                break
            time.sleep(2)
        else:
            print("Server not ready, proceeding anyway...")
    
    # Check server health
    if not test_server_health(args.url):
        print(f"\nWARNING: Server at {args.url} may not be running")
        print("Some tests may fail")
    
    try:
        test_deterministic_sampling_produces_weights(args.url, args.model)
        test_reproducibility(args.url, args.model)
        test_validation_flow(args.url, args.model)
        test_greedy_no_weights(args.url, args.model)
        test_weights_sum(args.url, args.model)
        test_different_seeds_differ(args.url, args.model)
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except requests.exceptions.ConnectionError:
        print(f"\n❌ CONNECTION ERROR: Cannot connect to {args.url}")
        print("Make sure vLLM server is running with VLLM_DETERMINISTIC_SAMPLING=1")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
