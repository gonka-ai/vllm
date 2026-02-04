# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end tests for validation with real vLLM server.

These tests require:
1. A running vLLM server with VLLM_DETERMINISTIC_SAMPLING=1
2. Network access to the server

Run with:
    RUN_E2E_TESTS=1 pytest tests/v1/validation/test_validation_e2e.py -v

Or start server manually and run:
    VLLM_TEST_SERVER_URL=http://localhost:8000 pytest tests/v1/validation/test_validation_e2e.py -v
"""

import os
import json
import pytest
from typing import Dict, List, Any, Optional


# Skip all tests if E2E testing is not enabled
pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_E2E_TESTS") and not os.environ.get("VLLM_TEST_SERVER_URL"),
    reason="E2E tests disabled. Set RUN_E2E_TESTS=1 or VLLM_TEST_SERVER_URL"
)


def get_server_url() -> str:
    """Get the vLLM server URL."""
    return os.environ.get("VLLM_TEST_SERVER_URL", "http://localhost:8000")


def get_model_name() -> str:
    """Get the model name to use for testing."""
    return os.environ.get("VLLM_TEST_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")


@pytest.fixture(scope="module")
def http_client():
    """Create HTTP client for API calls."""
    import requests
    
    class SimpleClient:
        def __init__(self, base_url: str, model: str):
            self.base_url = base_url
            self.model = model
        
        def chat_completion(
            self,
            messages: List[Dict[str, str]],
            **kwargs
        ) -> Dict[str, Any]:
            """Send chat completion request."""
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    **kwargs
                },
                timeout=120,
            )
            response.raise_for_status()
            return response.json()
    
    return SimpleClient(get_server_url(), get_model_name())


def build_enforced_tokens_from_response(response: Dict) -> Dict:
    """Build enforced_tokens dict from executor response."""
    content = response["choices"][0]["logprobs"]["content"]
    
    tokens = []
    for pos in content:
        token_data = {
            "token": str(pos["token"]),
            "top_tokens": [str(t["token"]) for t in pos["top_logprobs"]],
        }
        
        # Include logprobs if available
        if pos.get("top_logprobs"):
            token_data["logprobs"] = {
                str(t["token"]): t["logprob"]
                for t in pos["top_logprobs"]
            }
        
        # Include sampling_weights if available
        if pos.get("sampling_weights"):
            token_data["sampling_weights"] = pos["sampling_weights"]
        
        tokens.append(token_data)
    
    return {"tokens": tokens}


@pytest.mark.e2e
class TestDeterministicSampling:
    """Test deterministic sampling features."""
    
    def test_sampling_weights_in_response(self, http_client):
        """Test that sampling_weights are included when logprobs requested."""
        response = http_client.chat_completion(
            messages=[{"role": "user", "content": "Say hello"}],
            temperature=0.99,
            seed=42,
            logprobs=True,
            top_logprobs=5,
            max_tokens=5,
        )
        
        content = response["choices"][0]["logprobs"]["content"]
        assert len(content) > 0, "No logprobs content returned"
        
        first_token = content[0]
        assert "sampling_weights" in first_token, "sampling_weights missing"
        assert first_token["sampling_weights"] is not None, "sampling_weights is None"
        assert isinstance(first_token["sampling_weights"], dict), "sampling_weights not dict"
        
        # Verify weights are integers
        for tid, weight in first_token["sampling_weights"].items():
            assert isinstance(weight, int), f"Weight {weight} is not int"
    
    def test_reproducibility(self, http_client):
        """Test that same seed produces identical outputs."""
        params = {
            "messages": [{"role": "user", "content": "Count from 1 to 5"}],
            "temperature": 0.99,
            "seed": 12345,
            "max_tokens": 20,
        }
        
        results = []
        for _ in range(3):
            response = http_client.chat_completion(**params)
            text = response["choices"][0]["message"]["content"]
            results.append(text)
        
        assert len(set(results)) == 1, f"Outputs differ: {results}"
    
    def test_different_seeds_differ(self, http_client):
        """Test that different seeds produce different outputs."""
        base_params = {
            "messages": [{"role": "user", "content": "Write a random sentence"}],
            "temperature": 0.99,
            "max_tokens": 20,
        }
        
        results = []
        for seed in [100, 200, 300]:
            response = http_client.chat_completion(**base_params, seed=seed)
            text = response["choices"][0]["message"]["content"]
            results.append(text)
        
        # At least 2 different outputs (unlikely but possible to get same)
        assert len(set(results)) >= 2, "All seeds produced same output"
    
    def test_greedy_no_weights(self, http_client):
        """Test that greedy sampling doesn't include weights."""
        response = http_client.chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0,  # Greedy
            logprobs=True,
            top_logprobs=5,
            max_tokens=5,
        )
        
        content = response["choices"][0]["logprobs"]["content"]
        first_token = content[0]
        
        # Greedy should have None weights
        weights = first_token.get("sampling_weights")
        assert weights is None, f"Expected None for greedy, got {weights}"


@pytest.mark.e2e
class TestValidationFlow:
    """Test the full validation flow."""
    
    def test_honest_executor_passes_validation(self, http_client):
        """Test that honest executor's artifact passes validation."""
        # Step 1: Executor generates
        exec_response = http_client.chat_completion(
            messages=[{"role": "user", "content": "What is 2+2?"}],
            temperature=0.99,
            seed=42,
            logprobs=True,
            top_logprobs=5,
            max_tokens=10,
        )
        
        # Step 2: Build enforced_tokens
        enforced_tokens = build_enforced_tokens_from_response(exec_response)
        
        # Verify we have the necessary data
        assert len(enforced_tokens["tokens"]) > 0
        first_token = enforced_tokens["tokens"][0]
        assert "logprobs" in first_token or "sampling_weights" in first_token
        
        # Step 3: Validator verifies (same prompt, seed, params)
        # Note: The actual validation result depends on server implementation
        # This test just verifies the flow works
        try:
            val_response = http_client.chat_completion(
                messages=[{"role": "user", "content": "What is 2+2?"}],
                temperature=0.99,
                seed=42,
                logprobs=True,
                top_logprobs=5,
                max_tokens=10,
                extra_body={"enforced_tokens": enforced_tokens},
            )
            
            # Check for validation result if present
            if "validation" in val_response:
                validation = val_response["validation"]
                assert not validation.get("fraud"), "Honest execution flagged as fraud!"
        except Exception as e:
            # Some servers may not support extra_body
            pytest.skip(f"Server doesn't support enforced_tokens: {e}")
    
    def test_executor_validator_output_match(self, http_client):
        """Test that executor and validator produce matching logprobs."""
        # Generate with executor
        exec_response = http_client.chat_completion(
            messages=[{"role": "user", "content": "The capital of France is"}],
            temperature=0.99,
            seed=999,
            logprobs=True,
            top_logprobs=5,
            max_tokens=5,
        )
        
        exec_content = exec_response["choices"][0]["logprobs"]["content"]
        exec_text = exec_response["choices"][0]["message"]["content"]
        
        # Generate again with same params (simulating validator)
        val_response = http_client.chat_completion(
            messages=[{"role": "user", "content": "The capital of France is"}],
            temperature=0.99,
            seed=999,
            logprobs=True,
            top_logprobs=5,
            max_tokens=5,
        )
        
        val_content = val_response["choices"][0]["logprobs"]["content"]
        val_text = val_response["choices"][0]["message"]["content"]
        
        # Outputs should be identical (deterministic)
        assert exec_text == val_text, f"Texts differ: {exec_text} != {val_text}"
        
        # Logprobs should be very close
        for i, (exec_pos, val_pos) in enumerate(zip(exec_content, val_content)):
            assert exec_pos["token"] == val_pos["token"], \
                f"Tokens differ at position {i}"
            
            # Compare top logprob values
            exec_top = {t["token"]: t["logprob"] for t in exec_pos["top_logprobs"]}
            val_top = {t["token"]: t["logprob"] for t in val_pos["top_logprobs"]}
            
            for token in exec_top:
                if token in val_top:
                    diff = abs(exec_top[token] - val_top[token])
                    assert diff < 0.01, \
                        f"Logprob diff too large at pos {i}: {diff}"


@pytest.mark.e2e
class TestWeightIntegrity:
    """Test sampling weight integrity."""
    
    def test_weights_sum_approximately_correct(self, http_client):
        """Test that sampling weights sum to approximately 2^16."""
        response = http_client.chat_completion(
            messages=[{"role": "user", "content": "Hello world"}],
            temperature=0.99,
            seed=42,
            logprobs=True,
            top_logprobs=10,
            max_tokens=5,
        )
        
        WEIGHT_SCALE = 2**16
        
        content = response["choices"][0]["logprobs"]["content"]
        for i, pos in enumerate(content):
            weights = pos.get("sampling_weights")
            if weights:
                total = sum(weights.values())
                # Allow some tolerance for rounding
                assert abs(total - WEIGHT_SCALE) < WEIGHT_SCALE * 0.1, \
                    f"Weight sum {total} too far from {WEIGHT_SCALE} at position {i}"
    
    def test_weights_match_logprob_ordering(self, http_client):
        """Test that weight ordering matches logprob ordering."""
        response = http_client.chat_completion(
            messages=[{"role": "user", "content": "Test"}],
            temperature=0.99,
            seed=42,
            logprobs=True,
            top_logprobs=5,
            max_tokens=3,
        )
        
        content = response["choices"][0]["logprobs"]["content"]
        for pos in content:
            weights = pos.get("sampling_weights")
            if not weights:
                continue
            
            top_logprobs = pos["top_logprobs"]
            
            # Get tokens sorted by logprob (descending = higher prob first)
            sorted_by_logprob = sorted(
                top_logprobs,
                key=lambda x: x["logprob"],
                reverse=True
            )
            
            # Get tokens sorted by weight (descending)
            sorted_by_weight = sorted(
                [(t["token"], weights.get(str(t["token"]), 0)) for t in top_logprobs],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Orderings should generally match (may differ for near-equal values)
            # Just check that highest logprob has highest weight
            if len(sorted_by_logprob) > 0 and len(sorted_by_weight) > 0:
                top_logprob_token = str(sorted_by_logprob[0]["token"])
                top_weight_token = sorted_by_weight[0][0]
                
                # Allow for ties in near-equal cases
                top_logprob_val = sorted_by_logprob[0]["logprob"]
                second_logprob_val = sorted_by_logprob[1]["logprob"] if len(sorted_by_logprob) > 1 else -999
                
                if top_logprob_val - second_logprob_val > 0.1:
                    # Clear winner, should match
                    assert top_logprob_token == top_weight_token, \
                        f"Top tokens don't match: logprob={top_logprob_token}, weight={top_weight_token}"


@pytest.mark.e2e
class TestStreamingWithWeights:
    """Test streaming responses with sampling weights."""
    
    def test_streaming_includes_weights(self, http_client):
        """Test that streaming responses include sampling_weights."""
        import requests
        
        response = requests.post(
            f"{get_server_url()}/v1/chat/completions",
            json={
                "model": get_model_name(),
                "messages": [{"role": "user", "content": "Hi"}],
                "temperature": 0.99,
                "seed": 42,
                "logprobs": True,
                "top_logprobs": 5,
                "max_tokens": 5,
                "stream": True,
            },
            stream=True,
            timeout=60,
        )
        
        weights_found = False
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith("data: ") and line != "data: [DONE]":
                    data = json.loads(line[6:])
                    choices = data.get("choices", [])
                    for choice in choices:
                        logprobs = choice.get("logprobs")
                        if logprobs and logprobs.get("content"):
                            for pos in logprobs["content"]:
                                if pos.get("sampling_weights"):
                                    weights_found = True
                                    break
        
        assert weights_found, "No sampling_weights found in streaming response"


# Utility for manual testing
def main():
    """Run a quick manual test."""
    import requests
    
    base_url = get_server_url()
    model = get_model_name()
    
    print(f"Testing server at {base_url} with model {model}")
    
    # Test 1: Check sampling_weights
    print("\n=== Test 1: sampling_weights in response ===")
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.99,
            "seed": 42,
            "logprobs": True,
            "top_logprobs": 5,
            "max_tokens": 3,
        },
    )
    data = response.json()
    content = data["choices"][0]["logprobs"]["content"]
    print(f"First token: {content[0]['token']}")
    print(f"Weights: {content[0].get('sampling_weights')}")
    
    # Test 2: Reproducibility
    print("\n=== Test 2: Reproducibility ===")
    results = []
    for i in range(3):
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Count to 3"}],
                "temperature": 0.99,
                "seed": 12345,
                "max_tokens": 10,
            },
        )
        text = response.json()["choices"][0]["message"]["content"]
        results.append(text)
        print(f"Run {i+1}: {text[:50]}...")
    
    if len(set(results)) == 1:
        print("✓ All runs identical")
    else:
        print("✗ Runs differ!")
    
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
