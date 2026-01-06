#!/usr/bin/env python3
"""
Smoke test for PoC using vLLM scheduler integration.

Usage:
    VLLM_USE_V1=0 python scripts/poc_smoke_test.py

This script:
1. Loads Qwen3-0.6B model using vLLM v0
2. Submits PoC requests through the scheduler using add_request(PoCParams)
3. Verifies outputs are PoCRequestOutput with correct distances
4. Tests determinism (same nonce = same distance)
5. Tests independence (different public_key = different distance)
"""

import os
import sys
import time

# Force v0 engine
os.environ["VLLM_USE_V1"] = "0"

from vllm import LLM
from vllm.poc.poc_params import PoCParams
from vllm.outputs import PoCRequestOutput


def main():
    print("=" * 60)
    print("PoC Scheduler Integration Smoke Test")
    print("=" * 60)
    
    # Load model
    print("\n[1/5] Loading model...")
    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        enforce_eager=True,
        gpu_memory_utilization=0.3,
        max_model_len=256,
    )
    
    engine = llm.llm_engine
    model_config = engine.model_config
    
    print(f"   Model: {model_config.model}")
    print(f"   Hidden size: {model_config.get_hidden_size()}")
    print(f"   Engine type: {type(engine).__name__}")
    
    # Test config
    block_hash = "smoke_test_block_hash_12345"
    public_key = "test_node_pubkey"
    block_height = 100
    r_target = 1.5
    seq_len = 32
    
    print("\n[2/5] Submitting PoC requests through scheduler...")
    print(f"   Block hash: {block_hash}")
    print(f"   Public key: {public_key}")
    print(f"   r_target: {r_target}")
    print(f"   seq_len: {seq_len}")
    
    # Submit multiple nonces as individual requests
    nonces = [0, 1, 2, 3]
    results = []
    start_time = time.time()
    
    for nonce in nonces:
        poc_params = PoCParams(
            block_hash=block_hash,
            public_key=public_key,
            block_height=block_height,
            nonce=nonce,
            r_target=r_target,
            seq_len=seq_len,
            return_vectors=False,
        )
        
        request_id = f"poc-test-{nonce}"
        
        # Create dummy prompt (embeddings are generated on GPU)
        prompt = {"prompt_token_ids": [0] * seq_len}
        
        # Add request to scheduler
        engine.add_request(
            request_id=request_id,
            prompt=prompt,
            params=poc_params,
        )
    
    print(f"   Submitted {len(nonces)} PoC requests")
    
    # Process requests through scheduler
    print("\n[3/5] Processing requests through scheduler...")
    
    while engine.has_unfinished_requests():
        step_outputs = engine.step()
        for output in step_outputs:
            if isinstance(output, PoCRequestOutput):
                results.append({
                    "request_id": output.request_id,
                    "nonce": output.outputs.nonce,
                    "distance": output.outputs.distance,
                    "finished": output.finished,
                })
                print(f"   Got result: nonce={output.outputs.nonce}, distance={output.outputs.distance:.6f}")
    
    elapsed = time.time() - start_time
    print(f"   Processed {len(results)} results in {elapsed:.3f}s")
    print(f"   Rate: {len(results) / elapsed:.1f} nonces/s")
    
    # Verify results
    print("\n[4/5] Verifying results...")
    
    all_distances = [r["distance"] for r in results]
    print(f"   Distances: {[f'{d:.4f}' for d in all_distances]}")
    
    valid_count = sum(1 for d in all_distances if d < r_target)
    print(f"   Valid (< {r_target}): {valid_count}/{len(results)}")
    
    # Test determinism - rerun same nonces
    print("\n   Testing determinism (rerunning nonce 0)...")
    poc_params_repeat = PoCParams(
        block_hash=block_hash,
        public_key=public_key,
        block_height=block_height,
        nonce=0,
        r_target=r_target,
        seq_len=seq_len,
    )
    
    engine.add_request(
        request_id="poc-test-repeat",
        prompt={"prompt_token_ids": [0] * seq_len},
        params=poc_params_repeat,
    )
    
    repeat_result = None
    while engine.has_unfinished_requests():
        step_outputs = engine.step()
        for output in step_outputs:
            if isinstance(output, PoCRequestOutput):
                repeat_result = output.outputs.distance
    
    original_distance = results[0]["distance"]
    diff = abs(original_distance - repeat_result)
    # Use larger tolerance for bfloat16 precision
    determinism_match = diff < 0.01
    print(f"   Original distance: {original_distance:.6f}")
    print(f"   Repeat distance:   {repeat_result:.6f}")
    print(f"   Diff: {diff:.6f}")
    print(f"   Determinism: {'MATCH' if determinism_match else 'MISMATCH'}")
    
    # Test different public key
    print("\n   Testing different public key...")
    poc_params_other = PoCParams(
        block_hash=block_hash,
        public_key="different_pubkey",  # Different public key
        block_height=block_height,
        nonce=0,
        r_target=r_target,
        seq_len=seq_len,
    )
    
    engine.add_request(
        request_id="poc-test-other-pk",
        prompt={"prompt_token_ids": [0] * seq_len},
        params=poc_params_other,
    )
    
    other_result = None
    while engine.has_unfinished_requests():
        step_outputs = engine.step()
        for output in step_outputs:
            if isinstance(output, PoCRequestOutput):
                other_result = output.outputs.distance
    
    distances_differ = abs(original_distance - other_result) > 0.01
    print(f"   Original public_key distance: {original_distance:.6f}")
    print(f"   Different public_key distance: {other_result:.6f}")
    print(f"   Different: {distances_differ}")
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    checks = [
        ("All results are PoCRequestOutput", len(results) == len(nonces)),
        ("Distances in valid range [0, 2]", all(0 <= d <= 2 for d in all_distances)),
        ("Deterministic (same nonce = same distance)", determinism_match),
        ("Different pubkey -> different distances", distances_differ),
        ("Scheduler integration works", len(results) > 0),
    ]
    
    all_passed = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        all_passed = all_passed and passed
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL CHECKS PASSED!")
    else:
        print("SOME CHECKS FAILED!")
        sys.exit(1)
    print("=" * 60)
    
    # Cleanup
    del llm


if __name__ == "__main__":
    main()
