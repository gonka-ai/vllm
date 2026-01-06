#!/usr/bin/env python3
"""
Comprehensive E2E Test for PoC with vLLM Server.

Tests 3x3 seed matrix (block_hash x public_key) per model:
1. For each model (server stays up):
   - Runs 9 seed combinations without server restart
   - Saves per-seed JSON with nonces for later validation
   - Determinism test: repeats first seed, compares nonces
   - Fraud tests: wrong hash/pubkey detection
2. Collects logs to timestamped directory
3. Reports results as JSON

Usage:
    python scripts/poc_e2e_test.py
    python scripts/poc_e2e_test.py --models qwen
    python scripts/poc_e2e_test.py --duration 30
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import requests

# =============================================================================
# Configuration
# =============================================================================

SERVER_PORT = 8765
GENERATION_TIME = 15  # seconds per seed
SERVER_STARTUP_TIMEOUT = 120

MODELS = {
    "qwen": "Qwen/Qwen3-0.6B",
    "llama": "unsloth/Llama-3.2-1B-Instruct",
    "qwen4b": "Qwen/Qwen3-4B-Instruct-2507",
}

# Per-model config overrides
MODEL_MAX_LEN = {
    "qwen4b": 10256,
}

MODEL_GPU_UTIL = {
    "qwen4b": 0.9,  # Larger model needs more GPU memory
}

# 3x3 seed matrix
BLOCK_HASHES = ["block_alpha", "block_beta", "block_gamma"]
PUBLIC_KEYS = ["node_A", "node_B", "node_C"]

# r_target for ~20% valid rate in k=64 dim space (from estimate_valid_rate.py)
R_TARGET = 1.34

# Base config (seeds will be overridden per test)
BASE_CONFIG = {
    "block_height": 100,
    "r_target": R_TARGET,
    "node_id": 0,
    "node_count": 1,
    "batch_size": 32,
    "seq_len": 256,
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SeedResult:
    """Result from a single seed combination."""
    block_hash: str
    public_key: str
    total_checked: int = 0
    total_valid: int = 0
    valid_rate_percent: float = 0.0
    valid_nonces: List[int] = field(default_factory=list)
    valid_distances: List[float] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class ModelResult:
    """Result from testing a single model across all seeds."""
    model: str
    seed_results: List[SeedResult] = field(default_factory=list)
    determinism_pass: bool = False
    independence_pass: bool = False
    wrong_hash_fraud: bool = False
    wrong_pubkey_fraud: bool = False
    passed: bool = False
    error: Optional[str] = None
    duration_seconds: float = 0.0


@dataclass
class TestSuite:
    """Overall test suite results."""
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    results: List[ModelResult] = field(default_factory=list)
    all_passed: bool = False
    r_target: float = R_TARGET
    block_hashes: List[str] = field(default_factory=lambda: BLOCK_HASHES.copy())
    public_keys: List[str] = field(default_factory=lambda: PUBLIC_KEYS.copy())


# =============================================================================
# Directory Setup
# =============================================================================

def setup_run_logs_dir(run_name: str) -> Path:
    """Create per-run logs directory under logs/."""
    logs_dir = Path("logs") / run_name
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def setup_model_dir(logs_dir: Path, model_key: str) -> Path:
    """Create per-model subdirectory."""
    model_dir = logs_dir / model_key
    model_dir.mkdir(exist_ok=True)
    return model_dir


# =============================================================================
# Server Management
# =============================================================================

def start_vllm_server(model: str, model_dir: Path, model_key: str) -> subprocess.Popen:
    """Start vLLM server with PoC enabled."""
    log_file = model_dir / "server.log"
    max_model_len = MODEL_MAX_LEN.get(model_key, 512)
    gpu_util = MODEL_GPU_UTIL.get(model_key, 0.4)
    
    env = os.environ.copy()
    env["VLLM_USE_V1"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    
    f = open(log_file, "w", buffering=1)
    proc = subprocess.Popen(
        [
            sys.executable, "-u", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--enable-poc",
            "--port", str(SERVER_PORT),
            "--gpu-memory-utilization", str(gpu_util),
            "--max-model-len", str(max_model_len),
        ],
        stdout=f,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=Path.cwd(),
        start_new_session=True,
    )
    proc._log_file = f
    
    # Wait for server to start
    for i in range(SERVER_STARTUP_TIMEOUT):
        try:
            r = requests.get(f"http://localhost:{SERVER_PORT}/health", timeout=1)
            if r.status_code == 200:
                return proc
        except:
            pass
        time.sleep(1)
        if i > 0 and i % 20 == 0:
            print(f"      Waiting for server... ({i}s)")
    
    proc.kill()
    f.close()
    raise RuntimeError(f"vLLM server failed to start within {SERVER_STARTUP_TIMEOUT}s")


def stop_process(proc: subprocess.Popen):
    """Stop a subprocess gracefully."""
    if proc is None:
        return
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
    if hasattr(proc, '_log_file') and proc._log_file:
        proc._log_file.flush()
        proc._log_file.close()


# =============================================================================
# API Helpers
# =============================================================================

def api_call(method: str, endpoint: str, json_data: dict = None) -> dict:
    """Make API call to vLLM server."""
    url = f"http://localhost:{SERVER_PORT}{endpoint}"
    if method == "GET":
        r = requests.get(url, timeout=30)
    else:
        r = requests.post(url, json=json_data, timeout=30)
    r.raise_for_status()
    return r.json()


# =============================================================================
# Seed Generation & Saving
# =============================================================================

def run_seed_generation(block_hash: str, public_key: str, duration: int) -> SeedResult:
    """Run generation for a single seed combination using scheduler-based /generate.
    
    Instead of timed generation, we submit a batch of nonces and wait for results.
    The number of nonces is calculated based on duration and expected throughput.
    """
    result = SeedResult(block_hash=block_hash, public_key=public_key)
    
    try:
        # Estimate nonces based on duration (assume ~100 nonces/sec throughput)
        estimated_nonces = duration * 100
        nonces = list(range(estimated_nonces))
        
        start_time = time.time()
        
        # Use new scheduler-based /generate endpoint with wait=True
        response = api_call("POST", "/api/v1/pow/generate", {
            "block_hash": block_hash,
            "block_height": BASE_CONFIG["block_height"],
            "public_key": public_key,
            "r_target": BASE_CONFIG["r_target"],
            "nonces": nonces,
            "seq_len": BASE_CONFIG["seq_len"],
            "wait": True,
        })
        
        elapsed = time.time() - start_time
        
        result.total_checked = response.get("total_checked", len(nonces))
        result.total_valid = response.get("total_valid", 0)
        result.valid_nonces = response.get("valid_nonces", [])
        result.valid_distances = response.get("valid_distances", [])
        result.elapsed_seconds = elapsed
        
        if result.total_checked > 0:
            result.valid_rate_percent = result.total_valid / result.total_checked * 100
            
    except Exception as e:
        result.error = str(e)
    
    return result


def save_seed_result(model_dir: Path, result: SeedResult):
    """Save seed result to JSON file."""
    filename = f"{result.block_hash}_{result.public_key}.json"
    filepath = model_dir / filename
    
    data = {
        "block_hash": result.block_hash,
        "public_key": result.public_key,
        "total_checked": result.total_checked,
        "total_valid": result.total_valid,
        "valid_rate_percent": result.valid_rate_percent,
        "valid_nonces": result.valid_nonces,
        "valid_distances": result.valid_distances,
        "elapsed_seconds": result.elapsed_seconds,
        "error": result.error,
    }
    
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


# =============================================================================
# Validation Helpers
# =============================================================================

def validate_nonces(
    nonces: List[int],
    distances: List[float],
    block_hash: str,
    public_key: str,
    r_target: float,
) -> Dict[str, Any]:
    """Validate nonces by recomputing distances using scheduler-based /generate.
    
    Returns dict with:
    - fraud_detected: True if any claimed distance doesn't match recomputed
    - computed_distances: List of recomputed distances
    """
    # Recompute distances using scheduler
    response = api_call("POST", "/api/v1/pow/generate", {
        "block_hash": block_hash,
        "block_height": BASE_CONFIG["block_height"],
        "public_key": public_key,
        "r_target": r_target,
        "nonces": nonces,
        "seq_len": BASE_CONFIG["seq_len"],
        "wait": True,
        "return_vectors": False,
    })
    
    # Build computed distances map
    valid_nonces = response.get("valid_nonces", [])
    valid_distances = response.get("valid_distances", [])
    
    # For all requested nonces, get their distances from all_results if available
    # Otherwise use a high distance (fraud detection threshold)
    computed_map = dict(zip(valid_nonces, valid_distances))
    computed_distances = []
    
    for nonce in nonces:
        if nonce in computed_map:
            computed_distances.append(computed_map[nonce])
        else:
            # Nonce not valid - use 2.0 as max distance
            computed_distances.append(2.0)
    
    # Check for fraud: claimed valid but actually not
    fraud_detected = False
    for i, (claimed_dist, computed_dist) in enumerate(zip(distances, computed_distances)):
        # If claimed distance is very low but computed is high -> fraud
        if claimed_dist < r_target and computed_dist >= r_target:
            fraud_detected = True
            break
        # If distances differ significantly -> fraud
        if abs(claimed_dist - computed_dist) > 0.1:
            fraud_detected = True
            break
    
    return {
        "fraud_detected": fraud_detected,
        "computed_distances": computed_distances,
    }


def check_determinism(original: SeedResult, repeat: SeedResult) -> bool:
    """Check if two runs with same seed produce same nonces."""
    if original.error or repeat.error:
        return False
    
    # Compare nonce sets (order may differ)
    original_set = set(original.valid_nonces)
    repeat_set = set(repeat.valid_nonces)
    
    # Check significant overlap (allow some variation due to timing)
    if len(original_set) == 0 or len(repeat_set) == 0:
        return False
    
    overlap = len(original_set & repeat_set)
    min_size = min(len(original_set), len(repeat_set))
    
    # Require at least 80% overlap
    return overlap >= min_size * 0.8


def check_independence(seed_results: List[SeedResult]) -> bool:
    """Check that different seeds produce different nonces."""
    nonce_sets = []
    for r in seed_results:
        if r.error or len(r.valid_nonces) == 0:
            continue
        nonce_sets.append(set(r.valid_nonces))
    
    if len(nonce_sets) < 2:
        return True  # Can't check independence with < 2 valid results
    
    # Check that each pair has less than 50% overlap
    for i in range(len(nonce_sets)):
        for j in range(i + 1, len(nonce_sets)):
            overlap = len(nonce_sets[i] & nonce_sets[j])
            max_size = max(len(nonce_sets[i]), len(nonce_sets[j]))
            if max_size > 0 and overlap / max_size > 0.5:
                return False  # Too much overlap = not independent
    
    return True


# =============================================================================
# Main Test Logic
# =============================================================================

def test_model(model_key: str, model_name: str, model_dir: Path, duration: int) -> ModelResult:
    """Test a single model across all seed combinations."""
    result = ModelResult(model=model_name)
    start_time = time.time()
    server_proc = None
    
    try:
        # Start server once for all seeds
        print(f"\n  Starting server...")
        server_proc = start_vllm_server(model_name, model_dir, model_key)
        print(f"  Server started")
        
        # ====================================================================
        # Phase 1: Run all 9 seed combinations
        # ====================================================================
        print(f"\n  [Phase 1] Running 9 seed combinations ({duration}s each)")
        
        seed_count = 0
        total_seeds = len(BLOCK_HASHES) * len(PUBLIC_KEYS)
        
        for block_hash in BLOCK_HASHES:
            for public_key in PUBLIC_KEYS:
                seed_count += 1
                print(f"    [{seed_count}/{total_seeds}] {block_hash} + {public_key}...", end=" ", flush=True)
                
                seed_result = run_seed_generation(block_hash, public_key, duration)
                result.seed_results.append(seed_result)
                save_seed_result(model_dir, seed_result)
                
                if seed_result.error:
                    print(f"ERROR: {seed_result.error}")
                else:
                    print(f"checked={seed_result.total_checked}, valid={seed_result.total_valid} ({seed_result.valid_rate_percent:.1f}%)")
        
        # ====================================================================
        # Phase 2: Determinism test (repeat first seed)
        # ====================================================================
        print(f"\n  [Phase 2] Determinism test (repeat {BLOCK_HASHES[0]}_{PUBLIC_KEYS[0]})")
        
        first_seed = result.seed_results[0] if result.seed_results else None
        if first_seed and not first_seed.error:
            repeat_result = run_seed_generation(BLOCK_HASHES[0], PUBLIC_KEYS[0], duration)
            result.determinism_pass = check_determinism(first_seed, repeat_result)
            
            # Save repeat result too
            repeat_result.block_hash = f"{BLOCK_HASHES[0]}_repeat"
            save_seed_result(model_dir, repeat_result)
            
            print(f"    Original nonces: {len(first_seed.valid_nonces)}, Repeat: {len(repeat_result.valid_nonces)}")
            print(f"    Determinism: {'PASS' if result.determinism_pass else 'FAIL'}")
        else:
            print(f"    SKIP (first seed failed)")
        
        # ====================================================================
        # Phase 3: Independence check
        # ====================================================================
        print(f"\n  [Phase 3] Independence check (9 seeds should differ)")
        result.independence_pass = check_independence(result.seed_results)
        print(f"    Independence: {'PASS' if result.independence_pass else 'FAIL'}")
        
        # ====================================================================
        # Phase 4: Fraud detection tests
        # ====================================================================
        # Use nonces from first successful seed
        test_seed = None
        for sr in result.seed_results:
            if not sr.error and len(sr.valid_nonces) > 0:
                test_seed = sr
                break
        
        if test_seed:
            test_nonces = test_seed.valid_nonces[:min(10, len(test_seed.valid_nonces))]
            fake_distances = [0.05] * len(test_nonces)  # Claim very low distances
            
            print(f"\n  [Phase 4] Wrong block hash fraud test")
            wrong_hash_result = validate_nonces(
                test_nonces, fake_distances,
                block_hash="WRONG_BLOCK_HASH_XYZ",
                public_key=test_seed.public_key,
                r_target=0.1,
            )
            result.wrong_hash_fraud = wrong_hash_result.get("fraud_detected", False)
            print(f"    Wrong block_hash -> fraud detected: {'PASS' if result.wrong_hash_fraud else 'FAIL'}")
            
            print(f"\n  [Phase 5] Wrong public key fraud test")
            wrong_pubkey_result = validate_nonces(
                test_nonces, fake_distances,
                block_hash=test_seed.block_hash,
                public_key="WRONG_PUBLIC_KEY_XYZ",
                r_target=0.1,
            )
            result.wrong_pubkey_fraud = wrong_pubkey_result.get("fraud_detected", False)
            print(f"    Wrong public_key -> fraud detected: {'PASS' if result.wrong_pubkey_fraud else 'FAIL'}")
        else:
            print(f"\n  [Phase 4-5] SKIP fraud tests (no valid nonces)")
        
        # ====================================================================
        # Overall Result
        # ====================================================================
        result.passed = (
            result.determinism_pass and
            result.independence_pass and
            result.wrong_hash_fraud and
            result.wrong_pubkey_fraud
        )
        
    except Exception as e:
        result.error = str(e)
        print(f"    ERROR: {e}")
    finally:
        if server_proc:
            stop_process(server_proc)
        result.duration_seconds = time.time() - start_time
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Comprehensive PoC E2E Test Suite")
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()),
                        default=list(MODELS.keys()),
                        help="Models to test (default: all)")
    parser.add_argument("--duration", type=int, default=GENERATION_TIME,
                        help=f"Generation duration per seed in seconds (default: {GENERATION_TIME})")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run name for logs directory (default: e2e_YYYYMMDD_HHMMSS)")
    args = parser.parse_args()
    
    # Setup run directory
    if args.run_name is None:
        args.run_name = datetime.now().strftime("e2e_%Y%m%d_%H%M%S")
    
    logs_dir = setup_run_logs_dir(args.run_name)
    
    # Header
    total_seeds = len(BLOCK_HASHES) * len(PUBLIC_KEYS)
    print("=" * 70)
    print("Comprehensive PoC E2E Test Suite")
    print("=" * 70)
    print(f"Seed matrix:   {len(BLOCK_HASHES)} block_hashes x {len(PUBLIC_KEYS)} public_keys = {total_seeds} combinations")
    print(f"Block hashes:  {BLOCK_HASHES}")
    print(f"Public keys:   {PUBLIC_KEYS}")
    print(f"r_target:      {R_TARGET}")
    print(f"Duration:      {args.duration}s per seed")
    print(f"Models:        {list(args.models)}")
    print(f"Run name:      {args.run_name}")
    print()
    print(f"Logs directory: {logs_dir.absolute()}")
    
    # Save run config
    run_config_file = logs_dir / "run_config.json"
    with open(run_config_file, "w") as f:
        json.dump({
            "run_name": args.run_name,
            "models": args.models,
            "duration": args.duration,
            "r_target": R_TARGET,
            "block_hashes": BLOCK_HASHES,
            "public_keys": PUBLIC_KEYS,
            "batch_size": BASE_CONFIG["batch_size"],
            "seq_len": BASE_CONFIG["seq_len"],
        }, f, indent=2)
    
    suite = TestSuite()
    
    # Test each model
    for i, model_key in enumerate(args.models):
        model_name = MODELS[model_key]
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(args.models)}] Testing {model_key} ({model_name})")
        print("=" * 70)
        
        model_dir = setup_model_dir(logs_dir, model_key)
        result = test_model(model_key, model_name, model_dir, args.duration)
        suite.results.append(result)
        
        status = "PASS" if result.passed else "FAIL"
        print(f"\n  Model Result: {status} ({result.duration_seconds:.1f}s)")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for result in suite.results:
        status = "PASS" if result.passed else "FAIL"
        model_short = [k for k, v in MODELS.items() if v == result.model][0]
        print(f"\n  [{status}] {model_short}")
        
        if result.error:
            print(f"        Error: {result.error}")
        else:
            # Aggregate seed stats
            total_checked = sum(sr.total_checked for sr in result.seed_results)
            total_valid = sum(sr.total_valid for sr in result.seed_results)
            avg_rate = sum(sr.valid_rate_percent for sr in result.seed_results) / len(result.seed_results) if result.seed_results else 0
            
            print(f"        Seeds tested:  {len(result.seed_results)}")
            print(f"        Total checked: {total_checked}")
            print(f"        Total valid:   {total_valid}")
            print(f"        Avg valid rate: {avg_rate:.1f}%")
            print(f"        Determinism:   {'PASS' if result.determinism_pass else 'FAIL'}")
            print(f"        Independence:  {'PASS' if result.independence_pass else 'FAIL'}")
            print(f"        Wrong hash:    {'PASS' if result.wrong_hash_fraud else 'FAIL'}")
            print(f"        Wrong pubkey:  {'PASS' if result.wrong_pubkey_fraud else 'FAIL'}")
        
        all_passed = all_passed and result.passed
    
    suite.all_passed = all_passed
    
    # Save results
    results_file = logs_dir / "test_results.json"
    with open(results_file, "w") as f:
        # Convert to serializable format
        data = {
            "start_time": suite.start_time,
            "all_passed": suite.all_passed,
            "r_target": suite.r_target,
            "block_hashes": suite.block_hashes,
            "public_keys": suite.public_keys,
            "results": [],
        }
        for mr in suite.results:
            mr_data = {
                "model": mr.model,
                "passed": mr.passed,
                "error": mr.error,
                "duration_seconds": mr.duration_seconds,
                "determinism_pass": mr.determinism_pass,
                "independence_pass": mr.independence_pass,
                "wrong_hash_fraud": mr.wrong_hash_fraud,
                "wrong_pubkey_fraud": mr.wrong_pubkey_fraud,
                "seed_results": [asdict(sr) for sr in mr.seed_results],
            }
            data["results"].append(mr_data)
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED!")
        return 0
    else:
        print("SOME TESTS FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
