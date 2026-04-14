"""
Cross-machine reproducibility test: float16 vs float32 vs float64 vs Decimal.

Pipelines computing logprobs -> integer weights -> SHA256 sampling:
  - FLOAT16  (numpy float16)  -- simulates GPU half precision
  - FLOAT32  (numpy float32)  -- simulates GPU single precision
  - FLOAT64  (Python math)    -- CPU double precision
  - DECIMAL p=10,18,34,50     -- guaranteed cross-platform, various precisions

The processing pipeline (logprobs -> weights) matches the documented design in
docs/DETERMINISTIC_SAMPLING_VALIDATION.md and the implementation in
vllm/v1/sample/deterministic_utils.py:
  1. Temperature scaling: logprob / T
  2. Softmax with log-sum-exp stability shift
  3. top_k filtering (keep K highest-prob tokens)
  4. top_p filtering (cumulative prob cutoff)
  5. min_p filtering (threshold = max_prob * min_p)
  6. Re-normalize after filtering
  7. Quantize: int_weight = round(prob * SCALE)
  8. Fix total to exactly SCALE (residual -> max-weight token, ties by tid)
  9. Sample via SHA256 counter-mode RNG with unbiased rejection sampling

The SHA256 RNG matches deterministic_utils.Sha256CounterRNG:
  u64 = first_8_bytes(SHA256(seed_bytes || counter_be_u64))

Run on different machines and compare hashes.

Usage:
  python scripts/benchmark_decimal_vs_native.py
"""

import hashlib
import json
import math
import platform
import struct
import sys
import time
from decimal import Decimal, ROUND_HALF_EVEN, localcontext

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_LOGPROBS = 5
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = None       # set to int to enable top_k filtering
MIN_P = None       # set to float to enable min_p filtering
SCALE = 2 ** 32
N_POSITIONS = 100_000
SEED = 42
RNG_SEED = "benchmark_seed|12345"
DECIMAL_PRECISIONS = [10, 18, 34, 50]

# ---------------------------------------------------------------------------
# Expected hashes (from Linux x86_64, glibc 2.39, Python 3.12, 100k positions)
# ---------------------------------------------------------------------------
EXPECTED = {
    "INPUT":              "e02dfe7227488678756ae081a1ee8247886643901240df44355cb90a3d6ca7a6",
    "FLOAT64 weights":    "d3de3fa4012ae83b974b3e53b805bdf83ae5d891172a95b789351323a61736ec",
    "FLOAT64 tokens":     "110e0331229f55cce77ce361ea8bf45f104d37c69526ffd97e61e4fa40af9d1c",
    "DECIMAL10 weights":  "c13c4bbd941484c2bb320d02ba8e18c1863e31b9282864549a1ea7ec44beb8ad",
    "DECIMAL10 tokens":   "110e0331229f55cce77ce361ea8bf45f104d37c69526ffd97e61e4fa40af9d1c",
    "DECIMAL18 weights":  "d3de3fa4012ae83b974b3e53b805bdf83ae5d891172a95b789351323a61736ec",
    "DECIMAL18 tokens":   "110e0331229f55cce77ce361ea8bf45f104d37c69526ffd97e61e4fa40af9d1c",
    "DECIMAL34 weights":  "d3de3fa4012ae83b974b3e53b805bdf83ae5d891172a95b789351323a61736ec",
    "DECIMAL34 tokens":   "110e0331229f55cce77ce361ea8bf45f104d37c69526ffd97e61e4fa40af9d1c",
    "DECIMAL50 weights":  "d3de3fa4012ae83b974b3e53b805bdf83ae5d891172a95b789351323a61736ec",
    "DECIMAL50 tokens":   "110e0331229f55cce77ce361ea8bf45f104d37c69526ffd97e61e4fa40af9d1c",
}


# ---------------------------------------------------------------------------
# SHA256 Counter-Mode RNG (matches deterministic_utils.Sha256CounterRNG)
# ---------------------------------------------------------------------------
def _u64_be(x: int) -> bytes:
    """Encode uint64 as big-endian 8 bytes."""
    return struct.pack(">Q", x & 0xFFFFFFFFFFFFFFFF)


class Sha256CounterRNG:
    """
    Portable, fully-specified RNG:
      u64 = first_8_bytes(SHA256(seed_bytes || counter_be_u64))

    Produces identical sequences in Python and Go given the same seed.
    The counter advances by 1 on each call to next_u64().
    """

    def __init__(self, seed_bytes: bytes, counter: int = 0):
        self.seed_bytes = seed_bytes
        self.counter = counter

    @classmethod
    def from_seed_string(cls, seed: str) -> "Sha256CounterRNG":
        return cls(seed.encode("utf-8"), 0)

    def next_u64(self) -> int:
        h = hashlib.sha256(self.seed_bytes + _u64_be(self.counter)).digest()
        self.counter += 1
        return int.from_bytes(h[:8], byteorder="big", signed=False)


def uint64_below(rng: Sha256CounterRNG, n: int) -> int:
    """Unbiased draw in [0, n) from 64-bit uniform values via rejection sampling."""
    if n <= 0:
        raise ValueError("n must be > 0")
    two64 = 1 << 64
    limit = two64 - (two64 % n)
    while True:
        x = rng.next_u64()
        if x < limit:
            return x % n


def sample_categorical_weights(weights: list, rng: Sha256CounterRNG) -> int:
    """
    Deterministic categorical sampler on non-negative integer weights.
    Uses unbiased rejection sampling (uint64_below) then linear scan
    over the cumulative weight distribution.
    """
    if not weights:
        raise ValueError("weights is empty")
    total = 0
    last_nonzero = -1
    for i, w in enumerate(weights):
        if w < 0:
            raise ValueError(f"Negative weight at index {i}: {w}")
        if w > 0:
            last_nonzero = i
        total += w
    if total <= 0:
        return len(weights) - 1

    r = uint64_below(rng, total)
    cum = 0
    for i, w in enumerate(weights):
        cum += w
        if r < cum:
            return i
    return last_nonzero if last_nonzero >= 0 else len(weights) - 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sha256_hex(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

def _sha256_floats(seed_str: str, count: int) -> list:
    values = []
    for i in range(count):
        h = hashlib.sha256(f"{seed_str}|{i}".encode()).digest()
        u64 = int.from_bytes(h[:8], "big")
        values.append((u64 >> 11) / (1 << 53))
    return values

def generate_token_positions(n_positions: int, n_logprobs: int, seed: int = SEED):
    positions = []
    for pos_idx in range(n_positions):
        tids = []
        for j in range(n_logprobs):
            h = hashlib.sha256(f"tid|{seed}|{pos_idx}|{j}".encode()).digest()
            tid = int.from_bytes(h[:4], "big") % 150_000
            tids.append(tid)
        seen = set()
        for j in range(len(tids)):
            while tids[j] in seen:
                tids[j] = (tids[j] + 1) % 150_000
            seen.add(tids[j])

        raw_floats = _sha256_floats(f"lp|{seed}|{pos_idx}", n_logprobs)
        logprobs_raw = sorted(
            [f * (-8.0 + 0.01) + (-0.01) for f in raw_floats],
            reverse=True,
        )
        position = {str(tid): lp for tid, lp in zip(tids, logprobs_raw)}
        positions.append(position)
    return positions

def hash_inputs(positions_str):
    blob = json.dumps(positions_str, sort_keys=True, separators=(",", ":"))
    return sha256_hex(blob)

def hash_weights(all_weights):
    blob = json.dumps(all_weights, sort_keys=True, separators=(",", ":"))
    return sha256_hex(blob)

def hash_tokens(tokens):
    blob = json.dumps(tokens, separators=(",", ":"))
    return sha256_hex(blob)

def sample_from_weights(weights_dict: dict, rng: Sha256CounterRNG) -> str:
    """Sample from {tid_str: int_weight} dict using the SHA256 RNG."""
    sorted_tids = sorted(weights_dict.keys())
    weight_list = [weights_dict[tid] for tid in sorted_tids]
    idx = sample_categorical_weights(weight_list, rng)
    return sorted_tids[idx]


# ---------------------------------------------------------------------------
# Pipeline: Native float64
# ---------------------------------------------------------------------------
def float64_logprobs_to_weights(logprobs, temperature, top_p, top_k, min_p):
    sorted_tids = sorted(logprobs.keys())
    scaled = {tid: logprobs[tid] / temperature for tid in sorted_tids}
    max_val = max(scaled.values())
    exps = {tid: math.exp(scaled[tid] - max_val) for tid in sorted_tids}
    total_exp = sum(exps[tid] for tid in sorted_tids)
    probs = {tid: exps[tid] / total_exp for tid in sorted_tids}

    # top_k filtering
    if top_k is not None and top_k < len(sorted_tids):
        top_k_tids = sorted(
            sorted_tids, key=lambda t: probs[t], reverse=True
        )[:top_k]
        probs = {tid: probs[tid] for tid in top_k_tids}
        sorted_tids = sorted(top_k_tids)

    # top_p filtering
    if top_p is not None:
        sorted_by_prob = sorted(
            sorted_tids, key=lambda t: probs[t], reverse=True
        )
        cumsum = 0.0
        kept = []
        for tid in sorted_by_prob:
            cumsum += probs[tid]
            kept.append(tid)
            if cumsum >= top_p:
                break
        probs = {tid: probs[tid] for tid in kept}
        sorted_tids = sorted(kept)

    # min_p filtering
    if min_p is not None:
        max_prob = max(probs[tid] for tid in sorted_tids)
        threshold = max_prob * min_p
        kept = [tid for tid in sorted_tids if probs[tid] >= threshold]
        if not kept:
            kept = [max(sorted_tids, key=lambda t: probs[t])]
        probs = {tid: probs[tid] for tid in kept}
        sorted_tids = sorted(kept)

    # Re-normalize after filtering
    kept_total = sum(probs[tid] for tid in sorted_tids)
    norm_probs = {tid: probs[tid] / kept_total for tid in sorted_tids}
    weights = {tid: round(norm_probs[tid] * SCALE) for tid in sorted_tids}
    residual = SCALE - sum(weights.values())
    max_tid = max(sorted_tids, key=lambda t: (weights[t], t))
    weights[max_tid] += residual
    return weights


# ---------------------------------------------------------------------------
# Pipeline: NumPy float32 / float16
# ---------------------------------------------------------------------------
def numpy_logprobs_to_weights(logprobs, temperature, top_p, top_k, min_p,
                              dtype):
    sorted_tids = sorted(logprobs.keys())
    lps = np.array([logprobs[tid] for tid in sorted_tids], dtype=dtype)
    temp = dtype(temperature)

    scaled = lps / temp
    max_val = scaled.max()
    exps = np.exp((scaled - max_val).astype(dtype))
    total_exp = exps.sum()
    probs_arr = exps / total_exp

    # Convert to dict for filtering logic (matching the doc's dict-based API)
    probs = {tid: float(probs_arr[i]) for i, tid in enumerate(sorted_tids)}

    # top_k filtering
    if top_k is not None and top_k < len(sorted_tids):
        top_k_tids = sorted(
            sorted_tids, key=lambda t: probs[t], reverse=True
        )[:top_k]
        probs = {tid: probs[tid] for tid in top_k_tids}
        sorted_tids = sorted(top_k_tids)

    # top_p filtering
    if top_p is not None:
        tp = float(dtype(top_p))
        sorted_by_prob = sorted(
            sorted_tids, key=lambda t: probs[t], reverse=True
        )
        cumsum = 0.0
        kept = []
        for tid in sorted_by_prob:
            cumsum += probs[tid]
            kept.append(tid)
            if cumsum >= tp:
                break
        probs = {tid: probs[tid] for tid in kept}
        sorted_tids = sorted(kept)

    # min_p filtering
    if min_p is not None:
        mp = float(dtype(min_p))
        max_prob = max(probs[tid] for tid in sorted_tids)
        threshold = max_prob * mp
        kept = [tid for tid in sorted_tids if probs[tid] >= threshold]
        if not kept:
            kept = [max(sorted_tids, key=lambda t: probs[t])]
        probs = {tid: probs[tid] for tid in kept}
        sorted_tids = sorted(kept)

    # Re-normalize after filtering
    kept_total = sum(probs[tid] for tid in sorted_tids)
    norm_probs = {tid: probs[tid] / kept_total for tid in sorted_tids}
    weights = {tid: round(norm_probs[tid] * SCALE) for tid in sorted_tids}
    residual = SCALE - sum(weights.values())
    max_tid = max(sorted_tids, key=lambda t: (weights[t], t))
    weights[max_tid] += residual
    return weights


# ---------------------------------------------------------------------------
# Pipeline: Decimal (parameterized precision)
# ---------------------------------------------------------------------------
def decimal_logprobs_to_weights(logprob_strings, temperature, top_p, top_k,
                                min_p, prec):
    with localcontext() as ctx:
        ctx.prec = prec
        ctx.rounding = ROUND_HALF_EVEN

        sorted_tids = sorted(logprob_strings.keys())
        T = Decimal(temperature)
        D_SCALE = Decimal(SCALE)

        # Temperature scaling
        scaled = {tid: Decimal(logprob_strings[tid]) / T
                  for tid in sorted_tids}

        # Softmax with log-sum-exp stability shift
        max_val = max(scaled[tid] for tid in sorted_tids)
        exps = {tid: (scaled[tid] - max_val).exp() for tid in sorted_tids}
        total_exp = sum(exps[tid] for tid in sorted_tids)
        probs = {tid: exps[tid] / total_exp for tid in sorted_tids}

        # top_k filtering
        if top_k is not None and top_k < len(sorted_tids):
            top_k_tids = sorted(
                sorted_tids, key=lambda t: probs[t], reverse=True
            )[:top_k]
            probs = {tid: probs[tid] for tid in top_k_tids}
            sorted_tids = sorted(top_k_tids)

        # top_p filtering
        if top_p is not None:
            tp = Decimal(top_p)
            sorted_by_prob = sorted(
                sorted_tids, key=lambda t: probs[t], reverse=True
            )
            cumsum = Decimal(0)
            kept = []
            for tid in sorted_by_prob:
                cumsum += probs[tid]
                kept.append(tid)
                if cumsum >= tp:
                    break
            probs = {tid: probs[tid] for tid in kept}
            sorted_tids = sorted(kept)

        # min_p filtering
        if min_p is not None:
            mp = Decimal(min_p)
            max_prob = max(probs[tid] for tid in sorted_tids)
            threshold = max_prob * mp
            kept = [tid for tid in sorted_tids if probs[tid] >= threshold]
            if not kept:
                kept = [max(sorted_tids, key=lambda t: probs[t])]
            probs = {tid: probs[tid] for tid in kept}
            sorted_tids = sorted(kept)

        # Re-normalize after filtering
        kept_total = sum(probs[tid] for tid in sorted_tids)
        norm_probs = {tid: probs[tid] / kept_total for tid in sorted_tids}

        # Quantize to integer weights
        weights = {tid: int((norm_probs[tid] * D_SCALE).to_integral_value())
                   for tid in sorted_tids}

        # Fix total to exactly SCALE (residual -> max-weight token, ties by tid)
        residual = SCALE - sum(weights.values())
        max_tid = max(sorted_tids, key=lambda t: (weights[t], t))
        weights[max_tid] += residual
        return weights


# ---------------------------------------------------------------------------
# Run a full pipeline
# ---------------------------------------------------------------------------
def run_pipeline(weight_fn, positions, seed):
    rng = Sha256CounterRNG.from_seed_string(seed)
    t0 = time.perf_counter()
    all_weights = []
    tokens = []
    for i, pos in enumerate(positions):
        w = weight_fn(pos)
        all_weights.append(w)
        tokens.append(sample_from_weights(w, rng))
    elapsed = time.perf_counter() - t0
    return all_weights, tokens, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 75)
    print("Cross-Machine Reproducibility Test (100k positions)")
    print("=" * 75)
    print(f"  Python:     {sys.version}")
    print(f"  Platform:   {platform.platform()}")
    print(f"  Machine:    {platform.machine()}")
    print(f"  Processor:  {platform.processor() or 'unknown'}")
    if HAS_NUMPY:
        print(f"  NumPy:      {np.__version__}")
    else:
        print(f"  NumPy:      NOT INSTALLED (float16/float32 pipelines skipped)")
    params = f"T={TEMPERATURE}, top_p={TOP_P}"
    if TOP_K is not None:
        params += f", top_k={TOP_K}"
    if MIN_P is not None:
        params += f", min_p={MIN_P}"
    print(f"  Config:     {N_POSITIONS:,} positions, {NUM_LOGPROBS} logprobs, "
          f"{params}")
    print()

    # --- Generate inputs ---
    positions_float = generate_token_positions(N_POSITIONS, NUM_LOGPROBS, SEED)
    positions_str = [
        {tid: repr(lp) for tid, lp in pos.items()}
        for pos in positions_float
    ]
    input_hash = hash_inputs(positions_str)
    print(f"INPUT hash:              {input_hash}")
    print()

    results = {}  # name -> (w_hash, t_hash, elapsed)

    # --- Float16 ---
    if HAS_NUMPY:
        w, t, e = run_pipeline(
            lambda pos: numpy_logprobs_to_weights(
                pos, TEMPERATURE, TOP_P, TOP_K, MIN_P, np.float16),
            positions_float, RNG_SEED,
        )
        hw, ht = hash_weights(w), hash_tokens(t)
        results["FLOAT16"] = (hw, ht, e)
        print(f"FLOAT16  weights hash:   {hw}")
        print(f"FLOAT16  tokens hash:    {ht}")
        print(f"  ({e:.2f}s, {e/N_POSITIONS*1e6:.1f} \u00b5s/pos)")
        print()

    # --- Float32 ---
    if HAS_NUMPY:
        w, t, e = run_pipeline(
            lambda pos: numpy_logprobs_to_weights(
                pos, TEMPERATURE, TOP_P, TOP_K, MIN_P, np.float32),
            positions_float, RNG_SEED,
        )
        hw, ht = hash_weights(w), hash_tokens(t)
        results["FLOAT32"] = (hw, ht, e)
        print(f"FLOAT32  weights hash:   {hw}")
        print(f"FLOAT32  tokens hash:    {ht}")
        print(f"  ({e:.2f}s, {e/N_POSITIONS*1e6:.1f} \u00b5s/pos)")
        print()

    # --- Float64 ---
    w, t, e = run_pipeline(
        lambda pos: float64_logprobs_to_weights(
            pos, TEMPERATURE, TOP_P, TOP_K, MIN_P),
        positions_float, RNG_SEED,
    )
    hw, ht = hash_weights(w), hash_tokens(t)
    results["FLOAT64"] = (hw, ht, e)
    print(f"FLOAT64  weights hash:   {hw}")
    print(f"FLOAT64  tokens hash:    {ht}")
    print(f"  ({e:.2f}s, {e/N_POSITIONS*1e6:.1f} \u00b5s/pos)")
    print()

    # --- Decimal at various precisions ---
    temp_str = repr(TEMPERATURE)
    top_p_str = repr(TOP_P) if TOP_P is not None else None
    min_p_str = repr(MIN_P) if MIN_P is not None else None
    for prec in DECIMAL_PRECISIONS:
        name = f"DECIMAL{prec}"
        w, t, e = run_pipeline(
            lambda pos, p=prec: decimal_logprobs_to_weights(
                pos, temp_str, top_p_str, TOP_K, min_p_str, p),
            positions_str, RNG_SEED,
        )
        hw, ht = hash_weights(w), hash_tokens(t)
        results[name] = (hw, ht, e)
        label = f"{name:9s}"
        print(f"{label}weights hash:   {hw}")
        print(f"{label}tokens hash:    {ht}")
        print(f"  ({e:.2f}s, {e/N_POSITIONS*1e6:.1f} \u00b5s/pos)")
        print()

    # --- Verify expected hashes ---
    all_names = []
    for n in ["FLOAT16", "FLOAT32", "FLOAT64"]:
        if n in results:
            all_names.append(n)
    for prec in DECIMAL_PRECISIONS:
        all_names.append(f"DECIMAL{prec}")

    print("=" * 75)
    print("Verification against expected hashes")
    print("=" * 75)

    all_pass = True
    any_checked = False

    def check(name, actual, expected):
        nonlocal all_pass, any_checked
        if expected is None:
            print(f"  {name:22s}: SKIP (no expected hash)")
            return
        any_checked = True
        if actual == expected:
            print(f"  {name:22s}: PASS")
        else:
            print(f"  {name:22s}: FAIL")
            print(f"    expected: {expected}")
            print(f"    actual:   {actual}")
            all_pass = False

    check("INPUT", input_hash, EXPECTED.get("INPUT"))
    for name in all_names:
        check(f"{name} weights", results[name][0], EXPECTED.get(f"{name} weights"))
        check(f"{name} tokens", results[name][1], EXPECTED.get(f"{name} tokens"))

    print()
    if not any_checked:
        print("First run -- paste hashes into EXPECTED dict and rerun.")
    elif all_pass:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED -- see details above")

    return all_pass if any_checked else True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
