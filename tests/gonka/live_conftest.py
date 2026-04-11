"""Shared helpers for live integration tests against a running vLLM server.

Uses the same EnforcedTokens format and distance2 metric as the production
validation pipeline (see benchmarks/src/validation/utils.py).
"""
import math
import os
import time
import httpx
import pytest

MODEL = os.environ.get("VLLM_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
PORT = int(os.environ.get("VLLM_TEST_PORT", "18199"))
BASE_URL = f"http://127.0.0.1:{PORT}"


def require_server():
    """Skip if the vLLM server isn't running."""
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=5)
        if r.status_code != 200:
            pytest.skip("vLLM server not healthy on port 18199")
    except Exception:
        pytest.skip("vLLM server not running on port 18199")


def stop_poc():
    try:
        httpx.post(f"{BASE_URL}/api/v1/pow/stop", timeout=5)
    except Exception:
        pass
    time.sleep(0.5)


def chat_request(messages, max_tokens=20, extra=None, timeout=60):
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.99,
        "seed": 42,
        "stream": False,
        "logprobs": True,
        "top_logprobs": 5,
        "n": 1,
        "skip_special_tokens": False,
    }
    if extra:
        body.update(extra)
    return httpx.post(
        f"{BASE_URL}/v1/chat/completions", json=body, timeout=timeout
    )


def build_enforced_tokens(content):
    """Build EnforcedTokens payload from logprobs content
    (matches EnforcedTokens.from_content in the validation library)."""
    tokens = []
    for position in content:
        token = str(position["token"])
        top_tokens = [str(x["token"]) for x in position["top_logprobs"]]
        tokens.append({"token": token, "top_tokens": top_tokens})
    return {"tokens": tokens}


def extract_result(response_json):
    """Extract token/logprobs in PositionResult-compatible format."""
    content = response_json["choices"][0]["logprobs"]["content"]
    results = []
    for position in content:
        logprobs = {
            str(lp["token"]): lp["logprob"]
            for lp in position["top_logprobs"]
        }
        results.append({
            "token": str(position["token"]),
            "logprobs": logprobs,
        })
    return results


def token_distance2(inf_pos, val_pos):
    """Per-position distance matching Go customDistance/positionDistance.

    Iterates validation tokens, builds fallback from inference side.
    """
    inf_lp = inf_pos["logprobs"]
    val_lp = val_pos["logprobs"]

    if not inf_lp or not val_lp:
        return 100.0, 0

    sorted_inf = sorted(inf_lp.values())
    if len(sorted_inf) >= 2:
        min1, min2 = sorted_inf[0], sorted_inf[1]
    else:
        min1 = sorted_inf[0]
        min2 = min1 - 100.0
    next_inf_logprob = min1 - (min2 - min1)

    dist = 0.0
    n_matches = 0
    for token, val_logprob in val_lp.items():
        if token in inf_lp:
            inf_logprob = inf_lp[token]
            n_matches += 1
        else:
            inf_logprob = next_inf_logprob

        denom = 1e-6 + abs(val_logprob) + abs(inf_logprob)
        if math.isnan(denom) or denom == 0:
            continue
        term = abs(val_logprob - inf_logprob) / denom / 2.0
        if not math.isnan(term):
            dist += term

    return dist, n_matches


def distance2(inf_results, val_results):
    """Sequence-level distance2 matching the production metric."""
    if len(inf_results) != len(val_results):
        return -1, -1
    if [r["token"] for r in inf_results] != [r["token"] for r in val_results]:
        return -1, -1

    total_dist = 0.0
    total_matches = 0
    for inf_pos, val_pos in zip(inf_results, val_results):
        d, m = token_distance2(inf_pos, val_pos)
        total_dist += d
        total_matches += m

    n_logprobs = len(inf_results[0]["logprobs"]) if inf_results[0]["logprobs"] else 1
    matches_ratio = total_matches / (len(inf_results) * n_logprobs)
    total_dist = total_dist / (max(100, len(inf_results)) * n_logprobs)
    return total_dist, matches_ratio
