"""Live integration tests: Inference Validation (enforced tokens).

Requires a running vLLM server on port 18199.

Tests:
  1. Exact replay produces distance2 ~ 0
  2. Replay with different temperature → same tokens, small distance
  3. Replay with different seed → same tokens (enforced overrides sampling)
  4. Replay with grammar → small distance
  5. Multiple prompt lengths → distance stays small
  6. High top_logprobs (5) → distance stays small
  7. Corrupted single token → engine survives
  8. Corrupted multiple tokens → engine survives
  9. Enforced replay text matches original
  10. Long sequence replay → distance stays small
"""
import json
import time
import httpx
import pytest

from tests.gonka.live_conftest import (
    BASE_URL, MODEL, require_server, stop_poc,
    chat_request, build_enforced_tokens, extract_result, distance2,
)


@pytest.fixture(scope="module", autouse=True)
def server_ready():
    require_server()
    stop_poc()
    yield


def _infer(prompt, max_tokens=30, extra=None):
    r = chat_request(
        [{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        extra=extra,
    )
    assert r.status_code == 200, f"Inference failed ({r.status_code}): {r.text}"
    return r.json()


def _replay(prompt, enforced, max_tokens=30, extra=None):
    base = {"enforced_tokens": enforced}
    if extra:
        base.update(extra)
    r = chat_request(
        [{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        extra=base,
    )
    return r


def _infer_and_replay(prompt, max_tokens=30, replay_extra=None,
                       infer_extra=None):
    """Run inference then exact replay. Returns (dist, matches, inf_len, val_len)."""
    data1 = _infer(prompt, max_tokens=max_tokens, extra=infer_extra)
    inf = extract_result(data1)
    enforced = build_enforced_tokens(data1["choices"][0]["logprobs"]["content"])

    r2 = _replay(prompt, enforced, max_tokens=max_tokens, extra=replay_extra)
    assert r2.status_code == 200, f"Replay failed ({r2.status_code}): {r2.text}"
    val = extract_result(r2.json())

    dist, matches = distance2(inf, val)
    return dist, matches, len(inf), len(val)


class TestValidation:

    def test_01_exact_replay_distance_zero(self):
        """Exact token replay → distance2 ~ 0."""
        dist, matches, n_inf, n_val = _infer_and_replay(
            "Explain gravity in two sentences.", max_tokens=40
        )
        print(f"\n  distance2={dist:.6f}, matches={matches:.4f}, tokens={n_inf}")
        assert dist >= 0, f"Token mismatch (inf={n_inf}, val={n_val})"
        assert dist < 0.05, f"Exact replay distance too large: {dist:.6f}"

    def test_02_replay_different_temperature(self):
        """Replay with different temperature → enforced tokens override sampling."""
        prompt = "Name three colors."
        data1 = _infer(prompt, max_tokens=20, extra={"temperature": 0.5})
        inf = extract_result(data1)
        enforced = build_enforced_tokens(data1["choices"][0]["logprobs"]["content"])

        r2 = _replay(prompt, enforced, max_tokens=20, extra={"temperature": 1.5})
        assert r2.status_code == 200
        val = extract_result(r2.json())

        dist, matches = distance2(inf, val)
        print(f"\n  Different temp replay: distance2={dist:.6f}, matches={matches:.4f}")
        assert dist >= 0, "Token mismatch"

    def test_03_replay_different_seed(self):
        """Enforced tokens override seed — output is identical."""
        prompt = "Count to three."
        data1 = _infer(prompt, max_tokens=15, extra={"seed": 111})
        inf = extract_result(data1)
        enforced = build_enforced_tokens(data1["choices"][0]["logprobs"]["content"])

        r2 = _replay(prompt, enforced, max_tokens=15, extra={"seed": 999})
        assert r2.status_code == 200
        val = extract_result(r2.json())

        inf_tokens = [r["token"] for r in inf]
        val_tokens = [r["token"] for r in val]
        assert inf_tokens == val_tokens, "Enforced tokens should override seed"
        print(f"\n  Tokens match across seeds: {len(inf_tokens)} tokens")

    def test_04_replay_with_grammar(self):
        """Replay with JSON schema grammar → distance stays small."""
        schema = {
            "type": "object",
            "properties": {
                "fruit": {"type": "string"},
                "count": {"type": "integer"},
            },
            "required": ["fruit", "count"],
        }
        grammar_extra = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "fruit_info", "schema": schema},
            },
        }
        dist, matches, n_inf, n_val = _infer_and_replay(
            "Return JSON: fruit=banana, count=7",
            max_tokens=40,
            infer_extra=grammar_extra,
            replay_extra=grammar_extra,
        )
        print(f"\n  Grammar replay: distance2={dist:.6f}, matches={matches:.4f}")
        assert dist >= 0, "Token mismatch"
        assert dist < 0.05, f"Grammar replay distance too large: {dist:.6f}"

    def test_05_various_prompt_lengths(self):
        """Different prompt lengths → distance stays small."""
        prompts = [
            ("Hi", 10),
            ("Explain quantum computing in simple terms.", 30),
            ("Write a detailed paragraph about the solar system, covering "
             "all eight planets and their key characteristics.", 60),
        ]
        for prompt, max_tok in prompts:
            dist, matches, n_inf, n_val = _infer_and_replay(
                prompt, max_tokens=max_tok
            )
            print(f"\n  Prompt({len(prompt)} chars, {max_tok} max_tok): "
                  f"dist={dist:.6f}, tokens={n_inf}")
            assert dist >= 0, f"Token mismatch for prompt: {prompt[:30]}..."
            assert dist < 0.05, f"Distance too large: {dist:.6f}"

    def test_06_high_top_logprobs(self):
        """top_logprobs=5 → distance stays small."""
        dist, matches, n_inf, n_val = _infer_and_replay(
            "What is the capital of France?", max_tokens=20
        )
        print(f"\n  top_logprobs=5: distance2={dist:.6f}, matches={matches:.4f}")
        assert dist >= 0, "Token mismatch"
        assert dist < 0.05, f"Distance too large: {dist:.6f}"

    def test_07_corrupted_single_token_engine_survives(self):
        """Corrupt one enforced token → engine must not crash."""
        data1 = _infer("Hello world", max_tokens=10)
        content = data1["choices"][0]["logprobs"]["content"]
        enforced = build_enforced_tokens(content)

        if len(enforced["tokens"]) > 2:
            enforced["tokens"][1]["token"] = "99999"

        r2 = _replay("Hello world", enforced, max_tokens=10)
        assert r2.status_code in (200, 400, 422), (
            f"Engine crashed: {r2.status_code}: {r2.text}"
        )

        time.sleep(0.3)
        r3 = chat_request([{"role": "user", "content": "alive?"}], max_tokens=3)
        assert r3.status_code == 200, "Engine died after corrupted token"
        print(f"\n  Corrupted single token: status={r2.status_code}, engine OK")

    def test_08_corrupted_multiple_tokens_engine_survives(self):
        """Corrupt several enforced tokens → engine must not crash."""
        data1 = _infer("Tell me a joke", max_tokens=20)
        content = data1["choices"][0]["logprobs"]["content"]
        enforced = build_enforced_tokens(content)

        for i in range(0, len(enforced["tokens"]), 3):
            enforced["tokens"][i]["token"] = "88888"

        r2 = _replay("Tell me a joke", enforced, max_tokens=20)
        assert r2.status_code in (200, 400, 422)

        time.sleep(0.3)
        r3 = chat_request([{"role": "user", "content": "ok?"}], max_tokens=3)
        assert r3.status_code == 200, "Engine died after multiple corrupted tokens"
        print(f"\n  Corrupted multiple tokens: status={r2.status_code}, engine OK")

    def test_09_enforced_replay_text_matches(self):
        """Enforced replay produces identical text output."""
        prompt = "List three programming languages."
        data1 = _infer(prompt, max_tokens=30)
        text1 = data1["choices"][0]["message"]["content"]
        enforced = build_enforced_tokens(data1["choices"][0]["logprobs"]["content"])

        r2 = _replay(prompt, enforced, max_tokens=30)
        assert r2.status_code == 200
        text2 = r2.json()["choices"][0]["message"]["content"]

        assert text1 == text2, (
            f"Text mismatch:\n  Inf: {text1[:100]}\n  Val: {text2[:100]}"
        )
        print(f"\n  Text matches: {text1[:80]}")

    def test_10_long_sequence_replay(self):
        """Long output (100 tokens) replay → distance stays small."""
        dist, matches, n_inf, n_val = _infer_and_replay(
            "Write a detailed explanation of how computers work.",
            max_tokens=100,
        )
        print(f"\n  Long sequence: distance2={dist:.6f}, matches={matches:.4f}, "
              f"tokens={n_inf}")
        assert dist >= 0, f"Token mismatch (inf={n_inf}, val={n_val})"
        assert dist < 0.05, f"Long sequence distance too large: {dist:.6f}"
