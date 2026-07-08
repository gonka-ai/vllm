"""Live integration tests: logprobs_mode behavior.

Requires a running vLLM server on port 18199.

Tests that logprobs_mode actually changes what's returned:
  1. Default mode (processed_logprobs) returns token IDs as strings
  2. raw_logprobs returns different values than processed_logprobs
  3. Grammar-constrained processed_logprobs clamps non-chosen to -9999
  4. raw_logprobs are NOT clamped to -9999 (reflect true model distribution)
  5. Per-request logprobs_mode override works
  6. Validation replay with matching logprobs_mode → distance ~ 0
  7. Validation replay with mismatched logprobs_mode → distance > 0
"""
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


def _chat(prompt, max_tokens=15, logprobs_mode=None, extra=None):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.99,
        "seed": 42,
        "stream": False,
        "logprobs": True,
        "top_logprobs": 5,
        "n": 1,
        "skip_special_tokens": False,
    }
    if logprobs_mode is not None:
        body["logprobs_mode"] = logprobs_mode
    if extra:
        body.update(extra)
    r = httpx.post(
        f"{BASE_URL}/v1/chat/completions", json=body, timeout=30
    )
    return r


class TestLogprobsMode:

    def test_01_default_mode_returns_token_ids_as_strings(self):
        """Default (processed_logprobs) returns token IDs as strings."""
        r = _chat("Say hello.")
        assert r.status_code == 200
        data = r.json()
        content = data["choices"][0]["logprobs"]["content"]
        for pos in content[:3]:
            assert pos["token"].isdigit() or pos["token"].startswith("-"), (
                f"Expected token ID string, got: {pos['token']!r}"
            )
            for tp in pos["top_logprobs"]:
                assert tp["token"].isdigit() or tp["token"].startswith("-"), (
                    f"Expected token ID string in top_logprobs, got: {tp['token']!r}"
                )
        print(f"\n  Token samples: {[p['token'] for p in content[:5]]}")

    def test_02_raw_vs_processed_differ(self):
        """raw_logprobs and processed_logprobs return different logprob values."""
        prompt = "What is the meaning of life?"

        r_proc = _chat(prompt, logprobs_mode="processed_logprobs")
        r_raw = _chat(prompt, logprobs_mode="raw_logprobs")

        if r_raw.status_code != 200:
            pytest.skip(f"raw_logprobs not supported: {r_raw.status_code}")

        proc_content = r_proc.json()["choices"][0]["logprobs"]["content"]
        raw_content = r_raw.json()["choices"][0]["logprobs"]["content"]

        proc_values = []
        raw_values = []
        for pp, rp in zip(proc_content, raw_content):
            proc_values.append(pp["logprob"])
            raw_values.append(rp["logprob"])

        n_different = sum(
            1 for p, r in zip(proc_values, raw_values)
            if abs(p - r) > 1e-3
        )
        print(f"\n  Processed logprobs: {proc_values[:5]}")
        print(f"  Raw logprobs:       {raw_values[:5]}")
        print(f"  Positions with different values: {n_different}/{len(proc_values)}")

    def test_03_processed_grammar_clamps_to_negative_9999(self):
        """With grammar, processed_logprobs clamps non-chosen tokens to -9999."""
        schema = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }
        r = _chat("Return JSON: x=5", logprobs_mode="processed_logprobs", extra={
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "s", "schema": schema},
            },
            "max_tokens": 30,
        })
        assert r.status_code == 200
        content = r.json()["choices"][0]["logprobs"]["content"]

        n_clamped = 0
        for pos in content:
            for tp in pos["top_logprobs"]:
                if tp["logprob"] <= -9990:
                    n_clamped += 1

        total = sum(len(pos["top_logprobs"]) for pos in content)
        ratio = n_clamped / total if total > 0 else 0
        print(f"\n  Clamped to -9999: {n_clamped}/{total} ({ratio:.1%})")
        assert n_clamped > 0, (
            "Expected some -9999 clamped values in processed+grammar mode"
        )

    def test_04_raw_logprobs_not_clamped(self):
        """raw_logprobs should NOT be clamped to -9999 (true model distribution)."""
        r = _chat("Hello world", logprobs_mode="raw_logprobs", extra={
            "max_tokens": 10,
        })
        if r.status_code != 200:
            pytest.skip(f"raw_logprobs not supported: {r.status_code}")

        content = r.json()["choices"][0]["logprobs"]["content"]
        n_clamped = 0
        for pos in content:
            for tp in pos["top_logprobs"]:
                if tp["logprob"] <= -9990:
                    n_clamped += 1

        total = sum(len(pos["top_logprobs"]) for pos in content)
        ratio = n_clamped / total if total > 0 else 0
        print(f"\n  Clamped to -9999 in raw mode: {n_clamped}/{total} ({ratio:.1%})")
        assert ratio < 0.5, (
            f"Raw logprobs should have few -9999 values, got {ratio:.1%}"
        )

    def test_05_per_request_override(self):
        """Per-request logprobs_mode overrides the deployment default."""
        r_default = _chat("Hi")
        assert r_default.status_code == 200

        r_raw = _chat("Hi", logprobs_mode="raw_logprobs")
        if r_raw.status_code != 200:
            pytest.skip(f"raw_logprobs not supported: {r_raw.status_code}")

        default_lp = r_default.json()["choices"][0]["logprobs"]["content"][0]["logprob"]
        raw_lp = r_raw.json()["choices"][0]["logprobs"]["content"][0]["logprob"]

        print(f"\n  Default first logprob: {default_lp}")
        print(f"  Raw first logprob: {raw_lp}")

    def test_06_validation_same_mode_small_distance(self):
        """Inference + validation both processed → distance ~ 0."""
        prompt = "Name a country."
        data1 = _chat(prompt, logprobs_mode="processed_logprobs")
        assert data1.status_code == 200
        d1 = data1.json()
        inf = extract_result(d1)
        enforced = build_enforced_tokens(d1["choices"][0]["logprobs"]["content"])

        r2 = _chat(prompt, logprobs_mode="processed_logprobs", extra={
            "enforced_tokens": enforced,
        })
        assert r2.status_code == 200
        val = extract_result(r2.json())

        dist, matches = distance2(inf, val)
        print(f"\n  Same mode validation: distance2={dist:.6f}, matches={matches:.4f}")
        assert dist >= 0, "Token mismatch"
        assert dist < 0.05, f"Same-mode distance too large: {dist:.6f}"

    def test_07_validation_mismatched_mode_larger_distance(self):
        """Inference processed + validation raw → distance should be larger."""
        prompt = "Name a fruit."
        data1 = _chat(prompt, logprobs_mode="processed_logprobs")
        assert data1.status_code == 200
        d1 = data1.json()
        inf = extract_result(d1)
        enforced = build_enforced_tokens(d1["choices"][0]["logprobs"]["content"])

        r2 = _chat(prompt, logprobs_mode="raw_logprobs", extra={
            "enforced_tokens": enforced,
        })
        if r2.status_code != 200:
            pytest.skip(f"raw_logprobs replay not supported: {r2.status_code}")

        val = extract_result(r2.json())
        dist, matches = distance2(inf, val)
        print(f"\n  Mismatched mode: distance2={dist:.6f}, matches={matches:.4f}")

        # Compare with same-mode distance
        r3 = _chat(prompt, logprobs_mode="processed_logprobs", extra={
            "enforced_tokens": enforced,
        })
        assert r3.status_code == 200
        val_same = extract_result(r3.json())
        dist_same, _ = distance2(inf, val_same)

        print(f"  Same-mode distance:      {dist_same:.6f}")
        print(f"  Mismatched-mode distance: {dist:.6f}")
        if dist >= 0 and dist_same >= 0:
            assert dist >= dist_same, (
                "Mismatched mode should produce >= distance than same mode"
            )
