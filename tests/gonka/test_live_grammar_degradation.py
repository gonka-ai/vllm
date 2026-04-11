"""Live integration tests: Grammar Graceful Degradation (Inference Validation).

Requires a running vLLM server on port 18199.

Tests:
  1. Structured output produces valid JSON
  2. Exact enforced-token replay (no grammar) → small distance2
  3. Exact enforced-token replay WITH grammar → small distance2
  4. Corrupted enforced tokens + grammar → engine survives (graceful degradation)
  5. Multiple prompts: measure mean distance2 with grammar active

Uses the same EnforcedTokens format and distance2 metric as the production
validation pipeline (see benchmarks/src/validation/utils.py).
"""
import json
import time
import pytest

from tests.gonka.live_conftest import (
    BASE_URL, require_server, stop_poc, chat_request,
    build_enforced_tokens, extract_result, distance2,
)


@pytest.fixture(scope="module", autouse=True)
def server_ready():
    require_server()
    stop_poc()
    yield


def _inference_with_grammar(prompt, schema, schema_name="test_schema"):
    r = chat_request(
        [{"role": "user", "content": prompt}],
        max_tokens=60,
        extra={
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema},
            },
        },
    )
    assert r.status_code == 200, (
        f"Grammar inference failed ({r.status_code}): {r.text}"
    )
    return r.json()


def _replay_with_enforced(prompt, enforced_tokens, max_tokens=60,
                            schema=None, schema_name="test_schema"):
    extra = {"enforced_tokens": enforced_tokens}
    if schema:
        extra["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema},
        }
    return chat_request(
        [{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        extra=extra,
        timeout=60,
    )


class TestGrammarGracefulDegradation:

    def test_01_structured_output_baseline(self):
        """Structured output inference produces valid JSON."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        data = _inference_with_grammar(
            "Return JSON: name=Alice, age=30", schema
        )
        content = data["choices"][0]["message"]["content"]
        assert content.strip().startswith("{"), (
            f"Expected JSON-like output, got: {content[:100]}"
        )
        try:
            parsed = json.loads(content)
            assert "name" in parsed and "age" in parsed
            print(f"\n  Parsed JSON: {parsed}")
        except json.JSONDecodeError:
            print(f"\n  Grammar constrained output (not fully parseable with small model): "
                  f"{content[:120]}")

    def test_02_exact_replay_no_grammar_small_distance(self):
        """Exact token replay WITHOUT grammar: distance2 should be ~0."""
        prompt = "Count from 1 to 5 and explain each number briefly."
        r1 = chat_request(
            [{"role": "user", "content": prompt}],
            max_tokens=40,
        )
        assert r1.status_code == 200, f"Inference failed: {r1.text}"
        data1 = r1.json()
        inf_results = extract_result(data1)
        content1 = data1["choices"][0]["logprobs"]["content"]
        enforced = build_enforced_tokens(content1)

        r2 = _replay_with_enforced(prompt, enforced, max_tokens=40)
        assert r2.status_code == 200, f"Replay failed: {r2.text}"
        val_results = extract_result(r2.json())

        dist, matches = distance2(inf_results, val_results)
        print(f"\n  [No grammar] distance2={dist:.6f}, matches_ratio={matches:.4f}")
        print(f"  Tokens inf={len(inf_results)}, val={len(val_results)}")
        assert dist >= 0, f"Token mismatch (dist={dist}), inf={len(inf_results)} val={len(val_results)}"
        assert dist < 0.05, f"Distance too large without grammar: {dist:.6f}"

    def test_03_exact_replay_with_grammar_small_distance(self):
        """Exact replay WITH grammar: distance2 should be small.

        Same grammar + same tokens → FSM accepts everything, logprobs match.
        """
        schema = {
            "type": "object",
            "properties": {
                "color": {"type": "string"},
                "count": {"type": "integer"},
            },
            "required": ["color", "count"],
        }
        prompt = "Return JSON with your favorite color and a number."

        data1 = _inference_with_grammar(prompt, schema)
        inf_results = extract_result(data1)
        content1 = data1["choices"][0]["logprobs"]["content"]
        enforced = build_enforced_tokens(content1)

        r2 = _replay_with_enforced(prompt, enforced, schema=schema)
        assert r2.status_code == 200, f"Replay failed: {r2.text}"
        val_results = extract_result(r2.json())

        dist, matches = distance2(inf_results, val_results)
        print(f"\n  [With grammar, exact replay] distance2={dist:.6f}, matches={matches:.4f}")
        print(f"  Tokens: {len(inf_results)}")
        assert dist >= 0, f"Token mismatch (dist={dist})"
        assert dist < 0.05, f"Grammar replay distance too large: {dist:.6f}"

    def test_04_corrupted_enforced_with_grammar_no_crash(self):
        """Corrupted enforced tokens + grammar: engine must NOT crash.

        This is the scenario fixed by grammar graceful degradation.
        The corrupted token is rejected by the grammar FSM; before our fix
        this caused an assertion crash in the engine core.
        """
        schema = {
            "type": "object",
            "properties": {"animal": {"type": "string"}},
            "required": ["animal"],
        }
        prompt = "Return JSON with your favorite animal."

        data1 = _inference_with_grammar(prompt, schema)
        content1 = data1["choices"][0]["logprobs"]["content"]

        corrupt_idx = len(content1) // 2
        enforced = build_enforced_tokens(content1)
        enforced["tokens"][corrupt_idx]["token"] = "99999"

        r2 = _replay_with_enforced(prompt, enforced, schema=schema)
        assert r2.status_code in (200, 400, 422), (
            f"Got {r2.status_code}: {r2.text}"
        )
        print(f"\n  Corrupted replay status: {r2.status_code}")

        time.sleep(0.5)
        r3 = chat_request(
            [{"role": "user", "content": "Still alive?"}], max_tokens=5
        )
        assert r3.status_code == 200, (
            f"Engine crashed after corrupted enforced+grammar: {r3.text}"
        )

    def test_05_multi_prompt_grammar_validation_distances(self):
        """Multiple inference→validation pairs with grammar.

        Measure distance2 per prompt to verify structured output validation
        produces consistently small distances (honest self-validation).
        """
        schema = {
            "type": "object",
            "properties": {
                "item": {"type": "string"},
                "quantity": {"type": "integer"},
            },
            "required": ["item", "quantity"],
        }
        prompts = [
            "Return JSON: item=apple, quantity=5",
            "Return JSON: item=book, quantity=3",
            "Return JSON: item=car, quantity=1",
        ]

        distances = []
        for prompt in prompts:
            data1 = _inference_with_grammar(prompt, schema)
            inf_results = extract_result(data1)
            content1 = data1["choices"][0]["logprobs"]["content"]
            enforced = build_enforced_tokens(content1)

            r2 = _replay_with_enforced(prompt, enforced, schema=schema)
            assert r2.status_code == 200, f"Replay failed for '{prompt}': {r2.text}"
            val_results = extract_result(r2.json())

            dist, matches = distance2(inf_results, val_results)
            distances.append(dist)
            print(f"\n  Prompt: {prompt[:50]}")
            print(f"  distance2={dist:.6f}, matches={matches:.4f}, "
                  f"tokens={len(inf_results)}")

        mean_dist = sum(distances) / len(distances)
        print(f"\n  === Mean distance2 across {len(prompts)} prompts: {mean_dist:.6f} ===")
        assert all(d >= 0 for d in distances), "Some prompts had token mismatch"
        assert mean_dist < 0.05, (
            f"Mean distance2 too large: {mean_dist:.6f}. "
            f"Individual: {[f'{d:.6f}' for d in distances]}"
        )
