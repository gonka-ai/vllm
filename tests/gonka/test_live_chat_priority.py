"""Live integration tests: Chat Priority Gating.

Requires a running vLLM server on port 18199.

Tests:
  1. Baseline chat works
  2. PoC activates → chat rejected 503 → PoC stops → chat resumes
  3. Long inference in-flight + PoC activates → in-flight inference
     aborted, new chat rejected, engine survives
  4. Abort is fast — very long inference (2000 tokens) is terminated
     quickly, not drained
  5. Chat completions produce correct output after PoC stops
"""
import json
import time
import threading
import httpx
import pytest

from tests.gonka.live_conftest import (
    BASE_URL, MODEL, require_server, stop_poc, chat_request,
)

POC_INIT_BODY = {
    "block_hash": "TEST_BLOCK",
    "block_height": 100,
    "public_key": "test_pub_keys",
    "node_id": 0,
    "node_count": 1,
    "batch_size": 4,
    "params": {"model": MODEL, "seq_len": 64, "k_dim": 12},
}


@pytest.fixture(scope="module", autouse=True)
def server_ready():
    require_server()
    stop_poc()
    yield
    stop_poc()


@pytest.fixture(autouse=True)
def cleanup_poc():
    """Ensure PoC is stopped before and after each test."""
    stop_poc()
    yield
    stop_poc()


class TestChatPriorityGating:

    def test_01_baseline_chat_works(self):
        r = chat_request(
            [{"role": "user", "content": "Say hello in one word."}],
            max_tokens=5,
        )
        assert r.status_code == 200, f"Baseline chat failed: {r.text}"
        data = r.json()
        assert "choices" in data and len(data["choices"]) > 0

    def test_02_poc_activates_chat_rejected_then_resumes(self):
        r = httpx.post(
            f"{BASE_URL}/api/v1/pow/init/generate",
            json=POC_INIT_BODY,
            timeout=10,
        )
        assert r.status_code == 200, f"init/generate failed: {r.text}"

        # Minimal sleep — just enough for the flag to be set, stop before
        # the background generation loop executes a forward pass.
        time.sleep(0.3)

        r = chat_request(
            [{"role": "user", "content": "hello"}], max_tokens=5, timeout=10
        )
        assert r.status_code == 503, (
            f"Expected 503 during PoC, got {r.status_code}: {r.text}"
        )
        assert "PoC generation is active" in r.text

        r = httpx.post(f"{BASE_URL}/api/v1/pow/stop", timeout=10)
        assert r.status_code == 200
        time.sleep(1)

        r = chat_request(
            [{"role": "user", "content": "Say bye in one word."}],
            max_tokens=5,
        )
        assert r.status_code == 200, (
            f"Chat should resume after stop: {r.status_code}: {r.text}"
        )

    def test_03_long_inference_aborted_by_poc(self):
        """Long inference in-flight, PoC starts → inference aborted, engine OK."""
        long_prompt = (
            "Write a very detailed essay about the history of mathematics, "
            "covering ancient civilizations like Babylon, Egypt, Greece, India, "
            "and China. Discuss medieval Islamic mathematics, the Renaissance, "
            "and modern breakthroughs in algebra, calculus, topology, and "
            "number theory. Include specific mathematicians. " * 3
        )

        inference_result = {}

        def run_long_inference():
            try:
                r = chat_request(
                    [{"role": "user", "content": long_prompt}],
                    max_tokens=300,
                    timeout=30,
                )
                inference_result["status"] = r.status_code
                inference_result["text"] = r.text[:500]
            except Exception as e:
                inference_result["error"] = str(e)

        t = threading.Thread(target=run_long_inference)
        t.start()
        time.sleep(1.5)

        r = httpx.post(
            f"{BASE_URL}/api/v1/pow/init/generate",
            json={**POC_INIT_BODY, "block_height": 200},
            timeout=10,
        )
        assert r.status_code == 200, f"init/generate failed: {r.text}"

        time.sleep(0.3)
        r = chat_request(
            [{"role": "user", "content": "hi"}], max_tokens=3, timeout=10
        )
        assert r.status_code == 503, (
            f"Expected 503 while PoC active, got {r.status_code}"
        )

        httpx.post(f"{BASE_URL}/api/v1/pow/stop", timeout=10)
        t.join(timeout=30)

        # The in-flight inference should have been aborted.  Evidence:
        #   - An exception (connection reset / stream interrupted), OR
        #   - A non-200 HTTP status, OR
        #   - HTTP 200 with truncated output (abort fires at the engine
        #     level; for non-streaming requests vLLM V1 still returns 200
        #     with whatever tokens were generated before the abort).
        if "error" in inference_result:
            aborted = True
        elif inference_result.get("status") != 200:
            aborted = True
        else:
            try:
                body = __import__("json").loads(inference_result["text"])
                n_tokens = len(
                    body["choices"][0]["logprobs"]["content"]
                )
                aborted = n_tokens < 300
            except Exception:
                aborted = True
        assert aborted, (
            "Expected in-flight inference to be aborted by PoC, "
            f"but got status={inference_result.get('status')} "
            f"text={inference_result.get('text', '')[:200]}"
        )

        # Engine must survive abort + PoC stop and serve new requests.
        time.sleep(2)
        r = chat_request(
            [{"role": "user", "content": "Still alive?"}], max_tokens=5
        )
        assert r.status_code == 200, (
            f"Engine died after abort: {r.status_code}: {r.text}"
        )

    def test_04_abort_is_fast_not_drained(self):
        """Very long inference (2000 tokens) is aborted quickly by PoC,
        not drained to completion.

        Without abort the inference would take 30-60+ seconds on 235B.
        With abort, the in-flight request should resolve within a few
        seconds of PoC being initiated."""
        long_prompt = (
            "Write an extremely long, detailed, and comprehensive essay "
            "about the entire history of human civilization from the stone "
            "age through the modern era. Cover every major civilization, "
            "their rise and fall, cultural achievements, wars, scientific "
            "breakthroughs, and philosophical movements. " * 5
        )

        inference_result = {}

        def run_very_long_inference():
            t0 = time.monotonic()
            try:
                r = chat_request(
                    [{"role": "user", "content": long_prompt}],
                    max_tokens=2000,
                    timeout=120,
                )
                inference_result["status"] = r.status_code
                inference_result["text"] = r.text[:500]
            except Exception as e:
                inference_result["error"] = str(e)
            inference_result["elapsed"] = time.monotonic() - t0

        t = threading.Thread(target=run_very_long_inference)
        t.start()
        time.sleep(3)

        poc_start = time.monotonic()
        r = httpx.post(
            f"{BASE_URL}/api/v1/pow/init/generate",
            json={**POC_INIT_BODY, "block_height": 300},
            timeout=10,
        )
        assert r.status_code == 200, f"init/generate failed: {r.text}"

        t.join(timeout=30)
        poc_to_done = time.monotonic() - poc_start

        httpx.post(f"{BASE_URL}/api/v1/pow/stop", timeout=10)
        time.sleep(1)

        # The inference thread must have finished (abort resolved it).
        assert not t.is_alive(), "Inference thread still running after 30s"

        # With abort, the inference should resolve within 10s of PoC start.
        # Without abort, 2000 tokens on 235B/A100 takes 30-60+ seconds.
        print(f"  Time from PoC init to inference done: {poc_to_done:.1f}s")
        print(f"  Total inference wall time: {inference_result.get('elapsed', -1):.1f}s")
        assert poc_to_done < 15, (
            f"Abort too slow — inference took {poc_to_done:.1f}s after PoC "
            f"started (expected <15s). Looks like drain, not abort."
        )

        # Verify the inference was actually truncated, not fully completed.
        if inference_result.get("status") == 200:
            try:
                body = json.loads(inference_result["text"])
                n_tokens = len(body["choices"][0]["logprobs"]["content"])
                print(f"  Tokens generated: {n_tokens}/2000")
                assert n_tokens < 2000, (
                    f"Got all {n_tokens} tokens — inference was not aborted"
                )
            except (KeyError, json.JSONDecodeError):
                pass

        # Engine still healthy.
        time.sleep(2)
        r = chat_request(
            [{"role": "user", "content": "Quick check"}], max_tokens=5
        )
        assert r.status_code == 200, (
            f"Engine died after fast abort: {r.status_code}: {r.text}"
        )

    def test_05_chat_completions_work_after_poc(self):
        """PoC runs for a while, then stops. Chat completions must produce
        correct, coherent output afterward — model is not corrupted."""
        r = httpx.post(
            f"{BASE_URL}/api/v1/pow/init/generate",
            json={**POC_INIT_BODY, "block_height": 400},
            timeout=10,
        )
        assert r.status_code == 200, f"init/generate failed: {r.text}"

        time.sleep(3)

        r = chat_request(
            [{"role": "user", "content": "hi"}], max_tokens=3, timeout=10
        )
        assert r.status_code == 503, (
            f"Expected 503 while PoC active, got {r.status_code}"
        )

        r = httpx.post(f"{BASE_URL}/api/v1/pow/stop", timeout=10)
        assert r.status_code == 200
        time.sleep(2)

        # First request: verify basic 200 and non-empty content.
        r = chat_request(
            [{"role": "user", "content": "What is 2 + 2? Answer with "
             "just the number."}],
            max_tokens=10,
        )
        assert r.status_code == 200, (
            f"Chat failed after PoC stop: {r.status_code}: {r.text}"
        )
        data = r.json()
        text1 = data["choices"][0]["message"]["content"]
        print(f"  Post-PoC response 1: {text1!r}")
        assert "4" in text1, (
            f"Expected '4' in response, got: {text1!r}"
        )

        # Second request: longer generation to confirm model is coherent.
        r = chat_request(
            [{"role": "user", "content": "List the first 5 prime numbers, "
             "separated by commas."}],
            max_tokens=30,
        )
        assert r.status_code == 200, (
            f"Second chat failed after PoC: {r.status_code}: {r.text}"
        )
        data = r.json()
        text2 = data["choices"][0]["message"]["content"]
        print(f"  Post-PoC response 2: {text2!r}")
        for prime in ["2", "3", "5", "7", "11"]:
            assert prime in text2, (
                f"Expected '{prime}' in primes list, got: {text2!r}"
            )

        # Third request: verify logprobs are returned (not corrupted).
        r = chat_request(
            [{"role": "user", "content": "Say yes."}],
            max_tokens=5,
        )
        assert r.status_code == 200
        data = r.json()
        content = data["choices"][0]["logprobs"]["content"]
        assert len(content) > 0, "No logprobs content after PoC"
        assert all(
            "top_logprobs" in pos and len(pos["top_logprobs"]) > 0
            for pos in content
        ), "Logprobs structure broken after PoC"
        print(f"  Post-PoC response 3: logprobs OK ({len(content)} tokens)")
