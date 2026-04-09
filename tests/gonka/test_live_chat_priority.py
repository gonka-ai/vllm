"""Live integration tests: Chat Priority Gating.

Requires a running vLLM server on port 18199.

Tests:
  1. Baseline chat works
  2. PoC activates → chat rejected 503 → PoC stops → chat resumes
  3. Long inference in-flight + PoC activates → new chat rejected,
     in-flight drains, engine survives
"""
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

    def test_03_long_inference_then_poc_then_chat(self):
        """Long inference in-flight, PoC starts, new chat rejected, engine OK."""
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
                    timeout=120,
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

        # Verify gating is active
        time.sleep(0.3)
        r = chat_request(
            [{"role": "user", "content": "hi"}], max_tokens=3, timeout=10
        )
        assert r.status_code == 503, (
            f"Expected 503 while PoC active, got {r.status_code}"
        )

        # Stop immediately to minimize engine disruption
        httpx.post(f"{BASE_URL}/api/v1/pow/stop", timeout=10)
        t.join(timeout=120)

        assert "error" not in inference_result, (
            f"In-flight inference error: {inference_result.get('error')}"
        )

        time.sleep(2)
        r = chat_request(
            [{"role": "user", "content": "Still alive?"}], max_tokens=5
        )
        assert r.status_code == 200, (
            f"Engine died after overlap: {r.status_code}: {r.text}"
        )
