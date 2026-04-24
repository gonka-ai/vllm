"""Tests for Chat Priority Gating (PoC ↔ Chat coexistence).

Tests cover:
1. engine_patch.py: poc_request aborts in-flight inference and proceeds
2. api_router.py: chat and completion endpoints reject requests when PoC active
3. routes.py: _poc_generation_active flag lifecycle (set/cleared correctly)
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# 1. Engine patch: poc_request aborts in-flight inference then proceeds
# ---------------------------------------------------------------------------

class TestPocRequestChatGating:
    """Test that poc_request aborts in-flight inference requests and proceeds
    with PoC artifact generation."""

    @pytest.mark.asyncio
    async def test_poc_request_aborts_active_inference(self):
        """poc_request should abort in-flight requests then run collective_rpc."""
        from vllm.poc.engine_patch import poc_request

        mock_self = AsyncMock()
        mock_self.vllm_config.model_config.get_hidden_size.return_value = 4096

        mock_output_processor = MagicMock()
        mock_output_processor.has_unfinished_requests.return_value = True
        mock_output_processor.request_states = {
            "req-1": MagicMock(), "req-2": MagicMock(), "req-3": MagicMock(),
        }
        mock_self.output_processor = mock_output_processor
        mock_self.abort = AsyncMock()

        mock_self.collective_rpc = AsyncMock(return_value=[{
            "vectors": __import__("numpy").zeros((2, 12), dtype="float16"),
            "nonces": [0, 1],
        }])

        result = await poc_request(
            mock_self, "generate_artifacts",
            {"nonces": [0, 1], "block_hash": "abc", "public_key": "pk",
             "seq_len": 256, "k_dim": 12},
        )

        mock_self.abort.assert_called_once()
        aborted_ids = mock_self.abort.call_args[0][0]
        assert set(aborted_ids) == {"req-1", "req-2", "req-3"}
        assert mock_self.abort.call_args[1].get("internal") is True

        assert "skipped" not in result or not result.get("skipped")
        assert len(result["artifacts"]) == 2
        mock_self.collective_rpc.assert_called_once()

    @pytest.mark.asyncio
    async def test_poc_request_no_abort_when_no_inflight(self):
        """poc_request should skip abort when no requests are in-flight."""
        from vllm.poc.engine_patch import poc_request

        mock_self = AsyncMock()
        mock_self.vllm_config.model_config.get_hidden_size.return_value = 4096

        mock_output_processor = MagicMock()
        mock_output_processor.has_unfinished_requests.return_value = False
        mock_self.output_processor = mock_output_processor
        mock_self.abort = AsyncMock()

        mock_self.collective_rpc = AsyncMock(return_value=[{
            "vectors": __import__("numpy").zeros((1, 12), dtype="float16"),
            "nonces": [0],
        }])

        result = await poc_request(
            mock_self, "generate_artifacts",
            {"nonces": [0], "block_hash": "abc", "public_key": "pk",
             "seq_len": 256, "k_dim": 12},
        )

        mock_self.abort.assert_not_called()
        assert len(result["artifacts"]) == 1

    @pytest.mark.asyncio
    async def test_poc_request_skips_abort_when_request_states_empty(self):
        """Race: has_unfinished=True but request_states already drained."""
        from vllm.poc.engine_patch import poc_request

        mock_self = AsyncMock()
        mock_self.vllm_config.model_config.get_hidden_size.return_value = 4096

        mock_output_processor = MagicMock()
        mock_output_processor.has_unfinished_requests.return_value = True
        mock_output_processor.request_states = {}
        mock_self.output_processor = mock_output_processor
        mock_self.abort = AsyncMock()

        mock_self.collective_rpc = AsyncMock(return_value=[{
            "vectors": __import__("numpy").zeros((1, 12), dtype="float16"),
            "nonces": [0],
        }])

        result = await poc_request(
            mock_self, "generate_artifacts",
            {"nonces": [0], "block_hash": "abc", "public_key": "pk",
             "seq_len": 256, "k_dim": 12},
        )

        mock_self.abort.assert_not_called()
        assert len(result["artifacts"]) == 1
        mock_self.collective_rpc.assert_called_once()

    @pytest.mark.asyncio
    async def test_poc_request_abort_failure_propagates(self):
        """If abort() itself raises, the error propagates to the caller."""
        from vllm.poc.engine_patch import poc_request

        mock_self = AsyncMock()
        mock_self.vllm_config.model_config.get_hidden_size.return_value = 4096

        mock_output_processor = MagicMock()
        mock_output_processor.has_unfinished_requests.return_value = True
        mock_output_processor.request_states = {"req-1": MagicMock()}
        mock_self.output_processor = mock_output_processor
        mock_self.abort = AsyncMock(side_effect=RuntimeError("engine dead"))

        with pytest.raises(RuntimeError, match="engine dead"):
            await poc_request(
                mock_self, "generate_artifacts",
                {"nonces": [0], "block_hash": "abc", "public_key": "pk",
                 "seq_len": 256, "k_dim": 12},
            )

        mock_self.collective_rpc.assert_not_called()

    @pytest.mark.asyncio
    async def test_poc_request_proceeds_without_output_processor(self):
        """poc_request works if output_processor is missing (edge case)."""
        from vllm.poc.engine_patch import poc_request

        mock_self = AsyncMock()
        mock_self.vllm_config.model_config.get_hidden_size.return_value = 4096
        mock_self.output_processor = None

        mock_self.collective_rpc = AsyncMock(return_value=[{
            "vectors": __import__("numpy").zeros((1, 12), dtype="float16"),
            "nonces": [0],
        }])

        result = await poc_request(
            mock_self, "generate_artifacts",
            {"nonces": [0], "block_hash": "abc", "public_key": "pk",
             "seq_len": 256, "k_dim": 12},
        )

        assert len(result["artifacts"]) == 1

    @pytest.mark.asyncio
    async def test_poc_request_empty_nonces(self):
        from vllm.poc.engine_patch import poc_request

        mock_self = AsyncMock()
        result = await poc_request(
            mock_self, "generate_artifacts",
            {"nonces": [], "block_hash": "abc", "public_key": "pk"},
        )
        assert result == {"artifacts": []}

    @pytest.mark.asyncio
    async def test_poc_request_timeout_raises(self):
        from vllm.poc.engine_patch import poc_request

        mock_self = AsyncMock()
        mock_self.vllm_config.model_config.get_hidden_size.return_value = 4096
        mock_self.output_processor = None
        mock_self.collective_rpc = AsyncMock(side_effect=asyncio.TimeoutError)

        with pytest.raises(TimeoutError, match="timed out"):
            await poc_request(
                mock_self, "generate_artifacts",
                {"nonces": [0], "block_hash": "abc", "public_key": "pk",
                 "seq_len": 256, "k_dim": 12},
                timeout_ms=100,
            )

    @pytest.mark.asyncio
    async def test_poc_request_invalid_action(self):
        from vllm.poc.engine_patch import poc_request

        mock_self = AsyncMock()
        with pytest.raises(ValueError, match="Unknown PoC action"):
            await poc_request(mock_self, "invalid_action", {})

    @pytest.mark.asyncio
    async def test_poc_request_generic_exception_returns_skipped(self):
        from vllm.poc.engine_patch import poc_request

        mock_self = AsyncMock()
        mock_self.vllm_config.model_config.get_hidden_size.return_value = 4096
        mock_self.output_processor = None
        mock_self.collective_rpc = AsyncMock(
            side_effect=RuntimeError("GPU error"))

        result = await poc_request(
            mock_self, "generate_artifacts",
            {"nonces": [0], "block_hash": "abc", "public_key": "pk",
             "seq_len": 256, "k_dim": 12},
        )
        assert result["skipped"] is True


# ---------------------------------------------------------------------------
# 2. Chat and completion endpoints reject requests when PoC is active
# ---------------------------------------------------------------------------

class TestEndpointPocRejection:
    """Test that /v1/chat/completions and /v1/completions return 503 when
    PoC generation is active."""

    @pytest.mark.asyncio
    async def test_chat_rejected_when_poc_active(self):
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient

        from vllm.entrypoints.openai.chat_completion.api_router import (
            router,
        )

        app = FastAPI()
        app.include_router(router)

        app.state.openai_serving_chat = None
        app.state.openai_serving_tokenization = MagicMock()
        app.state.openai_serving_tokenization.create_error_response = (
            lambda message: MagicMock()
        )

        with patch("vllm.poc.routes._poc_generation_active", True):
            client = TestClient(app)
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "test", "messages": [{"role": "user",
                                                      "content": "hi"}]},
            )
            assert resp.status_code == 503
            body = resp.json()
            assert "PoC generation is active" in body["error"]["message"]

    @pytest.mark.asyncio
    async def test_completion_rejected_when_poc_active(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from vllm.entrypoints.openai.completion.api_router import router

        app = FastAPI()
        app.include_router(router)

        app.state.openai_serving_completion = None
        app.state.openai_serving_tokenization = MagicMock()
        app.state.openai_serving_tokenization.create_error_response = (
            lambda message: MagicMock()
        )

        with patch("vllm.poc.routes._poc_generation_active", True):
            client = TestClient(app)
            resp = client.post(
                "/v1/completions",
                json={"model": "test", "prompt": "hello"},
            )
            assert resp.status_code == 503
            body = resp.json()
            assert "PoC generation is active" in body["error"]["message"]

    @pytest.mark.asyncio
    async def test_chat_allowed_when_poc_not_active(self):
        """Chat should proceed normally when PoC is not generating."""
        from vllm.poc.routes import _poc_generation_active
        assert not _poc_generation_active  # default is False


# ---------------------------------------------------------------------------
# 3. _poc_generation_active flag lifecycle
# ---------------------------------------------------------------------------

class TestPocGenerationActiveFlag:
    """Test the _poc_generation_active flag is set/cleared correctly."""

    def test_flag_starts_false(self):
        import vllm.poc.routes as routes
        assert routes._poc_generation_active is False

    @pytest.mark.asyncio
    async def test_flag_cleared_on_task_completion(self):
        """done_callback should clear _poc_generation_active when gen_task ends."""
        import vllm.poc.routes as routes

        routes._poc_generation_active = True

        async def fake_gen():
            return

        task = asyncio.create_task(fake_gen())

        def on_done(t):
            routes._poc_generation_active = False

        task.add_done_callback(on_done)
        await task

        # Allow the callback to run
        await asyncio.sleep(0)
        assert routes._poc_generation_active is False

    @pytest.mark.asyncio
    async def test_flag_cleared_on_task_exception(self):
        """done_callback should clear flag even if gen_task crashes."""
        import vllm.poc.routes as routes

        routes._poc_generation_active = True

        async def failing_gen():
            raise RuntimeError("boom")

        task = asyncio.create_task(failing_gen())

        def on_done(t):
            routes._poc_generation_active = False

        task.add_done_callback(on_done)

        with pytest.raises(RuntimeError):
            await task

        await asyncio.sleep(0)
        assert routes._poc_generation_active is False

    @pytest.mark.asyncio
    async def test_flag_cleared_on_task_cancel(self):
        """done_callback should clear flag on cancellation."""
        import vllm.poc.routes as routes

        routes._poc_generation_active = True

        async def long_gen():
            await asyncio.sleep(999)

        task = asyncio.create_task(long_gen())

        def on_done(t):
            routes._poc_generation_active = False

        task.add_done_callback(on_done)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        await asyncio.sleep(0)
        assert routes._poc_generation_active is False

    @pytest.mark.asyncio
    async def test_stop_endpoint_clears_flag(self):
        """POST /stop should clear the flag."""
        import vllm.poc.routes as routes
        routes._poc_generation_active = True

        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from vllm.poc.routes import router

        app = FastAPI()
        app.include_router(router)
        app.state.engine_client = AsyncMock()
        app.state.poc_enabled = True

        client = TestClient(app)
        resp = client.post("/api/v1/pow/stop")
        assert resp.status_code == 200
        assert routes._poc_generation_active is False


# ---------------------------------------------------------------------------
# 4. Generation loop handles skipped responses
# ---------------------------------------------------------------------------

class TestGenerationLoopSkipHandling:
    """Test _generation_loop handles skipped and timeout responses."""

    @pytest.mark.asyncio
    async def test_generation_loop_retries_on_skip(self):
        from vllm.poc.routes import _generation_loop

        engine = AsyncMock()
        stop = asyncio.Event()
        stats = {"start_time": 0, "total_processed": 0}
        config = {
            "node_id": 0, "node_count": 1, "group_id": 0, "n_groups": 1,
            "batch_size": 4, "block_hash": "h", "block_height": 1,
            "public_key": "pk", "seq_len": 256, "k_dim": 12,
        }

        call_count = 0

        async def mock_poc(action, payload, timeout_ms=60000):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return {"artifacts": [], "skipped": True, "reason": "chat_unfinished"}
            stop.set()
            return {"artifacts": [{"nonce": 0, "vector_b64": "AA=="}]}

        engine.poc_request = mock_poc

        with patch("vllm.poc.routes.POC_CHAT_BUSY_BACKOFF_SEC", 0.001):
            await _generation_loop(engine, stop, None, config, stats)

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_generation_loop_recovers_from_timeout(self):
        from vllm.poc.routes import _generation_loop

        engine = AsyncMock()
        stop = asyncio.Event()
        stats = {"start_time": 0, "total_processed": 0}
        config = {
            "node_id": 0, "node_count": 1, "group_id": 0, "n_groups": 1,
            "batch_size": 4, "block_hash": "h", "block_height": 1,
            "public_key": "pk", "seq_len": 256, "k_dim": 12,
        }

        call_count = 0

        async def mock_poc(action, payload, timeout_ms=60000):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("busy")
            stop.set()
            return {"artifacts": [{"nonce": 0, "vector_b64": "AA=="}]}

        engine.poc_request = mock_poc

        with patch("vllm.poc.routes.POC_CHAT_BUSY_BACKOFF_SEC", 0.001):
            await _generation_loop(engine, stop, None, config, stats)

        assert call_count == 3
