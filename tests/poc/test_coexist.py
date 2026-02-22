"""Tests for PoC+Chat coexistence and priority modes."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vllm.poc.routes import _generation_loop
from vllm.poc.state import is_poc_active, set_poc_active


@pytest.fixture
def mock_engine_client():
    """Create a mock engine client for testing."""
    client = AsyncMock()
    client.poc_request = AsyncMock()
    client.poc_request.return_value = {
        "artifacts": [],
    }
    return client


@pytest.fixture(autouse=True)
def _reset_poc_state():
    """Ensure PoC state is clean before/after every test."""
    set_poc_active(False)
    yield
    set_poc_active(False)


class TestPoCPriorityMode:
    """PoC always aborts active chat requests before running GPU work."""

    def test_generate_artifacts_aborts_chat_requests(self):
        """PoC must abort every chat request in the scheduler."""
        from vllm.engine.multiprocessing.engine import MQLLMEngine
        from vllm.poc.data import Artifact

        mock_llm_engine = MagicMock()
        mock_llm_engine.has_unfinished_requests.return_value = False
        mock_scheduler = MagicMock()
        mock_seq_group = MagicMock()
        mock_seq_group.request_id = "req-123"
        mock_scheduler.waiting = [mock_seq_group]
        mock_scheduler.running = []
        mock_scheduler.swapped = []
        mock_llm_engine.scheduler = [mock_scheduler]

        mq_engine = MagicMock()
        mq_engine.engine = mock_llm_engine
        mq_engine._engine_step_in_progress = False
        mq_engine.input_socket.poll.return_value = 0
        mq_engine._abort_all_chat_requests = (
            MQLLMEngine._abort_all_chat_requests.__get__(
                mq_engine, MQLLMEngine))

        mock_manager = MagicMock()
        mock_manager.generate_artifacts.return_value = [
            Artifact(nonce=0, vector_b64="AAA="),
        ]
        mq_engine._poc_manager = mock_manager
        mq_engine._get_poc_manager = lambda: mock_manager

        result = MQLLMEngine._process_poc_action(
            mq_engine, "generate_artifacts", {
                "nonces": [0],
                "block_hash": "hash",
                "public_key": "key",
                "seq_len": 256,
                "k_dim": 12,
            })

        mock_llm_engine.abort_request.assert_called_with("req-123")
        mock_manager.generate_artifacts.assert_called_once()
        assert "skipped" not in result or result.get("skipped") is not True

    def test_generate_artifacts_skips_when_engine_step_in_progress(self):
        """Safety: PoC must not run on GPU while engine.step() is active."""
        from vllm.engine.multiprocessing.engine import MQLLMEngine

        mock_llm_engine = MagicMock()
        mock_llm_engine.has_unfinished_requests.return_value = False
        mock_llm_engine.scheduler = []

        mq_engine = MagicMock()
        mq_engine.engine = mock_llm_engine
        mq_engine._engine_step_in_progress = True
        mq_engine.input_socket.poll.return_value = 0
        mq_engine._abort_all_chat_requests = (
            MQLLMEngine._abort_all_chat_requests.__get__(
                mq_engine, MQLLMEngine))

        mock_manager = MagicMock()
        mq_engine._poc_manager = mock_manager
        mq_engine._get_poc_manager = lambda: mock_manager

        result = MQLLMEngine._process_poc_action(
            mq_engine, "generate_artifacts", {"nonces": [0, 1, 2]})

        assert result["skipped"] is True
        assert result["reason"] == "engine_step_in_progress"
        mock_manager.generate_artifacts.assert_not_called()

    def test_generate_artifacts_sets_poc_active(self):
        """_process_poc_action must set the PoC-active flag."""
        from vllm.engine.multiprocessing.engine import MQLLMEngine
        from vllm.poc.data import Artifact

        mock_llm_engine = MagicMock()
        mock_llm_engine.has_unfinished_requests.return_value = False
        mock_llm_engine.scheduler = []
        mq_engine = MagicMock()
        mq_engine.engine = mock_llm_engine
        mq_engine._engine_step_in_progress = False
        mq_engine.input_socket.poll.return_value = 0
        mq_engine._abort_all_chat_requests = (
            MQLLMEngine._abort_all_chat_requests.__get__(
                mq_engine, MQLLMEngine))

        mock_manager = MagicMock()
        mock_manager.generate_artifacts.return_value = [
            Artifact(nonce=0, vector_b64="AAA=")]
        mq_engine._poc_manager = mock_manager
        mq_engine._get_poc_manager = lambda: mock_manager

        assert not is_poc_active()
        MQLLMEngine._process_poc_action(
            mq_engine, "generate_artifacts", {
                "nonces": [0], "block_hash": "h", "public_key": "k",
                "seq_len": 256, "k_dim": 12,
            })
        assert is_poc_active()


class TestEndSession:
    """The ``end_session`` action clears the PoC flag in the engine process."""

    def test_end_session_clears_flag(self):
        from vllm.engine.multiprocessing.engine import MQLLMEngine

        mq_engine = MagicMock()
        set_poc_active(True)
        assert is_poc_active()

        result = MQLLMEngine._process_poc_action(
            mq_engine, "end_session", {})

        assert result.get("ok") is True
        assert not is_poc_active()


class TestPoCConditions:
    """PoC must return skip reasons for all guard conditions."""

    def test_generate_artifacts_skips_when_pending_input(self):
        from vllm.engine.multiprocessing.engine import MQLLMEngine

        mock_llm_engine = MagicMock()
        mock_llm_engine.scheduler = []

        mq_engine = MagicMock()
        mq_engine.engine = mock_llm_engine
        mq_engine._engine_step_in_progress = False
        mq_engine.input_socket.poll.return_value = 1
        mq_engine._abort_all_chat_requests = (
            MQLLMEngine._abort_all_chat_requests.__get__(
                mq_engine, MQLLMEngine))

        mock_manager = MagicMock()
        mq_engine._poc_manager = mock_manager
        mq_engine._get_poc_manager = lambda: mock_manager

        result = MQLLMEngine._process_poc_action(
            mq_engine, "generate_artifacts", {"nonces": [0, 1, 2]})

        assert result["skipped"] is True
        assert result["reason"] == "pending_input"
        mock_manager.generate_artifacts.assert_not_called()

    def test_generate_artifacts_skips_when_chat_unfinished(self):
        from vllm.engine.multiprocessing.engine import MQLLMEngine

        mock_llm_engine = MagicMock()
        mock_llm_engine.scheduler = []
        mock_llm_engine.has_unfinished_requests.return_value = True

        mq_engine = MagicMock()
        mq_engine.engine = mock_llm_engine
        mq_engine._engine_step_in_progress = False
        mq_engine.input_socket.poll.return_value = 0
        mq_engine._abort_all_chat_requests = (
            MQLLMEngine._abort_all_chat_requests.__get__(
                mq_engine, MQLLMEngine))

        mock_manager = MagicMock()
        mq_engine._poc_manager = mock_manager
        mq_engine._get_poc_manager = lambda: mock_manager

        result = MQLLMEngine._process_poc_action(
            mq_engine, "generate_artifacts", {"nonces": [0, 1, 2]})

        assert result["skipped"] is True
        assert result["reason"] == "chat_unfinished"
        mock_manager.generate_artifacts.assert_not_called()


class TestChatRejectionWhilePoC:
    """_handle_process_request must reject chat RPCs when PoC is active."""

    def test_handle_process_request_rejects_during_poc(self):
        from vllm.engine.multiprocessing.engine import MQLLMEngine
        from vllm.engine.multiprocessing import RPCProcessRequest, RPCError

        set_poc_active(True)

        mq_engine = MagicMock()
        mq_engine._errored_with = None
        outputs_sent = []
        mq_engine._send_outputs = lambda o: outputs_sent.append(o)

        request = MagicMock(spec=RPCProcessRequest)
        request.request_id = "chat-req-1"

        MQLLMEngine._handle_process_request(mq_engine, request)

        assert len(outputs_sent) == 1
        rpc_err = outputs_sent[0]
        assert isinstance(rpc_err, RPCError)
        assert rpc_err.request_id == "chat-req-1"
        assert not rpc_err.is_engine_errored
        mq_engine.engine.add_request.assert_not_called()

    def test_handle_process_request_allows_when_poc_inactive(self):
        from vllm.engine.multiprocessing.engine import MQLLMEngine
        from vllm.engine.multiprocessing import RPCProcessRequest

        set_poc_active(False)

        mq_engine = MagicMock()
        mq_engine._errored_with = None
        mq_engine.log_requests = False

        request = MagicMock(spec=RPCProcessRequest)
        request.request_id = "chat-req-2"
        request.prompt = "Hello"
        request.params = MagicMock()
        request.lora_request = None
        request.trace_headers = None
        request.prompt_adapter_request = None
        request.priority = 0

        MQLLMEngine._handle_process_request(mq_engine, request)

        mq_engine.engine.add_request.assert_called_once()


class TestGenerationLoopBackoff:
    """Generation loop retries with backoff when engine returns skipped."""

    @pytest.mark.asyncio
    async def test_generation_loop_backs_off_on_skip(
            self, mock_engine_client):
        stop_event = asyncio.Event()
        config = {
            "block_hash": "hash",
            "block_height": 100,
            "public_key": "key",
            "node_id": 0,
            "node_count": 1,
            "group_id": 0,
            "n_groups": 1,
            "batch_size": 4,
            "seq_len": 256,
            "k_dim": 12,
        }
        stats = {"start_time": 0, "total_processed": 0}

        call_count = 0

        async def mock_poc_request(action, payload, timeout_ms=None):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return {"skipped": True, "artifacts": []}
            stop_event.set()
            return {"artifacts": [], "skipped": True}

        mock_engine_client.poc_request = mock_poc_request

        with patch('vllm.poc.routes.POC_CHAT_BUSY_BACKOFF_SEC', 0.001):
            task = asyncio.create_task(
                _generation_loop(mock_engine_client, stop_event,
                                 None, config, stats))
            await asyncio.sleep(0.1)
            stop_event.set()
            try:
                await asyncio.wait_for(task, timeout=1.0)
            except asyncio.CancelledError:
                pass

        assert call_count >= 2


class TestUnknownAction:
    """Both engines must reject unrecognised PoC actions."""

    def test_mp_engine_rejects_unknown_action(self):
        from vllm.engine.multiprocessing.engine import MQLLMEngine

        mq_engine = MagicMock()
        mq_engine._get_poc_manager = MagicMock()

        with pytest.raises(ValueError, match="Unknown PoC action"):
            MQLLMEngine._process_poc_action(
                mq_engine, "unknown_action", {})

    def test_mp_engine_rejects_old_actions(self):
        from vllm.engine.multiprocessing.engine import MQLLMEngine

        mq_engine = MagicMock()
        mq_engine._get_poc_manager = MagicMock()

        for old_action in ["init", "start_generate", "stop",
                           "status", "run_batch"]:
            with pytest.raises(ValueError, match="Unknown PoC action"):
                MQLLMEngine._process_poc_action(
                    mq_engine, old_action, {})
