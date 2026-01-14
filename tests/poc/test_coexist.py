"""Tests for PoC+Chat coexistence (chat-priority gating)."""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.poc.routes import (
    router, _generation_loop,
    POC_CHAT_BUSY_BACKOFF_SEC,
)
from vllm.poc.config import PoCState


@pytest.fixture
def mock_engine_client():
    """Create a mock engine client for testing."""
    client = AsyncMock()
    client.poc_request = AsyncMock()
    client.poc_request.return_value = {
        "artifacts": [],
    }
    return client


class TestChatPriorityGating:
    """Tests for chat-priority gating in PoC GPU actions."""
    
    def test_generate_artifacts_skips_when_pending_input(self):
        """Test generate_artifacts returns skip when there's pending input (chat waiting)."""
        from vllm.engine.multiprocessing.engine import MQLLMEngine
        
        mock_llm_engine = MagicMock()
        
        mq_engine = MagicMock()
        mq_engine.engine = mock_llm_engine
        mq_engine._engine_step_in_progress = False
        mq_engine.input_socket.poll.return_value = 1  # Pending input
        
        mock_manager = MagicMock()
        mq_engine._poc_manager = mock_manager
        mq_engine._get_poc_manager = lambda: mock_manager
        
        result = MQLLMEngine._process_poc_action(mq_engine, "generate_artifacts", {
            "nonces": [0, 1, 2],
        })
        
        assert result["skipped"] is True
        assert result["reason"] == "pending_input"
        mock_manager.generate_artifacts.assert_not_called()
    
    def test_generate_artifacts_skips_when_engine_step_in_progress(self):
        """Test generate_artifacts returns skip when _engine_step_in_progress is True."""
        from vllm.engine.multiprocessing.engine import MQLLMEngine
        
        mock_llm_engine = MagicMock()
        
        mq_engine = MagicMock()
        mq_engine.engine = mock_llm_engine
        mq_engine._engine_step_in_progress = True
        mq_engine.input_socket.poll.return_value = 0  # No pending input
        
        mock_manager = MagicMock()
        mq_engine._poc_manager = mock_manager
        mq_engine._get_poc_manager = lambda: mock_manager
        
        result = MQLLMEngine._process_poc_action(mq_engine, "generate_artifacts", {
            "nonces": [0, 1, 2],
        })
        
        assert result["skipped"] is True
        assert result["reason"] == "engine_step_in_progress"
        mock_manager.generate_artifacts.assert_not_called()
    
    def test_generate_artifacts_skips_when_chat_unfinished(self):
        """Test generate_artifacts returns skip when chat has unfinished requests."""
        from vllm.engine.multiprocessing.engine import MQLLMEngine
        
        mock_llm_engine = MagicMock()
        mock_llm_engine.has_unfinished_requests.return_value = True
        
        mq_engine = MagicMock()
        mq_engine.engine = mock_llm_engine
        mq_engine._engine_step_in_progress = False
        mq_engine.input_socket.poll.return_value = 0  # No pending input
        
        mock_manager = MagicMock()
        mq_engine._poc_manager = mock_manager
        mq_engine._get_poc_manager = lambda: mock_manager
        
        result = MQLLMEngine._process_poc_action(mq_engine, "generate_artifacts", {
            "nonces": [0, 1, 2],
        })
        
        assert result["skipped"] is True
        assert result["reason"] == "chat_unfinished"
        mock_manager.generate_artifacts.assert_not_called()
    
    def test_generate_artifacts_proceeds_when_all_checks_pass(self):
        """Test generate_artifacts proceeds when no pending input, not in step, and no chat."""
        from vllm.engine.multiprocessing.engine import MQLLMEngine
        from vllm.poc.data import Artifact
        
        mock_llm_engine = MagicMock()
        mock_llm_engine.has_unfinished_requests.return_value = False
        
        mq_engine = MagicMock()
        mq_engine.engine = mock_llm_engine
        mq_engine._engine_step_in_progress = False
        mq_engine.input_socket.poll.return_value = 0  # No pending input
        
        mock_manager = MagicMock()
        mock_manager.generate_artifacts.return_value = [
            Artifact(nonce=0, vector_b64="AAA="),
            Artifact(nonce=1, vector_b64="BBB="),
        ]
        mq_engine._poc_manager = mock_manager
        mq_engine._get_poc_manager = lambda: mock_manager
        
        result = MQLLMEngine._process_poc_action(mq_engine, "generate_artifacts", {
            "nonces": [0, 1],
            "block_hash": "hash",
            "public_key": "key",
            "seq_len": 256,
            "k_dim": 12,
        })
        
        # Should call _prepare_for_poc_gpu_work before generate_artifacts
        mq_engine._prepare_for_poc_gpu_work.assert_called_once()
        mock_manager.generate_artifacts.assert_called_once()
        assert "skipped" not in result or result.get("skipped") is not True
        assert len(result["artifacts"]) == 2


# Note: AsyncLLMEngine (in-process mode) tests are skipped because _AsyncLLMEngine
# is difficult to mock correctly due to class proxy behavior at module load time.
# The main PoC behavior is tested via MQLLMEngine (MP mode) tests above.


class TestGenerationLoopBackoff:
    """Tests for generation loop backoff behavior."""
    
    @pytest.mark.asyncio
    async def test_generation_loop_backs_off_on_skip(self, mock_engine_client):
        """Test that generation loop backs off when engine returns skipped."""
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
                _generation_loop(mock_engine_client, stop_event, None, config, stats)
            )
            await asyncio.sleep(0.1)
            stop_event.set()
            try:
                await asyncio.wait_for(task, timeout=1.0)
            except asyncio.CancelledError:
                pass
        
        assert call_count >= 2


class TestUnknownAction:
    """Tests for unknown action handling."""
    
    def test_mp_engine_rejects_unknown_action(self):
        """Test MP engine rejects unknown actions."""
        from vllm.engine.multiprocessing.engine import MQLLMEngine
        
        mq_engine = MagicMock()
        mq_engine._get_poc_manager = MagicMock()
        
        with pytest.raises(ValueError, match="Unknown PoC action"):
            MQLLMEngine._process_poc_action(mq_engine, "unknown_action", {})
    
    def test_mp_engine_rejects_old_actions(self):
        """Test MP engine rejects old actions like run_batch, init, etc."""
        from vllm.engine.multiprocessing.engine import MQLLMEngine
        
        mq_engine = MagicMock()
        mq_engine._get_poc_manager = MagicMock()
        
        for old_action in ["init", "start_generate", "stop", "status", "run_batch"]:
            with pytest.raises(ValueError, match="Unknown PoC action"):
                MQLLMEngine._process_poc_action(mq_engine, old_action, {})
    
    # Note: async_engine_rejects_unknown_action test is skipped because
    # _AsyncLLMEngine is difficult to mock correctly. The behavior is
    # covered by the MP engine test above.
