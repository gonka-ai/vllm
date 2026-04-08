"""Tests for PoC API routes."""
import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.poc.routes import (
    router, _poc_tasks, _is_generation_active,
    POC_BATCH_SIZE_DEFAULT, PoCInitGenerateRequest, PoCGenerateRequest,
    NonceIterator,
)
from vllm.poc.generate_queue import GenerateJob, GenerateResult, get_queue, clear_queue, POC_MAX_QUEUED_NONCES
from vllm.poc.config import PoCState


async def _mock_generation_loop(engine_client, stop_event, callback_sender, config, stats):
    try:
        await stop_event.wait()
    except asyncio.CancelledError:
        pass


@pytest.fixture
def mock_engine_client():
    client = AsyncMock()
    client.poc_request.return_value = {"artifacts": []}
    return client


@pytest.fixture
def app_with_poc(mock_engine_client):
    app = FastAPI()
    app.include_router(router)
    app.state.engine_client = mock_engine_client
    app.state.poc_deployed = {"model": "test-model", "seq_len": 256, "k_dim": 12}
    mock_base_path = MagicMock()
    mock_base_path.model_path = "test-model"
    mock_base_path.name = "test-model"
    mock_serving_models = MagicMock()
    mock_serving_models.base_model_paths = [mock_base_path]
    app.state.openai_serving_models = mock_serving_models
    return app


@pytest.fixture
def client(app_with_poc):
    _poc_tasks.clear()
    with patch('vllm.poc.routes._generation_loop', _mock_generation_loop):
        yield TestClient(app_with_poc)
    for app_id, tasks in list(_poc_tasks.items()):
        if tasks.get("stop_event"):
            tasks["stop_event"].set()
        if tasks.get("gen_task"):
            tasks["gen_task"].cancel()
    _poc_tasks.clear()


class TestPoCInitGenerate:
    def test_init_generate_starts_generation(self, client, mock_engine_client):
        mock_engine_client.poc_request.return_value = {"artifacts": [{"nonce": 0, "vector_b64": "AAAA"}]}
        response = client.post("/api/v1/pow/init/generate", json={
            "block_hash": "abc123", "block_height": 100, "public_key": "pubkey123",
            "node_id": 0, "node_count": 1, "batch_size": 32,
            "params": {"model": "test-model", "seq_len": 256, "k_dim": 12},
        })
        assert response.status_code == 200
        assert response.json()["status"] == "OK"
        assert response.json()["pow_status"]["status"] == "GENERATING"

    def test_init_generate_conflict_when_already_generating(self, client, app_with_poc):
        app_id = id(app_with_poc)
        mock_task = MagicMock()
        mock_task.done.return_value = False
        _poc_tasks[app_id] = {"gen_task": mock_task, "stop_event": asyncio.Event(), "config": {}, "stats": {}}
        response = client.post("/api/v1/pow/init/generate", json={
            "block_hash": "abc456", "block_height": 101, "public_key": "pubkey123",
            "node_id": 0, "node_count": 1,
            "params": {"model": "test-model", "seq_len": 256, "k_dim": 12},
        })
        assert response.status_code == 409

    def test_init_generate_params_mismatch(self, client):
        response = client.post("/api/v1/pow/init/generate", json={
            "block_hash": "abc123", "block_height": 100, "public_key": "pubkey123",
            "node_id": 0, "node_count": 1,
            "params": {"model": "wrong-model", "seq_len": 256, "k_dim": 12},
        })
        assert response.status_code == 409

    def test_init_generate_extra_params_rejected(self, client):
        response = client.post("/api/v1/pow/init/generate", json={
            "block_hash": "abc123", "block_height": 100, "public_key": "pubkey123",
            "node_id": 0, "node_count": 1,
            "params": {"model": "test-model", "seq_len": 256, "k_dim": 12, "extra": "bad"},
        })
        assert response.status_code == 422

    def test_init_generate_poc_stronger_rng_stored_in_config(self, client, mock_engine_client, app_with_poc):
        """poc_stronger_rng=True must be stored in the generation config."""
        mock_engine_client.poc_request.return_value = {"artifacts": []}
        response = client.post("/api/v1/pow/init/generate", json={
            "block_hash": "abc123", "block_height": 100, "public_key": "pubkey123",
            "node_id": 0, "node_count": 1,
            "params": {"model": "test-model", "seq_len": 256, "k_dim": 12},
            "poc_stronger_rng": True,
        })
        assert response.status_code == 200
        assert _poc_tasks[id(app_with_poc)]["config"]["poc_stronger_rng"] is True


class TestPoCGenerate:
    def test_generate_returns_artifacts(self, client, mock_engine_client):
        mock_engine_client.poc_request.return_value = {
            "artifacts": [{"nonce": 0, "vector_b64": "AAAA"}, {"nonce": 1, "vector_b64": "BBBB"}],
        }
        response = client.post("/api/v1/pow/generate", json={
            "block_hash": "abc123", "block_height": 100, "public_key": "pubkey123",
            "node_id": 0, "node_count": 1, "nonces": [0, 1],
            "params": {"model": "test-model", "seq_len": 256, "k_dim": 12}, "wait": True,
        })
        assert response.status_code == 200
        assert response.json()["status"] == "completed"
        assert len(response.json()["artifacts"]) == 2

    def test_generate_wait_false_returns_queued(self, client):
        response = client.post("/api/v1/pow/generate", json={
            "block_hash": "abc123", "block_height": 100, "public_key": "pubkey123",
            "node_id": 0, "node_count": 1, "nonces": [0, 1, 2],
            "params": {"model": "test-model", "seq_len": 256, "k_dim": 12}, "wait": False,
        })
        assert response.status_code == 200
        assert response.json()["status"] == "queued"
        assert response.json()["queued_count"] == 3

    def test_generate_with_validation_detects_mismatch(self, client, mock_engine_client):
        mock_engine_client.poc_request.return_value = {
            "artifacts": [{"nonce": 0, "vector_b64": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"}],
        }
        response = client.post("/api/v1/pow/generate", json={
            "block_hash": "abc123", "block_height": 100, "public_key": "pubkey123",
            "node_id": 0, "node_count": 1, "nonces": [0],
            "params": {"model": "test-model", "seq_len": 256, "k_dim": 12}, "wait": True,
            "validation": {"artifacts": [{"nonce": 0, "vector_b64": "ADwAPAA8ADwAPAA8ADwAPAA8ADwAPAA8"}]},
        })
        assert response.status_code == 200
        assert response.json()["n_mismatch"] == 1
        assert response.json()["fraud_detected"] is True

    def test_generate_validation_nonce_mismatch_error(self, client):
        response = client.post("/api/v1/pow/generate", json={
            "block_hash": "abc123", "block_height": 100, "public_key": "pubkey123",
            "node_id": 0, "node_count": 1, "nonces": [0, 1],
            "params": {"model": "test-model", "seq_len": 256, "k_dim": 12}, "wait": True,
            "validation": {"artifacts": [{"nonce": 0, "vector_b64": "AAA="}, {"nonce": 5, "vector_b64": "BBB="}]},
        })
        assert response.status_code == 400

    def test_generate_propagates_poc_stronger_rng(self, client, mock_engine_client):
        """poc_stronger_rng=True must reach the engine_client.poc_request payload."""
        mock_engine_client.poc_request.return_value = {
            "artifacts": [{"nonce": 0, "vector_b64": "AAAA"}],
        }
        client.post("/api/v1/pow/generate", json={
            "block_hash": "abc123", "block_height": 100, "public_key": "pubkey123",
            "node_id": 0, "node_count": 1, "nonces": [0],
            "params": {"model": "test-model", "seq_len": 256, "k_dim": 12},
            "wait": True, "poc_stronger_rng": True,
        })
        payload = mock_engine_client.poc_request.call_args[0][1]
        assert payload["poc_stronger_rng"] is True


class TestPoCStatus:
    def test_get_status_idle(self, client):
        response = client.get("/api/v1/pow/status")
        assert response.status_code == 200
        assert response.json()["status"] == "IDLE"

    def test_get_status_generating(self, client, app_with_poc):
        app_id = id(app_with_poc)
        mock_task = MagicMock()
        mock_task.done.return_value = False
        _poc_tasks[app_id] = {
            "gen_task": mock_task, "stop_event": asyncio.Event(),
            "config": {"block_hash": "abc123", "block_height": 100, "public_key": "pk",
                       "node_id": 0, "node_count": 1, "seq_len": 256, "k_dim": 12},
            "stats": {"start_time": time.time(), "total_processed": 500},
        }
        response = client.get("/api/v1/pow/status")
        assert response.status_code == 200
        assert response.json()["status"] == "GENERATING"


class TestPoCStop:
    def test_stop_round(self, client, mock_engine_client):
        mock_engine_client.poc_request.return_value = {"artifacts": []}
        client.post("/api/v1/pow/init/generate", json={
            "block_hash": "abc123", "block_height": 100, "public_key": "pubkey123",
            "node_id": 0, "node_count": 1,
            "params": {"model": "test-model", "seq_len": 256, "k_dim": 12},
        })
        response = client.post("/api/v1/pow/stop")
        assert response.status_code == 200
        assert response.json()["pow_status"]["status"] == "STOPPED"
        assert client.get("/api/v1/pow/status").json()["status"] == "IDLE"


class TestNonceIterator:
    def test_single_node_single_group(self):
        it = NonceIterator(node_id=0, n_nodes=1, group_id=0, n_groups=1)
        assert it.take(5) == [0, 1, 2, 3, 4]

    def test_multi_node_single_group(self):
        it0 = NonceIterator(node_id=0, n_nodes=3, group_id=0, n_groups=1)
        it1 = NonceIterator(node_id=1, n_nodes=3, group_id=0, n_groups=1)
        it2 = NonceIterator(node_id=2, n_nodes=3, group_id=0, n_groups=1)
        assert it0.take(3) == [0, 3, 6]
        assert it1.take(3) == [1, 4, 7]
        assert it2.take(3) == [2, 5, 8]

    def test_multi_group(self):
        # 2 nodes, 2 groups: step = 2*2 = 4
        # group 0, node 0: offset=0, nonces: 0, 4, 8, 12...
        # group 0, node 1: offset=1, nonces: 1, 5, 9, 13...
        # group 1, node 0: offset=2, nonces: 2, 6, 10, 14...
        # group 1, node 1: offset=3, nonces: 3, 7, 11, 15...
        it_g0_n0 = NonceIterator(node_id=0, n_nodes=2, group_id=0, n_groups=2)
        it_g0_n1 = NonceIterator(node_id=1, n_nodes=2, group_id=0, n_groups=2)
        it_g1_n0 = NonceIterator(node_id=0, n_nodes=2, group_id=1, n_groups=2)
        it_g1_n1 = NonceIterator(node_id=1, n_nodes=2, group_id=1, n_groups=2)
        assert it_g0_n0.take(4) == [0, 4, 8, 12]
        assert it_g0_n1.take(4) == [1, 5, 9, 13]
        assert it_g1_n0.take(4) == [2, 6, 10, 14]
        assert it_g1_n1.take(4) == [3, 7, 11, 15]

    def test_all_nonces_disjoint(self):
        # 3 nodes, 2 groups = 6 total iterators covering all nonces
        all_nonces = set()
        for group_id in range(2):
            for node_id in range(3):
                it = NonceIterator(node_id=node_id, n_nodes=3, group_id=group_id, n_groups=2)
                nonces = it.take(10)
                assert len(set(nonces) & all_nonces) == 0, "Nonces overlap!"
                all_nonces.update(nonces)
        # Should cover 0..59 exactly
        assert all_nonces == set(range(60))


class TestGenerateQueue:
    def test_poll_unknown_request_returns_404(self, client):
        assert client.get("/api/v1/pow/generate/unknown-id").status_code == 404

    def test_poll_queued_request_returns_status(self, client):
        response = client.post("/api/v1/pow/generate", json={
            "block_hash": "abc123", "block_height": 100, "public_key": "pubkey123",
            "node_id": 0, "node_count": 1, "nonces": [0],
            "params": {"model": "test-model", "seq_len": 256, "k_dim": 12}, "wait": False,
        })
        request_id = response.json()["request_id"]
        poll = client.get(f"/api/v1/pow/generate/{request_id}")
        assert poll.status_code == 200
        assert poll.json()["status"] in ["queued", "running", "completed"]


class TestQueueCap:
    @pytest.mark.asyncio
    async def test_queue_nonce_cap_enforced(self):
        queue = get_queue()
        await queue.clear_all()
        mock_client = AsyncMock()
        big_job = GenerateJob(
            request_id="big", engine_client=mock_client, app_id=1,
            block_hash="abc", block_height=100, public_key="pk",
            node_id=0, node_count=1, nonces=list(range(POC_MAX_QUEUED_NONCES + 1)),
            seq_len=256, k_dim=12, batch_size=1000,
        )
        assert await queue.enqueue(big_job) is None
        await queue.clear_all()


class TestGenerateQueueIntegration:
    @pytest.mark.asyncio
    async def test_queue_process_job(self):
        from vllm.poc.generate_queue import GenerateQueue
        queue = GenerateQueue()
        mock_client = AsyncMock()
        mock_client.poc_request.return_value = {"artifacts": [{"nonce": 0, "vector_b64": "AAAA"}]}
        job = GenerateJob(
            request_id="job1", engine_client=mock_client, app_id=1,
            block_hash="abc", block_height=100, public_key="pk",
            node_id=0, node_count=1, nonces=[0], seq_len=256, k_dim=12, batch_size=10,
        )
        result = await queue._process_job(job)
        assert result["status"] == "completed"


class TestBatchSizeDefaults:
    def test_batch_size_default_constant_exists(self):
        assert POC_BATCH_SIZE_DEFAULT == 32

    def test_init_generate_uses_batch_size_default(self):
        req = PoCInitGenerateRequest(
            block_hash="abc", block_height=100, public_key="pk",
            node_id=0, node_count=1,
            params={"model": "test", "seq_len": 256, "k_dim": 12},
        )
        assert req.batch_size == POC_BATCH_SIZE_DEFAULT

    def test_generate_uses_batch_size_default(self):
        req = PoCGenerateRequest(
            block_hash="abc", block_height=100, public_key="pk",
            node_id=0, node_count=1, nonces=[0, 1],
            params={"model": "test", "seq_len": 256, "k_dim": 12},
        )
        assert req.batch_size == POC_BATCH_SIZE_DEFAULT

    def test_batch_size_can_be_overridden(self):
        req = PoCGenerateRequest(
            block_hash="abc", block_height=100, public_key="pk",
            node_id=0, node_count=1, nonces=[0, 1], batch_size=100,
            params={"model": "test", "seq_len": 256, "k_dim": 12},
        )
        assert req.batch_size == 100
