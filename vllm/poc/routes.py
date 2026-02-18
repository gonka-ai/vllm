"""PoC API routes for vLLM server."""
import asyncio
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, ConfigDict

from vllm.logger import init_logger
from .config import PoCState
from .data import Artifact, DEFAULT_DIST_THRESHOLD, DEFAULT_P_MISMATCH, DEFAULT_FRAUD_THRESHOLD
from .callbacks import CallbackSender
from .generate_queue import GenerateJob, get_queue, clear_queue, POC_MAX_QUEUED_NONCES
from .validation import run_validation

logger = init_logger(__name__)

router = APIRouter(prefix="/api/v1/pow", tags=["PoC"])

POC_CALLBACK_INTERVAL_SEC = float(os.environ.get("POC_CALLBACK_INTERVAL_SEC", "5"))
POC_GENERATE_CHUNK_TIMEOUT_SEC = float(os.environ.get("POC_GENERATE_CHUNK_TIMEOUT_SEC", "60"))
POC_CHAT_BUSY_BACKOFF_SEC = 0.05
POC_RPC_TIMEOUT_MS = int(os.environ.get("POC_RPC_TIMEOUT_MS", "60000"))
POC_BATCH_SIZE_DEFAULT = int(os.environ.get("POC_BATCH_SIZE_DEFAULT", "32"))

_poc_tasks: Dict[int, Dict[str, Any]] = {}


# =============================================================================
# Request/Response Models
# =============================================================================

class PoCParamsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    seq_len: int
    k_dim: int = 12


class PoCInitGenerateRequest(BaseModel):
    block_hash: str
    block_height: int
    public_key: str
    node_id: int
    node_count: int
    group_id: int = 0
    n_groups: int = 1
    batch_size: int = POC_BATCH_SIZE_DEFAULT
    params: PoCParamsModel
    url: Optional[str] = None


@dataclass
class NonceIterator:
    """Iterator for nonces with multi-node and multi-group support."""
    node_id: int
    n_nodes: int
    group_id: int
    n_groups: int
    _current_x: int = 0

    def __iter__(self):
        return self

    def __next__(self) -> int:
        offset = self.node_id + self.group_id * self.n_nodes
        step = self.n_groups * self.n_nodes
        value = offset + self._current_x * step
        self._current_x += 1
        return value

    def take(self, n: int) -> List[int]:
        """Take the next n nonces."""
        return [next(self) for _ in range(n)]


class ArtifactModel(BaseModel):
    nonce: int
    vector_b64: str


class ValidationModel(BaseModel):
    artifacts: List[ArtifactModel]


class StatTestModel(BaseModel):
    dist_threshold: float = DEFAULT_DIST_THRESHOLD
    p_mismatch: float = DEFAULT_P_MISMATCH
    fraud_threshold: float = DEFAULT_FRAUD_THRESHOLD


class PoCGenerateRequest(BaseModel):
    block_hash: str
    block_height: int
    public_key: str
    node_id: int
    node_count: int
    nonces: List[int]
    params: PoCParamsModel
    batch_size: int = POC_BATCH_SIZE_DEFAULT
    wait: bool = False
    url: Optional[str] = None
    validation: Optional[ValidationModel] = None
    stat_test: Optional[StatTestModel] = None


# =============================================================================
# Helpers
# =============================================================================

async def get_engine_client(request: Request):
    engine_client = getattr(request.app.state, 'engine_client', None)
    if engine_client is None:
        raise HTTPException(status_code=503, detail="Engine not available")
    return engine_client


def check_params_match(request: Request, params: PoCParamsModel):
    """Check params match deployed config. Raises 409 on mismatch."""
    serving_models = getattr(request.app.state, 'openai_serving_models', None)
    if serving_models and hasattr(serving_models, 'base_model_paths'):
        base_paths = serving_models.base_model_paths
        if base_paths:
            model_path = base_paths[0].model_path
            served_names = [p.name for p in base_paths]
            valid_models = {model_path} | set(served_names)
            if params.model not in valid_models:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "params mismatch",
                        "requested": {"model": params.model, "seq_len": params.seq_len, "k_dim": params.k_dim},
                        "deployed": {"model": list(valid_models), "seq_len": None, "k_dim": None},
                    }
                )
    
    deployed = getattr(request.app.state, 'poc_deployed', None)
    if deployed:
        mismatches = []
        if deployed.get("model") and params.model != deployed["model"]:
            mismatches.append("model")
        if deployed.get("seq_len") and params.seq_len != deployed["seq_len"]:
            mismatches.append("seq_len")
        if deployed.get("k_dim") and params.k_dim != deployed["k_dim"]:
            mismatches.append("k_dim")
        
        if mismatches:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "params mismatch",
                    "fields": mismatches,
                    "requested": {"model": params.model, "seq_len": params.seq_len, "k_dim": params.k_dim},
                    "deployed": deployed,
                }
            )


def _is_generation_active(app_id: int) -> bool:
    tasks = _poc_tasks.get(app_id)
    if not tasks:
        return False
    gen_task = tasks.get("gen_task")
    return gen_task is not None and not gen_task.done()


def _get_api_status(app_id: int) -> dict:
    tasks = _poc_tasks.get(app_id)
    
    if not tasks or not _is_generation_active(app_id):
        return {"status": PoCState.IDLE.value, "config": None, "stats": None}
    
    config = tasks.get("config", {})
    stats = tasks.get("stats", {})
    start_time = stats.get("start_time", 0)
    total_processed = stats.get("total_processed", 0)
    elapsed = time.time() - start_time if start_time > 0 else 0
    nonces_per_second = total_processed / elapsed if elapsed > 0 else 0
    
    return {
        "status": PoCState.GENERATING.value,
        "config": {
            "block_hash": config.get("block_hash"),
            "block_height": config.get("block_height"),
            "public_key": config.get("public_key"),
            "node_id": config.get("node_id"),
            "node_count": config.get("node_count"),
            "group_id": config.get("group_id"),
            "n_groups": config.get("n_groups"),
            "seq_len": config.get("seq_len"),
            "k_dim": config.get("k_dim"),
        },
        "stats": {
            "total_processed": total_processed,
            "nonces_per_second": nonces_per_second,
        },
    }


async def _cancel_poc_tasks(app_id: int):
    tasks = _poc_tasks.pop(app_id, None)
    if tasks:
        if tasks.get("stop_event"):
            tasks["stop_event"].set()
        if tasks.get("gen_task"):
            tasks["gen_task"].cancel()
            try:
                await tasks["gen_task"]
            except asyncio.CancelledError:
                pass
        if tasks.get("callback_sender"):
            tasks["callback_sender"].clear()



async def _compute_artifacts_chunk(
    engine_client,
    nonces: List[int],
    block_hash: str,
    public_key: str,
    seq_len: int,
    k_dim: int,
    timeout_sec: float = POC_GENERATE_CHUNK_TIMEOUT_SEC,
    check_cancelled: Optional[callable] = None,
) -> List[Dict]:
    """Compute artifacts for a chunk with backoff on skip."""
    chunk_start_time = time.time()
    
    while True:
        if check_cancelled and check_cancelled():
            raise RuntimeError("Cancelled")
        
        result = await engine_client.poc_request("generate_artifacts", {
            "nonces": nonces,
            "block_hash": block_hash,
            "public_key": public_key,
            "seq_len": seq_len,
            "k_dim": k_dim,
        })
        
        if not result.get("skipped"):
            return result.get("artifacts", [])
        
        elapsed = time.time() - chunk_start_time
        if elapsed >= timeout_sec:
            raise RuntimeError(f"Timeout after {elapsed:.1f}s")
        
        await asyncio.sleep(POC_CHAT_BUSY_BACKOFF_SEC)


# =============================================================================
# Generation Loop
# =============================================================================

async def _generation_loop(
    engine_client,
    stop_event: asyncio.Event,
    callback_sender: Optional[CallbackSender],
    config: dict,
    stats: dict,
):
    nonce_iter = NonceIterator(
        node_id=config["node_id"],
        n_nodes=config["node_count"],
        group_id=config["group_id"],
        n_groups=config["n_groups"],
    )
    batch_size = config["batch_size"]
    
    start_time = time.time()
    stats["start_time"] = start_time
    stats["total_processed"] = 0
    last_report_time = start_time
    
    logger.info(f"PoC generation started (node {config['node_id']}/{config['node_count']}, group {config['group_id']}/{config['n_groups']})")
    skip_count = 0
    timeout_count = 0
    pending_nonces = None
    
    try:
        while not stop_event.is_set():
            nonces = pending_nonces if pending_nonces else nonce_iter.take(batch_size)
            
            try:
                result = await engine_client.poc_request(
                    "generate_artifacts",
                    {
                        "nonces": nonces,
                        "block_hash": config["block_hash"],
                        "public_key": config["public_key"],
                        "seq_len": config["seq_len"],
                        "k_dim": config["k_dim"],
                    },
                    timeout_ms=POC_RPC_TIMEOUT_MS
                )
                timeout_count = 0
            except TimeoutError:
                timeout_count += 1
                if timeout_count == 1 or timeout_count % 10 == 0:
                    logger.warning(f"PoC timed out (#{timeout_count}), engine busy")
                pending_nonces = nonces
                await asyncio.sleep(POC_CHAT_BUSY_BACKOFF_SEC * 2)
                continue
            
            if result.get("skipped"):
                skip_count += 1
                if skip_count % 100 == 1:
                    logger.debug(f"PoC yielding to chat (skip #{skip_count})")
                pending_nonces = nonces
                await asyncio.sleep(POC_CHAT_BUSY_BACKOFF_SEC)
                continue
            
            skip_count = 0
            pending_nonces = None
            artifacts = result.get("artifacts", [])
            
            if artifacts and callback_sender:
                artifact_objs = [Artifact(nonce=a["nonce"], vector_b64=a["vector_b64"]) for a in artifacts]
                callback_sender.add_artifacts(artifact_objs, {
                    "public_key": config["public_key"],
                    "block_hash": config["block_hash"],
                    "block_height": config["block_height"],
                    "node_id": config["node_id"],
                })
            
            stats["total_processed"] += len(nonces)
            
            current_time = time.time()
            if current_time - last_report_time >= 5.0:
                elapsed_min = (current_time - start_time) / 60
                rate = stats["total_processed"] / elapsed_min if elapsed_min > 0 else 0
                logger.info(f"Generated: {stats['total_processed']} nonces ({rate:.0f}/min)")
                last_report_time = current_time
            
    except asyncio.CancelledError:
        elapsed_min = (time.time() - start_time) / 60
        logger.info(f"PoC stopped: {stats['total_processed']} nonces in {elapsed_min:.2f}min")
    except Exception as e:
        logger.error(f"PoC generation crashed: {e}", exc_info=True)
        raise


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/init/generate")
async def init_generate(request: Request, body: PoCInitGenerateRequest) -> dict:
    logger.info(f"PoC /init/generate: {body.block_hash}, {body.block_height}, {body.public_key}, {body.node_id}, {body.node_count}, {body.group_id}, {body.n_groups}, {body.batch_size}, {body.params}, {body.url}")
    check_params_match(request, body.params)
    engine_client = await get_engine_client(request)
    
    app_id = id(request.app)
    
    if _is_generation_active(app_id):
        raise HTTPException(status_code=409, detail="Already generating")
    
    await _cancel_poc_tasks(app_id)
    
    config = {
        "block_hash": body.block_hash,
        "block_height": body.block_height,
        "public_key": body.public_key,
        "node_id": body.node_id,
        "node_count": body.node_count,
        "group_id": body.group_id,
        "n_groups": body.n_groups,
        "batch_size": body.batch_size,
        "seq_len": body.params.seq_len,
        "k_dim": body.params.k_dim,
    }
    
    stats = {"start_time": 0, "total_processed": 0}
    stop_event = asyncio.Event()
    
    callback_sender = None
    callback_task = None
    if body.url:
        callback_sender = CallbackSender(body.url, stop_event, body.params.k_dim)
        callback_task = asyncio.create_task(callback_sender.run())
    
    gen_task = asyncio.create_task(
        _generation_loop(engine_client, stop_event, callback_sender, config, stats)
    )
    
    _poc_tasks[app_id] = {
        "gen_task": gen_task,
        "callback_task": callback_task,
        "callback_sender": callback_sender,
        "stop_event": stop_event,
        "config": config,
        "stats": stats,
    }
    
    return {"status": "OK", "pow_status": {"status": "GENERATING"}}


@router.post("/generate")
async def generate(request: Request, body: PoCGenerateRequest) -> dict:
    logger.info(f"PoC /generate: {body.block_hash}, {body.block_height}, {body.public_key}, {body.node_id}, {body.node_count}, {body.nonces}, {body.params}, {body.batch_size}, {body.wait}, {body.url}, {body.validation}, {body.stat_test}")
    check_params_match(request, body.params)
    engine_client = await get_engine_client(request)
    
    app_id = id(request.app)
    
    if body.validation:
        validation_nonces = set(a.nonce for a in body.validation.artifacts)
        if validation_nonces != set(body.nonces):
            raise HTTPException(status_code=400, detail="validation.artifacts nonces must match nonces field")
    
    validation_map = {a.nonce: a.vector_b64 for a in body.validation.artifacts} if body.validation else None
    stat_test = body.stat_test or StatTestModel()
    
    if not body.wait:
        queue = get_queue()
        queue.set_generation_active_check(_is_generation_active)
        
        if queue.queued_nonces + len(body.nonces) > POC_MAX_QUEUED_NONCES:
            raise HTTPException(
                status_code=429,
                detail=f"Queue full: {queue.queued_nonces} nonces queued, limit is {POC_MAX_QUEUED_NONCES}"
            )
        
        job = GenerateJob(
            request_id=str(uuid.uuid4()),
            engine_client=engine_client,
            app_id=app_id,
            block_hash=body.block_hash,
            block_height=body.block_height,
            public_key=body.public_key,
            node_id=body.node_id,
            node_count=body.node_count,
            nonces=body.nonces,
            seq_len=body.params.seq_len,
            k_dim=body.params.k_dim,
            batch_size=body.batch_size,
            validation_artifacts=validation_map,
            stat_test_dist_threshold=stat_test.dist_threshold,
            stat_test_p_mismatch=stat_test.p_mismatch,
            stat_test_fraud_threshold=stat_test.fraud_threshold,
            callback_url=body.url,
        )
        
        request_id = await queue.enqueue(job)
        if request_id is None:
            raise HTTPException(
                status_code=429,
                detail=f"Queue full: {queue.queued_nonces} nonces queued, limit is {POC_MAX_QUEUED_NONCES}"
            )
        
        await queue.ensure_worker_running(engine_client, app_id)
        
        return {"status": "queued", "request_id": request_id, "queued_count": len(body.nonces)}
    
    while _is_generation_active(app_id):
        await asyncio.sleep(0.1)
    
    total_nonces = len(body.nonces)
    n_chunks = (total_nonces + body.batch_size - 1) // body.batch_size
    logger.info(f"PoC /generate: {total_nonces} nonces, batch_size={body.batch_size}, chunks={n_chunks}")
    
    start_time = time.time()
    computed_artifacts = []
    
    for i in range(0, total_nonces, body.batch_size):
        chunk = body.nonces[i:i + body.batch_size]
        chunk_idx = i // body.batch_size
        
        def check_cancelled():
            return False
        
        while _is_generation_active(app_id):
            await asyncio.sleep(0.1)
        
        try:
            artifacts = await _compute_artifacts_chunk(
                engine_client, chunk, body.block_hash, body.public_key,
                body.params.seq_len, body.params.k_dim,
                POC_GENERATE_CHUNK_TIMEOUT_SEC, check_cancelled
            )
            computed_artifacts.extend(artifacts)
            logger.debug(f"PoC /generate: chunk {chunk_idx+1}/{n_chunks} done ({len(chunk)} nonces)")
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
    
    elapsed = time.time() - start_time
    rate = total_nonces / elapsed if elapsed > 0 else 0
    logger.info(f"PoC /generate completed: {total_nonces} nonces in {elapsed:.2f}s ({rate:.0f}/s)")
    
    if not body.validation:
        return {
            "status": "completed",
            "request_id": str(uuid.uuid4()),
            "artifacts": computed_artifacts,
            "encoding": {"dtype": "f16", "k_dim": body.params.k_dim, "endian": "le"},
        }
    
    validation_result = run_validation(
        computed_artifacts,
        validation_map,
        len(body.nonces),
        stat_test.dist_threshold,
        stat_test.p_mismatch,
        stat_test.fraud_threshold,
    )
    
    return {
        "status": "completed",
        "request_id": str(uuid.uuid4()),
        **validation_result,
    }


@router.get("/generate/{request_id}")
async def get_generate_result(request: Request, request_id: str) -> dict:
    queue = get_queue()
    record = queue.get_result(request_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Request {request_id} not found")
    
    response = {"status": record.status, "request_id": request_id}
    
    if record.status == "completed" and record.result:
        response.update(record.result)
    elif record.status == "failed" and record.error:
        response["error"] = record.error
    
    return response


@router.get("/status")
async def get_status(request: Request) -> dict:
    return _get_api_status(id(request.app))


@router.post("/stop")
async def stop_round(request: Request) -> dict:
    app_id = id(request.app)
    
    await _cancel_poc_tasks(app_id)
    await clear_queue()
    
    return {"status": "OK", "pow_status": {"status": "STOPPED"}}
