"""PoC generate queue with bounded nonce cap and result store."""
import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from vllm.logger import init_logger
from .validation import run_validation
from .callbacks import send_oneshot_callback
from .data import DEFAULT_DIST_THRESHOLD, DEFAULT_P_MISMATCH, DEFAULT_FRAUD_THRESHOLD

logger = init_logger(__name__)

POC_GENERATE_CHUNK_TIMEOUT_SEC = float(os.environ.get("POC_GENERATE_CHUNK_TIMEOUT_SEC", "60"))
POC_CHAT_BUSY_BACKOFF_SEC = 0.05
POC_GENERATE_RESULT_TTL_SEC = float(os.environ.get("POC_GENERATE_RESULT_TTL_SEC", "300"))
POC_MAX_QUEUED_NONCES = int(os.environ.get("POC_MAX_QUEUED_NONCES", "100000"))


@dataclass
class GenerateJob:
    """A queued /generate request."""
    request_id: str
    engine_client: Any
    app_id: int
    block_hash: str
    block_height: int
    public_key: str
    node_id: int
    node_count: int
    nonces: List[int]
    seq_len: int
    k_dim: int
    batch_size: int
    validation_artifacts: Optional[Dict[int, str]] = None
    stat_test_dist_threshold: float = DEFAULT_DIST_THRESHOLD
    stat_test_p_mismatch: float = DEFAULT_P_MISMATCH
    stat_test_fraud_threshold: float = DEFAULT_FRAUD_THRESHOLD
    callback_url: Optional[str] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class GenerateResult:
    """Result record for a queued /generate request."""
    status: str  # "queued", "running", "completed", "failed", "cancelled"
    nonce_count: int = 0
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class GenerateQueue:
    """Bounded queue for /generate jobs with result tracking."""
    
    def __init__(self):
        self._queue: asyncio.Queue[GenerateJob] = asyncio.Queue()
        self._results: Dict[str, GenerateResult] = {}
        self._queued_nonces: int = 0
        self._lock: asyncio.Lock = asyncio.Lock()
        self._worker_task: Optional[asyncio.Task] = None
        self._stop_event: asyncio.Event = asyncio.Event()
        self._is_generation_active: Optional[Callable[[int], bool]] = None
    
    def set_generation_active_check(self, fn: Callable[[int], bool]):
        """Set callback to check if /init/generate is active."""
        self._is_generation_active = fn
    
    @property
    def queued_nonces(self) -> int:
        return self._queued_nonces
    
    async def enqueue(self, job: GenerateJob) -> Optional[str]:
        """Enqueue a job. Returns None if cap exceeded."""
        async with self._lock:
            new_total = self._queued_nonces + len(job.nonces)
            if new_total > POC_MAX_QUEUED_NONCES:
                return None
            
            self._queued_nonces = new_total
            self._results[job.request_id] = GenerateResult(
                status="queued",
                nonce_count=len(job.nonces)
            )
            await self._queue.put(job)
            return job.request_id
    
    def get_result(self, request_id: str) -> Optional[GenerateResult]:
        """Get result for a request_id."""
        return self._results.get(request_id)
    
    async def clear_all(self):
        """Clear queue and results."""
        async with self._lock:
            while not self._queue.empty():
                try:
                    job = self._queue.get_nowait()
                    if job.request_id in self._results:
                        self._results[job.request_id].status = "cancelled"
                        self._results[job.request_id].completed_at = time.time()
                except asyncio.QueueEmpty:
                    break
            
            self._queued_nonces = 0
            self._results.clear()
            self._stop_event.set()
    
    def cleanup_old_results(self):
        """Remove completed/failed results older than TTL."""
        now = time.time()
        expired = [
            rid for rid, rec in self._results.items()
            if rec.status in ("completed", "failed", "cancelled")
            and rec.completed_at
            and now - rec.completed_at > POC_GENERATE_RESULT_TTL_SEC
        ]
        for rid in expired:
            del self._results[rid]
    
    async def ensure_worker_running(self, engine_client, app_id: int):
        """Ensure the worker task is running."""
        if self._worker_task is None or self._worker_task.done():
            self._stop_event.clear()
            self._worker_task = asyncio.create_task(
                self._worker_loop(engine_client, app_id)
            )
    
    async def stop_worker(self):
        """Stop the worker task."""
        self._stop_event.set()
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
    
    async def _worker_loop(self, engine_client, app_id: int):
        """Background worker that processes queued jobs."""
        logger.info("Generate queue worker started")
        
        while not self._stop_event.is_set():
            try:
                try:
                    job = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                if job.request_id in self._results:
                    self._results[job.request_id].status = "running"
                
                try:
                    if self._is_generation_active:
                        while self._is_generation_active(job.app_id):
                            if self._stop_event.is_set():
                                break
                            await asyncio.sleep(0.1)
                    
                    if self._stop_event.is_set():
                        break
                    
                    result = await self._process_job(job)
                    
                    if job.request_id in self._results:
                        self._results[job.request_id].status = "completed"
                        self._results[job.request_id].completed_at = time.time()
                        self._results[job.request_id].result = result
                    
                    if job.callback_url:
                        await self._send_callback(job, result)
                    
                except Exception as e:
                    logger.error(f"Generate job {job.request_id} failed: {e}", exc_info=True)
                    if job.request_id in self._results:
                        self._results[job.request_id].status = "failed"
                        self._results[job.request_id].completed_at = time.time()
                        self._results[job.request_id].error = str(e)
                
                finally:
                    async with self._lock:
                        self._queued_nonces -= len(job.nonces)
                        self._queued_nonces = max(0, self._queued_nonces)
                
                self.cleanup_old_results()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Generate worker error: {e}", exc_info=True)
                await asyncio.sleep(1)
        
        logger.info("Generate queue worker stopped")
    
    async def _process_job(self, job: GenerateJob) -> Dict[str, Any]:
        """Process a single generate job."""
        total_nonces = len(job.nonces)
        n_chunks = (total_nonces + job.batch_size - 1) // job.batch_size
        logger.info(f"PoC queue job {job.request_id[:8]}: {total_nonces} nonces, batch_size={job.batch_size}, chunks={n_chunks}")
        
        start_time = time.time()
        computed_artifacts = []
        
        for i in range(0, total_nonces, job.batch_size):
            chunk = job.nonces[i:i + job.batch_size]
            chunk_idx = i // job.batch_size
            chunk_start_time = time.time()
            
            while True:
                if self._stop_event.is_set():
                    raise RuntimeError("Job cancelled")
                
                if self._is_generation_active and self._is_generation_active(job.app_id):
                    await asyncio.sleep(0.1)
                    continue
                
                result = await job.engine_client.poc_request("generate_artifacts", {
                    "nonces": chunk,
                    "block_hash": job.block_hash,
                    "public_key": job.public_key,
                    "seq_len": job.seq_len,
                    "k_dim": job.k_dim,
                })
                
                if not result.get("skipped"):
                    computed_artifacts.extend(result.get("artifacts", []))
                    logger.debug(f"PoC queue job {job.request_id[:8]}: chunk {chunk_idx+1}/{n_chunks} done ({len(chunk)} nonces)")
                    break
                
                elapsed = time.time() - chunk_start_time
                if elapsed >= POC_GENERATE_CHUNK_TIMEOUT_SEC:
                    raise RuntimeError(f"Timeout waiting for engine: chunk {chunk_idx}")
                
                await asyncio.sleep(POC_CHAT_BUSY_BACKOFF_SEC)
        
        elapsed = time.time() - start_time
        rate = total_nonces / elapsed if elapsed > 0 else 0
        logger.info(f"PoC queue job {job.request_id[:8]} completed: {total_nonces} nonces in {elapsed:.2f}s ({rate:.0f}/s)")
        
        if job.validation_artifacts is None:
            return {
                "status": "completed",
                "request_id": job.request_id,
                "artifacts": computed_artifacts,
                "encoding": {"dtype": "f16", "k_dim": job.k_dim, "endian": "le"},
            }
        
        validation_result = run_validation(
            computed_artifacts,
            job.validation_artifacts,
            len(job.nonces),
            job.stat_test_dist_threshold,
            job.stat_test_p_mismatch,
            job.stat_test_fraud_threshold,
        )
        
        return {
            "status": "completed",
            "request_id": job.request_id,
            **validation_result,
        }
    
    async def _send_callback(self, job: GenerateJob, result: Dict[str, Any]):
        """Send callback for completed job."""
        if job.validation_artifacts is None:
            payload = {
                "request_id": job.request_id,
                "block_hash": job.block_hash,
                "block_height": job.block_height,
                "public_key": job.public_key,
                "node_id": job.node_id,
                "artifacts": result.get("artifacts", []),
                "encoding": result.get("encoding", {}),
            }
            await send_oneshot_callback(job.callback_url, "generated", payload, self._stop_event)
        else:
            payload = {
                "request_id": job.request_id,
                "block_hash": job.block_hash,
                "block_height": job.block_height,
                "public_key": job.public_key,
                "node_id": job.node_id,
                "n_total": result.get("n_total", 0),
                "n_mismatch": result.get("n_mismatch", 0),
                "mismatch_nonces": result.get("mismatch_nonces", []),
                "p_value": result.get("p_value", 1.0),
                "fraud_detected": result.get("fraud_detected", False),
            }
            await send_oneshot_callback(job.callback_url, "validated", payload, self._stop_event)


_queue_instance: Optional[GenerateQueue] = None


def get_queue() -> GenerateQueue:
    """Get or create singleton queue instance."""
    global _queue_instance
    if _queue_instance is None:
        _queue_instance = GenerateQueue()
    return _queue_instance


async def clear_queue():
    """Clear the queue singleton."""
    global _queue_instance
    if _queue_instance:
        await _queue_instance.clear_all()
        await _queue_instance.stop_worker()
        _queue_instance = None
