"""PoC Engine Patch for vLLM 0.15.1 V1 Engine.

This module patches the V1 AsyncLLM class to add poc_request support,
enabling PoC (Proof of Compute) artifact generation.

PoC Priority:
    PoC has priority over inference. When PoC generation is active, the
    chat and completion API endpoints reject new requests with 503.

    IMPORTANT: PoC's execute_poc_forward reuses KV cache blocks starting
    from block 0 (both as scratch for inputs_embeds and as the attention
    slot mapping). If any inference request still has KV blocks allocated,
    PoC will overwrite them and permanently corrupt the model output.

    Therefore poc_request aborts all in-flight inference requests before
    issuing collective_rpc. The API-level 503 gating prevents new
    requests from arriving while PoC generation is active.

Usage:
    Import this module early in the application startup to apply the patch.
"""
import asyncio
import math
import time
from vllm.logger import init_logger

logger = init_logger(__name__)

POC_BORROW_TIMEOUT_MS = 10_000

_patched = False


async def poc_request(self, action: str, payload: dict, timeout_ms: int = 60000,
                      borrowed_block_ids: list | None = None) -> dict:
    """Send a PoC (Proof of Compute) request to the engine.
    
    Only supports 'generate_artifacts' action. All PoC state (generation
    loop, nonce counter, stats) is managed in the API layer.
    
    Before issuing the GPU work this method aborts all in-flight inference
    requests.  This is required because execute_poc_forward writes into
    KV-cache blocks starting from block 0; if any request still holds
    those blocks the KV data is corrupted and the model produces garbage
    for the rest of its lifetime.
    
    The API-level 503 gating (chat and completion api_router.py) prevents
    new requests from arriving while PoC is active.
    
    Args:
        action: The PoC action to perform (only 'generate_artifacts' supported)
        payload: Dict containing nonces, block_hash, public_key, seq_len, k_dim
        timeout_ms: Timeout in milliseconds for the RPC call
        
    Returns:
        Dict with 'artifacts' list and optionally 'skipped' boolean
        
    Raises:
        TimeoutError: If engine doesn't respond within timeout
    """
    if action != "generate_artifacts":
        raise ValueError(f"Unknown PoC action: {action}")
    
    # Import PoC modules here to avoid circular imports
    from vllm.poc.poc_model_runner import execute_poc_forward
    from vllm.poc.data import encode_vector
    
    nonces = payload.get("nonces", [])
    block_hash = payload.get("block_hash", "")
    public_key = payload.get("public_key", "")
    seq_len = payload.get("seq_len", 256)
    k_dim = payload.get("k_dim", 12)
    poc_stronger_rng = payload.get("poc_stronger_rng", False)
    
    if not nonces:
        return {"artifacts": []}
    
    # Either the caller pre-reserved disjoint KV blocks (validation, via
    # /generate) — reuse them, never abort — or there are none, so we abort
    # in-flight inference and use the low blocks (/init/generate and queue).
    if borrowed_block_ids is None:
        await _abort_inflight(self)
    
    # Get model config for hidden_size
    # V1 engine stores config differently
    try:
        vllm_config = self.vllm_config
        hidden_size = vllm_config.model_config.get_hidden_size()
    except AttributeError:
        # Fallback - try to get from model config
        try:
            hidden_size = self.model_config.get_hidden_size()
        except Exception:
            # Default for Qwen models
            hidden_size = 8192
            logger.warning(f"Could not get hidden_size from config, using default: {hidden_size}")
    
    try:
        # Use collective_rpc to execute PoC forward on all workers
        timeout_sec = timeout_ms / 1000.0
        results = await self.collective_rpc(
            execute_poc_forward,
            timeout=timeout_sec,
            args=(
                block_hash,
                public_key,
                nonces,
                seq_len,
                hidden_size,
                k_dim,
                poc_stronger_rng,
                borrowed_block_ids,
            ),
        )
        
        # Only the last PP rank returns a result
        result = next((r for r in results if r is not None), None)
        
        if result is None:
            return {"artifacts": [], "skipped": True}
        
        # Convert result to artifact format
        vectors = result.get("vectors")  # FP16 numpy array
        result_nonces = result.get("nonces", nonces)
        
        artifacts = []
        for i, nonce in enumerate(result_nonces):
            vector_b64 = encode_vector(vectors[i])
            artifacts.append({"nonce": nonce, "vector_b64": vector_b64})
        
        return {"artifacts": artifacts}
        
    except asyncio.TimeoutError:
        logger.warning(f"PoC request timed out after {timeout_ms}ms")
        raise TimeoutError(f"PoC request timed out after {timeout_ms}ms")
    except Exception as e:
        logger.error(f"PoC request failed: {e}")
        return {"artifacts": [], "skipped": True}

async def _abort_inflight(self) -> None:
    """Abort in-flight inference so the PoC forward can reuse its KV blocks."""
    output_processor = getattr(self, "output_processor", None)
    if output_processor is None or not output_processor.has_unfinished_requests():
        return
    request_ids = list(output_processor.request_states.keys())
    if request_ids:
        logger.info("PoC aborting %d in-flight inference request(s)", len(request_ids))
        await self.abort(request_ids, internal=True)
        await asyncio.sleep(0.05)

async def _borrow_blocks(self, num_nonces: int, seq_len: int, timeout_ms: int):
    """Borrow disjoint KV blocks for a validation forward, waiting until the
    pool can spare them. Returns block ids, or None on timeout."""
    try:
        block_size = self.vllm_config.cache_config.block_size
    except Exception:
        block_size = 16
    needed = num_nonces * math.ceil(seq_len / block_size)
    deadline = time.monotonic() + timeout_ms / 1000.0
    while True:
        try:
            block_ids = await self.borrow_poc_blocks(needed)
        except Exception as e:
            logger.warning("PoC borrow failed: %s", e)
            block_ids = None
        if block_ids is not None:
            return block_ids
        if time.monotonic() > deadline:
            logger.warning("PoC could not borrow %d blocks within %dms", needed, timeout_ms)
            return None
        await asyncio.sleep(0.05)

async def borrow_poc_blocks(self, num_blocks: int):
    """Frontend-side wrapper: ask EngineCore (where the BlockPool lives) to
    borrow `num_blocks` free KV blocks. Returns block ids or None."""
    return await self.engine_core.call_utility_async("borrow_poc_blocks", num_blocks)

async def return_poc_blocks(self, block_ids) -> None:
    """Frontend-side wrapper: return previously borrowed KV blocks."""
    await self.engine_core.call_utility_async("return_poc_blocks", block_ids)

async def reserve_poc_blocks(self, num_nonces: int, seq_len: int,
                             timeout_ms: int = POC_BORROW_TIMEOUT_MS):
    """Reserve disjoint KV blocks for a PoC validation: wait up to timeout_ms
    for the pool, and if it stays full, abort in-flight inference once to free
    blocks and retry. Returns block ids (caller MUST return_poc_blocks) or None.

    Reserve once per request/queue and reuse across its chunks — cheaper than
    per-chunk borrow and avoids a mid-validation borrow failure."""
    block_ids = await _borrow_blocks(self, num_nonces, seq_len, timeout_ms)
    if block_ids is None:
        logger.warning("PoC reserve timed out; aborting inference to free KV blocks")
        await _abort_inflight(self)
        block_ids = await _borrow_blocks(self, num_nonces, seq_len, timeout_ms)
    return block_ids

def apply_patch():
    """Apply the PoC patch to vLLM V1 AsyncLLM class."""
    global _patched
    
    if _patched:
        logger.debug("PoC engine patch already applied")
        return
    
    try:
        from vllm.v1.engine.async_llm import AsyncLLM
        
        # Add poc_request method to AsyncLLM
        AsyncLLM.poc_request = poc_request
        # On-demand KV block borrow/return (routed to EngineCore via UTILITY RPC)
        AsyncLLM.borrow_poc_blocks = borrow_poc_blocks
        AsyncLLM.return_poc_blocks = return_poc_blocks
        AsyncLLM.reserve_poc_blocks = reserve_poc_blocks

        _patched = True
        logger.info("PoC engine patch applied successfully to AsyncLLM (V1)")
        
    except ImportError as e:
        logger.warning(f"Could not import V1 AsyncLLM, trying V0: {e}")
        
        try:
            from vllm.engine.async_llm_engine import AsyncLLMEngine
            
            # For V0 engine, the implementation is slightly different
            AsyncLLMEngine.poc_request = poc_request
            
            _patched = True
            logger.info("PoC engine patch applied successfully to AsyncLLMEngine (V0)")
            
        except ImportError as e2:
            logger.error(f"Could not import any LLM engine: {e2}")
            raise


# Auto-apply patch when module is imported
apply_patch()
