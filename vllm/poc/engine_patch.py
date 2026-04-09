"""PoC Engine Patch for vLLM 0.15.1 V1 Engine.

This module patches the V1 AsyncLLM class to add poc_request support,
enabling PoC (Proof of Compute) artifact generation.

Chat Priority Gating:
    PoC has priority over chat. When PoC generation is active, the chat
    API endpoint (api_router.py) rejects new inference requests with 503.

    IMPORTANT: PoC's execute_poc_forward reuses KV cache blocks starting
    from block 0 (both as scratch for inputs_embeds and as the attention
    slot mapping).  If any inference request still has KV blocks allocated,
    PoC will overwrite them and permanently corrupt the model output.

    Therefore poc_request MUST wait for all in-flight requests to fully
    complete (drain) before issuing the collective_rpc.  The API-level
    gating ensures no *new* requests arrive, so the drain is bounded by
    the longest in-flight generation (max_tokens).

Usage:
    Import this module early in the application startup to apply the patch.
"""
import asyncio
from typing import Dict, Any, Optional, TYPE_CHECKING
from vllm.logger import init_logger

logger = init_logger(__name__)

_patched = False


async def poc_request(self, action: str, payload: dict, timeout_ms: int = 60000) -> dict:
    """Send a PoC (Proof of Compute) request to the engine.
    
    Only supports 'generate_artifacts' action. All PoC state (generation
    loop, nonce counter, stats) is managed in the API layer.
    
    Before issuing the GPU work this method waits for every in-flight
    inference request to complete.  This is required because
    execute_poc_forward writes into KV-cache blocks starting from block 0;
    if any request still holds those blocks the KV data is corrupted and
    the model produces garbage for the rest of its lifetime.
    
    The API-level chat gating (api_router.py) prevents new requests from
    arriving, so the drain is bounded by the longest in-flight generation.
    
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
    
    if not nonces:
        return {"artifacts": []}
    
    # Drain all in-flight inference before touching the GPU.
    # execute_poc_forward reuses KV-cache blocks from block 0, so any
    # request that still holds allocated blocks would get its KV data
    # destroyed, permanently corrupting model output.
    # The API-level gating (api_router.py) already blocks new requests,
    # so we only need to wait for existing ones to finish.
    output_processor = getattr(self, 'output_processor', None)
    if output_processor is not None and output_processor.has_unfinished_requests():
        n = output_processor.get_num_unfinished_requests()
        logger.info("PoC waiting for %d in-flight inference request(s) to drain", n)
        drain_start = asyncio.get_event_loop().time()
        while output_processor.has_unfinished_requests():
            await asyncio.sleep(0.1)
            elapsed = asyncio.get_event_loop().time() - drain_start
            if elapsed > timeout_ms / 1000.0:
                n = output_processor.get_num_unfinished_requests()
                logger.warning("PoC drain timed out after %.1fs with %d "
                               "request(s) still in-flight, skipping",
                               elapsed, n)
                return {"artifacts": [], "skipped": True}
        elapsed = asyncio.get_event_loop().time() - drain_start
        logger.info("Inference drained in %.1fs, proceeding with PoC", elapsed)
    
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
