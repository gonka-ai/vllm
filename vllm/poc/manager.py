"""PoC Manager - handles artifact generation for proof of compute.

NOTE: This class uses synchronous collective_rpc and was designed for V0
engine. In V1, poc_request is monkey-patched onto AsyncLLM by engine_patch.py
and uses async collective_rpc directly. This module is kept for V0
compatibility but is unused in the V1 code path.

This is a minimal, stateless manager that only provides the generate_artifacts
operation. All state (generation loop, nonce counter, stats) is managed in
the API layer (routes.py).
"""
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from .data import Artifact, encode_vector

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.executor.executor_base import ExecutorBase


class PoCManager:
    """Manages PoC artifact generation (stateless)."""
    
    def __init__(
        self,
        model_executor: "ExecutorBase",
        model_config,
        vllm_config: "VllmConfig",
    ):
        self.model_executor = model_executor
        self.model_config = model_config
        self.vllm_config = vllm_config
    
    def _run_forward(
        self,
        block_hash: str,
        public_key: str,
        nonces: List[int],
        seq_len: int,
        k_dim: int,
    ) -> Optional[Dict[str, Any]]:
        """Run forward pass via collective_rpc.
        
        Returns dict with 'nonces' and 'vectors' (FP16 numpy array).
        """
        from .poc_model_runner import execute_poc_forward
        
        results = self.model_executor.collective_rpc(
            execute_poc_forward,
            args=(
                block_hash,
                public_key,
                nonces,
                seq_len,
                self.model_config.get_hidden_size(),
                k_dim,
            ),
        )
        
        # Only the last PP rank returns a result
        return next((r for r in results if r is not None), None)
    
    def generate_artifacts(
        self,
        nonces: List[int],
        block_hash: str,
        public_key: str,
        seq_len: int,
        k_dim: int,
    ) -> List[Artifact]:
        """Generate artifacts for specific nonces.
        
        This is the only public API. The caller provides nonces explicitly;
        nonce progression logic lives in the API layer.
        """
        result = self._run_forward(
            block_hash,
            public_key,
            nonces,
            seq_len,
            k_dim,
        )
        
        if result is None:
            return []
        
        vectors = result["vectors"]  # FP16 numpy array
        artifacts = []
        for i, nonce in enumerate(result["nonces"]):
            vector_b64 = encode_vector(vectors[i])
            artifacts.append(Artifact(nonce=nonce, vector_b64=vector_b64))
        
        return artifacts
