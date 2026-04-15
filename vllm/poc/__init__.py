# Apply PoC engine patch for vLLM 0.15.1 V1 engine
from . import engine_patch

from .config import PoCConfig, PoCState
from .data import (
    PoCParams,
    Artifact,
    Encoding,
    ArtifactBatch,
    ValidationResult,
    encode_vector,
    decode_vector,
    is_mismatch,
    fraud_test,
    compare_artifacts,
)
from .manager import PoCManager
from .routes import router as poc_router
from .layer_hooks import LayerHouseholderHook

__all__ = [
    "PoCConfig",
    "PoCState",
    "PoCParams",
    "Artifact",
    "Encoding",
    "ArtifactBatch",
    "ValidationResult",
    "encode_vector",
    "decode_vector",
    "is_mismatch",
    "fraud_test",
    "compare_artifacts",
    "PoCManager",
    "poc_router",
    "LayerHouseholderHook",
]
