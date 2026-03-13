"""PoC data types and helpers for artifact-based validation."""
import base64
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
from scipy.stats import binomtest


# Default validation parameters
DEFAULT_DIST_THRESHOLD = 0.02
DEFAULT_P_MISMATCH = 0.001
DEFAULT_FRAUD_THRESHOLD = 0.01


@dataclass
class PoCParams:
    """Strict params for PoC requests - exactly 3 fields."""
    model: str
    seq_len: int
    k_dim: int = 12


@dataclass
class Artifact:
    """Single nonce artifact with base64-encoded vector."""
    nonce: int
    vector_b64: str


@dataclass
class Encoding:
    """Metadata for vector encoding."""
    dtype: str = "f16"
    k_dim: int = 12
    endian: str = "le"


@dataclass
class ArtifactBatch:
    """Batch of artifacts for callback payloads."""
    public_key: str
    block_hash: str
    block_height: int
    node_id: int
    artifacts: List[Artifact]
    encoding: Encoding


@dataclass
class ValidationResult:
    """Result of artifact validation."""
    public_key: str
    block_hash: str
    block_height: int
    node_id: int
    nonces: List[int]
    n_total: int
    n_mismatch: int
    mismatch_nonces: List[int]
    fraud_threshold: float = DEFAULT_FRAUD_THRESHOLD
    p_value: Optional[float] = None
    fraud_detected: Optional[bool] = None


def encode_vector(vector: np.ndarray) -> str:
    """Encode FP32 vector to base64 FP16 little-endian."""
    f16 = vector.astype('<f2')  # '<f2' = little-endian float16
    return base64.b64encode(f16.tobytes()).decode('ascii')


def decode_vector(b64: str) -> np.ndarray:
    """Decode base64 FP16 little-endian to FP32."""
    data = base64.b64decode(b64)
    f16 = np.frombuffer(data, dtype='<f2')
    return f16.astype(np.float32)


def is_mismatch(
    computed_vector: np.ndarray,
    received_b64: str,
    dist_threshold: float = DEFAULT_DIST_THRESHOLD,
) -> bool:
    """Check if vectors differ beyond threshold."""
    received = decode_vector(received_b64)
    if not np.all(np.isfinite(received)):
        return True
    distance = float(np.linalg.norm(computed_vector - received))
    return distance > dist_threshold


def fraud_test(
    n_mismatch: int,
    n_total: int,
    p_mismatch: float = DEFAULT_P_MISMATCH,
    fraud_threshold: float = DEFAULT_FRAUD_THRESHOLD,
) -> Tuple[float, bool]:
    """
    Run binomial test for fraud detection.
    
    Args:
        n_mismatch: Number of nonces where vectors differ beyond threshold
        n_total: Total nonces checked
        p_mismatch: Expected mismatch rate for honest nodes (baseline)
        fraud_threshold: p-value below which fraud is detected
    
    Returns:
        (p_value, fraud_detected)
    """
    if n_total == 0:
        return 1.0, False

    result = binomtest(
        k=n_mismatch,
        n=n_total,
        p=p_mismatch,
        alternative='greater'
    )
    p_value = float(result.pvalue)
    fraud_detected = p_value < fraud_threshold
    return p_value, fraud_detected


def compare_artifacts(
    computed_vectors: List[np.ndarray],
    received_artifacts: List[Artifact],
    dist_threshold: float = DEFAULT_DIST_THRESHOLD,
) -> Tuple[int, List[int]]:
    """
    Compare computed vectors against received artifacts.
    
    Args:
        computed_vectors: List of computed FP32 vectors
        received_artifacts: List of received Artifact objects
        dist_threshold: L2 distance threshold for mismatch
    
    Returns:
        (n_mismatch, mismatch_nonces)
    """
    n_mismatch = 0
    mismatch_nonces = []
    
    for vec, artifact in zip(computed_vectors, received_artifacts):
        if is_mismatch(vec, artifact.vector_b64, dist_threshold):
            n_mismatch += 1
            mismatch_nonces.append(artifact.nonce)
    
    return n_mismatch, mismatch_nonces
