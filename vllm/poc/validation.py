"""PoC artifact validation logic."""
from typing import Dict, List, Tuple

import numpy as np

from .data import decode_vector, fraud_test, DEFAULT_DIST_THRESHOLD, DEFAULT_P_MISMATCH, DEFAULT_FRAUD_THRESHOLD


def validate_artifacts(
    computed_artifacts: List[Dict],
    validation_map: Dict[int, str],
    dist_threshold: float = DEFAULT_DIST_THRESHOLD,
) -> Tuple[int, List[int]]:
    """Compare computed artifacts against validation artifacts.
    
    Args:
        computed_artifacts: List of {"nonce": int, "vector_b64": str}
        validation_map: Dict mapping nonce -> vector_b64
        dist_threshold: L2 distance threshold for mismatch
    
    Returns:
        (n_mismatch, mismatch_nonces)
    """
    n_mismatch = 0
    mismatch_nonces = []
    
    for artifact in computed_artifacts:
        nonce = artifact["nonce"]
        received_b64 = validation_map.get(nonce)
        if not received_b64:
            continue
        
        computed_vec = decode_vector(artifact["vector_b64"])
        received_vec = decode_vector(received_b64)
        distance = float(np.linalg.norm(computed_vec - received_vec))
        
        if distance > dist_threshold:
            n_mismatch += 1
            mismatch_nonces.append(nonce)
    
    return n_mismatch, mismatch_nonces


def run_validation(
    computed_artifacts: List[Dict],
    validation_map: Dict[int, str],
    n_total: int,
    dist_threshold: float = DEFAULT_DIST_THRESHOLD,
    p_mismatch: float = DEFAULT_P_MISMATCH,
    fraud_threshold: float = DEFAULT_FRAUD_THRESHOLD,
) -> Dict:
    """Run full validation with fraud test.
    
    Returns:
        Dict with n_total, n_mismatch, mismatch_nonces, p_value, fraud_detected
    """
    n_mismatch, mismatch_nonces = validate_artifacts(
        computed_artifacts, validation_map, dist_threshold
    )
    p_value, fraud_detected = fraud_test(n_mismatch, n_total, p_mismatch, fraud_threshold)
    
    return {
        "n_total": n_total,
        "n_mismatch": n_mismatch,
        "mismatch_nonces": mismatch_nonces,
        "p_value": p_value,
        "fraud_detected": fraud_detected,
    }
