import hashlib
from typing import Optional, Tuple


def generate_run_seed(user_seed: Optional[int], inference_id: str) -> str:
    if user_seed is None:
        return ""

    combined = f"{user_seed}{inference_id}"
    hash_digest = hashlib.sha256(combined.encode()).hexdigest()
    return hash_digest


def compute_derived_seed(user_seed: Optional[int], inference_id: str) -> Tuple[Optional[int], str]:
    if user_seed is None:
        return None, ""

    run_seed = generate_run_seed(user_seed, inference_id)
    derived_seed = int(run_seed[:16], 16) & 0x7FFFFFFFFFFFFFFF

    return derived_seed, run_seed
