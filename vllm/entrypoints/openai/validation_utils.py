import hashlib
from typing import Optional, Tuple


def generate_run_seed(user_seed: Optional[int], inference_id: str) -> str:
    if user_seed is None:
        return ""

    combined = f"{user_seed}{inference_id}"
    hash_digest = hashlib.sha256(combined.encode()).hexdigest()
    return hash_digest


def derive_seed_from_run_seed(run_seed: str) -> int:
    return int(run_seed[:16], 16) & 0x7FFFFFFFFFFFFFFF


def compute_derived_seed(
    user_seed: Optional[int],
    inference_id: str,
    run_seed: Optional[str] = None
) -> Tuple[Optional[int], str]:
    if run_seed:
        derived_seed = derive_seed_from_run_seed(run_seed)
        return derived_seed, run_seed

    if user_seed is None:
        return None, ""

    computed_run_seed = generate_run_seed(user_seed, inference_id)
    derived_seed = derive_seed_from_run_seed(computed_run_seed)

    return derived_seed, computed_run_seed
