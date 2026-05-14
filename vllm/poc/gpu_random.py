"""Deterministic GPU-based random generation for PoC.

Core primitives for generating reproducible random tensors seeded by
(block_hash, public_key, nonce). Used by the production inference pipeline.

OPTIMIZED: Serial Python loops replaced with batched GPU operations.
"""
import hashlib
import math
from typing import List

import torch


def _seed_from_string(seed_string: str) -> int:
    h = hashlib.sha256(seed_string.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _murmur3_32(keys: torch.Tensor, seed: int) -> torch.Tensor:
    """Murmur3 hash for int32 keys. Returns int64 to preserve full uint32 range."""
    c1, c2 = 0xCC9E2D51, 0x1B873593

    h = torch.full_like(keys, seed & 0xFFFFFFFF, dtype=torch.int64)
    k = keys.to(torch.int64) & 0xFFFFFFFF

    k = (k * c1) & 0xFFFFFFFF
    k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
    k = (k * c2) & 0xFFFFFFFF

    h = h ^ k
    h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
    h = (h * 5 + 0xE6546B64) & 0xFFFFFFFF

    h = h ^ (h >> 16)
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h = h ^ (h >> 13)
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return h


def _batched_murmur3_32(keys: torch.Tensor, seeds: torch.Tensor) -> torch.Tensor:
    """Batched Murmur3 hash with per-row seeds.

    Args:
        keys: [batch_size, n] int32 tensor
        seeds: [batch_size, 1] int64 tensor
    Returns:
        [batch_size, n] int64 tensor
    """
    c1, c2 = 0xCC9E2D51, 0x1B873593

    h = (seeds & 0xFFFFFFFF).expand_as(keys.to(torch.int64))
    k = keys.to(torch.int64) & 0xFFFFFFFF

    k = (k * c1) & 0xFFFFFFFF
    k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
    k = (k * c2) & 0xFFFFFFFF

    h = h ^ k
    h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
    h = (h * 5 + 0xE6546B64) & 0xFFFFFFFF

    h = h ^ (h >> 16)
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h = h ^ (h >> 13)
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return h


def _batched_normal(seeds: list, n: int, device: torch.device) -> torch.Tensor:
    """Generate batched normal random numbers for multiple seeds.

    Args:
        seeds: List of integer seeds
        n: Number of random numbers per seed
        device: Target device

    Returns:
        Tensor of shape [len(seeds), n]
    """
    batch_size = len(seeds)
    n_pairs = (n + 1) // 2
    total = n_pairs * 2

    indices = torch.arange(total, device=device, dtype=torch.int32).unsqueeze(0).expand(batch_size, -1)
    seed_tensor = torch.tensor(seeds, dtype=torch.int64, device=device).unsqueeze(1)

    h = _batched_murmur3_32(indices, seed_tensor)
    u = h.to(torch.float32) / 4294967296.0

    u1 = u[:, :n_pairs]
    u2 = u[:, n_pairs:]
    u1 = torch.clamp(u1, min=1e-10)

    z0 = torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * math.pi * u2)
    z1 = torch.sqrt(-2.0 * torch.log(u1)) * torch.sin(2.0 * math.pi * u2)
    return torch.cat([z0, z1], dim=1)[:, :n]


def _uniform(seed: int, n: int, device: torch.device) -> torch.Tensor:
    indices = torch.arange(n, device=device, dtype=torch.int32)
    hashes = _murmur3_32(indices, seed)
    return hashes.to(torch.float32) / 4294967296.0


def _normal(seed: int, n: int, device: torch.device) -> torch.Tensor:
    n_pairs = (n + 1) // 2
    u = _uniform(seed, n_pairs * 2, device)
    u1, u2 = u[:n_pairs], u[n_pairs:]
    u1 = torch.clamp(u1, min=1e-10)
    z0 = torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * math.pi * u2)
    z1 = torch.sqrt(-2.0 * torch.log(u1)) * torch.sin(2.0 * math.pi * u2)
    return torch.cat([z0, z1])[:n]


def generate_inputs(
    block_hash: str,
    public_key: str,
    nonces: List[int],
    dim: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Generate deterministic input embeddings for PoC."""
    batch_size = len(nonces)
    result = torch.empty(batch_size, seq_len, dim, device=device, dtype=dtype)
    for i, nonce in enumerate(nonces):
        seed_str = f"{block_hash}_{public_key}_nonce{nonce}"
        seed = _seed_from_string(seed_str)
        normal = _normal(seed, seq_len * dim, device)
        result[i] = normal.view(seq_len, dim).to(dtype)
    return result


def generate_inputs_concat_murmur(
    block_hash: str,
    public_key: str,
    nonces: List[int],
    dim: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Generate deterministic input embeddings using concat-murmur (stronger RNG).

    Uses all 256 bits of SHA256 by splitting into 8 × 32-bit sub-seeds.
    Each sub-seed generates one segment of length ceil(n/8) via the existing
    murmur3 pipeline; segments are concatenated.
    """
    batch_size = len(nonces)
    result = torch.empty(batch_size, seq_len, dim, device=device, dtype=dtype)
    n = seq_len * dim
    seg_len = (n + 7) // 8  # ceil(n/8); last segment may be shorter

    for i, nonce in enumerate(nonces):
        h = hashlib.sha256(
            f"{block_hash}_{public_key}_nonce{nonce}".encode()
        ).digest()
        sub_seeds = [int.from_bytes(h[j:j + 4], 'big') for j in range(0, 32, 4)]

        segments = [
            _normal(s, min(seg_len, n - k * seg_len), device)
            for k, s in enumerate(sub_seeds)
            if k * seg_len < n
        ]
        flat = torch.cat(segments)[:n]
        result[i] = flat.view(seq_len, dim).to(dtype)

    return result


def generate_target(
    block_hash: str,
    public_key: str,
    dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate deterministic target unit vector."""
    seed_str = f"{block_hash}_{public_key}_target"
    seed = _seed_from_string(seed_str)
    normal = _normal(seed, dim, device)
    target = normal.to(dtype)
    target = target / target.norm()
    return target


def generate_householder_vector(
    seed_str: str,
    dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate a single unit vector for Householder reflection."""
    seed = _seed_from_string(seed_str)
    v = _normal(seed, dim, device)
    return v / v.norm()


def apply_householder(
    x: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Apply Householder reflection: H @ x = x - 2*(v.x)*v"""
    dot = (x * v).sum(dim=-1, keepdim=True)
    return x - 2 * dot * v


def random_pick_indices(
    block_hash: str,
    public_key: str,
    nonces: List[int],
    dim: int,
    k: int,
    device: torch.device,
) -> torch.Tensor:
    """Pick k dimensions per nonce deterministically (vectorized)."""
    if k <= 0 or k > dim:
        raise ValueError(f"k must be in [1, dim], got k={k}, dim={dim}")

    batch_size = len(nonces)

    seeds = []
    for nonce in nonces:
        seeds.append(_seed_from_string(
            f"{block_hash}_{public_key}_nonce_{nonce}_pick_{k}"
        ))

    all_idx = torch.arange(dim, device=device, dtype=torch.int32).unsqueeze(0).expand(batch_size, -1)
    seed_tensor = torch.tensor(seeds, dtype=torch.int64, device=device).unsqueeze(1)
    scores = _batched_murmur3_32(all_idx, seed_tensor)

    _, chosen = torch.topk(-scores, k=k, largest=True, sorted=False, dim=1)
    return chosen.to(torch.int64)


def apply_haar_rotation(
    block_hash: str,
    public_key: str,
    nonces: List[int],
    x: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Apply Haar-random rotation via k-1 Householder reflections (vectorized)."""
    batch_size, k = x.shape
    if k <= 0:
        raise ValueError(f"k must be positive, got k={k}")

    y = x.clone()

    all_seeds_by_step = []
    for j in range(k - 1):
        step_seeds = []
        for nonce in nonces:
            step_seeds.append(_seed_from_string(
                f"{block_hash}_{public_key}_nonce_{nonce}_haar_hh_{k}_{j}"
            ))
        all_seeds_by_step.append(step_seeds)

    for j in range(k - 1):
        v_batch = _batched_normal(all_seeds_by_step[j], k, device)
        v_batch = v_batch / (v_batch.norm(dim=-1, keepdim=True) + 1e-30)
        v_batch = v_batch.to(y.dtype)

        dot = (y * v_batch).sum(dim=-1, keepdim=True)
        y = y - 2 * dot * v_batch

    return y
