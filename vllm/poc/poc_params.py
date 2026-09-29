# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field


@dataclass
class PoCParams:
    """Parameters for a single PoC nonce request."""

    block_hash: str
    public_key: str
    block_height: int
    nonce: int
    seq_len: int = 256
    k_dim: int = 12
    # Decode scheme: max_tokens decode steps after the prefill.
    poc_decode: bool = False
    max_tokens: int = 0
    # Validation: the prover's k_points_steps; each step is seeded from the
    # reference so both sides run the same forward.
    enforced_k_steps: list[int] | None = field(default=None, repr=False)
    # Collect per-step sphere slices.
    debug: bool = False
    # Seed reflections by (block_hash, nonce); prover and validator must match.
    per_nonce_reflection: bool = False

    def __post_init__(self):
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}")
        if self.k_dim <= 0:
            raise ValueError(f"k_dim must be positive, got {self.k_dim}")
        if self.max_tokens < 0:
            raise ValueError(f"max_tokens must be >= 0, got {self.max_tokens}")
