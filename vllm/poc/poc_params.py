# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parameters for PoC (Proof of Compute) requests."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class PoCParams:
    """Parameters for a PoC request.
    
    Each PoC request represents a single nonce to be evaluated. The scheduler
    packs multiple PoC requests into batches based on token budget.
    
    Args:
        block_hash: Block hash for seeding the random generation.
        public_key: Public key for seeding the random generation.
        block_height: Block height for the PoC round.
        nonce: The nonce value to evaluate.
        r_target: Target distance threshold for valid proofs.
        seq_len: Sequence length for the generated embeddings.
        return_vectors: Whether to return the output vectors (for debugging).
    """
    block_hash: str
    public_key: str
    block_height: int
    nonce: int
    r_target: float = 1.5
    seq_len: int = 256
    return_vectors: bool = False

    def clone(self) -> "PoCParams":
        return PoCParams(
            block_hash=self.block_hash,
            public_key=self.public_key,
            block_height=self.block_height,
            nonce=self.nonce,
            r_target=self.r_target,
            seq_len=self.seq_len,
            return_vectors=self.return_vectors,
        )

    def __repr__(self) -> str:
        return (f"PoCParams(block_hash={self.block_hash!r}, "
                f"public_key={self.public_key!r}, "
                f"block_height={self.block_height}, "
                f"nonce={self.nonce}, "
                f"r_target={self.r_target}, "
                f"seq_len={self.seq_len})")


