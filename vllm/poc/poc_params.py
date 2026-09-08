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
    # Decode-mode parameters (enabled by --poc-decode server flag)
    poc_decode: bool = False  # run decode steps after prefill
    max_tokens: int = 0  # number of decode steps (0 = prefill-only)
    # Validation mode: when set, this request tracks deviations from an
    # inference run instead of freely generating its own k-id sequence.
    # The list contains k_points_steps from the reference inference run
    # (index 0 = prefill, 1..N = decode steps).  At each step the validation
    # server computes its own k-id, compares against the reference, and uses
    # the reference k-id to seed the *next* decode embedding so that both
    # servers always run the same forward pass regardless of local deviations.
    enforced_k_steps: list[int] | None = field(default=None, repr=False)
    # Debug mode: collect per-step sphere indices and values for mismatch analysis.
    debug: bool = False
    # Per-nonce Householder seeding: reflection vectors are seeded by
    # (block_hash, nonce) instead of block_hash alone, so every nonce measures
    # with its own independent draw (block statistics average over n independent
    # instruments instead of sharing one). Forward-affecting: the prover and the
    # validator MUST use the same value for a nonce or their chains diverge —
    # which is why this is a per-request parameter, never an env toggle.
    per_nonce_reflection: bool = False

    @property
    def is_validation(self) -> bool:
        return self.enforced_k_steps is not None

    def clone(self) -> "PoCParams":
        return PoCParams(
            block_hash=self.block_hash,
            public_key=self.public_key,
            block_height=self.block_height,
            nonce=self.nonce,
            seq_len=self.seq_len,
            k_dim=self.k_dim,
            poc_decode=self.poc_decode,
            max_tokens=self.max_tokens,
            enforced_k_steps=(
                list(self.enforced_k_steps)
                if self.enforced_k_steps is not None
                else None
            ),
            debug=self.debug,
            per_nonce_reflection=self.per_nonce_reflection,
        )

    def __post_init__(self):
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}")
        if self.k_dim <= 0:
            raise ValueError(f"k_dim must be positive, got {self.k_dim}")
        if self.max_tokens < 0:
            raise ValueError(f"max_tokens must be >= 0, got {self.max_tokens}")
