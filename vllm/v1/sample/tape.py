# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Public sampling tape: seeded Gumbel-max sampling a third party can replay.

A request with ``extra_args["gonka_tape"]`` samples output token t as the argmax
of ``logits/T + g(seed, t, token)`` over its min-p / top-k / top-p set. That is
ordinary sampling at T, but for a fixed seed the answer is fixed, so a validator
replaying the committed tokens scores each one against its own winner.

g is MurmurHash3_x86_32 of (seed_lo, seed_hi, t, token) under a protocol domain,
turned into Gumbel noise in fp32 by one kernel: the same integers on every GPU.
t is read from the batch counters, which count in-flight tokens.

Speculative decoding verifies a draft by equality with the tape winner, so the
output does not depend on the drafter and a validator need not know about it.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    DispatchSpec,
    TritonWarmupTensor,
    triton_kernel_dispatcher_with_warmup,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, tldevice, triton
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.torch_utils import async_tensor_h2d

if HAS_TRITON:
    from vllm.v1.sample.ops.topk_topp_triton import (
        _SPLIT_MAX_BATCH,
        apply_top_k_top_p_triton,
    )

TAPE_ARG = "gonka_tape"
TAG_ARG = "gonka_tape_tag"
LEN_ARG = "gonka_tape_len"
GREEDY_ARG = "gonka_tape_greedy"

# MurmurHash3 seed of this noise definition. Changing the noise needs a new one.
TAPE_DOMAIN = 0x67A9E502

_GREEDY_TEMPERATURE = 1e-5
_MASKED_GAP = 1e9
_MAX_TAGS = 512
_MAX_PENDING = 256
_BLOCK_SIZE = 1024

# tag -> {output position: (mismatch, gap)}
_records: dict[str, dict[int, tuple[bool, float]]] = {}
# Scores still on the GPU: reading them per step would stall async scheduling.
_pending: list[tuple[list[tuple[str, int]], torch.Tensor]] = []


@triton.jit
def _rotl32(x, n: tl.constexpr):
    return (x << n) | (x >> (32 - n))


@triton.jit
def _murmur_mix(h, k):
    k = k * 0xCC9E2D51
    k = _rotl32(k, 15)
    k = k * 0x1B873593
    h = h ^ k
    h = _rotl32(h, 13)
    return h * 5 + 0xE6546B64


@triton.jit
def _row_hash(seeds_ptr, positions_ptr, i, DOMAIN: tl.constexpr):
    lo = tl.load(seeds_ptr + 2 * i).to(tl.uint32, bitcast=True)
    hi = tl.load(seeds_ptr + 2 * i + 1).to(tl.uint32, bitcast=True)
    t = tl.load(positions_ptr + i).to(tl.uint32, bitcast=True)
    return _murmur_mix(_murmur_mix(_murmur_mix((lo ^ lo) ^ DOMAIN, lo), hi), t)


@triton.jit
def _gumbel(row_hash, tokens):
    h = _murmur_mix(row_hash, tokens.to(tl.uint32)) ^ 16
    h = h ^ (h >> 16)
    h = h * 0x85EBCA6B
    h = h ^ (h >> 13)
    h = h * 0xC2B2AE35
    h = h ^ (h >> 16)
    # (h + 1/2) * 2^-32 from the exact 16-bit halves.
    hi16 = (h >> 16).to(tl.int32)
    lo16 = (h & 0xFFFF).to(tl.int32)
    u = (
        hi16.to(tl.float32) * 1.52587890625e-05
        + (lo16.to(tl.float32) + 0.5) * 2.3283064365386963e-10
    )
    # -log1p(-u): a fixed series keeps the winning tail u -> 0 exact on any GPU.
    p = 1.0 / 8.0
    p = 1.0 / 7.0 + u * p
    p = 1.0 / 6.0 + u * p
    p = 1.0 / 5.0 + u * p
    p = 1.0 / 4.0 + u * p
    p = 1.0 / 3.0 + u * p
    p = 1.0 / 2.0 + u * p
    p = 1.0 + u * p
    direct = -tldevice.log(tl.maximum(1.0 - u, 5.960464477539063e-08))
    return -tldevice.log(tl.where(u < 0.25, u * p, direct))


@triton.jit
def _winner_kernel(
    logits_ptr,
    logits_stride,
    seeds_ptr,
    positions_ptr,
    best_ptr,
    index_ptr,
    num_blocks,
    vocab_size,
    DOMAIN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0)
    block = tl.program_id(1)
    tokens = block * BLOCK + tl.arange(0, BLOCK)
    logits = tl.load(
        logits_ptr + i * logits_stride + tokens,
        mask=tokens < vocab_size,
        other=float("-inf"),
    )
    noise = _gumbel(_row_hash(seeds_ptr, positions_ptr, i, DOMAIN), tokens)
    best, index = tl.max(logits + noise, axis=0, return_indices=True)
    tl.store(best_ptr + i * num_blocks + block, best)
    tl.store(index_ptr + i * num_blocks + block, block * BLOCK + index)


@triton.jit
def _noise_kernel(
    seeds_ptr, positions_ptr, tokens_ptr, noise_ptr, DOMAIN: tl.constexpr
):
    i = tl.program_id(0)
    tokens = tl.load(tokens_ptr + i)
    noise = _gumbel(_row_hash(seeds_ptr, positions_ptr, i, DOMAIN), tokens)
    tl.store(noise_ptr + i, noise)


def _winner_warmup_inputs() -> dict[str, object]:
    return dict(
        logits=TritonWarmupTensor(torch.float32),
        logits_stride=1,
        seeds=TritonWarmupTensor(torch.int32, shape=(2,)),
        positions=TritonWarmupTensor(torch.int32),
        best=TritonWarmupTensor(torch.float32),
        index=TritonWarmupTensor(torch.int64),
        num_blocks=1,
        vocab_size=1,
    )


@triton_kernel_dispatcher_with_warmup(
    kernel=_winner_kernel, warmup_inputs=_winner_warmup_inputs
)
def _launch_winners(
    logits: torch.Tensor,
    logits_stride: int,
    seeds: torch.Tensor,
    positions: torch.Tensor,
    best: torch.Tensor,
    index: torch.Tensor,
    num_blocks: int,
    vocab_size: int,
) -> DispatchSpec:
    return (logits.shape[0], num_blocks), dict(
        DOMAIN=TAPE_DOMAIN, BLOCK=_BLOCK_SIZE, enable_fp_fusion=False
    )


def _noise_warmup_inputs() -> dict[str, object]:
    return dict(
        seeds=TritonWarmupTensor(torch.int32, shape=(2,)),
        positions=TritonWarmupTensor(torch.int32),
        tokens=TritonWarmupTensor(torch.int64),
        noise=TritonWarmupTensor(torch.float32),
    )


@triton_kernel_dispatcher_with_warmup(
    kernel=_noise_kernel, warmup_inputs=_noise_warmup_inputs
)
def _launch_noise(
    seeds: torch.Tensor,
    positions: torch.Tensor,
    tokens: torch.Tensor,
    noise: torch.Tensor,
) -> DispatchSpec:
    return (tokens.shape[0],), dict(DOMAIN=TAPE_DOMAIN, enable_fp_fusion=False)


def register_tape_warmups() -> None:
    if HAS_TRITON and not current_platform.is_cpu():
        _launch_winners.register_warmup()
        _launch_noise.register_warmup()


def tape_winners(
    logits: torch.Tensor, seeds: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Winner token and score of each row; masked tokens are -inf."""
    num_rows, vocab_size = logits.shape
    num_blocks = triton.cdiv(vocab_size, _BLOCK_SIZE)
    best = logits.new_empty(num_rows, num_blocks, dtype=torch.float32)
    index = logits.new_empty(num_rows, num_blocks, dtype=torch.int64)
    _launch_winners(
        logits,
        logits.stride(0),
        seeds,
        positions,
        best,
        index,
        num_blocks,
        vocab_size,
    )
    block = best.argmax(dim=-1, keepdim=True)
    return index.gather(1, block).view(-1), best.gather(1, block).view(-1)


def tape_noise(
    seeds: torch.Tensor, positions: torch.Tensor, tokens: torch.Tensor
) -> torch.Tensor:
    """Noise of one token per row."""
    noise = torch.empty(tokens.shape[0], dtype=torch.float32, device=tokens.device)
    _launch_noise(seeds, positions, tokens, noise)
    return noise


def cut_logits(
    logits: torch.Tensor,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
    min_p: torch.Tensor | None,
) -> torch.Tensor:
    """Mask contiguous rows to their sampling sets in place: min-p, then vLLM's
    top-k and top-p. ``None`` means that no row uses the cut."""
    if min_p is not None:
        floor = logits.amax(dim=-1) + min_p.log()
        logits.masked_fill_(logits < floor.unsqueeze(1), float("-inf"))
    if top_k is None and top_p is None:
        return logits
    # Above this size top-p takes another kernel that rounds its fp32 mass in
    # another order: stay on one, so that a row's set hardly depends on its batch.
    for start in range(0, logits.shape[0], _SPLIT_MAX_BATCH):
        end = start + _SPLIT_MAX_BATCH
        apply_top_k_top_p_triton(
            logits[start:end],
            top_k[start:end] if top_k is not None else None,
            top_p[start:end] if top_p is not None else None,
        )
    return logits


@dataclass
class TapeRow:
    """What a request on the tape needs from its sampling params."""

    seed: int
    min_p: float
    temperature: float = 1.0
    committed: list[int] = field(default_factory=list)
    tag: str | None = None
    greedy: bool = False


def tape_row(params: Any) -> TapeRow | None:
    args = params.extra_args or {}
    if not args.get(TAPE_ARG):
        return None
    row = TapeRow(
        (params.seed or 0) & 0xFFFFFFFFFFFFFFFF, params.min_p, params.temperature
    )
    committed = params.enforced_token_ids
    if args.get(TAG_ARG) and committed:
        row.committed = committed[: args.get(LEN_ARG) or len(committed)]
        row.tag = str(args[TAG_ARG])
        row.greedy = bool(args.get(GREEDY_ARG))
    elif params.temperature < _GREEDY_TEMPERATURE:
        # Greedy sampling draws nothing.
        return None
    return row


class TapeBatch:
    """The rows of a batch on the tape; positions are refreshed every step.

    Nothing here waits for the GPU, a blocking copy included: async scheduling
    prepares the next step while the current one is still running.
    """

    def __init__(
        self,
        rows: Sequence[int],
        tape_rows: Sequence[TapeRow],
        device: torch.device,
        covers_random: bool,
    ):
        self.covers_random = covers_random
        self.device = device
        self.rows_np = np.array(rows, dtype=np.int64)
        self.rows = async_tensor_h2d(self.rows_np, device)
        # The bits of the seed: CUDA cannot index uint32 tensors.
        self.seeds = async_tensor_h2d(
            np.array(
                [(r.seed & 0xFFFFFFFF, r.seed >> 32) for r in tape_rows],
                dtype=np.uint32,
            ).astype(np.int32),
            device,
        )
        self.min_p = (
            async_tensor_h2d([r.min_p for r in tape_rows], device, torch.float32)
            if any(r.min_p > 0.0 for r in tape_rows)
            else None
        )
        self.replays = [(i, r) for i, r in enumerate(tape_rows) if r.tag is not None]
        self.positions_np = np.zeros(len(rows), dtype=np.int32)
        self.positions = async_tensor_h2d(self.positions_np, device)

    def set_positions(self, output_lengths: np.ndarray) -> None:
        """``output_lengths``: tokens each batch row has output, in-flight ones
        included, which is the index of the token it samples next."""
        self.positions_np = np.maximum(output_lengths[self.rows_np], 0).astype(np.int32)
        self.positions = async_tensor_h2d(self.positions_np, self.device)

    def advanced(self, num_draft_tokens: Sequence[int]) -> TapeBatch:
        """The batch at the bonus token, past each row's drafts."""
        shifted = copy.copy(self)
        drafts = np.asarray(num_draft_tokens, dtype=np.int32)[self.rows_np]
        shifted.positions_np = self.positions_np + drafts
        shifted.positions = async_tensor_h2d(shifted.positions_np, self.device)
        return shifted

    def select(self, remap: dict[int, int]) -> TapeBatch:
        """The batch after the sampler was narrowed to the rows of ``remap``."""
        selected = copy.copy(self)
        selected.rows_np = np.array([remap[r] for r in self.rows_np.tolist()])
        selected.rows = async_tensor_h2d(selected.rows_np, self.device)
        return selected

    def sample(
        self,
        logits: torch.Tensor,
        top_k: torch.Tensor | None,
        top_p: torch.Tensor | None,
        in_place: bool = False,
    ) -> torch.Tensor:
        """Winners over logits at temperature; replay rows are scored as well.

        Args:
            logits: The whole batch at temperature.
            top_k: The batch's top-k, ``None`` if no row uses it.
            top_p: The batch's top-p, ``None`` if no row uses it.
            in_place: Whether the logits may be left masked.

        Returns:
            The winner of each tape row.
        """
        live = [
            (i, r, int(self.positions_np[i]))
            for i, r in self.replays
            if self.positions_np[i] < len(r.committed)
        ]
        committed = self._read_committed(logits, live) if live else None
        if self.rows.shape[0] != logits.shape[0]:
            logits = logits[self.rows]
            top_k = top_k[self.rows] if top_k is not None else None
            top_p = top_p[self.rows] if top_p is not None else None
        elif not in_place:
            logits = logits.clone()
        winners, best = tape_winners(
            cut_logits(logits, top_k, top_p, self.min_p), self.seeds, self.positions
        )
        if committed is not None:
            self._score(live, committed, winners, best)
        return winners

    def verify(
        self,
        target_logits: torch.Tensor,
        num_draft_tokens: Sequence[int],
        temperature: torch.Tensor,
        top_k: torch.Tensor | None,
        top_p: torch.Tensor | None,
    ) -> torch.Tensor:
        """Target logits flattened over the drafts -> tape winner of each draft
        position, -1 at positions of rows not on the tape."""
        counts = np.asarray(num_draft_tokens, dtype=np.int64)
        starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
        per_row = counts[self.rows_np]
        owner = np.repeat(np.arange(self.rows_np.shape[0]), per_row)
        targets = torch.full(
            (target_logits.shape[0],), -1, dtype=torch.int64, device=self.device
        )
        if not owner.size:
            return targets
        offset = np.concatenate([np.arange(n) for n in per_row])
        tokens = np.repeat(starts[self.rows_np], per_row) + offset
        index = async_tensor_h2d(tokens, self.device)
        owners = async_tensor_h2d(owner, self.device)
        rows = self.rows[owners]
        winners, _ = tape_winners(
            cut_logits(
                target_logits[index] / temperature[rows].unsqueeze(1),
                top_k[rows] if top_k is not None else None,
                top_p[rows] if top_p is not None else None,
                self.min_p[owners] if self.min_p is not None else None,
            ),
            self.seeds[owners],
            async_tensor_h2d(
                (self.positions_np[owner] + offset).astype(np.int32), self.device
            ),
        )
        targets[index] = winners
        return targets

    def _read_committed(self, logits: torch.Tensor, live: list) -> tuple:
        """What a replay scores from the logits, read before they are masked."""
        at = async_tensor_h2d([i for i, _, _ in live], self.device, torch.int64)
        x = async_tensor_h2d(
            [r.committed[t] for _, r, t in live], self.device, torch.int64
        )
        rows = self.rows[at]
        top = logits[rows].max(dim=-1) if any(r.greedy for _, r, _ in live) else None
        return at, x, logits[rows, x], top

    def _score(
        self, live: list, committed: tuple, winners: torch.Tensor, best: torch.Tensor
    ) -> None:
        at, x, lx, top = committed
        noise = tape_noise(self.seeds[at], self.positions[at], x)
        # A committed token outside this model's set is scored by its uncut
        # score: boundary jitter gives no gap, a masked token _MASKED_GAP.
        z, sz, sx = winners[at], best[at], lx.double() + noise.double()
        if top is not None:
            greedy = async_tensor_h2d(
                [r.greedy for _, r, _ in live], self.device, torch.bool
            )
            z = torch.where(greedy, top.indices, z)
            sz = torch.where(greedy, top.values, sz)
            sx = torch.where(greedy, lx.double(), sx)
        # In logits as the model gave them: the jitter between two honest runs does
        # not grow as the temperature falls, but its score does.
        temperature = async_tensor_h2d(
            [r.temperature for _, r, _ in live], self.device, torch.float64
        )
        gap = torch.nan_to_num(
            (sz.double() - sx) * temperature, nan=0.0, posinf=_MASKED_GAP
        )
        stats = torch.stack([(x != z).double(), gap.clamp(min=0.0)], dim=1)
        _pending.append(([(r.tag, t) for _, r, t in live], stats))
        if len(_pending) >= _MAX_PENDING:
            with gpu_sync_allowed():
                _read_pending()


def _read_pending() -> None:
    for keys, stats in _pending:
        for (tag, t), (mismatch, gap) in zip(keys, stats.tolist()):
            _records.setdefault(tag, {})[t] = (bool(mismatch), gap)
    _pending.clear()
    while len(_records) > _MAX_TAGS:
        _records.pop(next(iter(_records)))


def take(tags: list[str]) -> dict[str, dict[str, Any]]:
    """Pop the per-response signals of finished replays."""
    _read_pending()
    out = {}
    for tag in tags:
        record = _records.pop(tag, None)
        if record is None:
            continue
        gaps = [gap for mismatch, gap in record.values() if mismatch]
        out[tag] = {
            "steps": len(record),
            "count": len(gaps),
            "max_gap": max(gaps, default=0.0),
            "sum_gap": sum(gaps),
        }
    return out
