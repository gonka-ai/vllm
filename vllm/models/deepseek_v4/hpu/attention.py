# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged caches of the DeepSeek-V4 attention on a plain torch device.

A layer keeps what a token contributes in caches addressed by position: its
window KV and the inputs of the compressors, one entry per token, and the
compressed KV and indexer key of a block, one entry per ``ratio`` tokens. A
block table then resolves any of them, a cached prefix is reusable as it is,
and no state lives outside the caches.

The per-token caches are sliding windows: the engine frees a block once the
window has left it. As in the CUDA model each has a block of its own, a few
tokens, so that its bytes in a layer come near those of a block of compressed
KV; the engine then cuts all of them from pages of one size and a prefix is
cached by the engine's block.
"""

import os
from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheLayout,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)

RATIOS = (4, 128)
# On the card the math of a step runs as compiled graphs of static shapes
# (model._graph), so a step is padded to one of a few sizes.
GRAPHS = os.environ.get("VLLM_HPU_DEEPSEEK_V4_GRAPHS", "1") == "1"
# The least candidate blocks a graph is built for.
_MIN_CANDIDATES = 64
# Bytes of a block of a per-token cache in one layer.
_WINDOWED_BLOCK_BYTES = 64 << 10


def bucket(n: int) -> int:
    """The smallest of 1, 2, 3, 4, 6, 8, 12, 16, 24, ... not below n."""
    power = 1 << (n - 1).bit_length()
    return power * 3 // 4 if power * 3 // 4 >= n else power


@dataclass
class BlockEnds:
    """The tokens of the step that close a compression block of one ratio, as
    entries of a per-token cache."""

    # [ends] position of the block's first token.
    starts: torch.Tensor
    # [ends * ratio] token entry of every token of the block.
    rows: torch.Tensor
    # [ends * ratio] token entry of every token of the previous block.
    previous_rows: torch.Tensor
    # [ends, 1, 1] 0, or -inf where no previous block exists.
    previous_bias: torch.Tensor


@dataclass
class Candidates:
    """The compressed blocks of every request of the step."""

    # [reqs, width] block entry of every block of a request, in order.
    states: torch.Tensor
    # [tokens] blocks closed at or before the token.
    counts: torch.Tensor


@dataclass
class DeepseekV4HPUMetadata:
    """The indices of a step into the caches of one shape, built on the host
    and resolved through the block table on the device. The device is never
    read back. Only what those caches are read by is built.

    A padded step has ``padded_tokens`` rows: the rows after ``num_tokens``
    attend to nothing and, as the padded block ends do, read and write entry
    0 of the null block."""

    num_tokens: int
    padded_tokens: int
    # [tokens] request of every token of the step.
    requests: torch.Tensor
    # A per-token cache: [tokens] the entry of every token.
    rows: torch.Tensor | None
    # The window KV: [tokens, window] entries of the window; 0 or -inf.
    window_rows: torch.Tensor | None
    window_bias: torch.Tensor | None
    # The states of a compressor, by ratio: [tokens] position within the
    # block, and the blocks the step closes.
    offsets: dict[int, torch.Tensor]
    ends: dict[int, BlockEnds | None]
    # A cache of compressed blocks, by ratio: [ends] the entry every closed
    # block compresses into, and the blocks a token may attend to.
    targets: dict[int, torch.Tensor | None]
    candidates: dict[int, Candidates | None]


class DeepseekV4HPUImpl(AttentionImpl):
    """The layer runs the attention itself; the backend only owns the caches."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class DeepseekV4HPUBackend(AttentionBackend):
    forward_includes_kv_cache_update: bool = True
    # A layer writes its caches and reads them back in one forward, and the
    # engine's KV cache groups alias the same bytes: a sequence cannot lie
    # on the same blocks in two groups. The PoC plugin, which lays out blocks
    # of its own, reads this.
    exclusive_kv_blocks: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16, torch.float32]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto"]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # Block-outermost: the engine then packs caches of different shapes
        # and lifetimes into one block and keeps the spec of every cache.
        return (KVCacheLayout.BLNHC,)

    @staticmethod
    def get_name() -> str:
        return "HPU_V4_MLA_SPARSE"

    @staticmethod
    def get_impl_cls() -> type[DeepseekV4HPUImpl]:
        return DeepseekV4HPUImpl

    @staticmethod
    def get_builder_cls() -> type["DeepseekV4HPUMetadataBuilder"]:
        return DeepseekV4HPUMetadataBuilder


class DeepseekV4HPUCache(nn.Module, AttentionLayerBase):
    """One paged cache of a layer, ``width`` values per entry.

    With ``window`` it holds an entry per token, of which only the last that
    many of a request are ever read: the window KV, or with ``ratio`` the
    states of a compressor. Without, it holds an entry per ``ratio`` tokens,
    the compressed blocks, all kept."""

    def __init__(
        self,
        prefix: str,
        width: int,
        dtype: torch.dtype,
        ratio: int | None = None,
        window: int | None = None,
    ) -> None:
        super().__init__()
        assert ratio is not None or window is not None
        self.prefix = prefix
        self.width = width
        self.dtype = dtype
        self.ratio = ratio
        self.window = window
        self.tokens_per_entry = 1 if window is not None or ratio is None else ratio
        # Bytes a block is padded to a multiple of, see align_rows.
        self.alignment: int | None = None
        self.kv_cache = torch.tensor([])
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return DeepseekV4HPUBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        block_size = vllm_config.cache_config.block_size
        if block_size % max(RATIOS):
            raise ValueError(
                f"DeepSeek-V4 needs a KV cache block of a multiple of {max(RATIOS)} "
                f"tokens, got --block-size {block_size}."
            )
        if self.window is not None:
            # A power of two up to 128, so that it divides the engine's block.
            tokens = max(_WINDOWED_BLOCK_BYTES // (self.width * self.dtype.itemsize), 1)
            tokens = min(1 << (tokens.bit_length() - 1), max(RATIOS))
            return SlidingWindowMLASpec(
                block_size=tokens,
                num_kv_heads=1,
                head_size=self.width,
                dtype=self.dtype,
                sliding_window=self.window,
                alignment=self.alignment,
            )
        return MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=self.width,
            dtype=self.dtype,
            tokens_per_state=self.tokens_per_entry,
            alignment=self.alignment,
        )

    @property
    def kv_cache(self) -> torch.Tensor:
        return self._kv_cache

    @kv_cache.setter
    def kv_cache(self, cache: torch.Tensor) -> None:
        """Takes [blocks, 1, entries per block, width] over the engine's buffer.

        The blocks of a cache need not be adjacent there, so the cache is
        addressed as rows of the whole buffer from its first entry on: entry
        ``n`` of block ``b`` is row ``b * step + n``."""
        self._kv_cache = cache
        self._rows = None
        self.step = 0
        if cache.dim() != 4:
            return
        blocks, heads, per_block, width = cache.shape
        stride = cache.stride()
        if (
            (heads, width, cache.dtype) != (1, self.width, self.dtype)
            or stride[2:] != (width, 1)
            or stride[0] % width
        ):
            raise ValueError(
                f"{self.prefix}: a cache of {self.width} {self.dtype} values per "
                f"entry got a view {tuple(cache.shape)} {cache.dtype} with strides "
                f"{stride}."
            )
        self.step = stride[0] // width
        self._rows = torch.as_strided(
            cache,
            ((blocks - 1) * self.step + per_block, width),
            (width, 1),
            cache.storage_offset(),
        )

    def entries(self) -> torch.Tensor:
        """The rows the cache is addressed by, [rows, width]."""
        assert self._rows is not None, f"{self.prefix} has no KV cache bound"
        return self._rows


def align_rows(caches: list[DeepseekV4HPUCache]) -> None:
    """Pads the block of every cache to the widest entry among them.

    The engine lays the blocks of all caches on one grid of bytes, and a cache
    is addressed as rows of that buffer: a block must be a whole number of
    entries of every cache."""
    widest = max(c.width * c.dtype.itemsize for c in caches)
    for cache in caches:
        assert widest % (cache.width * cache.dtype.itemsize) == 0
        cache.alignment = widest


class DeepseekV4HPUMetadataBuilder(AttentionMetadataBuilder[DeepseekV4HPUMetadata]):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = kv_cache_spec.block_size
        self.pad = GRAPHS and device.type == "hpu"
        self.max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        # The caches of a builder share a shape, hence the rows of a block.
        context = vllm_config.compilation_config.static_forward_context
        caches: list[DeepseekV4HPUCache] = [context[name] for name in layer_names]
        self.cache = caches[0]
        self.window = None
        if any(c.ratio is None for c in caches):
            self.window = vllm_config.model_config.hf_config.sliding_window
        ratios = [(c.ratio, c.window is not None) for c in caches if c.ratio]
        self.token_ratios = tuple(sorted({r for r, windowed in ratios if windowed}))
        self.block_ratios = tuple(sorted({r for r, windowed in ratios if not windowed}))

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV4HPUMetadata:
        cm = common_attn_metadata
        # A KV cache group holds caches of several shapes and the runner builds
        # once per shape; shapes read alike with the same rows per block share
        # the result.
        built = getattr(cm, "_deepseek_v4_hpu", None)
        if built is None:
            built = {}
            cm._deepseek_v4_hpu = built  # type: ignore[attr-defined]
        step = self.cache.step
        key = (step, self.window, self.token_ratios, self.block_ratios)
        if key not in built:
            num_reqs = cm.num_reqs
            query_start = cm.query_start_loc_cpu[: num_reqs + 1].to(torch.int64)
            seq_lens = cm.seq_lens_cpu_upper_bound
            if seq_lens is None:
                seq_lens = cm.seq_lens.cpu()
            padded = None
            if self.pad:
                # No further than the rows the engine and the PoC plugin hold.
                num_tokens = int(query_start[-1])
                padded = min(bucket(num_tokens), max(self.max_tokens, num_tokens))
            built[key] = build_metadata(
                query_start,
                seq_lens[:num_reqs].to(torch.int64),
                cm.block_table_tensor[:num_reqs],
                self.block_size,
                self.window,
                step,
                padded,
                self.token_ratios,
                self.block_ratios,
            )
        return built[key]


def build_metadata(
    query_start: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    window: int | None,
    step: int | None = None,
    padded_tokens: int | None = None,
    token_ratios: tuple[int, ...] = RATIOS,
    block_ratios: tuple[int, ...] = RATIOS,
) -> DeepseekV4HPUMetadata:
    """The indices of a step, as rows of the caches of one shape.

    Args:
        query_start: [reqs + 1] first token of every request, on the host.
        seq_lens: [reqs] length of every request after the step, on the host.
        block_table: [reqs, blocks per request], on the device.
        block_size: Tokens of a block of these caches.
        window: The attention window, if a window KV is among the caches.
        step: Rows from a block of the cache to the next; by default the
            entries of a block, that is blocks stored back to back.
        padded_tokens: Pads the step to that many tokens, and the block ends
            and the candidate blocks to a ``bucket`` each.
        token_ratios: The compressors whose per-token states are among the
            caches.
        block_ratios: The compressors whose blocks are among the caches.
    """
    device = block_table.device
    num_reqs = seq_lens.shape[0]
    width = block_table.shape[1]
    # One entry more, the null block: where the padding points.
    table = torch.cat([block_table.reshape(-1), block_table.new_zeros(1)])
    null = num_reqs * width

    def entries_of(
        reqs: torch.Tensor,
        index: torch.Tensor,
        per_block: int,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cache rows of the index-th entries of requests, all on the host;
        entry 0 of the null block where ``valid`` is False."""
        index = index.clamp(min=0)
        slots, offsets = reqs * width + index // per_block, index % per_block
        if valid is not None:
            slots = slots.masked_fill(~valid, null)
            offsets = offsets.masked_fill(~valid, 0)
        blocks = table.index_select(0, slots.reshape(-1).to(device))
        return blocks * (step or per_block) + offsets.reshape(-1).to(device)

    query_lens = query_start[1:] - query_start[:-1]
    num_tokens = int(query_start[-1])
    padded = max(padded_tokens or 0, num_tokens)
    reqs = torch.repeat_interleave(torch.arange(num_reqs), query_lens)
    positions = (seq_lens - query_lens)[reqs] + (
        torch.arange(num_tokens) - query_start[:-1][reqs]
    )
    real = None
    if padded > num_tokens:
        reqs = F.pad(reqs, (0, padded - num_tokens))
        positions = F.pad(positions, (0, padded - num_tokens))
        real = torch.arange(padded) < num_tokens

    rows = window_rows = window_bias = None
    if window is not None or token_ratios:
        rows = entries_of(reqs, positions, block_size, real)
    if window is not None:
        in_window = positions[:, None] - (window - 1) + torch.arange(window)
        unseen = in_window < 0
        window_real = None
        if real is not None:
            window_real = real[:, None].expand(-1, window)
            unseen = unseen | ~window_real
        window_rows = entries_of(
            reqs[:, None].expand(-1, window), in_window, block_size, window_real
        ).view(padded, window)
        window_bias = torch.zeros(in_window.shape, dtype=torch.float32)
        window_bias = window_bias.masked_fill_(unseen, float("-inf")).to(device)

    offsets: dict[int, torch.Tensor] = {}
    ends: dict[int, BlockEnds | None] = {}
    targets: dict[int, torch.Tensor | None] = {}
    candidates: dict[int, Candidates | None] = {}
    for ratio in sorted({*token_ratios, *block_ratios}):
        closes = (positions + 1) % ratio == 0
        if real is not None:
            closes = closes & real
        closing = torch.nonzero(closes).flatten()
        live = None
        extra = bucket(len(closing)) - len(closing) if len(closing) else 0
        if padded_tokens is not None and extra:
            live = torch.arange(len(closing) + extra) < len(closing)
            closing = F.pad(closing, (0, extra))
        last = positions[closing]

        if ratio in token_ratios:
            offsets[ratio] = (positions % ratio).to(device)
            ends[ratio] = None
        if ratio in token_ratios and len(closing) > 0:
            in_block = last[:, None] - (ratio - 1) + torch.arange(ratio)
            block_reqs = reqs[closing, None].expand(-1, ratio)
            no_previous = (last < 2 * ratio - 1)[:, None, None]
            starts = in_block[:, 0]
            live_rows = None
            if live is not None:
                starts = starts.masked_fill(~live, 0)
                live_rows = live[:, None].expand(-1, ratio)
            ends[ratio] = BlockEnds(
                starts=starts.contiguous().to(device),
                rows=entries_of(block_reqs, in_block, block_size, live_rows),
                previous_rows=entries_of(
                    block_reqs, in_block - ratio, block_size, live_rows
                ),
                previous_bias=torch.zeros(no_previous.shape, dtype=torch.float32)
                .masked_fill_(no_previous, float("-inf"))
                .to(device),
            )

        if ratio not in block_ratios:
            continue
        targets[ratio] = None
        if len(closing) > 0:
            targets[ratio] = entries_of(
                reqs[closing], last // ratio, block_size // ratio, live
            )
        max_blocks = int((seq_lens // ratio).max())
        candidates[ratio] = None
        if max_blocks > 0:
            columns = max_blocks
            if padded_tokens is not None:
                columns = max(bucket(max_blocks), _MIN_CANDIDATES)
            index = torch.arange(columns)[None, :].expand(num_reqs, -1)
            states = entries_of(
                torch.arange(num_reqs)[:, None].expand(-1, columns),
                index,
                block_size // ratio,
                None if columns == max_blocks else index < max_blocks,
            )
            counts = (positions + 1) // ratio
            if real is not None:
                counts = counts.masked_fill(~real, 0)
            candidates[ratio] = Candidates(
                # int32: the card gathers no int64, and the layer gathers these.
                states=states.view(num_reqs, columns).to(torch.int32),
                counts=counts.to(device),
            )

    return DeepseekV4HPUMetadata(
        num_tokens=num_tokens,
        padded_tokens=padded,
        requests=reqs.to(device),
        rows=rows,
        window_rows=window_rows,
        window_bias=window_bias,
        offsets=offsets,
        ends=ends,
        targets=targets,
        candidates=candidates,
    )
