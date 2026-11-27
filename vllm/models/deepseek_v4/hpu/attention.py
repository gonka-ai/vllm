# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged caches of the DeepSeek-V4 attention on a plain torch device.

A layer keeps what a token contributes in caches addressed by position: its
window KV and the inputs of the compressors, one entry per token, and the
compressed KV and indexer key of a block, one entry per ``ratio`` tokens. A
block table then resolves any of them, a cached prefix is reusable as it is,
and no state lives outside the caches.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch
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


@dataclass
class BlockEnds:
    """The tokens of the step that close a compression block of one ratio."""

    # [ends] token index within the step.
    tokens: torch.Tensor
    # [ends] position of the block's first token.
    starts: torch.Tensor
    # [ends * ratio] token entry of every token of the block.
    rows: torch.Tensor
    # [ends * ratio] token entry of every token of the previous block.
    previous_rows: torch.Tensor
    # [ends, 1, 1] 0, or -inf where no previous block exists.
    previous_bias: torch.Tensor
    # [ends] block entry the block compresses into.
    states: torch.Tensor


@dataclass
class Candidates:
    """The compressed blocks of every request of the step."""

    # [reqs, width] block entry of every block of a request, in order.
    states: torch.Tensor
    # [tokens] blocks closed at or before the token.
    counts: torch.Tensor


@dataclass
class DeepseekV4HPUMetadata:
    """Every index of a step, built on the host and resolved through the
    block table on the device. The device is never read back."""

    num_tokens: int
    # [tokens] token entry and request of every token of the step.
    rows: torch.Tensor
    requests: torch.Tensor
    # [tokens, window] token entries of the sliding window; 0 or -inf.
    window_rows: torch.Tensor
    window_bias: torch.Tensor
    # ratio -> [tokens] position within the block.
    offsets: dict[int, torch.Tensor]
    ends: dict[int, BlockEnds | None]
    candidates: dict[int, Candidates | None]


class DeepseekV4HPUImpl(AttentionImpl):
    """The layer runs the attention itself; the backend only owns the caches."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class DeepseekV4HPUBackend(AttentionBackend):
    forward_includes_kv_cache_update: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16, torch.float32]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto"]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(max(RATIOS))]

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
    """One paged cache of a layer: ``width`` values per ``tokens_per_entry``
    tokens. With ``window`` only the last that many tokens of a request are
    ever read, and the engine frees the blocks behind them."""

    def __init__(
        self,
        prefix: str,
        width: int,
        dtype: torch.dtype,
        tokens_per_entry: int = 1,
        window: int | None = None,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.width = width
        self.dtype = dtype
        self.tokens_per_entry = tokens_per_entry
        self.window = window
        # Engine blocks a block of this cache spans, see pack_blocks.
        self.block_scale = 1
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
            return SlidingWindowMLASpec(
                block_size=block_size,
                num_kv_heads=1,
                head_size=self.width,
                dtype=self.dtype,
                sliding_window=self.window,
            )
        return MLAAttentionSpec(
            block_size=block_size * self.block_scale,
            num_kv_heads=1,
            head_size=self.width,
            dtype=self.dtype,
            tokens_per_state=self.tokens_per_entry,
        )

    @property
    def bytes_per_token(self) -> float:
        return self.width * self.dtype.itemsize / self.tokens_per_entry

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


def pack_blocks(caches: list[DeepseekV4HPUCache]) -> None:
    """Sizes the blocks of the caches that are kept whole.

    The engine cuts its memory into blocks of one size, set by the windowed
    caches, which are far wider per token. A block of the kept caches spans as
    many times more tokens as fit that size, or the memory of a context would
    be that of the windowed caches again."""
    windowed = sum(c.bytes_per_token for c in caches if c.window is not None)
    kept = sum(c.bytes_per_token for c in caches if c.window is None)
    scale = max(int(windowed // kept), 1) if kept else 1
    for cache in caches:
        if cache.window is None:
            cache.block_scale = scale


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
        self.window = vllm_config.model_config.hf_config.sliding_window
        # The caches of a builder share a shape, hence the rows of a block.
        context = vllm_config.compilation_config.static_forward_context
        self.cache: DeepseekV4HPUCache = context[layer_names[0]]

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV4HPUMetadata:
        cm = common_attn_metadata
        # A KV cache group holds caches of several shapes and the runner builds
        # once per shape; shapes with the same rows per block share the result.
        built = getattr(cm, "_deepseek_v4_hpu", None)
        if built is None:
            built = {}
            cm._deepseek_v4_hpu = built  # type: ignore[attr-defined]
        step = self.cache.step
        if step not in built:
            num_reqs = cm.num_reqs
            query_start = cm.query_start_loc_cpu[: num_reqs + 1].to(torch.int64)
            seq_lens = cm.seq_lens_cpu_upper_bound
            if seq_lens is None:
                seq_lens = cm.seq_lens.cpu()
            built[step] = build_metadata(
                query_start,
                seq_lens[:num_reqs].to(torch.int64),
                cm.block_table_tensor[:num_reqs],
                self.block_size,
                self.window,
                step,
            )
        return built[step]


def build_metadata(
    query_start: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    window: int,
    step: int | None = None,
) -> DeepseekV4HPUMetadata:
    """The indices of a step, as rows of a cache.

    Args:
        query_start: [reqs + 1] first token of every request, on the host.
        seq_lens: [reqs] length of every request after the step, on the host.
        block_table: [reqs, blocks per request], on the device.
        step: Rows from a block of the cache to the next; by default the
            entries of a block, that is blocks stored back to back.
    """
    device = block_table.device
    num_reqs = seq_lens.shape[0]
    width = block_table.shape[1]
    table = block_table.reshape(-1)

    def entries_of(
        reqs: torch.Tensor, index: torch.Tensor, per_block: int
    ) -> torch.Tensor:
        """Cache rows of the index-th entries of requests, both on the host."""
        index = index.clamp(min=0)
        blocks = table.index_select(
            0, (reqs * width + index // per_block).reshape(-1).to(device)
        )
        offsets = (index % per_block).reshape(-1).to(device)
        return blocks * (step or per_block) + offsets

    query_lens = query_start[1:] - query_start[:-1]
    num_tokens = int(query_start[-1])
    reqs = torch.repeat_interleave(torch.arange(num_reqs), query_lens)
    positions = (seq_lens - query_lens)[reqs] + (
        torch.arange(num_tokens) - query_start[:-1][reqs]
    )

    in_window = positions[:, None] - (window - 1) + torch.arange(window)
    window_rows = entries_of(reqs[:, None].expand(-1, window), in_window, block_size)
    window_bias = torch.zeros(in_window.shape, dtype=torch.float32)
    window_bias.masked_fill_(in_window < 0, float("-inf"))

    offsets: dict[int, torch.Tensor] = {}
    ends: dict[int, BlockEnds | None] = {}
    candidates: dict[int, Candidates | None] = {}
    for ratio in RATIOS:
        offsets[ratio] = (positions % ratio).to(device)
        closing = torch.nonzero((positions + 1) % ratio == 0).flatten()
        ends[ratio] = None
        if len(closing) > 0:
            last = positions[closing]
            in_block = last[:, None] - (ratio - 1) + torch.arange(ratio)
            block_reqs = reqs[closing, None].expand(-1, ratio)
            no_previous = (last < 2 * ratio - 1)[:, None, None]
            ends[ratio] = BlockEnds(
                tokens=closing.to(device),
                starts=in_block[:, 0].contiguous().to(device),
                rows=entries_of(block_reqs, in_block, block_size),
                previous_rows=entries_of(block_reqs, in_block - ratio, block_size),
                previous_bias=torch.zeros(no_previous.shape, dtype=torch.float32)
                .masked_fill_(no_previous, float("-inf"))
                .to(device),
                states=entries_of(reqs[closing], last // ratio, block_size // ratio),
            )
        max_blocks = int((seq_lens // ratio).max())
        candidates[ratio] = None
        if max_blocks > 0:
            candidates[ratio] = Candidates(
                states=entries_of(
                    torch.arange(num_reqs)[:, None].expand(-1, max_blocks),
                    torch.arange(max_blocks)[None, :].expand(num_reqs, -1),
                    block_size // ratio,
                ).view(num_reqs, max_blocks),
                counts=((positions + 1) // ratio).to(device),
            )

    return DeepseekV4HPUMetadata(
        num_tokens=num_tokens,
        rows=entries_of(reqs, positions, block_size),
        requests=reqs.to(device),
        window_rows=window_rows.view(num_tokens, window),
        window_bias=window_bias.to(device),
        offsets=offsets,
        ends=ends,
        candidates=candidates,
    )
