# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged attention in plain torch, for devices with no attention kernel."""

import os
from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayout

_SEQ_BUCKET = 128


class TorchPagedAttentionBackend(AttentionBackend):
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto"]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.LBNHC,)

    @staticmethod
    def get_name() -> str:
        return "TORCH_PAGED"

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @staticmethod
    def get_impl_cls() -> type["TorchPagedAttentionImpl"]:
        return TorchPagedAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["TorchPagedAttentionMetadataBuilder"]:
        return TorchPagedAttentionMetadataBuilder

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


@dataclass
class TorchPagedAttentionMetadata:
    """Every index a layer needs, built once per step and without reading
    from the device: on an accelerator a readback stalls the whole forward."""

    num_actual_tokens: int
    slot_mapping: torch.Tensor
    num_reqs: int
    max_query_len: int
    max_seq_len: int
    # [num_reqs * max_seq_len] cache slot of every context position.
    kv_slots: torch.Tensor
    # [num_reqs * max_query_len] token of every query row, padded per request.
    query_rows: torch.Tensor
    # [num_actual_tokens] row of every token in the padded output.
    output_rows: torch.Tensor
    # [num_reqs, 1, 1, max_query_len, max_seq_len], 0 for the keys a query may
    # attend to and -inf for the rest.
    mask: torch.Tensor
    # [num_reqs, max_query_len] position of every query, on the host.
    query_positions: torch.Tensor


class TorchPagedAttentionMetadataBuilder(
    AttentionMetadataBuilder[TorchPagedAttentionMetadata]
):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = kv_cache_spec.block_size

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TorchPagedAttentionMetadata:
        cm = common_attn_metadata
        num_reqs = cm.num_reqs
        num_tokens = cm.num_actual_tokens
        device = cm.block_table_tensor.device
        query_start = cm.query_start_loc_cpu[: num_reqs + 1].to(torch.int64)
        seq_lens = cm.seq_lens_cpu_upper_bound
        if seq_lens is None:
            seq_lens = cm.seq_lens.cpu()
        seq_lens = seq_lens[:num_reqs].to(torch.int64)
        query_lens = query_start[1:] - query_start[:-1]
        max_query = int(query_lens.max())
        # Rounded up so that a compiled forward sees a new shape once per
        # bucket, not once per token; the mask hides the padding.
        max_seq = -(-int(seq_lens.max()) // _SEQ_BUCKET) * _SEQ_BUCKET

        positions = torch.arange(max_seq)
        table = cm.block_table_tensor[:num_reqs]
        block_ids = table.index_select(
            1, (positions // self.block_size).clamp(max=table.shape[1] - 1).to(device)
        )
        kv_slots = block_ids * self.block_size + (positions % self.block_size).to(
            device
        )

        steps = torch.arange(max_query)
        valid = steps[None, :] < query_lens[:, None]
        query_rows = (query_start[:-1, None] + steps[None, :]).clamp(max=num_tokens - 1)
        output_rows = (torch.arange(num_reqs)[:, None] * max_query + steps[None, :])[
            valid
        ]
        # A query at position p attends to the keys at positions <= p.
        query_positions = (seq_lens - query_lens)[:, None] + steps[None, :]
        mask = positions[None, None, :] <= query_positions[:, :, None]

        return TorchPagedAttentionMetadata(
            num_actual_tokens=num_tokens,
            slot_mapping=cm.slot_mapping,
            num_reqs=num_reqs,
            max_query_len=max_query,
            max_seq_len=max_seq,
            kv_slots=kv_slots.reshape(-1),
            query_rows=query_rows.reshape(-1).to(device),
            output_rows=output_rows.to(device),
            mask=torch.zeros(mask.shape, dtype=torch.float32)
            .masked_fill_(~mask, float("-inf"))[:, None, None]
            .to(device),
            query_positions=query_positions,
        )


def _attention(
    query: torch.Tensor,
    cache: torch.Tensor,
    kv_slots: torch.Tensor,
    mask: torch.Tensor,
    num_reqs: int,
    num_kv_heads: int,
    group: int,
    head_size: int,
    scale: float,
) -> torch.Tensor:
    """Attention of every padded query row over its request's context.

    The query heads of one KV head attend together, so keys and values are
    not repeated per query head: [reqs, kv_heads, group, queries, .].
    """
    cached = cache.index_select(0, kv_slots).view(
        num_reqs, -1, num_kv_heads, 2 * head_size
    )
    cached = cached.permute(0, 2, 1, 3).unsqueeze(2)
    k = cached[..., :head_size]
    v = cached[..., head_size:]
    q = query.view(num_reqs, -1, num_kv_heads, group, head_size)
    q = q.permute(0, 2, 3, 1, 4).to(k.dtype)
    scores = torch.matmul(q, k.transpose(-1, -2)).float() * scale + mask
    out = torch.matmul(torch.softmax(scores, dim=-1).to(v.dtype), v)
    return out.permute(0, 3, 1, 2, 4).reshape(-1, num_kv_heads * group, head_size)


# One compiled core for every layer: a compiled function per layer would
# compile each new shape once per layer.
_attend = _attention
if os.environ.get("VLLM_HPU_COMPILE_ATTN") == "1":
    # Static shapes: the Habana graph compiler rejects symbolic ones.
    _attend = torch.compile(_attention, backend="hpu_backend", dynamic=False)


class TorchPagedAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        if alibi_slopes is not None or logits_soft_cap or sinks is not None:
            raise NotImplementedError(
                "TORCH_PAGED supports neither ALiBi, logits soft cap nor sinks"
            )
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

    @torch.compiler.disable
    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        slots = slot_mapping[: key.shape[0]]
        self._slots(kv_cache).index_copy_(
            0, slots, torch.cat([key, value], dim=-1).to(kv_cache.dtype)
        )

    def _slots(self, kv_cache: torch.Tensor) -> torch.Tensor:
        """The cache as [num_slots, num_kv_heads, 2 * head_size].

        The layer sees [num_blocks, num_kv_heads, block_size, 2 * head_size]
        over memory laid out slot-major (LBNHC), so a slot is one flat index;
        the Habana bridge segfaults on x[i, :, j]."""
        slot_major = kv_cache.permute(0, 2, 1, 3)
        assert slot_major.is_contiguous(), tuple(kv_cache.shape)
        return slot_major.view(-1, self.num_kv_heads, 2 * self.head_size)

    @torch.compiler.disable
    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TorchPagedAttentionMetadata | None,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """query: [num_tokens, num_heads, head_size]; the answer goes to output."""
        if attn_metadata is None:
            return output
        m = attn_metadata
        mask = m.mask
        if self.sliding_window is not None and self.sliding_window > 0:
            positions = torch.arange(m.max_seq_len)[None, None, :]
            outside = positions <= (m.query_positions[:, :, None] - self.sliding_window)
            mask = mask.masked_fill(
                outside[:, None, None].to(mask.device), float("-inf")
            )
        out = _attend(
            query.index_select(0, m.query_rows),
            self._slots(kv_cache),
            m.kv_slots,
            mask,
            m.num_reqs,
            self.num_kv_heads,
            self.num_queries_per_kv,
            self.head_size,
            self.scale,
        )
        output[: m.num_actual_tokens] = out.index_select(0, m.output_rows).to(
            output.dtype
        )
        return output
