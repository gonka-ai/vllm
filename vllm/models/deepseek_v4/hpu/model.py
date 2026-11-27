# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 on a plain torch device (Intel Gaudi).

The math is DeepSeek's inference/model.py. The routed experts stay packed in
MXFP4 as in the checkpoint and run in the device's MoE kernel, which takes the
activation in bfloat16; the FP8 weights are held dequantized; attention is a
gather, a GEMM and a softmax over paged caches. Attention and the shared
expert are whole on every tensor-parallel rank, the routed experts are split
between the ranks.
"""

import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

import regex as re
import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.models.deepseek_v4.hpu import ops
from vllm.models.deepseek_v4.hpu.attention import (
    DeepseekV4HPUCache,
    DeepseekV4HPUMetadata,
    pack_blocks,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

# Elements one attention chunk may gather.
_ATTENTION_CHUNK = 1 << 27
_EXPERT = re.compile(r"layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)")
_RENAMES = (
    (".ffn.gate.bias", ".ffn.gate.e_score_correction_bias"),
    (".ffn.shared_experts.w", ".ffn.shared_w"),
)


def _param(*shape: int, dtype: torch.dtype) -> nn.Parameter:
    return nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)


def _gemm_dtype() -> torch.dtype:
    """The card multiplies in bfloat16; elsewhere the model is a reference."""
    if current_platform.device_type == "hpu":
        return torch.bfloat16
    return torch.float32


@dataclass
class _Rope:
    """A RoPE table and its rows at the positions of the step."""

    cos_table: torch.Tensor
    sin_table: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor


def _rotate_tail(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """RoPE on the last 2 * cos.shape[-1] dims of x."""
    dim = 2 * cos.shape[-1]
    tail = ops.apply_rope(x[..., -dim:], cos, sin, inverse)
    return torch.cat([x[..., :-dim], tail], dim=-1)


class DeepseekV4HPUCompressor(nn.Module):
    """Compressor: a block of ``ratio`` tokens pooled into one KV."""

    def __init__(
        self, config, prefix: str, ratio: int, head_dim: int, rotate: bool
    ) -> None:
        super().__init__()
        self.ratio = ratio
        self.head_dim = head_dim
        self.rotate = rotate
        self.overlap = ratio == 4
        self.eps = config.rms_norm_eps
        width = (1 + self.overlap) * head_dim
        self.ape = _param(ratio, width, dtype=torch.float32)
        self.wkv = _param(width, config.hidden_size, dtype=torch.float32)
        self.wgate = _param(width, config.hidden_size, dtype=torch.float32)
        self.norm = _param(head_dim, dtype=torch.float32)
        # Per token: the KV and the score it adds to its block.
        # A block draws on its own tokens and, at ratio 4, on the block before:
        # never further back than the attention window.
        self.states = DeepseekV4HPUCache(
            f"{prefix}.states", 2 * width, torch.float32, window=config.sliding_window
        )
        self.blocks = DeepseekV4HPUCache(
            f"{prefix}.blocks", head_dim, torch.bfloat16, ratio
        )
        if rotate:
            rotation = ops.hadamard(head_dim) * head_dim**-0.5
            self.register_buffer(
                "rotation", rotation.to(self.norm.device), persistent=False
            )

    def forward(
        self,
        x: torch.Tensor,
        metadata: dict[str, DeepseekV4HPUMetadata],
        rope: _Rope,
        dtype: torch.dtype,
    ) -> None:
        """Stores the step's tokens and compresses the blocks they close.

        Args:
            x: [tokens, hidden] float32.
            metadata: The indices of every cache, by its name.
            dtype: The activation dtype.
        """
        tokens = metadata[self.states.prefix]
        kv = F.linear(x, self.wkv)
        score = F.linear(x, self.wgate) + self.ape.index_select(
            0, tokens.offsets[self.ratio]
        )
        states = self.states.entries()
        states.index_copy_(0, tokens.rows, torch.cat([kv, score], dim=-1))
        ends = tokens.ends[self.ratio]
        if ends is None:
            return

        d = self.head_dim
        block = states.index_select(0, ends.rows).view(-1, self.ratio, 2, kv.shape[-1])
        kvs, scores = block[:, :, 0], block[:, :, 1]
        if self.overlap:
            # The first half of the dims belongs to the window that overlaps
            # the previous block, the second half to the block itself.
            previous = states.index_select(0, ends.previous_rows).view(block.shape)
            kvs = torch.cat([previous[:, :, 0, :d], kvs[..., d:]], dim=1)
            scores = torch.cat(
                [previous[:, :, 1, :d] + ends.previous_bias, scores[..., d:]], dim=1
            )
        pooled = (kvs * scores.softmax(dim=1)).sum(dim=1)
        out = ops.rms_norm(pooled.to(dtype), self.norm, self.eps)
        out = _rotate_tail(
            out,
            rope.cos_table.index_select(0, ends.starts),
            rope.sin_table.index_select(0, ends.starts),
        )
        if self.rotate:
            out = torch.matmul(out.float(), self.rotation).to(dtype)
            out = ops.fp4_quant_dequant(out)
        else:
            rope_dim = 2 * rope.cos.shape[-1]
            head = ops.fp8_quant_dequant(out[..., :-rope_dim], 64)
            out = torch.cat([head, out[..., -rope_dim:].float()], dim=-1)
        targets = metadata[self.blocks.prefix].ends[self.ratio]
        assert targets is not None
        cache = self.blocks.entries()
        cache.index_copy_(0, targets.states, out.to(cache.dtype))


class DeepseekV4HPUIndexer(nn.Module):
    """Indexer: picks the compressed blocks a token attends to."""

    def __init__(self, config, prefix: str, gemm_dtype: torch.dtype) -> None:
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.topk = config.index_topk
        self.wq_b = _param(
            self.n_heads * self.head_dim, config.q_lora_rank, dtype=gemm_dtype
        )
        self.weights_proj = _param(self.n_heads, config.hidden_size, dtype=gemm_dtype)
        self.compressor = DeepseekV4HPUCompressor(
            config, f"{prefix}.compressor", 4, self.head_dim, rotate=True
        )
        self.weight_scale = self.head_dim**-0.5 * self.n_heads**-0.5

    def queries(
        self, x: torch.Tensor, qr: torch.Tensor, rope: _Rope
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The index queries [tokens, heads, dim] and head weights [tokens, heads].

        Args:
            qr: The low-rank query through its FP8 image, in the GEMM dtype.
        """
        dtype = x.dtype
        q = F.linear(qr, self.wq_b).to(dtype).view(-1, self.n_heads, self.head_dim)
        q = _rotate_tail(q, rope.cos[:, None], rope.sin[:, None])
        q = torch.matmul(q.float(), self.compressor.rotation).to(dtype)
        q = ops.fp4_quant_dequant(q).to(dtype)
        weights = F.linear(x.to(self.weights_proj.dtype), self.weights_proj).to(dtype)
        return q, weights * self.weight_scale

    def select(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        states: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        """The top blocks of every token, as columns of ``states`` [tokens, n]."""
        keys = self.compressor.blocks.entries().index_select(0, states.reshape(-1))
        keys = keys.view(*states.shape, self.head_dim).to(self.wq_b.dtype)
        scores = torch.matmul(q.to(keys.dtype), keys.transpose(1, 2)).float().relu()
        scores = (scores * weights.float().unsqueeze(-1)).sum(dim=1)
        # Not ``+ bias``: the key of a block that is not closed may be NaN.
        scores = scores.masked_fill(bias < 0, float("-inf"))
        return scores.topk(self.topk, dim=-1).indices


class DeepseekV4HPUAttention(nn.Module):
    """Attention: a sliding window plus the compressed blocks before it."""

    def __init__(
        self, vllm_config: VllmConfig, layer_id: int, prefix: str, gemm_dtype
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.ratio = config.compress_ratios[layer_id]
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.window = config.sliding_window
        self.n_groups = config.o_groups
        self.eps = config.rms_norm_eps
        self.scale = self.head_dim**-0.5
        hidden = config.hidden_size
        self.attn_sink = _param(self.n_heads, dtype=torch.float32)
        self.wq_a = _param(config.q_lora_rank, hidden, dtype=gemm_dtype)
        self.q_norm = _param(config.q_lora_rank, dtype=torch.float32)
        self.wq_b = _param(
            self.n_heads * self.head_dim, config.q_lora_rank, dtype=gemm_dtype
        )
        self.wkv = _param(self.head_dim, hidden, dtype=gemm_dtype)
        self.kv_norm = _param(self.head_dim, dtype=torch.float32)
        self.wo_a = _param(
            self.n_groups,
            config.o_lora_rank,
            self.n_heads * self.head_dim // self.n_groups,
            dtype=gemm_dtype,
        )
        self.wo_b = _param(hidden, self.n_groups * config.o_lora_rank, dtype=gemm_dtype)
        self.kv = DeepseekV4HPUCache(
            f"{prefix}.kv", self.head_dim, torch.bfloat16, window=self.window
        )
        self.compressor = None
        self.indexer = None
        if self.ratio:
            self.compressor = DeepseekV4HPUCompressor(
                config, f"{prefix}.compressor", self.ratio, self.head_dim, rotate=False
            )
        if self.ratio == 4:
            self.indexer = DeepseekV4HPUIndexer(config, f"{prefix}.indexer", gemm_dtype)

    def forward(self, x: torch.Tensor, rope: _Rope) -> torch.Tensor:
        context = get_forward_context().attn_metadata
        if not isinstance(context, dict):
            # A profile run has no caches.
            return torch.zeros_like(x)
        metadata = cast(dict[str, DeepseekV4HPUMetadata], context)
        tokens = metadata[self.kv.prefix]
        assert x.shape[0] == tokens.num_tokens, (x.shape, tokens.num_tokens)
        dtype = x.dtype
        gemm = self.wq_a.dtype

        xq = ops.fp8_quant_dequant(x).to(gemm)
        qr = ops.rms_norm(F.linear(xq, self.wq_a).to(dtype), self.q_norm, self.eps)
        qr = ops.fp8_quant_dequant(qr).to(gemm)
        q = F.linear(qr, self.wq_b).to(dtype).view(-1, self.n_heads, self.head_dim)
        q = _rotate_tail(
            ops.unit_rms(q, self.eps), rope.cos[:, None], rope.sin[:, None]
        )

        kv = ops.rms_norm(F.linear(xq, self.wkv).to(dtype), self.kv_norm, self.eps)
        kv = _rotate_tail(kv, rope.cos, rope.sin)
        rope_dim = 2 * rope.cos.shape[-1]
        head = ops.fp8_quant_dequant(kv[..., :-rope_dim], 64).to(dtype)
        cache = self.kv.entries()
        cache.index_copy_(
            0,
            tokens.rows,
            torch.cat([head, kv[..., -rope_dim:]], dim=-1).to(cache.dtype),
        )

        index_q = index_weights = None
        if self.compressor is not None:
            xf = x.float()
            self.compressor(xf, metadata, rope, dtype)
            if self.indexer is not None:
                self.indexer.compressor(xf, metadata, rope, dtype)
                index_q, index_weights = self.indexer.queries(x, qr, rope)

        out = self._attend(q.to(gemm), metadata, index_q, index_weights).to(dtype)
        out = _rotate_tail(out, rope.cos[:, None], rope.sin[:, None], inverse=True)
        out = out.view(out.shape[0], self.n_groups, -1).transpose(0, 1).to(gemm)
        out = torch.matmul(out, self.wo_a.transpose(1, 2)).transpose(0, 1).to(dtype)
        return ops.fp8_linear(out.flatten(1), self.wo_b)

    def _attend(
        self,
        q: torch.Tensor,
        metadata: dict[str, DeepseekV4HPUMetadata],
        index_q: torch.Tensor | None,
        index_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        tokens = metadata[self.kv.prefix]
        kv_cache = self.kv.entries()
        candidates = None
        if self.compressor is not None:
            block_cache = self.compressor.blocks.entries()
            candidates = metadata[self.compressor.blocks.prefix].candidates[self.ratio]

        num_blocks = 0 if candidates is None else candidates.states.shape[1]
        # The indexer ranks the blocks once there are more than it keeps.
        ranker = self.indexer
        if ranker is not None and num_blocks <= ranker.topk:
            ranker = None
        kept = num_blocks if ranker is None else ranker.topk
        per_token = (self.window + kept) * self.head_dim
        index_keys = None
        if ranker is not None:
            per_token = max(per_token, num_blocks * ranker.head_dim)
            # The indexer's keys sit in a cache of their own.
            index_keys = metadata[ranker.compressor.blocks.prefix].candidates[
                self.ratio
            ]
        chunk = max(1, _ATTENTION_CHUNK // per_token)

        outs = []
        for start in range(0, q.shape[0], chunk):
            part = slice(start, start + chunk)
            bias = tokens.window_bias[part]
            n = bias.shape[0]
            keys = kv_cache.index_select(0, tokens.window_rows[part].reshape(-1))
            keys = keys.view(n, self.window, self.head_dim)
            if candidates is not None:
                states = candidates.states.index_select(0, tokens.requests[part])
                order = torch.arange(num_blocks, device=states.device)
                closed = order[None, :] < candidates.counts[part, None]
                block_bias = torch.zeros(
                    states.shape, dtype=torch.float32, device=states.device
                ).masked_fill(~closed, float("-inf"))
                if ranker is not None:
                    assert index_q is not None and index_weights is not None
                    assert index_keys is not None
                    top = ranker.select(
                        index_q[part],
                        index_weights[part],
                        index_keys.states.index_select(0, tokens.requests[part]),
                        block_bias,
                    )
                    states = states.gather(1, top)
                    block_bias = block_bias.gather(1, top)
                compressed = block_cache.index_select(0, states.reshape(-1))
                # A block that is not closed is a row nobody wrote: bytes left
                # by a cache of another dtype, NaN among them. The bias alone
                # would let those through the softmax.
                compressed = compressed.view(n, kept, self.head_dim).masked_fill(
                    (block_bias < 0).unsqueeze(-1), 0
                )
                keys = torch.cat([keys, compressed], dim=1)
                bias = torch.cat([bias, block_bias], dim=1)
            outs.append(
                ops.sparse_attn(
                    q[part], keys.to(q.dtype), bias, self.attn_sink, self.scale
                )
            )
        return outs[0] if len(outs) == 1 else torch.cat(outs)


_native_moe = None


def _native_experts(x, ids, weights, w13, w2, w13_scale, w2_scale, first, last):
    """The card's MoE kernel on packed MXFP4 experts; it runs compiled only."""
    global _native_moe
    if _native_moe is None:

        def moe(x, ids, weights, w13, w2, w13_scale, w2_scale, first, last):
            return torch.ops.hpu.mixture_of_experts.mxfp4_fused_weights(
                x,
                ids,
                weights,
                w13,
                w2,
                w13_scale,
                w2_scale,
                block_size=32,
                permuted_weights=True,
                activation="silu",
                experts_min=first,
                experts_max=last,
                is_fp4=True,
            )

        _native_moe = torch.compile(moe, backend="hpu_backend")
    return _native_moe(x, ids, weights, w13, w2, w13_scale, w2_scale, first, last)


class DeepseekV4HPUNorm(nn.Module):
    """RMSNorm with a float32 weight."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = _param(dim, dtype=torch.float32)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ops.rms_norm(x, self.weight, self.eps)


class DeepseekV4HPUGate(nn.Module):
    """The router: the logit of every expert, and on a hash layer the experts
    of every token id."""

    def __init__(self, config, layer_id: int) -> None:
        super().__init__()
        num_experts = config.n_routed_experts
        self.weight = _param(num_experts, config.hidden_size, dtype=torch.float32)
        if layer_id < config.num_hash_layers:
            self.register_parameter("e_score_correction_bias", None)
            self.tid2eid = _param(
                config.vocab_size, config.num_experts_per_tok, dtype=torch.int32
            )
        else:
            self.e_score_correction_bias = _param(num_experts, dtype=torch.float32)
            self.register_parameter("tid2eid", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x.float(), self.weight)


class DeepseekV4HPUExperts(nn.Module):
    """The routed experts of this rank, packed as in the checkpoint."""

    def __init__(self, config, gemm_dtype: torch.dtype) -> None:
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.limit = config.swiglu_limit
        self.gemm_dtype = gemm_dtype
        self.hidden = hidden = config.hidden_size
        self.inter = inter = config.moe_intermediate_size
        tp_size = get_tensor_model_parallel_world_size()
        assert self.num_experts % tp_size == 0
        self.num_local = local = self.num_experts // tp_size
        self.first = get_tensor_model_parallel_rank() * local
        self.native = (
            current_platform.device_type == "hpu"
            and os.environ.get("VLLM_HPU_DEEPSEEK_V4_MOE", "native") == "native"
        )
        # Gate (w1) over up (w3), two float4_e2m1 codes per byte, and one
        # E8M0 scale per 32 codes of a row.
        self.w13 = _param(local, 2 * inter, hidden // 2, dtype=torch.uint8)
        self.w13_scale = _param(local, 2 * inter, hidden // 32, dtype=torch.uint8)
        self.w2 = _param(local, hidden, inter // 2, dtype=torch.uint8)
        self.w2_scale = _param(local, hidden, inter // 32, dtype=torch.uint8)
        self.pieces_loaded = 0
        self._lists: tuple | None = None

    def load_expert(
        self, expert: int, matrix: str, kind: str, tensor: torch.Tensor
    ) -> None:
        local = expert - self.first
        if not 0 <= local < self.num_local:
            return
        if matrix == "w2":
            target = (self.w2 if kind == "weight" else self.w2_scale).data[local]
        else:
            target = (self.w13 if kind == "weight" else self.w13_scale).data[local]
            half = slice(0, self.inter) if matrix == "w1" else slice(self.inter, None)
            target = target[half]
        target.copy_(tensor.view(torch.uint8))
        self.pieces_loaded += 1

    def forward(
        self, x: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """The share of this rank's experts in the output, float32.

        Args:
            indices: [tokens, top_k] expert ids over all ranks.
            weights: [tokens, top_k] float32.
        """
        if not self.native:
            return self._torch(x, indices, weights)
        if self._lists is None:
            self._lists = tuple(
                tuple(p[i] for i in range(self.num_local))
                for p in (self.w13, self.w2, self.w13_scale, self.w2_scale)
            )
        w13, w2, w13_scale, w2_scale = self._lists
        return _native_experts(
            x,
            indices.to(torch.int32),
            weights.to(x.dtype),
            w13,
            w2,
            w13_scale,
            w2_scale,
            self.first,
            self.first + self.num_local - 1,
        ).float()

    def _torch(
        self, x: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """The experts in plain torch, dequantized as they are used."""
        gemm = self.gemm_dtype
        out = torch.zeros(x.shape[0], self.hidden, dtype=torch.float32, device=x.device)
        xg = x.to(gemm)
        for local in range(self.num_local):
            rows, slots = torch.where(indices == self.first + local)
            if rows.numel() == 0:
                continue
            w13 = ops.dequant_fp4_weight(self.w13[local], self.w13_scale[local])
            gate_up = F.linear(xg[rows], w13.to(gemm)).to(x.dtype)
            hidden = ops.swiglu(
                gate_up[:, : self.inter], gate_up[:, self.inter :], self.limit
            )
            hidden = (hidden * weights[rows, slots, None]).to(x.dtype)
            w2 = ops.dequant_fp4_weight(self.w2[local], self.w2_scale[local])
            out.index_add_(
                0, rows, F.linear(hidden.to(gemm), w2.to(gemm)).to(x.dtype).float()
            )
        return out


class DeepseekV4HPUMoE(nn.Module):
    """MoE: a gate, the routed experts and the shared expert. The names
    ``gate`` and ``experts`` are what the PoC plugin looks for."""

    def __init__(self, vllm_config: VllmConfig, layer_id: int, gemm_dtype) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.top_k = config.num_experts_per_tok
        self.route_scale = config.routed_scaling_factor
        self.limit = config.swiglu_limit
        self.tp_size = get_tensor_model_parallel_world_size()
        hidden, inter = config.hidden_size, config.moe_intermediate_size
        self.gate = DeepseekV4HPUGate(config, layer_id)
        self.experts = DeepseekV4HPUExperts(config, gemm_dtype)
        self.shared_w1 = _param(inter, hidden, dtype=gemm_dtype)
        self.shared_w3 = _param(inter, hidden, dtype=gemm_dtype)
        self.shared_w2 = _param(hidden, inter, dtype=gemm_dtype)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor | None) -> torch.Tensor:
        hash_experts = None
        if self.gate.tid2eid is not None:
            if input_ids is None:
                raise ValueError("DeepSeek-V4 hash MoE routing requires input_ids.")
            hash_experts = self.gate.tid2eid.index_select(0, input_ids).long()
        weights, indices = ops.route(
            self.gate(x),
            self.gate.e_score_correction_bias,
            hash_experts,
            self.top_k,
            self.route_scale,
        )
        routed = self.experts(x, indices, weights)
        if self.tp_size > 1:
            routed = tensor_model_parallel_all_reduce(routed)

        gemm = self.shared_w1.dtype
        xq = ops.fp8_quant_dequant(x).to(gemm)
        hidden = ops.swiglu(
            F.linear(xq, self.shared_w1).to(x.dtype),
            F.linear(xq, self.shared_w3).to(x.dtype),
            self.limit,
        )
        shared = ops.fp8_linear(hidden.to(x.dtype), self.shared_w2)
        return (routed + shared.float()).to(x.dtype)


class DeepseekV4HPULayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, layer_id: int, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        gemm_dtype = _gemm_dtype()
        self.eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.hc_iters = config.hc_sinkhorn_iters
        hidden, hc = config.hidden_size, config.hc_mult
        self.attn = DeepseekV4HPUAttention(
            vllm_config, layer_id, f"{prefix}.attn", gemm_dtype
        )
        self.ffn = DeepseekV4HPUMoE(vllm_config, layer_id, gemm_dtype)
        self.attn_norm = _param(hidden, dtype=torch.float32)
        self.ffn_norm = _param(hidden, dtype=torch.float32)
        for name in ("attn", "ffn"):
            setattr(
                self,
                f"hc_{name}_fn",
                _param((2 + hc) * hc, hc * hidden, dtype=torch.float32),
            )
            setattr(self, f"hc_{name}_base", _param((2 + hc) * hc, dtype=torch.float32))
            setattr(self, f"hc_{name}_scale", _param(3, dtype=torch.float32))

    def _hc_pre(self, h: torch.Tensor, name: str):
        return ops.hc_pre(
            h,
            getattr(self, f"hc_{name}_fn"),
            getattr(self, f"hc_{name}_scale"),
            getattr(self, f"hc_{name}_base"),
            self.eps,
            self.hc_eps,
            self.hc_iters,
        )

    def forward(
        self, h: torch.Tensor, input_ids: torch.Tensor | None, rope: _Rope
    ) -> torch.Tensor:
        x, post, comb = self._hc_pre(h, "attn")
        x = self.attn(ops.rms_norm(x, self.attn_norm, self.eps), rope)
        h = ops.hc_post(x, h, post, comb)
        x, post, comb = self._hc_pre(h, "ffn")
        x = self.ffn(ops.rms_norm(x, self.ffn_norm, self.eps), input_ids)
        return ops.hc_post(x, h, post, comb)


class DeepseekV4Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        hidden, hc = config.hidden_size, config.hc_mult
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, hidden, prefix=f"{prefix}.embed_tokens"
        )
        self.layers = nn.ModuleList(
            DeepseekV4HPULayer(vllm_config, i, f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)
        )
        # A module: the PoC plugin wraps the final norm.
        self.norm = DeepseekV4HPUNorm(hidden, config.rms_norm_eps)
        self.hc_head_fn = _param(hc, hc * hidden, dtype=torch.float32)
        self.hc_head_base = _param(hc, dtype=torch.float32)
        self.hc_head_scale = _param(1, dtype=torch.float32)
        pack_blocks([m for m in self.modules() if isinstance(m, DeepseekV4HPUCache)])

        # A layer with no compressor rotates by the base theta without YaRN.
        scaling = config.rope_scaling
        max_len = vllm_config.model_config.max_model_len
        for name, original_len, theta in (
            ("window", 0, config.rope_theta),
            (
                "compress",
                scaling["original_max_position_embeddings"],
                config.compress_rope_theta,
            ),
        ):
            cos, sin = ops.rope_table(
                config.qk_rope_head_dim,
                max_len,
                original_len,
                theta,
                scaling["factor"],
                scaling["beta_fast"],
                scaling["beta_slow"],
            )
            device = self.hc_head_fn.device
            self.register_buffer(f"{name}_cos", cos.to(device), persistent=False)
            self.register_buffer(f"{name}_sin", sin.to(device), persistent=False)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        h = inputs_embeds.unsqueeze(1).repeat(1, self.config.hc_mult, 1)
        ropes = {}
        for compressed, name in ((False, "window"), (True, "compress")):
            cos, sin = getattr(self, f"{name}_cos"), getattr(self, f"{name}_sin")
            ropes[compressed] = _Rope(
                cos, sin, cos.index_select(0, positions), sin.index_select(0, positions)
            )
        for layer in self.layers:
            h = layer(h, input_ids, ropes[layer.attn.ratio > 0])
        h = ops.hc_head(
            h,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.config.rms_norm_eps,
            self.config.hc_eps,
        )
        return self.norm(h)


class DeepseekV4ForCausalLM(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.model = DeepseekV4Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Loads the checkpoint as DeepSeek ships it: an FP8 weight meets its
        block scales here and is stored dequantized, a routed expert is stored
        packed, and the speculative-decoding stage (mtp.*) is left out."""
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        halves: dict[str, dict[str, torch.Tensor]] = {}

        def store(name: str, tensor: torch.Tensor) -> None:
            param = params[name]
            assert param.numel() == tensor.numel(), (name, param.shape, tensor.shape)
            param.data.copy_(tensor.view(param.shape))
            loaded.add(name)

        for name, tensor in weights:
            if name.startswith("mtp."):
                continue
            if expert := _EXPERT.fullmatch(name):
                layer, index, matrix, kind = expert.groups()
                self.model.layers[int(layer)].ffn.experts.load_expert(
                    int(index), matrix, kind, tensor
                )
                continue
            if name in ("embed.weight", "head.weight"):
                target = (
                    "model.embed_tokens.weight"
                    if name == "embed.weight"
                    else "lm_head.weight"
                )
                params[target].weight_loader(params[target], tensor)
                loaded.add(target)
                continue
            for old, new in _RENAMES:
                name = name.replace(old, new)
            stem, _, kind = name.rpartition(".")
            if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
                pair = halves.setdefault(stem, {})
                pair[kind] = tensor
                if len(pair) == 2:
                    del halves[stem]
                    store(
                        f"model.{stem}",
                        ops.dequant_fp8_weight(pair["weight"], pair["scale"]),
                    )
            elif f"model.{name}" in params:
                store(f"model.{name}", tensor)
            else:
                assert kind == "weight", name
                store(f"model.{stem}", tensor)

        assert not halves, sorted(halves)
        for index, layer in enumerate(self.model.layers):
            experts = layer.ffn.experts
            pieces = experts.pieces_loaded
            assert pieces == 6 * experts.num_local, (index, pieces)
            for name in ("w13", "w13_scale", "w2", "w2_scale"):
                loaded.add(f"model.layers.{index}.ffn.experts.{name}")
        return loaded
