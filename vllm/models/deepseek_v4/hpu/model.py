# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 on a plain torch device (Intel Gaudi).

The math is DeepSeek's inference/model.py. The routed experts stay packed in
MXFP4 as in the checkpoint and run in the device's MoE kernel, which takes the
activation in bfloat16; the FP8 weights are held dequantized; attention is a
gather, a GEMM and a softmax over paged caches. Attention and the shared
expert are whole on every tensor-parallel rank, the routed experts are split
between the ranks.

On the card the math between two cache accesses runs as compiled graphs
(``_graph``); the cache reads and writes and the modules the PoC plugin wraps
stay calls of their own.
"""

import functools
import os
from collections.abc import Callable, Iterable
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
    GRAPHS,
    DeepseekV4HPUCache,
    DeepseekV4HPUMetadata,
    align_rows,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

# Elements one attention chunk may gather.
_ATTENTION_CHUNK = 1 << 29
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


# A function compiles once per shape it meets: the token counts, candidate
# counts and block-end counts a step is padded to, by the kind of layer.
_RECOMPILE_LIMITS = {"recompile_limit": 1024, "accumulated_recompile_limit": 8192}


def _compile(fn: Callable, static: bool = True) -> Callable:
    config = torch._dynamo.config
    for name, limit in _RECOMPILE_LIMITS.items():
        setattr(config, name, max(getattr(config, name), limit))
    return torch.compile(fn, backend="hpu_backend", dynamic=False if static else None)


def _graph(fn: Callable) -> Callable:
    """``fn``, a pure function of tensors, as one graph of the card.

    Op by op the card takes about 60 us per call whatever the size, and a
    layer is some two thousand calls. A graph is static: compiled for changing
    shapes, the card resolves some of them wrongly or fails to patch them,
    hence the padding of a step (attention.bucket). Off the card ``fn`` runs
    as it is."""
    compiled = None

    @functools.wraps(fn)
    def run(*args):
        nonlocal compiled
        if not GRAPHS or args[0].device.type != "hpu":
            return fn(*args)
        if compiled is None:
            compiled = _compile(fn)
        return compiled(*args)

    return run


@_graph
def _token_state(
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    ape: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """What a token adds to its block: its KV and its score, float32."""
    x = x.float()
    score = F.linear(x, wgate) + ape.index_select(0, offsets)
    return torch.cat([F.linear(x, wkv), score], dim=-1)


@_graph
def _pool(
    block: torch.Tensor,
    previous: torch.Tensor | None,
    previous_bias: torch.Tensor | None,
    norm: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotation: torch.Tensor | None,
    head_dim: int,
    eps: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """The KV of closed blocks from the states of their tokens.

    Args:
        block: [ends, ratio, 2, width] KV and score of every token.
        previous: The same of the block before, at ratio 4.
        rotation: The Hadamard rotation of an indexer key, or None for a KV.
    """
    kvs, scores = block[:, :, 0], block[:, :, 1]
    if previous is not None:
        # The first half of the dims belongs to the window that overlaps
        # the previous block, the second half to the block itself.
        d = head_dim
        kvs = torch.cat([previous[:, :, 0, :d], kvs[..., d:]], dim=1)
        scores = torch.cat(
            [previous[:, :, 1, :d] + previous_bias, scores[..., d:]], dim=1
        )
    pooled = (kvs * scores.softmax(dim=1)).sum(dim=1)
    out = _rotate_tail(ops.rms_norm(pooled.to(dtype), norm, eps), cos, sin)
    if rotation is not None:
        out = torch.matmul(out.float(), rotation).to(dtype)
        return ops.fp4_quant_dequant(out)
    rope_dim = 2 * cos.shape[-1]
    head = ops.fp8_quant_dequant(out[..., :-rope_dim], 64)
    return torch.cat([head, out[..., -rope_dim:].float()], dim=-1)


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
        # Per token: the KV and the score it adds to its block. A block draws
        # on its own tokens and, at ratio 4, on the block before.
        self.states = DeepseekV4HPUCache(
            f"{prefix}.states",
            2 * width,
            torch.float32,
            ratio,
            window=(1 + self.overlap) * ratio,
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
    ) -> None:
        """Stores the step's tokens and compresses the blocks they close.

        Args:
            x: [tokens, hidden].
            metadata: The indices of every cache, by its name.
        """
        tokens = metadata[self.states.prefix]
        assert tokens.rows is not None
        states = self.states.entries()
        states.index_copy_(
            0,
            tokens.rows,
            _token_state(x, self.wkv, self.wgate, self.ape, tokens.offsets[self.ratio]),
        )
        ends = tokens.ends[self.ratio]
        if ends is None:
            return

        shape = (-1, self.ratio, 2, self.wkv.shape[0])
        block = states.index_select(0, ends.rows).view(shape)
        previous = previous_bias = None
        if self.overlap:
            previous = states.index_select(0, ends.previous_rows).view(shape)
            previous_bias = ends.previous_bias
        out = _pool(
            block,
            previous,
            previous_bias,
            self.norm,
            rope.cos_table.index_select(0, ends.starts),
            rope.sin_table.index_select(0, ends.starts),
            self.rotation if self.rotate else None,
            self.head_dim,
            self.eps,
            x.dtype,
        )
        targets = metadata[self.blocks.prefix].targets[self.ratio]
        assert targets is not None
        cache = self.blocks.entries()
        cache.index_copy_(0, targets, out.to(cache.dtype))


@_graph
def _index_queries(
    x: torch.Tensor,
    qr: torch.Tensor,
    wq_b: torch.Tensor,
    weights_proj: torch.Tensor,
    rotation: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    n_heads: int,
    weight_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = x.dtype
    q = F.linear(qr, wq_b).to(dtype).view(x.shape[0], n_heads, -1)
    q = _rotate_tail(q, cos[:, None], sin[:, None])
    q = torch.matmul(q.float(), rotation).to(dtype)
    q = ops.fp4_quant_dequant(q).to(dtype)
    weights = F.linear(x.to(weights_proj.dtype), weights_proj).to(dtype)
    return q, weights * weight_scale


@_graph
def _rank(
    q: torch.Tensor,
    weights: torch.Tensor,
    keys: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    scores = torch.matmul(q.to(keys.dtype), keys.transpose(1, 2)).float().relu()
    scores = (scores * weights.float().unsqueeze(-1)).sum(dim=1)
    # Not ``+ bias``: the key of a block that is not closed may be NaN.
    scores = scores.masked_fill(bias < 0, float("-inf"))
    return scores.topk(topk, dim=-1).indices


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
        return _index_queries(
            x,
            qr,
            self.wq_b,
            self.weights_proj,
            self.compressor.rotation,
            rope.cos,
            rope.sin,
            self.n_heads,
            self.weight_scale,
        )

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
        return _rank(q, weights, keys, bias, self.topk)


@_graph
def _queries(
    x: torch.Tensor,
    wq_a: torch.Tensor,
    q_norm: torch.Tensor,
    wq_b: torch.Tensor,
    wkv: torch.Tensor,
    kv_norm: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    n_heads: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The queries [tokens, heads, dim] and the low-rank query under them, both
    in the GEMM dtype, and the KV of every token as its cache holds it."""
    dtype, gemm = x.dtype, wq_a.dtype
    xq = ops.fp8_quant_dequant(x).to(gemm)
    qr = ops.rms_norm(F.linear(xq, wq_a).to(dtype), q_norm, eps)
    qr = ops.fp8_quant_dequant(qr).to(gemm)
    q = F.linear(qr, wq_b).to(dtype).view(x.shape[0], n_heads, -1)
    q = _rotate_tail(ops.unit_rms(q, eps), cos[:, None], sin[:, None])

    kv = ops.rms_norm(F.linear(xq, wkv).to(dtype), kv_norm, eps)
    kv = _rotate_tail(kv, cos, sin)
    rope_dim = 2 * cos.shape[-1]
    head = ops.fp8_quant_dequant(kv[..., :-rope_dim], 64).to(dtype)
    return q.to(gemm), qr, torch.cat([head, kv[..., -rope_dim:]], dim=-1)


@_graph
def _sparse_attn(
    q: torch.Tensor,
    keys: torch.Tensor,
    bias: torch.Tensor,
    blocks: torch.Tensor | None,
    block_bias: torch.Tensor | None,
    attn_sink: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Attention over the window keys and, after them, compressed blocks."""
    if blocks is not None:
        assert block_bias is not None
        keys = torch.cat([keys, blocks], dim=1)
        bias = torch.cat([bias, block_bias], dim=1)
    # A key a token does not attend to may be a row nobody wrote: a block that
    # is not closed, the null block under a padded row. Bytes left there by a
    # cache of another dtype read as NaN, and the bias alone would let that
    # through the softmax.
    keys = keys.masked_fill((bias < 0).unsqueeze(-1), 0)
    return ops.sparse_attn(q, keys.to(q.dtype), bias, attn_sink, scale)


@_graph
def _project(
    out: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """The attention output [tokens, heads, dim] through the grouped low-rank
    output projection."""
    gemm, n_groups = wo_a.dtype, wo_a.shape[0]
    out = _rotate_tail(out.to(dtype), cos[:, None], sin[:, None], inverse=True)
    out = out.view(out.shape[0], n_groups, -1).transpose(0, 1).to(gemm)
    out = torch.matmul(out, wo_a.transpose(1, 2)).transpose(0, 1).to(dtype)
    return ops.fp8_linear(out.flatten(1), wo_b)


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
        assert x.shape[0] == tokens.padded_tokens, (x.shape, tokens.padded_tokens)
        q, qr, kv = _queries(
            x,
            self.wq_a,
            self.q_norm,
            self.wq_b,
            self.wkv,
            self.kv_norm,
            rope.cos,
            rope.sin,
            self.n_heads,
            self.eps,
        )
        assert tokens.rows is not None
        cache = self.kv.entries()
        cache.index_copy_(0, tokens.rows, kv.to(cache.dtype))

        index_q = index_weights = None
        if self.compressor is not None:
            self.compressor(x, metadata, rope)
            if self.indexer is not None:
                self.indexer.compressor(x, metadata, rope)
                index_q, index_weights = self.indexer.queries(x, qr, rope)

        out = self._attend(q, metadata, index_q, index_weights)
        return _project(out, rope.cos, rope.sin, self.wo_a, self.wo_b, x.dtype)

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
        # A power of two: it cuts a padded step into chunks of padded sizes.
        chunk = 1 << (max(1, _ATTENTION_CHUNK // per_token).bit_length() - 1)

        window_rows, window_bias = tokens.window_rows, tokens.window_bias
        assert window_rows is not None and window_bias is not None
        outs = []
        for start in range(0, q.shape[0], chunk):
            part = slice(start, start + chunk)
            bias = window_bias[part]
            n = bias.shape[0]
            keys = kv_cache.index_select(0, window_rows[part].reshape(-1))
            keys = keys.view(n, self.window, self.head_dim)
            compressed = block_bias = None
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
                compressed = compressed.view(n, kept, self.head_dim)
            outs.append(
                _sparse_attn(
                    q if n == q.shape[0] else q[part],
                    keys,
                    bias,
                    compressed,
                    block_bias,
                    self.attn_sink,
                    self.scale,
                )
            )
        return outs[0] if len(outs) == 1 else torch.cat(outs)


def _limited_moe(x, ids, weights, w13, w2, s13, s2, b13, b2, *consts):
    first, last, total, limit = consts
    return torch.ops.hpu.mixture_of_experts.bias_mxfp4_fused_weights(
        x,
        ids.to(torch.int32),
        weights.to(x.dtype),
        w13,
        w2,
        b13,
        b2,
        s13,
        s2,
        block_size=32,
        permuted_weights=True,
        experts_min=first,
        experts_max=last,
        is_fp4=True,
        total_experts=total,
        alpha=1.0,
        limit=limit,
    ).float()


def _unlimited_moe(x, ids, weights, w13, w2, s13, s2, *consts):
    first, last, total, _ = consts
    return torch.ops.hpu.mixture_of_experts.mxfp4_fused_weights(
        x,
        ids.to(torch.int32),
        weights.to(x.dtype),
        w13,
        w2,
        s13,
        s2,
        block_size=32,
        permuted_weights=True,
        activation="silu",
        experts_min=first,
        experts_max=last,
        is_fp4=True,
        total_experts=total,
    ).float()


_native_moe: dict[bool, Callable] = {}


def _native_experts(x, ids, weights, lists, first, last, total, limit):
    """The card's MoE kernel on packed MXFP4 experts; it runs compiled only.

    Its time is that of reading the experts the tokens use, some 0.1 ms each
    whatever the tokens; told how many experts the ids span, it is a third
    faster on a few tokens.

    Args:
        lists: A tensor per expert for the gate and up rows, the down rows and
            the scales of both; with two more, the biases of both, the kernel
            that has a SwiGLU limit is used.
    """
    limited = len(lists) == 6
    if limited not in _native_moe:
        # Without graphs a step is not padded and its token count is free.
        moe = _limited_moe if limited else _unlimited_moe
        _native_moe[limited] = _compile(moe, static=GRAPHS)
    return _native_moe[limited](x, ids, weights, *lists, first, last, total, limit)


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
    """The routed experts of this rank, packed as in the checkpoint.

    On the card they run in its MoE kernel, by ``VLLM_HPU_DEEPSEEK_V4_MOE``:

    - ``unlimited``, the default: the kernel without a SwiGLU limit. Where a
      value passes the limit the output of that expert is off by its own size
      and more.
    - ``limited``: the kernel with a limit. It computes
      silu(min(gate, limit)) * (clamp(up + bias, -limit, limit) + 1); with a
      bias of -1 on up that is clamp(up, 1 - limit, 1 + limit), the model's
      limit on the gate and one off it on up, and within a tenth of the
      model's output where the limit acts. A third more time at a hundred
      tokens.
    - ``torch``: plain torch with the model's limit, as off the card.

    Scored on text against ``torch``, the two kernels are equally far, and as
    far as from each other: there the limit acts too seldom to tell them apart.
    """

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
        mode = os.environ.get("VLLM_HPU_DEEPSEEK_V4_MOE", "unlimited")
        self.native = current_platform.device_type == "hpu" and mode != "torch"
        # The limited kernel takes gate (w1) and up (w3) rows by turns, the
        # unlimited one all gate rows and then all up rows.
        self.interleaved = mode == "limited"
        # Two float4_e2m1 codes per byte, one E8M0 scale per 32 codes of a row.
        self.w13 = _param(local, 2 * inter, hidden // 2, dtype=torch.uint8)
        self.w13_scale = _param(local, 2 * inter, hidden // 32, dtype=torch.uint8)
        self.w2 = _param(local, hidden, inter // 2, dtype=torch.uint8)
        self.w2_scale = _param(local, hidden, inter // 32, dtype=torch.uint8)
        if self.native and self.interleaved:
            # A row per expert: the kernel takes no tensor twice.
            device = self.w13.device
            bias = torch.zeros(local, 2 * inter, dtype=gemm_dtype)
            bias[:, 1::2] = -1.0
            self.register_buffer("w13_bias", bias.to(device), persistent=False)
            bias = torch.zeros(local, hidden, dtype=gemm_dtype, device=device)
            self.register_buffer("w2_bias", bias, persistent=False)
        self.pieces_loaded = 0
        self._waiting: dict[tuple[int, str], tuple[str, torch.Tensor]] = {}
        self._lists: tuple | None = None

    def load_expert(
        self, expert: int, matrix: str, kind: str, tensor: torch.Tensor
    ) -> None:
        local = expert - self.first
        if not 0 <= local < self.num_local:
            return
        self.pieces_loaded += 1
        piece = tensor.view(torch.uint8)
        if matrix == "w2":
            (self.w2 if kind == "weight" else self.w2_scale).data[local].copy_(piece)
            return
        target = (self.w13 if kind == "weight" else self.w13_scale).data[local]
        if not self.interleaved:
            half = slice(0, self.inter) if matrix == "w1" else slice(self.inter, None)
            target[half].copy_(piece)
            return
        # w1 and w3 come apart: the first waits on the host for the other.
        waiting = self._waiting.pop((local, kind), None)
        if waiting is None:
            self._waiting[local, kind] = (matrix, piece)
            return
        gate, up = (waiting[1], piece) if waiting[0] == "w1" else (piece, waiting[1])
        target.copy_(torch.stack([gate, up], dim=1).flatten(0, 1))

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
            stacked = [self.w13, self.w2, self.w13_scale, self.w2_scale]
            if self.interleaved:
                stacked += [self.w13_bias, self.w2_bias]
            self._lists = tuple(
                tuple(p[i] for i in range(self.num_local)) for p in stacked
            )
        return _native_experts(
            x,
            indices,
            weights,
            self._lists,
            self.first,
            self.first + self.num_local - 1,
            self.num_experts,
            self.limit,
        )

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
            if self.interleaved:
                gate, up = gate_up[:, 0::2], gate_up[:, 1::2]
            else:
                gate, up = gate_up[:, : self.inter], gate_up[:, self.inter :]
            hidden = ops.swiglu(gate, up, self.limit)
            hidden = (hidden * weights[rows, slots, None]).to(x.dtype)
            w2 = ops.dequant_fp4_weight(self.w2[local], self.w2_scale[local])
            out.index_add_(
                0, rows, F.linear(hidden.to(gemm), w2.to(gemm)).to(x.dtype).float()
            )
        return out


_route = _graph(ops.route)


@_graph
def _add_shared(
    x: torch.Tensor,
    routed: torch.Tensor,
    w1: torch.Tensor,
    w3: torch.Tensor,
    w2: torch.Tensor,
    limit: float,
) -> torch.Tensor:
    """The shared expert on top of the routed experts' output."""
    gemm = w1.dtype
    xq = ops.fp8_quant_dequant(x).to(gemm)
    hidden = ops.swiglu(
        F.linear(xq, w1).to(x.dtype), F.linear(xq, w3).to(x.dtype), limit
    )
    shared = ops.fp8_linear(hidden.to(x.dtype), w2)
    return (routed + shared.float()).to(x.dtype)


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
        weights, indices = _route(
            self.gate(x),
            self.gate.e_score_correction_bias,
            hash_experts,
            self.top_k,
            self.route_scale,
        )
        routed = self.experts(x, indices, weights)
        if self.tp_size > 1:
            routed = tensor_model_parallel_all_reduce(routed)
        return _add_shared(
            x, routed, self.shared_w1, self.shared_w3, self.shared_w2, self.limit
        )


def _hc_enter(h, fn, scale, base, norm, rms_eps, eps, iters):
    """hc_pre and the norm of the half of a layer it leads into."""
    x, post, comb = ops.hc_pre(h, fn, scale, base, rms_eps, eps, iters)
    return ops.rms_norm(x, norm, rms_eps), post, comb


def _hc_cross(x, h, post, comb, fn, scale, base, norm, rms_eps, eps, iters):
    """hc_post of the attention and hc_pre of the MoE."""
    h = ops.hc_post(x, h, post, comb)
    return (h, *_hc_enter(h, fn, scale, base, norm, rms_eps, eps, iters))


_enter = _graph(_hc_enter)
_cross = _graph(_hc_cross)
_leave = _graph(ops.hc_post)
_head = _graph(ops.hc_head)


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

    def forward(
        self, h: torch.Tensor, input_ids: torch.Tensor | None, rope: _Rope
    ) -> torch.Tensor:
        consts = (self.eps, self.hc_eps, self.hc_iters)
        x, post, comb = _enter(
            h,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            self.attn_norm,
            *consts,
        )
        x = self.attn(x, rope)
        h, x, post, comb = _cross(
            x,
            h,
            post,
            comb,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.ffn_norm,
            *consts,
        )
        x = self.ffn(x, input_ids)
        return _leave(x, h, post, comb)


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
        align_rows([m for m in self.modules() if isinstance(m, DeepseekV4HPUCache)])

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
        # A step padded to a static size: the rows added here attend to
        # nothing and are dropped before the final norm, which the PoC plugin
        # reads.
        num_tokens = inputs_embeds.shape[0]
        context = get_forward_context().attn_metadata
        extra = 0
        if isinstance(context, dict):
            metadata = cast(dict[str, DeepseekV4HPUMetadata], context)
            extra = next(iter(metadata.values())).padded_tokens - num_tokens
        if extra:
            inputs_embeds = F.pad(inputs_embeds, (0, 0, 0, extra))
            positions = F.pad(positions, (0, extra))
            if input_ids is not None:
                input_ids = F.pad(input_ids, (0, extra))
        h = inputs_embeds.unsqueeze(1).repeat(1, self.config.hc_mult, 1)
        ropes = {}
        for compressed, name in ((False, "window"), (True, "compress")):
            cos, sin = getattr(self, f"{name}_cos"), getattr(self, f"{name}_sin")
            ropes[compressed] = _Rope(
                cos, sin, cos.index_select(0, positions), sin.index_select(0, positions)
            )
        for layer in self.layers:
            h = layer(h, input_ids, ropes[layer.attn.ratio > 0])
        h = _head(
            h,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.config.rms_norm_eps,
            self.config.hc_eps,
        )
        return self.norm(h[:num_tokens] if extra else h)


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
            assert not experts._waiting, (index, sorted(experts._waiting))
            for name in ("w13", "w13_scale", "w2", "w2_scale"):
                loaded.add(f"model.layers.{index}.ffn.experts.{name}")
        return loaded
