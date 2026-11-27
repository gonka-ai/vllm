# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4's reference math (inference/model.py and inference/kernel.py of
the checkpoint) in plain torch, for a device with no TileLang, Triton or CUDA
kernel. Tokens are flat: every function takes [tokens, ...]."""

import math

import torch
import torch.nn.functional as F

FP8_MAX = 448.0
FP4_MAX = 6.0
_FP4_CODES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _pow2_ceil(v: torch.Tensor) -> torch.Tensor:
    """The smallest power of two not below v, a positive normal float32.

    By frexp, not by the bits of v viewed as int32: compiled, the card takes
    no dtype view of a tensor whose shape changes, and its log2 is not exact."""
    mantissa, exponent = torch.frexp(v)
    exponent = exponent - (mantissa == 0.5).to(exponent.dtype)
    return torch.ldexp(torch.ones_like(v), exponent)


def _fp8_round(v: torch.Tensor) -> torch.Tensor:
    """v, a float32 within the E4M3 range, rounded to that grid, a tie to even.

    Not a cast to float8_e4m3fn: the card's cast rounds a tie away from zero."""
    # Three mantissa bits; below 2**-6 the subnormals keep the step of 2**-9.
    _, exponent = torch.frexp(v.abs().clamp(min=2.0**-6))
    step = torch.ldexp(torch.ones_like(v), exponent - 4)
    return torch.round(v / step) * step


def fp8_quant_dequant(x: torch.Tensor, block: int = 128) -> torch.Tensor:
    """act_quant(scale_fmt="ue8m0"): x as the float32 value of its FP8 E4M3
    image under a power-of-two scale per ``block`` values of the last dim."""
    v = x.float().unflatten(-1, (-1, block))
    amax = v.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    scale = _pow2_ceil(amax * (1.0 / FP8_MAX))
    q = _fp8_round((v / scale).clamp(-FP8_MAX, FP8_MAX))
    return (q * scale).flatten(-2)


def fp4_quant_dequant(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """fp4_act_quant: x as the float32 value of its float4_e2m1 image under a
    power-of-two scale per ``block`` values of the last dim."""
    v = x.float().unflatten(-1, (-1, block))
    # The reference floors at 6 * 2**-126, a scale of 2**-126, which the card's
    # ldexp turns into 0 and a zero block into NaN. The image is the same
    # under either floor unless the largest value lies below 6 * 2**-120.
    amax = v.abs().amax(dim=-1, keepdim=True).clamp(min=FP4_MAX * 2.0**-120)
    scale = _pow2_ceil(amax * (1.0 / FP4_MAX))
    m = (v / scale).clamp(-FP4_MAX, FP4_MAX)
    a = m.abs()
    # The codes are 0.5 apart below 2, 1 apart below 4 and 2 apart above;
    # torch.round sends a tie to the even code, as the cast does.
    step = 0.5 * (1 + (a >= 2).to(a.dtype)) * (1 + (a >= 4).to(a.dtype))
    return (torch.sign(m) * torch.round(a / step) * step * scale).flatten(-2)


def dequant_fp8_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """An FP8 E4M3 weight [out, in] under its 128x128 E8M0 block scales, as
    float32 on the host. The values fit bfloat16 exactly."""
    out_dim, in_dim = weight.shape
    s = scale.float().repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    return weight.float() * s[:out_dim, :in_dim]


def dequant_fp4_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """A packed float4_e2m1 weight [out, in / 2] under its E8M0 scales
    [out, in / 32], as float32: two codes per byte, the low nibble first."""
    codes = torch.tensor(_FP4_CODES + tuple(-c for c in _FP4_CODES))
    codes = codes.to(device=weight.device, dtype=torch.float32)
    packed = weight.view(torch.uint8).long()
    values = torch.stack([codes[packed & 0xF], codes[packed >> 4]], dim=-1)
    # An E8M0 byte is the biased exponent of a float32.
    s = (scale.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)
    return values.flatten(-2) * s.repeat_interleave(32, dim=1)


def hadamard(n: int) -> torch.Tensor:
    """The Hadamard matrix of rotate_activation, [n, n] float32, unscaled."""
    h = torch.ones(1, 1, dtype=torch.float32)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h


def rope_table(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """precompute_freqs_cis as (cos, sin), each [seqlen, dim / 2] float32."""

    def correction_dim(num_rotations):
        return (
            dim
            * math.log(original_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low = max(math.floor(correction_dim(beta_fast)), 0)
        high = min(math.ceil(correction_dim(beta_slow)), dim - 1)
        if low == high:
            high += 0.001
        ramp = (torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)
        smooth = 1 - ramp.clamp(0, 1)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    angles = torch.outer(torch.arange(seqlen, dtype=torch.float32), freqs)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """apply_rotary_emb on x [..., dim]; cos and sin broadcast to [..., dim / 2]."""
    pairs = x.float().unflatten(-1, (-1, 2))
    a, b = pairs[..., 0], pairs[..., 1]
    if inverse:
        sin = -sin
    rotated = torch.stack([a * cos - b * sin, a * sin + b * cos], dim=-1)
    return rotated.flatten(-2).to(x.dtype)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm in float32 with a float32 weight, back in the dtype of x."""
    v = x.float()
    v = v * torch.rsqrt(v.square().mean(-1, keepdim=True) + eps)
    return (v * weight).to(x.dtype)


def unit_rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    v = x.float()
    return (v * torch.rsqrt(v.square().mean(-1, keepdim=True) + eps)).to(x.dtype)


def fp8_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """linear() on an FP8 weight held dequantized: the activation goes through
    its FP8 image, block 128, as in fp8_gemm."""
    return F.linear(fp8_quant_dequant(x).to(weight.dtype), weight).to(x.dtype)


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    bias: torch.Tensor,
    attn_sink: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """sparse_attn with the gather done by the caller.

    Args:
        q: [tokens, heads, dim].
        kv: [tokens, keys, dim], one latent per key for all heads.
        bias: [tokens, keys] float32, 0 for a key the token attends to and
            -inf for the rest.
        attn_sink: [heads], a logit that takes part in the softmax only.
    """
    scores = torch.matmul(q, kv.transpose(1, 2)).float() * scale + bias.unsqueeze(1)
    # The head count spelled out: compiled for a changing token count, the
    # card resolves a -1 here to another dim.
    heads = attn_sink.shape[0]
    sink = attn_sink.view(1, heads, 1).expand(scores.shape[0], heads, 1)
    weights = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]
    return torch.matmul(weights.to(kv.dtype), kv)


def hc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    eps: float,
    iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Block.hc_pre on residual [tokens, hc, dim].

    Returns:
        The layer input [tokens, dim], post [tokens, hc] and
        comb [tokens, hc, hc], the last two float32.
    """
    hc = residual.shape[1]
    streams = residual.float()
    flat = streams.flatten(1)
    mixes = F.linear(flat, fn) * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + rms_eps
    )
    pre = torch.sigmoid(mixes[:, :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[:, hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (mixes[:, 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).view(-1, hc, hc)
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    x = (pre.unsqueeze(-1) * streams).sum(dim=1)
    return x.to(residual.dtype), post, comb


def hc_post(
    x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
) -> torch.Tensor:
    """Block.hc_post: stream k is post[k] * x + sum_j comb[j, k] * residual[j]."""
    mixed = torch.matmul(comb.transpose(1, 2), residual.float())
    return (post.unsqueeze(-1) * x.float().unsqueeze(1) + mixed).to(x.dtype)


def hc_head(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    eps: float,
) -> torch.Tensor:
    """Block.hc_head on residual [tokens, hc, dim]."""
    streams = residual.float()
    flat = streams.flatten(1)
    mixes = F.linear(flat, fn) * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + rms_eps
    )
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + eps
    return (pre.unsqueeze(-1) * streams).sum(dim=1).to(residual.dtype)


def route(
    logits: torch.Tensor,
    gate_bias: torch.Tensor | None,
    hash_experts: torch.Tensor | None,
    top_k: int,
    route_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gate.forward after its linear, with sqrtsoftplus scores.

    Args:
        logits: [tokens, experts] float32.
        hash_experts: [tokens, top_k], the experts of a hash layer, or None.

    Returns:
        The routing weights [tokens, top_k] float32 and the expert ids.
    """
    scores = F.softplus(logits).sqrt()
    if hash_experts is not None:
        indices = hash_experts
    else:
        indices = (scores + gate_bias).topk(top_k, dim=-1).indices
    weights = scores.gather(1, indices)
    weights = weights / weights.sum(dim=-1, keepdim=True) * route_scale
    return weights, indices


def swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    gate, up = gate.float(), up.float()
    if limit > 0:
        up = up.clamp(-limit, limit)
        gate = gate.clamp(max=limit)
    return F.silu(gate) * up
