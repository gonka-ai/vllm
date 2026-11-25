# Gonka MLNode Integration: Decode PoC

Extends [`poc-integration.md`](poc-integration.md) (prefill PoC). Reuse its RNG primitives ([`gpu_random.py`](https://github.com/gonka-ai/gonka-vllm-plugins/blob/v0.2.1/src/gonka_poc/poc/gpu_random.py)) and API.

Reference implementation: [`gonka-poc` v0.2.1](https://github.com/gonka-ai/gonka-vllm-plugins/tree/v0.2.1/src/gonka_poc) on [vLLM 0.30](https://github.com/gonka-ai/vllm/tree/92aa868134deabbeec758f42508fe0719b8f2e33).

The upgrade to v0.2.17 keeps prefill as the regular scheme and runs decode as a trial in the first confirmation PoC of each epoch. Governance switches the regular scheme later. A node must support both schemes.

## Overview

A nonce is a synthetic prompt of `seq_len` embeddings and then `N = max_tokens` decode steps on the same KV cache (256 and 256). No token is sampled. Each step snaps the final hidden state to one of 16 codebook points; the point seeds the next step.

```
prompt embeddings -> prefill -> k_0 -> decode 1 -> k_1 -> ... -> decode N -> k_N
```

The artifact is `[k_0, ..., k_N]`, 257 integers in `0..15`.

## Forward pass

Notation: `seed(s)` is the first 4 bytes of SHA-256, big-endian (`_seed_from_string`). `mm(key, seed)` is `_murmur3_32`. `normal(seed, n)` is `_normal`. `H` is the hidden size. `base = seed(f"{block_hash}_{public_key}_nonce{nonce}")`.

```
step_seed(i, salt) = mm((k_{i-1} * 0x9E3779B1 + i * 0x85EBCA77 + salt) mod 2^32, base)
```

| | Prefill (step 0) | Decode step `i` = 1..N |
| --- | --- | --- |
| position | `0..seq_len-1` | `seq_len + i - 1` |
| input embedding | `normal(base, seq_len * H)` as `[seq_len, H]`, same as prefill PoC | `normal(step_seed(i, 0x0D), H)` |
| pick seed | `seed(f"{block_hash}_{public_key}_nonce_{nonce}_pick_256_decode0")` | `step_seed(i, 0x91)` |

`poc_stronger_rng` does not apply to decode. Code: [`decode_random.py`](https://github.com/gonka-ai/gonka-vllm-plugins/blob/v0.2.1/src/gonka_poc/poc/decode_random.py#L57-L123).

**Householder.** Same as prefill PoC: after every decoder layer `l`, vector `normal(seed(f"{block_hash}_layer_{l}_householder"), H)` normalized to unit length. If the layer returns `(hidden, residual)`, reflect both. If it keeps several copies per token, reflect each. Code: [`native.py`](https://github.com/gonka-ai/gonka-vllm-plugins/blob/v0.2.1/src/gonka_poc/mixed/native.py#L87-L126).

**MoE routing.** Replace the router logits ([`_forced_logits`](https://github.com/gonka-ai/gonka-vllm-plugins/blob/v0.2.1/src/gonka_poc/poc/decode_random.py#L147-L171)). The model's own scoring and weighting then run on them unchanged.

```
start  = mm(step, seed(f"{block_hash}_n{nonce}_route_layer_{j}")) mod n_experts
expert (start + r) mod n_experts  ->  100 + top_k - r     for r = 0..top_k-1
all other experts                 ->  -10000
```

- `j` counts the seeded MoE layers from 0. Dense layers and layers routed by token id are not counted.
- `step` is 0 for every prompt token, then 1..N.

**Layers routed by token id** (DeepSeek-V4 `tid2eid`). Keep their router logits. Use these token ids:

- prompt position `p`: `mm(p, seed(f"{block_hash}_{public_key}_nonce{nonce}_input_ids")) mod vocab_size`
- decode step `i`: `mm(0, step_seed(i, 0x57)) mod vocab_size`

## Snap

Input: the last position's hidden state after the final norm, float32.

1. Take the 256 coordinates `d` with the smallest `mm(d, pick_seed)`. Order: the 255 smallest by ascending `d`, then the 256th.
2. Normalize to unit length: `q`.
3. `scores = codebook @ q`, `k = argmax(scores)`.

The order in step 1 is what `torch.topk(..., sorted=False)` returns on CUDA. It matters: with fully sorted indices, `k` matched in 13% of random inputs.

Codebook: [`sphere_codebook.pt`](https://github.com/gonka-ai/gonka-vllm-plugins/blob/v0.2.1/src/gonka_poc/poc/sphere_codebook.pt), `[16, 256]` float32. SHA-256 of the raw float32 data: `2201d1093583f43f328dd4643ddf1f34737f6da8c1956395c43fd093e357e34f`. Load this file, do not rebuild it. Code: [`sphere.py`](https://github.com/gonka-ai/gonka-vllm-plugins/blob/v0.2.1/src/gonka_poc/poc/sphere.py).

## Validation

Run the same forward, seeding step `i` with the prover's point `c_{i-1}` instead of `k_{i-1}`, so a disagreement does not change later steps ([`runtime.py`](https://github.com/gonka-ai/gonka-vllm-plugins/blob/v0.2.1/src/gonka_poc/mixed/runtime.py#L689-L749), [`validation.py`](https://github.com/gonka-ai/gonka-vllm-plugins/blob/v0.2.1/src/gonka_poc/poc/validation.py#L149-L238)).

1. Step margin, where `k_i != c_i`: `max(scores) - scores[c_i]`. A `c_i` outside `0..15` gives 2.0.
2. Nonce distance: the largest margin. The nonce mismatches if it exceeds `dist_threshold`.
3. Verdict: one-sided binomial test of the mismatch count against `p_mismatch`. Fraud if `p_value < fraud_threshold`.

A nonce with a non-finite hidden state is dropped. When validating, report it in `n_excluded`; the API node then abstains.

Thresholds come in the request's `stat_test`. Chain values in v0.2.17:

| Model | `dist_threshold` | `p_mismatch` | `fraud_threshold` |
| --- | --- | --- | --- |
| MiniMax-M2.7 | 0.0395 | 0.1 | 0.01 |
| DeepSeek-V4-Flash-0731 | 0.027 | 0.1 | 0.01 |
| GLM-5.3-Flash | 0.017 | 0.1 | 0.01 |

For 200 nonces, fraud starts at 31 mismatches.

## API

Same endpoints as prefill PoC. Differences:

- `params`: `"decode": true, "seq_len": 256, "max_tokens": 256`. Without `decode`, run prefill PoC.
- artifact: `{"nonce": n, "vector_b64": "", "k_points_steps": [k_0, ..., k_N]}` in `/generated` callbacks, `/generate` responses and `validation.artifacts`.
- `/validated` callback: `n_total`, `n_mismatch`, `mismatch_nonces`, `p_value`, `fraud_detected`, `n_excluded`, `excluded_nonces`.

> **vLLM:** each nonce is an engine request ([`scheduler.py`](https://github.com/gonka-ai/vllm/blob/92aa868134deabbeec758f42508fe0719b8f2e33/vllm/v1/core/sched/scheduler.py#L620)). Its prompt is prefilled in one step and never read from the prefix cache. Chat gets 503 during mining and is paused during validation.

## Check

1. Compare seeds, pick indices and routing with the plugin. They are integer-only and must match exactly.
2. Run [`validate.py --scheme decode`](https://github.com/gonka-ai/gonka/tree/upgrade-v0.2.17/mlnode/packages/benchmarks/scripts/poc_validation) against the golden references. Pass the chain thresholds: the references carry looser ones.

    ```bash
    python3 validate.py --mlnode-url "$URL" --model MiniMaxAI/MiniMax-M2.7 --scheme decode \
        --threshold 0.0395 --p-mismatch 0.1 --fraud-threshold 0.01
    ```

    For another model, record a reference on an NVIDIA MLNode with `make_artifact.py --scheme decode --num-nonces 200`.
3. Validate custom-node artifacts on NVIDIA GPUs.
4. Confirm that a quantized or pruned model is rejected.
