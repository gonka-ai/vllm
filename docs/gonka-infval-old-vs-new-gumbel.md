# Inference validation: old (logprob-distance) vs new (public-tape Gumbel draft-rejection)

Empirical comparison of two inference-validation schemes on the real mainnet model
**DeepSeek-V4-Flash-0731 (fp8)**, 1×B300, gonka vLLM fork (`mlnode:3.1.0-vllm-0.28.0`).
Harness and raw data: `benchmark-results/infval-oldnew-b300-2026-10-01/`.

## 1. Old validation — step by step (devshard v5.0.2, verified against code)

1. DAPI/chain rewrites the request: `logprobs=true, top_logprobs=5, min_tokens>=64,
   seed=inference_id, logprobs_mode=processed`.
2. The executor generates by **ordinary sampling** (its own RNG + seed); commits
   `token_ids` + per-position top-5 logprobs + usage; `ResponseHash=sha256`, signed.
3. Sampling for validation: each validator with probability ~`rate*slots`.
4. Downloads the executor payload (committed tokens + **reported** top-5 logprobs).
5. Re-runs P with `enforced_tokens`=committed tokens (**forces the same sequence**),
   `stream=false`, reads P's top-5 logprobs at each forced position.
6. `positionDistance = mean_top5 |a-b|/(1e-6+|a|+|b|)/2`;
   `similarity = 1 - sum_t positionDistance_t / max(100, N)`.
7. Verdict: pass iff `similarity > threshold` (DeepSeek 0.900). Structural shortcuts:
   4xx -> autopass, token-count inflation, min_tokens floor, length/token mismatch.
8. Vote (weighted by slots) -> aggregate -> settlement.

**Essence:** it forces the executor's tokens and only checks whether the logprob
*values* are plausible. It never asks whether P would have *chosen* those tokens.
-> launderable (report genuine P logprobs via one teacher-forced P pass), blind to
trajectory substitution.

## 2. New validation — step by step (Gumbel draft-rejection)

1. The job fixes `id, target P, manifest` and a **public seed S_gen** (unbiasable
   beacon) -> a public tape `u_{t,i}` over all (position, token),
   `g_{t,i} = -log(-log u_{t,i})`.
2. The executor samples via **shared Gumbel**: `x_t = argmax_i(log P_i(.|history) + g_{t,i})`
   (same marginal distribution, but reproducible given the tape); commits
   `token_ids + length + stop_reason` (**logprobs are not trusted**).
3. After commit, an **independent** `S_aud` picks a hidden depth/positions `K`.
4. The validator teacher-forces P over the committed history; `z_t = argmax_i(log P_i + g_{t,i})`.
5. `R_K = sum_{t<=K} 1{x_t != z_t}`; with a **margin-gate** (don't count numerical
   near-ties) and/or cumulative `Sum_m` (adjusted-score deficit).
6. Verdict: challenge iff `R_K`/`Sum_m` > a calibrated cutoff.

**Essence:** it checks that the committed tokens are P's *choice* under shared
randomness. Reported logprobs are not used -> cannot be laundered; any cheaper/different
model diverges, bounded below by `Pr[x_t != z_t] >= TV(P_t, Q_t)` (shared-Gumbel coupling).

## 3. Measurement method

One pass of P: a custom `AdapterLogitsProcessor` sees P's full logits, builds an
attack `L^Q=face(L^P)`, forces `x_t=argmax(L^Q+g_t)`, and records the rows; a
clean-P validator (solo) produces `z_t`, P's top-16 and the margin. The executor
runs **batched** (prod-like), the validator runs **solo** (a real numerical twin ->
honest FPR from batch composition). Both validators are scored offline on the **same
commits** (`analyze3.py`). 40 prompts x 192 tokens; the NEW cutoff is calibrated on
`honest_solo` -> 0% false positives on the sample.

## 4. Results (against a smart = laundering fraudster; 0% FPR)

| attack | OLD naive | **OLD laundering** | NEW margin-gated |
|---|---|---|---|
| honest_solo (FPR) | 0% | 0% | 0% |
| real weight-quant fp4-e2m1 (NVFP4-style, group-16) | 12% | **0%** | **100%** |
| real weight-quant int4 / int2 (group-16) | 0% / 22% | **0%** | **100%** |
| fraudulent spec-decode (any cheap drafter, laundered) | — | **0%** | **100%** |
| distribution tuning tau in {0.5,0.7,1.0,1.5}, top-p 0.9 (on int4) | 0-95% | **0%** | **100%** (min at tau=0.5) |
| mix 1% / 10% noise | 0% / 0% | **0%** | 42% / **100%** |
| temperature-only (0.7) | 98% | **0%** | 0% |

Continuous `Sum_m` (honest ~= 3): fp4 2894, int4 1005, int2 935 — monotonically far
from the honest floor.

### Why NEW catches real quant 100% while OLD does not
Under a fixed tape, a tiny logit difference (x43 MoE layers) deterministically flips
the tape-winner, and over 192 autoregressive steps on a drifting history the
divergence saturates (`R_K ~= 160/192`, with large margin). NEW sees this; OLD
compares logprob *values*, which per position stay close to P under quant -> blind;
under laundering -> exactly 0%.

### Distribution tuning does not evade NEW
`Pr[x != z] >= TV(P,Q)`: tuning a **fixed** Q (tau/top-k/top-p) can only push the
disagreement below the honest floor (~0.05) if `TV(P,Q) < 0.05` (i.e. Q ~= P = P's
work). Empirically the best tuning (tau=0.5) still yields 108/192 = 100% detection.
By contrast, OLD *is* fooled by tuning.

### The only evasion of NEW is adaptive, seed-aware fraud
See `g_t`, then pick the token (A9-A11/A15, outside the theorem). But picking a token
P would accept as its winner = computing `argmax(log P + g_t)` = P's work. Mitigated
by a hidden post-commit audit depth, an unbiasable beacon, rate limits and an
economic penalty.

## 5. EOS / length / padding

- **EOS suppression / padding / running past P's stop point**: caught by R_K — the
  committed non-EOS token != P's EOS winner at that position (or pad != winner).
- **Early stop with a fake EOS** (EOS where P would not stop): caught — EOS != `z_t`.
- **Early truncation without EOS** (honest prefix + claim "length limit"): R_K over the
  short prefix ~= 0 -> **not caught by the R_K metric**; closed by the commitment
  binding (`length + stop_reason + min_tokens`) — in OLD this is caught structurally by
  the min_tokens floor. Takeaway: NEW must bind length/stop into the commit (present in
  the proposal, but outside R_K). Measured: honest R_K over a 30-token prefix is ~1/30.

## 6. What to add to the new scheme
1. **margin-gate + continuous Sum_m** (proposal A2/A1) — the core: 0% FPR, recovers
   subtle attacks.
2. **bind length/stop_reason/min_tokens** into the commit (see section 5).
3. **batch-invariant kernels / fixed numerical profile** — lower the honest floor,
   tighten the margin threshold.
4. activation fingerprinting is **not needed** — the public tape + token-trajectory
   check already close quant and spec-fraud.

## 7. Reproduce
```
# real weight-quant (expert QDQ) + clean-P validator:
python3 harness5_exec.py --qbits 4 --qmode fp4 --group 16 --face g16fp4 --out commits_g16fp4.jsonl
python3 harness5_val.py  --commits "commits_*.jsonl" --out records.jsonl
python3 analyze3.py records.jsonl
```
Run inside the `mlnode:3.1.0-vllm-0.28.0` container, with
`VLLM_ENABLE_V1_MULTIPROCESSING=0`, `kv_cache_dtype=fp8`,
`tokenizer_mode=deepseek_v4`, `enforce_eager=True`.
