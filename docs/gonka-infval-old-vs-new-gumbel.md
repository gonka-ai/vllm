# Inference validation: old (logprob distance) vs new (public-tape Gumbel draft rejection)

Both validators scored on identical commits of **DeepSeek-V4-Flash-0731 fp8**, 1×B300,
gonka vLLM fork `0.28.0.dev0+glm53.gonka.sampler1` (`mlnode:3.1.0-vllm-0.28.0`),
40 prompts × 192 tokens. Harness: `benchmark-results/infval-oldnew-b300-2026-10-01/`.

- The old validator is blind to a laundered commit: if the executor reports the logprobs of
  one teacher-forced pass of the real model, it passes every attack measured here (0 %).
- The new validator does not read reported logprobs. It flags any commit from a model that
  disagrees with the target on a large share of steps, and a margin gate lifts detection of
  close perturbations (logit-space fp4: 25 % → 62 %).
- Open: close quantization (production NVFP4/AWQ) was not measured, and thresholds were set
  as in-sample maxima over 40 honest sequences; on a second honest set the margin gate
  flagged 1 of 40.

## 1. The two schemes

**Old** (devshard v5.0.2, threshold 0.900 for DeepSeek):

1. The executor's devshardd rewrites the request: `logprobs=true`, `top_logprobs=5`,
   `min_tokens≥64`, `seed=int32(inference_id)` unless the client set one,
   `logprobs_mode=processed_logprobs`. It samples normally and stores token ids with
   per-position top-5 logprobs; `ResponseHash = sha256(stored)`, signed.
2. Each validator samples the inference with `p = rate · own_slots / (16 − executor_slots)`,
   `rate` = 10 %, downloads the payload and replays the prompt with `enforced_tokens` =
   committed tokens, so the validator follows the same sequence.
3. Per position, over the validator's top-5: `|a − b| / (1e-6 + |a| + |b|) / 2`, where `b`
   is the reported logprob of the same token or, if absent, `m1 − (m2 − m1)` from the two
   lowest reported values; averaged over the 5 tokens.
   `similarity = 1 − Σ_t / max(100, N)`, pass iff `similarity > 0.900`.

**New** (draft rejection):

1. A public seed fixes a tape `u_{t,i}` for every position `t` and token id `i`,
   `g = −log(−log u)`.
2. The executor commits `x_t = argmax_i(logit_i + g_{t,i})`: the same marginal as sampling at
   temperature 1, reproducible from the tape. Only token ids, length and stop reason are
   committed.
3. The validator teacher-forces the target on the commit and computes
   `z_t = argmax_i(logit_i + g_{t,i})`. Evidence per step: `x_t ≠ z_t` and the margin
   `m_t = s(z_t) − s(x_t) ≥ 0`, `s = logit + g` in fp32.
4. Scores: `R_K = Σ 1{x_t ≠ z_t}`; margin-gated `R_K` counts only `m_t > ε`;
   `Σm = Σ m_t`. A commit is challenged above a cutoff calibrated on honest runs.

## 2. Harness

- **Launch:** `LLM(model, kv_cache_dtype="fp8", tokenizer_mode="deepseek_v4",
  enforce_eager=True, logits_processors=[...])` with `VLLM_ENABLE_V1_MULTIPROCESSING=0`, so
  the per-request processor runs in-process. Load takes 88 s with the TileLang/Triton caches
  warm and about 600 s cold.
- **Forcing:** a v1 `AdapterLogitsProcessor` gets per-request parameters through
  `SamplingParams.extra_args` and, at each step, sees the full logits `L` on the committed
  history. It records what it needs and returns `+1e4` at the chosen token and `−1e4`
  elsewhere. Requests run with `temperature=0`, `ignore_eos=True`, `max_tokens=192`.
- **Tape:** `torch.Generator("cuda").manual_seed(seed·1000003 + pos)`, `u = rand(V)` clamped
  at 1e-12, with `seed = 100000 + prompt index` and `pos = prompt_len + t`. This tape comes
  from the device RNG: reproducible here because the executor and the validator share one
  GPU, but not specified across GPUs or torch versions.
- **Recorded per step:** the token `x`, the target winner `z`, the margin `m`, and the top-16
  of `log_softmax` for both the validator's target logits and the executor's own logits
  (the executor's report).
- **Logit-space faces** (one model load): the executor commits from `face(L)` — temperature
  `L/τ`, mixture `log((1−a)·softmax(L) + a/V)`, top-k truncation, or per-vector QDQ of the
  logits (int-k, fp4 e2m1 in groups of 16).
- **Weight-space faces** (separate executor load): every MoE expert tensor, 172 fused tensors
  in all, goes through QDQ in place along the input dimension with groups of 16. The QDQ is
  symmetric int-k round/clamp or fp4 e2m1 levels {0, ½, 1, 1½, 2, 3, 4, 6}·amax/6. It is
  computed in bf16 chunks and written back to fp8. The τ / top-p sweep applies to the
  quantized model's logits before the tape. A clean-target load then validates all commits.
- **Validator passes:** all faces in one batched pass; honest commits also one request at a
  time (`honest_solo`). Both passes run on one GPU, so batch composition is the only source
  of honest disagreement.
- **Old scoring:** a port of `positionDistance`/`CompareLogits` on the recorded top-5, with the
  executor reporting either its own logprobs (*naive*) or the target's (*laundered*). It
  models the metric, not the deployed pipeline: no stream-chunk parsing, 4xx autopass or
  token-count check.
- **New cutoffs:** maxima over the 40 `honest_solo` sequences — `c` = 21 / 27 for `R_K`,
  `ε` = 3.54 / 4.36 for the margin (logit / weight run).
- **Cost:** executor 40 × 192 steps in about 31 s; batched validator 160–440 sequences in
  41–55 s; solo validator about 23 s per sequence.

## 3. Results

| face | median `R_K` /192 | old, naive | old, laundered | new, `R_K > c` | new, margin-gated | median `Σm` |
|---|---|---|---|---|---|---|
| honest_solo (calibration) | 10 | 0 % | 0 % | 0 % | 0 % | 3 |
| honest, second set (weight run) | 8.5 | 0 % | 0 % | 0 % | **2.5 % (1/40)** | 3 |
| logit fp4 QDQ | 14 | 0 % | 0 % | 25 % | 62 % | 23 |
| temperature 0.7 | 9 | 98 % | 0 % | 8 % | 0 % | 4.5 |
| top-k 1 | 23 | 100 % | 0 % | 55 % | 90 % | 32 |
| mixture 1 % / 10 % / 100 % noise | 1 / 18 / 136 | 0 / 0 / 100 % | 0 % | 0 / 30 / 100 % | 42 / 100 / 100 % | 17 / 138 / 439 |
| weight fp4 e2m1, group 16 | 163 | 12–22 % | 0 % | 100 % | 100 % | 2.9k |
| weight int4 / int2, group 16 | 162 / 156 | 0 / 22 % | 0 % | 100 % | 100 % | 1007 / 935 |
| weight int4, τ 0.5 / 0.7 / 1.0 / 1.5, top-p 0.9 | 108 / 125 / 162 / 172 / 154 | 95 / 52 / 0 / 60 / 2 % | 0 % | 100 % | 100 % | 514–1103 |

Logit faces use the logit-run cutoffs and weight faces the weight-run cutoffs. Old naive for
weight fp4 came out at 12 % and 22 % in two validator passes over the same commits.

**Old validator.** Laundering costs one teacher-forced pass and removes every detection. The
naive numbers depend on how the executor reports and are unstable between passes.

**Weight quantization as run here is coarse:** 81–85 % of steps disagree with the target,
17–21 % of tokens repeat the previous one (honest 0 %), and the distinct-token share falls
from 0.53 to 0.17–0.30. The new validator flags it trivially. The close regime is represented
by logit-space fp4, at 7.9 % of steps against 5.1 % honest: plain `R_K` catches 25 %, the
margin gate 62 %. Production NVFP4/AWQ checkpoints were not measured.

**Honest floor.** Batch composition alone flips 5 % of steps (`R_K` 1–27 of 192), almost all
near-ties (margin p50 0.26, p90 1.14). One honest sequence in the second set reached a margin
of 4.46 against `ε` = 4.36. Cross-hardware pairs would raise the floor further, and an
in-sample maximum over 40 sequences is not a deployable threshold.

**Tuning.** On the int4 model the best temperature for the attacker, τ = 0.5, lowers `R_K`
from 162 to 108 and stays flagged. This matches the coupling bound
`Pr[x_t ≠ z_t] ≥ TV(P_t, Q_t)`: a fixed `Q` cannot drop below the honest floor without
becoming the target. Tuning was not tested on close quantization.

**Temperature only.** τ = 0.7 changes logprob values, so naive old catches it, but its token
flips are near-ties. The new validator does not see it.

**Speculative decoding with a cheap drafter** is a commit from a cheaper model with laundered
logprobs, which the weight and mixture rows cover. A separate drafter was not run: setting
`RoutedExperts.top_k = 1` after load left the output unchanged.

## 4. EOS and length

| attack | what the tape-based score does | evidence |
|---|---|---|
| suppressed or fake EOS | reads the committed token at that step, so any token that differs from the target's winner (EOS or not) counts as a disagreement | by construction, not measured |
| truncation without EOS | does not see it: the honest prefix gives `R_K` median 1/30, max 7/30 | measured |

The old validator catches truncation through its `min_tokens` floor. The new one has to bind
length and stop reason in the commit.

## 5. What the new scheme still needs

1. A margin-based score, calibrated on a large honest corpus across hardware, TP and batch
   composition, with a target false-challenge rate rather than an in-sample maximum.
2. A tape addressed by (seed, position, token id) through a hash or counter-based PRF. The
   per-position device RNG used here holds only on one GPU.
3. Length, stop reason and `min_tokens` bound in the commit.
4. Batch-invariant kernels or a fixed numerical profile, to lower the honest floor.
5. A measurement of close quantization on real calibrated checkpoints (NVFP4, AWQ).

## 6. Reproduce

Run inside `mlnode:3.1.0-vllm-0.28.0` with the flags of §2:

```
python3 harness4.py --n 40                                   # logit faces, records4.jsonl
python3 harness5_exec.py --qbits 4 --qmode fp4 --group 16 --face g16fp4 --out commits_g16fp4.jsonl
python3 harness5_exec.py --qbits 4 --face int4_t05 --tau 0.5 --out commits_int4_t05.jsonl
python3 harness5_val.py --commits "commits_*.jsonl" --out records.jsonl
python3 analyze3.py records.jsonl
```
