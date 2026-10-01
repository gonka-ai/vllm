# Old vs New inference validation — DeepSeek-V4-Flash fp8, 1×B300, 2026-10-01

Comparison of the **old** (devshard v5.0.2 logprob-distance, DeepSeek threshold 0.900)
and **new** (public-tape Gumbel draft-rejection) inference validators on the real
mainnet model.

## Method
One pass of target P (fp8) through the gonka vLLM fork (`mlnode:3.1.0-vllm-0.28.0`) with
a custom `AdapterLogitsProcessor`: per position it sees P's full logit vector, applies an
attack `L^Q=face(L^P)`, draws the public Gumbel tape `g_t`, and forces the commit
`x_t=argmax(L^Q+g_t)`. A clean-P validator (solo) recomputes `z_t=argmax(L^P+g_t)`.
Offline: OLD = port of positionDistance/CompareLogits; NEW = `R_K=sum 1{x!=z}` + margin-gate
+ cumulative `Sum_m`. Executor runs batched (prod-like), validator runs solo (a real
numerical twin -> honest FPR).

## Files
- `harness.py` — the 40 prompts + single-pass emulation harness.
- `harness2/3/4.py` — iterations: 3 = two-pass, 4 = + adjusted-score margin.
- `harness5_exec.py` / `harness5_val.py` — **real weight-quant**: expert QDQ (int/fp4,
  group-wise, `--qbits/--qmode/--group`) + tau/top-p sweep (`--tau/--topp`); clean-P validator.
- `analyze3.py` — the main scorer (OLD naive+laundering, NEW R_K, margin-gate, Sum_m); 0% FPR vs honest_solo.
- `records4.jsonl` — emulation: honest, temp0.7, nvfp4-emul, int8/4/3/2, mix0.01/0.1/1.0, topk1 (+honest_solo).
- `records_real.jsonl` — real per-channel int4/3/2 (broken model, contrast).
- `records_realg.jsonl` (= `records_sweep.jsonl`) — real group-16 fp4 + int4 tau sweep.
- `commits_*.jsonl` — raw executor commits per face.

## Key results (40 prompts x 192 tokens, 0% FPR)
Against a smart (laundering) fraudster:

| attack | OLD (laundering) | NEW (margin-gated) |
|---|---|---|
| honest_solo (FPR) | 0% | 0% |
| real weight-quant fp4/NVFP4-style | 0% | 100% |
| real int4/int2 weights | 0% | 100% |
| fraudulent spec-decode (any cheap drafter) | 0% | 100% |
| distribution tuning (tau in [0.5..1.5], top-p 0.9) | 0% | 100% (min at tau=0.5) |
| mix 1% noise | 0% | 42% |
| temperature-only | 0% | 0% (only shared blind spot) |

**OLD checks logprob values (trusts the report) -> launderable -> blind to token
substitution. NEW checks that the committed tokens are P's winners under the shared tape
(does not trust the report) -> not launderable.** The only evasion of NEW is adaptive
seed-aware fraud (costs P-level work, mitigated by a hidden audit depth).

## Reproduce
```
python3 analyze3.py records4.jsonl          # emulation + margin
python3 analyze3.py records_realg.jsonl     # real weight-quant + tau sweep (adjust the face list)
```
Node: 1×B300, weights at `/home/ubuntu/hf/fp8`, load ~88 s (cached). Raw records/commits/logs
are git-ignored. Full write-up: `docs/gonka-infval-old-vs-new-gumbel.md`.
