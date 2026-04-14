# Deterministic Sampling & Validation for Decentralized Inference

## Broader Context

This system enables **verifiable inference** for decentralized LLM networks. An "executor" node runs inference, and a "validator" node independently verifies the executor's work was honest. The key challenge: how does the validator confirm the executor actually ran the model and sampled tokens correctly, without re-running the full inference?

The solution has three components:
1. **Deterministic sampling** -- the executor samples tokens using a reproducible SHA256-based RNG seeded from the prompt + user seed
2. **Reproducible decimal processing** -- both executor and validator derive integer weights from logprobs using Python's `decimal` library, guaranteeing bit-identical results across any machine
3. **Two-check validation** -- the validator verifies honesty using logprob distance (Check 1) and exact sampling replay (Check 2)

## Architecture Overview

### Executor Pipeline (per token position)

```
GPU (model + logit processing):
  ┌─────────────────────────────────────────────────────────────────┐
  │  1. Model.forward(input_ids) → raw logits [batch, vocab_size]   │
  │     (standard transformer forward pass)                         │
  │                              │                                  │
  │                              ▼                                  │
  │  2. logits.to(float32)                                          │
  │                              │                                  │
  │                              ▼                                  │
  │  3. Apply penalties + bias FIRST (reordered for post-penalty     │
  │     logprobs):                                                  │
  │     ├── apply_all_penalties()                                   │
  │     │   (repetition_penalty, frequency_penalty, presence_penalty│
  │     │    modifies logits of tokens in prompt/output history)    │
  │     ├── apply_logit_bias() (additive bias, after penalties)     │
  │     │                                                           │
  │     ├── apply allowed_token_ids mask  (hard mask → -inf)        │
  │     ├── apply bad_words mask          (hard mask → -inf)        │
  │     └── apply min_tokens mask         (hard mask EOS → -inf)    │
  │                              │                                  │
  │                              ▼                                  │
  │  4. log_softmax(logits) → post-penalty logprobs [batch, vocab]  │
  │     (these logprobs reflect penalties + hard masks)              │
  │                              │                                  │
  │                              ▼                                  │
  │  5. torch.topk(logprobs, K) → top-K token IDs + logprob values │
  │     K = max_num_logprobs from request (e.g. 5)                  │
  │                              │                                  │
  │                              ▼                                  │
  │  6. Transfer to CPU: K ints + K floats (~40 bytes for K=5)      │
  └──────────────────────────────┼──────────────────────────────────┘
                                 │
CPU (decimal pipeline + sampling):
  ┌──────────────────────────────┼──────────────────────────────────┐
  │                              ▼                                  │
  │  7. Convert logprob floats to Decimal: Decimal(repr(float))      │
  │     e.g. -0.0500... → Decimal("-0.05000000074505806")           │
  │     (repr gives shortest round-trip-safe string for Decimal)    │
  │                              │                                  │
  │                              ▼                                  │
  │  8. Temperature scaling: logprob / T  (Decimal division)        │
  │                              │                                  │
  │                              ▼                                  │
  │  9. Softmax: exp(scaled) / sum(exp(scaled))                     │
  │     (Decimal exp with log-sum-exp stability shift)              │
  │                              │                                  │
  │                              ▼                                  │
  │ 10. Filtering (all in Decimal arithmetic):                      │
  │     ├── top_k: keep K highest-prob tokens                       │
  │     ├── top_p: cumulative prob cutoff                           │
  │     └── min_p: threshold = max_prob * min_p                     │
  │                              │                                  │
  │                              ▼                                  │
  │ 11. Re-normalize after filtering                                │
  │                              │                                  │
  │                              ▼                                  │
  │ 12. Quantize: int_weight = round(prob * 2^16)                   │
  │     Fix total to exactly 2^16 (residual → max-weight token)     │
  │                              │                                  │
  │                              ▼                                  │
  │ 13. sample_categorical_weights(weights, SHA256_RNG)             │
  │     → sampled token ID                                          │
  └─────────────────────────────────────────────────────────────────┘

  14. Report in API response:
      - post-penalty logprobs as JSON float values (standard format)
      - sampled token ID
      - seed (for RNG reconstruction)
```

### Validator Pipeline

```
Validator receives executor's artifact:
  ┌──────────────────────────────────────────────────────────────────────┐
  │  Artifact from executor (per token position):                        │
  │    • logprobs: {"791": -0.050..., "578": -3.200..., ...}            │
  │      (post-penalty logprob floats, keyed by token ID string)         │
  │    • token: "791" (sampled token ID)                                 │
  │    • seed: 42 (user seed for RNG reconstruction)                     │
  │    • params: temperature=0.7, top_p=0.9, top_k=5, min_p=...        │
  └──────────────────────────┬───────────────────────────────────────────┘
                             │
               ┌─────────────┴─────────────┐
               ▼                           ▼
  ┌─────────────────────────┐  ┌─────────────────────────────────────────┐
  │       CHECK 1            │  │              CHECK 2                    │
  │  (logprob distance)      │  │         (sampling replay)               │
  │  detects model           │  │   detects sampling manipulation         │
  │  substitution            │  │              EXACT                      │
  └────────────┬─────────────┘  └──────────────────┬──────────────────────┘
               │                                   │
               ▼                                   ▼

CHECK 1 — Logprob Distance (tolerant, detects wrong model):
  ┌───────────────────────────────────────────────────────────────────────┐
  │                                                                       │
  │  GPU (validator re-runs model):                                       │
  │  ┌─────────────────────────────────────────────────────────────────┐  │
  │  │  1. Model.forward(same prompt) → raw logits                     │  │
  │  │                          │                                      │  │
  │  │                          ▼                                      │  │
  │  │  2. Apply same penalties + logit_bias + hard masks (reordered,   │  │
  │  │     same as executor)                                           │  │
  │  │                          │                                      │  │
  │  │                          ▼                                      │  │
  │  │  3. log_softmax → validator's post-penalty logprobs             │  │
  │  │                          │                                      │  │
  │  │                          ▼                                      │  │
  │  │  4. torch.topk(logprobs, K) → validator's top-K logprobs        │  │
  │  └──────────────────────────┼──────────────────────────────────────┘  │
  │                             │                                         │
  │  CPU (distance comparison): │                                         │
  │  ┌──────────────────────────┼──────────────────────────────────────┐  │
  │  │                          ▼                                      │  │
  │  │  5. Compare validator logprobs vs executor logprobs             │  │
  │  │     position_distance(validator_top_k, executor_top_k)          │  │
  │  │                          │                                      │  │
  │  │                          ▼                                      │  │
  │  │  6. distance < threshold?                                       │  │
  │  │     ├── YES: same model (small GPU float diffs OK) ──→ PASS     │  │
  │  │     └── NO:  different/cheaper model suspected    ──→ FAIL      │  │
  │  └─────────────────────────────────────────────────────────────────┘  │
  │                                                                       │
  └───────────────────────────────────────────────────────────────────────┘

CHECK 2 — Sampling Replay (exact, detects token manipulation):
  ┌───────────────────────────────────────────────────────────────────────┐
  │                                                                       │
  │  CPU (decimal pipeline — identical to executor's):                    │
  │  ┌─────────────────────────────────────────────────────────────────┐  │
  │  │  1. Parse executor's logprob floats → Decimal objects            │  │
  │  │     e.g. -0.050000000745... → Decimal(repr(f)) → exact Decimal  │  │
  │  │                          │                                      │  │
  │  │                          ▼                                      │  │
  │  │  2. Temperature scaling: logprob / T  (Decimal division)        │  │
  │  │                          │                                      │  │
  │  │                          ▼                                      │  │
  │  │  3. Softmax: exp(scaled) / sum(exp(scaled))                     │  │
  │  │     (Decimal exp with log-sum-exp stability shift)              │  │
  │  │                          │                                      │  │
  │  │                          ▼                                      │  │
  │  │  4. Filtering (all in Decimal arithmetic):                      │  │
  │  │     ├── top_k: keep K highest-prob tokens                       │  │
  │  │     ├── top_p: cumulative prob cutoff                           │  │
  │  │     └── min_p: threshold = max_prob * min_p                     │  │
  │  │                          │                                      │  │
  │  │                          ▼                                      │  │
  │  │  5. Re-normalize after filtering                                │  │
  │  │                          │                                      │  │
  │  │                          ▼                                      │  │
  │  │  6. Quantize: int_weight = round(prob * 2^16)                   │  │
  │  │     Fix total to exactly 2^16 (residual → max-weight token)     │  │
  │  │     ⚡ Bit-identical to executor's weights (Decimal guarantee)   │  │
  │  └──────────────────────────┼──────────────────────────────────────┘  │
  │                             │                                         │
  │  CPU (SHA256 sampling):     │                                         │
  │  ┌──────────────────────────┼──────────────────────────────────────┐  │
  │  │                          ▼                                      │  │
  │  │  7. Reconstruct RNG: seed_str = f"{seed}|{prompt_token_ids}"    │  │
  │  │     Create Sha256CounterRNG(seed_str), advance to same position │  │
  │  │                          │                                      │  │
  │  │                          ▼                                      │  │
  │  │  8. sample_categorical_weights(weights, rng) → replayed token   │  │
  │  │                          │                                      │  │
  │  │                          ▼                                      │  │
  │  │  9. replayed_token == executor's reported token?                 │  │
  │  │     ├── YES: sampling was honest           ──→ PASS             │  │
  │  │     └── NO:  token was manipulated         ──→ FRAUD            │  │
  │  │              (ZERO tolerance — any mismatch = fraud)             │  │
  │  └─────────────────────────────────────────────────────────────────┘  │
  │                                                                       │
  └───────────────────────────────────────────────────────────────────────┘

  Combined verdict:
    ┌─────────────┬──────────────┬─────────────────────────────────────┐
    │  Check 1    │   Check 2    │  Verdict                            │
    ├─────────────┼──────────────┼─────────────────────────────────────┤
    │  PASS       │   PASS       │  ✓ Honest inference                 │
    │  PASS       │   FAIL       │  ✗ Sampling manipulation (fraud)    │
    │  FAIL       │   PASS       │  ✗ Model substitution (fraud)       │
    │  FAIL       │   FAIL       │  ✗ Both model + sampling (fraud)    │
    └─────────────┴──────────────┴─────────────────────────────────────┘
```

### Why This Works

The key insight: if both sides run the same deterministic function on the same inputs, they get the same outputs. Python's `decimal` module (backed by `libmpdec`, implementing IEEE 754-2008 decimal floating-point) guarantees that all operations -- including `exp()` -- are **correctly rounded** to the working precision. This means identical results on any machine, any OS, any CPU architecture, as long as both run CPython.

By moving the precision-sensitive operations (temperature, softmax, quantization) from GPU float32 to CPU `decimal`, we eliminate every floating-point reproducibility issue and remove the need to transmit integer weights entirely. The weights are derived, not reported -- making weight tampering impossible by construction.

### Why Penalties Are Applied First (Reordering)

In the standard vLLM sampler, the processing order is:
1. allowed_token_ids mask, 2. bad_words mask, 3. non-argmax-invariant processors
(min_tokens, logit_bias), 4. penalties.

For the decimal pipeline, we need post-penalty logprobs as input. To get them cleanly,
we reorder to: **penalties → logit_bias → hard masks → log_softmax**. This reordering
is safe because:

- **Hard masks** (allowed_token_ids, bad_words, min_tokens) set tokens to `-inf`. Whether
  you penalize before or after masking to `-inf`, the result is the same: masked tokens
  stay at `-inf`, unmasked tokens get the same penalty. **Order doesn't matter.**
- **logit_bias** is additive and doesn't commute with the multiplicative repetition
  penalty. In standard vLLM, logit_bias is applied before penalties; in deterministic
  mode, penalties are applied first, then logit_bias. This means the numerical result
  differs slightly from standard vLLM when both logit_bias and repetition_penalty are
  active on the same tokens. However, this is NOT a verification issue — both executor
  and validator use the same reordered pipeline, so they agree exactly. The semantic
  difference is acceptable because deterministic mode is a distinct operating mode.

The full GPU processing order in deterministic mode:
1. `apply_all_penalties()` — repetition, frequency, presence penalties
2. `apply_logit_bias()` — additive bias to specific token logits
3. Hard masks — allowed_token_ids, bad_words, min_tokens (→ `-inf`)
4. `log_softmax` → post-penalty logprobs

The reordering is ONLY applied when `VLLM_DETERMINISTIC_SAMPLING=1`. The standard
(non-deterministic) sampler path is unchanged.

### Performance

The decimal pipeline adds ~18µs per token position (with 5 logprobs, precision=10). This is negligible compared to model inference latency (~30-50ms per token). The SHA256 sampling is CPU-only by design (see `sample_categorical_weights()` in `deterministic_utils.py`), so no GPU-CPU transfer beyond the top-K logprobs is needed.

Benchmark results (see `scripts/benchmark_decimal_vs_native.py`):
- Native float64: ~5.6 µs/position
- Decimal (precision=10): ~18.3 µs/position
- Slowdown: ~3.3x (entirely dominated by `Decimal.exp()`)
- For 100-token generation: ~1.8ms total overhead

## Two-Check Validation Scheme

### Check 1: Logprob Distance (detects model substitution)
- Validator runs the same model on the same prompt with the same penalties
- Compares its post-penalty logprobs with executor's post-penalty logprobs
- Small differences expected (GPU float precision, different hardware)
- Large differences indicate the executor used a different/cheaper model
- Tolerance: configurable distance threshold
- Also implicitly verifies the token set: if the executor's top-K tokens don't match
  the validator's, the logprob distance will be large

### Check 2: Sampling Replay (detects sampling manipulation) -- EXACT
- Validator takes executor's reported logprobs (JSON floats, converted to Decimal via `Decimal(repr(f))`) + sampling params
- Runs the decimal pipeline: temperature -> softmax -> filtering -> quantize
- Produces **bit-identical** integer weights (guaranteed by `decimal` determinism)
- Samples using SHA256 RNG with the same seed -> must produce the identical token
- **Zero tolerance** -- any mismatch is fraud
- This replaces the old design where the executor reported weights alongside logprobs

### Why the Old Weight-Reporting Design Was Eliminated

In the previous design, the executor reported integer weights alongside logprobs. The
validator had to: (a) verify those weights were consistent with the logprobs (Check 2),
then (b) replay sampling from the reported weights (Check 3). This had two problems:

1. **Float precision edge cases**: Reconstructing weights from logprobs using native
   floats could produce different results on different machines (~13 identified failure
   modes including `exp()` implementation differences, `total_weight` modular arithmetic
   divergence, top_p boundary effects, etc.)

2. **Unnecessary trust surface**: The executor could fabricate weights that happened to
   produce a desired token, independent of the actual logprobs. Check 2 caught this
   only with tolerance, creating a gap.

With decimal, weights are **derived** from logprobs deterministically. There is nothing
to report, nothing to compare, and no tolerance needed. The validator independently
computes the exact same weights from the same logprobs.

## Validation via Chat Completion

### How the Validator Invokes Validation

The validator performs validation by sending a standard **chat completion request** to its
own vLLM instance, with two additional fields from the executor's artifact:

1. **Enforced tokens** -- the executor's sampled token sequence (already supported by the
   enforced-token feature in `vllm/validation.py`)
2. **Executor's logprobs** -- the post-penalty logprobs for each token position, attached
   to each `EnforcedToken` as a `logprobs: Dict[str, float]` field (keyed by token ID
   string, values are JSON floats)

The `EnforcedToken` data model is extended:
```python
@dataclass
class EnforcedToken:
    token: str
    top_tokens: List[str]
    logprobs: Optional[Dict[str, float]] = None  # executor's post-penalty logprobs
    # ...existing fields...
```

### Execution Order: Check 2 BEFORE Inference, Check 1 AFTER

The two checks run at different points in the validation pipeline:

```
Validator receives artifact (enforced tokens + logprobs + seed + params)
                            │
                            ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │  CHECK 2 — Sampling Replay (BEFORE inference, pure CPU)             │
  │                                                                      │
  │  For each token position:                                            │
  │    1. Take executor's logprobs (from enforced token data)            │
  │    2. Run decimal pipeline: Decimal(repr(float)) → temperature →     │
  │       softmax → filtering → quantize to int weights                  │
  │    3. Reconstruct SHA256 RNG from seed + prompt token IDs            │
  │    4. sample_categorical_weights(weights, rng) → replayed token      │
  │    5. replayed_token == executor's reported token? → PASS/FAIL       │
  │                                                                      │
  │  If ALL positions pass → deterministic_sampling_valid = true         │
  │  If ANY position fails → deterministic_sampling_valid = false (fraud)│
  │                                                                      │
  │  This check is cheap (pure CPU, ~18µs/position). Run it FIRST to    │
  │  catch sampling fraud before burning GPU time on inference.           │
  └──────────────────────────┬───────────────────────────────────────────┘
                             │
                             ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │  MODEL INFERENCE (GPU) — with enforced tokens                        │
  │                                                                      │
  │  Standard chat completion with enforced tokens. The model generates   │
  │  the same token sequence as the executor (tokens are enforced). At   │
  │  each position, the validator's vLLM produces its own post-penalty   │
  │  logprobs using the same reordered pipeline.                         │
  └──────────────────────────┬───────────────────────────────────────────┘
                             │
                             ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │  CHECK 1 — Logprob Distance (AFTER inference)                        │
  │                                                                      │
  │  For each token position:                                            │
  │    1. Take validator's post-penalty logprobs (from model inference)   │
  │    2. Take executor's logprobs (from enforced token data)            │
  │    3. position_distance(validator_top_k, executor_top_k) → float     │
  │                                                                      │
  │  Returns per-position distance values. The caller (external          │
  │  validator logic) decides the threshold for pass/fail.                │
  └──────────────────────────┬───────────────────────────────────────────┘
                             │
                             ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │  RESPONSE — Standard chat completion + validation fields             │
  │                                                                      │
  │  The response is a standard ChatCompletion response with two         │
  │  additional fields:                                                   │
  │    • deterministic_sampling_valid: bool                              │
  │        true if Check 2 passed for ALL positions                      │
  │    • distances: List[float]                                          │
  │        per-position logprob distance values from Check 1             │
  │        (one float per token position in the generated sequence)       │
  └──────────────────────────────────────────────────────────────────────┘
```

### Why Check 2 Runs First

Check 2 (sampling replay) is pure CPU computation — no model inference needed. It takes
the executor's logprobs and sampling params, runs the decimal pipeline, and verifies each
token was sampled correctly. This costs ~18µs per token position (~1.8ms for 100 tokens).

If Check 2 detects fraud (sampling manipulation), there may be no need to run the
expensive GPU inference for Check 1. Running Check 2 first is an optimization: catch
cheap-to-detect fraud before burning GPU time.

Check 1 (logprob distance) requires the model to actually run inference with the same
prompt and enforced tokens, so it necessarily runs after model inference completes.

### Response Format

The validation response extends the standard `ChatCompletion` response:
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "choices": [{ "message": { "content": "The answer is 4.", ... }, ... }],
  "usage": { ... },
  "deterministic_sampling_valid": true,
  "distances": [0.0012, 0.0008, 0.0015, 0.0003, ...]
}
```

- `deterministic_sampling_valid` (bool): `true` if Check 2 (sampling replay) passed for
  every token position. `false` means at least one token was not sampled correctly from
  the reported logprobs — this is fraud with zero tolerance.
- `distances` (List[float]): per-position logprob distance from Check 1. One value per
  token in the generated sequence. The threshold for pass/fail is applied by the external
  validator logic (not by vLLM itself), since the threshold is a network policy decision.

These fields are ONLY present when the request includes enforced tokens with logprobs
(i.e., this is a validation request). For normal chat completions, the response format
is unchanged.

### Validation Logic: Two Separate Files

The validation logic is implemented in two standalone Python files, imported by the
serving layer:

1. **`vllm/validation_sampling.py`** — Check 2 (sampling replay):
   - `verify_sampling_from_logprobs(logprobs, seed_str, temperature, top_p, top_k, min_p, reported_token) -> bool`
   - Derives integer weights via `logprobs_to_weights()` from `deterministic_utils.py`
   - Replays SHA256 sampling, compares against reported token
   - Zero tolerance: returns `True` (honest) or `False` (fraud)
   - Pure Python, no torch dependency

2. **`vllm/validation_distance.py`** — Check 1 (logprob distance):
   - `position_distance(validator_logprobs, executor_logprobs) -> float`
   - `compute_distances(validator_positions, executor_positions) -> List[float]`
   - Logprob distance calculation per position
   - Pure Python, no torch dependency

Both files are imported in `serving_chat.py` where the validation pipeline is
orchestrated:
```python
from vllm.validation_sampling import verify_sampling_from_logprobs
from vllm.validation_distance import compute_distances
```

## Key Design Decisions

### Decimal Library for Reproducible Float Operations

Python's `decimal` module with a fixed context (precision=10, ROUND_HALF_EVEN) is used
for all precision-sensitive operations: temperature division, `exp()`, softmax
normalization, and quantization. Precision=10 provides ~3 guard digits beyond float32's
~7 significant digits, which is more than sufficient.

The decimal context is set once globally:
```python
from decimal import Decimal, getcontext, ROUND_HALF_EVEN
getcontext().prec = 10
getcontext().rounding = ROUND_HALF_EVEN
```

### Logprobs Transmitted as JSON Floats

Logprobs are transmitted as standard JSON float values in the API response (e.g.,
`-0.05000000074505806`), not as strings. This keeps the artifact compact and compatible
with the standard OpenAI response format.

Both executor and validator convert these floats to `Decimal` objects using the same
deterministic conversion:
```python
def float_to_decimal(f: float) -> Decimal:
    return Decimal(repr(f))
```

This is safe because:
- Python's `repr(float)` produces the shortest string that uniquely identifies the
  float64 value (guaranteed round-trip fidelity since Python 3.1)
- Python's `json` module serializes floats with full float64 precision (uses `repr`
  internally) and `json.loads` recovers the exact same float64
- Both sides start with the same float64 → same `repr()` → same `Decimal`

**Requirement**: logprob values must survive JSON round-trip at full float64 precision.
Any intermediary (proxy, logger) that truncates float precision will break Check 2.
In practice, all modern JSON libraries preserve doubles.

### Fixed total_weight Normalization

After quantizing probabilities to integer weights (`round(prob * 2^16)`), the weights
may not sum to exactly 2^16 due to independent rounding. To guarantee `total_weight` is
identical on both sides (critical for the SHA256 modular arithmetic), the rounding
residual is assigned to the token with the highest weight, with ties broken by token ID:

```python
SCALE = 2 ** 16
residual = SCALE - sum(weights.values())
max_tid = max(token_ids, key=lambda t: (weights[t], t))
weights[max_tid] += residual
```

This ensures `total_weight == 2^16` on every machine.

### Token Ordering

All operations that iterate over tokens use a fixed order: **sorted by token ID string**.
This eliminates any ambiguity in accumulation order for softmax sums, cumulative sums
for top_p, etc.

### Seed Derivation (no per-token seed needed)
The RNG seed is derived from `{user_seed}|{prompt_token_ids}` (see `gpu_model_runner.py`).
The RNG is counter-based (SHA256), so each token generation advances the counter.
The validator can reconstruct the seed by tokenizing the prompt.

### Integer Weights as Sampling Input
The executor samples FROM the quantized integer weights, not from float probabilities.
The decimal softmax is only used to derive the integers. Once quantized, the SHA256
sampling operates entirely on integers, which is trivially reproducible.
See `sample_categorical_weights()` in `deterministic_utils.py`.

### Top-K Clamping
When `VLLM_DETERMINISTIC_SAMPLING=1` and logprobs are requested, the sampler restricts
to the top-K tokens (K = `max_num_logprobs`). This ensures the API response logprobs
fully describe the sampling distribution. Without this, the sampler might pick from
tokens not visible in the response, making verification impossible.

### Token IDs Everywhere
The `_get_decoded_token` in `serving_engine.py` always returns `str(token_id)` (e.g. `"9707"`),
never decoded text. This keeps logprob keys and token fields in the same namespace.

### Temperature = 0 (Greedy) Handling
When temperature = 0, the decimal pipeline is skipped (greedy is deterministic by argmax
on GPU). The validator verifies by comparing its own argmax with the executor's token.
**Important**: `request.temperature` must use `is not None` check (not truthiness) to avoid
treating `temperature=0` as "not specified" (Python falsy-zero trap).

### Post-Penalty Logprobs (DECIDED)

The executor reports logprobs **after** penalties and hard masks but **before** the
decimal pipeline (temperature, filtering, softmax). This is the resolved approach:

1. Penalties are applied FIRST on GPU (reordered from standard vLLM processing order)
2. `logit_bias` is applied after penalties (additive)
3. Hard masks (allowed tokens, bad words, min_tokens) are applied after logit_bias
4. `log_softmax` is computed on the processed logits → post-penalty logprobs
5. Top-K is extracted from these post-penalty logprobs
6. The decimal pipeline operates on these post-penalty logprobs

This approach was chosen because:

- The reported logprobs ARE the decimal pipeline input -- no mismatch between what's
  reported and what was sampled from
- The validator doesn't need to reproduce penalty or logit_bias application -- it
  compares its own post-penalty logprobs with the executor's in Check 1, and uses the
  executor's logprobs directly for the decimal pipeline in Check 2
- Penalty reordering is safe (hard masks commute with penalties, logit_bias ordering
  is a minor semantic difference but both sides agree — see "Why Penalties Are Applied
  First" above)
- No need to transmit penalty/logit_bias metadata or implement them in the decimal pipeline

## Tricky Moments / Known Issues

### 1. Post-Penalty Logprobs Mode -- RESOLVED

**Decision**: When `VLLM_DETERMINISTIC_SAMPLING=1`, the sampler reorders processing so
that penalties are applied first, then hard masks, then `log_softmax`. The resulting
post-penalty logprobs are used both for the decimal pipeline AND as the logprobs reported
in the API response. This overrides the standard `--logprobs-mode` behavior in
deterministic mode.

No new `--logprobs-mode` flag is needed. No penalty metadata is transmitted. The
reordering is safe for all processors — `logit_bias` is applied after penalties but
before hard masks (see "Why Penalties Are Applied First" for the full ordering).

### 2. Model Default Sampling Params
Models ship with default params (e.g. Qwen2.5: `temperature: 0.7, top_k: 20, top_p: 0.8,
repetition_penalty: 1.1`). When user doesn't specify temperature, it falls back to
model defaults via `model_config.get_diff_sampling_param()`. The serving layer's
temperature resolution uses `is not None` (not truthiness) to handle `temperature=0`.

Both executor and validator must agree on the effective sampling params. The artifact
should record the **resolved** params (after applying defaults), not just the user-specified ones.

### 3. Python Version / `libmpdec` Consistency

The `decimal` module uses `libmpdec` as backend (since CPython 3.3). All CPython versions
3.3+ produce identical results for the same precision context. In a decentralized
network, both executor and validator should run CPython (not PyPy, Jython, etc.).
The protocol should specify a minimum Python version.

### 4. Softmax Shift-Invariance

`softmax(logprob_i / T) = softmax(logit_i / T)` for any fixed token set, because
raw logprobs differ from logits by a constant (the log-sum-exp of all logits), and
softmax is shift-invariant. This means the decimal pipeline produces correct
probabilities from logprobs -- no need to recover the original logits.

This holds for post-penalty logprobs too, since penalties modify logits before
log_softmax, so the post-penalty logprobs are just `penalty_logit_i - C'` for a
different constant `C'`.

## The Decimal Pipeline (Reference Implementation)

```python
from decimal import Decimal, getcontext, ROUND_HALF_EVEN

getcontext().prec = 10
getcontext().rounding = ROUND_HALF_EVEN

SCALE = 2 ** 16

def logprobs_to_weights(
    logprobs: dict[str, float],
    temperature: float,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
) -> dict[str, int]:
    """
    Deterministic logprobs -> integer weights pipeline.
    Both executor and validator call this with identical inputs.
    Produces bit-identical results on any machine running CPython 3.3+.
    Floats are converted to Decimal via repr() for exact reproducibility.
    """
    T = Decimal(repr(temperature))
    sorted_tids = sorted(logprobs.keys())

    # Temperature scaling
    scaled = {tid: Decimal(repr(logprobs[tid])) / T for tid in sorted_tids}

    # Softmax with log-sum-exp stability
    max_val = max(scaled[tid] for tid in sorted_tids)
    exps = {tid: (scaled[tid] - max_val).exp() for tid in sorted_tids}
    total_exp = sum(exps[tid] for tid in sorted_tids)
    probs = {tid: exps[tid] / total_exp for tid in sorted_tids}

    # top_k filtering
    if top_k is not None and top_k < len(sorted_tids):
        top_k_tids = sorted(sorted_tids, key=lambda t: probs[t], reverse=True)[:top_k]
        probs = {tid: probs[tid] for tid in top_k_tids}
        sorted_tids = sorted(top_k_tids)

    # top_p filtering
    if top_p is not None:
        tp = Decimal(repr(top_p))
        sorted_by_prob = sorted(sorted_tids, key=lambda t: probs[t], reverse=True)
        cumsum = Decimal(0)
        kept = []
        for tid in sorted_by_prob:
            cumsum += probs[tid]
            kept.append(tid)
            if cumsum >= tp:
                break
        probs = {tid: probs[tid] for tid in kept}
        sorted_tids = sorted(kept)

    # min_p filtering
    if min_p is not None:
        mp = Decimal(repr(min_p))
        max_prob = max(probs[tid] for tid in sorted_tids)
        threshold = max_prob * mp
        kept = [tid for tid in sorted_tids if probs[tid] >= threshold]
        if not kept:
            kept = [max(sorted_tids, key=lambda t: probs[t])]
        probs = {tid: probs[tid] for tid in kept}
        sorted_tids = sorted(kept)

    # Re-normalize after filtering
    kept_total = sum(probs[tid] for tid in sorted_tids)
    norm_probs = {tid: probs[tid] / kept_total for tid in sorted_tids}

    # Quantize to integer weights
    D_SCALE = Decimal(SCALE)
    weights = {tid: int((norm_probs[tid] * D_SCALE).to_integral_value())
               for tid in sorted_tids}

    # Fix total to exactly SCALE (deterministic residual assignment)
    residual = SCALE - sum(weights.values())
    max_tid = max(sorted_tids, key=lambda t: (weights[t], t))
    weights[max_tid] += residual

    return weights
```

## File Inventory

Files are marked: **[EXISTS]** = already implemented on this branch,
**[MODIFY]** = exists in base vLLM and needs changes, **[CREATE]** = new file to write.

### Core Sampling
- **[EXISTS]** `vllm/v1/sample/deterministic_utils.py` -- SHA256 counter-mode RNG (`Sha256CounterRNG`), `sample_categorical_weights()`, `logprobs_to_weights()` (decimal pipeline), `decimal_sample_from_logprobs()`. Pure Python, no torch dependency. This is the only deterministic sampling file that exists; all other files below need to be modified or created.
- **[MODIFY]** `vllm/v1/sample/sampler.py` -- Production sampler. Add a deterministic branch in `forward()`: when `VLLM_DETERMINISTIC_SAMPLING=1`, reorder penalties first, compute post-penalty logprobs, extract top-K, run decimal pipeline on CPU, skip GPU temperature/top-k/top-p/softmax path.
- `vllm/v1/sample/ops/topk_topp_sampler.py` -- Standard GPU sampling path (NOT used in deterministic mode, no changes needed)
- `vllm/v1/outputs.py` -- `SamplerOutput` (no changes needed; no `deterministic_weights` field)

### API / Serving Layer
- **[MODIFY]** `vllm/entrypoints/openai/serving_chat.py` -- Orchestrate validation pipeline: run Check 2 (sampling replay) before inference, run Check 1 (distance) after inference. Import from `validation_sampling.py` and `validation_distance.py`. Add `deterministic_sampling_valid` and `distances` fields to response when validating.
- **[MODIFY]** `vllm/entrypoints/openai/serving_engine.py` -- Modify `_get_decoded_token` to always return `str(token_id)` (e.g. `"9707"`) instead of decoded text, when in deterministic mode.
- **[MODIFY]** `vllm/entrypoints/openai/protocol.py` -- Add `deterministic_sampling_valid: Optional[bool]` and `distances: Optional[List[float]]` fields to `ChatCompletion` response model.

### Validation Logic
- **[MODIFY]** `vllm/validation.py` -- Extend `EnforcedToken` with `logprobs: Optional[Dict[str, float]]` field (JSON floats, converted to Decimal via `Decimal(repr(f))`). The base version already has `EnforcedToken`/`EnforcedTokens` with basic `token`/`top_tokens` fields from the enforced-token feature.
- **[CREATE]** `vllm/validation_sampling.py` -- Check 2: `verify_sampling_from_logprobs()` derives weights via decimal pipeline from `deterministic_utils.py`, replays SHA256 sampling, zero tolerance. Pure Python, no torch dependency.
- **[CREATE]** `vllm/validation_distance.py` -- Check 1: `position_distance()`, `compute_distances()` for logprob distance calculation. Pure Python, no torch dependency.

### Worker
- **[MODIFY]** `vllm/v1/worker/gpu_model_runner.py` -- Add seed derivation: `seed_str = f"{seed}|{prompt_token_ids}"`, create `Sha256CounterRNG` and pass in `sampling_metadata`.
- **[MODIFY]** `vllm/v1/sample/metadata.py` -- Add `deterministic_rngs` field to `SamplingMetadata` to carry per-request RNG instances from the worker to the sampler.

### Environment
- **[MODIFY]** `vllm/envs.py` -- Add `VLLM_DETERMINISTIC_SAMPLING` environment variable (bool, default False).

### Tests (to be created)
- `tests/v1/sample/test_deterministic_utils.py` -- RNG reproducibility, `logprobs_to_weights` determinism, categorical sampling, full decimal pipeline tests
- **`tests/v1/validation/test_sampling_verification.py`** -- Check 2 tests (see Test Coverage section)
- **`tests/v1/validation/test_distance_calculation.py`** -- Check 1 tests (see Test Coverage section)
- `tests/v1/validation/test_validation_e2e.py` -- E2E tests against running server (needs `RUN_E2E_TESTS=1`)

## Artifact Format

Each line in a JSONL artifact:
```json
{
  "prompt": "What is 2+2?",
  "inference_result": {
    "text": "The answer is 4.",
    "results": [
      {
        "token": "791",
        "logprobs": {"791": -0.05000000074505806, "578": -3.200000047683716, "1234": -5.099999904632568, "99": -6.0, "42": -7.5}
      }
    ]
  },
  "model": {"name": "Qwen/Qwen2.5-1.5B-Instruct", "url": "http://..."},
  "request_params": {"max_tokens": 100, "temperature": 0.7, "seed": 42, "top_logprobs": 5, "top_p": 0.9}
}
```

- `token`: sampled token ID as string
- `logprobs`: post-penalty logprobs as **JSON float values** (standard format), keyed by
  token ID string. Both sides convert to Decimal via `Decimal(repr(float_value))`.
- No `sampling_weights` field -- weights are derived by the validator from logprobs
- `temperature`, `top_p`, and other float params are standard JSON numbers
- Token IDs are consistent across logprobs keys and token field
- **JSON precision requirement**: logprob float values must survive JSON round-trip at
  full float64 precision (Python's `json` module guarantees this)

## Running Tests

Test files are listed in the File Inventory and need to be created as part of the
implementation. Example commands once they exist:

```bash
# Unit tests for deterministic_utils.py (decimal pipeline, RNG):
python -m pytest tests/v1/sample/test_deterministic_utils.py -v

# Validation logic tests:
python -m pytest tests/v1/validation/ -v

# E2E tests (needs running server with VLLM_DETERMINISTIC_SAMPLING=1):
VLLM_DETERMINISTIC_SAMPLING=1 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-1.5B-Instruct --enforce-eager \
  --gpu-memory-utilization 0.7 --max-model-len 2048

# Then in another terminal:
RUN_E2E_TESTS=1 python -m pytest tests/v1/validation/test_validation_e2e.py -v
```

## Implementation Plan

Starting point: base vLLM at commit `848a7932f` with enforced-token support already
present (`vllm/validation.py`, `vllm/v1/sample/metadata.py`). The only deterministic
sampling file that exists is `vllm/v1/sample/deterministic_utils.py` (Step 1 below).

### Step 1: `deterministic_utils.py` -- DONE
`logprobs_to_weights()` and `decimal_sample_from_logprobs()` are implemented alongside
the SHA256 RNG and `sample_categorical_weights()`. Pure Python, no torch dependency.

### Step 2: Add `VLLM_DETERMINISTIC_SAMPLING` env var
- `vllm/envs.py`: add `VLLM_DETERMINISTIC_SAMPLING: bool = False` environment variable.
  This is the master switch for all deterministic sampling behavior.

### Step 3: Modify `Sampler.forward()` with deterministic branch
In `vllm/v1/sample/sampler.py`, when `deterministic_rngs` is present in
`sampling_metadata` and `VLLM_DETERMINISTIC_SAMPLING=1`:
1. Reorder: apply penalties FIRST, then hard masks (allowed tokens, bad words, min_tokens)
2. Compute `log_softmax` on processed logits -> post-penalty logprobs
3. `torch.topk(logprobs, K)` where K = `max_num_logprobs` (required in deterministic mode)
4. Transfer top-K token IDs + logprob values to CPU
5. For each request: convert to strings via `repr(float)`, call `decimal_sample_from_logprobs()`
6. Return sampled tokens + post-penalty logprobs as `logprobs_tensors`
7. The GPU temperature/top-k/top-p/softmax path (TopKTopPSampler) is NOT called

The greedy path (temperature=0) stays unchanged -- argmax on GPU.

Also modify `vllm/v1/sample/metadata.py`: add `deterministic_rngs: Optional[Dict[int, Sha256CounterRNG]]`
field to `SamplingMetadata`.

### Step 4: Wire up seed derivation in the worker
- `vllm/v1/worker/gpu_model_runner.py`: when `VLLM_DETERMINISTIC_SAMPLING=1` and request
  has a seed, derive `seed_str = f"{seed}|{prompt_token_ids}"`, create `Sha256CounterRNG`,
  and pass it into `sampling_metadata.deterministic_rngs`.

### Step 5: Update serving layer
- `serving_chat.py`: orchestrate validation pipeline — run Check 2 before inference,
  Check 1 after inference. Import from `validation_sampling.py` and `validation_distance.py`.
  Add `deterministic_sampling_valid` and `distances` fields to response when validating.
- `serving_engine.py`: modify `_get_decoded_token` to return `str(token_id)` when in
  deterministic mode.
- `protocol.py`: add `deterministic_sampling_valid: Optional[bool]` and
  `distances: Optional[List[float]]` fields to `ChatCompletion` response model.

### Step 6: Extend validation data models
- `validation.py`: extend `EnforcedToken` with `logprobs: Optional[Dict[str, float]]`
  field (JSON floats, converted to Decimal via `Decimal(repr(f))` on both sides).
  No `sampling_weights` field -- weights are derived, not reported.

### Step 7: Create validation logic (two files)
- Create `validation_sampling.py` (Check 2 — sampling replay):
  - `verify_sampling_from_logprobs(logprobs, seed_str, temperature, ...)`:
    derives integer weights via `logprobs_to_weights()`, replays SHA256 sampling,
    compares against reported token. Zero tolerance. Pure Python.
- Create `validation_distance.py` (Check 1 — logprob distance):
  - `position_distance()` / `compute_distances()`: logprob distance calculation for
    model substitution detection. Pure Python.

### Step 8: Ensure JSON float precision for sampling params
Temperature, top_p, min_p transmitted as standard JSON float values in the artifact/API
response. Both sides convert to Decimal via `Decimal(repr(float_value))`. No string
wrapping needed — Python's `json` module preserves full float64 precision.

### Step 9: Write tests
- RNG reproducibility and decimal pipeline determinism tests
- Sampling replay verification tests
- Distance calculation tests
- E2E tests against running server

## Test Coverage

Both validation checks must be covered by unit tests. Tests should be runnable without
a GPU or a running vLLM server (except E2E tests).

### Check 2 Tests (`tests/v1/validation/test_sampling_verification.py`)

These test `vllm/validation_sampling.py` — the sampling replay check.

1. **Basic replay correctness**: Given known logprobs + seed + params, verify that
   `verify_sampling_from_logprobs()` returns `True` for the correctly sampled token.
2. **Fraud detection**: Same inputs but with a wrong reported token — must return `False`.
3. **Cross-platform determinism**: Verify that the decimal pipeline produces bit-identical
   integer weights from the same float logprobs on the test machine. (This is a sanity
   check; true cross-platform testing requires CI on multiple architectures.)
4. **Temperature variations**: Test with different temperatures (0.1, 0.7, 1.0, 1.5).
5. **Filtering combinations**: Test top_k only, top_p only, min_p only, and combinations.
6. **Edge cases**:
   - Single token in logprobs (probability = 1.0, must always select it)
   - All logprobs equal (uniform distribution)
   - Very skewed distribution (one token dominates)
   - Greedy (temperature = 0) bypass — verify argmax is used, not decimal pipeline
7. **RNG advancement**: Verify that after sampling N tokens, the RNG counter is at N,
   and replaying from counter=0 with the same seed produces the same N tokens.

### Check 1 Tests (`tests/v1/validation/test_distance_calculation.py`)

These test `vllm/validation_distance.py` — the logprob distance calculation.

1. **Identical logprobs**: Distance should be 0.0 (or very near zero).
2. **Small perturbation**: Slightly different logprob values (simulating same model on
   different hardware) — distance should be small.
3. **Different model**: Very different logprob distributions — distance should be large.
4. **Disjoint token sets**: Executor and validator top-K have no tokens in common —
   distance should be maximal.
5. **Partial overlap**: Some shared tokens, some different — distance should be intermediate.
6. **Consistency with Go implementation**: If distance functions exist in the Gonka
   codebase, verify Python produces the same values for known test vectors.

### E2E Tests (`tests/v1/validation/test_validation_e2e.py`)

These require a running vLLM server with `VLLM_DETERMINISTIC_SAMPLING=1` and are gated
behind `RUN_E2E_TESTS=1`.

1. **Round-trip**: Send a chat completion with seed → get response with logprobs →
   send validation request with enforced tokens + logprobs → verify
   `deterministic_sampling_valid=true` and distances are near zero.
2. **Tampered token**: Modify one token in the artifact → verify
   `deterministic_sampling_valid=false`.
3. **Tampered logprobs**: Modify one logprob value → verify distance spike + possible
   sampling mismatch.

## Known Limitations

1. **`logit_bias` ordering differs from standard vLLM**: In deterministic mode, penalties
   are applied before `logit_bias` (reversed from standard order). This means the
   numerical result differs slightly from non-deterministic mode when both `logit_bias`
   and `repetition_penalty` are active on the same tokens. This is a minor semantic
   difference, not a verification issue — both executor and validator use the same
   reordered pipeline. See "Why Penalties Are Applied First" for details.

2. **`max_num_logprobs` is required and bounds the sampling distribution**: In deterministic
   mode, the sampler must know how many logprobs to extract (K). If the request doesn't
   specify `top_logprobs`, the server should use a reasonable default (e.g. 5). The sampled
   token will always come from within these K tokens. This means the request's `top_k`
   sampling parameter is effectively clamped to `min(top_k, max_num_logprobs)` — if the
   user sets `top_k=20` but only 5 logprobs are extracted, the decimal pipeline only sees
   5 tokens.

3. **Greedy (temperature=0) bypasses decimal pipeline**: Greedy sampling uses GPU argmax
   directly. The validator verifies by comparing argmax results. This is inherently
   deterministic (the highest logprob token) so no decimal pipeline is needed.

## Related Codebases
- Gonka validation benchmarks: `/root/gonka/mlnode/packages/benchmarks/`
  - `src/validation/data.py` -- `ValidationItem`, `Result`, `PositionResult` data models
  - `src/validation/utils.py` -- `inference()`, `validation()`, `_extract_logprobs()`, distance functions
  - `src/validation/runner.py` -- Parallel validation runner
  - `scripts/inference_same_machine.py` -- Multi-language prompt loading (`en,sp,ch,hi,ar`)
