# Gonka Integration Tests

Tests for Gonka-specific features in the vLLM fork: Proof of Computation (PoC), inference validation with enforced tokens, chat priority gating, grammar graceful degradation, and logprobs mode behavior.

---

## Test Files

### Unit Tests (no server required)

| File | What it tests |
|------|---------------|
| `test_chat_priority_gating.py` | Chat priority gating logic with mocked AsyncLLM and FastAPI components |
| `test_grammar_graceful_degradation.py` | Grammar FSM graceful degradation with mocked xgrammar backend |

### Live Tests (require running vLLM server)

| File | What it tests |
|------|---------------|
| `test_live_chat_priority.py` | PoC activates → chat rejected 503 → PoC stops → chat resumes; long inference aborted by PoC, engine survives |
| `test_live_grammar_degradation.py` | Structured output + enforced tokens replay; corrupted tokens don't crash engine; distance2 with grammar |
| `test_live_inference.py` | Basic chat, logprobs, structured output, temperature, seed determinism, max_tokens, top_logprobs |
| `test_live_validation.py` | Enforced token replay across params (temperature, seed, grammar, prompt lengths); corrupted tokens; text match; distance2 |
| `test_live_poc.py` | PoC artifact generation, self-validation L2 < 0.2, batch generation, different block hashes, server-side validation |
| `test_live_logprobs_mode.py` | `processed_logprobs` vs `raw_logprobs` behavior; -9999 clamping; per-request override; validation distance with matching/mismatched modes |

### Shared Helpers

| File | Role |
|------|------|
| `live_conftest.py` | Shared helpers for live tests: `chat_request`, `build_enforced_tokens`, `extract_result`, `token_distance2`, `distance2` (matching the production validation pipeline) |

---

## Prerequisites

### Unit Tests

No prerequisites. Run directly:

```bash
cd /path/to/vllm
python3 -m pytest tests/gonka/test_chat_priority_gating.py tests/gonka/test_grammar_graceful_degradation.py -v --noconftest
```

### Live Tests

A running vLLM server with a loaded model. The tests read `VLLM_TEST_MODEL` and `VLLM_TEST_PORT` env vars, defaulting to `Qwen/Qwen2.5-0.5B-Instruct` on port `18199`.

#### Quick start with Docker (small model)

```bash
# Build the image (from repo root)
docker build -f Dockerfile.quick -t vllm:test .

# Run the server
docker run -d --rm \
  --gpus '"device=0"' \
  --entrypoint bash \
  --name vllm-test \
  -p 18199:18199 \
  --shm-size=4g \
  vllm:test -c \
  "python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --dtype float16 \
    --host 0.0.0.0 \
    --port 18199 \
    --max-model-len 4096 \
    --enforce-eager \
    --gpu-memory-utilization 0.4"

# Wait for server to be ready
while ! curl -s http://127.0.0.1:18199/health; do sleep 2; done
```

#### Running with Qwen3-235B-A22B (FP8)

The same tests work against the production-scale MoE model. This validates PoC, inference, and validation on the actual deployment target. Requires 4× A100-80GB (or H100/H200/B200) and the model pre-downloaded in the HF cache.

```bash
docker build -f Dockerfile.quick -t vllm:test .

docker run -d \
  --gpus '"device=4,5,6,7"' \
  --ipc=host \
  --entrypoint bash \
  --name vllm-235b-test \
  -p 18200:18200 \
  --shm-size=16g \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -v /path/to/huggingface/cache:/root/.cache/huggingface \
  vllm:test -c \
  "python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
    --dtype auto \
    --host 0.0.0.0 \
    --port 18200 \
    --tensor-parallel-size 4 \
    --max-model-len 4096 \
    --served-model-name Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"

while ! curl -s http://127.0.0.1:18200/health; do sleep 5; done
```

> **A100 FP8 note:** A100 GPUs lack native FP8 compute, so vLLM uses Marlin weight-only FP8 decompression. This works correctly but is slower than native FP8 on H100/H200/B200.

Run tests against the 235B server:

```bash
VLLM_TEST_MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8" \
VLLM_TEST_PORT=18200 \
python3 -m pytest tests/gonka/test_live_*.py -v -s --noconftest
```

#### Run live tests

```bash
python3 -m pytest tests/gonka/test_live_*.py -v -s --noconftest
```

If the server is not running, live tests are **automatically skipped** (not failed).

#### Run specific test suites

```bash
# Only inference + validation
python3 -m pytest tests/gonka/test_live_inference.py tests/gonka/test_live_validation.py -v -s --noconftest

# Only PoC
python3 -m pytest tests/gonka/test_live_poc.py -v -s --noconftest

# Only chat priority gating
python3 -m pytest tests/gonka/test_live_chat_priority.py -v -s --noconftest
```

Tests can be run in **any order**. Each test file cleans up after itself (PoC is stopped before and after every test via fixtures).

---

## Distance Metrics

The validation tests use the same `distance2` metric as the production validation pipeline (`benchmarks/src/validation/utils.py`):

- **`token_distance2`**: Per-position normalized logprob distance. Iterates validation-side tokens, builds fallback from inference-side sorted logprobs.
- **`distance2`**: Sequence-level mean of `token_distance2` over all positions, normalized by `max(100, n_positions) * n_logprobs`.

Expected values for honest self-validation (same server, same model):
- **distance2 < 0.05** for all inference validation (with or without grammar)

For PoC self-validation:
- **L2 distance < 0.2** for individual pairs (test_02)
- **Mean L2 < 0.1, max L2 < 0.3** across 20 pairs (test_06 — wider max to avoid flakes from float16 variance)
