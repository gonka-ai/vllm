# Gonka MLNode Integration: Decode PoC

What an inference engine has to implement to serve as a Gonka MLNode from the v0.2.17 upgrade on, and how to check it against the reference implementation, vLLM 0.30 on NVIDIA GPUs. [`poc-integration.md`](poc-integration.md) describes the prefill scheme on vLLM 0.20. That scheme is unchanged and stays in use until the network switches to decode, so a node needs both.

## Plan

### 1. Proof of Compute

Proof of Compute measures how much model compute a node has. v0.2.17 adds the decode scheme next to prefill: the chain runs it as a trial in the first confirmation PoC of each epoch, and governance later makes it the regular scheme. Validators recompute a sample of a node's nonces and accept only small numerical differences, so most of the integration risk is in this step.

1. Add the PoC modifications to the model forward pass: seeded inputs, a reflection after every layer, seeded MoE routing, and a snap of the last hidden state to a fixed codebook after every step.
2. Experiment: produce trajectories for `(block_hash, public_key, nonces)` and validate the golden decode references for [MiniMax-M2.7](https://github.com/gonka-ai/gonka/blob/upgrade-v0.2.17/mlnode/packages/benchmarks/scripts/poc_validation/artifacts/minimaxai-minimax-m2.7-decode.json), [DeepSeek-V4-Flash](https://github.com/gonka-ai/gonka/blob/upgrade-v0.2.17/mlnode/packages/benchmarks/scripts/poc_validation/artifacts/deepseek-ai-deepseek-v4-flash-0731-decode.json) and [GLM-5.3-Flash](https://github.com/gonka-ai/gonka/blob/upgrade-v0.2.17/mlnode/packages/benchmarks/scripts/poc_validation/artifacts/zai-org-glm-5.3-flash-decode.json).

    The check is sensitive, so expect several iterations. Starting with a small model is simplest: record a reference for it on an NVIDIA MLNode with `make_artifact.py --scheme decode` (see [Commands](#commands)).

3. Implement the API for PoC generation and validation. The decode scheme uses the prefill endpoints with a few extra fields.
4. Test the full flow:
    - artifacts generated with vLLM on NVIDIA GPUs pass validation on the custom node
    - artifacts generated on the custom node pass validation on NVIDIA GPUs
    - artifacts from a quantized or pruned model fail validation on the custom node

    Hosts run these checks with the [`mlnode-validate` skill](https://github.com/gonka-ai/gonka/blob/upgrade-v0.2.17/skills/mlnode-validate/SKILL.md) and [`validate.py --scheme decode`](https://github.com/gonka-ai/gonka/tree/upgrade-v0.2.17/mlnode/packages/benchmarks/scripts/poc_validation).

### 2. Inference Validation

Gonka inference uses the OpenAI-compatible `/v1/chat/completions` API. A validator replays a sample of inferences with the executor's tokens and compares logprobs.

1. Add `enforced_tokens` replay to the sampling flow, and return logprob tokens as token ids.
2. Experiment:
    - inference results produced by the custom node are validated on NVIDIA GPUs
    - inference results produced on NVIDIA GPUs are validated by the custom node
    - inference results produced by a modified model on NVIDIA GPUs are rejected by the custom node

    Reference tests: [inference validation scripts](https://github.com/gonka-ai/gonka/tree/upgrade-v0.2.17/mlnode/packages/benchmarks/scripts/inference_validation). The validator's own scoring: [`common/validation`](https://github.com/gonka-ai/gonka/tree/main/common/validation).

3. Check `/v1/chat/completions` parameter compatibility against [`docs/chat-api/README.md`](https://github.com/gonka-ai/gonka/blob/upgrade-v0.2.17/docs/chat-api/README.md). Resolve any differences in the proxy.

---

## Details

Reference implementation: [gonka-ai/vllm `release/v0.30-decode-int`](https://github.com/gonka-ai/vllm/tree/92aa868134deabbeec758f42508fe0719b8f2e33) (engine changes: scheduler, sampler, API) and the plugin [gonka-vllm-plugins `v0.2.1`](https://github.com/gonka-ai/gonka-vllm-plugins/tree/v0.2.1/src/gonka_poc) (the PoC itself). Other engines may use different hooks and memory management.

### 1. Proof of Compute

#### Concept

A nonce is a fixed computation on the real model weights that any node can repeat: a synthetic prompt of 256 embeddings, its prefill, then 256 decode steps on the same KV cache. No token is sampled. After the prefill and after every decode step, the last hidden state is reduced to one of 16 codebook points, and that point decides the input of the next step. The proof for a nonce is its 257 points.

Why it is built this way:

- Decode is where inference spends its compute. The proof runs as ordinary requests through the same scheduler, KV cache and CUDA graphs as chat, so a node that is fast at the proof is fast at the paid work.
- Each step depends on the previous point and attends over the KV cache of all earlier steps. The steps have to run in order; none can be skipped or computed before the block is known.
- A point is a symbol that survives small numerical differences between GPUs, kernels and frameworks. Only a hidden state close to the border between two cells can flip, and then the claimed point is almost as close as the validator's own. A cheaper model (quantized, pruned) moves hidden states into other cells by a large margin.

For an integration this means:

- The seeded values (inputs, reflection vectors, chosen coordinates, expert choice, codebook) must match the reference bit for bit. They are integer hashes and float32 math.
- The model forward must stay numerically close to vLLM's. Validation tolerates a few mismatched nonces per sample: fraud is declared from 31 of 200. vLLM on different NVIDIA GPU types gives 0–8.

#### Forward Pass

The usual transformer forward pass:

```
input  = embedding(input_tokens)          <--- seq_len x hidden_size
hidden = norm(layer_n(... layer_1(input)))
logits = lm_head(hidden[-1])
```

The PoC forward pass, for each `block_hash` and nonce:

```
input  = generate_inputs(block_hash, public_key, nonce, step, previous point)
model  = layer_1 -> hh_1 -> layer_2 -> hh_2 -> ... -> layer_n -> hh_n -> norm -> snap
         each MoE router picks a seeded set of experts
```

| Change | Why | vLLM implementation |
| --- | --- | --- |
| Prompt embeddings seeded by `(block_hash, public_key, nonce)` replace the token embeddings | No text is needed, and no input exists before the block | [`generate_inputs`](https://github.com/gonka-ai/gonka-vllm-plugins/blob/v0.2.1/src/gonka_poc/poc/gpu_random.py#L132-L149), [`PoCEmbeddingWrapper`](https://github.com/gonka-ai/gonka-vllm-plugins/blob/v0.2.1/src/gonka_poc/mixed/native.py#L129-L190) |
| The input of decode step `i` is seeded by the nonce, `i` and the previous point | Chains the steps | [`generate_decode_inputs_gpu`](https://github.com/gonka-ai/gonka-vllm-plugins/blob/v0.2.1/src/gonka_poc/poc/decode_random.py#L93-L104) |
| Householder reflection of the hidden state and the residual after every decoder layer, one vector per `(block_hash, layer)` | The computation is new for every block and cannot be cached | [`PoCLayerWrapper`](https://github.com/gonka-ai/gonka-vllm-plugins/blob/v0.2.1/src/gonka_poc/mixed/native.py#L87-L126) |
| MoE router logits replaced by a seeded run of `top_k` experts | The node must hold every expert, and routing stops depending on numerical noise | [`PoCRouterWrapper`](https://github.com/gonka-ai/gonka-vllm-plugins/blob/v0.2.1/src/gonka_poc/mixed/native.py#L250-L301), [`_forced_logits`](https://github.com/gonka-ai/gonka-vllm-plugins/blob/v0.2.1/src/gonka_poc/poc/decode_random.py#L147-L171) |
| Seeded pseudo token ids for MoE layers that route by token id (DeepSeek-V4 hash layers) | A constant id would pin these layers to one expert set | [`derive_pseudo_input_ids`](https://github.com/gonka-ai/gonka-vllm-plugins/blob/v0.2.1/src/gonka_poc/poc/gpu_random.py#L189-L213), [`decode_pseudo_token_ids`](https://github.com/gonka-ai/gonka-vllm-plugins/blob/v0.2.1/src/gonka_poc/poc/decode_random.py#L174-L186) |
| Snap after the final norm | Turns the hidden state into one of 16 points | [`PoCSnapWrapper`](https://github.com/gonka-ai/gonka-vllm-plugins/blob/v0.2.1/src/gonka_poc/mixed/native.py#L193-L225), [`sphere.py`](https://github.com/gonka-ai/gonka-vllm-plugins/blob/v0.2.1/src/gonka_poc/poc/sphere.py#L169-L189) |
| No LM head and no sampling for PoC rows | No tokens are produced | [`compute_logits`](https://github.com/gonka-ai/gonka-vllm-plugins/blob/v0.2.1/src/gonka_poc/mixed/bridge.py#L282-L290) |

#### Artifact Generation

For each nonce ([`runtime.py`](https://github.com/gonka-ai/gonka-vllm-plugins/blob/v0.2.1/src/gonka_poc/mixed/runtime.py#L597-L749)):

1. Prefill the 256 prompt embeddings at positions 0..255. Snap the last position: `k_0`.
2. For `i = 1..256`: run one decode token at position `255 + i` with the input seeded by `k_{i-1}`. Snap: `k_i`.
3. Return `{"nonce": n, "vector_b64": "", "k_points_steps": [k_0, ..., k_256]}`.

Snap: take 256 of the `hidden_size` coordinates chosen by a seed, normalize them to unit length, and pick the codebook point with the highest cosine.

The seeded values, which must match exactly. `seed(s)` is the first 4 bytes of SHA-256, big-endian. `mm(key, seed)` and `normal(seed, n)` are `_murmur3_32` and `_normal` from [`gpu_random.py`](https://github.com/gonka-ai/gonka-vllm-plugins/blob/v0.2.1/src/gonka_poc/poc/gpu_random.py#L22-L129). `H` is the hidden size.

```
base               = seed(f"{block_hash}_{public_key}_nonce{nonce}")
step_seed(i, salt) = mm((k_{i-1} * 0x9E3779B1 + i * 0x85EBCA77 + salt) mod 2^32, base)

prompt embeddings  = normal(base, 256 * H), shape [256, H]
input of step i    = normal(step_seed(i, 0x0D), H)
reflection, layer l: x - 2 (x·v) v,  v = unit(normal(seed(f"{block_hash}_layer_{l}_householder"), H))

pick seed, k_0     = seed(f"{block_hash}_{public_key}_nonce_{nonce}_pick_256_decode0")
pick seed, step i  = step_seed(i, 0x91)
coordinates        = the 256 values d < H with the smallest mm(d, pick seed)
q                  = unit(hidden[coordinates]), float32
k                  = argmax(codebook @ q)

MoE gate j         : start = mm(step, seed(f"{block_hash}_n{nonce}_route_layer_{j}")) mod n_experts
                     logit of expert (start + r) mod n_experts = 100 + top_k - r, r = 0..top_k-1; others -10000
token-id layers    : prompt position p: mm(p, seed(f"{block_hash}_{public_key}_nonce{nonce}_input_ids")) mod vocab
                     step i: mm(0, step_seed(i, 0x57)) mod vocab
```

- Coordinate order: the 255 smallest by ascending `d`, then the 256th. This is what `torch.topk(sorted=False)` returns on CUDA; another order gives other points.
- Codebook: [`sphere_codebook.pt`](https://github.com/gonka-ai/gonka-vllm-plugins/blob/v0.2.1/src/gonka_poc/poc/sphere_codebook.pt), `[16, 256]` float32, SHA-256 of the data `2201d1093583f43f328dd4643ddf1f34737f6da8c1956395c43fd093e357e34f`. Load it, do not rebuild it.
- Embeddings and reflection vectors are computed in float32 and cast to the model dtype.
- `j` counts only the gates whose logits are replaced, from 0. `step` is 0 for prompt tokens. The model's own scoring, bias and top-k then run on these logits.
- Gates that route by token id (DeepSeek-V4 `tid2eid`) keep their logits and read the pseudo token ids instead.

#### Validation

The validator receives the prover's trajectory `c` and runs the same nonce, but feeds `c_{i-1}` into step `i`, so one disagreement does not change the steps after it ([`runtime.py`](https://github.com/gonka-ai/gonka-vllm-plugins/blob/v0.2.1/src/gonka_poc/mixed/runtime.py#L703-L749), [`validation.py`](https://github.com/gonka-ai/gonka-vllm-plugins/blob/v0.2.1/src/gonka_poc/poc/validation.py#L149-L238)).

1. At each step where its own `k_i` differs from `c_i`: margin = `max(scores) - scores[c_i]`. A claimed point outside 0..15 counts as 2.0.
2. The nonce distance is its largest margin. The nonce mismatches if the distance exceeds `dist_threshold` (τ).
3. A one-sided binomial test compares the number of mismatched nonces with `p_mismatch`. Fraud if the p-value is below `fraud_threshold`.

The thresholds arrive in the request's `stat_test`. v0.2.17 sets τ to 0.0395 for MiniMax-M2.7, 0.027 for DeepSeek-V4-Flash and 0.017 for GLM-5.3-Flash, with `p_mismatch` 0.1 and `fraud_threshold` 0.01. A nonce with a non-finite value is dropped. A validator that drops one reports it in `n_excluded`, and the API node then abstains from the vote.

> **vLLM execution:** each nonce is a separate engine request with a dummy prompt, a unique cache salt and priority −1 ([`async_llm.py`](https://github.com/gonka-ai/vllm/blob/92aa868134deabbeec758f42508fe0719b8f2e33/vllm/v1/engine/async_llm.py#L386-L405), [`compute_nonce_artifacts`](https://github.com/gonka-ai/gonka-vllm-plugins/blob/v0.2.1/src/gonka_poc/poc/generate_queue.py#L73-L221)). The scheduler prefills its 256 positions in one step, never from the prefix cache, then schedules one token per step ([`poc_step_tokens`](https://github.com/gonka-ai/gonka-vllm-plugins/blob/v0.2.1/src/gonka_poc/mixed/admission.py#L28-L41)). A row mask applies the PoC changes to PoC rows only, inside the captured CUDA graphs. While PoC rows are live, chat requests are paused and keep their KV; a PoC row that needs that KV preempts the lowest-priority chat request ([`scheduler.py`](https://github.com/gonka-ai/vllm/blob/92aa868134deabbeec758f42508fe0719b8f2e33/vllm/v1/core/sched/scheduler.py#L1571-L1589)).

#### Endpoints

The endpoints are those of the prefill scheme ([`poc-integration.md`](poc-integration.md#endpoints), [`routes.py`](https://github.com/gonka-ai/gonka-vllm-plugins/blob/v0.2.1/src/gonka_poc/poc/routes.py#L520-L872)). For decode:

- `params` carries `"decode": true, "seq_len": 256, "max_tokens": 256`. Without `decode`, the node runs the prefill scheme.
- `POST /api/v1/pow/init/generate` keeps up to `batch_size` nonces in flight; MLNode sends 0, which means the engine's `--max-num-seqs`. New chat requests get 503 and in-flight ones are aborted.
- `POST /api/v1/pow/generate` with `validation.artifacts` needs `k_points_steps` for every nonce. The result and the `/validated` callback add `n_excluded` and `excluded_nonces`.
- `/generated` callbacks and `/generate` responses carry `k_points_steps`.

### 2. Inference Validation

#### Validation Replay

The API node turns on `logprobs`, sets `top_logprobs` to 5, `return_token_ids` and `skip_special_tokens: false`, and a seed on every inference, then stores the response ([`request.go`](https://github.com/gonka-ai/gonka/blob/main/common/completionapi/request.go#L28-L89)).

A validator replays a sampled inference with the recorded tokens in `enforced_tokens` (token ids as strings, with the 5 top token ids per position), the same seed and `logprobs_mode`. Its engine replaces each sampled token with the recorded one without changing the logits, and returns its own logprobs. The validator then requires the same tokens and length, computes a relative distance between the two top-5 logprob sets at each position, and passes the inference if 1 − mean distance exceeds the model's `validation_threshold` on chain: 0.922 for MiniMax-M2.7, 0.900 for DeepSeek-V4-Flash, 0.951 for GLM-5.3-Flash. DeepSeek and GLM also have per-position checks ([`validation.go`](https://github.com/gonka-ai/gonka/blob/main/common/validation/validation.go#L316-L393), [`scoring.go`](https://github.com/gonka-ai/gonka/blob/main/common/validation/scoring.go#L86-L148)).

Engine side, in the vLLM 0.30 fork:

- Logprob tokens are returned as numeric token-id strings, not decoded text ([`serving.py`](https://github.com/gonka-ai/vllm/blob/92aa868134deabbeec758f42508fe0719b8f2e33/vllm/entrypoints/generate/base/serving.py#L330-L331)).
- `enforced_tokens` is a `/v1/chat/completions` field ([`protocol.py`](https://github.com/gonka-ai/vllm/blob/92aa868134deabbeec758f42508fe0719b8f2e33/vllm/entrypoints/openai/chat_completion/protocol.py#L393-L396)). It becomes `enforced_token_ids`, with EOS appended ([`serving.py`](https://github.com/gonka-ai/vllm/blob/92aa868134deabbeec758f42508fe0719b8f2e33/vllm/entrypoints/openai/chat_completion/serving.py#L333-L370)).
- The sampler computes logprobs first, then replaces the sampled token ([`sampler.py`](https://github.com/gonka-ai/vllm/blob/92aa868134deabbeec758f42508fe0719b8f2e33/vllm/v1/sample/sampler.py#L148-L157)). The enforced token is picked by output position ([`gpu_input_batch.py`](https://github.com/gonka-ai/vllm/blob/92aa868134deabbeec758f42508fe0719b8f2e33/vllm/v1/worker/gpu_input_batch.py#L860-L889)); 4.0.0-sigma3 predates this fix, gonka-ai/vllm#132.
- `logprobs_mode` (raw or processed) is set per request and must match the original inference. If absent, it is detected from the top tokens ([`validation.py`](https://github.com/gonka-ai/vllm/blob/92aa868134deabbeec758f42508fe0719b8f2e33/vllm/validation.py#L61-L84)).
- A grammar that rejects an enforced token is switched off for that request ([`backend_xgrammar.py`](https://github.com/gonka-ai/vllm/blob/92aa868134deabbeec758f42508fe0719b8f2e33/vllm/v1/structured_output/backend_xgrammar.py#L158-L186)).

---

## Commands

### Versions

| | |
| --- | --- |
| MLNode image | `ghcr.io/gonka-ai/mlnode:4.0.0-sigma3@sha256:ce449de9545b55a8c55d495eaa512beebb72cd29478d4f4b672a1c231ce89d1f` |
| MLNode image, NVIDIA driver older than 580 | `ghcr.io/gonka-ai/mlnode:4.0.0-sigma3-cu12@sha256:3e4588d1dc33f4a09a1d099cb491348cdc80849ad7bad883da184ccf055f9aca` |
| vLLM in the image | [gonka-ai/vllm](https://github.com/gonka-ai/vllm/tree/c694cd48fb04c11b608c52384a3e508b14d3c915) `release/v0.30-decode-int` at `c694cd48fb` |
| PoC plugin in the image | [gonka-ai/gonka-vllm-plugins](https://github.com/gonka-ai/gonka-vllm-plugins/tree/v0.2.1) `v0.2.1` (commit `c98fc00`, reports version 0.2.0) |
| Presets and scripts | [gonka-ai/gonka](https://github.com/gonka-ai/gonka/tree/aa3c763ab814aa1af9694de76df6fe7dabd3ef50) `upgrade-v0.2.17` at `aa3c763ab` |

### Models and GPUs

| `MODEL` | `COMMIT` | `NAME` | `TAU` | GPUs per instance: H100 / H200 / B200 / B300 |
| --- | --- | --- | --- | --- |
| `MiniMaxAI/MiniMax-M2.7` | `d494266a4affc0d2995ba1fa35c8481cbd84294b` | `minimaxm27` | 0.0395 | 4 / 2 / 2 / 1 |
| `deepseek-ai/DeepSeek-V4-Flash-0731` | `7872f01b1d1fe23eabc4c98b48bffcef5a386062` | `deepseekv4flash0731` | 0.027 | 4 / 2 / 2 / 1 |
| `zai-org/GLM-5.3-Flash` | `04c4e9e95c5da8862dced7e5056455116f83a7e0` | `glm53flash` | 0.017 | 8 / 4 / 4 / 2 |

The vLLM arguments are in `deploy/join/node-config-<NAME>-<GPU>-decode-max.json`. MLNode starts one instance per `--tensor-parallel-size` GPUs. Two additions:

- GLM: add `--max-model-len 400000`. The chain sends it, the preset does not have it.
- DeepSeek on 2×H200: start the container with `-e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`.

On another GPU, start from the nearest preset:

- `--tensor-parallel-size`: enough GPUs for the weights.
- `--max-num-seqs`: at most the KV cache size in tokens (vLLM prints it at startup) divided by 512. Set `max_cudagraph_capture_size` to the same value.
- Keep `--kv-cache-dtype fp8`. Do not use pipeline parallelism: decode stalls.

Another model runs with the same commands and its own arguments. It has no golden reference and no network thresholds: record a reference in step 5.

### Run

```bash
git clone https://github.com/gonka-ai/gonka && cd gonka && git checkout aa3c763ab
pip install requests

export IMAGE=ghcr.io/gonka-ai/mlnode:4.0.0-sigma3@sha256:ce449de9545b55a8c55d495eaa512beebb72cd29478d4f4b672a1c231ce89d1f
export MODEL=MiniMaxAI/MiniMax-M2.7 COMMIT=d494266a4affc0d2995ba1fa35c8481cbd84294b
export NAME=minimaxm27 GPU=B300 TAU=0.0395
export URL=http://127.0.0.1:8080
export CFG=deploy/join/node-config-$NAME-$GPU-decode-max.json
export POC=mlnode/packages/benchmarks/scripts/poc_validation
```

1. Start MLNode.

    ```bash
    docker run -d --name mlnode --gpus all --ipc host --ulimit nofile=65535:65535 \
        -v "$HOME/.cache:/root/.cache" -e HF_HOME=/root/.cache -p 8080:8080 \
        $IMAGE uvicorn api.app:app --host=0.0.0.0 --port=8080
    ```

2. Download the model. Repeat the second command until it returns `DOWNLOADED`.

    ```bash
    curl $URL/api/v1/models/download -H 'Content-Type: application/json' \
        -d "{\"hf_repo\":\"$MODEL\",\"hf_commit\":\"$COMMIT\"}"
    curl $URL/api/v1/models/status -H 'Content-Type: application/json' \
        -d "{\"hf_repo\":\"$MODEL\",\"hf_commit\":\"$COMMIT\"}"
    ```

3. Start vLLM. Repeat the second command until `is_running` is true. For GLM, replace the last field with `additional_args: (.[0].models[$m].args + ["--max-model-len", "400000"])`.

    ```bash
    jq -c --arg m "$MODEL" '{model: $m, dtype: "auto", additional_args: .[0].models[$m].args}' $CFG \
        | curl $URL/api/v1/inference/up/async -H 'Content-Type: application/json' -d @-
    curl $URL/api/v1/inference/up/status
    ```

4. Measure the nonce rate and validate the golden reference, 200 nonces recorded on another node.

    ```bash
    python3 $POC/validate.py --mlnode-url $URL --model $MODEL --scheme decode \
        --skip-download --skip-deploy \
        --threshold $TAU --p-mismatch 0.1 --fraud-threshold 0.01
    ```

    Exit code 0 is `PASS`, 2 is `FAIL`. The rate is the sum over all instances; it grows during the first minute because decode mining starts in groups, so read the last sample. The three threshold flags are the v0.2.17 chain values. Without them the script uses the values stored in the reference, which differ.

5. Check the node in the other direction. Record its nonces and validate them on a second node.

    ```bash
    python3 $POC/make_artifact.py --mlnode-url $URL --model $MODEL --scheme decode \
        --num-nonces 200 --out ref.json
    python3 $POC/validate.py --mlnode-url $URL2 --model $MODEL --scheme decode --reference ref.json \
        --skip-download --skip-deploy --skip-throughput \
        --threshold $TAU --p-mismatch 0.1 --fraud-threshold 0.01
    ```

### Expected result

With these thresholds, fraud starts at 31 mismatches of 200. Honest nodes gave 0–8 against the golden references on B300, H200 and H100; B200, and DeepSeek on H100, were not measured. MiniMax AWQ gave 43, DeepSeek REAP 200.

On one RTX 3090 with Qwen3-0.6B, steps 1–5 give PASS with 0 of 200, and a reference with one point changed per nonce gives FAIL with 196 of 200.

Compare nodes only through validation. Free-running trajectories from two honest nodes differ: one near-tie changes every later step.

### Build the images

```bash
git clone https://github.com/gonka-ai/vllm && cd vllm && git checkout c694cd48fb
docker build -f docker/Dockerfile.gonka-poc -t vllm-poc:0.30.0 .

cd ../gonka && git submodule update --init mlnode/packages/train/third_party/gorilla && cd mlnode
docker build -f packages/api/Dockerfile --target app --build-arg VLLM_BASE_IMAGE=vllm-poc:0.30.0 -t mlnode:local .
```
