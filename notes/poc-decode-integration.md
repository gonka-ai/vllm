# Decode PoC: how to reproduce it

## Versions

| | |
| --- | --- |
| MLNode image | `ghcr.io/gonka-ai/mlnode:4.0.0-sigma3@sha256:ce449de9545b55a8c55d495eaa512beebb72cd29478d4f4b672a1c231ce89d1f` |
| MLNode image, NVIDIA driver older than 580 | `ghcr.io/gonka-ai/mlnode:4.0.0-sigma3-cu12@sha256:3e4588d1dc33f4a09a1d099cb491348cdc80849ad7bad883da184ccf055f9aca` |
| vLLM in the image | [gonka-ai/vllm](https://github.com/gonka-ai/vllm/tree/c694cd48fb04c11b608c52384a3e508b14d3c915), branch `release/v0.30-decode-int`, commit `c694cd48fb` |
| PoC plugin in the image | [gonka-ai/gonka-vllm-plugins](https://github.com/gonka-ai/gonka-vllm-plugins/tree/v0.2.1), tag `v0.2.1` (commit `c98fc00`, reports version 0.2.0) |
| Presets and scripts | [gonka-ai/gonka](https://github.com/gonka-ai/gonka/tree/aa3c763ab814aa1af9694de76df6fe7dabd3ef50), branch `upgrade-v0.2.17`, commit `aa3c763ab` |

To build the image: `docker/Dockerfile.gonka-poc` in the vLLM commit, then `mlnode/packages/api/Dockerfile` in the gonka commit on top of it (`--build-arg VLLM_BASE_IMAGE`).

## Models and GPUs

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

## Run

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

## Expected result

With these thresholds, fraud starts at 31 mismatches of 200. Honest nodes gave 0–8 against the golden references on B300, H200 and H100. B200, and DeepSeek on H100, were not measured. MiniMax AWQ gave 43, DeepSeek REAP 200.

Compare nodes only through validation. Free-running trajectories from two honest nodes differ: one near-tie changes every later step.

## What happens

1. The API node sends `block_hash`, `public_key` and nonces, with `params: {model, decode: true, seq_len: 256, max_tokens: 256}`. `/pow/init/generate` mines until `/pow/stop`; `/pow/generate` computes the given nonces.
2. Each nonce becomes one vLLM request. Its prompt is 256 embeddings derived from `(block_hash, public_key, nonce)`, prefilled in one step.
3. During the forward, every decoder layer output is reflected (Householder, seeded by `block_hash` and layer), and MoE routers pick experts from a seed instead of from the hidden state.
4. The last hidden state after the final norm is snapped to one of 16 fixed points: take 256 seeded coordinates, normalize, choose the point with the highest cosine. This is `k_0`.
5. `k_0` seeds the next input embedding and the next 256 coordinates. The model runs one decode step on the same KV cache, which gives `k_1`. This repeats 256 times.
6. The artifact is `k_points_steps = [k_0, ..., k_256]`. No token is sampled. `vector_b64` is empty.
7. A validator runs the same nonce but feeds the prover's points into the chain. At each step where its own point differs, the margin is `best_score - score_of_the_claimed_point`. A nonce mismatches if its largest margin exceeds τ.
8. A one-sided binomial test compares the number of mismatched nonces with `p_mismatch = 0.1`. Fraud if the p-value is below 0.01.

While mining, chat gets 503. While validating, chat waits. A nonce with a non-finite value is dropped; a validator that drops one abstains.

## Exact formulas

Needed only for another engine. `seed(s)` is the first 4 bytes of SHA-256, big-endian. `mm(key, seed)`, `normal(seed, n)` are `_murmur3_32`, `_normal` from [`gpu_random.py`](https://github.com/gonka-ai/gonka-vllm-plugins/blob/v0.2.1/src/gonka_poc/poc/gpu_random.py#L22-L129). `H` is the hidden size.

```
base               = seed(f"{block_hash}_{public_key}_nonce{nonce}")
step_seed(i, salt) = mm((k_{i-1} * 0x9E3779B1 + i * 0x85EBCA77 + salt) mod 2^32, base)

prompt embeddings  = normal(base, seq_len * H), shape [seq_len, H], positions 0..seq_len-1
input of step i    = normal(step_seed(i, 0x0D), H), position seq_len + i - 1, i = 1..max_tokens
reflection, layer l: x - 2 (x·v) v,  v = unit(normal(seed(f"{block_hash}_layer_{l}_householder"), H))

pick seed, k_0     = seed(f"{block_hash}_{public_key}_nonce_{nonce}_pick_256_decode0")
pick seed, step i  = step_seed(i, 0x91)
coordinates        = the 256 values d with the smallest mm(d, pick_seed)
q                  = unit(hidden[coordinates]), float32
k                  = argmax(codebook @ q)

router, MoE gate j : start = mm(step, seed(f"{block_hash}_n{nonce}_route_layer_{j}")) mod n_experts
                     logit of expert (start + r) mod n_experts = 100 + top_k - r, r = 0..top_k-1; others -10000
```

- Coordinate order: the 255 smallest by ascending `d`, then the 256th. This is `torch.topk(sorted=False)` on CUDA. Another order gives other points.
- [`sphere_codebook.pt`](https://github.com/gonka-ai/gonka-vllm-plugins/blob/v0.2.1/src/gonka_poc/poc/sphere_codebook.pt): `[16, 256]` float32, SHA-256 of the data `2201d1093583f43f328dd4643ddf1f34737f6da8c1956395c43fd093e357e34f`. Do not rebuild it.
- Embeddings and `v` are computed in float32, then cast to the model dtype.
- If a layer returns `(hidden, residual)`, reflect both.
- `j` counts only the gates with replaced logits, from 0. `step` is 0 for prompt tokens. The model then applies its own scoring, bias and top-k to these logits.
- DeepSeek-V4 gates that route by token id (`tid2eid`) keep their logits. Token ids there: `mm(p, seed(f"{block_hash}_{public_key}_nonce{nonce}_input_ids")) mod vocab` for prompt position `p`, `mm(0, step_seed(i, 0x57)) mod vocab` for step `i`.

Code: [`decode_random.py`](https://github.com/gonka-ai/gonka-vllm-plugins/blob/v0.2.1/src/gonka_poc/poc/decode_random.py), [`native.py`](https://github.com/gonka-ai/gonka-vllm-plugins/blob/v0.2.1/src/gonka_poc/mixed/native.py), [`sphere.py`](https://github.com/gonka-ai/gonka-vllm-plugins/blob/v0.2.1/src/gonka_poc/poc/sphere.py), [`runtime.py`](https://github.com/gonka-ai/gonka-vllm-plugins/blob/v0.2.1/src/gonka_poc/mixed/runtime.py), [`validation.py`](https://github.com/gonka-ai/gonka-vllm-plugins/blob/v0.2.1/src/gonka_poc/poc/validation.py).
