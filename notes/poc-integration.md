# Gonka MLNode Integration

## Plan

### 1. Proof of Compute 

Proof of Compute proves computational capacity on-chain. Most integration risk is in this step because validators measure artifact distance and allow only a small rate of numerical mismatches.

1. Add PoC modifications to the model forward flow 
2. Experiment: produce artifacts from `(block_hash, public_key, nonces)` and compare them with the [Kimi-K2.6 golden artifacts](https://github.com/gonka-ai/gonka/blob/main/mlnode/packages/benchmarks/scripts/poc_validation/artifacts/moonshotai-kimi-k2.6.json).

    This will likely require multiple iterations because the check is sensitive. It would be simplest to start with a small model if a local development flow is available.

3. Implement APIs for PoC generation and artifact validation.
4. Test the full flow: 
    - PoC artifacts generated with vLLM on NVIDIA GPUs are validated on custom nodes, and the statistical test passes
    - PoC artifacts generated on custom nodes are validated on NVIDIA GPUs, and the statistical test passes

    Hosts can verify PoC validation correctness with the [`mlnode-validate` skill](https://github.com/gonka-ai/gonka/blob/main/skills/mlnode-validate/SKILL.md).

### 2. Inference Validation

Gonka inference uses the OpenAI-compatible `/v1/chat/completions` API. Most parameters should work unchanged. Compatibility differences can be handled by the proxy.

1. Add `enforced_tokens` replay to the sampling flow.
2. Experiment:
    - inference results and artifacts produced by custom nodes are validated on NVIDIA GPUs
    - inference results and artifacts produced on NVIDIA GPUs are validated by custom nodes
    - inference results and artifacts produced by a modified model on NVIDIA GPUs are rejected by custom nodes

    Reference tests: [inference validation scripts](https://github.com/gonka-ai/gonka/tree/mlnode-v3.0.14/mlnode/packages/benchmarks/scripts/inference_validation).

    This step also carries some risk because frameworks may collect logprobs at different stages of the sampling pipeline. vLLM itself has had such differences.

3. Check `/v1/chat/completions` parameter compatibility against [`docs/chat-api/README.md`](https://github.com/gonka-ai/gonka/blob/main/docs/chat-api/README.md). Resolve any differences in the proxy.

---

## Details

Implementation diff against vLLM `release/v0.20.0`: [`release/v0.20.0...gm/poc-integration-notes`](https://github.com/gonka-ai/vllm/compare/release/v0.20.0...gm/poc-integration-notes).

The source links below show the vLLM reference implementation. Other engines may use different hooks and memory management.

### 1. Proof of Compute

#### Forward Pass

PoC requires these changes to the model forward pass:

- the embedding lookup is bypassed with `generate_inputs(block_hash, public_key, nonces)` ([`vllm/poc/gpu_random.py`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/poc/gpu_random.py#L119-L136)). The generated batch is passed to the model as `inputs_embeds` ([`vllm/poc/poc_model_runner.py`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/poc/poc_model_runner.py#L273-L285)).
- a Householder reflection (`hh`) is applied after each `layer_i`. The vectors are generated and the hooks are registered in [`vllm/poc/layer_hooks.py`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/poc/layer_hooks.py#L76-L113). The hooks are enabled during the PoC forward pass ([`vllm/poc/poc_model_runner.py`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/poc/poc_model_runner.py#L273-L285)).
- additional transformations are applied to the output after the model forward pass in [`vllm/poc/poc_model_runner.py`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/poc/poc_model_runner.py#L299-L330).

The usual transformer forward pass:
```
model = layer1 -> layer2 -> layer3 -> .... layer_n
input_tokens = [batch_size x seq_len]
input = embedding(input_tokens) <--- batch_size x seq_len x hidden_size

last_hidden = model(input)[:, -1, :] <---- batch_size x hidden_size
```

For each PoC session (`block_hash`):

```
model = layer1 -> hh_1 -> layer2 -> hh_2 -> layer3 -> hh_3 .... layer_n -> hh_n
```

#### Artifact Generation

It iterates over an assigned subsequence of `0, 1, 2, 3, ...` ([`vllm/poc/routes.py`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/poc/routes.py#L71-L92)):

PoC model forward pass (using the same loaded model):
```
input = generate_inputs(block_hash, public_key, nonces) <--- batch_size x seq_len x hidden_size
last_hidden = model_poc(input)[:, -1, :] <---- batch_size x hidden_size
```

Post-forward output transformations:
1. normalize to the unit sphere ([`vllm/poc/poc_model_runner.py`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/poc/poc_model_runner.py#L318-L319))
2. deterministically select `k_dim` coordinates (12 by default) from the normalized vector ([`vllm/poc/gpu_random.py`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/poc/gpu_random.py#L212-L237))
3. apply a Haar rotation ([`vllm/poc/gpu_random.py`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/poc/gpu_random.py#L240-L271))
4. normalize to the unit sphere ([`vllm/poc/poc_model_runner.py`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/poc/poc_model_runner.py#L326-L330))

> **vLLM memory management:** `/generate` reserves free KV-cache blocks. `/init/generate` reuses low KV-cache blocks after aborting in-flight inference ([`vllm/poc/engine_patch.py`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/poc/engine_patch.py#L46-L95), [`vllm/poc/engine_patch.py`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/poc/engine_patch.py#L163-L230)).

#### Endpoints

End-to-end process:

- [`POST /api/v1/pow/init/generate`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/poc/routes.py#L370-L438): start continuous PoC generation
  - new chat and completion requests return 503. In-flight requests are aborted before generation ([`vllm/entrypoints/openai/chat_completion/api_router.py`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/entrypoints/openai/chat_completion/api_router.py#L54-L73), [`vllm/poc/engine_patch.py`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/poc/engine_patch.py#L90-L95))
  - if `url` is provided, PoC artifacts are sent to `{url}/generated` ([`vllm/poc/callbacks.py`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/poc/callbacks.py#L114-L127))
- [`POST /api/v1/pow/generate`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/poc/routes.py#L441-L567): generate artifacts for specific nonces, with optional validation
  - if validation artifacts are provided, compare them with the generated artifacts and run the statistical test
  - with `wait=false`, queue the request and return a `request_id`
  - with `wait=false` and `url`, send the result to `{url}/generated` or `{url}/validated`
- [`GET /api/v1/pow/status`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/poc/routes.py#L587-L589): get the current generation status
- [`POST /api/v1/pow/stop`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/poc/routes.py#L592-L601): stop continuous PoC generation
- [`GET /api/v1/pow/versions`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/poc/routes.py#L604-L611): get version information

### 2. Inference Validation

#### Validation Replay

During inference, the client records each generated token and its `top_logprobs`. During validation, the engine replays the original token sequence and returns recomputed `logprobs` and `top_logprobs`. The client compares both results.

Tokens in `logprobs` and `top_logprobs` are returned as numeric token-ID strings, not decoded text ([`vllm/entrypoints/openai/engine/serving.py`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/entrypoints/openai/engine/serving.py#L723-L734)).

`enforced_tokens` is the `/v1/chat/completions` request parameter ([`vllm/entrypoints/openai/chat_completion/protocol.py`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/entrypoints/openai/chat_completion/protocol.py#L342-L345)). It is converted to internal `enforced_token_ids` ([`vllm/entrypoints/openai/chat_completion/serving.py`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/entrypoints/openai/chat_completion/serving.py#L306-L331)). During sampling, the next enforced token replaces the sampled token without changing logits ([`vllm/v1/sample/sampler.py`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/v1/sample/sampler.py#L139-L173)).

`logprobs_mode` must match the original inference. If not provided, replay detects raw or processed mode from `enforced_tokens.tokens[].top_tokens` ([`vllm/validation.py`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/validation.py#L51-L74)).
