# Gonka MLNode Integration

## Plan

### 1. Proof of Compute 

Proof of Compute is the mechanism used to prove computational capacity on-chain. The code changes are localized, but most integration risks are in this step because it explicitly checks how closely computation results match. 


1.1. Add PoC modifications to the model forward flow 
1.2. Experiment: produce artifacts from `(block_hash, pubkey, nonces)` and compare them with the [Kimi-K2.6 golden artifacts](https://github.com/gonka-ai/gonka/blob/main/mlnode/packages/benchmarks/scripts/poc_validation/artifacts/moonshotai-kimi-k2.6.json).

This will likely require multiple iterations because the check is sensitive. It would be simplest to start with a small model if a local development flow is available.

1.3. Implement APIs to start and stop PoC generation and validation. Operational risks remain around KV-cache handling and inference interruption. 
1.4. Test the full flow: 
    - PoC artifacts generated with vLLM on NVIDIA GPUs are validated on custom nodes, and the statistical test passes
    - PoC artifacts generated on custom nodes are validated on NVIDIA GPUs, and the statistical test passes

Hosts can verify PoC validation correctness with the [`mlnode-validate` skill](https://github.com/gonka-ai/gonka/blob/main/skills/mlnode-validate/SKILL.md).


### 2. Inference Validation

Gonka inference is served through vLLM's OpenAI-compatible `/v1/chat/completions` API. I expect the overall parameter set to be the same and require no modifications, or only minimal modifications that can be made in the final step or on the proxy side. 

2.1. Add `enforced_tokens` replay to the sampling flow.
2.2. Experiment:
    - inference results and artifacts produced by custom nodes are validated on NVIDIA GPUs
    - inference results and artifacts produced on NVIDIA GPUs are validated by custom nodes
    - inference results and artifacts produced by a modified model on NVIDIA GPUs are rejected by custom nodes

[Inference validation test scripts](https://github.com/gonka-ai/gonka/tree/mlnode-v3.0.14/mlnode/packages/benchmarks/scripts/inference_validation).

This step also carries some risk because frameworks may collect logprobs at different stages of the sampling pipeline. vLLM itself has had such differences.

2.3. Check `/chat/completions` parameter compatibility against [`docs/chat-api/README.md`](https://github.com/gonka-ai/gonka/blob/main/docs/chat-api/README.md). If there are any issues, I expect they can be resolved on the proxy side. 





----

## Details

### 1. Proof of Compute

#### Forward Pass

We need to modify the forward pass to enable PoC mode:
- the embedding layer is bypassed and replaced with `generate_inputs(block_hash, pubkey, nonce)` ([`vllm/poc/gpu_random.py`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/poc/gpu_random.py#L119-L136)). The generated tensor is passed to the model as `inputs_embeds` ([`vllm/poc/poc_model_runner.py`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/poc/poc_model_runner.py#L257-L285)). The input is pseudorandom and deterministically generated from chain-derived values
- a Householder reflection (`hh`) is applied after each `layer_i`. The vectors are generated and the hooks are registered in [`vllm/poc/layer_hooks.py`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/poc/layer_hooks.py#L76-L113). The hooks are enabled during the PoC forward pass ([`vllm/poc/poc_model_runner.py`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/poc/poc_model_runner.py#L273-L285))
- additional transformations are applied to the output after the model forward pass in [`vllm/poc/poc_model_runner.py`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/poc/poc_model_runner.py#L299-L330)


The usual forward pass:
```
model = layer1 -> layer2 -> layer3 -> .... layer_n
input_tokens = [seq_len]
input = embedding(input_tokens) <--- seq_len x hidden_size

last_hidden = model(input)[-1] <---- 1 x hidden_size
```


For each PoC session (`block_hash`) and host (`pubkey`):

```
model = layer1 -> hh_1 -> layer2 -> hh_2 -> layer3 -> hh_3 .... layer_n -> hh_n
```

#### Artifact Generation

It then iterates over `nonces = 1, 2, 3, ....`:

PoC model forward pass (using the same loaded model):
```
input = generate_inputs(block_hash, pubkey, nonce)
last_hidden = model_poc(input)[-1] <---- 1 x hidden_size
```

Post-forward output transformations:
1. normalize to the unit sphere ([`vllm/poc/poc_model_runner.py`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/poc/poc_model_runner.py#L318-L319))
2. deterministically select 12 coordinates from the normalized vector ([`vllm/poc/gpu_random.py`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/poc/gpu_random.py#L212-L237))
3. apply a Haar rotation ([`vllm/poc/gpu_random.py`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/poc/gpu_random.py#L240-L271))
4. normalize to the unit sphere ([`vllm/poc/poc_model_runner.py`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/poc/poc_model_runner.py#L326-L330))

> **Memory note:** Allocation is engine-specific. VRAM can be reserved in advance or borrowed from free KV-cache blocks. vLLM uses reservation for `/generate` and shared blocks for `/init/generate` after aborting in-flight inference ([`vllm/poc/engine_patch.py`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/poc/engine_patch.py#L46-L95), [`vllm/poc/engine_patch.py`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/poc/engine_patch.py#L163-L230)).

#### Endpoints

End-to-end process:
-> [`POST /api/v1/pow/init/generate`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/poc/routes.py#L370-L438) to start `POC GENERATION`
  in a loop:
    - a batch of PoC artifacts is generated and sent to `{url}/generated` ([`vllm/poc/callbacks.py`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/poc/callbacks.py#L114-L127))

-> [`POST /api/v1/pow/generate`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/poc/routes.py#L441-L567) to generate artifacts for specific nonces and validate them with the statistical test
  - if validation artifacts are provided, compare them with the generated artifacts and run the statistical test
  - with `wait=false`, queue the request and return a `request_id`
  - with `wait=false` and `url`, send the result to `{url}/generated` or `{url}/validated`

-> [`GET /api/v1/pow/status`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/poc/routes.py#L587-L589) to get the current generation status

-> [`POST /api/v1/pow/stop`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/poc/routes.py#L592-L601) to stop `POC GENERATION`

-> [`GET /api/v1/pow/versions`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/poc/routes.py#L604-L611) to get the version

### 2. Inference Validation

#### Validation Replay

During inference, each generated token and its `top_logprobs` are recorded for later validation. During validation, the original token sequence is replayed, and the recomputed `logprobs` and `top_logprobs` are compared with the recorded values.

`enforced_tokens` is the `/v1/chat/completions` request parameter ([`vllm/entrypoints/openai/chat_completion/protocol.py`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/entrypoints/openai/chat_completion/protocol.py#L342-L345)). It is converted to internal `enforced_token_ids` in [`vllm/entrypoints/openai/chat_completion/serving.py`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/entrypoints/openai/chat_completion/serving.py#L306-L324). During sampling, the next enforced token replaces the sampled token without changing logits. The replay returns the token logprob and the model's actual top-k for comparison ([`vllm/v1/sample/sampler.py`](https://github.com/gonka-ai/vllm/blob/gm/poc-integration-notes/vllm/v1/sample/sampler.py#L139-L173)).


