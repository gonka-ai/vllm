# vLLM modifications

Implementation reference: <https://github.com/gonka-ai/vllm/compare/v0.15.1...tg/scratchpad_for_mode>

## PoC

PoC makes the compute capacity of a vLLM deployment statistically verifiable on the actual deployed model. It profiles the same model weights and transformer layers used for inference, then emits compact artifacts that validators recompute for sampled nonces.

When PoC is active, vLLM runs proof batches back-to-back to keep the GPU compute path busy. `poc_request()` in `vllm/poc/engine_patch.py` starts the GPU proof work.

The current implementation measures prefill, also called the encoding phase: a fixed-length random context is passed through the model for each nonce. Token IDs are ignored. Instead, `execute_poc_forward()` in `vllm/poc/poc_model_runner.py` calls the model with deterministic `inputs_embeds`.

The current PoC already uses normal inference compute primitives in the prefill path. Future protocol versions are expected to add decode steps so PoC also profiles KV-cache reads/writes, memory-access behavior, and attention over cached tokens.

### Current compute stages:

- **Deterministic random input**: `generate_inputs()` and `generate_inputs_concat_murmur()` in `vllm/poc/gpu_random.py` create random embedding tensors with the same shape as normal token embeddings: `[batch, seq_len, hidden_size]`. These inputs are derived from `block_hash + public_key + nonce`, so each nonce produces a deterministic prefill workload without depending on tokenizer output. vLLM runs this over consecutive nonce batches, so production nodes can process thousands of nonces per minute on large deployed models. The RNG path is SHA-256-derived seed, Murmur3-32, uniform values, then Box-Muller normal values. The input tensors are generated on GPU and executed through the usual vLLM inference path.

- **Model forward over the deployed transformer**: `execute_poc_forward()` builds V1 attention metadata, slot mappings, and KV block tables, then calls the loaded model with `inputs_embeds` under `set_forward_context(..., skip_compiled=True)`. This exercises the model's attention, MLP, normalization, tensor-parallel, and pipeline-parallel paths.

- **Layer-level randomization**: `LayerHouseholderHook` in `vllm/poc/layer_hooks.py` applies a per-layer Householder reflection during PoC forwards only. The reflection vector is seeded by block hash and layer index. This adds deterministic transformations between transformer layers and makes the proof less dependent on stable model-specific output structure.

- **Final dimensionality reduction**: The last-token hidden state is L2-normalized, reduced from `hidden_size` to `k_dim` by `random_pick_indices()`, then randomized with `apply_haar_rotation()` in `vllm/poc/gpu_random.py`. The result is normalized again and becomes the artifact vector.

- **Artifact validation**: `encode_vector()` in `vllm/poc/data.py` stores each artifact as base64 little-endian FP16. The `/generate` PoC path recomputes artifacts for sampled `(block_hash, public_key, nonce)` tuples. `run_validation()` in `vllm/poc/validation.py` compares vectors with an L2 threshold and applies a binomial fraud test.

## Inference Validation

Inference validation checks that a submitted output sequence is consistent with the claimed model and sampling parameters. The executor exports the generated token and top logprobs for each output position. The validator reruns the model on the same prompt, forces the same output tokens, recomputes top logprobs, and compares the per-token distributions.

This reuses the normal vLLM decode path. The changes are:

- **Top-logprob export mode**: `logprobs_mode` is added to the chat and completion protocols in `vllm/entrypoints/openai/chat_completion/protocol.py` and `vllm/entrypoints/openai/completion/protocol.py`, then carried through `SamplingParams` in `vllm/sampling_params.py`. `Sampler.forward()` in `vllm/v1/sample/sampler.py` can export either raw logprobs before sampling processors or processed logprobs after sampling processors.

- **Forced token sequence**: `EnforcedTokens` in `vllm/validation.py` represents the submitted sequence. Chat serving converts it into `SamplingParams.enforced_token_ids`. `InputBatch._build_enforced_tensor()` in `vllm/v1/worker/gpu_input_batch.py` selects the required next token for each decode step, and `Sampler.forward()` overrides only the sampled token ID. Logits and logprobs are still computed normally, so validation follows the same decode computation as ordinary inference.
