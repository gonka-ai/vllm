# PoC and Inference Validation Migration to v0.23.0

This document is the migration ledger for porting
`b8160878f07fe6aff02deb12bc842df3fa4a9237..gm/poc-integration-1` onto
`releases/v0.23.0`.

Keep this file current during the migration. Every non-trivial issue, skipped
source-branch change, compatibility decision, and open question should be
recorded here before the port is considered done.

## Goals

- Port PoC v2 and inference validation onto `releases/v0.23.0`.
- Keep the migration clean and reviewable: no broad file overwrites where
  v0.23.0 has diverged, no scratch artifacts, and no unrelated default changes.
- Prove the port locally with `Qwen/Qwen3-0.6B` first.
- Compare against `gm/poc-integration-1` as the previous-version baseline.
- Run final acceptance against a container built from `Dockerfile.quick`, not a
  local editable build.
- Preserve enough artifacts to audit PoC correctness, inference validation,
  cross-version validation, and PoC throughput.

## Clean Migration Rules

- Start from `releases/v0.23.0` on branch `gm/port-pocv2-vllm-0.23.0`.
- Stage paths explicitly because the current worktree contains many untracked
  scratch/data files.
- Copy additive files only when they do not overwrite v0.23.0 logic.
- Manually port changes into files that changed upstream in v0.23.0.
- Preserve v0.23.0 behavior unless PoC or inference validation requires a
  deliberate change.
- Do not migrate noisy operational defaults, Docker changes, CI changes, or
  old docs unless they are needed for the local test pipeline.
- Do not use a full local build as final proof. Local editable installs are
  acceptable for quick development smoke checks only; signoff must use the
  `Dockerfile.quick` image because it is cleaner and closer to deployment.
- Record every manual decision below with the affected files and test evidence.

## Feature Summary

PoC adds a vLLM-native `/api/v1/pow` surface for deterministic vector
generation and validation. The core implementation lives under `vllm/poc/` and
uses model-forward hooks, deterministic RNG, queue/callback handling, and API
priority gating so normal inference does not run while PoC generation is
active.

Inference validation adds `enforced_tokens`, `enforced_str`, and
`logprobs_mode` support. Inference responses provide token/logprob artifacts;
validation requests force the same token sequence while preserving measured
logprob distributions; Gonka compares the two with `distance2`.

## Known Migration Issues

### Worktree and Branch Hygiene

Status: resolved

Affected areas:
- Repository root
- Existing untracked files and directories

Issue:
- The current tracked tree is clean, but there are many untracked scratch,
  experiment, log, notebook, config, and test files.

Decision needed:
Decision:
- Created `gm/port-pocv2-vllm-0.23.0` from `releases/v0.23.0`.
- Tracked files were clean before branching.
- Stage only intended migration paths.
- Do not delete or modify unrelated untracked files.

Test evidence:
- `git status --short --branch` showed no tracked diffs before branch creation.

### Additive PoC Modules

Status: open

Affected files:
- `vllm/poc/__init__.py`
- `vllm/poc/callbacks.py`
- `vllm/poc/config.py`
- `vllm/poc/data.py`
- `vllm/poc/engine_patch.py`
- `vllm/poc/generate_queue.py`
- `vllm/poc/gpu_random.py`
- `vllm/poc/layer_hooks.py`
- `vllm/poc/manager.py`
- `vllm/poc/poc_model_runner.py`
- `vllm/poc/routes.py`
- `vllm/poc/validation.py`

Issue:
- These files are mostly source-only and can be copied forward, but they depend
  on vLLM internals that changed between the source branch and v0.23.0.

Migration work:
- Copy the files, then fix imports and API calls against v0.23.0.
- Verify `engine_patch.py` still works with v0.23.0 `AsyncLLM`, abort handling,
  output processing, and worker RPC APIs.
- Verify `poc_model_runner.py` against v0.23.0 attention metadata, forward
  context, KV cache layout, and distributed execution.

Test evidence:
- Pending.

### OpenAI API Route Integration

Status: open

Affected files:
- `vllm/entrypoints/openai/api_server.py`
- `vllm/entrypoints/openai/chat_completion/api_router.py`
- `vllm/entrypoints/openai/completion/api_router.py`
- `vllm/entrypoints/openai/server_utils.py`
- `vllm/entrypoints/serve/utils/*`

Issue:
- The source branch integrated the PoC router and queue cleanup through older
  OpenAI server structure. v0.23.0 reorganized some server utilities, and
  `server_utils.py` is not a safe direct target.

Migration work:
- Include the PoC router in the v0.23.0 API server.
- Port PoC priority gating without losing v0.23.0 route behavior.
- Move any server utility hooks to the v0.23.0 utility location.
- Ensure `/api/v1/pow/*` works directly on vLLM and through MLNode proxy paths.

Test evidence:
- Pending.

### Chat and Completion Request Schema

Status: open

Affected files:
- `vllm/entrypoints/openai/chat_completion/protocol.py`
- `vllm/entrypoints/openai/chat_completion/serving.py`
- `vllm/entrypoints/openai/completion/protocol.py`
- `vllm/entrypoints/openai/completion/serving.py`
- `vllm/entrypoints/openai/engine/serving.py`

Issue:
- The source branch adds `logprobs_mode`, `enforced_tokens`, and
  `enforced_str`. v0.23.0 has newer OpenAI schema and serving behavior that
  should not be overwritten.

Migration work:
- Add validation fields to v0.23.0 schema while preserving newer fields.
- Convert enforced payloads to token IDs and append EOS consistently.
- Preserve auto-detection of `logprobs_mode` when validation requests omit it.
- Preserve numeric-string token IDs in logprob output where Gonka requires it.
- Confirm completion behavior if completion validation remains supported.

Test evidence:
- Pending.

### `/v1/responses` Compatibility

Status: open

Affected files:
- `vllm/entrypoints/openai/responses/*`

Issue:
- v0.23.0 includes the Responses API. The source branch only covered chat and
  completion paths.

Decision needed:
- Decide whether PoC priority gating and validation fields must apply to
  `/v1/responses` now, or document it as explicitly unsupported.

Test evidence:
- Pending.

### Sampling Params

Status: open

Affected files:
- `vllm/sampling_params.py`

Issue:
- The source branch adds `logprobs_mode` and `enforced_token_ids`. v0.23.0
  added or changed other sampling fields and validation logic.

Migration work:
- Add the new fields without dropping v0.23.0 fields such as newer logprob,
  bad-word, thinking-budget, routed-expert, and request-behavior options.
- Preserve validation and serialization behavior.

Test evidence:
- Pending.

### Sampling Metadata and Sampler

Status: open

Affected files:
- `vllm/v1/sample/metadata.py`
- `vllm/v1/sample/sampler.py`
- `vllm/v1/sample/ops/topk_topp_sampler.py`

Issue:
- The source branch adds per-request logprobs modes, processed-logprobs paths,
  and enforced-token override after sampling. v0.23.0 has newer sampler logic.

Migration work:
- Preserve v0.23.0 sampler behavior including optimized logprob paths, fp64
  Gumbel, XPU paths, FlashInfer helpers, speculative decoding interactions,
  and any thinking-budget state.
- Force enforced tokens only after logprob measurement, so validation compares
  the validator model distribution rather than a post-forced distribution.
- Ensure mixed batches with raw and processed logprobs still work.

Test evidence:
- Pending.

### Worker Input Batch and Model Runner

Status: open

Affected files:
- `vllm/v1/worker/gpu_input_batch.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/worker/gpu/input_batch.py`
- `vllm/v1/worker/gpu/model_runner.py`

Issue:
- v0.23.0 worker layout and active runtime paths may differ from the source
  branch. Enforced-token progression must survive batching and scheduling.

Migration work:
- Identify the active v0.23.0 input-batch/model-runner files.
- Track `enforced_token_ids`, current enforced index, and per-request
  `logprobs_mode` across request additions, removals, condense/swap, reorder,
  speculative decode, EOS, and max-token termination.
- Keep PoC runner integration compatible with v0.23.0 distributed workers.

Test evidence:
- Pending.

### PoC Engine Execution

Status: open

Affected files:
- `vllm/poc/engine_patch.py`
- `vllm/poc/poc_model_runner.py`
- `vllm/poc/layer_hooks.py`
- `vllm/v1/worker/gpu_model_runner.py`
- Active v0.23.0 GPU model-runner path

Issue:
- PoC execution reaches deep vLLM internals: worker RPC, attention metadata,
  forward context, KV scratch reuse, tensor parallelism, and pipeline
  parallelism.

Migration work:
- Reconcile `collective_rpc` and worker method registration.
- Reconcile attention metadata construction, FlashInfer/MLA handling, and
  `seq_lens_cpu_upper_bound`.
- Verify KV cache scratch allocation and reuse under v0.23.0.
- Verify TP/PP behavior and multi-replica behavior.

Test evidence:
- Pending.

### Structured Output Degradation

Status: open

Affected files:
- `vllm/v1/structured_output/__init__.py`
- `vllm/v1/structured_output/backend_xgrammar.py`

Issue:
- The source branch makes grammar token rejection degrade gracefully during
  enforced-token replay. v0.23.0 may have changed structured-output internals.

Migration work:
- Preserve v0.23.0 structured-output features while adding graceful degradation.
- Confirm corrupted enforced tokens do not crash the engine.

Test evidence:
- Pending.

### Config and Operational Defaults

Status: open

Affected files:
- `vllm/config/model.py`
- `vllm/config/vllm.py`
- `vllm/engine/arg_utils.py`
- `vllm/model_executor/models/config.py`
- `Dockerfile.quick`
- `.github/workflows/build-stage1.yml`

Issue:
- The source branch contains operational changes such as Docker base bumps,
  dtype defaults, attention backend defaults, compilation config, custom-op
  defaults, `max_num_batched_tokens`, and `gpu_memory_utilization`.
- Final validation must run from a container built with `Dockerfile.quick`, so
  that file has to be kept functional for v0.23.0 without pulling in unrelated
  packaging or CI changes.

Decision needed:
- Migrate only the defaults required for Qwen3-0.6B acceptance and later
  production PoC correctness.
- Avoid noisy default changes unless they have a direct test-backed reason.
- Decide whether Docker/CI changes belong in this migration branch or a
  separate follow-up.
- Decide the exact `Dockerfile.quick` image tag/name used for old and migrated
  branch comparisons.

Test evidence:
- Pending.

### Container-Based Final Validation

Status: open

Affected files:
- `Dockerfile.quick`
- Gonka benchmark and validation scripts run against the containerized server

Issue:
- Final acceptance must use a container image built from `Dockerfile.quick`,
  not a local editable vLLM build. This keeps the migration cleaner and avoids
  local environment leakage.

Migration work:
- Build an image from `Dockerfile.quick` on the migrated branch.
- Start vLLM/MLNode test servers from that image for final live tests.
- Use the same container-based approach for the old `gm/poc-integration-1`
  baseline when collecting cross-version and throughput artifacts.
- Record image tags, container commands, ports, model, GPU visibility, and
  artifact paths.

Test evidence:
- Pending.

### Tests and Documentation

Status: open

Affected files:
- `tests/gonka/*`
- `tests/gonka/README.md`
- Potential v0.23.0 OpenAI/sampler tests

Issue:
- Source tests are useful, but docs still include `python3` examples that
  conflict with repo workflow rules. Tests may need v0.23.0 fixture updates.

Migration work:
- Port Gonka tests.
- Update commands to use `.venv/bin/python`.
- Add or adjust focused tests when v0.23.0 behavior requires a local assertion.
- Keep live tests skippable when no server is running.

Test evidence:
- Pending.

### Gonka PoC Validation Pipeline

Status: open

Affected paths:
- `/home/ubuntu/workspace/gonka/mlnode/packages/benchmarks/scripts/poc_validation/validate.py`
- `/home/ubuntu/workspace/gonka/mlnode/packages/benchmarks/scripts/poc_validation/artifacts/qwen-qwen3-0.6b.json`

Issue:
- Acceptance uses MLNode `validate.py`, which deploys or checks a model,
  measures PoC throughput, validates against a committed reference, and writes
  reports.

Migration work:
- Run with `--mlnode-url <local_mlnode_url> --model Qwen/Qwen3-0.6B`.
- Use the committed Qwen3-0.6B reference unless local deployment shape requires
  a custom reference.
- Preserve the output experiment directory and report paths.

Required artifacts:
- `validate_config.json`
- `validate_report.json`
- `validate_report.txt`

Test evidence:
- Pending.

### Gonka Inference Validation Pipeline

Status: open

Affected paths:
- `/home/ubuntu/workspace/gonka/mlnode/packages/benchmarks/scripts/inference_validation/inference.py`
- `/home/ubuntu/workspace/gonka/mlnode/packages/benchmarks/scripts/inference_validation/validation.py`
- `/home/ubuntu/workspace/gonka/mlnode/packages/benchmarks/scripts/analysis/inference_length_vs_distance.py`

Issue:
- Acceptance requires old/new self-validation and cross-validation artifacts.

Migration work:
- Run old inference, old validation, new inference, new validation, old->new,
  and new->old with identical prompt set and request params.
- Record model, URL, prompt count, sampling params, `logprobs_mode`, and
  artifact paths.

Required artifacts:
- `inference_config.json`
- `inference_results.jsonl`
- `validation_config.json`
- `inference_validation_results.jsonl`
- Analysis output from `inference_length_vs_distance.py`

Test evidence:
- Pending.

### PoC Cross-Version and Throughput Comparison

Status: open

Affected paths:
- `/home/ubuntu/workspace/gonka/mlnode/packages/benchmarks/scripts/poc_validation/collect_data.py`
- `/home/ubuntu/workspace/gonka/mlnode/packages/benchmarks/scripts/analysis/poc_l2_histogram.py`

Issue:
- `validate.py` gives a golden-reference pass/fail and throughput report, while
  `collect_data.py` plus `poc_l2_histogram.py` gives cross-version L2
  comparison. Both are useful.

Migration work:
- Collect old and new PoC vectors with identical config.
- Compare L2 distances and preserve histogram/stat output.
- Compare median PoC throughput across repeated runs on the same hardware.

Decision needed:
- Define how many repeated runs are enough for throughput comparison.
- Define whether "at least the same" means no median regression or allows a
  small noise band.

Test evidence:
- Pending.

## Open Questions

- Production-scale Qwen3-235B validation remains open. This migration was
  proven locally on `Qwen/Qwen3-0.6B` because that was the selected acceptance
  target.
- MLNode `poc_validation/validate.py` was not run against the migrated vLLM
  container because the local test target was a direct vLLM OpenAI server, not
  an MLNode API that exposes `/api/v1/inference/*`. Direct PoC artifacts were
  collected through `/api/v1/pow/generate` instead.
- The v0.23.0 V2 runner is active for this model. Both V1 and V2 sampling paths
  were ported, but V2 is the path validated by the local container tests.
- Mixed raw/processed logprobs in a single V2 batch should be treated as a
  remaining larger-scale stress case. Homogeneous request batches and
  per-request overrides are covered by live tests.
- `poc_stronger_rng` remains request/config-controlled; it is not enabled by
  default.

## Acceptance Artifacts

- Final migrated image: `vllm-poc-v023:latest`, built from `Dockerfile.quick`
  with base `vllm/vllm-openai:v0.23.0-cu129`.
- Old baseline image: `vllm-poc-old:latest`, built from `gm/poc-integration-1`
  with its `Dockerfile.quick`.
- Final live tests: `41 passed` against `vllm-poc-v023:latest` serving
  `Qwen/Qwen3-0.6B` on port `18199`.
- Final unit tests: `33 passed` inside `vllm-poc-v023:latest` with
  `tests/gonka` mounted read-only.
- Migrated inference results:
  `/home/ubuntu/workspace/gonka/mlnode/packages/benchmarks/data/experiments/vllm_v023_qwen06b_processed_smoke_2026-06-18_221506/inference_results.jsonl`
- Migrated validation results:
  `/home/ubuntu/workspace/gonka/mlnode/packages/benchmarks/data/experiments/vllm_v023_qwen06b_processed_smoke_2026-06-18_221506/validation_results.jsonl`
  - Summary: `n=5`, `bad=0`, `mean=0.000062`, `max=0.000205`,
    `min_match=0.9933`.
- Old baseline inference results:
  `/home/ubuntu/workspace/gonka/mlnode/packages/benchmarks/data/experiments/vllm_old_qwen06b_smoke_2026-06-18_220530/inference_results.jsonl`
- Old baseline self-validation:
  `/home/ubuntu/workspace/gonka/mlnode/packages/benchmarks/data/experiments/vllm_old_qwen06b_smoke_2026-06-18_220530/validation_results.jsonl`
  - Summary: `n=5`, `bad=0`, `mean=0.000664`, `max=0.002772`,
    `min_match=0.9867`.
- Cross validation, new inference -> old validator:
  `/home/ubuntu/workspace/gonka/mlnode/packages/benchmarks/data/experiments/vllm_v023_qwen06b_smoke_2026-06-18_220307/validation_results__old_validator.jsonl`
  - Summary: `n=5`, `bad=0`, `mean=0.001696`, `max=0.001775`,
    `min_match=0.9733`.
- Cross validation, old inference -> new validator:
  `/home/ubuntu/workspace/gonka/mlnode/packages/benchmarks/data/experiments/vllm_old_qwen06b_smoke_2026-06-18_220530/validation_results__new_validator_processed.jsonl`
  - Summary: `n=5`, `bad=0`, `mean=0.001347`, `max=0.003498`,
    `min_match=0.9867`.
- Migrated direct PoC report:
  `/home/ubuntu/workspace/gonka/mlnode/packages/benchmarks/data/experiments/vllm_v023_qwen06b_final_direct_poc_2026-06-18_221538/poc_artifacts.json`
  - `artifact_count=8`, `n_mismatch=0`, `fraud_detected=false`,
    `nonces_per_second=27.445`.
- Old direct PoC report:
  `/home/ubuntu/workspace/gonka/mlnode/packages/benchmarks/data/experiments/vllm_old_qwen06b_direct_poc_2026-06-18_221646/poc_artifacts.json`
  - `artifact_count=8`, `n_mismatch=0`, `fraud_detected=false`,
    `nonces_per_second=18.199`.
- PoC throughput comparison: migrated smoke throughput was higher than old on
  the same RTX 5060 Ti setup (`27.445` vs `18.199` nonces/s). This is a small
  smoke sample, not a production benchmark.

## Decision Log

- Created `gm/port-pocv2-vllm-0.23.0` from `releases/v0.23.0`.
- Ported additive PoC modules/tests and manually reconciled OpenAI routes,
  request schemas, sampling params, V1/V2 samplers, V1 input batch, structured
  output graceful degradation, and Qwen3 MoE config.
- Added PoC priority gating to `/v1/chat/completions`, `/v1/completions`, and
  `/v1/responses`. Validation-specific `enforced_tokens` remains chat-only.
- Kept numeric token IDs as strings for OpenAI logprob payloads, matching the
  Gonka validator contract.
- Set the default `logprobs_mode` to `processed_logprobs`; cross-version
  validation proved this was required for old inference artifacts to validate
  cleanly on the migrated server.
- Disabled PoC KV-cache scratch reuse for v0.23.0. Reusing KV tensors as PoC
  input scratch caused degenerate post-PoC inference under the V2 runner; using
  separate PoC input tensors fixed the full live suite.
- Detached PoC layer hooks after each PoC forward so normal inference does not
  retain PoC hooks.
- Updated `Dockerfile.quick` to use `vllm/vllm-openai:v0.23.0-cu129` and added
  `pytest-asyncio` so the container can run the Gonka unit tests.
- Adjusted the post-PoC live test to check for degenerate repeated output
  instead of exact arithmetic/prime answers, because Qwen3-0.6B emits reasoning
  text before final answers.
