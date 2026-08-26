# Porting the PoC residual onto a new vLLM base

This branch (`poc-residual-glm53`) carries the Gonka PoC residual on top of the
GLM-5.3-Flash vLLM build. It is the same stack that `gonka-ai/vllm`
`release/v0.25.1` carries, replayed onto a newer base.

## What the stack is

Eight commits, cherry-picked in this order from `gonka-ai/vllm@release/v0.25.1`:

| # | Subject |
|---|---------|
| 1 | Merge pull request #78 — port PoC sampler residual (the whole V1 stack, squashed) |
| 2 | fix(poc): stop a disabled grammar from leaking another request's mask (#79) |
| 3 | fix(poc): range-check and bound replay ids before the engine sees them (#80) |
| 4 | fix(poc): make the shm ring size a knob instead of a hardcoded 64 (#82) |
| 5 | feat(poc): port replay hooks to the V2 model runner (#92) |
| 6 | chore(poc): update plugin to v0.1.2 |
| 7 | fix(poc): restore 32768-token API batch default |
| 8 | chore(poc): update plugin to v0.1.3 |

On top of those, two fixes that the release line currently carries as Stage-4
layers in `mlnode-foundry` only because the published images stopped building
from a residual tree. This branch IS that tree, so they belong here and the
corresponding S4 layers are unnecessary for anything built on it:

| # | Subject | S4 equivalent |
|---|---------|---------------|
| 9 | fix(sched): skip requests absent from req_id_to_index | `sched-req-index-guard` (kaitakuai/vllm#19) |
| 10-12 | keep replaying requests out of speculative decoding, guard the padding at its source, pin a replay's max_tokens | kaitakuai/vllm#21, not yet an S4 layer |

Rows 10-12 are not optional for this model: GLM-5.3-Flash config declares
`num_nextn_predict_layers: 1`, so it speculates, and an unpatched replay
diverges from the executor exactly as measured on Hy3 (82 of 100 length
mismatches). The `max_tokens` pin matters whenever a validator runs a larger
limit than the executor did.

The remaining Stage-4 layers stay where they are: `triton-ptxas-from-system-cuda`,
`flashinfer-jit-uninstall`, `libcuda-compat-580-driver`, `nvidia-headers-symlinks`
and `cold-start-tolerance` are hardware and timeout tuning, not in-tree fixes.
`content-type-injector` is the S4 form of `patches/0001`, which our own Stage 3
already applies. `libnvrtc-symlink` merged upstream as gonka#1560.
`dsv4-nvfp4-draft-moe` is DeepSeek-NVFP4-only. `poc-householder-compile` targets
the old in-tree `vllm/poc/`, which the plugin line does not have.

Take them from `release/v0.25.1`, not from `poc-sampler-residual-v0.25`: that
branch predates #92 and still pins the V1 runner, which the canonical stack
stopped doing once the V2 replay hooks landed.

## Base

`vllm/vllm-openai:glm53-flash`. GLM-5.3-Flash support is not in any upstream
release — it lives in the open PR `vllm-project/vllm#53906`, whose diff reaches
into `.cu` and cmake, so the kernels cannot come from a Python-only overlay and
the base has to be a pre-built image.

That image was built from an untagged tree: it reports
`0.1.dev20051+g487ecf187`, and that commit resolves in neither
`vllm-project/vllm` nor `ZJY0516/vllm`. Before trusting the overlay, every
`vllm/**.py` this stack touches was compared byte-for-byte against
`ZJY0516:glm-release`: 19 of 21 identical, the other two (`vllm/validation.py`,
`vllm/v1/worker/gpu/sample/replay.py`) ours and absent upstream. Redo that
comparison if the base tag is ever moved — the overlay assumes it.

## Conflicts seen on this port

* `sampling_params.py`, `gpu_input_batch.py`, `gpu_model_runner.py`,
  `chat_completion/protocol.py` — upstream added a field next to one of ours
  (`logprob_token_ids`, `use_replayssm`, `slot_mapping_modes`,
  `ec_transfer_params`). Keep both; the additions are independent.
* `topk_topp_sampler.py` — upstream replaced the literal mode tuple with
  `PROCESSED_LOGPROBS_MODES` and grew a `use_fp64_gumbel` branch. Take the
  upstream form of the assert and thread `need_processed_logprobs` through the
  new branch too.
* `completion/protocol.py` — upstream rewrote the same `logprobs=` line the
  patch touches (`None if self.logprob_token_ids else self.logprobs`). Keep
  upstream's expression and add `logprobs_mode` beside it.
* `chat_completion/serving.py` — the patch's hunk carried an
  `is_mistral_tool_parser` import that upstream has since dropped. Take only
  `validate_enforced_token_ids`; re-adding the other import leaves it unused.
* `gpu/sample/sampler.py` — `ReplayState` now sits beside
  `ThinkingBudgetState`; the per-request logprobs-mode branch goes first and
  upstream's mode test becomes the `elif`.
* `gpu/spec_decode/rejection_sampler.py` — upstream folded logprobs into
  `_verify_in_chunks`, which now returns them. Keep that call and graft only
  the replay override after it; `_get_logprobs_tensors` is gone.

## Plugin

`gonka_poc._compat` dispatches on the installed vLLM minor and refuses to load
on an unregistered one rather than falling back — a stale shim can corrupt PoC
output instead of failing. So a new base needs a registered shim:
`kaitakuai/gonka-vllm-plugins@v0.1.4` adds `_compat/v0_28.py`.

Every private surface the 0.25 shim depends on was checked against this base
and is unchanged: `CommonAttentionMetadata` (all fields the shim passes),
`GPUModelRunner.kv_caches`, `EngineCore` with `self.scheduler`,
`Scheduler.kv_cache_manager`, `KVCacheManager.block_pool`, `BlockPool`'s
`get_new_blocks`/`free_blocks`/`get_num_free_blocks`, and
`output_processor.request_states` on `AsyncLLM`.

Because the base reports a version naming no minor, `Dockerfile.gonka-poc`
restates it after the overlay so the dispatch resolves; the base's own string
stays in the rewritten `_version.py`.

## Building

`.github/workflows/build-vllm-poc.yml`, `workflow_dispatch` from this branch.
Publishes `ghcr.io/kaitakuai/vllm-poc:<tag>` plus an immutable `-<sha9>` tag —
the Stage-2 name `mlnode-foundry` accepts. Overridable: `base_image`,
`image_tag`, `expected_vllm_version`, `gonka_poc_ref`, `gonka_poc_repo`.

## Not verified

Nothing here has run on a GPU yet. The build proves the overlay applies, the
plugin installs and the versions line up; it proves nothing about PoC output.
Before this image goes anywhere near the network it needs the usual pass:
self-validation, nonce/min, and a replay cross-check against a known-good box.
