# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract tests for the sampler surfaces the PoC residual depends on.

The residual in this tree adds a few hooks to the V1 sampling stack that an
out-of-tree plugin consumes: per-request ``logprobs_mode`` and
``enforced_token_ids``, ``need_processed_logprobs`` threading through the
top-k/top-p sampler, the matching ``InputBatch`` bookkeeping, and
structured-output graceful degradation. All of them live on private classes,
so a refactor can move them without producing an import error: the plugin
keeps loading and silently stops enforcing tokens or stops returning the
requested logprobs.

These tests pin the shape of each touched surface. A failure means the
surface moved and the corresponding residual patch must be re-applied at its
new location; every assertion message names the file to look at.

Scope: read-only inspection of vllm modules. No GPU, no engine startup, no
forward pass.
"""

from __future__ import annotations

import importlib
import inspect

import pytest


def test_completion_request_forwards_logprobs_mode() -> None:
    mod = importlib.import_module("vllm.entrypoints.openai.completion.protocol")
    request = mod.CompletionRequest(
        model="test", prompt="test", logprobs_mode="processed_logprobs"
    )
    assert request.to_sampling_params(1).logprobs_mode == "processed_logprobs"


def test_sampling_params_has_poc_fields() -> None:
    """Pin the per-request SamplingParams fields the plugin sets."""
    pytest.importorskip("vllm")
    mod = importlib.import_module("vllm.sampling_params")
    cls = getattr(mod, "SamplingParams", None)
    assert cls is not None, "vllm.sampling_params.SamplingParams missing"

    # SamplingParams is a msgspec.Struct, whose declared field set is
    # __struct_fields__; msgspec generates __init__ dynamically, so
    # inspect.signature(cls.__init__) only reports (*args, **kwargs). Fall
    # back to the dataclass and signature views so the assertion survives a
    # future change of class machinery.
    field_names = set(getattr(cls, "__struct_fields__", None) or ())
    if not field_names:
        field_names = set(getattr(cls, "__dataclass_fields__", None) or ())
    if not field_names:
        field_names = set(inspect.signature(cls.__init__).parameters)
    for name in ("logprobs_mode", "enforced_token_ids"):
        assert name in field_names, (
            f"SamplingParams.{name} is not a declared field — re-apply the "
            f"residual patch to vllm/sampling_params.py."
        )

    # from_optional is the path the API entrypoints take: a field that is
    # declared but not forwarded here never reaches the sampler.
    from_optional_params = set(inspect.signature(cls.from_optional).parameters)
    for name in ("logprobs_mode", "enforced_token_ids"):
        assert name in from_optional_params, (
            f"SamplingParams.from_optional no longer forwards {name} — "
            f"requests carrying it would be silently dropped before the "
            f"sampler sees it."
        )

    # Both fields must be opt-in: requests that do not set them must behave
    # exactly like unpatched vLLM.
    defaults = cls()
    assert defaults.logprobs_mode is None
    assert defaults.enforced_token_ids is None

    params = cls(
        logprobs=1,
        logprobs_mode="processed_logprobs",
        enforced_token_ids=[7, 11],
    )
    assert params.logprobs_mode == "processed_logprobs"
    assert params.enforced_token_ids == [7, 11]

    # The value check runs from __post_init__; without it an unusable mode
    # reaches the sampler and is resolved to the deployment default.
    with pytest.raises(ValueError):
        cls(logprobs_mode="not_a_mode")


def test_sampling_metadata_has_poc_fields() -> None:
    """Pin the sampler-side SamplingMetadata fields the residual populates."""
    pytest.importorskip("vllm")
    mod = importlib.import_module("vllm.v1.sample.metadata")
    cls = getattr(mod, "SamplingMetadata", None)
    assert cls is not None, "vllm.v1.sample.metadata.SamplingMetadata missing"

    dc_fields = getattr(cls, "__dataclass_fields__", None)
    assert dc_fields is not None, (
        "SamplingMetadata is no longer a dataclass — re-check how "
        "vllm/v1/worker/gpu_input_batch.py constructs it."
    )

    # batch_logprobs_mode drives the mode resolution in Sampler.forward, and
    # logprobs_is_processed selects raw vs processed per row of a mixed
    # batch; enforced_next_token_ids drives the post-sampling override.
    for name in (
        "batch_logprobs_mode",
        "logprobs_is_processed",
        "enforced_next_token_ids",
    ):
        assert name in dc_fields, (
            f"SamplingMetadata.{name} missing — Sampler.forward would fall "
            f"back to the deployment-wide logprobs mode and skip the "
            f"enforced-token override without failing. Re-apply the residual "
            f"patch to vllm/v1/sample/metadata.py."
        )
        assert dc_fields[name].default is None, (
            f"SamplingMetadata.{name} no longer defaults to None — every "
            f"other construction site of SamplingMetadata must keep working "
            f"without the residual fields."
        )


def test_sampler_call_accepts_enforced_tokens() -> None:
    """Pin the Sampler entry points that carry the residual state."""
    pytest.importorskip("vllm")
    mod = importlib.import_module("vllm.v1.sample.sampler")
    cls = getattr(mod, "Sampler", None)
    assert cls is not None, "vllm.v1.sample.sampler.Sampler missing"
    assert hasattr(cls, "forward"), (
        "Sampler.forward missing — the method may have been renamed to "
        "__call__ or split; the residual hooks in it need re-porting."
    )

    forward_params = set(inspect.signature(cls.forward).parameters)
    assert "sampling_metadata" in forward_params, (
        f"Sampler.forward no longer takes sampling_metadata; present params "
        f"= {sorted(forward_params)!r}. The residual reads the per-request "
        f"mode and the enforced tokens off it."
    )

    sample_params = set(inspect.signature(cls.sample).parameters)
    assert "need_processed_logprobs" in sample_params, (
        f"Sampler.sample no longer takes need_processed_logprobs; present "
        f"params = {sorted(sample_params)!r}. Without it a per-request "
        f"processed-logprobs override cannot reach the top-k/top-p sampler."
    )

    meta_mod = importlib.import_module("vllm.v1.sample.metadata")
    meta_cls = getattr(meta_mod, "SamplingMetadata", None)
    assert meta_cls is not None, "vllm.v1.sample.metadata.SamplingMetadata missing"
    meta_fields = getattr(meta_cls, "__dataclass_fields__", None) or {}
    assert "enforced_next_token_ids" in meta_fields, (
        "Sampler.forward takes sampling_metadata but SamplingMetadata lacks "
        "enforced_next_token_ids — validation replay would be a no-op."
    )

    # The override is a tensor read inside forward(), observable only during
    # a real forward pass on device tensors, which this contract test does
    # not run; the source is the only place its removal shows up.
    forward_src = inspect.getsource(cls.forward)
    assert "enforced_next_token_ids" in forward_src, (
        "Sampler.forward does not reference enforced_next_token_ids — the "
        "post-sampling override was dropped; re-apply the residual patch to "
        "vllm/v1/sample/sampler.py."
    )


def test_topk_topp_sampler_need_processed_logprobs() -> None:
    """Pin need_processed_logprobs on every TopKTopPSampler forward path."""
    pytest.importorskip("vllm")
    mod = importlib.import_module("vllm.v1.sample.ops.topk_topp_sampler")
    cls = getattr(mod, "TopKTopPSampler", None)
    assert cls is not None, (
        "vllm.v1.sample.ops.topk_topp_sampler.TopKTopPSampler missing — the "
        "module was restructured and the residual threading needs re-porting."
    )

    # Every backend path must accept the kwarg: the dispatch target depends
    # on the platform, so one missing path is a runtime TypeError there and
    # nowhere else.
    for method_name in (
        "forward_native",
        "forward_cuda",
        "forward_cpu",
        "forward_hip",
        "forward_xpu",
    ):
        fn = getattr(cls, method_name, None)
        assert fn is not None, (
            f"TopKTopPSampler.{method_name} missing — this forward path was "
            f"renamed or consolidated; re-port the need_processed_logprobs "
            f"threading to its replacement."
        )
        params = set(inspect.signature(fn).parameters)
        assert "need_processed_logprobs" in params, (
            f"TopKTopPSampler.{method_name} no longer takes "
            f"need_processed_logprobs; present params = {sorted(params)!r}. "
            f"On this path a mixed-mode batch would silently return raw "
            f"logprobs where processed ones were requested."
        )

    # sample() is the wrapper Sampler.sample calls: it falls back to
    # forward_native when the active fast path (FlashInfer, aiter) cannot
    # produce processed logprobs.
    sample = getattr(cls, "sample", None)
    assert sample is not None, (
        "TopKTopPSampler.sample missing — Sampler.sample would call the "
        "platform forward directly and raise TypeError on the fast paths "
        "when processed logprobs are requested."
    )
    sample_params = set(inspect.signature(sample).parameters)
    assert "need_processed_logprobs" in sample_params, (
        f"TopKTopPSampler.sample no longer takes need_processed_logprobs; "
        f"present params = {sorted(sample_params)!r}."
    )


def test_input_batch_logprobs_modes_dict() -> None:
    """Pin the InputBatch bookkeeping that feeds SamplingMetadata."""
    pytest.importorskip("vllm")
    mod = importlib.import_module("vllm.v1.worker.gpu_input_batch")
    cls = getattr(mod, "InputBatch", None)
    assert cls is not None, (
        "vllm.v1.worker.gpu_input_batch.InputBatch missing — the module was "
        "restructured; re-apply the residual bookkeeping at its new location."
    )

    init_params = set(inspect.signature(cls.__init__).parameters)
    assert "logprobs_mode_default" in init_params, (
        f"InputBatch.__init__ no longer takes logprobs_mode_default; present "
        f"params = {sorted(init_params)!r}. The model runner passes the "
        f"deployment-wide mode through it as the per-request fallback."
    )

    mode_attr = inspect.getattr_static(cls, "batch_logprobs_mode", None)
    assert isinstance(mode_attr, property), (
        "InputBatch.batch_logprobs_mode is not a property — the aggregation "
        "of per-request modes into a batch-level mode (including 'mixed') "
        "is gone; SamplingMetadata.batch_logprobs_mode cannot be filled."
    )

    assert hasattr(cls, "_build_enforced_tensor"), (
        "InputBatch._build_enforced_tensor missing — the per-step enforced "
        "token tensor is no longer built, so validation replay is a no-op."
    )

    # logprobs_modes and req_enforced_token_ids are instance attributes, and
    # InputBatch cannot be constructed here (it needs a full VllmConfig, a KV
    # cache spec and a device), so check the constructor source instead.
    init_src = inspect.getsource(cls.__init__)
    for needle in ("self.logprobs_modes", "self.req_enforced_token_ids"):
        assert needle in init_src, (
            f"InputBatch.__init__ no longer initializes {needle} — "
            f"add_request cannot record per-request modes or enforced "
            f"tokens; re-apply the residual patch to "
            f"vllm/v1/worker/gpu_input_batch.py."
        )


def test_structured_output_graceful_degradation_hook() -> None:
    """Pin the grammar surfaces the graceful-degradation hook patches."""
    pytest.importorskip("vllm")

    backend = importlib.import_module("vllm.v1.structured_output.backend_xgrammar")
    grammar_cls = getattr(backend, "XgrammarGrammar", None)
    assert grammar_cls is not None, (
        "vllm.v1.structured_output.backend_xgrammar.XgrammarGrammar missing "
        "— the backend was relocated or renamed; re-apply the residual patch "
        "at its new location."
    )

    # Enforced tokens can conflict with the grammar FSM. The hook turns that
    # rejection into a per-request disable instead of a hard failure, using
    # a _grammar_failed flag that rollback() and fill_bitmask() also honor.
    grammar_fields = getattr(grammar_cls, "__dataclass_fields__", None) or {}
    assert "_grammar_failed" in grammar_fields, (
        "XgrammarGrammar._grammar_failed missing — a grammar rejection would "
        "again abort the request instead of degrading gracefully."
    )
    assert grammar_fields["_grammar_failed"].default is False, (
        "XgrammarGrammar._grammar_failed must default to False so grammar "
        "enforcement stays on until a token is actually rejected."
    )
    for method_name in ("accept_tokens", "rollback", "fill_bitmask", "reset"):
        assert hasattr(grammar_cls, method_name), (
            f"XgrammarGrammar.{method_name} missing — the grammar interface "
            f"drifted and the degradation hook no longer covers every entry "
            f"point that touches the matcher."
        )

    mgr_mod = importlib.import_module("vllm.v1.structured_output")
    mgr_cls = getattr(mgr_mod, "StructuredOutputManager", None)
    assert mgr_cls is not None, (
        "vllm.v1.structured_output.StructuredOutputManager missing — the "
        "module was restructured; re-apply the residual patch."
    )
    bitmask_fn = getattr(mgr_cls, "grammar_bitmask", None)
    assert bitmask_fn is not None, (
        "StructuredOutputManager.grammar_bitmask missing — the patched "
        "bitmask path moved; find the new call site of accept_tokens."
    )
    # The patch replaces a raise at this call site with a warning plus a
    # bitmask disable. There is no structural way to observe a call site, and
    # exercising it needs a compiled grammar and a real speculative-decode
    # batch, so pin it by source.
    assert "accept_tokens" in inspect.getsource(bitmask_fn), (
        "StructuredOutputManager.grammar_bitmask no longer calls "
        "accept_tokens — the patched call site moved; re-apply the residual "
        "patch to vllm/v1/structured_output/__init__.py at its new location."
    )
