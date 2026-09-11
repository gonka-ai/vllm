# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract pins for the residual engine seams.

Each pin asserts a non-negotiable seam behavior is present in engine source.
Every one of these encodes an incident: losing it reproduces a shipped bug.
Brittle by design -- a refactor that moves one MUST consciously re-pin it.
"""

import pathlib

R = pathlib.Path(__file__).resolve().parents[3]


def _src(p):
    return (R / p).read_text()


def test_async_placeholder_cap_exempts_poc():
    """502-steps/s incident: under async, the loop-top output-placeholder cap
    treated PoC rows (engine max_tokens=1, no sampled tokens ever) as finished
    and scheduled them every other step."""
    s = _src("vllm/v1/core/sched/scheduler.py")
    i = s.find("num_output_placeholders > 0")
    assert i > 0
    assert "poc_params is None" in s[i - 200 : i], (
        "PoC rows are no longer exempt from the async placeholder cap"
    )


def test_poc_finish_precedes_token_processing():
    """check_stop-inertness depends on ordering: the PoC artifact-finish branch
    must run BEFORE generated-token handling in update_from_output."""
    s = _src("vllm/v1/core/sched/scheduler.py")
    assert (
        0
        < s.find("poc_params is not None:\n                # PoC finish")
        < s.find("req_id_to_index[req_id]")
    )


def test_poc_requests_get_unique_cache_salt():
    """Prefix-cache incident: identical dummy prompts cross-matched between
    nonces -> shared KV -> consensus corruption."""
    s = _src("vllm/v1/engine/async_llm.py")
    i = s.find("poc_params=poc_params")
    assert "cache_salt=request_id" in s[i - 600 : i], (
        "PoC requests no longer carry a unique cache salt"
    )


def test_v2_sampler_slot_neutralized_for_poc():
    """V2 engine-death incident: reused slot columns kept the previous
    occupant's prompt-logprobs flags for a PoC row."""
    s = _src("vllm/v1/worker/gpu/model_runner.py")
    assert "clear_slot" in s
    assert "elif self.is_last_pp_rank:" in s


def test_v2_hooks_precede_full_replay_branch():
    """Stale-metas incident: hooks placed in the else-branch were bypassed by
    FULL cudagraph replay steps."""
    s = _src("vllm/v1/worker/gpu/model_runner.py")
    assert (
        0 < s.find("_poc_bridge.pre_forward") < s.find("cg_mode == CUDAGraphMode.FULL")
    )


def test_native_uses_class_level_patching():
    """torch.compile incident: instance-level forward patches are ignored in
    compiled regions; module replacement breaks the compiled param map."""
    s = _src("vllm/poc/native.py")
    assert "_install_poc_patch" in s
    assert "cls.forward = _poc_forward" in s
    assert "layers[i] = PoCLayerWrapper" not in s


def test_decode_pool_sized_from_resolved_cap():
    """Empty-pool incident: manager sized from the raw config value (0 under
    lazy AUTO) -> no decode state -> prefill-only artifacts."""
    s = _src("vllm/poc/mixed_decode.py")
    i = s.find("def get_decode_manager")
    assert "resolve_poc_max_batch_size" in s[i : i + 900]
