# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Model Runner V2 enforced tokens (validation replay).

A request with ``SamplingParams.enforced_token_ids`` must emit ``ids[s]`` at
output step ``s`` (``ids[-1]`` past the end), as in the V1 sampler. The step is
``total_len - prompt_len``. These tests run on CPU: the override is plain torch
and the GPU-only helpers of ``Sampler.__call__`` are monkeypatched.
"""

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.states import RequestState

DEVICE = "cpu"
VOCAB = 16


@pytest.fixture(autouse=True)
def _cpu_buffers(monkeypatch):
    """No pinned memory or Triton on a CPU-only host: plain buffers and a torch
    stand-in for the staged-write kernel."""
    from vllm.v1.worker.gpu import buffer_utils

    def apply_write(self):
        start = 0
        for idx, col, end in zip(
            self._staged_write_indices,
            self._staged_write_starts,
            self._staged_write_cu_lens,
        ):
            vals = self._staged_write_contents[start:end]
            self.gpu[idx, col : col + len(vals)] = torch.tensor(vals, dtype=self.dtype)
            start = end
        self.clear_staged_writes()

    monkeypatch.setattr(buffer_utils, "is_uva_available", lambda: False)
    monkeypatch.setattr(buffer_utils.StagedWriteTensor, "apply_write", apply_write)


def _i32(x) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.int32, device=DEVICE)


def _i64(x) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.int64, device=DEVICE)


def _req_states(max_num_reqs: int) -> RequestState:
    return cast(
        RequestState,
        SimpleNamespace(
            max_num_reqs=max_num_reqs,
            max_model_len=2048,
            vocab_size=VOCAB,
            device=torch.device(DEVICE),
            total_len=SimpleNamespace(
                gpu=torch.zeros(max_num_reqs, dtype=torch.int32, device=DEVICE)
            ),
            prompt_len=SimpleNamespace(
                gpu=torch.zeros(max_num_reqs, dtype=torch.int32, device=DEVICE)
            ),
        ),
    )


def _set_step(req_states, req_idx: int, step: int, prompt_len: int = 5) -> None:
    req_states.prompt_len.gpu[req_idx] = prompt_len
    req_states.total_len.gpu[req_idx] = prompt_len + step


def _state(max_num_reqs: int = 4):
    from vllm.v1.worker.gpu.sample.enforced_tokens import EnforcedTokensState

    req_states = _req_states(max_num_reqs)
    return EnforcedTokensState(req_states), req_states


def test_step_indexing_and_last_id_repeat():
    """ids[s] for s < len(ids), ids[-1] afterwards."""
    state, rs = _state()
    state.add_request(0, SamplingParams(enforced_token_ids=[3, 4, 5]))
    state.apply_staged_writes()
    idx_mapping = _i32([0])
    expected = [3, 4, 5, 5, 5]
    for step, want in enumerate(expected):
        _set_step(rs, 0, step)
        sampled = _i64([-7])
        state.apply(sampled, idx_mapping, np.array([0]))
        assert sampled.tolist() == [want]


def test_mixed_batch_and_padding_rows_untouched():
    """Unenforced rows and idx_mapping < 0 keep their sampled token."""
    state, rs = _state()
    state.add_request(0, SamplingParams())
    state.add_request(1, SamplingParams(enforced_token_ids=[9, 8]))
    state.apply_staged_writes()
    _set_step(rs, 0, 0)
    _set_step(rs, 1, 1)
    sampled = _i64([11, 12, 13, 14])
    state.apply(sampled, _i32([0, 1, -1, 1]), np.array([0, 1, -1, 1]))
    assert sampled.tolist() == [11, 8, 13, 8]


def test_logprob_is_of_forced_token_under_unmodified_logits(monkeypatch):
    """The override leaves logits alone, so the forced token's logprob is the
    model's own."""
    from vllm.v1.worker.gpu.sample import sampler as sampler_mod

    logits = torch.randn(1, VOCAB)
    original = logits.clone()
    sampler, input_batch, seen = _build_sampler(
        monkeypatch, sampler_mod, logits, sampled_value=0, ids=[7]
    )
    sampler(logits, input_batch)
    assert seen["sampled"].tolist() == [7]
    assert torch.equal(seen["logits"], original)
    logprob = torch.log_softmax(seen["logits"].float(), dim=-1)[0, 7]
    assert logprob == torch.log_softmax(original.float(), dim=-1)[0, 7]


def test_slot_lifecycle_and_overlong_list():
    """Slot reuse clears old ids; an over-long list raises with both numbers."""
    from vllm.v1.worker.gpu.sample.enforced_tokens import MAX_ENFORCED_IDS

    state, rs = _state(2)
    state.add_request(0, SamplingParams(enforced_token_ids=[3, 4]))
    state.apply_staged_writes()
    assert state.num_ids.tolist() == [2, 0]

    state.add_request(0, SamplingParams())
    state.apply_staged_writes()
    _set_step(rs, 0, 0)
    sampled = _i64([42])
    state.apply(sampled, _i32([0]), np.array([0]))
    assert sampled.tolist() == [42]

    too_long = list(range(MAX_ENFORCED_IDS + 1))
    with pytest.raises(ValueError, match=rf"{len(too_long)}.*{MAX_ENFORCED_IDS}"):
        state.add_request(1, SamplingParams(enforced_token_ids=too_long))


def test_sampler_call_emits_enforced_token(monkeypatch):
    """Sampler.__call__ outputs the enforced token (fails when ignored)."""
    from vllm.v1.worker.gpu.sample import sampler as sampler_mod

    logits = torch.randn(1, VOCAB)
    sampler, input_batch, _ = _build_sampler(
        monkeypatch, sampler_mod, logits, sampled_value=0, ids=[6, 2]
    )
    out = sampler(logits, input_batch)
    assert out.sampled_token_ids.tolist() == [[6]]


def test_no_gpu_op_without_enforced_rows(monkeypatch):
    """A batch with no enforced row launches no torch op in either entry point."""
    from vllm.v1.worker.gpu.sample import enforced_tokens as mod

    state, rs = _state()
    state.add_request(0, SamplingParams())
    state.add_request(1, SamplingParams(enforced_token_ids=[9]))
    state.apply_staged_writes()

    def boom(*args, **kwargs):
        raise AssertionError("GPU work launched without enforced rows")

    monkeypatch.setattr(mod, "apply_enforced_tokens", boom)
    monkeypatch.setattr(torch, "where", boom)
    sampled = _i64([[11, 12], [13, 14]])
    num_sampled = _i32([2, 2])
    idx_np = np.array([0, -1])
    state.apply(sampled[:, 0], _i32([0, -1]), idx_np)
    state.apply_to_verified(sampled, num_sampled, _i32([0, -1]), idx_np)
    assert sampled.tolist() == [[11, 12], [13, 14]]
    assert num_sampled.tolist() == [2, 2]


def _build_sampler(monkeypatch, sampler_mod, logits, sampled_value, ids):
    from vllm.v1.worker.gpu import buffer_utils

    # No pinned memory on a CPU-only host: use the explicit-copy buffers.
    monkeypatch.setattr(buffer_utils, "is_uva_available", lambda: False)
    req_states = _req_states(2)
    req_states.prefill_len = SimpleNamespace(gpu=_i32([0, 0]))
    sampler = sampler_mod.Sampler(
        max_num_reqs=2,
        vocab_size=VOCAB,
        device=torch.device(DEVICE),
        req_states=req_states,
    )
    sampler.add_request(0, 5, SamplingParams(enforced_token_ids=ids, logprobs=1))
    sampler.apply_staged_writes()
    _set_step(req_states, 0, 0)

    seen: dict = {}

    def fake_sample(self, logits, *args, **kwargs):
        return _i64([sampled_value]), logits

    def fake_topk(logits, num_logprobs, sampled, *args, **kwargs):
        seen["sampled"] = sampled.clone()
        seen["logits"] = logits.clone()

    monkeypatch.setattr(sampler_mod.Sampler, "sample", fake_sample)
    monkeypatch.setattr(sampler_mod, "compute_topk_scores", fake_topk)
    monkeypatch.setattr(
        sampler_mod,
        "get_num_sampled_and_rejected",
        lambda *a, **k: (_i32([1]), _i32([0])),
    )
    input_batch = SimpleNamespace(
        expanded_idx_mapping=_i32([0]),
        idx_mapping=_i32([0]),
        idx_mapping_np=np.array([0]),
        cu_num_logits_np=np.array([0, 1]),
        cu_num_logits=_i32([0, 1]),
        expanded_local_pos=_i32([0]),
        positions=_i64([5]),
        input_ids=_i32([1]),
        logits_indices=_i64([0]),
        seq_lens=_i32([6]),
        num_reqs=1,
    )
    return sampler, input_batch, seen


def _verify_with_fake_kernel(ids_by_req, steps, sampled, num_sampled):
    """Run RejectionSampler._verify_in_chunks with a stubbed verify kernel.

    Row layout: req 0 has 2 drafts (3 logits), req 1 has none (1 logit),
    req 2 has 2 drafts (3 logits).
    """
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

    state, rs = _state(3)
    for idx, ids in ids_by_req.items():
        state.add_request(idx, SamplingParams(enforced_token_ids=ids))
        _set_step(rs, idx, steps[idx])
    state.apply_staged_writes()

    logits = torch.randn(7, VOCAB)
    rejection_sampler = object.__new__(RejectionSampler)
    rejection_sampler.sampler = SimpleNamespace(
        enforced_tokens_state=state, logprobs_mode="raw_logprobs"
    )
    rejection_sampler.enable_adaptive_verification = False
    seen: dict = {}

    def fake_verify(*args, **kwargs):
        return logits, sampled.clone(), num_sampled.clone()

    def fake_logprobs(sampled, num_sampled, logits, *args):
        seen["sampled"] = sampled.clone()
        seen["logits"] = logits.clone()
        return None

    rejection_sampler._verify = fake_verify
    rejection_sampler._get_logprobs_tensors = fake_logprobs
    input_batch = SimpleNamespace(
        num_reqs=3,
        cu_num_logits_np=np.array([0, 3, 4, 7]),
        cu_num_logits=_i32([0, 3, 4, 7]),
        idx_mapping=_i32([0, 1, 2]),
        idx_mapping_np=np.array([0, 1, 2]),
        expanded_idx_mapping=_i32([0, 0, 0, 1, 2, 2, 2]),
        expanded_local_pos=_i32([0, 1, 2, 0, 0, 1, 2]),
    )
    out = rejection_sampler._verify_in_chunks(
        logits, input_batch, None, _i64([0] * 7), _i64([0] * 7), 100, 1
    )
    return out, seen, logits


def test_rejection_sampler_forces_enforced_row_only():
    """Enforced 0-draft row emits ids[step] with one sampled token; others as is."""
    sampled = _i64([[4, 5, 6], [1, 0, 0], [2, 3, 4]])
    num_sampled = _i32([3, 1, 3])
    (out_sampled, out_num, _), _, _ = _verify_with_fake_kernel(
        {1: [9, 8]}, {1: 1}, sampled, num_sampled
    )
    assert out_sampled.tolist() == [[4, 5, 6], [8, 0, 0], [2, 3, 4]]
    assert out_num.tolist() == [3, 1, 3]


def test_rejection_sampler_logprobs_see_forced_token_and_raw_logits():
    sampled = _i64([[4, 5, 6], [1, 0, 0], [2, 3, 4]])
    num_sampled = _i32([3, 1, 3])
    (_, _, _), seen, logits = _verify_with_fake_kernel(
        {1: [9, 8]}, {1: 0}, sampled, num_sampled
    )
    assert seen["sampled"][1, 0].item() == 9
    assert torch.equal(seen["logits"], logits)


def test_rejection_sampler_enforced_row_with_drafts_is_cut_to_one_token():
    """Defensive rule: an enforced row that still has drafts accepts none."""
    sampled = _i64([[4, 5, 6], [1, 0, 0], [2, 3, 4]])
    num_sampled = _i32([3, 1, 3])
    (out_sampled, out_num, _), _, _ = _verify_with_fake_kernel(
        {2: [7]}, {2: 4}, sampled, num_sampled
    )
    assert out_sampled[2, 0].item() == 7
    assert out_num.tolist() == [3, 1, 1]
    assert out_sampled[0].tolist() == [4, 5, 6]


def test_v2_enforced_tokens_marker():
    from vllm import validation

    assert validation.V2_ENFORCED_TOKENS is True
