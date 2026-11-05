# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the "tape" rejection sample method of Model Runner V2.

Tape verification keeps a draft while it equals the non-speculative seeded draw
(the "tape") of its row, so the output equals non-speculative sampling. These
tests run on CPU: the Gumbel kernel is replaced by a fake that returns chosen
tape tokens and records its arguments.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.spec_decode import rejection_sampler as rs_mod
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

DEVICE = "cpu"
VOCAB = 16
NUM_SPEC = 3


def _i32(x) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.int32, device=DEVICE)


def _i64(x) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.int64, device=DEVICE)


def _setup(monkeypatch, tape_fn):
    """Build a tape-mode sampler; `tape_fn(pos) -> tape tokens` fakes the draw."""
    calls: dict = {}
    marker_logits = torch.randn(1, VOCAB)

    def fake_gumbel(logits, idx_map, temperature, seed, pos, **kwargs):
        calls["logits"] = logits
        calls["args"] = (idx_map, temperature, seed, pos)
        calls["kwargs"] = kwargs
        return _i64(tape_fn(pos))

    def fake_top_k_top_p(logits, k, p):
        calls["top_k_top_p"] = (logits, k, p)
        return logits

    def must_not_run(*args, **kwargs):
        pytest.fail("rejection_sample must not run in tape mode")

    monkeypatch.setattr(rs_mod, "gumbel_sample", fake_gumbel, raising=False)
    monkeypatch.setattr(rs_mod, "apply_top_k_top_p", fake_top_k_top_p, raising=False)
    monkeypatch.setattr(rs_mod, "rejection_sample", must_not_run)

    states = SimpleNamespace(
        temperature=SimpleNamespace(gpu=torch.ones(4)),
        seeds=SimpleNamespace(gpu=_i64([11, 12, 13, 14])),
        get_top_k_top_p=lambda *a: (None, None),
    )

    def apply_params(logits, *args, **kwargs):
        calls["apply_kwargs"] = kwargs
        return marker_logits

    sampler = SimpleNamespace(
        apply_sampling_params=apply_params,
        sampling_states=states,
        use_fp64_gumbel=False,
        logprobs_mode="raw_logprobs",
    )
    spec_config = SimpleNamespace(
        num_speculative_tokens=NUM_SPEC,
        enable_adaptive_verification=False,
        rejection_sample_method="tape",
        synthetic_acceptance_rates=None,
    )
    rejection_sampler = RejectionSampler(sampler, spec_config, torch.device(DEVICE))
    calls["marker_logits"] = marker_logits
    return rejection_sampler, calls


def _verify(rejection_sampler, drafts, pos, cu=(0, 4)):
    num_reqs = len(cu) - 1
    n = cu[-1]
    return rejection_sampler._verify(
        torch.randn(n, VOCAB),
        None,
        _i32(drafts),
        _i64(pos),
        _i32(list(cu)),
        _i32(list(range(num_reqs))),
        np.arange(num_reqs),
        _i32([0] * n),
        _i32([0] * n),
    )


def _run(monkeypatch, tape, drafts):
    rejection_sampler, calls = _setup(monkeypatch, lambda pos: tape)
    _, sampled, num_sampled = _verify(rejection_sampler, drafts, [10, 11, 12, 13])
    return sampled, num_sampled, calls


def test_tape_partial_accept_emits_tape_token(monkeypatch):
    """The first mismatching draft is replaced by its row's tape token."""
    sampled, num_sampled, _ = _run(monkeypatch, [5, 9, 2, 7], [3, 5, 9, 4])
    assert num_sampled.tolist() == [3]
    assert sampled[0, :3].tolist() == [5, 9, 2]


def test_tape_all_accepted_appends_bonus(monkeypatch):
    sampled, num_sampled, _ = _run(monkeypatch, [5, 9, 2, 7], [3, 5, 9, 2])
    assert num_sampled.tolist() == [4]
    assert sampled[0].tolist() == [5, 9, 2, 7]


def test_tape_placeholder_draft_rejected(monkeypatch):
    """A negative draft id never matches, even if the tape could."""
    sampled, num_sampled, _ = _run(monkeypatch, [5, 9, 2, 7], [3, -1, 9, 2])
    assert num_sampled.tolist() == [1]
    assert sampled[0, :1].tolist() == [5]


def test_tape_output_equals_nonspeculative_tape(monkeypatch):
    """Two requests of different lengths: each emits a prefix of its own tape."""
    full = {10: 1, 11: 2, 12: 3, 13: 4, 20: 8, 21: 9}

    def tape_fn(pos):
        return [full[int(p)] for p in pos]

    rejection_sampler, _ = _setup(monkeypatch, tape_fn)
    drafts = [0, 1, 2, 6, 0, 8]
    _, sampled, num_sampled = rejection_sampler._verify(
        torch.randn(6, VOCAB),
        None,
        _i32(drafts),
        _i64([10, 11, 12, 13, 20, 21]),
        _i32([0, 4, 6]),
        _i32([0, 1]),
        np.arange(2),
        _i32([0] * 6),
        _i32([0] * 6),
    )
    assert num_sampled.tolist() == [3, 2]
    assert sampled[0, :3].tolist() == [1, 2, 3]
    assert sampled[1, :2].tolist() == [8, 9]


def test_tape_gumbel_args_match_nonspeculative(monkeypatch):
    rejection_sampler, calls = _setup(monkeypatch, lambda pos: [5, 9, 2, 7])
    states = rejection_sampler.sampler.sampling_states
    pos = [10, 11, 12, 13]
    _, _, _ = _verify(rejection_sampler, [3, 5, 9, 2], pos)
    assert calls["apply_kwargs"] == {"skip_top_k_top_p": True}
    assert calls["top_k_top_p"][0] is calls["marker_logits"]
    assert calls["logits"] is calls["marker_logits"]
    _, temperature, seed, got_pos = calls["args"]
    assert temperature is states.temperature.gpu
    assert seed is states.seeds.gpu
    assert got_pos.tolist() == pos
    assert calls["kwargs"] == {
        "apply_temperature": False,
        "is_drafting": False,
        "use_fp64": False,
    }


def _spec_config(**kwargs):
    from vllm.config import SpeculativeConfig

    return SpeculativeConfig(
        method="ngram",
        num_speculative_tokens=2,
        rejection_sample_method="tape",
        **kwargs,
    )


def test_tape_config_conflicts_rejected():
    cfg = _spec_config(draft_sample_method="probabilistic")
    assert cfg.rejection_sample_method == "tape"
    with pytest.raises(ValueError, match="enable_adaptive_verification"):
        _spec_config(
            draft_sample_method="probabilistic", enable_adaptive_verification=True
        )
    with pytest.raises(ValueError, match="draft_sample_method"):
        _spec_config(draft_sample_method="greedy")


def test_tape_refused_by_v1_model_runner():
    from vllm.config import VllmConfig

    fake = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1, enable_batch_sharded_sampling=False
        ),
        speculative_config=SimpleNamespace(
            method="dspark",
            rejection_sample_method="tape",
            enable_adaptive_verification=False,
        ),
        model_config=None,
        _dflash_needs_multi_kv_group=lambda: False,
        _is_dflash2_draft=lambda: False,
    )
    unsupported = VllmConfig._get_v1_model_runner_unsupported_features(fake)
    assert "tape rejection sampling" in unsupported
    fake.speculative_config.rejection_sample_method = "standard"
    unsupported = VllmConfig._get_v1_model_runner_unsupported_features(fake)
    assert "tape rejection sampling" not in unsupported


def test_tape_with_enforced_tokens_emits_enforced_token(monkeypatch):
    """An enforced request has no drafts: its tape token is overridden."""
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu import buffer_utils
    from vllm.v1.worker.gpu.sample.enforced_tokens import EnforcedTokensState

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

    req_states = SimpleNamespace(
        max_num_reqs=4,
        max_model_len=2048,
        vocab_size=VOCAB,
        device=torch.device(DEVICE),
        total_len=SimpleNamespace(gpu=torch.full((4,), 5, dtype=torch.int32)),
        prompt_len=SimpleNamespace(gpu=torch.full((4,), 5, dtype=torch.int32)),
    )
    enforced = EnforcedTokensState(req_states)
    enforced.add_request(0, SamplingParams(enforced_token_ids=[8]))
    enforced.apply_staged_writes()

    rejection_sampler, _ = _setup(monkeypatch, lambda pos: [5])
    rejection_sampler.sampler.enforced_tokens_state = enforced
    input_batch = SimpleNamespace(
        num_reqs=1,
        cu_num_logits=_i32([0, 1]),
        cu_num_logits_np=np.array([0, 1]),
        idx_mapping=_i32([0]),
        idx_mapping_np=np.array([0]),
        expanded_idx_mapping=_i32([0]),
        expanded_local_pos=_i32([0]),
    )
    sampled, num_sampled, logprobs = rejection_sampler._verify_in_chunks(
        torch.randn(1, VOCAB), input_batch, None, _i32([3]), _i64([10]), 1 << 20, -1
    )
    assert logprobs is None
    assert num_sampled.tolist() == [1]
    assert sampled[0, 0].item() == 8
