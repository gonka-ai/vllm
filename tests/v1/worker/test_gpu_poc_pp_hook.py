# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sample_tokens must let non-last PP ranks join the decode-PoC prev_k receive."""

from types import SimpleNamespace

import pytest

from vllm.v1.worker import gpu_model_runner as gmr
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def _make(calls, *, is_last, async_sched=True, bridge=True):
    fake = SimpleNamespace(
        execute_model_state=None,
        kv_connector_output=None,
        use_async_scheduling=async_sched,
        _pp_receive_prev_sampled_token_ids_to_input_batch=lambda: calls.append(
            "token_recv"
        ),
    )
    if bridge:
        fake._poc_bridge = SimpleNamespace(
            pp_recv_prev_k=lambda: calls.append("prev_k")
        )
    return fake


@pytest.fixture
def set_rank(monkeypatch):
    def _set(is_last):
        monkeypatch.setattr(
            gmr, "get_pp_group", lambda: SimpleNamespace(is_last_rank=is_last)
        )

    return _set


def test_non_last_rank_calls_pp_recv_prev_k(set_rank):
    set_rank(False)
    calls = []
    out = GPUModelRunner.sample_tokens(_make(calls, is_last=False), None)
    assert calls.count("prev_k") == 1
    assert out == gmr.ModelRunnerOutput.with_kv_conn_output_only(None)


def test_pp_recv_after_async_token_receive(set_rank):
    set_rank(False)
    calls = []
    GPUModelRunner.sample_tokens(_make(calls, is_last=False), None)
    assert calls == ["token_recv", "prev_k"]


def test_last_rank_does_not_call_pp_recv(set_rank):
    set_rank(True)
    calls = []
    GPUModelRunner.sample_tokens(_make(calls, is_last=True), None)
    assert calls == []


def test_no_bridge_no_call(set_rank):
    set_rank(False)
    calls = []
    GPUModelRunner.sample_tokens(_make(calls, is_last=False, bridge=False), None)
    assert calls == ["token_recv"]
