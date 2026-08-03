# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The V2 sampler's replay bookkeeping, exercised without a GPU or engine.

ReplayState touches exactly two RequestState tensors (total_len,
prompt_len), so a two-field stand-in suffices; the test names state the
pinned semantics.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vllm.v1.worker.gpu.sample.replay import ReplayState  # noqa: E402


class _Field:
    def __init__(self, values):
        self.gpu = torch.tensor(values, dtype=torch.int32)


class _ReqStates:
    def __init__(self, total_len, prompt_len):
        self.total_len = _Field(total_len)
        self.prompt_len = _Field(prompt_len)


class _Params:
    def __init__(self, etids=None, mode=None):
        self.enforced_token_ids = etids
        self.logprobs_mode = mode


def test_enforced_follows_emitted_position_and_exhausts_to_minus_one():
    # Slot 0: replay [11, 22]; slot 1: plain request.
    # total_len - prompt_len => emitted so far: slot0 has 1, slot1 has 5.
    rs = ReplayState(
        4,
        _ReqStates(total_len=[6, 15, 0, 0], prompt_len=[5, 10, 0, 0]),
        torch.device("cpu"),
    )
    rs.add_request(0, 5, _Params(etids=[11, 22]))
    rs.add_request(1, 10, _Params())

    rows = torch.tensor([0, 1], dtype=torch.int64)
    out = rs.enforced_for_batch(rows).tolist()
    assert out == [22, -1], "slot0 emitted 1 token, so position 1 => id 22"

    # Advance slot0 past the end of the replay: nothing left to enforce.
    rs.req_states.total_len.gpu[0] = 7
    assert rs.enforced_for_batch(rows).tolist() == [-1, -1]


def test_batch_predicate_is_slot_scoped():
    rs = ReplayState(4, _ReqStates([0] * 4, [0] * 4), torch.device("cpu"))
    rs.add_request(2, 1, _Params(etids=[7]))
    import numpy as np

    assert rs.batch_has_replay(np.array([2, 3]))
    assert not rs.batch_has_replay(np.array([0, 1, 3]))
    # Slot reuse must clear the flag.
    rs.add_request(2, 1, _Params())
    assert not rs.batch_has_replay(np.array([2]))


def test_processed_mask_mixes_override_with_engine_default():
    rs = ReplayState(4, _ReqStates([0] * 4, [0] * 4), torch.device("cpu"))
    rs.add_request(0, 1, _Params(mode="raw_logprobs"))
    rs.add_request(1, 1, _Params(mode="processed_logprobs"))
    rs.add_request(2, 1, _Params())  # no override -> engine default

    rows = torch.tensor([0, 1, 2], dtype=torch.int64)
    assert rs.processed_rows_mask(rows, engine_processed=True).tolist() == [
        False,
        True,
        True,
    ]
    assert rs.processed_rows_mask(rows, engine_processed=False).tolist() == [
        False,
        True,
        False,
    ]


def test_sampler_calls_replay_methods_with_the_signatures_they_declare():
    """The sampler's call sites must match ReplayState's signatures.

    Regression: the mode-mask call site passed three positional args to a
    two-arg method, and passed the host-side index array where the body
    indexes GPU buffers. Every replay request died with TypeError inside
    the engine, while the tests above stayed green because they call the
    methods directly. Signatures alone are checked here — no GPU, no
    engine — so the mismatch cannot come back unnoticed.
    """
    import ast
    import inspect
    from pathlib import Path

    from vllm.v1.worker.gpu.sample import replay as replay_mod

    sampler_src = Path(inspect.getfile(replay_mod)).with_name("sampler.py").read_text()
    tree = ast.parse(sampler_src)

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "replay_state"
    ]
    assert calls, "sampler.py no longer calls replay_state — did the hook move?"

    for call in calls:
        assert isinstance(call.func, ast.Attribute)
        name = call.func.attr
        method = getattr(ReplayState, name, None)
        assert method is not None, f"sampler.py calls ReplayState.{name}, which is gone"
        params = list(inspect.signature(method).parameters)[1:]  # drop self
        passed = len(call.args) + len(call.keywords)
        assert passed <= len(params), (
            f"sampler.py passes {passed} args to ReplayState.{name}, "
            f"which declares {len(params)} ({', '.join(params)})"
        )
        # Positional args must line up with the declared parameter names:
        # passing idx_mapping_np where the body indexes GPU buffers is the
        # exact mistake this guards against.
        for pos, arg in enumerate(call.args):
            if isinstance(arg, ast.Name) and params[pos].endswith("idx_mapping"):
                assert arg.id.endswith("idx_mapping"), (
                    f"ReplayState.{name} expects {params[pos]!r} at position "
                    f"{pos}, but sampler.py passes {arg.id!r}"
                )
