# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("xgrammar")

from vllm.v1.structured_output.backend_xgrammar import (  # noqa: E402
    XgrammarGrammar,
)


def test_failed_grammar_is_unconstrained() -> None:
    grammar = object.__new__(XgrammarGrammar)
    grammar._grammar_failed = True

    bitmask = torch.zeros((2, 4), dtype=torch.int32)
    bitmask[1][0] = 0b0010
    grammar.fill_bitmask(bitmask, 1)

    assert torch.equal(bitmask[1], torch.full((4,), -1, dtype=torch.int32))
    assert torch.equal(bitmask[0], torch.zeros(4, dtype=torch.int32))
    assert grammar.validate_tokens([11, 22, 33]) == [11, 22, 33]
