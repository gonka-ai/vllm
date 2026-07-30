# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import pytest

pytest.importorskip("vllm")

import vllm.validation as validation  # noqa: E402


def test_replay_ids_are_range_checked() -> None:
    vocab_size = 32000
    validation.validate_enforced_token_ids([0, 1, vocab_size - 1], vocab_size)

    for bad in ([vocab_size], [-1], [-5]):
        with pytest.raises(ValueError):
            validation.validate_enforced_token_ids(bad, vocab_size)


def test_replay_payload_is_size_bounded() -> None:
    with pytest.raises(ValueError):
        validation.EnforcedTokens(
            tokens=[validation.EnforcedToken(token="1")]
            * (validation.MAX_ENFORCED_TOKENS + 1),
        )
    with pytest.raises(ValueError):
        validation.EnforcedToken(
            token="1",
            top_tokens=["2"] * (validation.MAX_TOP_TOKENS_PER_POSITION + 1),
        )
