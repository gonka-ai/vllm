# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for logprobs validation, ported from Go decentralized-api tests."""

import json
import os

from vllm.entrypoints.openai.protocol import (
    ChatCompletionLogProb,
    ChatCompletionLogProbsContent,
)
from vllm.logprobs_validation import compare_logprobs

TESTDATA_DIR = os.path.join(os.path.dirname(__file__), "logprobs_validation_testdata")


def load_logprobs_content(path: str) -> list[ChatCompletionLogProbsContent]:
    with open(path) as f:
        data = json.load(f)
    content = data["choices"][0]["logprobs"]["content"]
    return [ChatCompletionLogProbsContent.model_validate(item) for item in content]


class TestCompareLogprobs:
    """Port of TestValidation and TestValidationQuant from Go."""

    def test_validation(self):
        """Same as Go TestValidation: standard float precision."""
        original = load_logprobs_content(
            os.path.join(TESTDATA_DIR, "inference_response.json")
        )
        validation = load_logprobs_content(
            os.path.join(TESTDATA_DIR, "validation_response.json")
        )

        result = compare_logprobs(original, validation)
        assert result.reason == "similarity"
        assert result.is_successful
        assert result.similarity > 0.99

    def test_validation_quant(self):
        """Same as Go TestValidationQuant: int4 vs fp8."""
        original = load_logprobs_content(
            os.path.join(TESTDATA_DIR, "inference_response_int4.json")
        )
        validation = load_logprobs_content(
            os.path.join(TESTDATA_DIR, "validation_response_fp8.json")
        )

        result = compare_logprobs(original, validation)
        assert result.reason == "similarity"
        print(f"Quant validation similarity: {result.similarity}")

    def test_different_length(self):
        original = [
            ChatCompletionLogProbsContent(
                token="a",
                logprob=-1.0,
                top_logprobs=[ChatCompletionLogProb(token="a", logprob=-1.0)],
            ),
            ChatCompletionLogProbsContent(
                token="b",
                logprob=-2.0,
                top_logprobs=[ChatCompletionLogProb(token="b", logprob=-2.0)],
            ),
        ]
        validation = [
            ChatCompletionLogProbsContent(
                token="a",
                logprob=-1.0,
                top_logprobs=[ChatCompletionLogProb(token="a", logprob=-1.0)],
            ),
        ]
        result = compare_logprobs(original, validation)
        assert result.reason == "different_length"
        assert not result.is_successful

    def test_different_tokens(self):
        original = [
            ChatCompletionLogProbsContent(
                token="a",
                logprob=-1.0,
                top_logprobs=[ChatCompletionLogProb(token="a", logprob=-1.0)],
            ),
        ]
        validation = [
            ChatCompletionLogProbsContent(
                token="b",
                logprob=-1.0,
                top_logprobs=[ChatCompletionLogProb(token="b", logprob=-1.0)],
            ),
        ]
        result = compare_logprobs(original, validation)
        assert result.reason == "different_tokens"
        assert not result.is_successful

    def test_identical_logprobs(self):
        logprobs = [
            ChatCompletionLogProbsContent(
                token="hello",
                logprob=-0.5,
                top_logprobs=[
                    ChatCompletionLogProb(token="hello", logprob=-0.5),
                    ChatCompletionLogProb(token="world", logprob=-2.0),
                    ChatCompletionLogProb(token="foo", logprob=-5.0),
                ],
            ),
        ]
        result = compare_logprobs(logprobs, logprobs)
        assert result.similarity == 1.0
        assert result.is_successful

    def test_validation_longer_ok(self):
        """Validation logprobs can be longer than original (Go allows this)."""
        original = [
            ChatCompletionLogProbsContent(
                token="a",
                logprob=-1.0,
                top_logprobs=[
                    ChatCompletionLogProb(token="a", logprob=-1.0),
                    ChatCompletionLogProb(token="b", logprob=-3.0),
                ],
            ),
        ]
        validation = [
            ChatCompletionLogProbsContent(
                token="a",
                logprob=-1.0,
                top_logprobs=[
                    ChatCompletionLogProb(token="a", logprob=-1.0),
                    ChatCompletionLogProb(token="b", logprob=-3.0),
                ],
            ),
            ChatCompletionLogProbsContent(
                token="c",
                logprob=-0.1,
                top_logprobs=[
                    ChatCompletionLogProb(token="c", logprob=-0.1),
                    ChatCompletionLogProb(token="d", logprob=-4.0),
                ],
            ),
        ]
        result = compare_logprobs(original, validation)
        assert result.reason == "similarity"
        assert result.similarity == 1.0
