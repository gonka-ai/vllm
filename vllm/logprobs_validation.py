# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Logprobs validation for decentralized inference.

Compares logprob distributions between original inference and validation
runs to verify that the same model was used.

Port of the validation logic from
decentralized-api/internal/validation/inference_validation.go
"""

from dataclasses import dataclass
from typing import Literal

from vllm.entrypoints.openai.protocol import (
    ChatCompletionLogProb,
    ChatCompletionLogProbsContent,
)

SIMILARITY_THRESHOLD = 0.99
TOPLOC_VALIDATION_USAGE = False


def set_validation_runtime_config(
    similarity_threshold: float | None = None,
    toploc_validation_usage: bool | None = None,
) -> None:
    """Set runtime validation config from server CLI args."""
    global SIMILARITY_THRESHOLD, TOPLOC_VALIDATION_USAGE

    if similarity_threshold is not None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                "similarity_threshold must be in [0.0, 1.0], "
                f"got {similarity_threshold!r}"
            )
        SIMILARITY_THRESHOLD = similarity_threshold

    if toploc_validation_usage is not None:
        TOPLOC_VALIDATION_USAGE = toploc_validation_usage


@dataclass
class ValidationResult:
    similarity: float
    reason: Literal[
        "similarity",
        "different_length",
        "different_tokens",
        "artifacts_mismatch",
    ]
    threshold: float = SIMILARITY_THRESHOLD
    artifacts_match: bool | None = None

    @property
    def is_successful(self) -> bool:
        if self.reason == "artifacts_mismatch":
            return False
        if self.artifacts_match is False:
            return False
        return self.reason == "similarity" and self.similarity > self.threshold


def compare_artifacts(
    original_artifacts: list[dict[str, str]] | None,
    validation_artifacts: list[dict[str, str]] | None,
) -> bool:
    """Compare input_artifacts from original and validation responses.

    Returns True if both are None or both are identical.
    """
    if original_artifacts is None and validation_artifacts is None:
        return True
    if original_artifacts is None or validation_artifacts is None:
        return False
    if len(original_artifacts) != len(validation_artifacts):
        return False
    for orig, val in zip(original_artifacts, validation_artifacts):
        if orig.get("modality") != val.get("modality"):
            return False
        if orig.get("artifact") != val.get("artifact"):
            return False
    return True


def compare_logprobs(
    original_logprobs: list[ChatCompletionLogProbsContent],
    validation_logprobs: list[ChatCompletionLogProbsContent],
    threshold: float = SIMILARITY_THRESHOLD,
    original_artifacts: list[dict[str, str]] | None = None,
    validation_artifacts: list[dict[str, str]] | None = None,
) -> ValidationResult:
    """Compare logprobs from original and validation inference runs.

    Args:
        original_logprobs: Per-token logprobs from the original inference.
        validation_logprobs: Per-token logprobs from the validation run
            (with enforced tokens).
        threshold: Minimum similarity for successful validation.
        original_artifacts: input_artifacts from original response (optional).
        validation_artifacts: input_artifacts from validation response (optional).

    Returns:
        ValidationResult with similarity score, reason, and artifacts_match.
    """
    artifacts_match = compare_artifacts(original_artifacts, validation_artifacts)
    if not artifacts_match:
        return ValidationResult(
            threshold=threshold,
            similarity=0.0,
            reason="artifacts_mismatch",
            artifacts_match=False,
        )

    # Important for quantization detection (int8 vs int4):
    # even if sequence lengths / token ids differ, we still compute similarity
    # on the common prefix to keep a usable numeric signal.
    prefix_len = min(len(original_logprobs), len(validation_logprobs))
    if prefix_len <= 0:
        similarity = 0.0
        tokens_match = False
    else:
        original_prefix = original_logprobs[:prefix_len]
        validation_prefix = validation_logprobs[:prefix_len]
        similarity_prefix = _custom_similarity(original_prefix, validation_prefix)
        # Down-weight similarity if one side is shorter.
        max_len = max(len(original_logprobs), len(validation_logprobs))
        len_factor = prefix_len / max_len if max_len > 0 else 0.0
        similarity = similarity_prefix * len_factor
        tokens_match = all(
            original_prefix[i].token == validation_prefix[i].token
            for i in range(prefix_len)
        )

    if len(validation_logprobs) != len(original_logprobs):
        reason: ValidationResult.reason = "different_length"
    elif not tokens_match:
        reason = "different_tokens"
    else:
        reason = "similarity"
    return ValidationResult(
        threshold=threshold,
        similarity=similarity,
        reason=reason,
        artifacts_match=artifacts_match,
    )


def _custom_similarity(
    original: list[ChatCompletionLogProbsContent],
    validation: list[ChatCompletionLogProbsContent],
) -> float:
    distance = _custom_distance(original, validation)
    similarity = 1.0 - distance
    return max(0.0, similarity)


def _custom_distance(
    original: list[ChatCompletionLogProbsContent],
    validation: list[ChatCompletionLogProbsContent],
) -> float:
    distance = 0.0
    for i in range(len(original)):
        distance += _position_distance(
            original[i].top_logprobs, validation[i].top_logprobs
        )
    total = max(100, len(original)) * len(original[0].top_logprobs)
    return distance / total


def _position_distance(
    original_top: list[ChatCompletionLogProb],
    validation_top: list[ChatCompletionLogProb],
) -> float:
    if not original_top or not validation_top:
        raise ValueError("Empty logprobs provided")

    original_map = {lp.token: lp.logprob for lp in original_top}
    sorted_values = sorted(original_map.values())

    if len(sorted_values) >= 2:
        min1, min2 = sorted_values[0], sorted_values[1]
    elif len(sorted_values) == 1:
        min1 = sorted_values[0]
        min2 = min1 - 100.0
    else:
        return 0.0

    # Estimate logprob for tokens not in the original top-k
    next_logprob = min1 - (min2 - min1)

    distance = 0.0
    for v in validation_top:
        orig_lp = original_map.get(v.token, next_logprob)
        denom = 1e-6 + abs(v.logprob) + abs(orig_lp)
        distance += abs(v.logprob - orig_lp) / denom / 2.0

    return distance
