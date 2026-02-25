"""
Logprobs validation for decentralized inference.

Compares logprob distributions between original inference and validation
runs to verify that the same model was used.

Port of the validation logic from
decentralized-api/internal/validation/inference_validation.go
"""

from dataclasses import dataclass
from typing import List

from vllm.entrypoints.openai.protocol import (
    ChatCompletionLogProb,
    ChatCompletionLogProbsContent,
)

SIMILARITY_THRESHOLD = 0.99


@dataclass
class ValidationResult:
    def __init__(self, threshold: float = SIMILARITY_THRESHOLD):
        self.threshold = threshold
    similarity: float
    reason: str  # "similarity" | "different_length" | "different_tokens"

    @property
    def is_successful(self) -> bool:
        return (self.reason == "similarity"
                and self.similarity > self.threshold)


def compare_logprobs(
    original_logprobs: List[ChatCompletionLogProbsContent],
    validation_logprobs: List[ChatCompletionLogProbsContent],
    threshold: float = SIMILARITY_THRESHOLD,
) -> ValidationResult:
    """Compare logprobs from original and validation inference runs.

    Args:
        original_logprobs: Per-token logprobs from the original inference.
        validation_logprobs: Per-token logprobs from the validation run
            (with enforced tokens).

    Returns:
        ValidationResult with similarity score and reason.
    """
    if len(validation_logprobs) < len(original_logprobs):
        return ValidationResult(threshold=threshold, similarity=0.0, reason="different_length",)

    for i in range(len(original_logprobs)):
        if original_logprobs[i].token != validation_logprobs[i].token:
            return ValidationResult(threshold=threshold, similarity=0.0, reason="different_tokens")

    similarity = _custom_similarity(original_logprobs, validation_logprobs)
    return ValidationResult(threshold=threshold, similarity=similarity, reason="similarity")


def _custom_similarity(
    original: List[ChatCompletionLogProbsContent],
    validation: List[ChatCompletionLogProbsContent],
) -> float:
    distance = _custom_distance(original, validation)
    similarity = 1.0 - distance
    return max(0.0, similarity)


def _custom_distance(
    original: List[ChatCompletionLogProbsContent],
    validation: List[ChatCompletionLogProbsContent],
) -> float:
    distance = 0.0
    for i in range(len(original)):
        distance += _position_distance(original[i].top_logprobs,
                                       validation[i].top_logprobs)
    total = max(100, len(original)) * len(original[0].top_logprobs)
    return distance / total


def _position_distance(
    original_top: List[ChatCompletionLogProb],
    validation_top: List[ChatCompletionLogProb],
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