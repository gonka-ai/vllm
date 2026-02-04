# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.validation import EnforcedTokens

if TYPE_CHECKING:
    from vllm.v1.sample.deterministic_utils import Sha256CounterRNG

@dataclass
class SamplingMetadata:
    temperature: torch.Tensor | None
    all_greedy: bool
    all_random: bool
    all_enforced: bool
    mixed_enforced: bool

    enforced_token_ids: dict[list[int]]
    enforced_tokens: dict[EnforcedTokens]
    enforced_req_ids: list[int]

    top_p: torch.Tensor | None
    top_k: torch.Tensor | None

    generators: dict[int, torch.Generator]

    # Deterministic RNGs for cross-platform reproducible sampling (validation)
    # When VLLM_DETERMINISTIC_SAMPLING=1, this contains Sha256CounterRNG
    # instances keyed by request index
    deterministic_rngs: "dict[int, Sha256CounterRNG]" = field(
        default_factory=dict)

    # None means no logprobs, 0 means sampled token logprobs only
    max_num_logprobs: int | None

    no_penalties: bool
    prompt_token_ids: torch.Tensor | None
    frequency_penalties: torch.Tensor
    presence_penalties: torch.Tensor
    repetition_penalties: torch.Tensor

    output_token_ids: list[list[int]]

    # `allowed_token_ids_mask` is a 2D bool tensor of shape (max batch size,
    # vocab size).
    allowed_token_ids_mask: torch.Tensor | None

    # req_index -> bad_words_token_ids
    bad_words_token_ids: dict[int, list[list[int]]]

    # Loaded logits processors
    logitsprocs: LogitsProcessors

    # Speculative token ids
    spec_token_ids: list[list[int]] | None = None
