# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field

import torch

from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.validation import EnforcedTokens


@dataclass
class SamplingMetadata:
    temperature: torch.Tensor | None
    all_greedy: bool
    all_random: bool
    all_enforced: bool
    mixed_enforced: bool

    top_p: torch.Tensor | None
    top_k: torch.Tensor | None

    generators: dict[int, torch.Generator]

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

    # Per-request logprobs mode: "raw_logprobs", "processed_logprobs",
    # "mixed", or None (no sampled-token logprobs requested).
    batch_logprobs_mode: str | None = None

    # Per-row bool mask: True = processed, False = raw.
    # Only materialized when batch_logprobs_mode == "mixed".
    logprobs_is_processed: torch.Tensor | None = None

    # Speculative token ids
    spec_token_ids: list[list[int]] | None = None

    # Enforced next token ids for validation replay (gonka PoC).
    # Shape [num_reqs], -1 means no enforcement for that request.
    enforced_next_token_ids: torch.Tensor | None = None

    enforced_token_ids: dict[int, list[int]] = field(default_factory=dict)
    enforced_tokens: dict[int, EnforcedTokens] = field(default_factory=dict)
    enforced_req_ids: list[int] = field(default_factory=list)
