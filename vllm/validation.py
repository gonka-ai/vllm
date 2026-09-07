# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Enforced-token support for Gonka inference validation."""

from typing import Any

from pydantic import BaseModel, Field

# Bound replay work; request-body limits belong to the proxy.
MAX_ENFORCED_TOKENS = 32768
MAX_TOP_TOKENS_PER_POSITION = 64


class EnforcedToken(BaseModel):
    token: str
    top_tokens: list[str] = Field(
        default_factory=list, max_length=MAX_TOP_TOKENS_PER_POSITION
    )
    token_id: int | None = Field(default=None, exclude=True)
    top_token_ids: list[int] = Field(default_factory=list, exclude=True)

    def encode(self, tokenizer) -> None:
        """Convert token strings to token IDs.
        Tokens from gonka API are already numeric strings (token IDs)."""
        try:
            self.token_id = int(self.token)
            self.top_token_ids = [int(t) for t in self.top_tokens]
        except ValueError:
            # Fallback: tokenize the string
            ids = tokenizer.encode(self.token, add_special_tokens=False)
            self.token_id = ids[0] if ids else None
            self.top_token_ids = []
            for t in self.top_tokens:
                t_ids = tokenizer.encode(t, add_special_tokens=False)
                if t_ids:
                    self.top_token_ids.append(t_ids[0])


class EnforcedTokens(BaseModel):
    tokens: list[EnforcedToken] = Field(max_length=MAX_ENFORCED_TOKENS)

    def encode(self, tokenizer) -> None:
        for token in self.tokens:
            token.encode(tokenizer)

    @classmethod
    def from_content(cls, content: list[dict[str, Any]]) -> "EnforcedTokens":
        tokens = []
        for position in content:
            token = position["token"]
            top_tokens = [x["token"] for x in position["top_logprobs"]]
            tokens.append(EnforcedToken(token=token, top_tokens=top_tokens))
        return cls(tokens=tokens)

    def get_enforced_token_ids(self) -> list[int]:
        token_ids = [token.token_id for token in self.tokens]
        if not token_ids or any(token_id is None for token_id in token_ids):
            raise ValueError("Enforced tokens are not encoded")
        return [token_id for token_id in token_ids if token_id is not None]

    def detect_logprobs_mode(self, threshold: float = 0.10) -> str | None:
        """Classify original inference logprobs mode from top_token_ids.

        In processed-logprobs results, empty top-k slots are padded with the
        lowest vocab IDs (0-3 for Qwen: ``!``, ``"``, ``#``, ``$``) at
        logprob -9999. This makes ~75% of top_token_ids entries < 4.
        In raw-logprobs results, only ~0.2% of entries are < 4.

        Must be called after encode().

        Returns ``'raw_logprobs'``, ``'processed_logprobs'``, or ``None``
        if there is insufficient data (fewer than 10 top-token entries).
        """
        total = 0
        low_id_count = 0
        for t in self.tokens:
            for tid in t.top_token_ids:
                total += 1
                if tid < 4:
                    low_id_count += 1
        if total < 10:
            return None
        ratio = low_id_count / total
        return "processed_logprobs" if ratio > threshold else "raw_logprobs"


def validate_enforced_token_ids(token_ids: list[int], vocab_size: int) -> None:
    """Reject replay ids that cannot be embedded."""
    for position, token_id in enumerate(token_ids):
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id < vocab_size
        ):
            raise ValueError(
                f"invalid enforced token at position {position}: "
                f"{token_id!r} not in [0, {vocab_size})"
            )
