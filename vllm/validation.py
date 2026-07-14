from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from vllm.transformers_utils.tokenizer import AnyTokenizer


class EnforcedToken(BaseModel):
    """
    Represents a single token position in the enforced tokens sequence.

    Used by validators to verify that:
    1. The token was sampled correctly from the probability distribution
    2. The probability distribution matches the claimed model

    Attributes:
        token: The token ID (as string) that was sampled at this position
        top_tokens: List of top-k token IDs (as strings) for this position
        logprobs: Raw logprobs for the top tokens (for distance calculation)
        sampling_weights: Integer weights (2^16 scale) used for deterministic
                         sampling verification
        token_id: Parsed integer token ID (excluded from serialization)
        top_token_ids: Parsed integer top token IDs (excluded from serialization)
    """
    token: str
    top_tokens: List[str] = Field(default_factory=list)

    # NEW: Raw logprobs for distance calculation (Stage 2 validation)
    # Maps token ID (as string) to logprob value
    logprobs: Optional[Dict[str, float]] = None

    # NEW: Integer weights for deterministic sampling verification (Stage 1)
    # Maps token ID (as string) to integer weight (2^16 scale)
    # These are the quantized probabilities used for sampling
    sampling_weights: Optional[Dict[str, int]] = None

    # Internal parsed fields (excluded from serialization)
    token_id: Optional[int] = Field(default=None, exclude=True)
    top_token_ids: List[int] = Field(default_factory=list, exclude=True)

    def encode(self, tokenizer: AnyTokenizer) -> None:
        """Parse token strings to integer IDs."""
        try:
            self.token_id = int(self.token)
            self.top_token_ids = [int(t) for t in self.top_tokens]
        except Exception as e:
            raise e


class EnforcedTokens(BaseModel):
    """
    Container for a sequence of enforced tokens with optional validation data.

    This is sent by validators to verify inference results. The tokens are
    "enforced" meaning the model is forced to produce these exact tokens,
    allowing comparison of the logprob distributions.
    """
    tokens: List[EnforcedToken]

    def encode(self, tokenizer: AnyTokenizer) -> None:
        """Parse all token strings to integer IDs."""
        for token in self.tokens:
            token.encode(tokenizer)

    @classmethod
    def from_content(
        cls,
        content: List[Dict[str, Any]],
        include_logprobs: bool = False,
        include_sampling_weights: bool = False,
    ) -> "EnforcedTokens":
        """
        Create EnforcedTokens from OpenAI-format logprobs content.

        Args:
            content: List of logprobs content dicts from ChatCompletionResponse
            include_logprobs: If True, include raw logprobs in the artifact
            include_sampling_weights: If True, include sampling_weights in artifact

        Returns:
            EnforcedTokens instance
        """
        tokens = []
        for position in content:
            token = position["token"]
            top_tokens = [x["token"] for x in position["top_logprobs"]]

            # Optionally include logprobs for distance calculation
            logprobs = None
            if include_logprobs and "top_logprobs" in position:
                logprobs = {
                    x["token"]: x["logprob"]
                    for x in position["top_logprobs"]
                }

            # Optionally include sampling weights for sampling verification
            sampling_weights = None
            if include_sampling_weights and "sampling_weights" in position:
                sampling_weights = position["sampling_weights"]

            tokens.append(
                EnforcedToken(
                    token=token,
                    top_tokens=top_tokens,
                    logprobs=logprobs,
                    sampling_weights=sampling_weights,
                ))
        return cls(tokens=tokens)

    def get_enforced_token_ids(self) -> List[int]:
        """Get the list of enforced token IDs (requires encode() to be called first)."""
        if not self.tokens or not self.tokens[0].token_id:
            raise ValueError("Enforced tokens are not encoded")
        return [token.token_id for token in self.tokens]

    def get_top_tokens(self) -> List[Dict[int, List[int]]]:
        return [
            {
                token.token_id: token.top_token_ids
            }
            for token in self.tokens
        ]

    def has_validation_data(self) -> bool:
        """Check if this artifact has data for validation (logprobs or weights)."""
        if not self.tokens:
            return False
        first = self.tokens[0]
        return first.logprobs is not None or first.sampling_weights is not None
