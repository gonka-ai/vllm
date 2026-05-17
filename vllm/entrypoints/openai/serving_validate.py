# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.protocol import (
    ChatCompletionLogProbsContent,
    ErrorInfo,
    ErrorResponse,
)
from vllm.logger import init_logger
from vllm.logprobs_validation import ValidationResult, compare_logprobs

logger = init_logger(__name__)


class OpenAIServingValidate:
    """Validation service.

    Handles the routes:
    - /v1/validate
    """

    def __init__(
        self,
        engine_client: EngineClient,
        threshold: float,
        toploc_validation_usage: bool,
    ):
        super().__init__()

        self.engine_client = engine_client
        self.threshold = threshold
        self.toploc_validation_usage = toploc_validation_usage

    async def validate(
        self,
        original_logprobs: list[ChatCompletionLogProbsContent],
        validation_logprobs: list[ChatCompletionLogProbsContent],
        original_artifacts: list[dict[str, str]] | None = None,
        validation_artifacts: list[dict[str, str]] | None = None,
    ) -> ValidationResult | ErrorResponse:
        """Validate logprobs and optionally input artifacts."""
        if self.toploc_validation_usage:
            raise NotImplementedError("Toploc validation usage is not implemented")
        # TODO: implement toploc validation usage
        try:
            return compare_logprobs(
                original_logprobs,
                validation_logprobs,
                self.threshold,
                original_artifacts=original_artifacts,
                validation_artifacts=validation_artifacts,
            )
        except Exception as e:
            return ErrorResponse(
                error=ErrorInfo(message=str(e), type="ValidationError", code=500)
            )
