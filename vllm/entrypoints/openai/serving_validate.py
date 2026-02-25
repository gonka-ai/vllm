# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import List

from vllm.engine.protocol import EngineClient

from vllm.entrypoints.openai.protocol import (
    ChatCompletionLogProbsContent,
    ErrorResponse,
    ErrorInfo,
)

from vllm.logprobs_validation import compare_logprobs, ValidationResult


from vllm.logger import init_logger

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
    ):
        super().__init__()

        self.engine_client = engine_client
        self.threshold = threshold

    async def validate(
        self,
        original_logprobs: List[ChatCompletionLogProbsContent],
        validation_logprobs: List[ChatCompletionLogProbsContent],
        ) -> ValidationResult | ErrorResponse:
        """Validate logprobs"""
        try:
            return compare_logprobs(original_logprobs, validation_logprobs, self.threshold)
        except Exception as e:
            return ErrorResponse(
                error=ErrorInfo(message=str(e), type="ValidationError", code=500))
        

