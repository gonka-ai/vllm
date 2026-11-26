# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 speculative decoding is not implemented on HPU. The registry
resolves these classes by name for a speculative config only; they exist so
that ``vllm.models.deepseek_v4`` imports on the platform."""

from vllm.config import VllmConfig


class DSparkDeepseekV4ForCausalLM:
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        raise NotImplementedError(
            "DeepSeek-V4 speculative decoding (DSpark) is not implemented on HPU."
        )


class DeepSeekV4MTP:
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        raise NotImplementedError(
            "DeepSeek-V4 speculative decoding (MTP) is not implemented on HPU."
        )
