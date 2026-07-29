# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract tests for the V2-model-runner containment the residual relies on.

vLLM ships a second GPU model runner (``vllm/v1/worker/gpu/``) with its own
InputBatch and sampler stack, which does not carry the residual sampler hooks
(per-request ``logprobs_mode``, ``enforced_token_ids``,
``need_processed_logprobs``). A request routed through it samples without any
of that, and without an error. Containment is therefore an env pin,
``VLLM_USE_V2_MODEL_RUNNER=0``, baked into docker/Dockerfile.gonka-poc.

The pin only works while two assumptions hold:

1. the env knob still exists — once renamed or dropped, the pin is a silent
   no-op and a future flip of the default re-exposes the V2 runner;
2. the V2 runner still has its own sampler stack — once the stacks are
   unified, either the pin is unnecessary or the hooks need a V2 port.

Scope: read-only inspection. No GPU, no engine startup.
"""

from __future__ import annotations

import importlib

import pytest


def test_v2_runner_env_knob_exists() -> None:
    """Require VLLM_USE_V2_MODEL_RUNNER to still be a recognized env knob."""
    pytest.importorskip("vllm")
    envs = importlib.import_module("vllm.envs")
    assert "VLLM_USE_V2_MODEL_RUNNER" in getattr(envs, "environment_variables", {}), (
        "VLLM_USE_V2_MODEL_RUNNER is gone from vllm.envs — the baked "
        "container pin no longer selects anything. Find the current "
        "V2-runner switch and re-pin, or confirm the V2 runner was removed."
    )


def test_v2_runner_still_separate_sampler() -> None:
    """Require the V2 runner to keep the separate sampler stack it is pinned for."""
    pytest.importorskip("vllm")
    try:
        v2_sampler = importlib.import_module("vllm.v1.worker.gpu.sample.sampler")
    except ModuleNotFoundError:
        pytest.skip(
            "V2 runner sampler module is gone — the stacks may have been "
            "unified; re-evaluate whether the env pin is still needed."
        )
    v1_sampler = importlib.import_module("vllm.v1.sample.sampler")
    assert v2_sampler.__name__ != v1_sampler.__name__, (
        "The V2 sampler resolves to the V1 module — unification happened, so "
        "re-evaluate the containment pin and whether the residual hooks now "
        "cover both runners."
    )
