# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Pytest configuration for validation tests.
"""

import os
import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "e2e: mark test as end-to-end (requires running vLLM server)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip E2E tests unless explicitly enabled."""
    if not os.environ.get("RUN_E2E_TESTS") and not os.environ.get("VLLM_TEST_SERVER_URL"):
        skip_e2e = pytest.mark.skip(
            reason="E2E tests disabled. Set RUN_E2E_TESTS=1 or VLLM_TEST_SERVER_URL"
        )
        for item in items:
            if "e2e" in item.keywords:
                item.add_marker(skip_e2e)


@pytest.fixture(scope="session")
def validation_test_seed():
    """Common seed for validation tests."""
    return 42


@pytest.fixture(scope="session")
def validation_test_prompt():
    """Common prompt for validation tests."""
    return "What is 2+2?"


@pytest.fixture(scope="session")
def validation_test_temperature():
    """Common temperature for validation tests."""
    return 0.99
