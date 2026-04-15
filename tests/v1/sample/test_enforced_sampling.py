# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.sampling_params import SamplingParams, SamplingType
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch


@pytest.fixture
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def input_batch(device):
    batch = InputBatch(
        max_num_reqs=8,
        max_model_len=128,
        max_num_batched_tokens=1024,
        device=device,
        vocab_size=32000,
        pin_memory=is_pin_memory_available(),
        block_sizes=[1],
        kernel_block_sizes=[1],
    )
    return batch


@pytest.mark.parametrize(
    "i1,i2,enforced_positions,enforced_tokens_data,expected_req_ids,expected_token_ids",
    [
        # Test 1: Only one position has enforced, swap must move it
        (0, 2, [0], {0: [100, 200, 300]}, [2], {2: [100, 200, 300]}),
        # Test 2: Both positions have enforced, both stay in list (values swap)
        (0, 2, [0, 2], {0: [100], 2: [200]}, {0, 2}, {0: [200], 2: [100]}),
        # Test 3: Neither position has enforced, no change
        (0, 2, [1, 3], {1: [100], 3: [200]}, [1, 3], {1: [100], 3: [200]}),
    ],
)
def test_swap_enforced_states(
    input_batch,
    i1,
    i2,
    enforced_positions,
    enforced_tokens_data,
    expected_req_ids,
    expected_token_ids,
):
    max_index = max(i1, i2, max(enforced_positions) if enforced_positions else 0)
    for req_idx in range(max_index + 1):
        req = CachedRequestState(
            req_id=f"req_{req_idx}",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(temperature=0.0),
            pooling_params=None,
            mm_features=[],
            block_ids=([],),
            generator=None,
            num_computed_tokens=0,
            output_token_ids=[],
        )
        input_batch.add_request(req)

    input_batch.enforced_req_ids = enforced_positions.copy()
    input_batch.enforced_token_ids = enforced_tokens_data.copy()
    input_batch.enforced_tokens = {}

    input_batch.swap_states(i1, i2)

    if isinstance(expected_req_ids, set):
        assert set(input_batch.enforced_req_ids) == expected_req_ids
    else:
        assert input_batch.enforced_req_ids == expected_req_ids

    assert input_batch.enforced_token_ids == expected_token_ids


@pytest.mark.parametrize(
    "greedy_reqs,random_reqs,enforced_reqs,expected",
    [
        (set(), set(), set(), False),  # empty batch
        (set(), set(), {"req_1", "req_2"}, True),  # only enforced
        ({"req_1"}, set(), {"req_2"}, False),  # mixed batch (enforced + greedy)
        ({"req_1"}, set(), set(), False),  # only greedy
    ],
)
def test_all_enforced_property(
    input_batch, greedy_reqs, random_reqs, enforced_reqs, expected
):
    """Test all_enforced property with different request combinations."""
    input_batch.greedy_reqs = greedy_reqs
    input_batch.random_reqs = random_reqs
    input_batch.enforced_reqs = enforced_reqs

    assert input_batch.all_enforced == expected


@pytest.mark.parametrize(
    "greedy_reqs,random_reqs,enforced_reqs,expected",
    [
        ({"req_1"}, set(), {"req_2"}, True),  # enforced mixed with greedy
        (set(), set(), {"req_1", "req_2"}, False),  # all enforced (not mixed)
        ({"req_1"}, {"req_2"}, set(), False),  # no enforced (not mixed)
    ],
)
def test_mixed_enforced_property(
    input_batch, greedy_reqs, random_reqs, enforced_reqs, expected
):
    """Test mixed_enforced property with different request combinations."""
    input_batch.greedy_reqs = greedy_reqs
    input_batch.random_reqs = random_reqs
    input_batch.enforced_reqs = enforced_reqs

    assert input_batch.mixed_enforced == expected


@pytest.mark.parametrize(
    "temperature,enforced_token_ids,enforced_tokens,expected_type",
    [
        (0.0, [100, 200, 300], None, SamplingType.ENFORCED),  # enforced_token_ids
        (
            0.0,
            None,
            [{100: [100, 200]}, {200: [200, 300]}],
            SamplingType.ENFORCED,
        ),  # enforced_tokens
        (0.0, None, None, SamplingType.GREEDY),  # no enforced, temp=0
        (1.0, None, None, SamplingType.RANDOM),  # no enforced, temp>0
    ],
)
def test_sampling_type_detection(
    temperature, enforced_token_ids, enforced_tokens, expected_type
):
    params = SamplingParams(
        temperature=temperature,
        enforced_token_ids=enforced_token_ids,
        enforced_tokens=enforced_tokens,
    )
    assert params.sampling_type == expected_type
