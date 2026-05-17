# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random

import pytest
import torch

from tests.utils import create_new_process_for_each_test
from tests.v1.sample.utils import (
    create_fake_logits,
    create_penalty_tensor,
    create_prompt_tokens_tensor,
)
from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.sample.logits_processor import build_logitsprocs
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

PIN_MEMORY_AVAILABLE = is_pin_memory_available()
VOCAB_SIZE = 1024
NUM_OUTPUT_TOKENS = 10
CUDA_DEVICES = [
    f"{current_platform.device_type}:{i}"
    for i in range(1 if current_platform.device_count() == 1 else 2)
]


def _create_enforced_sampling_metadata(
    batch_size: int,
    vocab_size: int,
    device: torch.device,
    enforced_req_ids: list[int],
    enforced_token_ids: dict[int, list[int]],
    enforced_tokens: dict[int, list[dict[int, list[int]]]] | None = None,
) -> SamplingMetadata:
    """Create sampling metadata with enforced tokens."""
    output_token_ids: list[list[int]] = []
    prompt_token_ids: list[list[int]] = []

    for i in range(batch_size):
        # For enforced requests, use shorter output to test step-based enforcement
        if i in enforced_req_ids:
            output_token_ids.append(
                [random.randint(0, vocab_size - 1) for _ in range(random.randint(0, 5))]
            )
        else:
            output_token_ids.append(
                [random.randint(0, vocab_size - 1) for _ in range(NUM_OUTPUT_TOKENS)]
            )
        prompt_token_ids.append([random.randint(0, vocab_size - 1) for _ in range(10)])

    logitsprocs = build_logitsprocs(
        vllm_config=VllmConfig(),
        device=device,
        is_pin_memory=PIN_MEMORY_AVAILABLE,
        is_pooling_model=False,
    )

    all_enforced = len(enforced_req_ids) == batch_size
    mixed_enforced = len(enforced_req_ids) > 0 and len(enforced_req_ids) < batch_size
    all_greedy = not all_enforced and not mixed_enforced
    all_random = False

    sampling_metadata = SamplingMetadata(
        temperature=torch.full(
            (batch_size,), 0.0 if all_greedy else 1.0, device=device
        ),
        all_greedy=all_greedy,
        all_random=all_random,
        all_enforced=all_enforced,
        mixed_enforced=mixed_enforced,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=3 if enforced_tokens else 0,
        prompt_token_ids=create_prompt_tokens_tensor(
            prompt_token_ids, vocab_size, device
        ),
        output_token_ids=output_token_ids,
        frequency_penalties=create_penalty_tensor(batch_size, 0.0, device),
        presence_penalties=create_penalty_tensor(batch_size, 0.0, device),
        repetition_penalties=create_penalty_tensor(batch_size, 1.0, device),
        no_penalties=True,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=logitsprocs,
        enforced_req_ids=enforced_req_ids,
        enforced_token_ids=enforced_token_ids,
        enforced_tokens=enforced_tokens or {},
    )
    return sampling_metadata


@create_new_process_for_each_test()
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_all_enforced_sampling(device: str, batch_size: int):
    """Test sampling when all requests have enforced tokens."""
    torch.set_default_device(device)
    device_obj = torch.device(device)

    enforced_req_ids = list(range(batch_size))
    enforced_token_ids = {
        i: [random.randint(0, VOCAB_SIZE - 1) for _ in range(5)]
        for i in range(batch_size)
    }

    logits = create_fake_logits(batch_size, VOCAB_SIZE).to(device_obj)
    sampling_metadata = _create_enforced_sampling_metadata(
        batch_size, VOCAB_SIZE, device_obj, enforced_req_ids, enforced_token_ids
    )

    sampler = Sampler()
    sampled, _ = sampler.sample(logits, sampling_metadata)

    assert sampled.shape == (batch_size,)
    for i in range(batch_size):
        step = len(sampling_metadata.output_token_ids[i])
        expected_token = enforced_token_ids[i][
            min(step, len(enforced_token_ids[i]) - 1)
        ]
        assert sampled[i].item() == expected_token, (
            f"Request {i}: expected token {expected_token} at step {step}, "
            f"got {sampled[i].item()}"
        )


@create_new_process_for_each_test()
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("batch_size,num_enforced", [(8, 2), (16, 5), (32, 10)])
def test_mixed_enforced_sampling(device: str, batch_size: int, num_enforced: int):
    """Test sampling when only some requests have enforced tokens."""
    torch.set_default_device(device)
    device_obj = torch.device(device)

    enforced_req_ids = random.sample(range(batch_size), num_enforced)
    enforced_token_ids = {
        i: [random.randint(0, VOCAB_SIZE - 1) for _ in range(5)]
        for i in enforced_req_ids
    }

    logits = create_fake_logits(batch_size, VOCAB_SIZE).to(device_obj)
    for i in range(batch_size):
        if i not in enforced_req_ids:
            logits[i, i % VOCAB_SIZE] = 100.0  # Clear winner

    sampling_metadata = _create_enforced_sampling_metadata(
        batch_size, VOCAB_SIZE, device_obj, enforced_req_ids, enforced_token_ids
    )

    sampler = Sampler()
    sampled, _ = sampler.sample(logits, sampling_metadata)

    assert sampled.shape == (batch_size,)
    for i in enforced_req_ids:
        step = len(sampling_metadata.output_token_ids[i])
        expected_token = enforced_token_ids[i][
            min(step, len(enforced_token_ids[i]) - 1)
        ]
        assert sampled[i].item() == expected_token, (
            f"Enforced request {i}: expected token {expected_token} at step {step}, "
            f"got {sampled[i].item()}"
        )

    for i in range(batch_size):
        if i not in enforced_req_ids:
            # Should pick the dominant token we set
            expected_token = i % VOCAB_SIZE
            assert sampled[i].item() == expected_token, (
                f"Non-enforced request {i}: expected greedy token {expected_token}, "
                f"got {sampled[i].item()}"
            )


@create_new_process_for_each_test()
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_enforced_tokens_with_logprobs(device: str):
    """Test that enforced tokens work correctly with logprobs gathering."""
    torch.set_default_device(device)
    device_obj = torch.device(device)
    batch_size = 4

    enforced_req_ids = [0, 2]
    enforced_token_ids = {
        0: [10, 20, 30],
        2: [40, 50, 60],
    }
    enforced_tokens = {
        0: [
            {10: [10, 11, 12]},  # step 0: token 10, top tokens [10, 11, 12]
            {20: [20, 21, 22]},  # step 1: token 20, top tokens [20, 21, 22]
            {30: [30, 31, 32]},  # step 2: token 30, top tokens [30, 31, 32]
        ],
        2: [
            {40: [40, 41, 42]},
            {50: [50, 51, 52]},
            {60: [60, 61, 62]},
        ],
    }

    logits = create_fake_logits(batch_size, VOCAB_SIZE).to(device_obj)
    sampling_metadata = _create_enforced_sampling_metadata(
        batch_size,
        VOCAB_SIZE,
        device_obj,
        enforced_req_ids,
        enforced_token_ids,
        enforced_tokens,
    )

    sampler = Sampler()
    sampled, _ = sampler.sample(logits, sampling_metadata)

    for i in enforced_req_ids:
        step = len(sampling_metadata.output_token_ids[i])
        expected_token = enforced_token_ids[i][
            min(step, len(enforced_token_ids[i]) - 1)
        ]
        assert sampled[i].item() == expected_token

    logprobs = sampler.compute_logprobs(logits)
    logprobs_tensors = sampler.gather_logprobs(
        logprobs, 3, sampled.long(), sampling_metadata
    )

    assert logprobs_tensors.logprob_token_ids.shape == (
        batch_size,
        4,
    )  # sampled + top 3
    assert logprobs_tensors.logprobs.shape == (batch_size, 4)
    assert logprobs_tensors.selected_token_ranks.shape == (batch_size,)


@create_new_process_for_each_test()
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("step", [0, 2, 5, 10])
def test_enforced_tokens_at_different_steps(device: str, step: int):
    """Test that enforced tokens work correctly at different generation steps."""
    torch.set_default_device(device)
    device_obj = torch.device(device)
    batch_size = 4

    enforced_req_ids = [1]
    enforced_sequence = [100, 200, 300, 400, 500]
    enforced_token_ids = {1: enforced_sequence}

    logits = create_fake_logits(batch_size, VOCAB_SIZE).to(device_obj)
    output_token_ids = [[] for _ in range(batch_size)]
    output_token_ids[1] = [0] * step  # Simulate 'step' tokens already generated

    prompt_token_ids = [
        [random.randint(0, VOCAB_SIZE - 1) for _ in range(10)]
        for _ in range(batch_size)
    ]

    logitsprocs = build_logitsprocs(
        vllm_config=VllmConfig(),
        device=device_obj,
        is_pin_memory=PIN_MEMORY_AVAILABLE,
        is_pooling_model=False,
    )

    sampling_metadata = SamplingMetadata(
        temperature=torch.full((batch_size,), 1.0, device=device_obj),
        all_greedy=False,
        all_random=False,
        all_enforced=False,
        mixed_enforced=True,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=0,
        prompt_token_ids=create_prompt_tokens_tensor(
            prompt_token_ids, VOCAB_SIZE, device_obj
        ),
        output_token_ids=output_token_ids,
        frequency_penalties=create_penalty_tensor(batch_size, 0.0, device_obj),
        presence_penalties=create_penalty_tensor(batch_size, 0.0, device_obj),
        repetition_penalties=create_penalty_tensor(batch_size, 1.0, device_obj),
        no_penalties=True,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=logitsprocs,
        enforced_req_ids=enforced_req_ids,
        enforced_token_ids=enforced_token_ids,
        enforced_tokens={},
    )

    sampler = Sampler()
    sampled, _ = sampler.sample(logits, sampling_metadata)
    expected_token = enforced_sequence[min(step, len(enforced_sequence) - 1)]
    assert sampled[1].item() == expected_token, (
        f"At step {step}: expected token {expected_token}, got {sampled[1].item()}"
    )


@create_new_process_for_each_test()
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_enforced_tokens_last_token_repeats(device: str):
    """Test that when step exceeds sequence length, last token is repeated."""
    torch.set_default_device(device)
    device_obj = torch.device(device)
    batch_size = 2

    enforced_req_ids = [0]
    enforced_sequence = [100, 200, 300]
    enforced_token_ids = {0: enforced_sequence}

    logits = create_fake_logits(batch_size, VOCAB_SIZE).to(device_obj)
    output_token_ids = [[0] * 5, []]
    prompt_token_ids = [
        [random.randint(0, VOCAB_SIZE - 1) for _ in range(10)]
        for _ in range(batch_size)
    ]

    logitsprocs = build_logitsprocs(
        vllm_config=VllmConfig(),
        device=device_obj,
        is_pin_memory=PIN_MEMORY_AVAILABLE,
        is_pooling_model=False,
    )

    sampling_metadata = SamplingMetadata(
        temperature=torch.full((batch_size,), 1.0, device=device_obj),
        all_greedy=False,
        all_random=False,
        all_enforced=False,
        mixed_enforced=True,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=0,
        prompt_token_ids=create_prompt_tokens_tensor(
            prompt_token_ids, VOCAB_SIZE, device_obj
        ),
        output_token_ids=output_token_ids,
        frequency_penalties=create_penalty_tensor(batch_size, 0.0, device_obj),
        presence_penalties=create_penalty_tensor(batch_size, 0.0, device_obj),
        repetition_penalties=create_penalty_tensor(batch_size, 1.0, device_obj),
        no_penalties=True,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=logitsprocs,
        enforced_req_ids=enforced_req_ids,
        enforced_token_ids=enforced_token_ids,
        enforced_tokens={},
    )

    sampler = Sampler()
    sampled, _ = sampler.sample(logits, sampling_metadata)

    assert sampled[0].item() == 300, (
        f"Expected last token (300) to repeat, got {sampled[0].item()}"
    )


@create_new_process_for_each_test()
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_enforced_tokens_empty_batch(device: str):
    """Test that empty enforced_req_ids doesn't break sampling."""
    torch.set_default_device(device)
    device_obj = torch.device(device)
    batch_size = 4

    enforced_req_ids = []
    enforced_token_ids = {}

    logits = create_fake_logits(batch_size, VOCAB_SIZE).to(device_obj)

    sampling_metadata = _create_enforced_sampling_metadata(
        batch_size, VOCAB_SIZE, device_obj, enforced_req_ids, enforced_token_ids
    )
    sampler = Sampler()
    sampled, _ = sampler.sample(logits, sampling_metadata)

    assert sampled.shape == (batch_size,)


@create_new_process_for_each_test()
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_enforced_tokens_ordering_after_condense(device: str):
    """Test that enforced tokens work correctly even when req_ids are out of order."""
    torch.set_default_device(device)
    device_obj = torch.device(device)
    batch_size = 4

    # Enforced req_ids out of order (simulating post-condense state)
    enforced_req_ids = [2, 0, 3]
    enforced_token_ids = {
        0: [100, 101, 102],
        2: [200, 201, 202],
        3: [300, 301, 302],
    }

    logits = create_fake_logits(batch_size, VOCAB_SIZE).to(device_obj)
    sampling_metadata = _create_enforced_sampling_metadata(
        batch_size, VOCAB_SIZE, device_obj, enforced_req_ids, enforced_token_ids
    )

    sampler = Sampler()
    sampled, _ = sampler.sample(logits, sampling_metadata)

    for req_id in enforced_req_ids:
        step = len(sampling_metadata.output_token_ids[req_id])
        expected_token = enforced_token_ids[req_id][
            min(step, len(enforced_token_ids[req_id]) - 1)
        ]
        assert sampled[req_id].item() == expected_token, (
            f"Request {req_id}: expected token {expected_token}, got {sampled[req_id].item()}"
        )
