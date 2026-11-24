# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Requests with enforced_token_ids (validation replay) get no draft tokens."""

import copy

from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput

from .utils import create_requests, create_scheduler

NUM_SPEC = 3
DRAFTS = [11, 12, 13]


def _runner_output(req_ids):
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        sampled_token_ids=[[0] for _ in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _add_requests(scheduler):
    normal, enforced = create_requests(num_requests=2, num_tokens=1)
    # create_requests shares one SamplingParams object between requests.
    enforced.sampling_params = copy.copy(enforced.sampling_params)
    enforced.sampling_params.enforced_token_ids = [5, 6, 7]
    scheduler.add_request(normal)
    scheduler.add_request(enforced)
    return normal.request_id, enforced.request_id


def test_sync_draft_update_skips_enforced_request():
    scheduler = create_scheduler(num_speculative_tokens=NUM_SPEC)
    normal_id, enforced_id = _add_requests(scheduler)
    ids = [normal_id, enforced_id]

    output = scheduler.schedule()
    scheduler.update_from_output(output, _runner_output(ids))
    scheduler.update_draft_token_ids(DraftTokenIds(ids, [list(DRAFTS), list(DRAFTS)]))

    output = scheduler.schedule()
    assert output.scheduled_spec_decode_tokens[normal_id] == DRAFTS
    assert enforced_id not in output.scheduled_spec_decode_tokens
    assert output.num_scheduled_tokens[enforced_id] == 1
    assert output.num_scheduled_tokens[normal_id] == 1 + len(DRAFTS)


def test_async_placeholder_drafts_dropped_for_enforced_request():
    scheduler = create_scheduler(
        num_speculative_tokens=NUM_SPEC,
        async_scheduling=True,
        speculative_method="ngram_gpu",
    )
    normal_id, enforced_id = _add_requests(scheduler)

    first = scheduler.schedule()
    assert not first.scheduled_spec_decode_tokens
    # The async scheduler now holds placeholder drafts on both requests.
    second = scheduler.schedule()
    assert len(second.scheduled_spec_decode_tokens[normal_id]) == NUM_SPEC
    assert enforced_id not in second.scheduled_spec_decode_tokens
    assert second.num_scheduled_tokens[enforced_id] == 1

    scheduler.update_draft_token_ids_in_output(
        DraftTokenIds([normal_id, enforced_id], [list(DRAFTS), list(DRAFTS)]), second
    )
    assert second.scheduled_spec_decode_tokens[normal_id] == DRAFTS
    assert enforced_id not in second.scheduled_spec_decode_tokens
