# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import random

import aiohttp
import pytest
import requests

from ...utils import RemoteOpenAIServer

MODEL = "hmellor/tiny-random-LlamaForCausalLM"
SERVER_ARGS = [
    "--enforce_eager",
    "--no_enable_prefix_caching",
    "--gpu-memory-utilization=0.7",
]
PROMPT = "Hello my name is Robert and I"

VOCAB = [str(i) for i in range(1000, 1050)]


def generate_random_enforced(max_tokens):
    n = max_tokens
    tokens = []
    used = set()
    for _ in range(n):
        tok = random.choice(VOCAB)
        while tok in used:
            tok = random.choice(VOCAB)
        used.add(tok)

        others = set(VOCAB) - {tok}
        top_two = random.choices(list(others), k=2)
        top = [tok] + top_two

        tokens.append({"token": tok, "top_tokens": top})
    return tokens


def validate_enforced_tokens(response_content, enforced_tokens):
    content = response_content
    enforced = enforced_tokens

    if len(content) < len(enforced):
        enforced = enforced[: len(content)]

    for i, enforced_item in enumerate(enforced):
        exp_tok = enforced_item["token"]
        exp_top = set(enforced_item["top_tokens"])

        model_tok = content[i]["token"]

        top_logits = content[i]["top_logprobs"]
        if isinstance(top_logits, dict):
            model_top = set(top_logits.keys())
        else:
            model_top = {t["token"] for t in top_logits}

        assert model_tok == exp_tok, (
            f"token mismatch at position {i}: {model_tok} != {exp_tok}"
        )
        assert model_top.issubset(exp_top), (
            f"top-logprobs mismatch at position {i}: {model_top} not subset of {exp_top}"
        )


def test_enforced_tokens():
    max_tokens = 10
    enforced = generate_random_enforced(max_tokens)

    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.99,
        "logprobs": True,
        "top_logprobs": 3,
        "enforced_tokens": {"tokens": enforced},
    }
    with RemoteOpenAIServer(MODEL, SERVER_ARGS) as remote_server:
        url = f"{remote_server.url_for('v1')}/chat/completions"
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        assert response.status_code == 200, response.text
        resp_json = response.json()

    content = resp_json["choices"][0]["logprobs"]["content"]
    validate_enforced_tokens(content, enforced)


@pytest.mark.parametrize("max_tokens", [5, 50, 100])
def test_enforced_tokens_different_lengths(max_tokens):
    enforced = generate_random_enforced(max_tokens)

    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.99,
        "logprobs": True,
        "top_logprobs": 3,
        "enforced_tokens": {"tokens": enforced},
    }
    with RemoteOpenAIServer(MODEL, SERVER_ARGS) as remote_server:
        url = f"{remote_server.url_for('v1')}/chat/completions"
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        assert response.status_code == 200, response.text
        resp_json = response.json()

    content = resp_json["choices"][0]["logprobs"]["content"]
    validate_enforced_tokens(content, enforced)


def test_enforced_tokens_batch_with_random_and_greedy():
    max_tokens = 10
    enforced = generate_random_enforced(max_tokens)

    headers = {"Content-Type": "application/json"}

    enforced_payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.99,
        "logprobs": True,
        "top_logprobs": 3,
        "enforced_tokens": {"tokens": enforced},
    }

    random_payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.99,
        "logprobs": True,
        "top_logprobs": 3,
    }

    greedy_payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 3,
    }

    with RemoteOpenAIServer(MODEL, SERVER_ARGS) as remote_server:
        url = f"{remote_server.url_for('v1')}/chat/completions"

        async def send_requests():
            async with aiohttp.ClientSession() as session:
                tasks = [
                    session.post(url, headers=headers, json=enforced_payload),
                    session.post(url, headers=headers, json=random_payload),
                    session.post(url, headers=headers, json=greedy_payload),
                ]
                responses = await asyncio.gather(*tasks)
                return responses

        responses = asyncio.run(send_requests())

        assert all(r.status == 200 for r in responses), "Some requests failed"

        enforced_resp = asyncio.run(responses[0].json())
        random_resp = asyncio.run(responses[1].json())
        greedy_resp = asyncio.run(responses[2].json())

    enforced_content = enforced_resp["choices"][0]["logprobs"]["content"]
    validate_enforced_tokens(enforced_content, enforced)

    random_content = random_resp["choices"][0]["logprobs"]["content"]
    greedy_content = greedy_resp["choices"][0]["logprobs"]["content"]

    enforced_tokens = [item["token"] for item in enforced_content]
    random_tokens = [item["token"] for item in random_content]
    greedy_tokens = [item["token"] for item in greedy_content]

    assert len(random_tokens) > 0, "Random sampling produced no tokens"
    assert len(greedy_tokens) > 0, "Greedy sampling produced no tokens"
    assert len(enforced_tokens) > 0, "Enforced sampling produced no tokens"
