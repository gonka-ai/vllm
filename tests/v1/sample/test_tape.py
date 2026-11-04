# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The public tape: its noise is a published function, its winners sample the
requested distribution, and its sampling set is the one vLLM's order defines."""

import math
import struct

import numpy as np
import pytest
import torch

from vllm.v1.sample.tape import (
    _MASKED_GAP,
    TAPE_DOMAIN,
    TapeBatch,
    TapeRow,
    cut_logits,
    take,
    tape_noise,
    tape_winners,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

U32 = 0xFFFFFFFF


def murmur3_x86_32(data: bytes, seed: int) -> int:
    h = seed
    for (k,) in struct.iter_unpack("<I", data):
        k = (k * 0xCC9E2D51) & U32
        k = ((k << 15) | (k >> 17)) & U32
        k = (k * 0x1B873593) & U32
        h ^= k
        h = ((h << 13) | (h >> 19)) & U32
        h = (h * 5 + 0xE6546B64) & U32
    h ^= len(data)
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & U32
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & U32
    return h ^ (h >> 16)


def reference_noise(seed: int, t: int, token: int) -> float:
    block = struct.pack("<IIII", seed & U32, seed >> 32, t, token)
    u = (murmur3_x86_32(block, TAPE_DOMAIN) + 0.5) * 2.0**-32
    return -math.log(-math.log1p(-u))


def seeds_tensor(seeds: list[int]) -> torch.Tensor:
    pairs = np.array([(s & U32, s >> 32) for s in seeds], dtype=np.uint32)
    return torch.from_numpy(pairs.astype(np.int32)).to("cuda")


def test_murmur3_reference_is_the_standard_hash():
    # Published vector of MurmurHash3_x86_32 (block-aligned input).
    assert murmur3_x86_32(b"\x00\x00\x00\x00", 0) == 0x2362F9DE
    assert murmur3_x86_32(b"\xff\xff\xff\xff", 0) == 0x76293B50


def test_noise_is_the_published_function_of_seed_position_and_token():
    seeds = [0x123456789AB * (i + 1) & 0xFFFFFFFFFFFFFFFF for i in range(512)]
    positions = [i * 7 for i in range(512)]
    tokens = [(i * 131) % 200000 for i in range(512)]
    noise = tape_noise(
        seeds_tensor(seeds),
        torch.tensor(positions, dtype=torch.int32, device="cuda"),
        torch.tensor(tokens, dtype=torch.int64, device="cuda"),
    )
    expected = [reference_noise(*args) for args in zip(seeds, positions, tokens)]
    assert np.allclose(noise.cpu().numpy(), expected, atol=4e-5, rtol=1e-5)


def test_winner_is_the_argmax_of_logits_plus_noise_whatever_the_batch():
    vocab, rows = 4096, 8
    torch.manual_seed(0)
    logits = (torch.randn(rows, vocab, device="cuda") * 3).float()
    seeds = [1000 + i for i in range(rows)]
    positions = torch.arange(rows, dtype=torch.int32, device="cuda") * 3
    winners, _ = tape_winners(logits, seeds_tensor(seeds), positions)
    for r in range(rows):
        noise = [reference_noise(seeds[r], int(positions[r]), t) for t in range(vocab)]
        scores = logits[r].cpu().double().numpy() + np.array(noise)
        assert int(winners[r]) == int(scores.argmax())
        alone, _ = tape_winners(
            logits[r : r + 1],
            seeds_tensor(seeds[r : r + 1]),
            positions[r : r + 1],
        )
        assert int(alone) == int(winners[r])


@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_winners_sample_the_distribution_at_temperature(temperature):
    raw = torch.tensor([3.0, 2.0, 1.0, 0.0, -2.0, -5.0], device="cuda")
    logits = raw / temperature
    n = 400_000
    winners, _ = tape_winners(
        logits.expand(n, -1).contiguous(),
        seeds_tensor([987654321] * n),
        torch.arange(n, dtype=torch.int32, device="cuda"),
    )
    freq = torch.bincount(winners, minlength=6).double() / n
    p = logits.softmax(-1).double()
    # Five standard errors per token.
    assert ((freq - p).abs() <= 5 * (p * (1 - p) / n).sqrt()).all()


def test_rare_tokens_are_reached_at_their_mass():
    # The tail where a truncated uniform or noise table goes wrong.
    logits = torch.tensor([0.0, -9.0, -12.0], device="cuda")
    n = 8_000_000
    winners, _ = tape_winners(
        logits.expand(n, -1).contiguous(),
        seeds_tensor([42] * n),
        torch.arange(n, dtype=torch.int32, device="cuda"),
    )
    counts = torch.bincount(winners, minlength=3).double()
    expected = logits.softmax(-1).double() * n
    assert ((counts - expected).abs() <= 5 * expected.sqrt())[1:].all()


def brute_force_kept(lg, top_k, top_p, min_p):
    probs = lg.softmax(-1)
    kept = probs >= min_p * probs.max() if min_p > 0 else torch.ones_like(lg).bool()
    values = lg[kept].sort(descending=True).values
    if 0 < top_k < lg.numel():
        values = values[:top_k]
    theta = values[-1]
    if top_p < 1.0:
        p = values.softmax(-1)
        above = p.cumsum(-1) - p
        theta = values[(above < top_p).sum() - 1]
    return lg >= theta


def cut(logits, top_k, top_p, min_p):
    rows, vocab = logits.shape
    return cut_logits(
        logits.clone(),
        torch.full((rows,), top_k or vocab, dtype=torch.int32, device="cuda"),
        torch.full((rows,), top_p, dtype=torch.float32, device="cuda"),
        torch.full((rows,), min_p, dtype=torch.float32, device="cuda"),
    )


CUTS = [
    (20, 1.0, 0.0),
    (0, 0.95, 0.0),
    (0, 1.0, 0.1),
    (40, 0.95, 0.0),
    (0, 0.95, 0.01),
    (20, 0.8, 0.05),
    (5000, 0.999, 0.0),
]


@pytest.mark.parametrize("top_k,top_p,min_p", CUTS)
@pytest.mark.parametrize("scale", [8.0, 0.05])
def test_cut_keeps_the_set_vllm_defines(top_k, top_p, min_p, scale):
    torch.manual_seed(1)
    logits = (torch.randn(6, 8192, device="cuda") * scale).float()
    masked = cut(logits, top_k, top_p, min_p)
    for r in range(6):
        kept = masked[r] > float("-inf")
        expected = brute_force_kept(logits[r], top_k, top_p, min_p)
        # fp32 cumulative sums may place the nucleus boundary a token apart.
        assert (kept ^ expected).sum() <= 2


@pytest.mark.parametrize("top_k,top_p,min_p", CUTS)
@pytest.mark.parametrize("scale", [8.0, 2.0, 0.05])
def test_cut_of_a_row_depends_on_its_batch_only_by_rounding(top_k, top_p, min_p, scale):
    # An executor and a validator batch differently, and vLLM's top-p sums fp32
    # mass in a batch-dependent order. The tokens that move carry that rounding
    # as their mass, which bounds how often one of them wins.
    torch.manual_seed(2)
    logits = (torch.randn(256, 8192, device="cuda") * scale).float()
    full = cut(logits, top_k, top_p, min_p) > float("-inf")
    probs = logits.softmax(-1)
    for size in (1, 8, 64, 65, 128):
        part = cut(logits[:size], top_k, top_p, min_p) > float("-inf")
        moved = ((part ^ full[:size]) * probs[:size]).sum(-1)
        assert moved.max() < 1e-3, size


def test_no_cut_leaves_every_token_in():
    logits = torch.zeros(2, 16, device="cuda")
    assert (cut_logits(logits, None, None, None) == 0.0).all()


def tape_batch(rows, positions):
    batch = TapeBatch(
        list(range(len(rows))), rows, torch.device("cuda"), covers_random=True
    )
    batch.set_positions(np.array(positions))
    return batch


def test_replay_signals_score_the_committed_token_against_the_winner():
    torch.manual_seed(4)
    logits = (torch.randn(3, 4096, device="cuda") * 3).float()
    winners, best = tape_winners(
        logits.clone(),
        seeds_tensor([5, 6, 7]),
        torch.arange(3, dtype=torch.int32, device="cuda"),
    )
    other = int(logits[1].argmin())
    batch = tape_batch(
        [
            TapeRow(5, 0.0, committed=[int(winners[0])], tag="right"),
            TapeRow(6, 0.0, committed=[0, other], tag="wrong"),
            TapeRow(7, 0.0, committed=[0], tag="past the end"),
        ],
        [0, 1, 2],
    )
    assert batch.sample(logits.clone(), None, None).tolist() == winners.tolist()
    signals = take(["right", "wrong", "past the end"])
    assert signals["right"] == {"steps": 1, "count": 0, "max_gap": 0.0, "sum_gap": 0.0}
    assert signals["wrong"]["count"] == 1
    expected = float(best[1]) - float(logits[1, other]) - reference_noise(6, 1, other)
    assert signals["wrong"]["max_gap"] == pytest.approx(expected, abs=1e-3)
    assert "past the end" not in signals


def test_replay_of_a_token_outside_the_grammar_gets_the_masked_gap():
    logits = torch.zeros(1, 64, device="cuda")
    logits[0, 9] = float("-inf")
    batch = tape_batch([TapeRow(5, 0.0, committed=[9], tag="m")], [0])
    batch.sample(logits, None, None)
    assert take(["m"])["m"]["max_gap"] == _MASKED_GAP


def test_verify_finds_the_winner_each_draft_position_would_draw():
    torch.manual_seed(5)
    vocab = 2048
    rows = [TapeRow(11, 0.0), TapeRow(12, 0.05)]
    batch = tape_batch(rows, [3, 8])
    num_drafts = [2, 3]
    target = (torch.randn(5, vocab, device="cuda") * 4).float()
    temperature = torch.tensor([1.0, 0.7], device="cuda")
    top_k = torch.tensor([20, 40], dtype=torch.int32, device="cuda")
    top_p = torch.tensor([0.95, 0.9], dtype=torch.float32, device="cuda")
    targets = batch.verify(target, num_drafts, temperature, top_k, top_p)
    at = [(0, 0, 3), (0, 1, 4), (1, 2, 8), (1, 3, 9), (1, 4, 10)]
    for row, token, position in at:
        logits = target[token : token + 1] / temperature[row]
        one = cut_logits(
            logits.clone(),
            top_k[row : row + 1],
            top_p[row : row + 1],
            batch.min_p[row : row + 1],
        )
        winner, _ = tape_winners(
            one,
            batch.seeds[row : row + 1],
            torch.tensor([position], dtype=torch.int32, device="cuda"),
        )
        assert int(targets[token]) == int(winner)


def test_select_follows_the_rows_when_the_sampler_is_narrowed():
    # Decode PoC rows leave the batch before the chat rows are sampled.
    torch.manual_seed(6)
    logits = (torch.randn(6, 2048, device="cuda") * 4).float()
    rows = [TapeRow(21, 0.0), TapeRow(22, 0.0)]
    batch = TapeBatch([2, 5], rows, torch.device("cuda"), covers_random=True)
    batch.set_positions(np.array([0, 0, 7, 0, 0, 9]))
    narrowed = batch.select({2: 0, 5: 1})
    expected, _ = tape_winners(
        logits[[2, 5]].clone(),
        seeds_tensor([21, 22]),
        torch.tensor([7, 9], dtype=torch.int32, device="cuda"),
    )
    assert (narrowed.sample(logits[[2, 5]], None, None) == expected).all()
