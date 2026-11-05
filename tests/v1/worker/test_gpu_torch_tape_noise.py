# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Model Runner V2 target draws on torch's V1 seeded noise layout.

With a step source (`prompt_len`), a non-drafting `gumbel_sample` adds the
Gumbel noise -log(q), where q is the value torch's seeded CUDA `exponential_`
writes for (seed, output step, token). The kernels run on CPU under the Triton
interpreter, which needs a few module globals rebound (vLLM disables Triton in
the CPU test venv); monkeypatch undoes it after each test.
"""

import importlib.util
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.sample import gumbel as gumbel_mod
from vllm.v1.worker.gpu.spec_decode import rejection_sampler as rs_mod
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

# (seed, step, token, q): values torch wrote for a 100000-wide row on an
# RTX 3090 (gonka-ai/gonka#2215, testdata/torch_exponential_100000.json).
TORCH_GOLDEN = [
    (0, 0, 0, 0.9186772704124451),
    (0, 0, 255, 0.35865160822868347),
    (0, 0, 256, 2.4537434577941895),
    (0, 0, 99999, 0.3030778765678406),
    (0, 1, 0, 0.02815130539238453),
    (0, 1, 255, 0.02432016097009182),
    (0, 1, 256, 0.8566632866859436),
    (0, 1, 99999, 1.162268042564392),
    (0, 2, 0, 3.9399378299713135),
    (0, 2, 255, 0.5455689430236816),
    (0, 2, 256, 0.2265876680612564),
    (0, 2, 99999, 0.43885233998298645),
    (4294967301, 0, 0, 6.043552398681641),
    (4294967301, 0, 255, 0.7329369783401489),
    (4294967301, 0, 256, 2.0591790676116943),
    (4294967301, 0, 99999, 0.5803092122077942),
    (4294967301, 1, 0, 2.6773438453674316),
    (4294967301, 1, 255, 1.2910457849502563),
    (4294967301, 1, 256, 0.3938697874546051),
    (4294967301, 1, 99999, 0.9006578922271729),
    (4294967301, 2, 0, 2.1033294200897217),
    (4294967301, 2, 255, 0.5015251040458679),
    (4294967301, 2, 256, 0.17981216311454773),
    (4294967301, 2, 99999, 2.3792500495910645),
    (9223372036854775807, 0, 0, 1.6152443885803223),
    (9223372036854775807, 0, 255, 1.1341379880905151),
    (9223372036854775807, 0, 256, 0.36708691716194153),
    (9223372036854775807, 0, 99999, 1.6379188299179077),
    (9223372036854775807, 1, 0, 3.4416913986206055),
    (9223372036854775807, 1, 255, 0.6063733696937561),
    (9223372036854775807, 1, 256, 0.9147003293037415),
    (9223372036854775807, 1, 99999, 1.825962781906128),
    (9223372036854775807, 2, 0, 0.5392026901245117),
    (9223372036854775807, 2, 255, 0.6069435477256775),
    (9223372036854775807, 2, 256, 0.5418905019760132),
    (9223372036854775807, 2, 99999, 0.18747922778129578),
]

VOCAB = 1500
_M0, _M1 = 0xD2511F53, 0xCD9E8D57
_W0, _W1 = 0x9E3779B9, 0xBB67AE85
_MASK = 0xFFFFFFFF


@pytest.fixture
def interp(monkeypatch):
    """Run vLLM's real Gumbel kernels under the Triton interpreter on CPU."""
    import triton
    import triton.language as tl
    from triton.runtime.interpreter import InterpretedFunction

    monkeypatch.setenv("TRITON_INTERPRET", "1")
    # Triton's own jit'd helpers were decorated before the interpreter was on;
    # load copies of them that run interpreted.
    for member in dir(tl.tensor):
        if not member.startswith("__"):
            monkeypatch.setattr(tl.tensor, member, getattr(tl.tensor, member))
    for mod_name in ("random", "standard"):
        path = os.path.join(os.path.dirname(tl.__file__), mod_name + ".py")
        spec = importlib.util.spec_from_file_location(
            f"triton.language._{mod_name}", path
        )
        copy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(copy)
        for name, value in vars(copy).items():
            if isinstance(value, InterpretedFunction) and hasattr(tl, name):
                monkeypatch.setattr(tl, name, value)
    monkeypatch.setattr(gumbel_mod, "triton", triton)
    monkeypatch.setattr(gumbel_mod, "tl", tl)
    monkeypatch.setattr(gumbel_mod, "_TL_RAND_MIN", tl.constexpr(4.6566127342e-10))
    monkeypatch.setattr(gumbel_mod, "_DRAFT_NOISE_SALT", tl.constexpr(1 << 30))
    # The interpreter has no libdevice.
    monkeypatch.setattr(
        gumbel_mod, "tldevice", SimpleNamespace(log1p=lambda x: tl.log(1.0 + x))
    )
    for name in (
        "tl_rand64",
        "tl_rand32",
        "tl_torch_gumbel",
        "tl_domain_word",
        "tl_rand32_accept",
        "gumbel_noised_argmax",
        "gumbel_block_argmax",
        "_gumbel_sample_kernel",
    ):
        monkeypatch.setattr(gumbel_mod, name, triton.jit(getattr(gumbel_mod, name)))
    return triton, tl


def _philox_word0(seed: int, step: int, token: int, c3: int = 0) -> int:
    """First Philox4x32-10 word for key (seed lo, hi), counter (step, token, c3)."""
    c = [step & _MASK, (step >> 32) & _MASK, token & _MASK, c3]
    k0, k1 = seed & _MASK, (seed >> 32) & _MASK
    for _ in range(10):
        p0, p1 = _M0 * c[0], _M1 * c[2]
        c = [
            (p1 >> 32) ^ c[1] ^ k0,
            p1 & _MASK,
            (p0 >> 32) ^ c[3] ^ k1,
            p0 & _MASK,
        ]
        k0, k1 = (k0 + _W0) & _MASK, (k1 + _W1) & _MASK
    return c[0]


def _torch_q(seed: int, step: int, token: int) -> float:
    word = np.float32(_philox_word0(seed, step, token))
    u = np.float32(word * np.float32(2.3283064365386963e-10)) + np.float32(
        1.1641532182693481e-10
    )
    if u >= np.float32(1.0) - np.float32(2.0**-24):
        return 2.0**-24
    return float(-np.log(np.float64(u)))


def _noise_kernel(interp):
    triton, tl = interp

    @triton.jit
    def kernel(out_ptr, keys_ptr, seed, step, n, USE_FP64: tl.constexpr):
        offs = tl.arange(0, 16)
        keys = tl.load(keys_ptr + offs, mask=offs < n, other=0)
        noise = gumbel_mod.tl_torch_gumbel(seed, step, keys, USE_FP64=USE_FP64)
        tl.store(out_ptr + offs, noise.to(tl.float64), mask=offs < n)

    return kernel


@pytest.mark.parametrize("use_fp64", [False, True])
def test_torch_noise_matches_golden_torch_values(interp, use_fp64):
    kernel = _noise_kernel(interp)
    for seed in sorted({g[0] for g in TORCH_GOLDEN}):
        for step in sorted({g[1] for g in TORCH_GOLDEN}):
            rows = [g for g in TORCH_GOLDEN if g[0] == seed and g[1] == step]
            keys = torch.tensor([g[2] for g in rows], dtype=torch.int64)
            out = torch.zeros(16, dtype=torch.float64)
            kernel[(1,)](out, keys, seed, step, len(rows), USE_FP64=use_fp64)
            for (_, _, token, q), noise in zip(rows, out[: len(rows)].tolist()):
                got = float(np.exp(-noise))
                assert got == pytest.approx(q, abs=2e-6, rel=1e-6), (
                    seed,
                    step,
                    token,
                )


def test_numpy_philox_helper_matches_torch_and_clamps_top_word():
    """torch maps u = 1.0 to q = 2**-24, not -log(1) = 0."""
    assert _torch_q(0, 0, 0) == pytest.approx(0.9186772704124451, abs=2e-6)
    word_max = np.float32(0xFFFFFFFF)
    u = np.float32(word_max * np.float32(2.3283064365386963e-10)) + np.float32(
        1.1641532182693481e-10
    )
    assert u >= np.float32(1.0) - np.float32(2.0**-24)


SEEDS = [-7, 4294967301, 123456789, 0]
PROMPT_LEN = [3, 5, 1, 4]


def _expected(logits, idx_map, pos):
    out = []
    for row, (req, p) in enumerate(zip(idx_map, pos)):
        step = p + 1 - PROMPT_LEN[req]
        gumbel = np.array(
            [-np.log(_torch_q(SEEDS[req], step, k)) for k in range(logits.shape[1])]
        )
        out.append(int(np.argmax(logits[row].astype(np.float64) + gumbel)))
    return out


def _states():
    return (
        torch.ones(4),
        torch.tensor(SEEDS, dtype=torch.int64),
        torch.tensor(PROMPT_LEN, dtype=torch.int32),
    )


def test_gumbel_sample_with_step_source_matches_numpy_reference(interp):
    torch.manual_seed(0)
    logits = torch.randn(5, VOCAB) * 2
    idx_map = [2, 0, 1, 3, 1]
    pos = [7, 4, 12, 9, 13]
    temperature, seeds, prompt_len = _states()
    got = gumbel_mod.gumbel_sample(
        logits,
        torch.tensor(idx_map, dtype=torch.int32),
        temperature,
        seeds,
        torch.tensor(pos, dtype=torch.int64),
        apply_temperature=False,
        is_drafting=False,
        prompt_len=prompt_len,
    )
    assert got.tolist() == _expected(logits.numpy(), idx_map, pos)


def test_gumbel_sample_without_step_source_uses_other_stream(interp):
    torch.manual_seed(1)
    logits = torch.randn(3, VOCAB) * 2
    temperature, seeds, prompt_len = _states()
    args = (
        logits,
        torch.tensor([0, 1, 2], dtype=torch.int32),
        temperature,
        seeds,
        torch.tensor([5, 6, 7], dtype=torch.int64),
    )
    kwargs = dict(apply_temperature=False, is_drafting=False)
    plain = gumbel_mod.gumbel_sample(*args, **kwargs)
    with_step = gumbel_mod.gumbel_sample(*args, **kwargs, prompt_len=prompt_len)
    assert with_step.tolist() == _expected(logits.numpy(), [0, 1, 2], [5, 6, 7])
    assert plain.tolist() != with_step.tolist()


def test_raw_seed_streams_are_disjoint(interp):
    triton, tl = interp

    @triton.jit
    def kernel(out_ptr, seed, pos):
        tl.store(out_ptr, gumbel_mod.tl_domain_word(seed, pos, DOMAIN=1))
        tl.store(out_ptr + 1, gumbel_mod.tl_domain_word(seed, pos, DOMAIN=2))

    for seed, pos in [(-7, 3), (4294967301, 40), (123456789, 5000000000)]:
        out = torch.zeros(2, dtype=torch.int64)
        kernel[(1,)](out, seed, pos)
        accept, subseed = (int(v) & _MASK for v in out.tolist())
        torch_word = _philox_word0(seed, pos, 0)
        assert accept == _philox_word0(seed, pos, 0, c3=1)
        assert subseed == _philox_word0(seed, pos, 0, c3=2)
        assert len({accept, subseed, torch_word}) == 3


def _tape_sampler(monkeypatch, num_spec):
    monkeypatch.setattr(rs_mod, "apply_top_k_top_p", lambda logits, k, p: logits)
    temperature, seeds, prompt_len = _states()
    states = SimpleNamespace(
        temperature=SimpleNamespace(gpu=temperature),
        seeds=SimpleNamespace(gpu=seeds),
        get_top_k_top_p=lambda *a: (None, None),
    )
    sampler = SimpleNamespace(
        apply_sampling_params=lambda logits, *a, **kw: logits,
        sampling_states=states,
        req_states=SimpleNamespace(prompt_len=SimpleNamespace(gpu=prompt_len)),
        use_fp64_gumbel=False,
        logprobs_mode="raw_logprobs",
    )
    spec_config = SimpleNamespace(
        num_speculative_tokens=num_spec,
        enable_adaptive_verification=False,
        rejection_sample_method="tape",
        synthetic_acceptance_rates=None,
    )
    return RejectionSampler(sampler, spec_config, torch.device("cpu")), sampler


def test_tape_verify_emits_non_speculative_tokens(interp, monkeypatch):
    """Tape verify with the real kernel emits what non-spec sampling would."""
    torch.manual_seed(2)
    rejection_sampler, sampler = _tape_sampler(monkeypatch, num_spec=2)
    cu = [0, 3, 5]
    row_req = [1, 1, 1, 3, 3]
    pos = [8, 9, 10, 6, 7]
    logits = torch.randn(5, VOCAB) * 2
    idx_map = torch.tensor(row_req, dtype=torch.int32)
    tape = gumbel_mod.gumbel_sample(
        logits,
        idx_map,
        sampler.sampling_states.temperature.gpu,
        sampler.sampling_states.seeds.gpu,
        torch.tensor(pos, dtype=torch.int64),
        apply_temperature=False,
        is_drafting=False,
        prompt_len=sampler.req_states.prompt_len.gpu,
    ).tolist()
    assert tape == _expected(logits.numpy(), row_req, pos)

    # Row r + 1 holds the draft that row r verifies. The first request keeps
    # its first draft and breaks on its second; the second keeps its only one.
    wrong = (tape[1] + 1) % VOCAB
    drafts = [0, tape[0], wrong, 0, tape[3]]
    _, sampled, num_sampled = rejection_sampler._verify(
        logits,
        None,
        torch.tensor(drafts, dtype=torch.int32),
        torch.tensor(pos, dtype=torch.int64),
        torch.tensor(cu, dtype=torch.int32),
        torch.tensor([1, 3], dtype=torch.int32),
        np.array([1, 3]),
        idx_map,
        torch.tensor([0, 1, 2, 0, 1], dtype=torch.int32),
    )
    assert num_sampled.tolist() == [2, 2]
    assert sampled[0, :2].tolist() == tape[:2]
    assert sampled[1, :2].tolist() == tape[3:5]
