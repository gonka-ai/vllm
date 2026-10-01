# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXEC pass with REAL weight quantization of MoE experts.
Loads a model, optionally QDQ-quantizes expert weight tensors to int-k per output channel
(real weight-domain fake-quant, not logits), then forces public-tape commits and dumps them.
"""

import argparse
import json
import time

import torch

from vllm import LLM, SamplingParams
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

TOPK = 16
HI, LO = 1e4, -1e4
EXEC = {}


def gumbel(seed, gpos, vocab, device, dtype=torch.float32):
    g = torch.Generator(device=device)
    g.manual_seed((seed * 1_000_003 + gpos) & 0x7FFFFFFFFFFFFFFF)
    u = torch.rand(vocab, generator=g, device=device, dtype=dtype).clamp_(1e-12, 1.0)
    return -torch.log(-torch.log(u))


def exec_cb(uid, seed, vocab, tau=1.0, topp=1.0):
    rec = EXEC.setdefault(uid, [])

    def cb(prompt_ids, output_ids, logits):
        t = len(output_ids)
        gpos = len(prompt_ids) + t
        LP = logits.detach().float()
        g = gumbel(seed, gpos, vocab, LP.device, LP.dtype)
        LT = LP / tau
        if topp < 1.0:
            sl, si = torch.sort(LT, descending=True)
            p = torch.softmax(sl, -1).cumsum(-1)
            mask = p - torch.softmax(sl, -1) >= topp
            sl = sl.masked_fill(mask, -1e30)
            LT = torch.full_like(LT, -1e30).scatter_(0, si, sl)
        x = int(torch.argmax(LT + g))  # tuned commit of the (quantized) executor
        tQ = torch.topk(torch.log_softmax(LP, -1), TOPK)
        rec.append(
            {
                "t": t,
                "x": x,
                "Qids": tQ.indices.tolist(),
                "Qlp": [round(v, 5) for v in tQ.values.tolist()],
            }
        )
        out = torch.full_like(logits, LO)
        out[x] = HI
        return out

    return cb


class ExecLP(AdapterLogitsProcessor):
    def is_argmax_invariant(self):
        return False

    def new_req_logits_processor(self, p):
        ea = getattr(p, "extra_args", None) or {}
        return (
            exec_cb(
                ea["uid"],
                int(ea["seed"]),
                int(ea["vocab"]),
                float(ea.get("tau", 1.0)),
                float(ea.get("topp", 1.0)),
            )
            if ea.get("mode") == "exec"
            else None
        )


def find_model(llm):
    cands = []

    def walk(o, d=0):
        if d > 6 or o is None:
            return
        if isinstance(o, torch.nn.Module):
            try:
                if any("experts" in n for n, _ in o.named_parameters()):
                    cands.append(o)
                    return
            except Exception:
                pass
        for a in (
            "model_executor",
            "driver_worker",
            "model_runner",
            "model",
            "engine_core",
            "llm_engine",
            "engine",
        ):
            if hasattr(o, a):
                try:
                    walk(getattr(o, a), d + 1)
                except Exception:
                    pass

    walk(llm)
    return cands[0] if cands else None


FP8 = getattr(torch, "float8_e4m3fn", None)


def _fp4_levels(device):
    lv = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=device)  # e2m1 magnitudes
    return torch.cat([-lv.flip(0)[:-1], lv])  # signed 15 levels


def quantize_experts(model, bits, group=16, qmode="int"):
    qmax = (1 << (bits - 1)) - 1
    n = 0
    LV = None
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "expert" not in name or "scale" in name or p.dim() < 2:
                continue
            W = p.data
            od = W.dtype
            C = W.shape[0]
            IN = W.shape[-1]
            ug = group if IN % group == 0 else 0
            if LV is None and qmode == "fp4":
                LV = _fp4_levels(W.device)
            step = max(1, C // ((W.numel() // (32 * 1024 * 1024)) + 1))
            for i in range(0, C, step):
                w = W[i : i + step].to(torch.bfloat16)
                sh = w.shape
                wg = w.reshape(*sh[:-1], IN // ug, ug) if ug else w
                amax = wg.abs().amax(-1, keepdim=True).clamp_min(1e-12)
                if qmode == "fp4":
                    xn = (wg / amax * 6.0).unsqueeze(-1)
                    wq = LV[(xn - LV).abs().argmin(-1)] / 6.0 * amax
                else:
                    s = amax / max(qmax, 1)
                    wq = torch.round(wg / s).clamp_(-qmax - 1, qmax) * s
                wq = wq.reshape(sh) if ug else wq
                W[i : i + step].copy_(wq.to(od))
                del w, wg, amax, wq
            n += 1
        torch.cuda.empty_cache()
    print(f"QUANTIZED {n} expert tensors {qmode}{bits} group{group}", flush=True)
    return n


from harness import PROMPTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/weights")
    ap.add_argument("--qbits", type=int, default=0)
    ap.add_argument("--qmode", default="int")
    ap.add_argument("--group", type=int, default=16)
    ap.add_argument("--drafttopk", type=int, default=0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--topp", type=float, default=1.0)
    ap.add_argument("--face", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    prompts = PROMPTS[: a.n]
    t0 = time.time()
    llm = LLM(
        model=a.model,
        trust_remote_code=True,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=2048,
        gpu_memory_utilization=0.80,
        kv_cache_dtype="fp8",
        tokenizer_mode="deepseek_v4",
        logits_processors=[ExecLP],
    )
    vocab = llm.llm_engine.vllm_config.model_config.get_vocab_size()
    print("LOAD_SEC", round(time.time() - t0, 1), "VOCAB", vocab, flush=True)
    if a.qbits > 0 or a.drafttopk > 0:
        m = find_model(llm)
        print("MODEL_FOUND", type(m).__name__ if m else None, flush=True)
        assert m is not None, "expert model not found"
        if a.drafttopk > 0:
            cnt = 0
            for name, mod in m.named_modules():
                if type(mod).__name__ == "RoutedExperts" and hasattr(mod, "top_k"):
                    mod.top_k = a.drafttopk
                    cnt += 1
            print(
                f"DRAFTER routed_experts.top_k={a.drafttopk} on {cnt} modules",
                flush=True,
            )
        if a.qbits > 0:
            quantize_experts(m, a.qbits, group=a.group, qmode=a.qmode)
    reqs = []
    sps = []
    for pi, pr in enumerate(prompts):
        uid = f"p{pi}_{a.face}"
        reqs.append(pr)
        sps.append(
            SamplingParams(
                temperature=0,
                top_p=1,
                top_k=0,
                min_p=0,
                repetition_penalty=1,
                max_tokens=a.max_tokens,
                ignore_eos=True,
                extra_args={
                    "mode": "exec",
                    "uid": uid,
                    "seed": 100000 + pi,
                    "vocab": vocab,
                    "tau": a.tau,
                    "topp": a.topp,
                },
            )
        )
    tg = time.time()
    llm.generate(reqs, sps)
    print("EXEC_SEC", round(time.time() - tg, 1), flush=True)
    with open(a.out, "w") as f:
        for uid, rows in EXEC.items():
            pi = int(uid.split("_")[0][1:])
            f.write(
                json.dumps(
                    {
                        "uid": uid,
                        "face": a.face,
                        "seed": 100000 + pi,
                        "rows": sorted(rows, key=lambda r: r["t"]),
                    }
                )
                + "\n"
            )
    print("WROTE", a.out, flush=True)


if __name__ == "__main__":
    main()
