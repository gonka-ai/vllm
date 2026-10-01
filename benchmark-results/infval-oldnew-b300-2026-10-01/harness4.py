# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""old-vs-new inference-validation eval, DeepSeek-V4-Flash fp8, one B300, one load.
PASS A exec (batched): force x_t=argmax(face(L^P)+g_t), record x + exec self-report top-k.
PASS B val (batched): recompute P, record z_t=argmax(L^P+g_t) + P top-k  -> attack detection.
PASS C val honest (SOLO batch=1): numerical twin -> realistic honest FPR / NEW cutoff.
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
VAL = {}


def gumbel(seed, gpos, vocab, device, dtype=torch.float32):
    g = torch.Generator(device=device)
    g.manual_seed((seed * 1_000_003 + gpos) & 0x7FFFFFFFFFFFFFFF)
    u = torch.rand(vocab, generator=g, device=device, dtype=dtype).clamp_(1e-12, 1.0)
    return -torch.log(-torch.log(u))


def face_logits(face, LP):
    if face == "honest":
        return LP
    if face.startswith("temp"):
        return LP / float(face[4:])
    if face.startswith("mix"):
        a = float(face[3:])
        p = torch.softmax(LP, -1)
        V = LP.numel()
        return torch.log(((1 - a) * p + a * (1.0 / V)).clamp_min(1e-30))
    if face.startswith("int"):
        b = int(face[3:])
        qmax = (1 << (b - 1)) - 1
        s = LP.abs().max() / max(qmax, 1)
        return LP if s <= 0 else torch.round(LP / s).clamp_(-qmax - 1, qmax) * s
    if face == "nvfp4":
        V = LP.numel()
        pad = (16 - V % 16) % 16
        x = torch.nn.functional.pad(LP, (0, pad)).view(-1, 16)
        amax = x.abs().amax(-1, keepdim=True).clamp_min(1e-12)
        lv = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=LP.device)
        lv = torch.cat([-lv.flip(0)[:-1], lv])
        xn = x / amax * 6.0
        q = lv[(xn.unsqueeze(-1) - lv).abs().argmin(-1)] / 6.0 * amax
        return q.view(-1)[:V]
    if face.startswith("topk"):
        k = int(face[4:])
        v, _ = torch.topk(LP, k)
        thr = v[-1]
        return torch.where(thr <= LP, LP, torch.full_like(LP, -1e30))
    raise ValueError(face)


def exec_cb(uid, face, seed, vocab):
    rec = EXEC.setdefault(uid, [])

    def cb(prompt_ids, output_ids, logits):
        t = len(output_ids)
        gpos = len(prompt_ids) + t
        LP = logits.detach().float()
        g = gumbel(seed, gpos, vocab, LP.device, LP.dtype)
        LQ = face_logits(face, LP)
        x = int(torch.argmax(LQ + g))
        tQ = torch.topk(torch.log_softmax(LQ, -1), TOPK)
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


def val_cb(uid, seed, vocab, committed):
    rec = VAL.setdefault(uid, [])

    def cb(prompt_ids, output_ids, logits):
        t = len(output_ids)
        gpos = len(prompt_ids) + t
        LP = logits.detach().float()
        s = LP + gumbel(seed, gpos, vocab, LP.device, LP.dtype)
        z = int(torch.argmax(s))
        x = committed[t] if t < len(committed) else z
        m = float(s[z] - s[x])  # adjusted-score margin (>=0; 0 iff x==z)
        lse = torch.log_softmax(LP, -1)
        tP = torch.topk(lse, TOPK)
        rec.append(
            {
                "t": t,
                "x": x,
                "z": z,
                "m": round(m, 5),
                "xlp": round(float(lse[x]), 5),
                "zlp": round(float(lse[z]), 5),
                "Pids": tP.indices.tolist(),
                "Plp": [round(v, 5) for v in tP.values.tolist()],
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
            exec_cb(ea["uid"], ea["face"], int(ea["seed"]), int(ea["vocab"]))
            if ea.get("mode") == "exec"
            else None
        )


class ValLP(AdapterLogitsProcessor):
    def is_argmax_invariant(self):
        return False

    def new_req_logits_processor(self, p):
        ea = getattr(p, "extra_args", None) or {}
        return (
            val_cb(ea["uid"], int(ea["seed"]), int(ea["vocab"]), ea["committed"])
            if ea.get("mode") == "val"
            else None
        )


from harness import PROMPTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument(
        "--faces",
        type=str,
        default="honest,temp0.7,nvfp4,int8,int4,int3,int2,mix0.01,mix0.1,mix1.0,topk1",
    )
    ap.add_argument("--out", type=str, default="/work/records4.jsonl")
    a = ap.parse_args()
    faces = a.faces.split(",")
    prompts = PROMPTS[: a.n]
    t0 = time.time()
    llm = LLM(
        model="/weights",
        trust_remote_code=True,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        kv_cache_dtype="fp8",
        tokenizer_mode="deepseek_v4",
        logits_processors=[ExecLP, ValLP],
    )
    vocab = llm.llm_engine.vllm_config.model_config.get_vocab_size()
    print("LOAD_SEC", round(time.time() - t0, 1), "VOCAB", vocab, flush=True)

    def sp(**ea):
        return SamplingParams(
            temperature=0,
            top_p=1,
            top_k=0,
            min_p=0,
            repetition_penalty=1,
            max_tokens=ea.pop("mt", a.max_tokens),
            ignore_eos=True,
            extra_args=ea,
        )

    # PASS A exec batched
    reqs = []
    sps = []
    for pi, pr in enumerate(prompts):
        for face in faces:
            reqs.append(pr)
            sps.append(
                sp(
                    mode="exec",
                    uid=f"p{pi}_{face}",
                    face=face,
                    seed=100000 + pi,
                    vocab=vocab,
                )
            )
    ta = time.time()
    llm.generate(reqs, sps)
    print("EXEC_SEC", round(time.time() - ta, 1), "nseq", len(reqs), flush=True)
    # PASS B val batched
    reqs = []
    sps = []
    for pi, pr in enumerate(prompts):
        for face in faces:
            uid = f"p{pi}_{face}"
            committed = [r["x"] for r in sorted(EXEC[uid], key=lambda r: r["t"])]
            reqs.append(pr)
            sps.append(
                sp(
                    mode="val",
                    uid=uid,
                    seed=100000 + pi,
                    vocab=vocab,
                    committed=committed,
                    mt=len(committed),
                )
            )
    tb = time.time()
    llm.generate(reqs, sps)
    print("VALB_SEC", round(time.time() - tb, 1), flush=True)
    # PASS C val honest SOLO (numerical twin for FPR)
    tc = time.time()
    for pi, pr in enumerate(prompts):
        uid = f"p{pi}_honest"
        solo = f"p{pi}_honest__solo"
        committed = [r["x"] for r in sorted(EXEC[uid], key=lambda r: r["t"])]
        llm.generate(
            [pr],
            [
                sp(
                    mode="val",
                    uid=solo,
                    seed=100000 + pi,
                    vocab=vocab,
                    committed=committed,
                    mt=len(committed),
                )
            ],
        )
    print("VALC_SEC", round(time.time() - tc, 1), flush=True)
    # merge
    with open(a.out, "w") as f:

        def emit(uid, face, pi):
            ex = sorted(EXEC[uid.replace("__solo", "")], key=lambda r: r["t"])
            vb = {r["t"]: r for r in VAL.get(uid, [])}
            rows = []
            for e in ex:
                v = vb.get(e["t"])
                if v:
                    rows.append(
                        {
                            "t": e["t"],
                            "x": e["x"],
                            "z": v["z"],
                            "m": v.get("m"),
                            "xlp": v.get("xlp"),
                            "zlp": v.get("zlp"),
                            "Qids": e["Qids"],
                            "Qlp": e["Qlp"],
                            "Pids": v["Pids"],
                            "Plp": v["Plp"],
                        }
                    )
            f.write(
                json.dumps(
                    {"uid": uid, "face": face, "seed": 100000 + pi, "rows": rows}
                )
                + "\n"
            )

        for pi in range(len(prompts)):
            for face in faces:
                emit(f"p{pi}_{face}", face, pi)
            if f"p{pi}_honest__solo" in VAL:
                emit(f"p{pi}_honest__solo", "honest_solo", pi)
    print("WROTE", a.out, flush=True)


if __name__ == "__main__":
    main()
