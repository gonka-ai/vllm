# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VALIDATOR pass with CLEAN P over committed sequences from several exec files.
Batched validation (z, margin, P top-k) for every committed face; plus honest SOLO twin for FPR.
Writes merged records (x,z,m,xlp,zlp,Qids,Qlp,Pids,Plp) for analyze3."""

import argparse
import glob
import json
import time

import torch

from vllm import LLM, SamplingParams
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

TOPK = 16
HI, LO = 1e4, -1e4
VAL = {}


def gumbel(seed, gpos, vocab, device, dtype=torch.float32):
    g = torch.Generator(device=device)
    g.manual_seed((seed * 1_000_003 + gpos) & 0x7FFFFFFFFFFFFFFF)
    u = torch.rand(vocab, generator=g, device=device, dtype=dtype).clamp_(1e-12, 1.0)
    return -torch.log(-torch.log(u))


def val_cb(uid, seed, vocab, committed):
    rec = VAL.setdefault(uid, [])

    def cb(prompt_ids, output_ids, logits):
        t = len(output_ids)
        gpos = len(prompt_ids) + t
        LP = logits.detach().float()
        s = LP + gumbel(seed, gpos, vocab, LP.device, LP.dtype)
        z = int(torch.argmax(s))
        x = committed[t] if t < len(committed) else z
        m = float(s[z] - s[x])
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
    ap.add_argument("--commits", required=True, help="glob of exec commit files")
    ap.add_argument("--out", default="/work/records_real.jsonl")
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    commits = {}  # uid -> {face,seed,committed}
    for fp in sorted(glob.glob(a.commits)):
        for l in open(fp):
            s = json.loads(l)
            commits[s["uid"]] = {
                "face": s["face"],
                "seed": s["seed"],
                "committed": [r["x"] for r in s["rows"]],
                "qrep": {r["t"]: (r["Qids"], r["Qlp"]) for r in s["rows"]},
            }
    print(
        "COMMITS",
        len(commits),
        "faces",
        sorted({v["face"] for v in commits.values()}),
        flush=True,
    )
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
        logits_processors=[ValLP],
    )
    vocab = llm.llm_engine.vllm_config.model_config.get_vocab_size()
    print("LOAD_SEC", round(time.time() - t0, 1), flush=True)

    def sp(uid, seed, committed):
        return SamplingParams(
            temperature=0,
            top_p=1,
            top_k=0,
            min_p=0,
            repetition_penalty=1,
            max_tokens=len(committed),
            ignore_eos=True,
            extra_args={
                "mode": "val",
                "uid": uid,
                "seed": seed,
                "vocab": vocab,
                "committed": committed,
            },
        )

    # batched validation of all committed faces
    reqs = []
    sps = []
    for uid, c in commits.items():
        pi = int(uid.split("_")[0][1:])
        reqs.append(PROMPTS[pi])
        sps.append(sp(uid, c["seed"], c["committed"]))
    tb = time.time()
    llm.generate(reqs, sps)
    print("VALB_SEC", round(time.time() - tb, 1), flush=True)
    # honest SOLO twin (numerical FPR): re-validate honest commits one-at-a-time
    tc = time.time()
    nsolo = 0
    for uid, c in commits.items():
        if c["face"] != "honest":
            continue
        solo = uid + "__solo"
        llm.generate(
            [PROMPTS[int(uid.split("_")[0][1:])]], [sp(solo, c["seed"], c["committed"])]
        )
        nsolo += 1
    print("VALC_SEC", round(time.time() - tc, 1), "nsolo", nsolo, flush=True)
    with open(a.out, "w") as f:
        for uid, c in commits.items():
            for tag, face in [(uid, c["face"])] + (
                [(uid + "__solo", "honest_solo")] if c["face"] == "honest" else []
            ):
                va = {r["t"]: r for r in VAL.get(tag, [])}
                rows = []
                for t, x in enumerate(c["committed"]):
                    v = va.get(t)
                    if not v:
                        continue
                    qids, qlp = c["qrep"].get(t, ([], []))
                    rows.append(
                        {
                            "t": t,
                            "x": x,
                            "z": v["z"],
                            "m": v["m"],
                            "xlp": v["xlp"],
                            "zlp": v["zlp"],
                            "Qids": qids,
                            "Qlp": qlp,
                            "Pids": v["Pids"],
                            "Plp": v["Plp"],
                        }
                    )
                f.write(
                    json.dumps(
                        {"uid": tag, "face": face, "seed": c["seed"], "rows": rows}
                    )
                    + "\n"
                )
    print("WROTE", a.out, flush=True)


if __name__ == "__main__":
    main()
