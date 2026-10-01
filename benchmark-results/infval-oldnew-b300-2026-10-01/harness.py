# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-validation evaluation harness for DeepSeek-V4-Flash (fp8) on one B300.

One honest target P is loaded once. For every (prompt, attack-face) we run a single
forced-decode pass whose per-request logits processor, at each position t:
  * sees P's full-vocab logits L^P_t along the committed history,
  * builds the attack's effective logits L^Q_t = face(L^P_t),
  * draws the public Gumbel tape g_t (seeded by (job_seed, global_pos)),
  * records what BOTH validators need,
  * forces the committed token x_t = argmax(L^Q_t + g_t).

Because the token is forced, P's recorded logits are exactly what the validator would
recompute on the committed history -> both validators are scored on identical commits.

Stored per position (compact): committed token x, P's public-Gumbel winner z,
top-K of softmax(L^P) (ids+logprobs) = validator re-run, and top-K of softmax(L^Q)
(ids+logprobs) = executor self-report. Old vs new validators are computed offline.
"""

import json
import time

import torch

from vllm import LLM, SamplingParams
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

VOCAB_TOPK = 16  # stored support width per position
FORCE_HI, FORCE_LO = 1e4, -1e4
RECORDS = {}  # uid -> list of per-position dicts
META = {}  # uid -> {prompt_len, face, seed}


# ---------- public Gumbel tape -------------------------------------------------
def gumbel_tape(seed: int, global_pos: int, vocab: int, device, dtype=torch.float32):
    g = torch.Generator(device=device)
    g.manual_seed((seed * 1_000_003 + global_pos) & 0x7FFFFFFFFFFFFFFF)
    u = torch.rand(vocab, generator=g, device=device, dtype=dtype).clamp_(1e-12, 1.0)
    return -torch.log(-torch.log(u))


# ---------- attack faces: L^Q = face(L^P) --------------------------------------
def face_logits(face: str, LP: torch.Tensor):
    if face == "honest":
        return LP
    if face.startswith("temp"):  # wrong sampling temperature
        tau = float(face[4:])
        return LP / tau
    if face.startswith("mix"):  # Q=(1-a)P + a*Uniform  (a=1 -> pure noise)
        a = float(face[3:])
        p = torch.softmax(LP, dim=-1)
        V = LP.numel()
        q = (1.0 - a) * p + a * (1.0 / V)
        return torch.log(q.clamp_min(1e-30))
    if face.startswith(
        "int"
    ):  # emulated weight-quant: per-vector symmetric QDQ of logits
        bits = int(face[3:])
        qmax = (1 << (bits - 1)) - 1
        scale = LP.abs().max() / max(qmax, 1)
        if scale <= 0:
            return LP
        return torch.round(LP / scale).clamp_(-qmax - 1, qmax) * scale
    if face.startswith("topk"):  # truncated support
        k = int(face[4:])
        v, _ = torch.topk(LP, k)
        thr = v[-1]
        return torch.where(thr <= LP, LP, torch.full_like(LP, -1e30))
    raise ValueError(face)


# ---------- per-request logits processor (classic 3-arg callable) --------------
def make_req_lp(uid: str, face: str, seed: int, vocab: int):
    rec = RECORDS.setdefault(uid, [])

    def req_lp(prompt_ids, output_ids, logits):
        t = len(output_ids)
        if not META.get(uid, {}).get("prompt_len"):
            META[uid] = {"prompt_len": len(prompt_ids), "face": face, "seed": seed}
        gpos = len(prompt_ids) + t
        LP = logits.detach().float()
        g = gumbel_tape(seed, gpos, vocab, LP.device, LP.dtype)
        LQ = face_logits(face, LP)
        z = int(torch.argmax(LP + g))  # NEW validator target winner
        x = int(torch.argmax(LQ + g))  # executor committed token
        lseP = torch.log_softmax(LP, dim=-1)
        lseQ = torch.log_softmax(LQ, dim=-1)
        tP = torch.topk(lseP, VOCAB_TOPK)
        tQ = torch.topk(lseQ, VOCAB_TOPK)
        # P's logprob at Q's reported ids (needed when Q's top ids are off P's top)
        qids = tQ.indices
        lpP_at_q = lseP[qids]
        rec.append(
            {
                "t": t,
                "x": x,
                "z": z,
                "Pids": tP.indices.tolist(),
                "Plp": [round(v, 5) for v in tP.values.tolist()],
                "Qids": qids.tolist(),
                "Qlp": [round(v, 5) for v in tQ.values.tolist()],
                "PlpAtQ": [round(v, 5) for v in lpP_at_q.tolist()],
            }
        )
        out = torch.full_like(logits, FORCE_LO)
        out[x] = FORCE_HI
        return out

    return req_lp


class EvalLP(AdapterLogitsProcessor):
    def is_argmax_invariant(self):
        return False

    def new_req_logits_processor(self, params):
        ea = getattr(params, "extra_args", None) or {}
        uid = ea.get("uid")
        if uid is None:
            return None
        return make_req_lp(uid, ea["face"], int(ea["seed"]), int(ea["vocab"]))


# ---------- prompts ------------------------------------------------------------
PROMPTS = [
    "Explain step by step why the sky appears blue during the day.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "A train travels 60 km in 45 minutes. What is its average speed in km/h? Show your work.",
    "Summarize the causes of the fall of the Western Roman Empire.",
    "Prove that the square root of 2 is irrational.",
    "Describe how a hash map handles collisions and give a worked example.",
    "What is the capital of Australia and what is it known for?",
    "Translate 'The weather is nice today' into French and German.",
    "Give three practical tips for reducing household energy consumption.",
    "Implement binary search in C and explain its time complexity.",
    "What happens biochemically when muscles get tired during exercise?",
    "Outline the plot of a short mystery story set on a train.",
    "Compute the derivative of f(x)=x^3*sin(x) and show the steps.",
    "Explain the difference between TCP and UDP with examples.",
    "List the first 10 prime numbers and explain how to test primality.",
    "Describe the water cycle and its main stages.",
    "Write a SQL query to find the second-highest salary in an employees table.",
    "What are the main differences between mitosis and meiosis?",
    "Explain how public-key cryptography enables secure communication.",
    "A recipe needs 2.5 cups of flour for 12 cookies. How much for 30 cookies?",
    "Discuss the pros and cons of renewable versus fossil-fuel energy.",
    "Explain recursion to a beginner using the factorial function.",
    "What caused the 2008 financial crisis? Keep it concise.",
    "Describe how vaccines train the immune system.",
    "Write a haiku about the ocean and then explain its imagery.",
    "Explain the concept of entropy in thermodynamics.",
    "How does a transformer neural network use attention? Brief overview.",
    "Convert the decimal number 156 to binary and hexadecimal, with steps.",
    "What are design patterns in software engineering? Name three.",
    "Explain why the inner planets are rocky and the outer ones are gaseous.",
    "Describe the process of photosynthesis at a high level.",
    "Give a regular expression that matches valid IPv4 addresses and explain it.",
    "What is the difference between correlation and causation? Give an example.",
    "Explain how garbage collection works in managed languages.",
    "Solve the equation 3x^2 - 12x + 9 = 0 and show each step.",
    "Describe the main layers of the Earth from the crust to the core.",
    "What is Big-O notation and why does it matter for algorithms?",
    "Explain the greenhouse effect and its role in climate change.",
    "Write pseudocode for Dijkstra's shortest-path algorithm.",
    "What distinguishes a virus from a bacterium?",
]


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument(
        "--faces",
        type=str,
        default="honest,temp0.7,int8,int4,int3,int2,mix0.03,mix0.3,mix1.0",
    )
    ap.add_argument("--out", type=str, default="/work/records.jsonl")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    faces = args.faces.split(",")
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
        logits_processors=[EvalLP],
    )
    print("LOAD_SEC", round(time.time() - t0, 1), flush=True)
    vocab = llm.get_tokenizer().vocab_size
    try:
        vocab = llm.llm_engine.vllm_config.model_config.get_vocab_size()
    except Exception:
        pass
    print("VOCAB", vocab, flush=True)

    if args.selftest:
        prompts, faces_run, mt = PROMPTS[:2], ["honest", "mix1.0"], 8
    else:
        prompts, faces_run, mt = PROMPTS[: args.n], faces, args.max_tokens

    reqs, sps = [], []
    for pi, prompt in enumerate(prompts):
        seed = 100000 + pi
        for face in faces_run:
            uid = f"p{pi}_{face}"
            reqs.append(prompt)
            sps.append(
                SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    top_k=0,
                    min_p=0.0,
                    repetition_penalty=1.0,
                    max_tokens=mt,
                    ignore_eos=True,
                    extra_args={"uid": uid, "face": face, "seed": seed, "vocab": vocab},
                )
            )
    tg = time.time()
    llm.generate(reqs, sps)
    print("GEN_SEC", round(time.time() - tg, 1), "nseq", len(reqs), flush=True)

    with open(args.out, "w") as f:
        for uid, rows in RECORDS.items():
            f.write(json.dumps({"uid": uid, **META.get(uid, {}), "rows": rows}) + "\n")
    print("WROTE", args.out, "uids", len(RECORDS), flush=True)

    if args.selftest:
        for uid, rows in RECORDS.items():
            face = META[uid]["face"]
            rej = sum(1 for r in rows if r["x"] != r["z"])
            print(
                "SELFTEST",
                uid,
                "len",
                len(rows),
                "R_K",
                rej,
                "x",
                [r["x"] for r in rows[:6]],
                "z",
                [r["z"] for r in rows[:6]],
                flush=True,
            )


if __name__ == "__main__":
    main()
