# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import statistics
import sys
from collections import defaultdict

THRESH = 0.900


def posdist(reported, validator):
    omap = {i: lp for i, lp in reported}
    vals = sorted(omap.values())
    if len(vals) >= 2:
        m1, m2 = vals[0], vals[1]
    elif len(vals) == 1:
        m1, m2 = vals[0], vals[0] - 100.0
    else:
        return 0.0
    fb = m1 - (m2 - m1)
    d = 0.0
    for vid, vlp in validator:
        olp = omap.get(vid, fb)
        if olp != olp or vlp != vlp:
            d += 0.5
            continue
        d += abs(vlp - olp) / (1e-6 + abs(vlp) + abs(olp)) / 2.0
    return d / max(len(validator), 1)


def old_sim(rows, report="naive", k=5):
    N = len(rows)
    tot = 0.0
    for r in rows:
        val = list(zip(r["Pids"][:k], r["Plp"][:k]))
        rep = list(zip(r["Qids"][:k], r["Qlp"][:k])) if report == "naive" else val
        tot += posdist(rep, val)
    return 1.0 - tot / max(100, N)


def RK(rows):
    return sum(1 for r in rows if r["x"] != r["z"])


def RKm(rows, eps):
    return sum(
        1 for r in rows if r["x"] != r["z"] and (r.get("m") or 0) > eps
    )  # margin-gated


def Sm(rows):
    return sum((r.get("m") or 0) for r in rows)  # cumulative margin deficit


seqs = [
    json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "records4.jsonl")
]
bf = defaultdict(list)
for s in seqs:
    bf[s["face"]].append(s)
HS = bf.get("honest_solo", [])
# plain cutoff
hs_rk = sorted(RK(s["rows"]) for s in HS)
c = max(hs_rk) if hs_rk else 14
# margin eps: max honest_solo per-position margin at flips -> removes honest near-tie flips
hs_flipm = [(r.get("m") or 0) for s in HS for r in s["rows"] if r["x"] != r["z"]]
eps = max(hs_flipm) if hs_flipm else 0.0
hs_rkm = sorted(RKm(s["rows"], eps) for s in HS)
cg = max(hs_rkm) if hs_rkm else 0
hs_sm = sorted(Sm(s["rows"]) for s in HS)
csm = max(hs_sm) if hs_sm else 0
print(
    f"# honest_solo: R_K max={c} | margin flips: eps={eps:.3f} -> gated R_K max={cg} | cumulative Σm max={csm:.1f}"
)
print(
    f"# honest_solo flip margins: p50={statistics.median(hs_flipm):.3f} p90={sorted(hs_flipm)[int(0.9 * len(hs_flipm))]:.3f} max={eps:.3f}  (near-ties)\n"
)

hdr = f"{'face':11} {'n':>2} | {'OLDnaive':>8} {'OLDlaun':>7} | {'NEW_RK':>6} {'NEWdet':>6} | {'NEWmargin':>9} {'Σm_med':>7} {'MGdet':>6}"
print(hdr)
print("-" * len(hdr))
for face in [
    "honest",
    "honest_solo",
    "nvfp4",
    "int8",
    "int4",
    "int3",
    "int2",
    "temp0.7",
    "mix0.01",
    "mix0.1",
    "topk1",
    "mix1.0",
]:
    ss = bf.get(face, [])
    if not ss:
        continue
    n = len(ss)
    od = sum(1 for s in ss if old_sim(s["rows"], "naive") <= THRESH) / n
    ol = sum(1 for s in ss if old_sim(s["rows"], "launder") <= THRESH) / n
    rk = sorted(RK(s["rows"]) for s in ss)
    nd = sum(1 for v in rk if v > c) / n
    rkm = sorted(RKm(s["rows"], eps) for s in ss)
    mgd = sum(1 for v in rkm if v > cg) / n
    sm = sorted(Sm(s["rows"]) for s in ss)
    smd = sum(1 for v in sm if v > csm) / n
    print(
        f"{face:11} {n:>2} | {od:>8.0%} {ol:>7.0%} | {statistics.median(rk):>6.0f} {nd:>6.0%} | {statistics.median(rkm):>9.0f} {statistics.median(sm):>7.0f} {smd:>6.0%}"
    )
print(
    "\n# OLDnaive/OLDlaun: OLD detection, honest self-report vs prefill-laundered P report."
)
print(
    f"# NEW_RK: median plain R_K; NEWdet: share R_K>{c}.  NEWmargin: median margin-gated R_K (eps={eps:.2f}); MGdet: share gated R_K>{cg}."
)
print("# Σm_med: median cumulative margin deficit (continuous NEW score).")
