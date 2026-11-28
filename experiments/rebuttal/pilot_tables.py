"""Rebuttal: the pilots (multi-step arithmetic, multiple-choice kinship; Qwen2.5-7B, Llama-3.1-8B, Phi-4) in the format of the W2
controls and of the search-cost comparison. (CPU only)
Usage: python pilot_tables.py <dir with <model>_<pilot>_<cond>.csv, written by pilot.py>
1. Boundaries (probe / patch / steer / handoff / onset, as in pilot_eval.py) compared with the top-k peaks (k = 1, 3, 5) of
     own           the same model and condition
     other task    the conditions of the other pilot on the same model (arith <-> kin)
     other model   the same condition on another model, peak layers converted by relative depth
     circular shift  own peaks shifted circularly within the late layers (10,000 draws)
   Cells: exact (within +-1), averaged over the swapped-in sources.
2. Layers tested to find the handoff and the alpha = 1 steering onset (first late layer passing the test, as in search_cost.py),
   with a verified stop: full sweep, binary search, and the optimal decision tree under the fixed prior over the top-3 peaks
   (peak_search.py). Correct = the layer a full sweep returns.
Writes <dir>/../pilot_tables.md."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", ".."))
from pilot_eval import bounds, rise, EARLY
from peak_controls import peaks, dist
from peak_search import Probe, bisect_rest, optimal, prior, FIXED
import glob, os, numpy as np, pandas as pd
D = _s.argv[1]; KS = (1, 3, 5); NPERM = 10000
SHORT = {"Qwen2.5-7B-Instruct": "Qwen-7B", "Llama-3.1-8B-Instruct": "Llama-8B", "Phi-4-14B": "Phi-4"}
C = {}
for f in sorted(glob.glob(f"{D}/*.csv")):
    M, rest = os.path.basename(f)[:-4].split("_", 1); p, cond = rest.split("_", 1); C[(M, p, cond)] = pd.read_csv(f).sort_values("layer").reset_index(drop=True)
NL = {k: len(d) for k, d in C.items()}; X = {k: d["shape"].values.astype(float) for k, d in C.items()}; B = {k: bounds(d) for k, d in C.items()}
rng = np.random.default_rng(0); out = ["# Pilots: peak controls and search cost\n", f"{len(C)} conditions, {sum(len(b) for b in B.values())} boundaries.\n",
                                     "## 1. Boundaries exactly on a peak (within ±1)\n", "| Peaks from | Top-1 | Top-3 | Top-5 |", "|---|---|---|---|"]
res = {}
for k in KS:
    P = {key: peaks(X[key], k) for key in C}; acc = {"Own": [], "Other task, same model": [], "Other model (relative depth)": []}
    for (M, p, cond), b in B.items():
        oth = [key for key in C if key[0] == M and key[1] != p]
        mods = [key for key in C if key[0] != M and key[1:] == (p, cond)]
        conv = {key: sorted({int(round(q / NL[key] * NL[(M, p, cond)])) for q in P[key]}) for key in mods}
        for y in b.values():
            acc["Own"].append([dist(y, P[(M, p, cond)])])
            acc["Other task, same model"].append([dist(y, P[o]) for o in oth])
            acc["Other model (relative depth)"].append([dist(y, conv[o]) for o in mods])
    for lab, L in acc.items():
        res.setdefault(lab, {})[k] = (np.mean([np.mean(np.array(d) == 0) for d in L]), np.mean([np.mean(np.array(d) <= 1) for d in L]))
    n0, n1 = np.zeros(NPERM), np.zeros(NPERM)
    for i in range(NPERM):
        h0, h1 = [], []
        for key, b in B.items():
            lo = int(round(NL[key] * EARLY)); sp = NL[key] - lo; z = rng.integers(0, sp); sh = [lo + (q - lo + z) % sp for q in P[key]]
            for y in b.values(): d = dist(y, sh); h0.append(d == 0); h1.append(d <= 1)
        n0[i], n1[i] = np.mean(h0), np.mean(h1)
    res.setdefault("Circular shift", {})[k] = (n0.mean(), n1.mean())
for lab, r in res.items(): out.append(f"| {lab} | " + " | ".join(f"{r[k][0]:.2f} ({r[k][1]:.2f})" for k in KS) + " |")
# 2. search cost
def tests(d):
    T = {}; pi = d.patch_in.values; m = int(np.nanargmax(pi))
    if pi[m] >= 0.3: T["handoff"] = lambda L, pi=pi, m=m: L > m and pi[L] < 0.5 * pi[m]
    st = d.steer.values; T["onset (α=1)"] = lambda L, st=st: st[L] >= 0.5
    return T
rows = []
for key, d in C.items():
    lo, hi = int(round(NL[key] * EARLY)), NL[key] - 1; pk = peaks(X[key], 3)
    for ev, f in tests(d).items():
        truth = next((L for L in range(lo, hi + 1) if f(L)), None)
        if truth is None: continue
        r = {"Full sweep": (truth, hi - lo + 1), "Binary search": bisect_rest(Probe(f, lo, hi)),
             "Peaks (optimal tree, fixed prior)": optimal(Probe(f, lo, hi), prior(lo, hi, [(q, i) for i, q in enumerate(pk)], FIXED))}
        for s, (L, n) in r.items(): rows.append(dict(model=SHORT.get(key[0], key[0]), pilot=key[1], cond=key[2], event=ev, strategy=s, tests=n, correct=L == truth))
R = pd.DataFrame(rows); R.to_csv(f"{D}/../pilot_search.csv", index=False)
order = list(dict.fromkeys(R.strategy)); g = R.pivot_table(index="strategy", columns="event", values="tests").loc[order]
cr = R.groupby("strategy").correct.mean().loc[order]; ne = R.drop_duplicates(["model", "pilot", "cond", "event"]).event.value_counts()
out += ["\n## 2. Layers tested to find the boundary (mean)\n", "| Strategy | " + " | ".join(f"{e} (n={ne[e]})" for e in g.columns) + " | correct |", "|---" * (len(g.columns) + 2) + "|"]
out += [f"| {s} | " + " | ".join(f"{v:.1f}" for v in r) + f" | {cr[s]:.2f} |" for s, r in g.iterrows()]
out += ["\n### By model and pilot (mean over both events)\n", "| model | pilot | full sweep | binary | peaks |", "|---|---|---|---|---|"]
h = R.pivot_table(index=["model", "pilot"], columns="strategy", values="tests")
out += [f"| {m} | {p} | {r['Full sweep']:.1f} | {r['Binary search']:.1f} | {r['Peaks (optimal tree, fixed prior)']:.1f} |" for (m, p), r in h.iterrows()]
open(f"{D}/../pilot_tables.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
