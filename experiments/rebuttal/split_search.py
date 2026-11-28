"""Rebuttal (DQWj): search cost when the peaks come from prompts disjoint from those that define the boundary. (CPU)
Usage: python split_search.py     (reads DATA/split/{A,B}; written by run_split.sh)
Events, measured on half B as in search_cost.py / split_eval.py:
  handoff  patching the input token has passed its maximum and fallen below half of it (maximum >= 0.2)
  onset    steering all answer positions (alpha = 1) switches >= half of the prompts
Strategies with a verified stop (peak_search.py): full sweep of the late layers, binary search, and the optimal decision tree under
the fixed prior over the top-3 peaks of half A (disjoint) or of half B (same prompts). Writes DATA/split/split_search.md and .csv."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", ".."))
from paths import DATA
from peak_controls import peaks
from peak_search import Probe, bisect_rest, optimal, prior, FIXED
from split_eval import curve, ARITH, SHORT, EARLY, S
import glob, os, numpy as np, pandas as pd
rows = []
for f in sorted(glob.glob(f"{S}/B/rebuttal/metrics/*.csv")):
    stem = os.path.basename(f)[:-4]; M, t = stem.rsplit("_", 1); fa = f"{S}/A/rebuttal/metrics/{stem}.csv"; fp = f"{S}/B/handoff_patch/{stem}.csv"
    fs = f"{S}/B/handoff/{stem}.csv" if t in ARITH else f"{S}/B/handoff_retrieval/{stem}.csv"
    if not all(os.path.exists(x) for x in (fa, fp, fs)): continue
    xa, xb = pd.read_csv(fa).procrustes.values, pd.read_csv(f).procrustes.values; NL = len(xb); lo, hi = int(round(NL * EARLY)), NL - 1
    PK = {"A": peaks(xa, 3), "B": peaks(xb, 3)}
    T = {}
    pin = curve(pd.read_csv(fp), "in_patch")
    if len(pin) and pin.max() >= 0.2:
        mx, am = pin.max(), int(pin.idxmax()); T["handoff"] = lambda L, c=pin, mx=mx, am=am: L > am and L in c.index and c[L] < mx / 2
    st = curve(pd.read_csv(fs), "ans_all_nat"); T["onset (α=1)"] = lambda L, c=st: L in c.index and c[L] >= 0.5
    for ev, fn in T.items():
        truth = next((L for L in range(lo, hi + 1) if fn(L)), None)
        if truth is None: continue
        r = {"Full sweep": (truth, hi - lo + 1), "Binary search": bisect_rest(Probe(fn, lo, hi)),
             "Peaks of half A (disjoint)": optimal(Probe(fn, lo, hi), prior(lo, hi, [(q, i) for i, q in enumerate(PK["A"])], FIXED)),
             "Peaks of half B (same prompts)": optimal(Probe(fn, lo, hi), prior(lo, hi, [(q, i) for i, q in enumerate(PK["B"])], FIXED))}
        for s_, (L, n) in r.items(): rows.append(dict(model=SHORT.get(M, M), task=t, event=ev, strategy=s_, tests=n, correct=L == truth))
R = pd.DataFrame(rows); R.to_csv(f"{S}/split_search.csv", index=False)
order = list(dict.fromkeys(R.strategy)); g = R.pivot_table(index="strategy", columns="event", values="tests").loc[order]
cr = R.groupby("strategy").correct.mean().loc[order]; ne = R.drop_duplicates(["model", "task", "event"]).event.value_counts()
out = ["# Search cost with peaks from disjoint prompts (layers tested, mean)\n", "| Strategy | " + " | ".join(f"{e} (n={ne[e]})" for e in g.columns) + " | correct |",
       "|---" * (len(g.columns) + 2) + "|"] + [f"| {s_} | " + " | ".join(f"{v:.1f}" for v in r) + f" | {cr[s_]:.2f} |" for s_, r in g.iterrows()]
open(f"{S}/split_search.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
