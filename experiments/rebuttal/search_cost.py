"""Rebuttal: how many interventions does it take to find each causal boundary, with and without the peaks? (CPU only, release data)
Usage: python search_cost.py
The three boundaries of the paper, each as "the first late layer at which a per-layer test passes":
  handoff      patching the input token falls below half of its maximum (Section 4; curves in handoff_patch/ for the 14B models
               and handoff_from_pdf.csv for the other four, the latter digitised from the paper's figures)
  onset        steering at the answer positions switches >= half of the prompts, alpha = 1 (Section 5; steer_sweep/, best of
               the three directions, as in Appendix G)
  transfer     a vector built at layer X transfers with T(X) >= 0.5 (Section 6; steer_xfer/; T as in Section 6, over the layers
               where the layer's own vector works). One test = evaluating T at one build layer, i.e. |Y| interventions.
Search strategies (one test = one layer): full sweep of the late layers; binary search (assumes the test keeps passing once it
passes); start at the first layer with A_lin > 0.1 (logit lens, Billa 2026) or at the highest peak, then walk down while the test
passes or up until it passes. A strategy is correct if it returns the layer a full sweep returns.
Sweep data cover only part of the late layers (steer_sweep stops 3 layers after the highest peak); untested layers take the nearest
tested value. The alpha = 1 onset lies inside the tested range in every pair, so its search costs are unaffected.
Writes DATA/rebuttal/search_cost.md and search_cost.csv."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", ".."))
from paths import DATA
from threshold_peaks import MODELS, SHORT, EARLY
from peak_controls import peaks
import numpy as np, pandas as pd

def series(x):
    x = x.dropna().sort_index(); return x[~x.index.duplicated()]

def nearest(c, L):
    if L in c.index: return c[L]
    return c.iloc[0] if L < c.index.min() else c.iloc[-1] if L > c.index.max() else c.iloc[np.argmin(abs(c.index.values - L))]

def tests(M, t, NL):
    out = {}
    hp = pd.read_csv(f"{DATA}/handoff_from_pdf.csv"); h = hp[(hp.model == M) & (hp.task == t) & (hp.cond == "in_patch")]
    if t != "emotion" and len(h):
        c = series(h.set_index("layer").hit); mx, am = c.max(), c.idxmax()
        if mx >= 0.2: out["1 handoff"] = lambda L, c=c, mx=mx, am=am: L > am and nearest(c, L) < mx / 2
    try:
        d = pd.read_csv(f"{DATA}/steer_sweep/{M}_{t}.csv"); d = d[(d["mode"] == "nat") & (d.s == 1.0)]
        c = series(d.groupby(["Y", "X"]).hit.mean().groupby("Y").max()); out["2 onset (α=1)"] = lambda L, c=c: L >= c.index.min() and nearest(c, L) >= 0.5
    except FileNotFoundError: pass
    try:
        X = pd.read_csv(f"{DATA}/steer_xfer/{M}_{t}.csv").groupby(["X", "Y"]).hit.mean().unstack()
        own = [y for y in X.columns if y in X.index and X.loc[y, y] >= 0.9]
        if own:
            c = series(X[own].mean(axis=1)); out["3 transfer (T ≥ .5)"] = lambda L, c=c: L >= c.index.min() and nearest(c, L) >= 0.5
            out["_ny"] = len(own)
    except FileNotFoundError: pass
    return out

def search(f, lo, hi, mode, start=None):
    seen = {}
    def ok(L):
        if L not in seen: seen[L] = bool(f(L))
        return seen[L]
    if mode == "full":
        for L in range(lo, hi + 1): ok(L)
        return next((L for L in range(lo, hi + 1) if seen[L]), None), len(seen)
    if mode == "binary":
        a, b = lo, hi
        if not ok(b): return None, len(seen)
        while a < b:
            m = (a + b) // 2
            if ok(m): b = m
            else: a = m + 1
        return a, len(seen)
    L = min(max(start, lo), hi)                          # anchored walk
    if ok(L):
        while L - 1 >= lo and ok(L - 1): L -= 1
        return L, len(seen)
    while L + 1 <= hi:
        L += 1
        if ok(L): return L, len(seen)
    return None, len(seen)

STRATS = [("full sweep", "full"), ("binary search", "binary"), ("start at logit lens", "lens"), ("start at highest peak", "peak")]
if __name__ == "__main__":
    sig = pd.read_csv(f"{DATA}/events_v2/signals_v2.csv"); rows = []
    for M in MODELS:
        S = sig[sig.model == M]
        for t in S.task.unique():
            s_ = S[S.task == t].sort_values("layer"); NL = len(s_); lo, hi = int(round(NL * EARLY)), NL - 1
            l1 = peaks(s_["shape"].values, 3)[0]; a0 = next((L for L in range(NL) if s_.a_lin.values[L] > 0.1), hi)
            T = tests(M, t, NL); ny = T.pop("_ny", np.nan)
            for ev, f in T.items():
                truth, _ = search(f, lo, hi, "full")
                if truth is None: continue
                for name, mode in STRATS:
                    L, n = search(f, lo, hi, "anchor" if mode in ("lens", "peak") else mode, start={"lens": a0, "peak": l1}.get(mode))
                    rows.append(dict(model=SHORT[M], task=t, event=ev, strategy=name, span=hi - lo + 1, tests=n, correct=L == truth,
                                     interventions=n * (ny if ev.startswith("3") else 1)))
    R = pd.DataFrame(rows); R.to_csv(f"{DATA}/rebuttal/search_cost.csv", index=False)
    order = [s for s, _ in STRATS]; out = ["# Search cost per causal boundary\n", "Layers tested (mean) and share of pairs where the strategy returns the full-sweep layer.\n"]
    for ev in sorted(R.event.unique()):
        a = R[R.event == ev]; g = a.groupby("strategy").agg(tests=("tests", "mean"), correct=("correct", "mean"), interventions=("interventions", "mean")).loc[order]
        out += [f"\n## {ev} ({a.task.size // len(order)} pairs; late layers {a.span.mean():.1f})\n", "| strategy | layers tested | correct |" + (" interventions |" if ev.startswith("3") else ""),
                "|---|---|---|" + ("---|" if ev.startswith("3") else "")]
        out += [f"| {s} | {r.tests:.1f} | {r.correct:.2f} |" + (f" {r.interventions:.0f} |" if ev.startswith("3") else "") for s, r in g.iterrows()]
        p = a.pivot_table(index=["model", "task"], columns="strategy", values="tests")
        out.append(f"\nPeak start cheaper than logit-lens start in {np.mean(p['start at highest peak'] < p['start at logit lens']):.2f} of pairs, dearer in {np.mean(p['start at highest peak'] > p['start at logit lens']):.2f}.")
    open(f"{DATA}/rebuttal/search_cost.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
