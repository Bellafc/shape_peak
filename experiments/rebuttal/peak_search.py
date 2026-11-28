"""Rebuttal: search strategies that use the top peaks and their neighbours, with a verified stop. (CPU only, release data)
Usage: python peak_search.py
Targets and per-layer tests are those of search_cost.py (handoff, steering onset at alpha = 1, transfer T >= 0.5). Every strategy
keeps a bracket (F, P]: F = deepest layer known to fail, P = shallowest layer known to pass; the boundary is found when P = F + 1
(or P is the first late layer), i.e. a tested failing layer sits directly below a tested passing one. Assumes that a test keeps
passing once it passes (true in all pairs, see search_cost.py).
Strategies
  walk from λ1 / walk from logit lens   as in search_cost.py
  binary search                         bisect (lo - 1, hi] without testing hi first
  peak pairs                            for the top-3 peaks in order of height, test p and p - 1 if they are inside the bracket,
                                        then bisect what is left
  prior search (fixed)                  bisect by the weighted median of a prior over layers: each top-3 peak spreads weight
                                        0.5 / 0.25 / 0.25 over p, p - 1, p + 1, peaks weighted 3 : 2 : 1 by rank, plus 10 % uniform
  prior search (learned, LOMO)          the same, with the weight of each (peak rank, offset -3..+3) estimated on the other five
                                        models for this boundary type (leave one model out), plus 10 % uniform
  prior search (learned) + logit lens   adds the offsets around the first layer with A_lin > 0.1 as a fourth anchor
Writes DATA/rebuttal/peak_search.md and peak_search.csv."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", ".."))
from paths import DATA
from threshold_peaks import MODELS, SHORT, EARLY
from peak_controls import peaks
from search_cost import tests, search
import numpy as np, pandas as pd
OFF = range(-3, 4); UNIF = 0.10

class Probe:
    def __init__(self, f, lo, hi): self.f, self.lo, self.hi, self.seen = f, lo, hi, {}; self.F, self.P = lo - 1, hi + 1
    def test(self, L):
        if L not in self.seen:
            self.seen[L] = bool(self.f(L))
            if self.seen[L]: self.P = min(self.P, L)
            else: self.F = max(self.F, L)
        return self.seen[L]
    def done(self): return self.P == self.F + 1 or (self.P == self.lo)

def bisect_rest(pr, w=None):
    """finish inside the bracket; with weights w (dict layer -> mass) use the weighted median, else the midpoint"""
    while not pr.done():
        a, b = pr.F, min(pr.P, pr.hi + 1)
        if b > pr.hi:  # nothing passed yet: the boundary may be anywhere above F
            b = pr.hi + 1
        cand = list(range(a + 1, b))
        if not cand: break
        if w is None: m = (a + 1 + b - 1) // 2 if b <= pr.hi else (a + 1 + pr.hi) // 2
        else:
            # boundary layer B is in (a, b]; testing m splits it into (a, m] (pass) and (m, b] (fail)
            mass = np.array([w.get(L, 0.0) for L in range(a + 1, min(b, pr.hi) + 1)]); tot = mass.sum()
            if tot <= 0: m = (a + 1 + min(b, pr.hi)) // 2
            else:
                cum = np.cumsum(mass) / tot; ms = list(range(a + 1, min(b, pr.hi) + 1))
                m = ms[int(np.argmin(np.abs(cum - 0.5)))]
                if m == min(b, pr.hi) and len(ms) > 1 and b <= pr.hi: m = ms[-2]   # testing P itself gives nothing
        pr.test(m)
    return pr.P, len(pr.seen)

def optimal(pr, w):
    """test layers by the decision tree that minimises the expected number of tests under the prior w (dynamic programming over
    brackets (a, c]; the boundary B lies in (a, c] and is found when c = a + 1)"""
    lo, hi = pr.lo, pr.hi; W = np.array([w.get(L, 0.0) for L in range(lo, hi + 1)]); C = np.concatenate([[0], np.cumsum(W)])
    mass = lambda a, c: C[min(c, hi) - lo + 1] - C[a - lo + 1]          # mass of B in (a, c]
    memo = {}
    def E(a, c):
        if c == a + 1 or mass(a, c) <= 0: return 0.0, None
        if (a, c) in memo: return memo[(a, c)]
        best = (np.inf, None)
        for m in range(a + 1, c):
            v = 1 + (mass(a, m) * E(a, m)[0] + mass(m, c) * E(m, c)[0]) / mass(a, c)
            if v < best[0]: best = (v, m)
        memo[(a, c)] = best; return best
    while not pr.done():
        m = E(pr.F, pr.P)[1]
        if m is None: m = (pr.F + 1 + min(pr.P, hi + 1) - 1) // 2
        pr.test(m)
    return pr.P, len(pr.seen)

def prior(lo, hi, anchors, weights):
    """anchors: list of (layer, key); weights: dict key -> {offset: weight}"""
    w = {L: UNIF / (hi - lo + 1) for L in range(lo, hi + 1)}; extra = {}
    for a, k in anchors:
        for d, v in weights[k].items():
            L = a + d
            if lo <= L <= hi: extra[L] = extra.get(L, 0.0) + v
    s = sum(extra.values())
    for L, v in extra.items(): w[L] += (1 - UNIF) * v / s
    return w

FIXED = {r: {d: wt * v for d, v in {-1: .25, 0: .5, 1: .25}.items()} for r, wt in zip((0, 1, 2), (3, 2, 1))}

if __name__ == "__main__":
    sig = pd.read_csv(f"{DATA}/events_v2/signals_v2.csv"); cases = []
    for M in MODELS:
        S = sig[sig.model == M]
        for t in S.task.unique():
            s_ = S[S.task == t].sort_values("layer"); NL = len(s_); lo, hi = int(round(NL * EARLY)), NL - 1
            pk = peaks(s_["shape"].values, 3); a0 = next((L for L in range(NL) if s_.a_lin.values[L] > 0.1), hi)
            T = tests(M, t, NL); T.pop("_ny", None)
            for ev, f in T.items():
                truth = search(f, lo, hi, "full")[0]
                if truth is not None: cases.append(dict(M=M, t=t, ev=ev, f=f, lo=lo, hi=hi, pk=pk, a0=a0, truth=truth))
    # learned offsets, leave one model out
    def learn(ev, M_out, with_lens):
        cnt = {k: {d: 0.0 for d in OFF} for k in (0, 1, 2, "lens")}
        for c in cases:
            if c["ev"] != ev or c["M"] == M_out: continue
            anc = [(p, r) for r, p in enumerate(c["pk"])] + ([(c["a0"], "lens")] if with_lens else [])
            for a, k in anc:
                d = c["truth"] - a
                if d in cnt[k]: cnt[k][d] += 1
        return cnt
    rows = []
    for c in cases:
        f, lo, hi, pk = c["f"], c["lo"], c["hi"], c["pk"]; res = {}
        res["walk from λ1"] = search(f, lo, hi, "anchor", start=pk[0])
        res["walk from logit lens"] = search(f, lo, hi, "anchor", start=c["a0"])
        res["binary search"] = bisect_rest(Probe(f, lo, hi))
        pr = Probe(f, lo, hi)
        for p in pk:
            for L in (p, p - 1):
                if pr.done(): break
                if pr.F < L < pr.P and lo <= L <= hi: pr.test(L)
        res["peak pairs"] = bisect_rest(pr)
        res["prior search (fixed)"] = bisect_rest(Probe(f, lo, hi), prior(lo, hi, [(p, r) for r, p in enumerate(pk)], FIXED))
        res["prior search (learned)"] = bisect_rest(Probe(f, lo, hi), prior(lo, hi, [(p, r) for r, p in enumerate(pk)], learn(c["ev"], c["M"], False)))
        res["prior search (learned) + logit lens"] = bisect_rest(Probe(f, lo, hi), prior(lo, hi, [(p, r) for r, p in enumerate(pk)] + [(c["a0"], "lens")], learn(c["ev"], c["M"], True)))
        res["optimal tree, prior (fixed)"] = optimal(Probe(f, lo, hi), prior(lo, hi, [(p, r) for r, p in enumerate(pk)], FIXED))
        res["optimal tree, prior (learned)"] = optimal(Probe(f, lo, hi), prior(lo, hi, [(p, r) for r, p in enumerate(pk)], learn(c["ev"], c["M"], False)))
        res["optimal tree, prior (learned) + logit lens"] = optimal(Probe(f, lo, hi), prior(lo, hi, [(p, r) for r, p in enumerate(pk)] + [(c["a0"], "lens")], learn(c["ev"], c["M"], True)))
        res["optimal tree, logit lens only (learned)"] = optimal(Probe(f, lo, hi), prior(lo, hi, [(c["a0"], "lens")], learn(c["ev"], c["M"], True)))
        res["floor: boundary known in advance"] = optimal(Probe(f, lo, hi), {c["truth"]: 1.0})
        for k, (L, n) in res.items(): rows.append(dict(model=SHORT[c["M"]], task=c["t"], event=c["ev"], strategy=k, tests=n, correct=L == c["truth"]))
    R = pd.DataFrame(rows); R.to_csv(f"{DATA}/rebuttal/peak_search.csv", index=False)
    order = list(dict.fromkeys(R.strategy))
    g = R.pivot_table(index="strategy", columns="event", values="tests").loc[order]; cr = R.groupby("strategy").correct.mean().loc[order]
    g["mean"] = g.mean(axis=1); g["correct"] = cr
    cols = list(g.columns)
    out = "\n".join(["# Search with peaks and their neighbours (layers tested, mean)\n", "| strategy | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
                    + [f"| {k} | " + " | ".join(f"{v:.2f}" for v in r) + " |" for k, r in g.iterrows()])
    open(f"{DATA}/rebuttal/peak_search.md", "w").write(out + "\n"); print(out)
