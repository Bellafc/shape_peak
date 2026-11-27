"""Rebuttal (W2): stricter peak-to-peak controls for Table 1. (CPU only, release data, all six models, 397 boundaries)
Usage: python peak_controls.py
Each boundary of a model-task pair is compared with the top-k shape-change peaks (k = 1, 3, 5; Appendix C rule) of
  own          the same model and task (Table 1)
  same family  another task of the same family (cyclic arithmetic / retrieval) on the same model
  other family a task of the other family on the same model
  other model  the same task on another model; its peak layers are converted to relative depth (layer / number of layers)
               and back to the target model's layers
Hit rates: exact (distance 0) and within +-1, averaged over the swapped-in sources. p: one source drawn at random per boundary,
20,000 draws; share of draws reaching the own rate. No optimisation and no intervention is used by any control.
Writes DATA/rebuttal/peak_controls.md and peak_controls.csv."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", ".."))
from paths import DATA
from threshold_peaks import load, MODELS, SHORT, EARLY
import numpy as np, pandas as pd
ARITH = {"weekday", "month", "hour", "letter"}; NDRAW = 20000; KS = (1, 3, 5)
TYPES = ["probe", "patch", "steer", "handoff", "onset1", "onset8", "transfer"]

def peaks(x, k):
    NL = len(x); lo = int(round(NL * EARLY)); med = np.nanmedian(x[lo:])
    pk = [l for l in range(max(1, lo), NL) if np.isfinite(x[l]) and x[l] >= np.nanmax(x[max(1, l - 1):l + 2]) and x[l] >= 1.5 * med]
    return sorted(pk, key=lambda l: -x[l])[:k]

def dist(y, pk): return min(abs(y - q) for q in pk) if pk else 99

if __name__ == "__main__":
    D = {M: load(M) for M in MODELS}; NL = {M: len(next(iter(D[M][0].values()))) for M in MODELS}
    rng = np.random.default_rng(0); rows = []
    for k in KS:
        P = {(M, t): peaks(X[t], k) for M, (X, B) in D.items() for t in X}
        items = []  # (type, own distance, {control: [distances to each source]})
        for M, (X, B) in D.items():
            for t in X:
                same = [u for u in X if u != t and (u in ARITH) == (t in ARITH)]; oth = [u for u in X if (u in ARITH) != (t in ARITH)]
                mods = [M2 for M2 in MODELS if M2 != M and (M2, t) in P]
                conv = {M2: sorted({int(round(q / NL[M2] * NL[M])) for q in P[(M2, t)]}) for M2 in mods}
                for b, y in B[t].items():
                    items.append((b, dist(y, P[(M, t)]), {"same family": [dist(y, P[(M, u)]) for u in same],
                                                            "other family": [dist(y, P[(M, u)]) for u in oth],
                                                            "other model": [dist(y, conv[M2]) for M2 in mods]}))
        for bt in ["all"] + TYPES:
            it = [i for i in items if bt == "all" or i[0] == bt]
            own = np.array([i[1] for i in it])
            r = dict(k=k, type=bt, n=len(it), source="own task, own model", exact=np.mean(own == 0), w1=np.mean(own <= 1), p_exact=np.nan, p_w1=np.nan)
            rows.append(r)
            for c in ["same family", "other family", "other model"]:
                src = [i[2][c] for i in it]
                ex = np.mean([np.mean(np.array(s) == 0) for s in src]); w1 = np.mean([np.mean(np.array(s) <= 1) for s in src])
                draw = np.array([[s[rng.integers(len(s))] for s in src] for _ in range(NDRAW // 10)])  # 2,000 draws per cell to keep runtime short
                p0 = (1 + ((draw == 0).mean(1) >= r["exact"]).sum()) / (len(draw) + 1); p1 = (1 + ((draw <= 1).mean(1) >= r["w1"]).sum()) / (len(draw) + 1)
                rows.append(dict(k=k, type=bt, n=len(it), source=c, exact=ex, w1=w1, p_exact=p0, p_w1=p1))
    S = pd.DataFrame(rows); S.to_csv(f"{DATA}/rebuttal/peak_controls.csv", index=False)
    out = ["# Peak-to-peak controls (W2)\n", "Cell: exact hit (within ±1). p: one control source drawn per boundary, 2,000 draws.\n"]
    for bt in ["all"] + TYPES:
        a = S[S.type == bt]; out += [f"\n## {bt} (n = {a.n.iloc[0]})\n", "| peaks from | top-1 | top-3 | top-5 | p (top-1 / 3 / 5, ±1) |", "|---|---|---|---|---|"]
        for src in a.source.unique():
            b = a[a.source == src].set_index("k")
            ps = " / ".join("–" if np.isnan(b.p_w1[k]) else f"{b.p_w1[k]:.4f}" for k in KS)
            out.append(f"| {src} | " + " | ".join(f"{b.exact[k]:.2f} ({b.w1[k]:.2f})" for k in KS) + f" | {ps} |")
    open(f"{DATA}/rebuttal/peak_controls.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
