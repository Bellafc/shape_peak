"""Rebuttal (DQWj): do peaks from one half of the prompts locate boundaries measured on the other, disjoint half? (CPU)
Usage: python split_eval.py     (reads DATA/split/{A,B}; written by run_split.sh)
Shape: Procrustes curve (Eq. 1) on half A and on half B (DATA/split/{A,B}/rebuttal/metrics/<model>_<task>.csv).
Boundaries, all on half B only (DATA/split/B):
  handoff     first layer after its maximum at which patching the input token falls below half of that maximum (>= 0.2)
  patching    largest one-layer rise (>= 0.1, late layers) of patching all answer positions
  steering    largest one-layer rise (>= 0.1, late layers) of steering all answer positions (alpha = 1, own-layer direction)
  onset       first late layer at which that steering switches >= half of the prompts
Peaks: top 3 by the rule of Appendix C. Compared: disjoint (peaks of A vs boundaries of B) and same pool (peaks of B vs boundaries
of B). Null: circular shift of the peaks (10,000 draws). Writes DATA/split/split_report.md and split_hits.csv."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", ".."))
from paths import DATA
from peak_controls import peaks, dist
import glob, os, numpy as np, pandas as pd
EARLY, NPERM = 0.25, 10000; S = f"{DATA}/split"
SHORT = {"Qwen2.5-7B-Instruct": "Qwen-7B", "Llama-3.1-8B-Instruct": "Llama-8B", "Mistral-7B-Instruct-v0.3": "Mistral-7B", "Gemma-2-9B-it": "Gemma-9B",
         "Qwen2.5-14B-Instruct": "Qwen-14B", "Phi-4-14B": "Phi-4"}
ARITH = {"weekday", "month", "hour", "letter"}

def curve(df, cond):
    x = df[df.cond == cond].groupby("layer").hit.mean().sort_index(); return x

def late_rise(x, NL, thr):
    lo = int(round(NL * EARLY)); d = x.diff(); d = d[d.index >= lo]
    return int(d.idxmax()) if len(d) and d.max() >= thr else None

def bounds(M, t, NL):
    b = {}
    p = pd.read_csv(f"{S}/B/handoff_patch/{M}_{t}.csv")
    pin = curve(p, "in_patch")
    if len(pin) and pin.max() >= 0.2:
        mx, am = pin.max(), pin.idxmax(); b["handoff"] = next((int(L) for L, v in pin.loc[am:].items() if v < mx / 2), None)
    b["patching"] = late_rise(curve(p, "ans_all_patch"), NL, 0.10)
    f = f"{S}/B/handoff/{M}_{t}.csv" if t in ARITH else f"{S}/B/handoff_retrieval/{M}_{t}.csv"
    if os.path.exists(f):
        st = curve(pd.read_csv(f), "ans_all_nat"); lo = int(round(NL * EARLY))
        b["steering"] = late_rise(st, NL, 0.10); b["onset"] = next((int(L) for L, v in st.items() if L >= lo and v >= 0.5), None)
    return {k: v for k, v in b.items() if v is not None}

if __name__ == "__main__":
    rows, P = [], {}
    for f in sorted(glob.glob(f"{S}/B/rebuttal/metrics/*.csv")):
        stem = os.path.basename(f)[:-4]; M, t = stem.rsplit("_", 1)
        fa = f"{S}/A/rebuttal/metrics/{stem}.csv"
        if not os.path.exists(fa) or not os.path.exists(f"{S}/B/handoff_patch/{M}_{t}.csv"): continue
        xa, xb = pd.read_csv(fa).procrustes.values, pd.read_csv(f).procrustes.values; NL = len(xb)
        pa, pb = peaks(xa, 3), peaks(xb, 3); P[(M, t)] = (pa, pb, NL)
        for k, y in bounds(M, t, NL).items():
            rows.append(dict(model=M, task=t, NL=NL, boundary=k, layer=y, peaksA=" ".join(map(str, pa)), peaksB=" ".join(map(str, pb)),
                             dA=dist(y, pa), dB=dist(y, pb)))
    H = pd.DataFrame(rows); H.to_csv(f"{S}/split_hits.csv", index=False)
    rng = np.random.default_rng(0)
    def null(col):
        keys = list(P); out = np.zeros(NPERM)
        for i in range(NPERM):
            sh = {}
            for key in keys:
                pa, pb, NL = P[key]; pk = pa if col == "dA" else pb; lo = int(round(NL * EARLY)); span = NL - lo; z = rng.integers(0, span)
                sh[key] = [lo + ((q - lo + z) % span) for q in pk]
            out[i] = np.mean([dist(r.layer, sh[(r.model, r.task)]) <= 1 for r in H.itertuples()])
        return out
    nA, nB = null("dA"), null("dB")
    agree = np.mean([len(set(pa) & {q + e for q in pb for e in (-1, 0, 1)}) / max(len(pa), 1) for pa, pb, _ in P.values()])
    L = ["# Peaks and boundaries from disjoint prompt halves\n",
         f"{len(P)} model-task pairs, {len(H)} boundaries measured on half B. Cells: on a peak (within ±1). Null: circular shift.\n",
         "| peaks from | on peak | within ±1 | null | p |", "|---|---|---|---|---|"]
    for lab, col, nl in (("half A (disjoint)", "dA", nA), ("half B (same prompts)", "dB", nB)):
        w1 = np.mean(H[col] <= 1); L.append(f"| {lab} | {np.mean(H[col] == 0):.2f} | **{w1:.2f}** | {nl.mean():.2f} | {(1 + (nl >= w1).sum()) / (NPERM + 1):.4f} |")
    L += [f"\nTop-3 peaks of half A within ±1 of a peak of half B: {agree:.2f}.\n", "## By boundary\n", "| boundary | n | half A ±1 | half B ±1 |", "|---|---|---|---|"]
    L += [f"| {b} | {len(g)} | {np.mean(g.dA <= 1):.2f} | {np.mean(g.dB <= 1):.2f} |" for b, g in H.groupby("boundary")]
    L += ["\n## By model\n", "| model | n | half A ±1 | half B ±1 |", "|---|---|---|---|"]
    L += [f"| {SHORT.get(m, m)} | {len(g)} | {np.mean(g.dA <= 1):.2f} | {np.mean(g.dB <= 1):.2f} |" for m, g in H.groupby("model")]
    open(f"{S}/split_report.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
