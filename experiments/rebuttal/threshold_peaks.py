"""Rebuttal: replace "top 3 peaks" by a threshold, so that the number of flagged layers is not fixed in advance. (CPU only)
Usage: python threshold_peaks.py        (reads the release data under DATA; all six models of the paper, full prompt set)
For each model x task, the shape-change curve (events_v2/signals_v2.csv, Eq. 1) is searched from round(0.25 L) as in Appendix C.
A layer is flagged when its shape change exceeds a threshold computed by the same statistic from that curve's searched layers:
  mad k      x > median + k * 1.4826 * MAD   (robust z-score > k; k = 3 is the usual outlier rule)
  fold c     x > c * median
  top3       the paper's rule (local maxima >= 1.5 x median, highest three), for reference
  argmax     the single highest layer
The number of flagged layers therefore varies by model and task. Boundaries (probe / patch / steer rise, handoff, steering onset at
alpha = 1 and 8, transfer rise) are those of Table 1, taken from the release data as in evaluate.py.
Scores: flagged layers per task, the share of the searched layers lying within +-1 of a flagged layer, boundaries on / within +-1 of a
flagged layer, precision (flagged layers within +-1 of a boundary of the same task), and the circular-shift null (10,000 draws).
Writes DATA/rebuttal/threshold_report.md and threshold_flags.csv."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import numpy as np, pandas as pd
EARLY, NPERM = 0.25, 10000
MODELS = ["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.3", "Gemma-2-9B-it", "Qwen2.5-14B-Instruct", "Phi-4-14B"]
SHORT = dict(zip(MODELS, ["Qwen-7B", "Llama-8B", "Mistral-7B", "Gemma-9B", "Qwen-14B", "Phi-4"]))
BTYPES = ["probe", "patch", "steer", "handoff", "onset1", "onset8", "transfer"]
RULES = [("top3", None), ("argmax", None), ("mad", 2.0), ("mad", 3.0), ("mad", 4.0), ("mad", 5.0), ("fold", 2.0), ("fold", 3.0), ("fold", 4.0)]

def rise(v, thr):  # largest one-layer rise within the late layers (from round(0.25 L)), as in Table 1
    d = np.diff(np.asarray(v, float)); d[~np.isfinite(d)] = -np.inf; d[:max(0, int(round(len(v) * EARLY)) - 1)] = -np.inf; i = int(np.argmax(d))
    return i + 1 if d[i] >= thr else None

def transfer_rise(M, t):
    X = pd.read_csv(f"{DATA}/steer_xfer/{M}_{t}.csv").groupby(["X", "Y"]).hit.mean().unstack()
    own = [y for y in X.columns if y in X.index and X.loc[y, y] >= 0.9]
    T = X[own].mean(axis=1).sort_index(); return int(T.diff().idxmax()) if own else None

def load(M):
    sig = pd.read_csv(f"{DATA}/events_v2/signals_v2.csv"); sig = sig[sig.model == M]
    ho = pd.read_csv(f"{DATA}/events_v2/handoff_types_v2.csv"); ho = ho[ho.model == M].set_index("task")
    on = pd.read_csv(f"{DATA}/steer_sweep/report_onset.csv"); on = on[(on.model == M) & (on["mode"] == "nat")]
    X, B = {}, {}
    for t in sorted(sig.task.unique()):
        s = sig[sig.task == t].sort_values("layer"); X[t] = s["shape"].values.astype(float)
        b = dict(probe=rise(s.id.values, 0.15), patch=rise(s.pa.values, 0.10), steer=rise(s.sa.values, 0.10),
                 handoff=int(ho.handoff[t]) if t in ho.index and np.isfinite(ho.handoff[t]) else None)
        for a in (1, 8):
            r = on[(on.task == t) & (on.s == a)]
            b[f"onset{a}"] = int(r.on_best.iloc[0]) if len(r) and np.isfinite(r.on_best.iloc[0]) else None
        try: b["transfer"] = transfer_rise(M, t)
        except FileNotFoundError: b["transfer"] = None
        B[t] = {k: v for k, v in b.items() if v is not None}
    return X, B

def flag(x, rule, par):
    NL = len(x); lo = int(round(NL * EARLY)); y = x[lo:]; L = np.arange(lo, NL); ok = np.isfinite(y)
    med = np.nanmedian(y)
    if rule == "top3":
        pk = [l for l in range(max(1, lo), NL) if np.isfinite(x[l]) and x[l] >= np.nanmax(x[max(1, l - 1):l + 2]) and x[l] >= 1.5 * med]
        return sorted(pk, key=lambda l: -x[l])[:3]
    if rule == "argmax": return [int(L[np.nanargmax(y)])]
    thr = med + par * 1.4826 * np.nanmedian(np.abs(y - med)) if rule == "mad" else par * med
    return [int(l) for l, v, o in zip(L, y, ok) if o and v > thr]

def dist(y, pk): return min(abs(y - q) for q in pk) if pk else 99

def score(F, B, NL, rng):
    T = [(t, y) for t in F for y in B[t].values()]
    d = np.array([dist(y, F[t]) for t, y in T])
    cover = np.mean([len({l + e for l in F[t] for e in (-1, 0, 1)} & set(range(int(round(NL[t] * EARLY)), NL[t])))
                     / (NL[t] - int(round(NL[t] * EARLY))) for t in F])
    prec = [dist(l, list(B[t].values())) <= 1 for t in F for l in F[t]]
    null = np.zeros(NPERM); null0 = np.zeros(NPERM)
    for i in range(NPERM):
        sh = {}
        for t in F:
            lo = int(round(NL[t] * EARLY)); span = NL[t] - lo; z = rng.integers(0, span); sh[t] = [lo + ((q - lo + z) % span) for q in F[t]]
        dd = np.array([dist(y, sh[t]) for t, y in T]); null[i] = np.mean(dd <= 1); null0[i] = np.mean(dd == 0)
    w1 = float(np.mean(d <= 1))
    return dict(n=len(T), k=np.mean([len(F[t]) for t in F]), kmin=min(len(F[t]) for t in F), kmax=max(len(F[t]) for t in F),
                cover=cover, on=float(np.mean(d == 0)), null0=null0.mean(), w1=w1, null=null.mean(), p=(1 + (null >= w1).sum()) / (NPERM + 1),
                prec=float(np.mean(prec)) if prec else np.nan)

if __name__ == "__main__":
    rng = np.random.default_rng(0); rows, flags = [], []
    for M in MODELS:
        X, B = load(M); NL = {t: len(X[t]) for t in X}
        for rule, par in RULES:
            name = rule if par is None else f"{rule} {par:g}"
            F = {t: flag(X[t], rule, par) for t in X}
            rows.append(dict(model=SHORT[M], L=max(NL.values()), rule=name, **score(F, B, NL, rng)))
            flags += [dict(model=M, rule=name, task=t, layers=" ".join(map(str, F[t])), boundaries=" ".join(f"{k}={v}" for k, v in B[t].items())) for t in X]
    S = pd.DataFrame(rows); pd.DataFrame(flags).to_csv(f"{DATA}/rebuttal/threshold_flags.csv", index=False)
    out = ["# Threshold instead of top 3\n", "Per model: flagged layers per task (mean, min–max), share of searched layers within ±1 of a flagged layer, "
           "boundaries on / within ±1 of a flagged layer, circular-shift null, p, precision (flagged layers within ±1 of a boundary).\n"]
    for rule in S.rule.unique():
        out += [f"\n## {rule}\n", "| model | layers | k (min–max) | ±1 covers | n | on (null) | within ±1 | null | p | precision |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in S[S.rule == rule].itertuples():
            out.append(f"| {r.model} | {r.L} | {r.k:.1f} ({r.kmin}–{r.kmax}) | {r.cover:.2f} | {r.n} | {r.on:.2f} ({r.null0:.2f}) | **{r.w1:.2f}** | {r.null:.2f} | {r.p:.4f} | {r.prec:.2f} |")
        a = S[S.rule == rule]; out.append(f"| **all** | | {a.k.mean():.1f} | {a.cover.mean():.2f} | {a.n.sum()} | {np.average(a.on, weights=a.n):.2f} ({np.average(a.null0, weights=a.n):.2f}) | "
                                          f"**{np.average(a.w1, weights=a.n):.2f}** | {np.average(a.null, weights=a.n):.2f} | | {a.prec.mean():.2f} |")
    open(f"{DATA}/rebuttal/threshold_report.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
