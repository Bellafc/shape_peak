"""Rebuttal pilots: do shape-change peaks locate the boundaries in multi-step arithmetic and multiple-choice kinship reasoning?
Usage: python pilot_eval.py     (reads every DATA/rebuttal/pilot/<model>_<pilot>_<cond>.csv)
Boundaries per condition, defined as in the paper (Appendix D / Table 1):
  probe     largest one-layer rise of the identity probe (counted if >= 0.15)
  patch     largest one-layer rise of patching all answer positions (>= 0.10)
  steer     largest one-layer rise of alpha = 1 steering at the answer positions (>= 0.10)
  handoff   first layer after its maximum at which patching the user message falls below half of that maximum
  onset     first layer at which steering succeeds on at least half of the prompts
Peaks: paper rule (from round(0.25 L), local maximum, >= 1.5 x median of the searched layers, top 3). Null: circular shift of each
condition's peaks within the late layers, 10,000 draws, pooled over the conditions of one pilot and model.
Writes DATA/rebuttal/pilot_report.md and pilot_<model>_<pilot>.png."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import os, glob, numpy as np, pandas as pd
R = f"{DATA}/rebuttal"; EARLY, PEAKR, NPERM = 0.25, 1.5, 10000
BT = ["probe", "patch", "steer", "handoff", "onset"]

def peaks(x):
    NL = len(x); lo = int(round(NL * EARLY)); med = np.nanmedian(x[lo:])
    pk = [L for L in range(max(1, lo), NL) if np.isfinite(x[L]) and x[L] >= np.nanmax(x[max(1, L - 1):L + 2]) and x[L] >= PEAKR * med]
    return sorted(pk, key=lambda L: -x[L])[:3]

def rise(v, thr):  # largest one-layer rise within the late layers (from round(0.25 L)), as in Table 1
    d = np.diff(np.asarray(v, float)); d[~np.isfinite(d)] = -np.inf; d[:max(0, int(round(len(v) * EARLY)) - 1)] = -np.inf; i = int(np.argmax(d))
    return i + 1 if d[i] >= thr else None

def bounds(d):
    b = dict(probe=rise(d.probe, 0.15), patch=rise(d.patch_ans, 0.10), steer=rise(d.steer, 0.10))
    pi = d.patch_in.values; m = int(np.nanargmax(pi)) if np.isfinite(pi).any() else None
    b["handoff"] = next((L for L in range(m + 1, len(pi)) if pi[L] < 0.5 * pi[m]), None) if m is not None and pi[m] >= 0.3 else None
    b["onset"] = next((int(L) for L, s in zip(d.layer, d.steer) if s >= 0.5), None)
    return {k: v for k, v in b.items() if v is not None}

def dist(y, pk): return min(abs(y - q) for q in pk) if pk else 99

def test(PK, B, NL, rng):
    T = [(c, y) for c in PK for y in B[c].values()]
    if not T: return dict(n=0)
    obs = np.mean([dist(y, PK[c]) <= 1 for c, y in T]); on = np.mean([dist(y, PK[c]) == 0 for c, y in T])
    null = np.zeros(NPERM)
    for i in range(NPERM):
        sh = {}
        for c in PK:
            lo = int(round(NL * EARLY)); span = NL - lo; z = rng.integers(0, span); sh[c] = [lo + ((q - lo + z) % span) for q in PK[c]]
        null[i] = np.mean([dist(y, sh[c]) <= 1 for c, y in T])
    return dict(n=len(T), on=on, within1=obs, null=null.mean(), p=(1 + (null >= obs).sum()) / (NPERM + 1))

if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    files = sorted(glob.glob(f"{R}/pilot/*.csv")); rng = np.random.default_rng(0)
    G = {}
    for f in files:
        stem = os.path.basename(f)[:-4]; m = None
        for pil in ("arith", "kin"):
            if f"_{pil}_" in stem: m, c = stem.split(f"_{pil}_"); p = pil
        if m is None: continue
        G.setdefault((m, p), {})[c] = pd.read_csv(f)
    L = ["# Pilots beyond single-word lookups\n",
         "Boundaries within ±1 layer of a top-3 shape-change peak (pooled over conditions); null = circular shift of peaks (10,000 draws).\n"]
    summ = ["| model | pilot | n | on peak | within ±1 | null | p |", "|---|---|---|---|---|---|---|"]
    detail = []
    for (m, p), conds in sorted(G.items()):
        order = sorted(conds, key=lambda c: (("long" in c), c))
        NL = len(next(iter(conds.values())))
        PK = {c: peaks(conds[c]["shape"].values) for c in order}; B = {c: bounds(conds[c]) for c in order}
        t = test(PK, B, NL, rng)
        if t["n"]: summ.append(f"| {m} | {p} | {t['n']} | {t['on']:.2f} | **{t['within1']:.2f}** | {t['null']:.2f} | {t['p']:.4f} |")
        detail += [f"\n## {m} · {p}\n", "| condition | N | acc | pairs | peaks (shape) | " + (" peaks (by relation word) |" if p == "kin" else "") + " ".join(f"{k} |" for k in BT),
                   "|---" * (5 + (p == "kin") + len(BT)) + "|"]
        for c in order:
            d = conds[c]; extra = f" {','.join(map(str, peaks(d['shape_content'].values)))} |" if p == "kin" else ""
            detail.append(f"| {c} | {int(d.N.iloc[0])} | {d.acc.iloc[0]:.2f} | {int(d.n_pairs.iloc[0])} | {','.join(map(str, PK[c]))} |{extra} "
                          + " ".join(f"{B[c].get(k, '–')} |" for k in BT))
        if p == "kin":
            PKc = {c: peaks(conds[c]["shape_content"].values) for c in order}; tc = test(PKc, B, NL, rng)
            if tc["n"]: detail.append(f"\nPeaks of the shape grouped by relation word instead of letter: within ±1 {tc['within1']:.2f} (null {tc['null']:.2f}, p {tc['p']:.4f}).")
        # figure
        fig, ax = plt.subplots(1, len(order), figsize=(3.6 * len(order), 3.2), sharey=True); ax = np.atleast_1d(ax)
        for a, c in zip(ax, order):
            d = conds[c]; x = d["shape"].values; lo = int(round(NL * EARLY))
            a.bar(d.layer, x / np.nanmax(x[lo:]), color="0.8", label="shape change (scaled)")
            for k, col in (("probe", "tab:purple"), ("patch_in", "tab:blue"), ("patch_ans", "tab:red"), ("steer", "tab:green")):
                a.plot(d.layer, d[k], color=col, lw=1.5, label=k)
            for q in PK[c]: a.axvline(q, color="tab:orange", lw=1, ls="--")
            a.set_title(f"{c} (acc {d.acc.iloc[0]:.2f})", fontsize=9); a.set_ylim(0, 1.05); a.set_xlabel("layer")
        ax[0].legend(fontsize=6, loc="upper left"); fig.suptitle(f"{m} · {p}: orange = top-3 shape peaks", fontsize=10); fig.tight_layout()
        fig.savefig(f"{R}/pilot_{m}_{p}.png", dpi=130); plt.close(fig)
    L += summ + detail
    open(f"{R}/pilot_report.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
