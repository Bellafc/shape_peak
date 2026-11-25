"""Rebuttal: do the alternative metrics' peaks locate the same boundaries as the Procrustes peaks? (CPU only)
Usage: python evaluate.py [model name]   (default Qwen2.5-7B-Instruct)
Reads DATA/rebuttal/metrics/<model>_<task>.csv and the boundary layers already in the release data
(they do not depend on the shape metric, so no intervention is re-run):
  probe / patch / steer   largest one-layer rise of the identity probe (>= 0.15) / answer-position patching / steering (>= 0.10)
                          (events_v2/signals_v2.csv: id, pa, sa)
  handoff                 events_v2/handoff_types_v2.csv
  onset1 / onset8         steering onset at alpha = 1 / 8 (steer_sweep/report_onset.csv, mode nat, on_best)
  transfer                largest one-layer rise of T(X) (steer_xfer/<model>_<task>.csv)
Peaks: the paper's rule (Appendix C): search from round(0.25 L), local maximum, >= 1.5 x median of the searched layers; top 3 by height.
Null: the paper's circular shift; each task's three peaks keep their spacing and are shifted together within the late layers
(one shift per task, shared by all boundary types of that task), 10,000 draws.
Writes DATA/rebuttal/report_<model>.md, peaks_<model>.csv, hits_<model>.csv, curves_<model>.png and prints the report."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import sys, os, numpy as np, pandas as pd
M = sys.argv[1] if len(sys.argv) > 1 else "Qwen2.5-7B-Instruct"
TASKS = "weekday month hour letter city continent capital2hop color ctxcolor emotion".split()
R = f"{DATA}/rebuttal"; NPERM = 10000; EARLY, PEAKR = 0.25, 1.5
GROUPS = [("Reference", ["procrustes"]),
          ("A. magnitude only", ["mag_scale", "mag_norm", "mag_sep", "mag_classnorm"]),
          ("C6. normalisation ablation (c=centre, s=unit norm, a=rotation)", [f"grid_c{c}_s{s}_a{a}" for c in (1, 0) for s in (1, 0) for a in (1, 0)]),
          ("C7. raw-coordinate direction change", ["dir_centroid", "dir_pair"])]
METRICS = [m for _, g in GROUPS for m in g]
BTYPES = ["probe", "patch", "steer", "handoff", "onset1", "onset8", "transfer"]

def peaks(x):
    NL = len(x); lo = int(round(NL * EARLY)); med = np.nanmedian(x[lo:])
    pk = [L for L in range(max(1, lo), NL) if np.isfinite(x[L]) and x[L] >= np.nanmax(x[max(1, L - 1):L + 2]) and x[L] >= PEAKR * med]
    return sorted(pk, key=lambda L: -x[L])[:3]

def rise(v, thr):
    d = np.diff(np.asarray(v, float)); d[~np.isfinite(d)] = -np.inf; i = int(np.argmax(d))
    return i + 1 if d[i] >= thr else None

def transfer_rise(t):
    X = pd.read_csv(f"{DATA}/steer_xfer/{M}_{t}.csv").groupby(["X", "Y"]).hit.mean().unstack()
    own = [y for y in X.columns if y in X.index and X.loc[y, y] >= 0.9]
    T = X[own].mean(axis=1).sort_index(); return int(T.diff().idxmax()) if own else None

def boundaries():
    sig = pd.read_csv(f"{DATA}/events_v2/signals_v2.csv"); sig = sig[sig.model == M]
    ho = pd.read_csv(f"{DATA}/events_v2/handoff_types_v2.csv"); ho = ho[ho.model == M].set_index("task")
    on = pd.read_csv(f"{DATA}/steer_sweep/report_onset.csv"); on = on[(on.model == M) & (on["mode"] == "nat")]
    B, paper = {}, {}
    for t in TASKS:
        s = sig[sig.task == t].sort_values("layer")
        b = dict(probe=rise(s.id.values, 0.15), patch=rise(s.pa.values, 0.10), steer=rise(s.sa.values, 0.10),
                 handoff=int(ho.handoff[t]) if t in ho.index and np.isfinite(ho.handoff[t]) else None)
        for a in (1, 8):
            r = on[(on.task == t) & (on.s == a)]
            b[f"onset{a}"] = int(r.on_best.iloc[0]) if len(r) and np.isfinite(r.on_best.iloc[0]) else None
        try: b["transfer"] = transfer_rise(t)
        except FileNotFoundError: b["transfer"] = None
        B[t] = {k: v for k, v in b.items() if v is not None}
        paper[t] = peaks(s["shape"].values)
    return B, paper

def dist(y, pk): return min(abs(y - q) for q in pk) if pk else 99

def score(PK, NL, B, types=BTYPES, rng=None):
    T = [(t, y) for t in TASKS if t in PK for k, y in B[t].items() if k in types]
    obs = np.array([dist(y, PK[t]) for t, y in T])
    out = dict(n=len(T), on_peak=float(np.mean(obs == 0)), within1=float(np.mean(obs <= 1)), mean_dist=float(np.mean(np.minimum(obs, NL[TASKS[0]]))))
    if rng is not None:
        tasks = [t for t in TASKS if t in PK]; null = np.zeros(NPERM)
        for i in range(NPERM):
            sh = {}
            for t in tasks:
                n = NL[t]; lo = int(round(n * EARLY)); span = n - lo; z = rng.integers(0, span)
                sh[t] = [lo + ((q - lo + z) % span) for q in PK[t]]
            null[i] = np.mean([dist(y, sh[t]) <= 1 for t, y in T])
        out.update(null_within1=float(null.mean()), p=float((1 + (null >= out["within1"]).sum()) / (NPERM + 1)))
    return out

if __name__ == "__main__":
    B, paper = boundaries()
    D = {t: pd.read_csv(f"{R}/metrics/{M}_{t}.csv") for t in TASKS if os.path.exists(f"{R}/metrics/{M}_{t}.csv")}
    missing = [t for t in TASKS if t not in D]
    TASKS[:] = [t for t in TASKS if t in D]
    NL = {t: len(D[t]) for t in TASKS}
    PK = {m: {t: peaks(D[t][m].values) for t in TASKS} for m in METRICS}
    rng = np.random.default_rng(0); rows = []
    for m in METRICS:
        r = dict(metric=m, **score(PK[m], NL, B, rng=rng))
        ref = PK["procrustes"]
        r["top3_match_procrustes"] = float(np.mean([dist(q, ref[t]) <= 1 for t in TASKS for q in PK[m][t]])) if any(PK[m][t] for t in TASKS) else np.nan
        r["highest_match_procrustes"] = float(np.mean([bool(PK[m][t]) and abs(PK[m][t][0] - ref[t][0]) <= 1 for t in TASKS]))
        rows.append(r)
    S = pd.DataFrame(rows); S.to_csv(f"{R}/hits_{M}.csv", index=False)
    pd.DataFrame([dict(metric=m, task=t, peaks=" ".join(map(str, PK[m][t]))) for m in METRICS for t in TASKS]).to_csv(f"{R}/peaks_{M}.csv", index=False)
    # per boundary type
    BT = pd.DataFrame([dict(metric=m, **{k: score(PK[m], NL, B, types=[k])["within1"] for k in BTYPES}) for m in METRICS])
    # report
    acc = {t: D[t].acc.iloc[0] for t in TASKS}; Ns = {t: int(D[t].N.iloc[0]) for t in TASKS}
    L = [f"# Alternative shape metrics: {M}\n",
         f"Tasks: {len(TASKS)} ({', '.join(f'{t} N={Ns[t]} acc={acc[t]:.2f}' for t in TASKS)})" + (f"; missing: {missing}" if missing else ""),
         f"Boundaries: {sum(len(B[t]) for t in TASKS)} (7 types x tasks; from the release data, independent of the metric). "
         f"Peaks: paper rule, top 3 from layer {int(round(NL[TASKS[0]] * EARLY))}. Null: circular shift of each task's peaks, {NPERM} draws.\n",
         "## Sanity check: Procrustes peaks on the half prompt set vs. the paper (all prompts)\n",
         "| task | paper | this run |", "|---|---|---|"]
    L += [f"| {t} | {' '.join(map(str, paper[t]))} | {' '.join(map(str, PK['procrustes'][t]))} |" for t in TASKS]
    L += ["", "## Boundaries per task\n", "| task | " + " | ".join(BTYPES) + " |", "|---" * (len(BTYPES) + 1) + "|"]
    L += [f"| {t} | " + " | ".join(str(B[t].get(k, "–")) for k in BTYPES) + " |" for t in TASKS]
    L += ["", "## Main table: boundaries within one layer of a top-3 peak\n",
          "| metric | n | on peak | within ±1 | null | p | mean dist | top-3 peaks ±1 of a Procrustes peak | highest peak ±1 of Procrustes highest |",
          "|---|---|---|---|---|---|---|---|---|"]
    for g, ms in GROUPS:
        L.append(f"| **{g}** | | | | | | | | |")
        for m in ms:
            r = S[S.metric == m].iloc[0]
            L.append(f"| {m} | {r.n} | {r.on_peak:.2f} | **{r.within1:.2f}** | {r.null_within1:.2f} | {r.p:.4f} | {r.mean_dist:.2f} | {r.top3_match_procrustes:.2f} | {r.highest_match_procrustes:.2f} |")
    L += ["", "## Within ±1 by boundary type\n", "| metric | " + " | ".join(BTYPES) + " |", "|---" * (len(BTYPES) + 1) + "|"]
    L += [f"| {r.metric} | " + " | ".join(f"{getattr(r, k):.2f}" for k in BTYPES) + " |" for r in BT.itertuples()]
    L += ["", "## Top-3 peaks per task (highest first)\n", "| metric | " + " | ".join(TASKS) + " |", "|---" * (len(TASKS) + 1) + "|"]
    L += [f"| {m} | " + " | ".join(",".join(map(str, PK[m][t])) or "–" for t in TASKS) + " |" for m in METRICS]
    open(f"{R}/report_{M}.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
    # figure: curves scaled to their maximum over the late layers, boundaries as ticks
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    SHOW = [("procrustes", "k", "-"), ("grid_c0_s0_a0", "tab:red", "-"), ("dir_pair", "tab:blue", "-"), ("mag_scale", "tab:green", "--"),
            ("mag_norm", "tab:orange", "--"), ("mag_classnorm", "tab:purple", ":")]
    fig, ax = plt.subplots(2, 5, figsize=(20, 7), sharey=True); ax = ax.ravel()
    for i, t in enumerate(TASKS):
        d = D[t]; lo = int(round(NL[t] * EARLY))
        for m, c, ls in SHOW:
            x = d[m].values; ax[i].plot(d.layer, x / np.nanmax(x[lo:]), color=c, ls=ls, lw=1.4, label=m)
        for k, y in B[t].items(): ax[i].plot([y], [1.08], "v", color="gray", ms=5)
        ax[i].axvspan(0, lo - 0.5, color="0.93"); ax[i].set_title(t); ax[i].set_ylim(0, 1.15); ax[i].set_xlabel("layer")
    ax[0].legend(fontsize=7, loc="upper left"); fig.suptitle(f"{M}: adjacent-layer change (scaled to max over late layers); triangles = boundary layers")
    fig.tight_layout(); fig.savefig(f"{R}/curves_{M}.png", dpi=130); print("figure", f"{R}/curves_{M}.png")
