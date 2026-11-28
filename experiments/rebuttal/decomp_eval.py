"""Rebuttal (rTuT C1): summarise steer_decomp.py. Usage: python decomp_eval.py <dir with <model>_<task>.csv>   (CPU)
Layers are grouped relative to the highest shape-change peak λ1 of the pair (release data, Eq. 1, App. C rule), late layers only.
Cells: share of prompts switched to the target (alpha = 1 unless rescaled). Writes <dir>/decomp_report.md."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", ".."))
from paths import DATA
from peak_controls import peaks
import glob, os, numpy as np, pandas as pd
D = sys_dir = _s.argv[1]; sig = pd.read_csv(f"{DATA}/events_v2/signals_v2.csv")
SHORT = {"Qwen2.5-7B-Instruct": "Qwen-7B", "Llama-3.1-8B-Instruct": "Llama-8B"}
CONDS = [("full", "full centroid difference"), ("answer", "answer part U"), ("blur", "blurred part B"),
         ("answer_eq", "U, rescaled to ‖full‖"), ("blur_eq", "B, rescaled to ‖full‖"), ("random", "random, ‖full‖")]
rows = []; per = []
for f in sorted(glob.glob(f"{D}/*.csv")):
    M, t = os.path.basename(f)[:-4].rsplit("_", 1); X = pd.read_csv(f)
    s = sig[(sig.model == M) & (sig.task == t)].sort_values("layer"); NL = len(s); lo = int(round(NL * .25)); pk = peaks(s["shape"].values, 3); l1 = pk[0]
    X = X[X.layer >= lo].copy(); X["rel"] = np.where(X.layer < l1, "before λ1", "at / after λ1"); X["pair"] = f"{SHORT.get(M, M)} {t}"
    rows.append(X); per.append(f"| {SHORT.get(M, M)} | {t} | {X.n.iloc[0]} | {' '.join(map(str, pk))} |")
X = pd.concat(rows)
g = X.groupby(["rel", "cond"]).hit.mean().unstack(0)[["before λ1", "at / after λ1"]]
L = ["# Steering with the parts of the centroid difference\n", "Qwen2.5-7B and Llama-3.1-8B, weekday and month; late layers (from round(0.25 L)); all answer positions.\n",
     "| model | task | prompts | top-3 peaks (λ1 first) |", "|---|---|---|---|"] + per
L += ["\n## Success, by layer relative to the highest peak (mean over layers and pairs)\n", "| direction | before λ1 | at / after λ1 |", "|---|---|---|"]
L += [f"| {lab} | {g.loc[c, 'before λ1']:.2f} | {g.loc[c, 'at / after λ1']:.2f} |" for c, lab in CONDS]
L += ["\n## Per pair (at / after λ1)\n", "| pair | " + " | ".join(c for c, _ in CONDS) + " |", "|---" * (len(CONDS) + 1) + "|"]
for p, h in X[X.rel == "at / after λ1"].groupby("pair"):
    m = h.groupby("cond").hit.mean(); L.append(f"| {p} | " + " | ".join(f"{m[c]:.2f}" for c, _ in CONDS) + " |")
L += ["\n## Layers where the full difference switches ≥ half of the prompts\n", "| direction | share of those layers where it also switches ≥ half |", "|---|---|"]
W = X.pivot_table(index=["pair", "layer"], columns="cond", values="hit"); w = W[W["full"] >= .5]
L += [f"| {lab} | {np.mean(w[c] >= .5):.2f} (of {len(w)}) |" for c, lab in CONDS if c != "full"]
open(f"{D}/decomp_report.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
