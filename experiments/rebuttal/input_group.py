"""Rebuttal (rTuT W1): does grouping the last-token states by the input (start item) instead of the correct answer give
shape-change peaks that locate the boundaries? (CPU)
Usage: python input_group.py <dir with <name>_<task>.pt from run_inputgroup.sh>
For each model x task: Eq. 1 (alt_metrics.grid, centred, unit norm, aligned) between adjacent layers, with centroids grouped by
the correct answer (as in the paper) or by the start item; top-k peaks by the App. C rule. Boundaries are those of Table 1 from
the release data (threshold_peaks.load). Cells: exactly on a peak (within ±1); null: circular shift of the peaks.
Writes <dir>/input_group.md."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", ".."))
from threshold_peaks import load, SHORT, EARLY
from peak_controls import peaks, dist
from alt_metrics import grid
import glob, os, numpy as np, torch
D = _s.argv[1]; rng = np.random.default_rng(0)

def curve(H, y):
    cls = np.unique(y); out = [np.nan]; prev = None
    for L in range(H.shape[1]):
        X = H[:, L]; C = torch.stack([X[torch.as_tensor(y == c)].mean(0) for c in cls])
        if prev is not None: out.append(grid(prev, C, 1, 1, 1))
        prev = C
    return np.array(out)

res = {g: {k: [] for k in (1, 3)} for g in ("answer", "input")}; shift = {g: {k: [] for k in (1, 3)} for g in ("answer", "input")}; per = []
for f in sorted(glob.glob(f"{D}/*.pt")):
    M, t = os.path.basename(f)[:-3].rsplit("_", 1); a = torch.load(f, weights_only=False)
    H = a["H"].float(); ans, src = np.asarray(a["ans"]), np.asarray(a["src"])
    X, B = load(M); bnd = list(B[t].values()); NL = H.shape[1]; lo = int(round(NL * EARLY))
    C = {"answer": curve(H, ans), "input": curve(H, src)}
    row = [SHORT.get(M, M), t, len(bnd)]; pair = []
    for g, x in C.items():
        cells = []
        for k in (1, 3):
            pk = peaks(x, k); d = [dist(y, pk) for y in bnd]; res[g][k] += d
            sh = []
            for _ in range(2000):
                z = rng.integers(0, NL - lo); s = [lo + (q - lo + z) % (NL - lo) for q in pk]; sh.append([dist(y, s) for y in bnd])
            sh = np.array(sh); shift[g][k].append(sh); d = np.array(d)
            cells += [f"{np.mean(d == 0):.2f} ({np.mean(d <= 1):.2f})", f"{np.mean(sh == 0):.2f} ({np.mean(sh <= 1):.2f})"]
        row.append(" ".join(map(str, peaks(x, 3)))); pair.append([SHORT.get(M, M), t, len(bnd), g] + cells)
    per.append(row); pairs = globals().setdefault("pairs", []); pairs += pair
L = ["# Grouping by the input instead of the answer\n", "| model | task | boundaries | top-3 peaks, answer-grouped | top-3 peaks, input-grouped |", "|---|---|---|---|---|"]
L += [f"| {' | '.join(map(str, r))} |" for r in per]
L += ["\nExactly on a peak (within ±1); shift = circularly shifted peaks of the same grouping.\n", "| grouping | Top-1 | Shift top-1 | Top-3 | Shift top-3 |", "|---|---|---|---|---|"]
for g in ("answer", "input"):
    cells = []
    for k in (1, 3):
        d = np.array(res[g][k]); s = np.concatenate(shift[g][k], axis=1)
        cells += [f"{np.mean(d == 0):.2f} ({np.mean(d <= 1):.2f})", f"{np.mean(s == 0):.2f} ({np.mean(s <= 1):.2f})"]
    L.append(f"| {g} | " + " | ".join(cells) + " |")
L += ["\n## Per model and task\n", "| model | task | n | grouping | Top-1 | Shift top-1 | Top-3 | Shift top-3 |", "|---|---|---|---|---|---|---|---|"]
L += [f"| {' | '.join(map(str, r))} |" for r in pairs]
open(f"{D}/input_group.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
