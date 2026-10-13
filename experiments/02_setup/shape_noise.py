"""形變的噪聲基線（用存下的答案位置激活 results/dims/<名稱>_<concept>_acts.pt）。
(1) bootstrap：每類別內對題目有放回重抽樣，重算逐層 change_full（B 次）→ 每層 95% 區間、第 1 峰位置的穩定度。
(2) 測量地板：同一層兩份獨立 bootstrap 樣本的 centroid 之間的 Procrustes 距離（沒有任何層間變化時，量到的「變化」）。
峰是否真實：峰值 vs 同層地板的 95 百分位；以及原始第 1 峰在 bootstrap 中仍是第 1 峰（±0 / ±1 層）的比例。
輸出 results/events/noise/<名稱>_<concept>.csv（逐層）與 results/events/noise_summary.csv（每組）。"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import glob, os, sys, numpy as np, pandas as pd, torch
R = f"{DATA}"; OUT = f"{R}/events/noise"; os.makedirs(OUT, exist_ok=True)
B = int(os.environ.get("B", 100)); EARLY = 0.25; torch.set_num_threads(8)
def disp(A, Bm):
    A = A - A.mean(0); Bm = Bm - Bm.mean(0); A = A / A.norm(); Bm = Bm / Bm.norm()
    Q, _ = torch.linalg.qr(torch.cat([A, Bm]).T); A, Bm = A @ Q, Bm @ Q
    return float(1 - torch.linalg.svdvals(A.T @ Bm).sum() ** 2)
def curve(H, idx_by_class):
    C = torch.stack([H[ix].mean(0) for ix in idx_by_class], 1)  # NL × n × D
    return C, np.array([np.nan] + [disp(C[L], C[L - 1]) for L in range(1, C.shape[0])])
rows = []
files = sorted(glob.glob(f"{R}/acts/*_acts.pt")) if len(sys.argv) < 2 else sys.argv[1:]
for f in files:
    name = os.path.basename(f)[:-8]; m, t = name.rsplit("_", 1)
    if os.path.exists(f"{OUT}/{name}.csv") and not os.environ.get("REDO"): continue
    a = torch.load(f, weights_only=False); H = a["H"].float(); y = np.asarray(a["ans"]); cls = np.unique(y)
    NL = H.shape[1]; lo = int(round(NL * EARLY)); rng = np.random.default_rng(0)
    base_idx = [np.where(y == c)[0] for c in cls]
    _, c0 = curve(H, base_idx)
    late = np.arange(lo, NL); top0 = late[np.nanargmax(c0[lo:])]
    boot, floor, tops = [], [], []
    for b in range(B):
        i1 = [rng.choice(ix, len(ix)) for ix in base_idx]; i2 = [rng.choice(ix, len(ix)) for ix in base_idx]
        C1, cb = curve(H, i1); boot.append(cb); tops.append(late[np.nanargmax(cb[lo:])])
        C2 = torch.stack([H[ix].mean(0) for ix in i2], 1)
        floor.append([disp(C1[L], C2[L]) for L in range(NL)])
    perm = []
    for b in range(B // 2):  # 標籤打亂：同樣的類別大小，隨機分組
        yp = rng.permutation(y); perm.append(curve(H, [np.where(yp == c)[0] for c in cls])[1])
    boot, floor, tops, perm = np.array(boot), np.array(floor), np.array(tops), np.array(perm)
    d = pd.DataFrame(dict(layer=range(NL), change=c0, boot_lo=np.nanpercentile(boot, 2.5, 0), boot_hi=np.nanpercentile(boot, 97.5, 0),
                          floor_med=np.median(floor, 0), floor_p95=np.percentile(floor, 95, 0),
                          perm_med=np.nanmedian(perm, 0), perm_p95=np.nanpercentile(perm, 95, 0)))
    d.to_csv(f"{OUT}/{name}.csv", index=False)
    # 晚期局部峰（≥ 1.5× 晚期中位數），按高度排序
    med = np.nanmedian(c0[lo:]); pk = sorted([L for L in late if c0[L] >= np.nanmax(c0[max(1, L - 1):L + 2]) and c0[L] >= 1.5 * med], key=lambda L: -c0[L])
    still = lambda L: np.mean([bb[L] >= np.nanmax(bb[max(1, L - 1):L + 2]) and bb[L] >= 1.5 * np.nanmedian(bb[lo:]) for bb in boot])
    r = dict(model=m, task=t, NL=NL, n=len(y), n_cls=len(cls), top1=int(top0), top1_change=round(c0[top0], 4),
             top1_perm_p95=round(d.perm_p95[top0], 4), top1_ratio_perm=round(c0[top0] / d.perm_p95[top0], 1),
             top1_floor_p95=round(d.floor_p95[top0], 4),
             top1_stable_exact=round(float((tops == top0).mean()), 2), top1_stable_pm1=round(float((np.abs(tops - top0) <= 1).mean()), 2),
             n_peaks=len(pk), n_peaks_above_perm=int(sum(c0[L] > d.perm_p95[L] for L in pk)),
             late_layers_above_perm=round(float(np.mean(c0[lo:] > d.perm_p95[lo:])), 2), top_peaks=",".join(map(str, pk[:3])),
             peaks_ratio_perm=",".join(f"{c0[L] / d.perm_p95[L]:.1f}" for L in pk[:3]),
             peaks_still_peak=",".join(f"{still(L):.2f}" for L in pk[:3]))
    rows.append(r); print(r, flush=True)
S = f"{R}/events/noise_summary.csv"
old = pd.read_csv(S) if os.path.exists(S) else pd.DataFrame()
pd.concat([old, pd.DataFrame(rows)]).drop_duplicates(["model", "task"], keep="last").to_csv(S, index=False)
