"""離線：答案位置 centroid 差值方向的跨層餘弦（所有類別對 i≠j 的 c_j − c_i，取平均餘弦）。
看形變峰前後是否分成兩塊：峰之後建的方向彼此相似（排列穩定），峰之前的方向與峰之後不同。
輸出 results/steer_xfer/dircos/<名稱>_<concept>.csv（X × Y 平均餘弦）與 dircos_summary.csv。"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import glob, os, numpy as np, pandas as pd, torch
R = f"{DATA}"; OUT = f"{R}/steer_xfer/dircos"; os.makedirs(OUT, exist_ok=True)
E = pd.read_csv(f"{R}/events/events.csv")
rows = []
for f in sorted(glob.glob(f"{R}/acts/*_acts.pt")):
    name = os.path.basename(f)[:-8]; m, t = name.rsplit("_", 1)
    a = torch.load(f, weights_only=False); H = a["H"].float(); y = np.asarray(a["ans"]); cls = np.unique(y); NL = H.shape[1]
    C = torch.stack([H[y == c].mean(0) for c in cls], 1)  # NL × n × D
    i, j = torch.triu_indices(len(cls), len(cls), 1)
    Dv = C[:, j] - C[:, i]; Dv = Dv / Dv.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # NL × pairs × D
    M = torch.einsum("xpd,ypd->xy", Dv, Dv).numpy() / Dv.shape[1]
    pd.DataFrame(M, index=range(NL), columns=range(NL)).to_csv(f"{OUT}/{name}.csv")
    e = E[(E.model == m) & (E.task == t) & (E.peak == "第1峰")]
    if not len(e): continue
    pk = int(e.layer.iloc[0]); last = NL - 1
    # 與最後一層方向的餘弦：每層；峰前 / 峰後平均；峰後到最後的平均跨層餘弦
    r = dict(model=m, task=t, NL=NL, peak=pk, cos_to_last_at_peak_m1=round(M[pk - 1, last], 3), cos_to_last_at_peak=round(M[pk, last], 3),
             cos_to_last_at_peak_p2=round(M[min(pk + 2, last), last], 3),
             within_post=round(M[pk:, pk:][np.triu_indices(NL - pk, 1)].mean(), 3),
             within_pre5=round(M[pk - 5:pk, pk - 5:pk][np.triu_indices(5, 1)].mean(), 3),
             pre_post=round(M[pk - 5:pk, pk:].mean(), 3),
             biggest_drop_layer=int(max(range(int(round(NL * 0.25)), NL), key=lambda L: 1 - M[L, L - 1])))
    rows.append(r); print(r, flush=True)
pd.DataFrame(rows).to_csv(f"{R}/steer_xfer/dircos_summary.csv", index=False)
