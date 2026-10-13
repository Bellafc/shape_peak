"""交叉驗證的形變（扣掉題目層面噪聲）：題目按類別分層隨機分成兩半 A / B。
相似度 sim(L) = ½(‖X_Aᵀ Y_B‖_* + ‖X_Bᵀ Y_A‖_*) / √(⟨X_A, X_B⟩ ⟨Y_A, Y_B⟩)，X = 第 L 層、Y = 第 L−1 層的中心化 centroid（n×D）。
兩半的題目噪聲獨立，交叉項與分母都不含噪聲平方，所以排列沒變時 change_cv = 1 − sim² 的期望接近 0（原始 change_full 則會被噪聲撐大）。
重複 S 次分半；另做標籤打亂（應接近 0）。輸出 results/events/cv/<名稱>_<concept>.csv 與 results/events/cv_summary.csv。"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import glob, os, sys, numpy as np, pandas as pd, torch
R = f"{DATA}"; OUT = f"{R}/events/cv"; os.makedirs(OUT, exist_ok=True)
S_ = int(os.environ.get("S", 50)); EARLY = 0.25; torch.set_num_threads(8)
def udist(CA, CB):
    """無偏的類間距離平方：⟨c_iA − c_jA, c_iB − c_jB⟩（兩份題目獨立，噪聲期望為 0），回傳上三角向量。"""
    GA, GB = CA - CA.mean(0), CB - CB.mean(0); G = GA @ GB.T; G = 0.5 * (G + G.T)
    D = G.diag()[:, None] + G.diag()[None, :] - 2 * G; iu = torch.triu_indices(len(G), len(G), 1); return D[iu[0], iu[1]]
cos = lambda a, b: float((a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-12))
def curve(H, y, cls, rng):
    Qs = [[], [], [], []]
    for c in cls:
        ix = rng.permutation(np.where(y == c)[0]); q = len(ix) // 4
        for k in range(4): Qs[k].append(ix[k * q:(k + 1) * q])
    C = [torch.stack([H[i].mean(0) for i in Q], 1) for Q in Qs]  # 每份：NL × n × D
    D12 = [udist(C[0][L], C[1][L]) for L in range(H.shape[1])]; D34 = [udist(C[2][L], C[3][L]) for L in range(H.shape[1])]
    rel = np.array([cos(D12[L], D34[L]) for L in range(H.shape[1])])  # 同層信度
    cross = np.array([np.nan] + [0.5 * (cos(D12[L], D34[L - 1]) + cos(D34[L], D12[L - 1])) for L in range(1, H.shape[1])])
    adj = cross / np.sqrt(np.clip(rel, 0.05, None) * np.clip(np.r_[np.nan, rel[:-1]], 0.05, None))
    return 1 - np.clip(adj, -1, 1), rel
rows = []
files = sorted(glob.glob(f"{R}/acts/*_acts.pt")) if len(sys.argv) < 2 else sys.argv[1:]
for f in files:
    name = os.path.basename(f)[:-8]; m, t = name.rsplit("_", 1)
    a = torch.load(f, weights_only=False); H = a["H"].float(); y = np.asarray(a["ans"]); cls = np.unique(y)
    NL = H.shape[1]; lo = int(round(NL * EARLY)); rng = np.random.default_rng(0)
    res = [curve(H, y, cls, rng) for _ in range(S_)]; cv = np.array([r_[0] for r_ in res]); rel = np.array([r_[1] for r_ in res]).mean(0)
    raw = pd.read_csv(f"{R}/events/noise/{name}.csv").change.values
    d = pd.DataFrame(dict(layer=range(NL), raw=raw, cv=np.nanmean(cv, 0), cv_lo=np.nanpercentile(cv, 5, 0), cv_hi=np.nanpercentile(cv, 95, 0),
                          rel=rel))
    d.to_csv(f"{OUT}/{name}.csv", index=False)
    late = np.arange(lo, NL); rl = late[rel[late] >= 0.5]
    base = np.nanmedian(d.cv.values[rl]) if len(rl) else np.nan  # 可靠晚期層的典型去噪形變
    medraw = np.nanmedian(raw[lo:])
    pk = sorted([L for L in late if raw[L] >= np.nanmax(raw[max(1, L - 1):L + 2]) and raw[L] >= 1.5 * medraw], key=lambda L: -raw[L])
    def kind(L):
        if rel[L] >= 0.5 and rel[L - 1] < 0.5: return "結構出現"
        if rel[L] < 0.5: return "結構不可靠"
        return "重組（顯著）" if np.mean(cv[:, L] > 2 * base) >= 0.8 else "不顯著"
    for rk, L in enumerate(pk[:3], 1):
        rows.append(dict(model=m, task=t, NL=NL, rank=rk, layer=L, raw=round(raw[L], 4), cv=round(d.cv[L], 4), cv_lo=round(d.cv_lo[L], 4),
                         base_cv=round(base, 4), frac_gt2base=round(float(np.mean(cv[:, L] > 2 * base)), 2), rel_prev=round(rel[L - 1], 2), rel=round(rel[L], 2), kind=kind(L)))
        print(rows[-1], flush=True)
    r = None
P = f"{R}/events/cv_summary.csv"; old = pd.read_csv(P) if os.path.exists(P) and len(sys.argv) >= 2 else pd.DataFrame()
pd.concat([old, pd.DataFrame(rows)]).drop_duplicates(["model", "task", "rank"], keep="last").to_csv(P, index=False)
