"""跨層遷移的三個補充分析（38 組，只用現有數據）。
(1) 邊界候選比較：形變第 1 峰 / 身份探針首次 ≥0.5 / 同層信度首次 ≥0.5（答案結構出現）/ lens 首次 ≥0.1。
    對每個候選層 B：B 之前 5 層的遷移力、B 及之後的遷移力、與真實邊界（遷移力首次 ≥0.5 / ≥0.8）的距離；按第 1 峰事件類型分組。
(2) 峰前已有結構：峰前、同層信度 ≥ 0.5 的層的遷移力，對照峰後同樣信度範圍的層。
(3) 置換檢驗：遷移力最大跳升落在第 1–3 峰（不分類型）的組數；虛無分布 = 各組峰位置保持間距、整體隨機平移（循環），10000 次。
輸出 results/steer_xfer/extra_*.csv、extra.md"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import glob, os, json, numpy as np, pandas as pd, torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
R = f"{DATA}"; O = f"{R}/steer_xfer"
E = pd.read_csv(f"{R}/events/events.csv"); S = pd.read_csv(f"{R}/events/signals.csv"); RR = pd.read_csv(f"{O}/report_rules.csv")
def id_from_acts(m, t):  # 與 group.py 相同的身份探針（PCA 64 + logistic，5 折）
    a = torch.load(f"{R}/acts/{m}_{t}_acts.pt", weights_only=False); H = a["H"].float().numpy(); y = np.asarray(a["ans"]); out = []
    for L in range(H.shape[1]):
        X = H[:, L]; X = (X - X.mean(0)) / (X.std(0) + 1e-6); X = PCA(64, random_state=0).fit_transform(X)
        out.append(float(cross_val_score(LogisticRegression(max_iter=2000, C=0.1), X, y, cv=5).mean()))
    return pd.Series(out)
first = lambda s, th, lo=0: next((int(L) for L in s.index if L >= lo and s[L] >= th), np.nan)
G = []
for r in RR.itertuples():
    m, t = r.model, r.task; d = pd.read_csv(f"{O}/{m}_{t}.csv"); M = d.groupby(["X", "Y"]).hit.mean().unstack()
    Ys = [Y for Y in M.columns if M[Y].notna().all()]; Yok = [Y for Y in Ys if M.loc[Y, Y] >= 0.9]
    tr = pd.Series({X: M.loc[X, Yok].mean() for X in M.index})
    s = S[(S.model == m) & (S.task == t)].set_index("layer"); NL = len(s); lo = int(round(NL * 0.25))
    idv = s["id"] if s["id"].notna().any() else id_from_acts(m, t)
    rel = pd.read_csv(f"{R}/events/cv/{m}_{t}.csv").set_index("layer").rel
    e = E[(E.model == m) & (E.task == t)]; pks = sorted(e[e.peak.isin(["第1峰", "第2–3峰"])].layer.astype(int).tolist())
    G.append(dict(m=m, t=t, NL=NL, lo=lo, tr=tr, rel=rel, typ=r.第1峰類型, pks=pks,
                  cand={"形變第1峰": int(r.第1峰), "身份≥0.5": first(idv, 0.5), "身份≥0.5（晚期）": first(idv, 0.5, lo),
                        "信度≥0.5": first(rel, 0.5), "信度≥0.5（晚期）": first(rel, 0.5, lo), "lens≥0.1": first(s.lens, 0.1) if s.lens.notna().any() else np.nan},
                  b5=first(tr, 0.5), b8=first(tr, 0.8)))
# ---------- (1) 邊界候選 ----------
rows = []
for g in G:
    tr = g["tr"]
    for k, B in g["cand"].items():
        if B != B: continue
        pre = tr[[X for X in tr.index if B - 5 <= X < B]]; post = tr[[X for X in tr.index if X >= B]]; near = tr[[X for X in tr.index if B <= X <= B + 2]]
        rows.append(dict(model=g["m"], task=g["t"], 第1峰類型=g["typ"], 候選=k, B=B, 之前5層=pre.mean() if len(pre) else np.nan, 之後=post.mean(), B至B加2=near.mean(),
                         誤差05=B - g["b5"], 誤差08=B - g["b8"]))
C = pd.DataFrame(rows); C.to_csv(f"{O}/extra_boundary.csv", index=False)
agg = lambda df: df.groupby("候選").agg(n=("B", "size"), 之前5層=("之前5層", "mean"), 之後=("之後", "mean"), B至B加2=("B至B加2", "mean"),
                                        MAE05=("誤差05", lambda x: x.abs().mean()), 偏差05=("誤差05", "mean"), MAE08=("誤差08", lambda x: x.abs().mean()), 偏差08=("誤差08", "mean")).round(2)
out = ["# 跨層遷移：三個補充分析\n", "## (1) 邊界候選比較（38 組）\n", "誤差 = 候選層 − 真實邊界（遷移力首次 ≥0.5 / ≥0.8）；負 = 候選偏早。「之前 5 層」越低、「之後」越高越好。\n", agg(C).to_markdown()]
for ty in ["可讀重組", "寫入"]:
    out += [f"\n### 第 1 峰為「{ty}」的組\n", agg(C[C.第1峰類型 == ty]).to_markdown()]
# 每組：哪個候選最準（|誤差05| 最小）
win = C.dropna(subset=["誤差05"]).assign(ae=lambda x: x.誤差05.abs()).pipe(lambda x: x[x.候選.isin(["形變第1峰", "身份≥0.5（晚期）", "信度≥0.5（晚期）", "lens≥0.1"])])
best = win.loc[win.groupby(["model", "task"]).ae.idxmin()]
out += ["\n每組最準的候選（四個晚期候選中 |誤差05| 最小；並列取先列者）：\n", best.候選.value_counts().to_markdown()]
# ---------- (2) 峰前已有結構 ----------
rows = []
for g in G:
    p1 = g["cand"]["形變第1峰"]; tr, rel = g["tr"], g["rel"]
    for X in tr.index:
        rows.append(dict(model=g["m"], task=g["t"], 第1峰類型=g["typ"], X=X, 相對峰=X - p1, 位置="峰前" if X < p1 else "峰及之後", rel=rel.get(X, np.nan), tr=tr[X]))
P = pd.DataFrame(rows); P.to_csv(f"{O}/extra_prepeak.csv", index=False)
P["信度區間"] = pd.cut(P.rel, [-1, 0.3, 0.5, 0.7, 0.9, 1.01], labels=["<0.3", "0.3–0.5", "0.5–0.7", "0.7–0.9", "≥0.9"])
tab = P.groupby(["信度區間", "位置"], observed=True).tr.agg(["mean", "median", "size"]).round(2).unstack("位置")
pre_rel = P[(P.位置 == "峰前") & (P.rel >= 0.5)]
grp = pre_rel.groupby(["model", "task", "第1峰類型"]).agg(層數=("X", "size"), 平均信度=("rel", "mean"), 遷移力=("tr", "mean"), 最高=("tr", "max")).round(2).reset_index()
out += ["\n## (2) 峰前已有可靠結構的層\n", "所有 (組, X) 按同層信度分箱，比較峰前與峰後的遷移力：\n", tab.to_markdown(),
        f"\n峰前且信度 ≥ 0.5 的層：{len(pre_rel)} 個，來自 {pre_rel.groupby(['model', 'task']).ngroups} 組；遷移力平均 {pre_rel.tr.mean():.2f}、中位 {pre_rel.tr.median():.2f}、> 0.5 的比例 {(pre_rel.tr > 0.5).mean():.2f}。\n",
        "各組明細：\n", grp.to_markdown(index=False)]
# ---------- (3) 置換檢驗 ----------
rng = np.random.default_rng(0); obs_exact = obs_pm1 = obs_p1 = 0; null = np.zeros((10000, 3)); per = []
for g in G:
    tr = g["tr"]; dj = tr.diff(); late = [int(L) for L in dj.index if np.isfinite(dj[L]) and L >= g["lo"]]
    big = int(dj[late].idxmax()); pks = [p for p in g["pks"] if p in late]; p1 = g["cand"]["形變第1峰"]
    ex = big in pks; pm = any(abs(big - p) <= 1 for p in pks); f1 = abs(big - p1) <= 1
    obs_exact += ex; obs_pm1 += pm; obs_p1 += f1
    lo_, hi_ = min(late), max(late); span = hi_ - lo_ + 1
    sh = rng.integers(0, span, size=10000)
    for j, k in enumerate(sh):
        q = [lo_ + (p - lo_ + k) % span for p in pks]; q1 = lo_ + (p1 - lo_ + k) % span
        null[j] += [big in q, any(abs(big - x) <= 1 for x in q), abs(big - q1) <= 1]
    per.append(dict(model=g["m"], task=g["t"], 最大跳升=big, 峰=",".join(map(str, pks)), 正好在峰上=ex, 在峰1層內=pm, 在第1峰1層內=f1))
pd.DataFrame(per).to_csv(f"{O}/extra_perm.csv", index=False)
pv = lambda o, col: (np.sum(null[:, col] >= o) + 1) / (len(null) + 1)
out += ["\n## (3) 置換檢驗：遷移力最大跳升是否落在形變峰上（不分事件類型）\n",
        "虛無分布：每組的第 1–3 峰保持彼此間距，在晚期層範圍內整體循環平移一個隨機量；10000 次。\n",
        "| 統計量 | 觀察值 | 虛無平均 | 虛無 95% | p |\n|---|---|---|---|---|",
        f"| 最大跳升正好在第 1–3 峰上 | {obs_exact}/{len(G)} | {null[:, 0].mean():.1f} | {np.percentile(null[:, 0], 95):.0f} | {pv(obs_exact, 0):.4f} |",
        f"| 最大跳升在第 1–3 峰 ±1 層內 | {obs_pm1}/{len(G)} | {null[:, 1].mean():.1f} | {np.percentile(null[:, 1], 95):.0f} | {pv(obs_pm1, 1):.4f} |",
        f"| 最大跳升在第 1 峰 ±1 層內 | {obs_p1}/{len(G)} | {null[:, 2].mean():.1f} | {np.percentile(null[:, 2], 95):.0f} | {pv(obs_p1, 2):.4f} |"]
open(f"{O}/extra.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
