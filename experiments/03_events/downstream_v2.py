"""用 v2 事件類型重算所有依賴事件類型的分析。"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import pandas as pd, numpy as np, glob, os
from scipy.stats import binomtest
E2 = pd.read_csv("events_v2.csv"); out = []; P = lambda *a: (print(*a), out.append(" ".join(map(str, a))))
TOP = {k: g[g.peak.isin(["第1峰", "第2–3峰"])].sort_values("layer") for k, g in E2.groupby(["model", "task"])}
first = {k: g[g.peak == "第1峰"].iloc[0] for k, g in E2.groupby(["model", "task"])}
# ---- 第四部分 ----
T = pd.read_csv(f"{DATA}/handoff_all/table.csv")
MAPT = {"星期": "weekday", "月份": "month", "小時": "hour", "字母": "letter", "城市": "city", "國家→洲": "continent", "首都→國家→洲": "capital2hop", "顏色": "color", "上下文顏色": "ctxcolor", "情緒": "emotion"}
T["task"] = T["任務"].map(MAPT); T = T.dropna(subset=["輸入失效_patching"])
rows = []
for r in T.itertuples():
    k = (r.模型, r.task); f = first[k]; h = int(r.輸入失效_patching)
    rows.append(dict(model=r.模型, task=r.task, peak=int(f.layer), type=f.type, handoff=h, off=h - int(f.layer), top3=min(abs(h - TOP[k].layer.values)) <= 1))
H = pd.DataFrame(rows); H.to_csv("handoff_types_v2.csv", index=False)
P("# 依賴事件類型的分析（v2 分類）\n"); P("## 第四部分：交接層 − 第 1 峰，按第 1 峰事件類型（35 組）\n")
P("| 事件 | n | off 中位 | ±1 內 | off>0 | off<0 | 符號檢驗 p |"); P("|---|---|---|---|---|---|---|")
for t, g in H.groupby("type"):
    pos, neg = (g.off > 0).sum(), (g.off < 0).sum(); p = binomtest(pos, pos + neg).pvalue if pos + neg else np.nan
    P(f"| {t} | {len(g)} | {g.off.median():+.0f} | {(g.off.abs() <= 1).sum()}/{len(g)} | {pos} | {neg} | {p:.3f} |")
P(f"\n交接在前三峰之一 ±1：{H.top3.sum()}/{len(H)}（不依賴事件類型，與 v1 相同）")
# ---- 第二部分：起效 − 第 1 峰，按類型 ----
O = pd.read_csv(f"{DATA}/steer_sweep/report_onset.csv"); full = {"Gemma": "Gemma-2-9B-it", "Llama": "Llama-3.1-8B-Instruct", "Mistral": "Mistral-7B-Instruct-v0.3", "Qwen2.5": "Qwen2.5-7B-Instruct"}
O["model"] = O.model.map(lambda x: full.get(x, x)); O = O[~((O.model == "Mistral-7B-Instruct-v0.3") & (O.task == "hour"))]
O["type2"] = [first[(r.model, r.task)].type for r in O.itertuples()]; O["d1"] = [r.on_best - first[(r.model, r.task)].layer for r in O.itertuples()]
P("\n## 第二部分：起效層 − 第 1 峰（中位；括號 = 早於峰 >1 層的組數 / 組數）\n"); P("| 第1峰類型 | α=1 | α=2 | α=4 | α=8 | rel 0.4 | rel 0.8 |"); P("|---|---|---|---|---|---|---|")
for t, g in O.groupby("type2"):
    c = []
    for mo, s in [("nat", 1), ("nat", 2), ("nat", 4), ("nat", 8), ("rel", 0.4), ("rel", 0.8)]:
        x = g[(g["mode"] == mo) & (g.s == s)].d1.dropna(); c.append(f"{x.median():+.1f}（{(x < -1).sum()}/{len(x)}）")
    P(f"| {t} | " + " | ".join(c) + " |")
g1 = O[(O["mode"] == "nat") & (O.s == 1)]; P(f"\nα=1：起效層 ≥ 第 1 峰 − 1 的組 {(g1.d1 >= -1).sum()}/{len(g1)}；例外：" + "、".join(f"{r.model.split('-')[0]} {r.task}（{r.type2}，{r.d1:+.0f}）" for r in g1.itertuples() if r.d1 < -1))
# ---- 第三部分：遷移跳升落在哪類峰 ----
XP = f"{DATA}/steer_xfer"; jr = []
for f in sorted(glob.glob(f"{XP}/*_*.csv")):
    nm = os.path.basename(f)[:-4]
    if nm.startswith("report"): continue
    m, t = nm.rsplit("_", 1); d = pd.read_csv(f)
    if "template" not in d or sorted(d.template.unique()) != [0, 1, 2] or d.groupby("template").n.first().sum() < 20 or (m, t) not in first: continue
    M = d.groupby(["X", "Y"]).hit.mean().unstack(); Yok = [Y for Y in M.columns if Y in M.index and M.loc[Y, Y] >= 0.9]
    if not Yok: continue
    tr = pd.Series({X: M.loc[X, Yok].mean() for X in M.index}); dj = tr.diff(); big = int(dj.idxmax())
    tp = TOP[(m, t)]; types = dict(zip(tp.layer, tp.type)); p1 = int(first[(m, t)].layer)
    jr.append(dict(model=m, task=t, big=big, on_peak=big in types, type_at=types.get(big, "（非峰）"), first_type=first[(m, t)].type,
                   pre5=tr[[X for X in tr.index if p1 - 5 <= X < p1]].mean(), post=tr[[X for X in tr.index if X >= p1]].mean()))
JR = pd.DataFrame(jr); JR.to_csv("transfer_types_v2.csv", index=False)
P(f"\n## 第三部分：遷移力最大跳升（{len(JR)} 組）\n")
P(f"- 正好在前三峰之一：{JR.on_peak.sum()}/{len(JR)}；跳升所在峰的類型：" + "、".join(f"{k} {v}" for k, v in JR.type_at.value_counts().items()))
P("- 按第 1 峰類型的峰前 5 層 / 峰後遷移力：")
for t, g in JR.groupby("first_type"): P(f"  - {t}（{len(g)} 組）：{g.pre5.mean():.2f} / {g.post.mean():.2f}")
open("downstream_v2.md", "w").write("\n".join(out) + "\n")
