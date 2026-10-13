import pandas as pd, numpy as np, itertools
from events_v2 import classify
sig = pd.read_csv("signals_v2.csv"); E2 = pd.read_csv("events_v2.csv")
STARTK = {"weekday", "month", "hour", "letter"}
out = []; P = lambda *a: (print(*a), out.append(" ".join(map(str, a))))
P("# 第一部分 v2：事件分類改用交接實驗的因果信號（39 組，4 個大模型）\n")
P("## 1. 形變峰對應的事件類型\n"); P("| | 寫入 | 可讀重組 | 集中 | 只有 lens | 無事件 | n |"); P("|---|---|---|---|---|---|---|")
for lab, sel in [("第 1 峰", E2.peak == "第1峰"), ("第 2–3 峰", E2.peak == "第2–3峰"), ("非峰層", E2.peak == "非峰")]:
    g = E2[sel]; v = g.type.value_counts(normalize=True)
    P(f"| {lab} | {v.get('寫入',0):.2f} | {v.get('可讀重組',0):.2f} | {v.get('集中',0):.2f} | {v.get('輸出化',0):.2f} | {v.get('無事件',0):.2f} | {len(g)} |")
f1 = E2[E2.peak == "第1峰"].assign(cat=lambda d: np.where(d.task.isin(STARTK), "計算", "檢索/語義"))
P("\n按任務類別（第 1 峰）："); P(pd.crosstab(f1.cat, f1.type).to_string())
# 2. 各信號跳升靠近峰
P("\n## 2. 各信號跳升落在前三峰 ±1 層內的比例\n")
top = E2[E2.peak.isin(["第1峰", "第2–3峰"])]
def near_top(m, t, L): p = top[(top.model == m) & (top.task == t)].layer.values; return len(p) and np.min(np.abs(p - L)) <= 1
rows = []
for (m, t), g in sig.groupby(["model", "task"]):
    g = g.sort_values("layer").reset_index(drop=True); lo = int(round(len(g) * 0.25))
    for k in ["id", "pa", "pl", "sa", "sl", "lens"]:
        v = g[k].values
        for i in range(max(1, lo), len(g)):
            th = 0.15 if k == "id" else 0.10
            if np.isfinite(v[i]) and np.isfinite(v[i - 1]) and v[i] - v[i - 1] >= th: rows.append((k, near_top(m, t, i)))
J = pd.DataFrame(rows, columns=["signal", "near"]); base = np.mean([near_top(r.model, r.task, r.layer) for r in E2.itertuples()])
NAME = {"id": "身份探針", "pa": "patching（全部答案位置）", "pl": "patching（只換最後位置）", "sa": "steering（全部答案位置）", "sl": "steering（只改最後位置）", "lens": "logit lens"}
P("| 信號 | 比例 | n |"); P("|---|---|---|")
for k, g in J.groupby("signal"): P(f"| {NAME[k]} | {g.near.mean():.2f} | {len(g)} |")
P(f"| 隨機晚期層（基準） | {base:.2f} | — |")
# 3. 可讀重組 vs 寫入
P("\n## 3. lens 看不到的事件（前三峰）\n"); P("| 事件 | n | 當層 lens < 0.01 | 當層 A_lin ≤ 0.1 | 因果 ≥ 0.3 晚幾層（中位 / 平均） | lens ≥ 0.1 晚幾層（中位 / 平均） |"); P("|---|---|---|---|---|---|")
for typ in ["可讀重組", "寫入"]:
    g = top[top.type == typ]; dc, dl = [], []
    for r in g.itertuples():
        s = sig[(sig.model == r.model) & (sig.task == r.task)].set_index("layer"); c = s[["pa", "pl", "sa", "sl"]].max(axis=1)
        a = next((L for L in s.index if L >= r.layer and c[L] >= 0.3), np.nan); b = next((L for L in s.index if L >= r.layer and s.lens[L] >= 0.1), np.nan)
        dc.append(a - r.layer); dl.append(b - r.layer)
    P(f"| {typ} | {len(g)} | {(g.lens < 0.01).mean():.2f} | {(g.a_lin <= 0.1).mean():.2f} | {np.nanmedian(dc):.0f} / {np.nanmean(dc):.1f} | {np.nanmedian(dl):.0f} / {np.nanmean(dl):.1f} |")
# 4. 門檻敏感度
P("\n## 4. 門檻敏感度（144 種：跳升 0.05–0.20 × 峰門檻 1.25–3 × 窗口 0/±1/±2 × 早期界線 20–33%）\n")
res = []
for JUMP, PEAKR, WIN, EARLY in itertools.product([0.05, 0.1, 0.15, 0.2], [1.25, 1.5, 2, 3], [0, 1, 2], [0.2, 0.25, 0.33]):
    e = classify(sig, JUMP, PEAKR, WIN, EARLY); f = e[e.peak == "第1峰"]; nonp = e[e.peak == "非峰"]
    w1, w0 = (f.type == "寫入").mean(), (nonp.type == "寫入").mean(); n1, n0 = (f.type == "無事件").mean(), (nonp.type == "無事件").mean()
    rr = e[e.peak.isin(["第1峰", "第2–3峰"]) & (e.type == "可讀重組")]
    res.append(dict(JUMP=JUMP, PEAKR=PEAKR, WIN=WIN, EARLY=EARLY, write_first=w1, write_non=w0, ratio=w1 / w0 if w0 else np.nan, none_first=n1, none_non=n0,
                    lensonly_first=(f.type == "輸出化").mean(), rr_lens0=(rr.lens < 0.01).mean() if len(rr) else np.nan))
Rr = pd.DataFrame(res); Rr.to_csv("sensitivity_v2.csv", index=False)
P(f"- 第 1 峰寫入比例 > 非峰層：{(Rr.write_first > Rr.write_non).sum()}/144（倍數中位 {Rr.ratio.median():.1f}，範圍 {Rr.ratio.min():.1f}–{Rr.ratio.max():.1f}）")
P(f"- 第 1 峰無事件 < 非峰層：{(Rr.none_first < Rr.none_non).sum()}/144")
P(f"- 第 1 峰「只有 lens」= 0：{(Rr.lensonly_first == 0).sum()}/144（最多 {Rr.lensonly_first.max():.2f}）")
P(f"- 可讀重組當層 lens < 0.01 全部成立：{(Rr.rr_lens0 == 1).sum()}/{Rr.rr_lens0.notna().sum()}")
open("summary_v2.md", "w").write("\n".join(out) + "\n")
