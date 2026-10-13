"""跨層遷移彙總（只用三個模板都完成的組）：
(1) 各建向量選法的遷移力；(2) 峰前 / 峰後；(3) 按 X − 第 1 峰對齊的逐層曲線；(4) 遷移力最大跳升是否落在第 1–3 峰（含隨機基準）；
(5) 按事件類型（第 1 峰是寫入 / 可讀重組 / 集中…）與任務類別分組。輸出 results/steer_xfer/report_*.csv，並印出表格。"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import glob, os, numpy as np, pandas as pd, json
R = f"{DATA}"; O = f"{R}/steer_xfer"
E = pd.read_csv(f"{R}/events/events.csv"); S = pd.read_csv(f"{R}/events/signals.csv")
CALC = {"weekday", "month", "hour", "letter"}
def acc(m, t):
    f = f"{R}/{m}/{t}/ans/row.json"
    return json.load(open(f)).get("acc") if os.path.exists(f) else np.nan
G = []
for f in sorted(glob.glob(f"{O}/*_*.csv")):
    n = os.path.basename(f)[:-4]
    if n.startswith(("summary", "dircos", "boundary", "jump", "queue", "report", "shared")): continue
    m, t = n.rsplit("_", 1); d = pd.read_csv(f)
    if sorted(d.template.unique()) != [0, 1, 2]: continue
    last_t = d[d.template == 2]; Yall = sorted(d[d.template == 0].Y.unique())
    if sorted(last_t.Y.unique()) != Yall: continue  # 第三個模板還沒跑完
    if d.groupby("template").n.first().sum() < 20: print(f"跳過 {n}：測試題 < 20"); continue
    M = d.groupby(["X", "Y"]).hit.mean().unstack(); Ys = [Y for Y in M.columns if M[Y].notna().all()]
    Yok = [Y for Y in Ys if M.loc[Y, Y] >= 0.9]
    if not Yok: continue
    tr = pd.Series({X: M.loc[X, Yok].mean() for X in M.index})
    e = E[(E.model == m) & (E.task == t)]; top = e[e.peak.isin(["第1峰", "第2–3峰"])]
    if not len(e[e.peak == "第1峰"]): continue
    p1 = int(e[e.peak == "第1峰"].layer.iloc[0]); typ1 = e[e.peak == "第1峰"].type.iloc[0]; pks = sorted(top.layer.tolist())
    s = S[(S.model == m) & (S.task == t)].set_index("layer"); NL = len(s)
    l10 = next((L for L in s.index if s.lens[L] >= 0.1), np.nan) if s.lens.notna().any() else np.nan
    st90 = next((Y for Y in sorted(Ys) if M.loc[Y, Y] >= 0.9), np.nan)
    G.append(dict(model=m, task=t, NL=NL, acc=acc(m, t), p1=p1, typ1=typ1, pks=pks, types={int(r.layer): r.type for r in top.itertuples()},
                  tr=tr, l10=l10, st90=st90))
print(f"完成的組數：{len(G)}")
rows, al, jr = [], {}, []
for g in G:
    tr, p1, NL = g["tr"], g["p1"], g["NL"]
    get = lambda X: tr.get(X, np.nan) if X == X else np.nan
    r = dict(model=g["model"], task=g["task"], 類別="計算" if g["task"] in CALC else "檢索/語義", acc=g["acc"], 第1峰=p1, 第1峰類型=g["typ1"],
             峰前5層=tr[[X for X in tr.index if p1 - 5 <= X < p1]].mean(), 峰及之後=tr[[X for X in tr.index if X >= p1]].mean(),
             第1峰當層=get(p1), 峰加2=get(p1 + 2), lens10=get(g["l10"]), 原生st90=get(g["st90"]), 深度80=get(int(round(0.8 * (NL - 1)))), 最後一層=get(NL - 1),
             最佳=tr.max(), 最佳X=int(tr.idxmax()), 峰後到08=next((X - p1 for X in tr.index if X >= p1 and tr[X] >= 0.8), np.nan))
    rows.append(r)
    for X in tr.index: al.setdefault(X - p1, []).append(tr[X])
    dj = tr.diff(); big = int(dj.idxmax()); late = [L for L in dj.index if np.isfinite(dj[L])]
    jr.append(dict(model=g["model"], task=g["task"], 第1峰=p1, 第1峰類型=g["typ1"], 第1峰跳幅=dj.get(p1, np.nan), 最大跳升層=big, 最大跳幅=dj[big],
                   最大跳升處類型=g["types"].get(big, "（非峰）"), 在峰上=big in g["pks"], 在峰1內=min(abs(big - p) for p in g["pks"]) <= 1,
                   隨機機率=len(set(g["pks"]) & set(late)) / len(late), 隨機機率1=len({L for p in g["pks"] for L in (p - 1, p, p + 1)} & set(late)) / len(late),
                   峰上跳幅佔比=sum(max(dj.get(p, 0), 0) for p in g["pks"]) / max(tr.max() - tr.min(), 1e-6)))
D = pd.DataFrame(rows); J = pd.DataFrame(jr)
D.to_csv(f"{O}/report_rules.csv", index=False); J.to_csv(f"{O}/report_jumps.csv", index=False)
cols = ["峰前5層", "峰及之後", "第1峰當層", "峰加2", "lens10", "原生st90", "深度80", "最後一層", "最佳"]
print("\n(1) 各選法遷移力：平均（中位）"); print(pd.DataFrame({c: [f"{D[c].mean():.2f}（{D[c].median():.2f}）"] for c in cols}).to_string(index=False))
print("\n按第 1 峰事件類型"); print(D.groupby("第1峰類型")[cols + ["峰後到08"]].mean().round(2).assign(n=D.groupby("第1峰類型").size()).to_string())
print("\n按任務類別"); print(D.groupby("類別")[cols + ["峰後到08"]].mean().round(2).assign(n=D.groupby("類別").size()).to_string())
print("\n按模型"); print(D.groupby("model")[cols + ["峰後到08"]].mean().round(2).assign(n=D.groupby("model").size()).to_string())
lo = D[D.acc < 0.55]
print(f"\n去掉準確率 <0.55 的 {len(lo)} 組：", {c: round(D[D.acc >= 0.55][c].mean(), 2) for c in ["峰前5層", "峰及之後", "第1峰當層", "最後一層"]})
print("\n(3) 按 X − 第 1 峰對齊")
A = pd.DataFrame([dict(offset=o, mean=np.mean(v), min=np.min(v), n=len(v)) for o, v in sorted(al.items()) if -3 <= o <= 16]); print(A.round(2).to_string(index=False))
A.to_csv(f"{O}/report_aligned.csv", index=False)
print("\n(4) 最大跳升落在第 1–3 峰：", int(J.在峰上.sum()), "/", len(J), f"（隨機期望 {J.隨機機率.mean():.2f}）；±1 層內：", int(J.在峰1內.sum()), f"（隨機期望 {J.隨機機率1.mean():.2f}）")
print("第 1 峰本身跳幅 ≥ 0.3：", int((J.第1峰跳幅 >= 0.3).sum()), "/", len(J))
print("最大跳升所在峰的類型："); print(J.最大跳升處類型.value_counts().to_string())
print("按第 1 峰類型：第 1 峰跳幅平均"); print(J.groupby("第1峰類型").第1峰跳幅.agg(["mean", "size"]).round(2).to_string())
print("峰上跳幅佔總上升：中位", round(J.峰上跳幅佔比.median(), 2), "範圍", round(J.峰上跳幅佔比.min(), 2), "–", round(J.峰上跳幅佔比.max(), 2))
