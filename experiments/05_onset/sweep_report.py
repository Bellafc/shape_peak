"""強度掃描彙總：各強度下 steering 起效層 − 形變第 1 峰。python sweep_report.py [閾值，預設 0.5]
起效層 = 第一個（3 模板按題數加權）命中 ≥ 閾值的 Y；「最佳方向」= 該 Y 上三個方向（X = Y、最後一層、事後最佳 X）取最高。
另報：峰前（< 峰 − 1）各強度的最高命中、隨機方向對照、起效時的 p_other（輸出落在所有類別之外的機率，看是否推壞）。"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import sys, glob, os, numpy as np, pandas as pd
R = f"{DATA}"; TH = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
E = pd.read_csv(f"{R}/events/events.csv")
CAT = {"weekday": "計算", "month": "計算", "hour": "計算", "letter": "計算", "city": "檢索", "continent": "檢索", "capital2hop": "多跳", "color": "顏色", "ctxcolor": "顏色", "emotion": "情緒"}
wm = lambda g: pd.Series({"hit": (g.hit * g.n).sum() / g.n.sum(), "p_other": (g.p_other * g.n).sum() / g.n.sum()})
rows, pre = [], []
for f in sorted(glob.glob(f"{R}/steer_sweep/*_*.csv")):
    name, task = os.path.basename(f)[:-4].rsplit("_", 1)
    lg = f.replace(".csv", ".log")
    if not os.path.exists(lg) or "done" not in open(lg).read(): continue  # 只彙總跑完的組（也排除 report_*.csv）
    d = pd.read_csv(f); d["X"] = d.X.astype(str)
    p = E[(E.model == name) & (E.task == task) & (E.peak == "第1峰")]; pk, ty = int(p.layer.iloc[0]), p.type.iloc[0]
    d["dir"] = np.where(d.X == "rand", "rand", np.where(d.X == d.Y.astype(str), "own", "late"))
    g = d.groupby(["dir", "X", "mode", "s", "Y"]).apply(wm).reset_index()
    for (mode, s), h in g[g.dir != "rand"].groupby(["mode", "s"]):
        best = h.groupby("Y").hit.max(); own = h[h.dir == "own"].set_index("Y").hit
        on = lambda ser: next((int(y) for y, v in ser.sort_index().items() if v >= TH), None)
        ob, oo = on(best), on(own)
        po = h[(h.Y == ob)].sort_values("hit").p_other.iloc[-1] if ob is not None else np.nan
        rows.append(dict(model=name, task=task, cat=CAT[task], type=ty, peak=pk, mode=mode, s=s, on_best=ob, on_own=oo,
                         d_best=None if ob is None else ob - pk, d_own=None if oo is None else oo - pk, p_other_at_on=round(po, 2),
                         pre_max=round(best[best.index < pk - 1].max(), 2) if (best.index < pk - 1).any() else np.nan))
    for (mode, s), h in g[g.dir == "rand"].groupby(["mode", "s"]):
        pre.append(dict(model=name, task=task, mode=mode, s=s, rand_max=round(h.hit.max(), 2)))
D = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
if D.empty: sys.exit("還沒有跑完的組")
D.to_csv(f"{R}/steer_sweep/report_onset{"" if TH == 0.5 else f"_{TH}"}.csv", index=False)
print(f"閾值 {TH}；已完成 {D.groupby(['model', 'task']).ngroups} 組\n")
print("## 各強度：最佳方向起效層 − 第 1 峰\n| 強度 | 有起效的組 | 早於峰 >1 層 | 峰 ±1 | 晚於峰 >1 層 | 中位差 | 最早 | 峰前最高命中（中位 / 最大） | 起效時 p_other（中位） |\n|---|---|---|---|---|---|---|---|---|")
for (mode, s), h in D.groupby(["mode", "s"]):
    x = h.d_best.dropna()
    print(f"| {mode} {s:g} | {len(x)}/{len(h)} | {(x < -1).sum()} | {((x >= -1) & (x <= 1)).sum()} | {(x > 1).sum()} | {x.median():+.0f} | {x.min():+.0f} | {h.pre_max.median():.2f} / {h.pre_max.max():.2f} | {h.p_other_at_on.median():.2f} |")
print("\n## 每組：最佳方向起效層（各強度），括號 = 第 1 峰")
W = D.assign(k=D["mode"] + " " + D.s.map("{:g}".format)).pivot_table(index=["model", "task", "type", "peak"], columns="k", values="on_best", aggfunc="first")
print(W.to_string())
if pre:
    P = pd.DataFrame(pre); print("\n## 隨機方向對照（全部 Y 的最高命中）"); print(P.pivot_table(index=["model", "task"], columns=["mode", "s"], values="rand_max").to_string())
