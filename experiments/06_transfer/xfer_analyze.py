"""跨層遷移 steering 的彙總。每組：hit[X, Y]（三模板平均）。
- 可用的 Y：原生（X=Y）成功率 ≥ 0.9 的層（Y_ok）。
- X 的遷移力 = 在所有 Y_ok 上的平均成功率；以及在 Y ≥ X 的 Y_ok 上的平均（向後遷移）。
- 比較各種「建向量的層」選法：形變第 1 峰、峰 +1 / +2、lens 首次 ≥ 0.1、原生 steering 首次 ≥ 0.9、80% 深度、最後一層、最佳 X（上限）。
輸出 results/steer_xfer/summary.csv、summary.md、<組>_heat.png"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import glob, os, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = f"{DATA}"; O = f"{R}/steer_xfer"
E = pd.read_csv(f"{R}/events/events.csv"); S = pd.read_csv(f"{R}/events/signals.csv")
rows = []
for f in sorted(glob.glob(f"{O}/*_*.csv")):
    name = os.path.basename(f)[:-4]
    if "_" not in name or name.startswith(("summary", "dircos", "boundary", "jump", "queue")): continue
    m, t = name.rsplit("_", 1); d = pd.read_csv(f)
    M = d.groupby(["X", "Y"]).hit.mean().unstack()  # 行 X、列 Y
    Ls = sorted(set(M.index) & set(M.columns))
    M = M.loc[Ls, Ls]; full = M.notna().all(axis=0); Ys = [Y for Y in Ls if full[Y]]
    if len(Ys) < 3: continue
    nat = pd.Series({Y: M.loc[Y, Y] for Y in Ys}); Yok = [Y for Y in Ys if nat[Y] >= 0.9]
    NL = int(E[(E.model == m) & (E.task == t)].layer.max()) + 1 if len(E[(E.model == m) & (E.task == t)]) else max(Ls) + 1
    tr = pd.Series({X: M.loc[X, Yok].mean() if Yok else np.nan for X in Ls})
    fwd = pd.Series({X: M.loc[X, [Y for Y in Yok if Y >= X]].mean() if [Y for Y in Yok if Y >= X] else np.nan for X in Ls})
    s = S[(S.model == m) & (S.task == t)].set_index("layer")
    pk = E[(E.model == m) & (E.task == t) & (E.peak == "第1峰")].layer
    pk = int(pk.iloc[0]) if len(pk) else np.nan
    lens10 = next((L for L in s.index if s.lens[L] >= 0.1), np.nan) if "lens" in s and s.lens.notna().any() else np.nan
    st90 = next((Y for Y in Ys if nat[Y] >= 0.9), np.nan)
    rules = {"形變第1峰": pk, "峰+1": pk + 1, "峰+2": pk + 2, "lens≥0.1": lens10, "原生steering≥0.9": st90, "80%深度": int(round(0.8 * (NL - 1))), "最後一層": max(Ls)}
    r = dict(model=m, task=t, NL=NL, done_Y=len(Ys), n_Yok=len(Yok), first_ok=st90, peak=pk, best_X=int(tr.idxmax()) if tr.notna().any() else np.nan, best_tr=round(tr.max(), 3))
    for k, X in rules.items():
        r[f"X_{k}"] = X; r[f"tr_{k}"] = round(tr.get(X, np.nan), 3) if X == X else np.nan; r[f"fwd_{k}"] = round(fwd.get(X, np.nan), 3) if X == X else np.nan
    # 峰之前 / 之後建的方向的平均遷移力
    r["tr_X<peak(5層)"] = round(tr[[X for X in Ls if pk - 5 <= X < pk]].mean(), 3) if pk == pk else np.nan
    r["tr_X>=peak"] = round(tr[[X for X in Ls if X >= pk]].mean(), 3) if pk == pk else np.nan
    rows.append(r)
    fig, ax = plt.subplots(figsize=(6.4, 5.4)); im = ax.imshow(M.loc[Ys, Ys].T.values, origin="lower", cmap="Blues", vmin=0, vmax=1, extent=[Ys[0] - .5, Ys[-1] + .5, Ys[0] - .5, Ys[-1] + .5])
    if pk == pk: ax.axvline(pk - .5, color="#e34948", lw=1.5, ls="--", label=f"shape peak L{pk}"); ax.axhline(pk - .5, color="#e34948", lw=1, ls=":")
    if st90 == st90: ax.axhline(st90 - .5, color="#008300", lw=1.5, ls="--", label=f"native steer ≥0.9 L{st90}")
    ax.set_xlabel("X: layer the direction is built at"); ax.set_ylabel("Y: layer it is added at"); ax.set_title(f"{m.split('-Instruct')[0]} · {t}: steering hit", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046); ax.legend(loc="upper left", fontsize=8, frameon=False); fig.tight_layout(); fig.savefig(f"{O}/{name}_heat.png", dpi=120); plt.close(fig)
D = pd.DataFrame(rows); D.to_csv(f"{O}/summary.csv", index=False)
cols = ["model", "task", "done_Y", "n_Yok", "first_ok", "peak", "best_X", "best_tr"] + [c for c in D.columns if c.startswith("tr_")]
print(D[cols].to_string(index=False))
