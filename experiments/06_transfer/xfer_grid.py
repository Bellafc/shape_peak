"""跨層遷移總圖：4 個模型 × 10 個任務的 X × Y steering 熱圖；豎線 = 形變第 1–3 峰（按事件類型上色），點線 = 原生 steering 首次 ≥ 0.9。輸出 results/steer_xfer/grid.png"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import pandas as pd, numpy as np, os, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
PLAIN = os.environ.get("PLAIN") == "1"  # 1：只標第 1 峰 / 第 2–3 峰，不分事件類型、不畫原生 steering 線，輸出 grid_peaks.png
O = f"{DATA}/steer_xfer"; E = pd.read_csv(f"{DATA}/events/events.csv")
MODELS = [("Qwen2.5-7B-Instruct", "Qwen2.5-7B"), ("Llama-3.1-8B-Instruct", "Llama-3.1-8B"), ("Mistral-7B-Instruct-v0.3", "Mistral-7B"), ("Gemma-2-9B-it", "Gemma-2-9B")]
TASKS = [("weekday", "weekday"), ("month", "month"), ("hour", "hour"), ("letter", "letter"), ("city", "city"), ("continent", "country→continent"),
         ("capital2hop", "capital→country→cont."), ("color", "color"), ("ctxcolor", "color in context"), ("emotion", "emotion")]
TC = {"寫入": "#e34948", "可讀重組": "#4a3aa7", "集中": "#008300", "輸出化": "#eda100", "無事件": "#898781"}
NOTE = {("Gemma-2-9B-it", "emotion"): "native never ≥0.9\n(excluded)"}
fig, axs = plt.subplots(4, 10, figsize=(34, 14.5))
for i, (m, ml) in enumerate(MODELS):
    for j, (t, tl) in enumerate(TASKS):
        ax = axs[i, j]; f = f"{O}/{m}_{t}.csv"
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=7)
        if i == 0: ax.set_title(tl, fontsize=11, pad=8)
        if j == 0: ax.set_ylabel(f"{ml}\nY: layer added at", fontsize=10)
        if i == 3: ax.set_xlabel("X: layer direction built at", fontsize=8)
        if not os.path.exists(f):
            ax.text(0.5, 0.5, "not run", ha="center", va="center", fontsize=9, color="#898781", transform=ax.transAxes); ax.set_xticks([]); ax.set_yticks([]); continue
        d = pd.read_csv(f); M = d.groupby(["X", "Y"]).hit.mean().unstack(); Ys = [Y for Y in M.columns if M[Y].notna().all()]; Xs = list(M.index)
        im = ax.imshow(M.loc[Xs, Ys].T.values, origin="lower", cmap="Blues", vmin=0, vmax=1, aspect="auto", extent=[Xs[0] - .5, Xs[-1] + .5, Ys[0] - .5, Ys[-1] + .5])
        for r in E[(E.model == m) & (E.task == t) & E.peak.isin(["第1峰", "第2–3峰"])].itertuples():
            if Xs[0] <= r.layer <= Xs[-1]: ax.axvline(r.layer - .5, color="#e34948" if PLAIN else TC.get(r.type, "#898781"), lw=2.0 if r.peak == "第1峰" else 1.0, ls="-" if r.peak == "第1峰" else "--")
        s90 = next((Y for Y in Ys if M.loc[Y, Y] >= 0.9), None)
        if s90 is not None and not PLAIN: ax.axhline(s90 - .5, color="#1baf7a", lw=1.2, ls=":")
        ntest = int(d.groupby("template").n.first().sum())
        note = NOTE.get((m, t)) or (f"only {ntest} test prompts\n(accuracy too low; noisy)" if ntest < 20 else None)
        if note: ax.text(0.03, 0.95, note, transform=ax.transAxes, fontsize=7.5, va="top", color="#3d3d3a", bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
h = [Line2D([], [], color=c, lw=2) for c in TC.values()] + [Line2D([], [], color="#3d3d3a", lw=2), Line2D([], [], color="#3d3d3a", lw=1, ls="--"), Line2D([], [], color="#1baf7a", lw=1.2, ls=":")]
lab = ["write", "readable reorg", "concentrate", "output (lens)", "no event", "shape peak #1 (solid, thick)", "shape peaks #2–3 (dashed)", "native steering first ≥0.9"]
if PLAIN:
    h = [Line2D([], [], color="#e34948", lw=2), Line2D([], [], color="#e34948", lw=1, ls="--")]; lab = ["shape peak #1", "shape peaks #2–3"]
    fig.legend(h, lab, loc="upper center", ncol=2, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 1.0))
else:
    fig.legend(h, lab, loc="upper center", ncol=8, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.005), title="vertical lines: shape peaks colored by event type", title_fontsize=11)
fig.colorbar(im, ax=axs, fraction=0.012, pad=0.008, label="steering hit (direction from layer X, added at layer Y, norm matched to Y)")
fig.savefig(f"{O}/grid_peaks.png" if PLAIN else f"{O}/grid.png", dpi=90, bbox_inches="tight"); print("ok")
