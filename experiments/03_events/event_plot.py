"""事件種類總圖：6 個代表組，上：形變（全維 Procrustes 層間變化）與第 1–3 峰事件類型；下：身份 / 因果 / lens。"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
R = f"{DATA}/events"; S = pd.read_csv(f"{R}/signals.csv"); E = pd.read_csv(f"{R}/events.csv")
G = [("Llama-3.1-8B-Instruct", "city"), ("Llama-3.1-8B-Instruct", "month"), ("Mistral-7B-Instruct-v0.3", "continent"),
     ("Qwen2.5-7B-Instruct", "city"), ("Qwen2.5-7B-Instruct", "month"), ("Gemma-2-9B-it", "month")]
TYPE_C = {"可讀重組": "#4a3aa7", "寫入": "#e34948", "集中": "#008300", "輸出化": "#eda100", "無事件": "#898781"}
TYPE_EN = {"可讀重組": "readable reorg", "寫入": "write", "集中": "concentrate", "輸出化": "output (lens)", "無事件": "none"}
LINES = [("id", "identity probe", "#2a78d6"), ("causal", "patching (max of last / all pos)", "#eb6834"), ("sa", "steering all pos α=1", "#1baf7a"), ("lens", "logit lens p(answer)", "#eda100")]
fig, ax = plt.subplots(2, 6, figsize=(22, 6.4), gridspec_kw=dict(height_ratios=[1, 1.3], hspace=0.08, wspace=0.18), sharex="col")
for j, (m, t) in enumerate(G):
    s = S[(S.model == m) & (S.task == t)].sort_values("layer"); L = s.layer.values
    s = s.assign(causal=s[["pl", "pa"]].max(axis=1))
    a = ax[0, j]; a.bar(L, s["shape"], width=0.7, color="#c3c2b7")
    e = E[(E.model == m) & (E.task == t) & E.peak.isin(["第1峰", "第2–3峰"])]
    for r in e.itertuples():
        a.bar(r.layer, r.shape, width=0.7, color=TYPE_C[r.type]); a.annotate(f"L{r.layer}", (r.layer, r.shape), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8, color="#3d3d3a")
    a.set_title(f"{m.split('-Instruct')[0]} · {t}", fontsize=10); a.set_ylim(0, s["shape"].max() * 1.25)
    b = ax[1, j]
    for k, lab, c in LINES:
        if s[k].notna().any(): b.plot(L, s[k], color=c, lw=2, label=lab)
    for r in e.itertuples(): b.axvline(r.layer, color=TYPE_C[r.type], lw=1, ls=":", alpha=0.8)
    b.set_ylim(-0.03, 1.05); b.set_xlabel("layer")
    for aa in (a, b):
        aa.spines[["top", "right"]].set_visible(False); aa.tick_params(labelsize=8, colors="#3d3d3a"); aa.grid(axis="y", color="#e8e6dc", lw=0.6); aa.set_axisbelow(True)
ax[0, 0].set_ylabel("shape change\n(full-dim Procrustes)", fontsize=9); ax[1, 0].set_ylabel("score", fontsize=9)
h1 = [plt.Rectangle((0, 0), 1, 1, color=c) for c in TYPE_C.values()] + [plt.Rectangle((0, 0), 1, 1, color="#c3c2b7")]
fig.legend(h1, [TYPE_EN[k] for k in TYPE_C] + ["other layers"], loc="upper center", ncol=6, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0), title="top 1–3 shape peaks, by event type", title_fontsize=9)
h, l = ax[1, 0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.04))
fig.savefig(f"{R}/grid.png", dpi=130, bbox_inches="tight"); print("ok")
