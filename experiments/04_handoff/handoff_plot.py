"""交接測試的圖與摘要（失效 / 生效用固定門檻 0.5）：python handoff_plot.py → data/handoff/handoff.png、summary.csv、SUMMARY.md 的表格"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import glob, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = f"{DATA}/handoff"; ref = pd.read_csv(f"{DATA}/depth_vs_shape.csv")
CN = {"weekday": "星期", "month": "月份"}
files = sorted(f for f in glob.glob(f"{D}/*.csv") if not f.endswith("summary.csv"))
def metrics(x, cin, cans):
    a, b = x[x.cond == cin].set_index("layer").hit, x[x.cond == cans].set_index("layer").hit
    am = int(a.idxmax()); drop = next((L for L in a.index if L > am and a[L] < 0.5), None)  # 固定門檻 0.5
    rise = next((L for L in b.index if b[L] >= 0.5), None); cross = next((L for L in b.index if b[L] >= a[L] and b[L] > 0.05), None)
    both = [L for L in a.index if a[L] >= 0.5 and b[L] >= 0.5]; neither = [L for L in a.index if drop is not None and rise is not None and drop <= L < rise and a[L] < 0.5 and b[L] < 0.5]
    return dict(in_max=round(float(a.max()), 2), in_drop=drop, ans_rise=rise, cross=cross, width=None if drop is None or rise is None else rise - drop,
                both_ok=f"L{both[0]}–L{both[-1]}" if both else "—", gap=len(neither))
rows = []; fig, axs = plt.subplots(2, 4, figsize=(22, 9)); axs = axs.ravel()
for k, f in enumerate(files):
    x = pd.read_csv(f); name, con = os.path.basename(f)[:-4].rsplit("_", 1); r = ref[(ref.model.str.startswith(name.split("-Instruct")[0].split("-it")[0])) & (ref.task == con)]
    peak, steer = (int(r.peak.iloc[0]), int(r.steer.iloc[0])) if len(r) else (None, None)
    row = dict(case=f"{name} {CN.get(con, con)}", NL=int(x.layer.max()) + 1, shape_peak=peak, steer_onset=steer)
    for tag, (ci, ca) in {"nat": ("in_nat", "ans_all_nat"), "rel1.0": ("in_rel1.0", "ans_all_rel1.0")}.items():
        for kk_, v in metrics(x, ci, ca).items(): row[f"{kk_}_{tag}"] = v
    b = x[x.cond == "ans_last_nat"].set_index("layer").hit; row["last_rise_nat"] = next((L for L in b.index if b[L] >= 0.5), None)
    rows.append(row)
    if k < len(axs):
        ax = axs[k]
        for cond, col, ls in (("in_nat", "C0", "-"), ("in_rel1.0", "C0", "--"), ("ans_all_nat", "C3", "-"), ("ans_all_rel1.0", "C3", "--"), ("ans_last_nat", "C1", "-"),
                              ("rand_in", "gray", ":"), ("rand_ans", "k", ":")):
            y = x[x.cond == cond].set_index("layer").hit; ax.plot(y.index, y.values, color=col, ls=ls, label=cond, lw=1.6 if ls == "-" else 1)
        if peak is not None: ax.axvline(peak, color="purple", ls=":", lw=1.5, label=f"shape peak L{peak}")
        if steer is not None: ax.axvline(steer, color="green", ls=":", lw=1.5, label=f"steer onset L{steer}")
        ax.set_title(f"{name} {con}", fontsize=10); ax.set_xlabel("layer"); ax.set_ylim(-0.02, 1.02); ax.grid(alpha=.3)
        if k == 0: ax.legend(fontsize=7, loc="center left")
plt.tight_layout(); plt.savefig(f"{D}/handoff.png", dpi=100)
S = pd.DataFrame(rows); S.to_csv(f"{D}/summary.csv", index=False); pd.set_option("display.width", 300); print(S.to_string(index=False))
