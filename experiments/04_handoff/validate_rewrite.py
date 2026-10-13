# 验证重写的 handoff_patch.py / lens_full.py：在原论文的 Qwen2.5-7B 上跑，与原始数据逐层比较
import sys as _s, os as _o; ROOT = _o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))); _s.path.insert(0, ROOT); from paths import DATA
import numpy as np, pandas as pd
M = "Qwen2.5-7B-Instruct"; R = f"{DATA}/validation"; D = DATA  # 新脚本在 Qwen2.5-7B 上的输出 vs 原论文数据
S2 = pd.read_csv(f"{D}/events_v2/signals_v2_orig4.csv"); S2 = S2[S2.model == M]; HP = pd.read_csv(f"{D}/handoff_from_pdf_orig4.csv"); HP = HP[HP.model == M]
HT = pd.read_csv(f"{D}/events_v2/handoff_types_v2_orig4.csv"); HT = HT[HT.model == M].set_index("task")
ARITH = {"weekday", "month", "hour", "letter"}
def handoff(s):  # 第一个「输入 token patching < 最大值一半」的层（在最大值之后）
    s = s.sort_index(); mx = s.max(); after = s.loc[s.idxmax():]; return next((int(L) for L, v in after.items() if v < mx / 2), None)
rows = []
for t in sorted(S2.task.unique()):
    o = S2[S2.task == t].set_index("layer"); r = {"task": t}
    try:
        p = pd.read_csv(f"{R}/handoff_patch/{M}_{t}.csv"); g = lambda c: p[p.cond == c].set_index("layer").hit
        r["pa 最大差"] = (g("ans_all_patch") - o.pa).abs().max(); r["pl 最大差"] = (g("ans_last_patch") - o.pl).abs().max()
        pin = HP[(HP.task == t) & (HP.cond == "in_patch")].set_index("layer").hit
        r["in_patch 与PDF曲线 最大差"] = (g("in_patch") - pin).abs().max(); r["handoff 新"] = handoff(g("in_patch"))
        r["handoff 原"] = int(HT.loc[t, "handoff"]) if t in HT.index else None; r["n"] = int(p.n.iloc[0])
    except FileNotFoundError: pass
    if t not in ARITH:
        try:
            s = pd.read_csv(f"{R}/handoff_retrieval/{M}_{t}.csv"); g2 = lambda c: s[s.cond == c].set_index("layer").hit
            r["sa 最大差"] = (g2("ans_all_nat") - o.sa).abs().max(); r["sl 最大差"] = (g2("ans_last_nat") - o.sl).abs().max()
            sin = HP[(HP.task == t) & (HP.cond == "in_steer")].set_index("layer").hit; r["in_steer 与PDF 最大差"] = (g2("in_nat") - sin).abs().max()
        except FileNotFoundError: pass
    else:
        try:
            s = pd.read_csv(f"{R}/handoff/{M}_{t}.csv"); g2 = lambda c: s[s.cond == c].set_index("layer").hit
            r["sa 最大差"] = (g2("ans_all_nat") - o.sa).abs().max(); r["sl 最大差"] = (g2("ans_last_nat") - o.sl).abs().max()
        except FileNotFoundError: pass
    try:
        lf = pd.read_csv(f"{R}/lens_full/{M}_{t}.csv").set_index("layer"); r["lens 最大差"] = (lf.lens_p - o.lens).abs().max(); r["a_lin 最大差"] = (lf.a_lin - o.a_lin).abs().max()
    except FileNotFoundError: pass
    rows.append(r)
pd.set_option("display.width", 250); print(pd.DataFrame(rows).round(3).to_string(index=False))
