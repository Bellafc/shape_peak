# 重建 handoff 汇总（原 results/handoff_all/table.csv 已遗失）：
#   原 4 个模型：handoff 层取原论文 events_v2/handoff_types_v2_orig4.csv；新模型：输入 token patching 在最大值之后首次 < 最大值一半的层（论文附录 D 的定义）
#   steering handoff 同理用 in_nat；emotion 没有单一输入 token，不定义 handoff
# 另把新模型的四条曲线按 handoff_from_pdf.csv 的格式追加（附录 handoff 图用）
import sys as _s, os as _o; ROOT = _o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))); _s.path.insert(0, ROOT); from paths import DATA
import pandas as pd, numpy as np, os
D = f"{DATA}"; NEW = ["Qwen2.5-14B-Instruct", "Phi-4-14B"]; ARITH = {"weekday", "month", "hour", "letter"}
ZH = {"weekday": "星期", "month": "月份", "hour": "小時", "letter": "字母", "city": "城市", "continent": "國家→洲", "capital2hop": "首都→國家→洲", "color": "顏色", "ctxcolor": "上下文顏色", "emotion": "情緒"}
def handoff(s, floor=0.2):
    s = s.sort_index()
    if s.max() < floor: return None
    mx = s.max(); return next((int(L) for L, v in s.loc[s.idxmax():].items() if v < mx / 2), None)
HT = pd.read_csv(f"{D}/events_v2/handoff_types_v2_orig4.csv")
rows = [dict(模型=r.model, 任務=ZH[r.task], 輸入失效_patching=r.handoff, 輸入失效_steering=np.nan, source="orig") for r in HT.itertuples()]
curves = []
for m in NEW:
    for t in ZH:
        p = pd.read_csv(f"{D}/handoff_patch/{m}_{t}.csv"); s = pd.read_csv(f"{D}/{'handoff' if t in ARITH else 'handoff_retrieval'}/{m}_{t}.csv")
        g = lambda d, c: d[d.cond == c].set_index("layer").hit
        if t != "emotion":  # 情绪故事没有单一输入 token，不定义 handoff（曲线仍画在附录图里）
            rows.append(dict(模型=m, 任務=ZH[t], 輸入失效_patching=handoff(g(p, "in_patch")), 輸入失效_steering=handoff(g(s, "in_nat")), n=int(p.n.iloc[0]), source="new"))
        for c_new, c_pdf, d in (("in_patch", "in_patch", p), ("ans_all_patch", "ans_patch", p), ("in_nat", "in_steer", s), ("ans_all_nat", "ans_steer", s)):
            x = g(d, c_new); curves += [dict(model=m, task=t, cond=c_pdf, layer=int(L), layer_raw=float(L), hit=float(v)) for L, v in x.items()]
os.makedirs(f"{D}/handoff_all", exist_ok=True); T = pd.DataFrame(rows); T.to_csv(f"{D}/handoff_all/table.csv", index=False)
HP = pd.read_csv(f"{DATA}/handoff_from_pdf_orig4.csv"); HP = pd.concat([HP[~HP.model.isin(NEW)], pd.DataFrame(curves)]); HP.to_csv(f"{D}/handoff_from_pdf.csv", index=False)
print(T[T.source == "new"].to_string(index=False))
