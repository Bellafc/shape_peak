"""附錄 G 用：在 A_lin 精確版（compare_a_lin.csv）上，把交接層換成 v2 的 35 組，重算全部目標的誤差、違反數、配對檢驗與逐組表。"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import pandas as pd, numpy as np
from scipy.stats import binomtest, wilcoxon
C = pd.read_csv('compare_a_lin.csv'); P = pd.read_csv('compare_lens_top.csv')
T = pd.read_csv(f'{DATA}/handoff_all/table.csv')
MAPT = {"星期":"weekday","月份":"month","小時":"hour","字母":"letter","城市":"city","國家→洲":"continent","首都→國家→洲":"capital2hop","顏色":"color","上下文顏色":"ctxcolor","情緒":"emotion"}
T['task'] = T['任務'].map(MAPT); H = {(r.模型, r.task): r.輸入失效_patching for r in T.itertuples()}
for D in (C, P): D['handoff'] = [H.get((m, t), np.nan) for m, t in zip(D.model, D.task)]
TG = [('onset_nat1','Onset $\\alpha{=}1$'),('onset_nat2','Onset $\\alpha{=}2$'),('onset_nat8','Onset $\\alpha{=}8$'),('onset_rel0.4','Onset $c{=}0.4$'),('onset_rel0.8','Onset $c{=}0.8$'),('transfer','Transfer boundary'),('handoff','Handoff')]
PR = [('shape_p1','First peak'),('shape_min3','Earliest peak'),('lap_emerge','$A_{\\mathrm{lin}}{>}0.1$'),('lap_argmax','$\\arg\\max A_{\\mathrm{lin}}$')]
def cell(g, t, p): e = g[t] - g[p]; return f"{e.abs().mean():.1f} / {np.mean(e.abs()<=1):.2f} / {(e<-1).sum()}"
out = {}
for name, D in (('exact', C), ('proxy', P)):
    rows = []
    for t, tl in TG:
        g = D.dropna(subset=[t]); rows.append(f"    {tl} & {len(g)} & " + ' & '.join(cell(g, t, p) for p, _ in PR) + ' \\\\')
    out[name] = '\n'.join(rows)
tests = []
for t, tl in TG:
    g = C.dropna(subset=[t]); a = (g[t]-g.shape_p1) < -1; b = (g[t]-g.lap_emerge) < -1
    ob, oa = int((b & ~a).sum()), int((a & ~b).sum()); p = binomtest(ob, oa+ob).pvalue if oa+ob else 1
    e1, e2 = (g[t]-g.shape_p1).abs(), (g[t]-g.lap_emerge).abs(); pw = wilcoxon(e1, e2).pvalue
    tests.append(f"    {tl} & {e1.mean():.2f} & {e2.mean():.2f} & {pw:.3f} & {oa} & {ob} & {p:.3f} \\\\")
out['tests'] = '\n'.join(tests)
L = {}
SH={'Qwen2.5-7B-Instruct':'Qwen2.5-7B','Llama-3.1-8B-Instruct':'Llama-3.1-8B','Mistral-7B-Instruct-v0.3':'Mistral-7B','Gemma-2-9B-it':'Gemma-2-9B'}
TN={'weekday':'Weekday','month':'Month','hour':'Hour','letter':'Letter','city':'Landmark $\\to$ city','continent':'Country $\\to$ cont.','capital2hop':'Capital $\\to$ cont.','color':'Object $\\to$ color','ctxcolor':'In-context color','emotion':'Story $\\to$ emotion'}
f = lambda v: '--' if v != v else str(int(v))
pp = []
for t in TN:
    for m in SH:
        r = C[(C.model==m)&(C.task==t)]
        if not len(r): continue
        r = r.iloc[0]
        pp.append(f"    {SH[m]} & {TN[t]} & {f(r.shape_p1)} & {f(r.shape_min3)} & {f(r.lap_emerge)} & {f(r.lap_argmax)} ({r.lap_peak:.2f}) & {f(r.onset_nat1)} & {f(r.onset_nat8)} & {f(r.transfer)} & {f(r.handoff)} \\\\")
out['perpair'] = '\n'.join(pp)
# steering where A_lin <= 0.1
lens = {}
for r in C.itertuples(): lens[(r.model, r.task)] = pd.read_csv(f'{DATA}/lens_full/{r.model}_{r.task}.csv').set_index('layer').a_lin
low = []
for t, tl in TG:
    if t == 'transfer': continue
    g = C.dropna(subset=[t]); n = sum(lens[(r.model, r.task)].get(int(getattr(r, t) if t != 'onset_rel0.4' and t != 'onset_rel0.8' else r[t]), 0) <= 0.1 for _, r in g.iterrows()) if False else sum(lens[(r['model'], r['task'])].get(int(r[t]), 0) <= 0.1 for _, r in g.iterrows())
    low.append((tl, int(n), int(len(g))))
out['low'] = low
import json; json.dump(out, open('appendix_tables.json', 'w'), indent=1)
print(out['exact']); print(out['tests']); print(low); print(len(pp))
