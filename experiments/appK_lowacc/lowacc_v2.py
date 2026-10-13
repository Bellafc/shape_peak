"""附錄 J：去掉準確率 < 0.55 的 4 組後，§4–§7 主要數字（與全部組並列）。"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import pandas as pd, numpy as np, ast
from scipy.stats import binomtest
LOW = {('Llama-3.1-8B-Instruct','letter'),('Mistral-7B-Instruct-v0.3','letter'),('Gemma-2-9B-it','hour'),('Mistral-7B-Instruct-v0.3','emotion')}
rng = np.random.default_rng(0)
E = pd.read_csv('events_v2/events_v2.csv'); S = pd.read_csv('events_v2/signals_v2.csv')
key = lambda d: [(m, t) for m, t in zip(d.model, d.task)]
TOP = {k: sorted(g[g.peak.isin(['第1峰','第2–3峰'])].layer) for k, g in E.groupby(['model','task'])}
P1 = {k: int(g[g.peak=='第1峰'].layer.iloc[0]) for k, g in E.groupby(['model','task'])}
NL = {k: int(g.NL.iloc[0]) for k, g in E.groupby(['model','task'])}
def perm(items, nperm=5000):
    obs = sum(min(abs(y - np.array(TOP[k]))) <= 1 for k, y in items); c = 0; tot = 0
    for _ in range(nperm):
        s = 0
        for k, y in items:
            n = NL[k]; lo = int(round(n*0.25)); sp = n - lo; sh = rng.integers(0, sp)
            s += min(abs(y - (lo + (np.array(TOP[k]) - lo + sh) % sp))) <= 1
        c += s >= obs; tot += s
    return obs, tot/nperm, (c+1)/(nperm+1)
rows = []
def add(sec, name, f):
    a = f(lambda k: True); b = f(lambda k: k not in LOW); rows.append((sec, name, a, b))
# §4
def first_write(sel):
    g = E[[sel(k) for k in key(E)]]; f = g[g.peak=='第1峰']; n = g[g.peak=='非峰']
    return f"{(f.type=='寫入').mean():.2f} vs {(n.type=='寫入').mean():.2f} (n={len(f)})"
add('§4','First peak is a write vs. non-peak layers', first_write)
def lensonly(sel):
    f = E[[sel(k) for k in key(E)]]; f = f[f.peak=='第1峰']; return f"{(f.type=='輸出化').sum()}"
add('§4','First peaks that are output-ization only', lensonly)
def rr(sel):
    g = E[[sel(k) for k in key(E)]]; r = g[g.peak.isin(['第1峰','第2–3峰']) & (g.type=='可讀重組')]
    return f"{(r.lens<0.01).sum()}/{len(r)}"
add('§4','Readable-reorganization peaks with logit lens < 0.01', rr)
# §5
T = pd.read_csv(f'{DATA}/handoff_all/table.csv'); MAPT={"星期":"weekday","月份":"month","小時":"hour","字母":"letter","城市":"city","國家→洲":"continent","首都→國家→洲":"capital2hop","顏色":"color","上下文顏色":"ctxcolor","情緒":"emotion"}
T['task'] = T['任務'].map(MAPT); T['model'] = T['模型']; T = T.dropna(subset=['輸入失效_patching'])
def gap(sel):
    g = T[[sel(k) for k in key(T)]]; return f"{(g['空檔_patching']<=1).sum()}/{len(g)}"
add('§5','Handoff within at most one layer (patching gap ≤ 1)', gap)
def hpk(sel):
    g = T[[sel(k) for k in key(T)]]; o, nm, p = perm([((r.model, r.task), int(r.輸入失效_patching)) for r in g.itertuples()])
    return f"{o}/{len(g)} (null {nm:.1f}, p<{max(p,1e-3):.3f})" if p < 0.001 else f"{o}/{len(g)} (null {nm:.1f}, p={p:.3f})"
add('§5','Handoff within ±1 of a top-3 peak', hpk)
def wr(sel):
    g = T[[sel(k) for k in key(T)]]; g = g[[E[(E.model==m)&(E.task==t)&(E.peak=='第1峰')].type.iloc[0]=='寫入' for m,t in key(g)]]
    return f"{sum(abs(r.輸入失效_patching - P1[(r.model,r.task)])<=1 for r in g.itertuples())}/{len(g)}"
add('§5','Write first peak = handoff (±1)', wr)
# §6
O = pd.read_csv(f'{DATA}/steer_sweep/report_onset.csv'); full={'Gemma':'Gemma-2-9B-it','Llama':'Llama-3.1-8B-Instruct','Mistral':'Mistral-7B-Instruct-v0.3','Qwen2.5':'Qwen2.5-7B-Instruct'}
O['model'] = O.model.map(full); O = O[~((O.model=='Mistral-7B-Instruct-v0.3')&(O.task=='hour'))]
def on1(sel):
    g = O[(O['mode']=='nat')&(O.s==1)&O.on_best.notna()]; g = g[[sel(k) for k in key(g)]]
    return f"{sum(r.on_best >= P1[(r.model,r.task)]-1 for r in g.itertuples())}/{len(g)}"
add('§6','Onset at α=1 no earlier than first peak − 1', on1)
def onpk(sel):
    out = []
    for mo, s in [('nat',1),('nat',8),('rel',0.8)]:
        g = O[(O['mode']==mo)&(O.s==s)&O.on_best.notna()]; g = g[[sel(k) for k in key(g)]]
        o = sum(min(abs(r.on_best - np.array(TOP[(r.model,r.task)]))) <= 1 for r in g.itertuples()); out.append(f"{o/len(g):.2f}")
    return ' / '.join(out)
add('§6','Onset within ±1 of a top-3 peak (α=1 / α=8 / c=0.8)', onpk)
def early(sel):
    out = []
    for mo, s in [('nat',1),('nat',8),('rel',0.8)]:
        g = O[(O['mode']==mo)&(O.s==s)&O.on_best.notna()]; g = g[[sel(k) for k in key(g)]]
        out.append(str(sum(r.on_best < min(TOP[(r.model,r.task)]) - 1 for r in g.itertuples())))
    return ' / '.join(out) + f" of {len(g)}"
add('§6','Onsets > 1 layer before the earliest peak (α=1 / α=8 / c=0.8)', early)
C = pd.read_csv('lap_compare/compare_a_lin.csv')
def lap(sel):
    g = C[[sel(k) for k in key(C)]].dropna(subset=['onset_nat1']); a = (g.onset_nat1-g.shape_p1) < -1; b = (g.onset_nat1-g.lap_emerge) < -1
    oa, ob = int((a&~b).sum()), int((b&~a).sum()); p = binomtest(ob, oa+ob).pvalue
    return f"{a.sum()} vs {b.sum()} (p={p:.3f})"
add('§6','Lower-bound violations at α=1: first peak vs A_lin > 0.1', lap)
# §7
R = pd.read_csv(f'{DATA}/steer_xfer/report_rules.csv'); J = pd.read_csv('events_v2/transfer_types_v2.csv')
def pre(sel):
    g = R[[sel(k) for k in key(R)]]; return f"{g['峰前5層'].mean():.2f} / {g['峰及之後'].mean():.2f} (n={len(g)})"
add('§7','Transfer before / at-or-after the first peak', pre)
def jmp(sel):
    g = J[[sel(k) for k in key(J)]]; return f"{g.on_peak.sum()}/{len(g)}"
add('§7','Largest transfer jump exactly on a top-3 peak', jmp)
X = pd.read_csv('theory/xfer_cum.csv').dropna(subset=['rho_cum_fixed_dist'])
def cum(sel):
    g = X[[sel(k) for k in key(X)]]; return f"{(g.rho_cum_fixed_dist<0).sum()}/{len(g)} (median ρ {g.rho_cum_fixed_dist.median():.2f})"
add('§7','Transfer decreases with cumulative shape change at fixed distance', cum)
D = pd.DataFrame(rows, columns=['sec','result','all','excl']); D.to_csv('lowacc_v2.csv', index=False); print(D.to_string(index=False))
