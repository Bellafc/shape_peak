import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import pandas as pd, numpy as np, ast
O=pd.read_csv(f'{DATA}/steer_sweep/report_onset.csv')
D=pd.read_csv('regroup.csv'); D['peaks']=D.peaks.apply(ast.literal_eval)
D['m']=D.model.str.split('-').str[0].replace({'Qwen2.5':'Qwen2.5'})
O['m']=O.model.str.split('-').str[0]
print(O.groupby(['mode','s']).size().head(20)); print(O.m.unique(), D.m.unique())
out=[]; P=lambda *a:(print(*a),out.append(' '.join(map(str,a))))
J=O.merge(D[['m','task','handoff','peaks']],on=['m','task'],how='left')
P('\n## 起效層 − 第1峰，按第1峰的事件類型（中位數；括號 = 早於峰 >1 層的組數 / 組數）\n')
P('| 第1峰類型 | n | α=1 | α=2 | α=4 | α=8 | rel 0.4 | rel 0.8 |'); P('|---|---|---|---|---|---|---|---|')
cols=[('nat',1),('nat',2),('nat',4),('nat',8),('rel',0.4),('rel',0.8)]
for t,g in O.groupby('type'):
    cells=[]
    for mo,s in cols:
        x=g[(g['mode']==mo)&(g.s==s)].d_best.dropna(); cells.append(f'{x.median():+.1f}（{(x<-1).sum()}/{len(x)}）')
    P(f'| {t} | {g[["m","task"]].drop_duplicates().shape[0]} | '+' | '.join(cells)+' |')
P('\n## α=1 起效層 vs 第4部分交接層（patching 輸入失效），有交接的 33 組\n')
for mo,s in cols:
    g=J[(J['mode']==mo)&(J.s==s)&J.handoff.notna()&J.on_best.notna()]
    d=g.on_best-g.handoff; P(f'- {mo} {s}: n={len(g)}，起效層 − 交接層 中位 {d.median():+.1f}，±1 內 {np.mean(d.abs()<=1):.2f}，早於交接 >1 層 {np.sum(d<-1)}')
g=J[(J['mode']=='nat')&(J.s==8)&J.handoff.notna()]
P('\nα=8 早於交接層 >1 的組：'+'; '.join(f'{r.m} {r.task} 起效{int(r.on_best)} 交接{int(r.handoff)} 峰{r.peaks}' for r in g.itertuples() if r.on_best-r.handoff<-1))
open('sweep_cross.md','w').write('\n'.join(out)+'\n')
