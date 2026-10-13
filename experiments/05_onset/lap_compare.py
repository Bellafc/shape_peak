"""形變峰 vs Billa (2604.15557) 的 logit-lens 可及性 A_lin，預測三個邊界層：
  steering 起效層（第2部分，各強度）、跨層遷移邊界（第3部分，遷移力首次 ≥0.5 的建向量層）、交接層（第4部分）。
A_lin 代理：lens_top = 最後位置 logit lens（final norm + lm_head）在「各答案類別 + 其他」上取 argmax 的正確率（比 Billa 的全詞表 argmax 嚴格）。
LENS_SRC 環境變數可改成精確版（全詞表 argmax）的目錄。"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import os, glob, ast, numpy as np, pandas as pd
from scipy.stats import spearmanr
LENS=os.environ.get('LENS_SRC','/workspace/others/results/lens'); COL=os.environ.get('LENS_COL','lens_top')
E=pd.read_csv(f'{DATA}/events/events.csv')
BIG=['Gemma-2-9B-it','Llama-3.1-8B-Instruct','Mistral-7B-Instruct-v0.3','Qwen2.5-7B-Instruct']
short=lambda m:m.split('-')[0]
O=pd.read_csv(f'{DATA}/steer_sweep/report_onset.csv')
H=pd.read_csv(f'{DATA}/peak_type/regroup.csv')
# 遷移邊界
XF=f'{DATA}/steer_xfer/'; TB={}
for f in glob.glob(XF+'*_*.csv'):
    n=os.path.basename(f)[:-4]
    if n.startswith('report'): continue
    m,t=n.rsplit('_',1); d=pd.read_csv(f)
    if 'template' not in d.columns: continue
    if sorted(d.template.unique())!=[0,1,2] or d.groupby('template').n.first().sum()<20: continue
    M=d.groupby(['X','Y']).hit.mean().unstack(); Ys=[Y for Y in M.columns if M[Y].notna().all()]
    Yok=[Y for Y in Ys if Y in M.index and M.loc[Y,Y]>=0.9]
    if not Yok: continue
    tr=pd.Series({X:M.loc[X,Yok].mean() for X in M.index})
    b=next((X for X in tr.index if tr[X]>=0.5),np.nan); TB[(m,t)]=b
rows=[]
for m in BIG:
    for t in E[E.model==m].task.unique():
        if m.startswith('Mistral') and t=='hour': continue
        e=E[(E.model==m)&(E.task==t)]; p1=e[e.peak=='第1峰']
        if not len(p1): continue
        top=sorted(e[e.peak.isin(['第1峰','第2–3峰'])].layer)
        f=f'{LENS}/{m}_{t}.csv'
        if not os.path.exists(f): continue
        L=pd.read_csv(f).set_index('layer')[COL]
        r=dict(model=m,task=t,shape_p1=int(p1.layer.iloc[0]),shape_min3=min(top),
               lap_argmax=int(L.idxmax()),lap_emerge=next((l for l in L.index if L[l]>0.1),np.nan),lap_emerge05=next((l for l in L.index if L[l]>0.05),np.nan),
               lap_peak=L.max(), transfer=TB.get((m,t),np.nan))
        h=H[(H.model==m)&(H.task==t)]; r['handoff']=h.handoff.iloc[0] if len(h) else np.nan
        for (mo,s),g in O[(O.model==short(m))&(O.task==t)].groupby(['mode','s']): r[f'onset_{mo}{s:g}']=g.on_best.iloc[0]
        r['_lens']=L; rows.append(r)
D=pd.DataFrame(rows)
PRED=[('shape_p1','形變第1峰'),('shape_min3','前三峰最早者'),('lap_emerge','A_lin 首次 >0.1（Billa 的門檻）'),('lap_emerge05','A_lin 首次 >0.05'),('lap_argmax','A_lin 最高層（Billa 的選層規則）')]
TGT=[('onset_nat1','起效層 α=1'),('onset_nat2','起效層 α=2'),('onset_nat8','起效層 α=8'),('onset_rel0.4','起效層 rel 0.4'),('onset_rel0.8','起效層 rel 0.8'),('transfer','遷移邊界'),('handoff','交接層')]
out=[]; P=lambda *a:(print(*a),out.append(' '.join(map(str,a))))
P(f'# 形變峰 vs logit-lens 可及性（A_lin = {COL}）\n')
P('每格：平均絕對誤差 / ±1 內比例 / 下界違反數（目標早於預測 >1 層，即「預測說之前不行，但其實行」）\n')
P('| 目標 | n | '+' | '.join(n for _,n in PRED)+' |'); P('|---|---|'+'---|'*len(PRED))
for tc,tn in TGT:
    g=D.dropna(subset=[tc]); cells=[]
    for pc,_ in PRED:
        x=g.dropna(subset=[pc]); e=x[tc]-x[pc]
        cells.append(f'{e.abs().mean():.2f} / {np.mean(e.abs()<=1):.2f} / {(e<-1).sum()}')
    P(f'| {tn} | {len(g)} | '+' | '.join(cells)+' |')
# Billa 自己的主張：峰值 A_lin 預測 steering 效果
P('\n## Billa 的 regime 2 在我們的設定下是否成立：起效層上 A_lin 是否已 >0.1\n')
for tc,tn in TGT[:5]:
    g=D.dropna(subset=[tc]); low=[r for r in g.itertuples() if r._11[int(getattr(r,tc))] <= 0.1] if False else None
    cnt=sum(1 for _,r in g.iterrows() if r['_lens'].get(int(r[tc]),0)<=0.1)
    P(f'- {tn}：{cnt}/{len(g)} 組在起效層 A_lin ≤ 0.1（Billa 預測此時均值差 steering 無效）')
g=D.dropna(subset=['handoff']); P(f'- 交接層：{sum(1 for _,r in g.iterrows() if r["_lens"].get(int(r["handoff"]),0)<=0.1)}/{len(g)} 組在交接層 A_lin ≤ 0.1')
P(f'\nA_lin 峰值 < 0.05（Billa：steering 不可行）的組：{(D.lap_peak<0.05).sum()}/{len(D)}；這些組 α=1 起效：{D[D.lap_peak<0.05].onset_nat1.notna().sum()}')
P('\n## 逐組\n'); P('| 模型 | 任務 | 第1峰 | 前三峰最早 | A_lin>0.1 | A_lin 最高層（值） | 起效 α=1 / α=8 | 遷移邊界 | 交接 |'); P('|---|---|---|---|---|---|---|---|---|')
for r in D.sort_values(['model','task']).itertuples():
    P(f'| {short(r.model)} | {r.task} | {r.shape_p1} | {r.shape_min3} | {r.lap_emerge} | {r.lap_argmax}（{r.lap_peak:.2f}） | {getattr(r,"onset_nat1",np.nan)} / {getattr(r,"onset_nat8",np.nan)} | {r.transfer} | {r.handoff} |')
D.drop(columns='_lens').to_csv(f'compare_{COL}.csv',index=False); open(f'compare_{COL}.md','w').write('\n'.join(out)+'\n')
