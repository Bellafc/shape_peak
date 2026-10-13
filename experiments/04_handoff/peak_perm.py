"""置換檢驗：目標層（交接層、各強度的 steering 起效層）是否落在前三個形變峰上。
虛無：各組前三峰保持間距，在晚期層 [round(0.25·NL), NL−1] 內整體循環平移（同第三部分 xfer_extra.py 的做法），10000 次。
統計量：目標層正好在某峰上的組數、在某峰 ±1 內的組數。"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import pandas as pd, numpy as np, ast
rng=np.random.default_rng(0); NPERM=10000
E=pd.read_csv(f'{DATA}/events/events.csv')
BIG=['Gemma-2-9B-it','Llama-3.1-8B-Instruct','Mistral-7B-Instruct-v0.3','Qwen2.5-7B-Instruct']
E=E[E.model.isin(BIG)]
PK={k:sorted(g[g.peak.isin(['第1峰','第2–3峰'])].layer) for k,g in E.groupby(['model','task'])}
NL={k:int(g.NL.iloc[0]) for k,g in E.groupby(['model','task'])}
# 交接層：33 組 + 顏色 2 組
H=pd.read_csv('regroup.csv')[['model','task','handoff']]
H=pd.concat([H,pd.DataFrame([dict(model='Gemma-2-9B-it',task='color',handoff=26),dict(model='Mistral-7B-Instruct-v0.3',task='color',handoff=17)])])
O=pd.read_csv(f'{DATA}/steer_sweep/report_onset.csv')
full={'Gemma':'Gemma-2-9B-it','Llama':'Llama-3.1-8B-Instruct','Mistral':'Mistral-7B-Instruct-v0.3','Qwen2.5':'Qwen2.5-7B-Instruct'}
O['model']=O.model.map(full); O=O[~((O.model=='Mistral-7B-Instruct-v0.3')&(O.task=='hour'))]
def test(T,label):
    T=[(k,int(v)) for k,v in T if k in PK and PK[k]]
    def stat(shifts):
        ex=on1=0
        for (k,y),sh in zip(T,shifts):
            n=NL[k]; lo=int(round(n*0.25)); span=n-lo
            p=[lo+((q-lo+sh)%span) for q in PK[k]]; d=min(abs(y-q) for q in p); ex+=d==0; on1+=d<=1
        return ex,on1
    obs=stat([0]*len(T))
    null=np.array([stat([rng.integers(0,NL[k]-int(round(NL[k]*0.25))) for k,_ in T]) for _ in range(NPERM)])
    p0=(1+(null[:,0]>=obs[0]).sum())/(NPERM+1); p1=(1+(null[:,1]>=obs[1]).sum())/(NPERM+1)
    return f'| {label} | {len(T)} | {obs[0]} | {null[:,0].mean():.1f} | {p0:.4f} | {obs[1]} | {null[:,1].mean():.1f}（95%: {np.percentile(null[:,1],95):.0f}） | {p1:.4f} |'
out=['# 置換檢驗：目標層是否落在前三個形變峰上（10000 次循環平移）\n','| 目標 | n | 正好在峰上 | 虛無平均 | p | 在峰 ±1 | 虛無平均 | p |','|---|---|---|---|---|---|---|---|']
out.append(test([((r.model,r.task),r.handoff) for r in H.itertuples()],'交接層（patching 輸入失效）'))
for mo,s in [('nat',1),('nat',2),('nat',4),('nat',8),('rel',0.4),('rel',0.8)]:
    g=O[(O['mode']==mo)&(O.s==s)&O.on_best.notna()]
    out.append(test([((r.model,r.task),r.on_best) for r in g.itertuples()],f'steering 起效層 {mo} {s:g}'))
print('\n'.join(out)); open('peak_perm.md','w').write('\n'.join(out)+'\n')
