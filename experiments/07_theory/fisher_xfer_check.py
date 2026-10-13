"""理論③：答案可解釋方差（Fisher 比）越高，steering 需要的強度越小。
理論④：跨層遷移 X→Y 受兩層之間累積形變限制（三角不等式給上界）；檢查在相同層距下，累積形變越大遷移越差。"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
S=pd.read_csv(f'{DATA}/events/signals.csv')
SW=f'{DATA}/steer_sweep/'; XF=f'{DATA}/steer_xfer/'
out=[]; P=lambda *a:(print(*a),out.append(' '.join(map(str,a))))
# ---------- ③ ----------
res=[]
for f in sorted(glob.glob(SW+'*.csv')):
    if 'report' in f or 'existing' in f: continue
    name,task=os.path.basename(f)[:-4].rsplit('_',1)
    d=pd.read_csv(f); d=d[d.X.astype(str)==d.Y.astype(str)]
    d=d.assign(w=d.hit*d.n); g=d.groupby(['mode','s','Y'])[['w','n']].sum().reset_index(); g['hit']=g.w/g.n
    sig=S[(S.model==name)&(S.task==task)].set_index('layer')
    if sig.fisher.isna().all(): continue
    for mode,levels in (('rel',[0.1,0.2,0.4,0.8]),('nat',[1,2,4,8])):
        rows=[]
        for Y,x in g[g['mode']==mode].groupby('Y'):
            ok=[s for s in levels if (x[x.s==s].hit>=0.5).any()]
            rows.append((Y,min(ok) if ok else levels[-1]*2,sig.fisher.get(Y,np.nan)))
        r=pd.DataFrame(rows,columns=['Y','need','fisher']).dropna()
        if r.need.nunique()>1 and len(r)>=5:
            rho=spearmanr(r.fisher,r.need).correlation
            rk=r[['Y','fisher','need']].rank()
            def resid(a,b): A=np.c_[np.ones(len(b)),b]; return a-A@np.linalg.lstsq(A,a,rcond=None)[0]
            pr=np.corrcoef(resid(rk.fisher.values,rk.Y.values),resid(rk.need.values,rk.Y.values))[0,1]
            rl=spearmanr(r.Y,r.need).correlation
            res.append(dict(model=name,task=task,mode=mode,rho=rho,rho_layer=rl,partial=pr,n=len(r)))
R3=pd.DataFrame(res)
P('## 理論③：Fisher 比 vs 需要的最小強度（每組逐層 Spearman，預測為負）\n')
for mode,g in R3.groupby('mode'):
    P(f'- {mode}：{len(g)} 組，ρ(Fisher, 強度) 中位 {g.rho.median():+.2f}（<0：{(g.rho<0).sum()}/{len(g)}）；對照 ρ(層號, 強度) 中位 {g.rho_layer.median():+.2f}；控制層號後偏相關中位 {g.partial.median():+.2f}（<0：{(g.partial<0).sum()}/{len(g)}）')
R3.to_csv('fisher_need.csv',index=False)
# ---------- ④ ----------
res=[]; pooled=[]
for f in sorted(glob.glob(XF+'*.csv')):
    b=os.path.basename(f)[:-4]
    if '_' not in b or b.startswith('report'): continue
    name,task=b.rsplit('_',1)
    if name.startswith('Mistral') and task=='hour': continue
    if not {'X','Y','hit','n'}<=set(pd.read_csv(f,nrows=1).columns): continue
    d=pd.read_csv(f); d=d.assign(w=d.hit*d.n); g=d.groupby(['X','Y'])[['w','n']].sum().reset_index(); g['hit']=g.w/g.n
    sh=S[(S.model==name)&(S.task==task)].set_index('layer')['shape'].fillna(0)
    if not len(sh): continue
    cs=sh.cumsum()
    g=g[g.X!=g.Y].copy(); g['dist']=(g.X-g.Y).abs()
    g['cum']=[cs[max(x,y)]-cs[min(x,y)] for x,y in zip(g.X,g.Y)]
    g['dir']=np.sign(g.Y-g.X)
    rh=[]
    for (dist,dr),x in g.groupby(['dist','dir']):
        if len(x)>=4 and x.hit.nunique()>1 and x.cum.nunique()>1: rh.append(spearmanr(x.cum,x.hit).correlation)
    rd=spearmanr(g.dist,g.hit).correlation; rc=spearmanr(g.cum,g.hit).correlation
    res.append(dict(model=name,task=task,rho_dist=rd,rho_cum=rc,rho_cum_fixed_dist=np.nanmean(rh) if rh else np.nan,n_pairs=len(g)))
R4=pd.DataFrame(res)
P('\n## 理論④：跨層遷移 vs 累積形變（每組 Spearman，預測為負）\n')
P(f'- 組數 {len(R4)}；遷移 vs 層距 ρ 中位 {R4.rho_dist.median():+.2f}；遷移 vs 累積形變 ρ 中位 {R4.rho_cum.median():+.2f}；累積形變比層距更負的組 {(R4.rho_cum<R4.rho_dist).sum()}/{len(R4)}')
x=R4.rho_cum_fixed_dist.dropna(); P(f'- 固定層距與方向後，遷移 vs 累積形變 ρ 中位 {x.median():+.2f}，<0 的組 {(x<0).sum()}/{len(x)}')
R4.to_csv('xfer_cum.csv',index=False); open('fisher_xfer.md','w').write('\n'.join(out)+'\n')
