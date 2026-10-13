"""統一用第1峰重算『形變峰 vs 交接層』，按事件類型分組；再試只用 lens 的選峰規則（不需干預）。"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import pandas as pd, numpy as np, itertools
from scipy.stats import binomtest
rng=np.random.default_rng(0)
EV=f'{DATA}/events/'; HT='/workspace/handoff_tentative_paper/tables/'
E=pd.read_csv(EV+'events.csv'); S=pd.read_csv(EV+'signals.csv'); N=pd.read_csv(EV+'noise_summary.csv')
M=pd.read_csv(HT+'master_table.csv'); CI=pd.read_csv(HT+'confidence_intervals.csv')
TASK={'星期':'weekday','月份':'month','小時':'hour','字母':'letter','城市':'city','國家→洲':'continent','首都→國家→洲':'capital2hop','顏色':'color','上下文顏色':'ctxcolor','情緒':'emotion'}
MOD={'Qwen2.5-7B-Instruct':'Qwen2.5-7B-Instruct','Llama-3.1-8B-Instruct':'Llama-3.1-8B-Instruct','Mistral-7B-Instruct-v0.3':'Mistral-7B-Instruct-v0.3','Gemma-2-9B-it':'Gemma-2-9B-it'}
M['task']=M['任務'].map(TASK); M['model']=M['模型']
M['model']=M['model'].replace({'Mistral-7B':'Mistral-7B-Instruct-v0.3','Gemma-2-9B':'Gemma-2-9B-it'})
rows=[]
for r in M.itertuples():
    h=pd.to_numeric(r.輸入失效_patching,errors='coerce')
    if not np.isfinite(h): continue
    e=E[(E.model==r.model)&(E.task==r.task)]
    p1=e[e.peak=='第1峰'].iloc[0]; others=e[e.peak=='第2–3峰'].layer.tolist()
    ci=CI[(CI.model==r.model)&(CI.task==r.task)&(CI.method=='patching')].iloc[0]
    ns=N[(N.model==r.model)&(N.task==r.task)]
    s=S[(S.model==r.model)&(S.task==r.task)].set_index('layer').lens
    rows.append(dict(類型=r.類型,model=r.model,task=r.task,NL=int(p1.NL),old_peak=r.形狀峰值,peak=int(p1.layer),type=p1.type,
        peaks=sorted([int(p1.layer)]+others),handoff=int(h),lo=int(ci.in_drop_lo),hi=int(ci.in_drop_hi),
        stable_pm1=float(ns.top1_stable_pm1.iloc[0]) if len(ns) else np.nan,lens=s))
D=pd.DataFrame(rows); D['off']=D.handoff-D.peak
D['off_lo']=D.lo-D.peak; D['off_hi']=D.hi-D.peak
out=[]
P=lambda *a: (print(*a), out.append(' '.join(map(str,a))))
P(f'# 形變峰（統一用第1峰）vs 交接層（patching 輸入失效），n={len(D)}\n')
P('off = 交接層 − 第1峰（正 = 交接在峰之後）；off 區間 = 交接層 95% bootstrap 區間 − 峰；峰穩定 = bootstrap 第1峰落在 ±1 層的比例\n')
P('| 類型 | 模型 | 任務 | 舊峰 | 第1峰 | 峰的事件 | 交接層 [95%] | off [區間] | 峰穩定 |'); P('|---|---|---|---|---|---|---|---|---|')
for r in D.sort_values(['type','類型','model']).itertuples():
    P(f'| {r.類型} | {r.model.split("-Inst")[0].replace("-it","")} | {r.task} | {r.old_peak} | {r.peak} | {r.type} | {r.handoff} [{r.lo}, {r.hi}] | {r.off:+d} [{r.off_lo:+d}, {r.off_hi:+d}] | {r.stable_pm1:.2f} |')
P('\n## 按峰的事件類型\n'); P('| 事件 | n | off 中位數 | ±1 內 | off>0 | off<0 | 符號檢驗 p（雙尾，排除 0） |'); P('|---|---|---|---|---|---|---|')
for t,g in D.groupby('type'):
    pos,neg=(g.off>0).sum(),(g.off<0).sum(); p=binomtest(pos,pos+neg).pvalue if pos+neg else np.nan
    P(f'| {t} | {len(g)} | {g.off.median():+.0f} | {(g.off.abs()<=1).sum()}/{len(g)} | {pos} | {neg} | {p:.3f} |')
w=D[D.type=='寫入']; bs=[np.mean(np.abs(w.off.values[rng.integers(0,len(w),len(w))])<=1) for _ in range(5000)]
P(f'\n寫入峰 ±1 內：{np.mean(w.off.abs()<=1):.2f}，組 bootstrap 95% [{np.percentile(bs,2.5):.2f}, {np.percentile(bs,97.5):.2f}]')
wn=w[w.task!='ctxcolor']; P(f'寫入峰（不含上下文顏色）±1 內：{(wn.off.abs()<=1).sum()}/{len(wn)}')
# 區間層面：交接層 95% 區間與峰 ±1 有交集
D['ci_hit']=(D.off_hi>=-1)&(D.off_lo<=1)
P(f'考慮交接層區間後（區間與峰±1有交集）：寫入 {D[D.type=="寫入"].ci_hit.sum()}/{len(w)}，全部 {D.ci_hit.sum()}/{len(D)}')
P('\n## 舊定義 vs 第1峰\n')
for col in ['old_peak','peak']:
    o=D.handoff-D[col]; P(f'- {col}: MAE {o.abs().mean():.2f}，±1 內 {(o.abs()<=1).mean():.2f}')

# ---------- 不需干預的選峰規則 ----------
def lens_at(s,l): return float(s.get(l,np.nan)) if l in s.index else (float(s.iloc[-1]) if l>s.index.max() else 0.0)
def pick(r,th_up,th_pre,win=3):
    """在第1–3峰中，跳過『之後 win 層 lens 仍 < th_up』的峰（可讀重組候選），也跳過『峰前 2 層 lens 已 ≥ th_pre』的峰（集中/輸出化候選）；
    取最早的剩餘峰；若全被跳過，取第1峰。"""
    s=r.lens; ok=[L for L in r.peaks if lens_at(s,L+win)>=th_up and lens_at(s,L-2)<th_pre]
    return min(ok) if ok else r.peak
def lens_onset(r,th=0.1):
    s=r.lens; x=s[s>=th]; return int(x.index.min()) if len(x) else int(s.index.max())
res=[]
for th_up,th_pre,win in itertools.product([0.05,0.1,0.2,0.3],[0.2,0.3,0.4,0.5],[2,3,4]):
    pr=np.array([pick(r,th_up,th_pre,win) for r in D.itertuples()]); e=D.handoff.values-pr
    res.append(dict(th_up=th_up,th_pre=th_pre,win=win,MAE=np.abs(e).mean(),pm1=np.mean(np.abs(e)<=1),exact=np.mean(e==0)))
R=pd.DataFrame(res)
base=D.handoff-D.peak
lo=np.array([lens_onset(r) for r in D.itertuples()]); lo_off=D.handoff.values-lo
loo=np.array([lo_off[i]-np.median(np.delete(lo_off,i)) for i in range(len(D))])
P('\n## 不需干預的選層：預測交接層（33 組）\n')
P(f'- 第1峰本身：MAE {base.abs().mean():.2f}，±1 內 {(base.abs()<=1).mean():.2f}，同層 {(base==0).mean():.2f}')
P(f'- lens 起點（≥0.1）+ 留一偏移：MAE {np.abs(loo).mean():.2f}，±1 內 {np.mean(np.abs(loo)<=1):.2f}')
dflt=R[(R.th_up==0.1)&(R.th_pre==0.3)&(R.win==3)].iloc[0]
P(f'- lens 篩選的峰（預設 th_up=0.1, th_pre=0.3, win=3）：MAE {dflt.MAE:.2f}，±1 內 {dflt.pm1:.2f}，同層 {dflt.exact:.2f}')
P(f'- 48 種門檻組合：MAE 中位數 {R.MAE.median():.2f}（範圍 {R.MAE.min():.2f}–{R.MAE.max():.2f}），±1 內中位數 {R.pm1.median():.2f}（{R.pm1.min():.2f}–{R.pm1.max():.2f}）；MAE 優於第1峰的組合 {np.mean(R.MAE<base.abs().mean()):.2f}')
D['rule']=[pick(r,0.1,0.3,3) for r in D.itertuples()]; D['rule_off']=D.handoff-D.rule
P('\n逐組（預設規則）：\n'); P('| 模型 | 任務 | 第1峰（事件） | 候選峰 | 規則選的峰 | 交接層 | 規則誤差 |'); P('|---|---|---|---|---|---|---|')
for r in D.sort_values(['model','task']).itertuples():
    P(f'| {r.model.split("-Inst")[0].replace("-it","")} | {r.task} | {r.peak}（{r.type}） | {r.peaks} | {r.rule} | {r.handoff} | {r.rule_off:+d} |')
for k,g in D.groupby('類型'): P(f'- {k}：第1峰 MAE {g.off.abs().mean():.2f} / ±1 {np.mean(g.off.abs()<=1):.2f}；規則 MAE {g.rule_off.abs().mean():.2f} / ±1 {np.mean(g.rule_off.abs()<=1):.2f}')
R.to_csv('rule_sensitivity.csv',index=False); D.drop(columns='lens').to_csv('regroup.csv',index=False)
open('regroup.md','w').write('\n'.join(out)+'\n')
