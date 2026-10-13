import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import pandas as pd, numpy as np, ast
rng=np.random.default_rng(0)
D=pd.read_csv('regroup.csv'); D['peaks']=D.peaks.apply(ast.literal_eval)
S=pd.read_csv(f'{DATA}/events/signals.csv')
out=[]; P=lambda *a:(print(*a),out.append(' '.join(map(str,a))))
hit=[min(abs(h-np.array(p)))<=1 for h,p in zip(D.handoff,D.peaks)]
# 基線：每組在晚期層（≥25% 深度，與峰定義同）隨機取 3 個不相鄰層
base=[]
for _ in range(2000):
    c=0
    for r in D.itertuples():
        lo=int(round(r.NL*0.25)); cand=rng.choice(np.arange(lo,r.NL),3,replace=False); c+=min(abs(r.handoff-cand))<=1
    base.append(c/len(D))
P(f'交接層落在第1–3峰之一 ±1 內：{sum(hit)}/{len(D)} = {np.mean(hit):.2f}；隨機 3 個晚期層基線 {np.mean(base):.2f}（95% 上界 {np.percentile(base,95):.2f}，p≈{np.mean(np.array(base)>=np.mean(hit)):.4f}）')
miss=D[[not h for h in hit]]; P('未命中：'+'; '.join(f'{r.model.split("-")[0]} {r.task} 峰{r.peaks} 交接{r.handoff}' for r in miss.itertuples()))
# 只在 3 個峰上做答案位置 patching（pa = 問句後全部位置整個殘差 patching，events 的信號）：取最早一個 pa(L)≥0.5 的峰
def rule(r,th=0.5,sig='pa'):
    s=S[(S.model==r.model)&(S.task==r.task)].set_index('layer')[sig]
    if s.isna().all(): return np.nan
    ok=[L for L in r.peaks if s.get(L,np.nan)>=th]; return min(ok) if ok else max(r.peaks)
for sig in ['pa','pl']:
    pr=np.array([rule(r,sig=sig) for r in D.itertuples()]); e=D.handoff.values-pr; m=np.isfinite(e)
    P(f'只在 3 個峰上做 patching（信號 {sig}≥0.5，取最早）：n={m.sum()}，MAE {np.abs(e[m]).mean():.2f}，±1 內 {np.mean(np.abs(e[m])<=1):.2f}，同層 {np.mean(e[m]==0):.2f}')
    if sig=='pa': D['rule_pa']=pr
f=D[np.isfinite(D.rule_pa)]; b=f.handoff-f.peak
P(f'  同一批組第1峰本身：MAE {b.abs().mean():.2f}，±1 內 {np.mean(b.abs()<=1):.2f}')
for r in D.itertuples():
    if np.isfinite(r.rule_pa) and r.rule_pa!=r.peak: P(f'  改選：{r.model.split("-")[0]} {r.task} 第1峰 {r.peak}（{r.type}）→ {int(r.rule_pa)}，交接 {r.handoff}')
open('top3.md','w').write('\n'.join(out)+'\n')
