"""用交接實驗本身的 patching 曲線核對『只在峰上做 patching』；對照：同樣 3 次查詢的二分搜尋。"""
import pandas as pd, numpy as np, ast
D=pd.read_csv('regroup.csv'); D['peaks']=D.peaks.apply(ast.literal_eval)
PM='/workspace/handoff_tentative_paper/patching/main/'
out=[]; P=lambda *a:(print(*a),out.append(' '.join(map(str,a))))
def curves(m,t):
    c=pd.read_csv(f'{PM}{m}_{t}.csv'); g=lambda k: c[c.cond==k].set_index('layer').hit
    return g('in_patch'), g('ans_all_patch')
def peak_rule(ans,peaks,th=0.5):
    ok=[L for L in peaks if ans.get(L,0)>=th]; return (min(ok) if ok else max(peaks)), len(peaks)
def bisect(ans,lo,hi,q,th=0.5):
    # 在 [lo,hi] 找 ans 首次 ≥ th 的層，最多 q 次查詢；回傳區間中點
    n=0
    while lo<hi and n<q:
        mid=(lo+hi)//2; n+=1
        if ans.get(mid,0)>=th: hi=mid
        else: lo=mid+1
    return (lo+hi)/2
rows=[]
for r in D.itertuples():
    inp,ans=curves(r.model,r.task); NL=r.NL; lo=int(round(NL*0.25))
    pr,_=peak_rule(ans,r.peaks)
    rise=int(ans[ans>=0.5].index.min())
    rows.append(dict(model=r.model,task=r.task,handoff=r.handoff,ans_rise=rise,peak1=r.peak,peaks=r.peaks,peak_rule=pr,
        bis3=bisect(ans,lo,NL-1,3),bis4=bisect(ans,lo,NL-1,4),bis5=bisect(ans,lo,NL-1,5)))
R=pd.DataFrame(rows)
P(f'n={len(R)}；目標 = patching 輸入失效層（交接層）\n')
P('| 方法 | 干預查詢次數 | MAE | ±1 內 | 同層 |'); P('|---|---|---|---|---|')
for k,name,q in [('peak1','第1峰（不干預）','0'),('peak_rule','只在第1–3峰做答案位置 patching，取最早 ≥0.5 的峰','3'),
                 ('bis3','晚期層二分搜尋','3'),('bis4','晚期層二分搜尋','4'),('bis5','晚期層二分搜尋','5'),('ans_rise','逐層掃描（答案生效層）','全部晚期層')]:
    e=R.handoff-R[k]; P(f'| {name} | {q} | {e.abs().mean():.2f} | {np.mean(e.abs()<=1):.2f} | {np.mean(e==0):.2f} |')
P('\n峰規則改選或出錯的組：')
for r in R.itertuples():
    if r.peak_rule!=r.peak1 or abs(r.handoff-r.peak_rule)>1: P(f'- {r.model.split("-")[0]} {r.task}：峰 {r.peaks}，第1峰 {r.peak1} → 選 {r.peak_rule}，交接 {r.handoff}（誤差 {r.handoff-r.peak_rule:+d}）')
R.to_csv('verify_patch.csv',index=False); open('verify_patch.md','w').write('\n'.join(out)+'\n')
