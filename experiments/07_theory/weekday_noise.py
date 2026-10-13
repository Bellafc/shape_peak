"""理論② 星期定量預測的噪聲底線檢驗。
預測（加性模型、k 均勻取 1..6、N=7）：峰前答案中心點 c(D) = −a(D)/6，即 cos(c, −a)=1、‖c‖/‖a‖=1/6。
原始估計會被抽樣噪聲墊高（每類只有 18 題）。這裡用模板分半：每個模板本身是完整的 s×k 平衡設計，
用模板 i 估 c、模板 j 估 a（i≠j），交叉內積對噪聲無偏：
  ‖c‖² ≈ <c_i, c_j>、‖a‖² ≈ <a_i, a_j>、<c, −a> ≈ <c_i, −a_j>，三對模板取平均。"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import torch, numpy as np, pandas as pd, glob, os
E=pd.read_csv(f'{DATA}/events/events.csv'); n=7; rows=[]
for f in sorted(glob.glob(f'{DATA}/acts/*_weekday_acts.pt')):
    name=os.path.basename(f)[:-len('_weekday_acts.pt')]
    a=torch.load(f,weights_only=False); H=a['H'].float(); N,NL,_=H.shape
    idx=np.arange(N); t=idx//42; s=(idx%42)//6; k=idx%6+1; D=(s+k)%n
    e=E[(E.model==name)&(E.task=='weekday')]; p1=int(e[e.peak=='第1峰'].layer.iloc[0])
    for L in range(max(1,p1-5),min(NL,p1+6)):
        X=H[:,L].double(); Cs,As=[],[]
        for tt in range(3):
            Xt=X[t==tt]; mu=Xt.mean(0); st=s[t==tt]; Dt=D[t==tt]
            As.append(torch.stack([Xt[st==i].mean(0)-mu for i in range(n)])); Cs.append(torch.stack([Xt[Dt==i].mean(0)-mu for i in range(n)]))
        pairs=[(0,1),(0,2),(1,2),(1,0),(2,0),(2,1)]
        cc=np.mean([float((Cs[i]*Cs[j]).sum()) for i,j in pairs]); aa=np.mean([float((As[i]*As[j]).sum()) for i,j in pairs])
        ca=np.mean([float((Cs[i]*-As[j]).sum()) for i,j in pairs])
        C=sum(Cs)/3; A_=sum(As)/3
        raw_cos=float((C*-A_).sum()/(C.norm()*A_.norm())); raw_ratio=float(C.norm()/A_.norm())
        cos_c=ca/np.sqrt(max(cc,1e-12)*max(aa,1e-12)) if cc>0 and aa>0 else np.nan
        rows.append(dict(model=name,L=L,rel=L-p1,raw_cos=raw_cos,cos_corr=cos_c,raw_ratio=raw_ratio,ratio_corr=np.sqrt(cc/aa) if cc>0 and aa>0 else np.nan))
R=pd.DataFrame(rows); R.to_csv('weekday_noise.csv',index=False)
out=['# 理論② 星期定量預測：噪聲校正（模板交叉估計）\n','預測：峰前 cos(c, −a) = 1、‖c‖/‖a‖ = 1/6 ≈ 0.167。\n',
     '| 模型 | 峰前 5 層：cos 原始 → 校正 | ‖c‖/‖a‖ 原始 → 校正 | 峰後 5 層：cos 校正 | ‖c‖/‖a‖ 校正 |','|---|---|---|---|---|']
for m,g in R.groupby('model'):
    pre=g[g.rel<0]; post=g[g.rel>0]
    out.append(f'| {m} | {pre.raw_cos.mean():.2f} → {pre.cos_corr.mean():.2f} | {pre.raw_ratio.mean():.3f} → {pre.ratio_corr.mean():.3f} | {post.cos_corr.mean():.2f} | {post.ratio_corr.mean():.3f} |')
pre=R[R.rel<0]; out.append(f'\n全部峰前層：cos 校正中位 {pre.cos_corr.median():.2f}，‖c‖/‖a‖ 校正中位 {pre.ratio_corr.median():.3f}（原始 {pre.raw_cos.median():.2f} / {pre.raw_ratio.median():.3f}）')
print('\n'.join(out)); open('weekday_noise.md','w').write('\n'.join(out)+'\n')
