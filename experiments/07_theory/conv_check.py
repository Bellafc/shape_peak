"""理論②（卷積模型）驗證：起點+k 任務，答案位置（最後 token）各層激活。
平衡設計 (模板 t, 起點 s, k)，加性擬合 h ≈ μ + a_s + b_k + c_t，殘差 r 為交互作用。
答案 centroid c(D)=E[h|D]−μ 拆成 c_pred(D)=E[a_s|D]（加性部分經 k 分布卷積）+ c_int(D)=E[r|D]（按答案組織的交互項）。
預測：峰前 c ≈ c_pred（星期：= −a(D)/6），峰後 c_int 主導；兩者交叉的層落在形變峰附近。"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); from paths import DATA
import torch, numpy as np, pandas as pd, glob, os
A=f'{DATA}/acts/'; E=pd.read_csv(f'{DATA}/events/events.csv')
NS={'weekday':7,'month':12,'hour':12,'letter':26}
def disp(A_,B_):
    A_=A_-A_.mean(0); B_=B_-B_.mean(0); A_=A_/A_.norm(); B_=B_/B_.norm()
    Q,_=torch.linalg.qr(torch.cat([A_,B_]).T); A_,B_=A_@Q,B_@Q
    return float(1-torch.linalg.svdvals(A_.T@B_).sum()**2)
rows=[]
for f in sorted(glob.glob(A+'*_acts.pt')):
    name,task=os.path.basename(f)[:-8].rsplit('_',1)
    if task not in NS: continue
    n=NS[task]; a=torch.load(f,weights_only=False); H=a['H'].float(); N,NL,d=H.shape
    if N!=3*n*6: print('skip',name,task,N); continue
    idx=np.arange(N); t=idx//(n*6); s=(idx%(n*6))//6; k=idx%6+1; D=(s+k)%n
    assert (D==a['ans']).all(), (name,task)
    e=E[(E.model==name)&(E.task==task)]
    p1=int(e[e.peak=='第1峰'].layer.iloc[0]) if len(e) else -1; top=sorted(e[e.peak.isin(['第1峰','第2–3峰'])].layer.tolist())
    prevC=None
    for L in range(NL):
        X=H[:,L].double(); mu=X.mean(0); Xc=X-mu
        As=torch.stack([Xc[s==i].mean(0) for i in range(n)]); Bk=torch.stack([Xc[k==j].mean(0) for j in range(1,7)]); Ct=torch.stack([Xc[t==j].mean(0) for j in range(3)])
        fit=As[s]+Bk[k-1]+Ct[t]; r=Xc-fit
        C=torch.stack([Xc[D==i].mean(0) for i in range(n)])
        Cp=torch.stack([As[[(i-j)%n for j in range(1,7)]].mean(0) for i in range(n)])
        Ci=torch.stack([r[D==i].mean(0) for i in range(n)])
        tot=(Xc**2).sum(1).mean()
        rows.append(dict(model=name,task=task,L=L,NL=NL,peak1=p1,top=top,
            R2_add=float(1-(r**2).sum(1).mean()/tot), eta2_D=float((C**2).sum(1).mean()/tot),
            share_pred=float((Cp**2).sum()/(C**2).sum()), share_int=float((Ci**2).sum()/(C**2).sum()),
            cos_c_pred=float((C*Cp).sum()/(C.norm()*Cp.norm())),
            cos_c_negA=float((C*-As).sum()/(C.norm()*As.norm())), ratio_c_A=float(C.norm()/As.norm()),
            shape=disp(C.float(),prevC) if prevC is not None else np.nan))
        prevC=C.float()
    print(name,task,'done',flush=True)
R=pd.DataFrame(rows); R.to_csv('conv_layers.csv',index=False)
